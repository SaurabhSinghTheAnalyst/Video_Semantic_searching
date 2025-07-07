import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

# Import reranker
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None

# Disable LLM globally to prevent OpenAI initialization
Settings.llm = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoSearchEngine:
    """
    Handles semantic search across video transcripts using LlamaIndex with optional reranking
    """
    
    def __init__(self, index: VectorStoreIndex, use_reranker: bool = True, reranker_model: str = "BAAI/bge-reranker-base"):
        """
        Initialize the search engine with a LlamaIndex vector store
        
        Args:
            index: VectorStoreIndex containing video transcript embeddings
            use_reranker: Whether to use reranking for improved results
            reranker_model: Name of the reranker model to use
        """
        self.index = index
        self.retriever = None
        self.reranker = None
        self.use_reranker = use_reranker and RERANKER_AVAILABLE
        
        # Initialize retriever
        self._setup_retriever()
        
        # Initialize reranker if requested and available
        if self.use_reranker:
            self._setup_reranker(reranker_model)
    
    def _setup_retriever(self):
        """Set up the vector retriever with default parameters"""
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=100,  # Get more candidates for reranking
        )
    
    def _setup_reranker(self, model_name: str):
        """Set up the reranker model"""
        try:
            logger.info(f"Loading reranker model: {model_name}")
            self.reranker = CrossEncoder(model_name)
            logger.info("✅ Reranker model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Falling back to semantic search only.")
            self.use_reranker = False
            self.reranker = None

    def semantic_search(
        self, 
        query: str, 
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        rerank_top_k_multiplier: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across video transcripts with optional reranking
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            similarity_threshold: Minimum similarity score for initial retrieval
            rerank_top_k_multiplier: Multiplier for initial candidates (before reranking)
            
        Returns:
            List of search results with video timestamps and metadata
        """
        try:
            logger.info(f"Searching for: '{query}' (reranking: {self.use_reranker})")
            
            # Determine how many initial candidates to retrieve
            initial_top_k = top_k * rerank_top_k_multiplier if self.use_reranker else top_k * 2
            self.retriever.similarity_top_k = initial_top_k
            
            # Stage 1: Execute initial semantic search
            nodes = self.retriever.retrieve(query)
            
            # Filter by similarity threshold
            filtered_nodes = [
                node for node in nodes 
                if hasattr(node, 'score') and node.score >= similarity_threshold
            ]
            
            if not filtered_nodes:
                logger.info("No results met similarity threshold")
                return []
            
            # Stage 2: Reranking (if enabled)
            if self.use_reranker and self.reranker is not None and len(filtered_nodes) > top_k:
                reranked_nodes = self._rerank_results(query, filtered_nodes, top_k)
            else:
                # Use original semantic search results
                reranked_nodes = filtered_nodes[:top_k]
            
            # Format and return results
            results = []
            for node in reranked_nodes:
                result = self._format_search_result(node, query)
                results.append(result)
            
            logger.info(f"Found {len(results)} relevant results")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def _rerank_results(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        """
        Rerank search results using the cross-encoder reranker
        
        Args:
            query: Search query
            nodes: Initial search results
            top_k: Number of top results to return
            
        Returns:
            Reranked list of nodes
        """
        try:
            logger.info(f"Reranking {len(nodes)} candidates to top {top_k}")
            
            # Prepare query-document pairs for reranking
            query_doc_pairs = [(query, node.text) for node in nodes]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # Combine nodes with rerank scores
            node_score_pairs = list(zip(nodes, rerank_scores))
            
            # Sort by rerank score (descending)
            node_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Update nodes with combined scores (keeping original semantic score)
            reranked_nodes = []
            for i, (node, rerank_score) in enumerate(node_score_pairs[:top_k]):
                # Create a new node with enhanced metadata
                enhanced_node = node
                if hasattr(enhanced_node, 'metadata'):
                    enhanced_node.metadata['rerank_score'] = float(rerank_score)
                    enhanced_node.metadata['rerank_position'] = i + 1
                    enhanced_node.metadata['original_score'] = float(node.score) if hasattr(node, 'score') else 0
                
                reranked_nodes.append(enhanced_node)
            
            # Get the top rerank score for logging
            top_rerank_score = node_score_pairs[0][1] if node_score_pairs else 0
            logger.info(f"✅ Reranking completed. Top score: {top_rerank_score:.3f}")
            return reranked_nodes
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Falling back to semantic search results.")
            return nodes[:top_k]
    
    def _format_search_result(self, node: NodeWithScore, query: str) -> Dict[str, Any]:
        """
        Format a search result node into a structured dictionary
        
        Args:
            node: NodeWithScore from LlamaIndex search
            query: Original search query
            
        Returns:
            Formatted search result dictionary
        """
        metadata = node.metadata if hasattr(node, 'metadata') else {}
        
        # Extract text content
        text = node.text if hasattr(node, 'text') else ""
        
        # Highlight query terms in text (simple implementation)
        highlighted_text = self._highlight_query_terms(text, query)
        
        # Calculate video timestamp
        video_file = metadata.get("video_file", "unknown.mp4")
        start_time = metadata.get("start_time", 0)
        end_time = metadata.get("end_time", 0)
        
        # Format timestamp for display
        formatted_timestamp = self._format_timestamp(start_time)
        
        # Get scoring information
        original_score = node.score if hasattr(node, 'score') else 0
        rerank_score = metadata.get('rerank_score')
        rerank_position = metadata.get('rerank_position')
        
        # Determine the primary relevance score (rerank if available, otherwise semantic)
        relevance_score = rerank_score if rerank_score is not None else original_score
        
        result = {
            "video_file": video_file,
            "video_path": f"Data/{video_file}",  # Assuming videos are in Data folder
            "start_time": start_time,
            "end_time": end_time,
            "timestamp_display": formatted_timestamp,
            "text": text,
            "highlighted_text": highlighted_text,
            "text_snippet": self._create_text_snippet(text, 150),
            "relevance_score": relevance_score,
            "confidence": metadata.get("confidence", 0),
            "chunk_number": metadata.get("chunk_number", 1),
            "word_count": metadata.get("word_count", 0),
            "duration": end_time - start_time,
            "source_file": metadata.get("source_file", ""),
            "paragraph_index": metadata.get("paragraph_index"),
            "sentence_index": metadata.get("sentence_index"),
            "metadata": metadata,  # Include full metadata for debugging
        }
        
        # Add reranking information if available
        if rerank_score is not None:
            result.update({
                "rerank_score": rerank_score,
                "original_semantic_score": metadata.get('original_score', original_score),
                "rerank_position": rerank_position,
                "search_method": "reranked"
            })
        else:
            result.update({
                "search_method": "semantic_only"
            })
        
        return result
    
    def _highlight_query_terms(self, text: str, query: str) -> str:
        """
        Simple query term highlighting in text
        
        Args:
            text: Text to highlight
            query: Query terms to highlight
            
        Returns:
            Text with highlighted query terms
        """
        if not text or not query:
            return text
        
        # Simple word-based highlighting
        query_words = query.lower().split()
        highlighted_text = text
        
        for word in query_words:
            if len(word) > 2:  # Only highlight words longer than 2 characters
                # Case-insensitive replacement with highlighting
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted_text = pattern.sub(f"**{word}**", highlighted_text)
        
        return highlighted_text
    
    def _create_text_snippet(self, text: str, max_length: int = 150) -> str:
        """
        Create a text snippet for display
        
        Args:
            text: Full text
            max_length: Maximum snippet length
            
        Returns:
            Text snippet with ellipsis if truncated
        """
        if len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        snippet = text[:max_length]
        last_space = snippet.rfind(' ')
        
        if last_space > max_length * 0.8:  # If space is reasonably close to end
            snippet = snippet[:last_space]
        
        return snippet + "..."
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds into MM:SS or HH:MM:SS format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def search_by_video(
        self, 
        query: str, 
        video_files: List[str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search within specific video files only
        
        Args:
            query: Search query
            video_files: List of video files to search in
            top_k: Number of results to return
            
        Returns:
            Filtered search results
        """
        # Get all results first
        all_results = self.semantic_search(query, top_k=top_k*3)  # Get more to filter
        
        # Filter by video files
        filtered_results = [
            result for result in all_results 
            if result["video_file"] in video_files
        ]
        
        return filtered_results[:top_k]
    
    def search_by_time_range(
        self, 
        query: str, 
        start_time: float, 
        end_time: float, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific time range across all videos
        
        Args:
            query: Search query
            start_time: Start time in seconds
            end_time: End time in seconds
            top_k: Number of results to return
            
        Returns:
            Time-filtered search results
        """
        # Get all results first
        all_results = self.semantic_search(query, top_k=top_k*3)
        
        # Filter by time range
        filtered_results = [
            result for result in all_results 
            if result["start_time"] >= start_time and result["end_time"] <= end_time
        ]
        
        return filtered_results[:top_k]
    
    def search_high_confidence(
        self, 
        query: str, 
        min_confidence: float = 0.8, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search only in high-confidence transcriptions
        
        Args:
            query: Search query
            min_confidence: Minimum transcription confidence
            top_k: Number of results to return
            
        Returns:
            High-confidence search results
        """
        # Get all results first
        all_results = self.semantic_search(query, top_k=top_k*3)
        
        # Filter by confidence
        filtered_results = [
            result for result in all_results 
            if result["confidence"] >= min_confidence
        ]
        
        return filtered_results[:top_k]
    
    def get_video_summary(self, video_file: str) -> Dict[str, Any]:
        """
        Get summary information for a specific video
        
        Args:
            video_file: Video filename
            
        Returns:
            Video summary information
        """
        try:
            # Search for all chunks of this video
            all_results = self.semantic_search("", top_k=100, similarity_threshold=0.0)
            video_chunks = [r for r in all_results if r["video_file"] == video_file]
            
            if not video_chunks:
                return {"error": f"No data found for {video_file}"}
            
            # Calculate summary statistics
            total_duration = sum(chunk["duration"] for chunk in video_chunks)
            total_words = sum(chunk["word_count"] for chunk in video_chunks)
            avg_confidence = sum(chunk["confidence"] for chunk in video_chunks) / len(video_chunks)
            
            # Get all text content
            all_text = " ".join(chunk["text"] for chunk in video_chunks if chunk["text"])
            
            return {
                "video_file": video_file,
                "total_chunks": len(video_chunks),
                "total_duration": total_duration,
                "total_words": total_words,
                "average_confidence": round(avg_confidence, 3),
                "full_transcript": all_text,
                "chunks": sorted(video_chunks, key=lambda x: x["start_time"])
            }
            
        except Exception as e:
            logger.error(f"Error getting video summary for {video_file}: {e}")
            return {"error": str(e)} 