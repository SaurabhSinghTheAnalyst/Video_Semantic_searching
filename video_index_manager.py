import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st

from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Disable LLM globally to prevent OpenAI initialization
Settings.llm = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoIndexManager:
    """
    Manages the creation and loading of LlamaIndex vector stores for video transcripts
    """
    
    def __init__(self, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the VideoIndexManager with local embedding model
        
        Args:
            embed_model_name: HuggingFace model name for embeddings
        """
        self.embed_model_name = embed_model_name
        self.embed_model = None
        self.index = None
        self.storage_path = "./embeddings/video_index"
        self.chroma_path = "./embeddings/chroma_db"
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the local embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.embed_model_name}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embed_model_name,
                device="cpu"  # Use CPU for compatibility
            )
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def load_transcript_documents(self, transcript_folder: str = "transcripts") -> List[Document]:
        """
        Load transcript JSON files and convert them to LlamaIndex Documents
        
        Args:
            transcript_folder: Path to folder containing transcript JSON files
            
        Returns:
            List of LlamaIndex Document objects
        """
        transcript_path = Path(transcript_folder)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript folder not found: {transcript_folder}")
        
        documents = []
        transcript_files = list(transcript_path.glob("*_transcript.json"))
        
        if not transcript_files:
            raise FileNotFoundError(f"No transcript files found in {transcript_folder}")
        
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                
                # Extract video info from filename
                filename = file_path.stem
                video_info = self._parse_filename(filename)
                
                # Prefer chunking by sentences under paragraphs if present
                paragraphs = None
                if 'paragraphs' in transcript_data:
                    paragraphs = transcript_data['paragraphs']
                elif 'paragraphs' in transcript_data.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0]:
                    paragraphs_obj = transcript_data['results']['channels'][0]['alternatives'][0]['paragraphs']
                    # Handle nested paragraphs structure: {'transcript': '...', 'paragraphs': [...]}
                    if isinstance(paragraphs_obj, dict) and 'paragraphs' in paragraphs_obj:
                        paragraphs = paragraphs_obj['paragraphs']
                    else:
                        paragraphs = paragraphs_obj
                if paragraphs and isinstance(paragraphs, list):
                    for para_idx, para in enumerate(paragraphs):
                        for sent_idx, sent in enumerate(para.get('sentences', [])):
                            text = sent.get('text', '').strip()
                            if not text:
                                continue
                            doc = Document(
                                text=text,
                                metadata={
                                    "video_file": video_info["video_file"],
                                    "chunk_number": video_info["chunk_number"],
                                    "paragraph_index": para_idx + 1,
                                    "sentence_index": sent_idx + 1,
                                    "start_time": sent.get("start", None),
                                    "end_time": sent.get("end", None),
                                    "word_count": len(text.split()),
                                    "source_file": str(file_path),
                                    "filename": filename
                                }
                            )
                            documents.append(doc)
                    continue  # skip utterances if paragraphs are present
                # Fallback: use utterances if available
                utterances = transcript_data.get("utterances", [])
                if utterances:
                    for idx, utt in enumerate(utterances):
                        text = utt.get("text", "").strip()
                        if not text:
                            continue
                        doc = Document(
                            text=text,
                            metadata={
                                "video_file": video_info["video_file"],
                                "chunk_number": video_info["chunk_number"],
                                "utterance_index": idx + 1,
                                "start_time": utt.get("start", None),
                                "end_time": utt.get("end", None),
                                "confidence": utt.get("confidence", transcript_data.get("confidence", 0)),
                                "word_count": len(text.split()),
                                "source_file": str(file_path),
                                "filename": filename
                            }
                        )
                        documents.append(doc)
                else:
                    # Fallback: use the full transcript text
                    text = ""
                    
                    # Try multiple ways to extract the transcript text
                    if 'results' in transcript_data:
                        results = transcript_data['results']
                        if 'channels' in results and len(results['channels']) > 0:
                            channel = results['channels'][0]
                            if 'alternatives' in channel and len(channel['alternatives']) > 0:
                                alternative = channel['alternatives'][0]
                                text = alternative.get('transcript', '').strip()
                    
                    # Fallback to direct text field if above didn't work
                    if not text:
                        text = transcript_data.get("text", "").strip()
                    
                    if not text or transcript_data.get("error"):
                        logger.warning(f"Skipping empty/error transcript: {filename}")
                        continue
                    
                    # Try to get duration from transcript data if available
                    duration = video_info["duration"]
                    end_time = video_info["end_time"] 
                    
                    # Check if we can extract timing from the transcript data structure
                    if 'results' in transcript_data:
                        results = transcript_data['results']
                        if 'channels' in results and len(results['channels']) > 0:
                            channel = results['channels'][0]
                            if 'alternatives' in channel and len(channel['alternatives']) > 0:
                                alternative = channel['alternatives'][0]
                                if 'words' in alternative and len(alternative['words']) > 0:
                                    words = alternative['words']
                                    if words:
                                        end_time = words[-1].get('end', 0)
                                        duration = end_time
                    
                    # Also get overall confidence if available  
                    confidence = transcript_data.get("confidence", 0)
                    if 'results' in transcript_data and 'channels' in transcript_data['results']:
                        channels = transcript_data['results']['channels']
                        if len(channels) > 0 and 'alternatives' in channels[0]:
                            alternatives = channels[0]['alternatives']
                            if len(alternatives) > 0:
                                confidence = alternatives[0].get('confidence', confidence)
                    
                    doc = Document(
                        text=text,
                        metadata={
                            "video_file": video_info["video_file"],
                            "chunk_number": video_info["chunk_number"],
                            "start_time": video_info["start_time"],
                            "end_time": end_time,
                            "duration": duration,
                            "confidence": confidence,
                            "word_count": len(text.split()),
                            "source_file": str(file_path),
                            "filename": filename
                        }
                    )
                    documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _parse_filename(self, filename: str) -> Dict:
        """
        Parse transcript filename to extract video info
        
        Examples: 
        - "1.mp3_transcript" -> video_file="1.mp4", chunk_number=1
        - "AI.mp3_transcript" -> video_file="AI.mp4", chunk_number=1  
        - "1_chunk_003_transcript" -> video_file="1.mp4", chunk_number=3
        """
        try:
            # Remove "_transcript" suffix
            name_part = filename.replace("_transcript", "")
            
            # Check for chunk-based format first: "1_chunk_003"
            if "_chunk_" in name_part:
                parts = name_part.split("_")
                if len(parts) >= 3 and parts[1] == "chunk":
                    video_id = parts[0]
                    chunk_number = int(parts[2])
                    
                    # Calculate timestamps (assuming 10-second chunks)
                    start_time = (chunk_number - 1) * 10
                    end_time = chunk_number * 10
                    duration = 10
                    
                    return {
                        "video_file": f"{video_id}.mp4",
                        "chunk_number": chunk_number,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration
                    }
            
            # Handle standard format: "filename.mp3" or just "filename"
            if name_part.endswith(".mp3"):
                # Remove .mp3 extension to get base name
                video_name = name_part[:-4]  # Remove ".mp3"
                return {
                    "video_file": f"{video_name}.mp4",  # Convert to mp4 for video file
                    "chunk_number": 1,
                    "start_time": 0,
                    "end_time": 0,  # Will be set from transcript data
                    "duration": 0   # Will be calculated from transcript data
                }
            else:
                # Fallback for any other format
                return {
                    "video_file": f"{name_part}.mp4",
                    "chunk_number": 1,
                    "start_time": 0,
                    "end_time": 0,
                    "duration": 0
                }
                
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            return {
                "video_file": "unknown.mp4",
                "chunk_number": 1,
                "start_time": 0,
                "end_time": 0,
                "duration": 0
            }
    
    def create_index_from_transcripts(self, transcript_folder: str = "transcripts") -> VectorStoreIndex:
        """
        Create a new vector index from transcript files
        
        Args:
            transcript_folder: Path to transcript files
            
        Returns:
            VectorStoreIndex object
        """
        logger.info("Creating new vector index from transcripts...")
        
        # Load documents
        documents = self.load_transcript_documents(transcript_folder)
        
        if not documents:
            raise ValueError("No valid documents found to create index")
        
        # Set up Chroma vector store
        chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = chroma_client.get_or_create_collection("video_transcripts")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index with custom node parser
        node_parser = SentenceSplitter(
            chunk_size=512,  # Smaller chunks for better precision
            chunk_overlap=50  # Some overlap to maintain context
        )
        
        # Create the index
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            transformations=[node_parser]
        )
        
        # Persist the index
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.storage_path)
        
        logger.info(f"✅ Index created and persisted to {self.storage_path}")
        return self.index
    
    def load_existing_index(self) -> Optional[VectorStoreIndex]:
        """
        Load an existing vector index from storage
        
        Returns:
            VectorStoreIndex object or None if not found
        """
        try:
            if not Path(self.storage_path).exists():
                logger.info("No existing index found")
                return None
            
            logger.info("Loading existing vector index...")
            
            # Set up Chroma vector store
            chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            chroma_collection = chroma_client.get_or_create_collection("video_transcripts")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=self.storage_path
            )
            
            # Load index
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.embed_model
            )
            
            logger.info("✅ Existing index loaded successfully")
            return self.index
            
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            return None
    
    def get_or_create_index(self, transcript_folder: str = "transcripts") -> VectorStoreIndex:
        """
        Get existing index or create new one if it doesn't exist
        
        Args:
            transcript_folder: Path to transcript files
            
        Returns:
            VectorStoreIndex object
        """
        # Try to load existing index first
        self.index = self.load_existing_index()
        
        if self.index is None:
            # Create new index if none exists
            self.index = self.create_index_from_transcripts(transcript_folder)
        
        return self.index
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"error": "No index loaded"}
        
        try:
            # Get document count (this is an approximation)
            retriever = self.index.as_retriever(similarity_top_k=1000)  # Large number to get all
            nodes = retriever.retrieve("test query to get stats")
            
            # Collect statistics
            video_files = set()
            total_chunks = 0
            total_duration = 0
            
            for node in nodes:
                if hasattr(node, 'metadata'):
                    video_files.add(node.metadata.get("video_file", "unknown"))
                    total_chunks += 1
                    total_duration += node.metadata.get("duration", 0)
            
            return {
                "total_videos": len(video_files),
                "total_chunks": total_chunks,
                "total_duration_minutes": round(total_duration / 60, 2),
                "video_files": sorted(list(video_files))
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)} 