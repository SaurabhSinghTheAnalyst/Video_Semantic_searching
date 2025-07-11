import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
import os

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    
    # Try HuggingFace embeddings first
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        HUGGINGFACE_AVAILABLE = True
    except ImportError:
        HUGGINGFACE_AVAILABLE = False
        logger.warning("HuggingFace embeddings not available")
    
    # Fallback to OpenAI embeddings if HuggingFace fails
    if not HUGGINGFACE_AVAILABLE:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            OPENAI_AVAILABLE = True
        except ImportError:
            OPENAI_AVAILABLE = False
            logger.warning("OpenAI embeddings not available")
    else:
        OPENAI_AVAILABLE = False
    
    # Disable LLM globally to prevent unwanted API calls
    Settings.llm = None
    LLAMAINDEX_AVAILABLE = True
    logger.info("✅ LlamaIndex dependencies loaded successfully")
    
except ImportError as e:
    LLAMAINDEX_AVAILABLE = False
    HUGGINGFACE_AVAILABLE = False
    OPENAI_AVAILABLE = False
    logger.error(f"❌ Failed to import LlamaIndex dependencies: {e}")
    logger.info("💡 The app will run in limited mode without semantic search")

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
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError("LlamaIndex dependencies not available. Please check requirements.txt")
        
        self.embed_model_name = embed_model_name
        self.embed_model = None
        self.index = None
        self.storage_path = "./embeddings/video_index"
        self.chroma_path = "./embeddings/chroma_db"
        
        # Initialize directory permissions early
        self._initialize_directories()
        
        # Initialize embedding model
        self._initialize_embedding_model()
    
    def _initialize_directories(self):
        """Initialize directories with proper permissions"""
        try:
            import os
            
            # Create directories if they don't exist
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            
            # Set proper permissions
            try:
                os.chmod(self.storage_path, 0o755)
                os.chmod(self.chroma_path, 0o755)
            except OSError:
                # Ignore permission errors on some systems
                pass
                
        except Exception as e:
            logger.warning(f"Could not initialize directories: {e}")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model with fallback options"""
        # Try HuggingFace first (if available)
        if HUGGINGFACE_AVAILABLE:
            try:
                logger.info(f"Loading HuggingFace embedding model: {self.embed_model_name}")
                self.embed_model = HuggingFaceEmbedding(
                    model_name=self.embed_model_name,
                    device="cpu"  # Use CPU for compatibility
                )
                logger.info("✅ HuggingFace embedding model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace model: {e}")
        
        # Fallback to OpenAI embeddings if HuggingFace fails
        if OPENAI_AVAILABLE:
            try:
                # Check for OpenAI API key
                openai_key = None
                try:
                    # Try Streamlit secrets first
                    openai_key = st.secrets.get("openai", {}).get("OPENAI_API_KEY")
                except:
                    pass
                
                # Fallback to environment variable
                if not openai_key:
                    openai_key = os.getenv("OPENAI_API_KEY")
                
                if openai_key:
                    logger.info("Loading OpenAI embedding model as fallback")
                    self.embed_model = OpenAIEmbedding(api_key=openai_key)
                    logger.info("✅ OpenAI embedding model loaded successfully")
                    return
                else:
                    logger.warning("OpenAI API key not found")
            except Exception as e:
                logger.warning(f"Failed to load OpenAI model: {e}")
        
        # If both fail, raise an error
        raise ImportError(
            "No embedding model could be initialized. Please ensure you have either:\n"
            "1. HuggingFace dependencies installed (sentence-transformers, torch)\n"
            "2. OpenAI API key configured in secrets or environment variables"
        )
    
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
        
        try:
            # Ensure directories exist with proper permissions
            Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            
            # Set up Chroma vector store with error handling
            import os
            
            # Try to fix permissions if needed
            if os.path.exists(self.chroma_path):
                try:
                    os.chmod(self.chroma_path, 0o755)
                except:
                    pass  # Ignore permission errors on some systems
            
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
            self.index.storage_context.persist(persist_dir=self.storage_path)
            
            logger.info(f"✅ Index created and persisted to {self.storage_path}")
            return self.index
            
        except Exception as e:
            if "readonly database" in str(e).lower():
                logger.error("Database permission error - trying alternative approach...")
                return self._create_index_alternative_approach(documents, transcript_folder)
            else:
                logger.error(f"Error creating index: {e}")
                raise
    
    def _create_index_alternative_approach(self, documents, transcript_folder: str) -> VectorStoreIndex:
        """
        Alternative index creation approach when database permissions fail
        Uses in-memory vector store as fallback
        """
        try:
            logger.info("🔄 Creating in-memory index as fallback...")
            
            # Use simple in-memory vector store
            from llama_index.core import SimpleDirectoryReader
            
            # Create index without persistent storage
            node_parser = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            
            # Create in-memory index
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=self.embed_model,
                transformations=[node_parser]
            )
            
            logger.info("✅ In-memory index created successfully")
            return self.index
            
        except Exception as e:
            logger.error(f"Failed to create fallback index: {e}")
            raise
    
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
    
    def rebuild_index(self, transcript_folder: str = "transcripts") -> VectorStoreIndex:
        """
        Force rebuild of the index to include all transcripts (including newly uploaded ones)
        
        Args:
            transcript_folder: Path to transcript files
            
        Returns:
            VectorStoreIndex object
        """
        logger.info("🔄 Force rebuilding index to include all transcripts...")
        
        try:
            # Delete existing index files with better error handling
            import shutil
            import os
            
            if Path(self.storage_path).exists():
                try:
                    # Try to fix permissions before deletion
                    for root, dirs, files in os.walk(self.storage_path):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o755)
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o644)
                    shutil.rmtree(self.storage_path)
                    logger.info(f"Deleted existing index at {self.storage_path}")
                except Exception as e:
                    logger.warning(f"Could not delete index directory: {e}")
            
            if Path(self.chroma_path).exists():
                try:
                    # Try to fix permissions before deletion
                    for root, dirs, files in os.walk(self.chroma_path):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o755)
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o644)
                    shutil.rmtree(self.chroma_path)
                    logger.info(f"Deleted existing Chroma DB at {self.chroma_path}")
                except Exception as e:
                    logger.warning(f"Could not delete Chroma directory: {e}")
            
            # Create fresh index with all transcripts
            self.index = self.create_index_from_transcripts(transcript_folder)
            
            logger.info("✅ Index rebuilt successfully with all transcripts")
            return self.index
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            # Try alternative approach if main method fails
            documents = self.load_transcript_documents(transcript_folder)
            return self._create_index_alternative_approach(documents, transcript_folder)
    
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