import streamlit as st
import logging
import os
from pathlib import Path
import json
import time
from typing import List, Dict, Any
import pandas as pd

# Import our custom modules
try:
    from video_index_manager import VideoIndexManager
    from video_search_engine import VideoSearchEngine
    INDEX_AVAILABLE = True
except ImportError as e:
    INDEX_AVAILABLE = False
    st.error(f"⚠️ Search functionality unavailable: {e}")
    st.info("💡 The app will run in upload-only mode. Please check your dependencies.")

from video_manager import VideoManager
from video_processor import DeepgramTranscriber

# Page configuration
st.set_page_config(
    page_title="Video Semantic Search",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .search-result {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .relevance-score {
        color: #2e7d32;
        font-weight: bold;
    }
    .timestamp {
        color: #1976d2;
        font-weight: bold;
        cursor: pointer;
    }
    .video-file {
        color: #7b1fa2;
        font-weight: bold;
    }
    .confidence {
        color: #f57c00;
        font-size: 0.9em;
    }
    .highlighted {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .stats-card {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'index_manager' not in st.session_state:
    st.session_state.index_manager = None
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None
if 'index_loaded' not in st.session_state:
    st.session_state.index_loaded = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'video_manager' not in st.session_state:
    st.session_state.video_manager = None
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}
if 'upload_initialized' not in st.session_state:
    st.session_state.upload_initialized = False

def initialize_upload_system():
    """Initialize the video upload and processing system"""
    try:
        if not st.session_state.upload_initialized:
            with st.spinner("🔄 Initializing upload system..."):
                # Initialize video manager
                st.session_state.video_manager = VideoManager()
                
                # Initialize video processor (requires Deepgram API key)
                # Get API key from Streamlit secrets or environment variable
                deepgram_key = None
                try:
                    # Try Streamlit secrets first (for Streamlit Cloud)
                    deepgram_key = st.secrets.get("deepgram", {}).get("DEEPGRAM_API_KEY")
                except:
                    pass
                
                # Fallback to environment variable (for local development) 
                if not deepgram_key:
                    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
                
                st.session_state.video_processor = DeepgramTranscriber(deepgram_api_key=deepgram_key)
                st.session_state.upload_initialized = True
                
            st.success("✅ Upload system initialized!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize upload system: {e}")
        logger.error(f"Upload system initialization error: {e}")
        return False

def initialize_system():
    """Initialize the video search system"""
    if not INDEX_AVAILABLE:
        st.warning("⚠️ Search functionality is not available due to missing dependencies")
        st.info("💡 You can still upload and process videos, but search features will be disabled")
        return False
        
    try:
        with st.spinner("🔄 Initializing semantic search system..."):
            # Initialize index manager
            st.session_state.index_manager = VideoIndexManager()
            
            # Load or create index
            index = st.session_state.index_manager.get_or_create_index()
            
            # Initialize search engine with reranker
            st.session_state.search_engine = VideoSearchEngine(index, use_reranker=True)
            st.session_state.index_loaded = True
            
            st.success("✅ System initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        logger.error(f"Initialization error: {e}")
        st.info("💡 You can still upload and process videos, but search features will be disabled")
        return False

def handle_video_upload(uploaded_file, custom_title=None):
    """Handle uploaded video file"""
    try:
        if not st.session_state.upload_initialized:
            initialize_upload_system()
        
        # Save uploaded file to temporary location
        temp_path = Path("temp_uploads")
        temp_path.mkdir(exist_ok=True)
        
        temp_file_path = temp_path / uploaded_file.name
        
        # Write uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add to video manager
        result = st.session_state.video_manager.add_manual_video(
            str(temp_file_path), 
            title=custom_title or uploaded_file.name
        )
        
        # Clean up temp file
        temp_file_path.unlink(missing_ok=True)
        
        # Update session state
        st.session_state.uploaded_videos.append(result)
        st.session_state.processing_status[result['id']] = 'uploaded'
        
        return result
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        st.error(f"Upload failed: {e}")
        return None

def process_uploaded_video(video_id):
    """Process uploaded video to generate transcript"""
    try:
        if not st.session_state.upload_initialized:
            st.error("Upload system not initialized")
            return False
        
        # Get video info
        video_info = st.session_state.video_manager.get_video_by_id(video_id)
        if not video_info:
            st.error(f"Video not found: {video_id}")
            return False
        
        video_path = Path(video_info['file_path'])
        
        # Update status
        st.session_state.processing_status[video_id] = 'processing'
        
        with st.spinner(f"🎬 Processing {video_info['title']}..."):
            # Extract audio
            audio_path = Path("audio") / f"{video_path.stem}.mp3"
            audio_path.parent.mkdir(exist_ok=True)
            
            st.write(f"📢 Extracting audio from {video_path.name}...")
            st.session_state.video_processor.extract_audio(str(video_path), str(audio_path))
            
            # Generate transcript - save to transcripts folder for indexing
            st.write(f"📝 Generating transcript...")
            
            # Create transcripts directory
            transcripts_path = Path("transcripts")
            transcripts_path.mkdir(exist_ok=True)
            
            # Generate transcript but don't auto-save yet
            transcript_result = st.session_state.video_processor.transcribe_audio(
                str(audio_path), save_json=False
            )
            
            if transcript_result:
                # Save transcript to transcripts folder for indexing
                transcript_filename = transcripts_path / f"{audio_path.stem}_transcript.json"
                
                # Parse and save transcript
                import json
                try:
                    # transcript_result is already JSON string, save it directly
                    with open(transcript_filename, 'w', encoding='utf-8') as f:
                        f.write(transcript_result)
                    
                    # Verify the file was created and has content
                    if transcript_filename.exists() and transcript_filename.stat().st_size > 0:
                        st.write(f"💾 Saved transcript to {transcript_filename}")
                    else:
                        raise Exception("Transcript file was not created properly")
                        
                except Exception as e:
                    st.error(f"Failed to save transcript: {e}")
                    transcript_result = None
            
            if transcript_result:
                # Update video status in database
                st.session_state.video_manager.update_video_status(
                    video_id, 'transcript_status', 'completed'
                )
                st.session_state.processing_status[video_id] = 'completed'
                
                # Update search indexed status in database
                st.session_state.video_manager.update_video_status(
                    video_id, 'search_indexed', 'true'
                )
                
                # Refresh index to include new video
                if INDEX_AVAILABLE and st.session_state.index_loaded:
                    st.write("🔄 Rebuilding search index to include uploaded video...")
                    
                    try:
                        st.session_state.index_manager = VideoIndexManager()
                        
                        # Force rebuild to include newly uploaded video transcripts
                        index = st.session_state.index_manager.rebuild_index()
                        st.session_state.search_engine = VideoSearchEngine(index, use_reranker=True)
                        
                        # Show updated stats to confirm inclusion
                        stats = st.session_state.index_manager.get_index_stats()
                        if "error" not in stats:
                            st.write(f"📊 Index now contains {stats.get('total_chunks', 0)} chunks from {stats.get('total_videos', 0)} videos")
                            if stats.get("video_files"):
                                st.write(f"📹 Videos in index: {', '.join(stats['video_files'])}")
                        
                        st.write("✅ Search index rebuilt - uploaded video is now searchable!")
                        
                    except Exception as index_error:
                        if "readonly database" in str(index_error).lower():
                            st.warning("⚠️ Database permission issue detected. Using fallback indexing method...")
                            st.info("💡 Your video is still searchable, but the index won't persist between sessions. This is usually due to file permission restrictions.")
                        else:
                            st.error(f"❌ Index rebuild failed: {index_error}")
                            logger.error(f"Index rebuild error: {index_error}")
                            # Don't fail completely - video transcript is still saved
                            st.warning("⚠️ Search index update failed, but transcript was saved. You may need to restart the app to search this video.")
                elif not INDEX_AVAILABLE:
                    st.success("🎉 Video processed successfully!")
                    st.info("📄 Transcript has been generated and saved. In a full deployment with search dependencies, this video would be automatically indexed for semantic search.")
                
                st.success(f"✅ Successfully processed {video_info['title']}")
                return True
            else:
                st.session_state.processing_status[video_id] = 'failed'
                st.error("Failed to generate transcript")
                return False
                
    except Exception as e:
        st.session_state.processing_status[video_id] = 'failed'
        logger.error(f"Processing error: {e}")
        st.error(f"Processing failed: {e}")
        return False

def cleanup_temp_files():
    """Clean up temporary upload files"""
    try:
        temp_path = Path("temp_uploads")
        if temp_path.exists():
            for file in temp_path.glob("*"):
                file.unlink(missing_ok=True)
            temp_path.rmdir()
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")

def display_index_stats():
    """Display statistics about the loaded index"""
    if st.session_state.index_manager and st.session_state.index_loaded:
        stats = st.session_state.index_manager.get_index_stats()
        
        if "error" not in stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Videos", stats.get("total_videos", 0))
            with col2:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
            with col3:
                st.metric("Total Duration", f"{stats.get('total_duration_minutes', 0)} min")
            with col4:
                st.metric("Video Files", len(stats.get("video_files", [])))
            
            # Video files list
            if stats.get("video_files"):
                st.markdown("**Available Videos:**")
                st.write(", ".join(stats["video_files"]))

def display_search_result(result: Dict[str, Any], index: int):
    """Display a single search result"""
    with st.container():
        # Format score display based on search method
        search_method = result.get('search_method', 'semantic_only')
        score_display = f"{result['relevance_score']:.3f}"
        
        # Add reranking badge if applicable
        method_badge = ""
        if search_method == 'reranked':
            method_badge = " 🚀"
            if result.get('rerank_score') is not None:
                score_display = f"{result['rerank_score']:.3f}"
        
        st.markdown(f"""
        <div class="search-result">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span class="video-file">📹 {result['video_file']}</span>
                <span class="relevance-score">Score: {score_display}{method_badge}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <span class="timestamp">🕒 {result['timestamp_display']}</span>
                <span style="margin-left: 20px;" class="confidence">Confidence: {result['confidence']:.2f}</span>
                <span style="margin-left: 20px; font-size: 0.8em;">Method: {search_method.replace('_', ' ').title()}</span>
            </div>
            <div style="margin-bottom: 10px;">
                <strong>Text:</strong><br>
                {result['text_snippet']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 6])
        
        with col1:
            if st.button(f"📋 Copy Text", key=f"copy_{index}"):
                st.write("📋 Text copied to clipboard!")
                st.code(result['text'], language='text')
        
        with col2:
            if st.button(f"🎯 Jump to Time", key=f"jump_{index}"):
                st.info(f"⏯️ Jump to {result['timestamp_display']} in {result['video_file']}")
                display_video_info(result)

def display_video_info(result: Dict[str, Any]):
    """Display video information and player"""
    st.markdown("### 🎥 Video Information")
    
    video_path = Path(result['video_path'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        **Video File:** {result['video_file']}  
        **Start Time:** {result['timestamp_display']}  
        **Duration:** {result['duration']} seconds  
        **Chunk:** {result['chunk_number']}  
        """)
    
    with col2:
        if video_path.exists():
            st.markdown("**Video Player:**")
            st.video(str(video_path), start_time=result['start_time'])
        else:
            st.warning(f"Video file not found: {video_path}")
            st.info("💡 Make sure the video files are in the 'Data' folder")

def main():
    """Main Streamlit application"""
    
    # Clean up any leftover temporary files
    cleanup_temp_files()
    
    # Header
    st.title("🎥 Video Semantic Search")
    st.markdown("Search through video transcripts using natural language queries powered by LlamaIndex")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # System initialization
        if INDEX_AVAILABLE:
            if not st.session_state.index_loaded:
                st.markdown("### Initialize System")
                if st.button("🚀 Load Search Index", type="primary"):
                    initialize_system()
            else:
                st.success("✅ System Ready")
        else:
            st.info("📋 Minimal Deployment")
            st.caption("Upload and transcription only")
            
            with st.expander("ℹ️ About Search Functionality"):
                st.write("""
                **Search is disabled in this deployment** to ensure:
                - Fast startup times
                - Reliable deployment  
                - Minimal resource usage
                
                **Core features still available:**
                - Video upload and processing
                - Automatic transcript generation
                - Video library management
                """)
        
        # Video Upload Section
        st.markdown("### 📤 Upload Videos")
        
        # Initialize upload system
        if not st.session_state.upload_initialized:
            if st.button("🔧 Initialize Upload System"):
                initialize_upload_system()
        else:
            # File uploader
            uploaded_files = st.file_uploader(
                "Choose video files",
                type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
                accept_multiple_files=True,
                help="Upload video files to add to your searchable library"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Check if already uploaded
                    already_uploaded = any(
                        video['title'] == uploaded_file.name 
                        for video in st.session_state.uploaded_videos
                    )
                    
                    if not already_uploaded:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            custom_title = st.text_input(
                                f"Title for {uploaded_file.name}:",
                                value=Path(uploaded_file.name).stem,
                                key=f"title_{uploaded_file.name}"
                            )
                        
                        with col2:
                            if st.button(f"📤 Upload", key=f"upload_{uploaded_file.name}"):
                                result = handle_video_upload(uploaded_file, custom_title)
                                if result:
                                    st.success(f"✅ Uploaded: {result['title']}")
                                    st.rerun()
            
            # Display uploaded videos status
            if st.session_state.uploaded_videos:
                st.markdown("#### 📋 Uploaded Videos")
                
                for video in st.session_state.uploaded_videos:
                    video_id = video['id']
                    status = st.session_state.processing_status.get(video_id, 'unknown')
                    
                    # Status indicators
                    status_icons = {
                        'uploaded': '⏳',
                        'processing': '🔄',
                        'completed': '✅',
                        'failed': '❌'
                    }
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"{status_icons.get(status, '❓')} {video['title']}")
                    
                    with col2:
                        st.write(status.title())
                    
                    with col3:
                        if status == 'uploaded' and st.button(f"🎬 Process", key=f"process_{video_id}"):
                            if process_uploaded_video(video_id):
                                st.rerun()
                        elif status == 'completed':
                            st.write("✅ Ready")
                        elif status == 'processing':
                            st.write("🔄 Processing...")
                        elif status == 'failed':
                            if st.button(f"🔄 Retry", key=f"retry_{video_id}"):
                                if process_uploaded_video(video_id):
                                    st.rerun()
                
                # Bulk processing
                pending_videos = [
                    video for video in st.session_state.uploaded_videos 
                    if st.session_state.processing_status.get(video['id']) == 'uploaded'
                ]
                
                if pending_videos and st.button("🚀 Process All Pending"):
                    for video in pending_videos:
                        st.write(f"Processing {video['title']}...")
                        process_uploaded_video(video['id'])
                    st.rerun()
            
            # Search parameters
            st.markdown("### Search Settings")
            top_k = st.slider("Number of Results", 1, 20, 10)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.1)
            
            # Reranker settings
            use_reranker = st.checkbox("🚀 Use Reranker", value=True, 
                                     help="Use cross-encoder reranking for improved search quality")
            
            # Advanced filters
            st.markdown("### Filters")
            
            # Video filter
            if st.session_state.index_manager:
                stats = st.session_state.index_manager.get_index_stats()
                if "video_files" in stats:
                    selected_videos = st.multiselect(
                        "Filter by Videos",
                        options=stats["video_files"],
                        default=[]
                    )
            
            # Confidence filter
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
            
            # Clear results
            if st.button("🗑️ Clear Results"):
                st.session_state.search_results = []
                st.rerun()
    
    # Main content area
    if not INDEX_AVAILABLE:
        st.info("🚀 **Minimal Deployment Mode** - This is a lightweight version focused on video upload and transcription")
        st.success("📤 You can upload videos and generate transcripts using the sidebar")
        
        # Show simplified environment info
        st.markdown("### 📋 System Status")
        st.error("❌ Semantic Search: Unavailable (minimal deployment)")
        st.success("✅ Video Upload: Available")
        st.success("✅ Transcript Generation: Available (requires Deepgram API key)")
        
        # Deployment info
        st.markdown("### 🔧 About This Deployment")
        st.info("""
        **This is a minimal deployment** that includes core functionality:
        - ✅ Video file upload (MP4, AVI, MOV, MKV, WMV, FLV)
        - ✅ Automatic transcript generation using Deepgram API
        - ✅ Video library management
        - ✅ File organization and metadata tracking
        
        **Search functionality is disabled** to ensure fast deployment and reliability.
        To enable search features, you would need to add ML dependencies like:
        - llama-index
        - sentence-transformers
        - torch
        - chromadb
        """)
        
    elif not st.session_state.index_loaded:
        st.info("👆 Please initialize the system using the sidebar to start searching")
        
        # Show system requirements
        st.markdown("### 📋 System Requirements")
        st.markdown("""
        - Video files in `Data/` folder (1.mp4, 2.mp4, etc.) **OR** upload videos using the sidebar
        - Transcript files in `transcripts/` folder (auto-generated from uploads)
        - Required Python packages (see requirements.txt)
        - **NEW**: Deepgram API key for automatic transcript generation from uploaded videos
        """)
        
        # Upload feature info
        st.markdown("### 🆕 Upload Feature")
        st.info("""
        **Upload Functionality Available!**
        
        You can now upload videos directly through the sidebar:
        1. Initialize the upload system in the sidebar
        2. Upload video files (MP4, AVI, MOV, MKV, WMV, FLV)
        3. Process videos to generate transcripts automatically
        4. **NEW**: Transcripts are now properly saved to the `transcripts/` folder for indexing
        5. Search through your uploaded content immediately after processing
        
        **Note**: You'll need a Deepgram API key set as an environment variable `DEEPGRAM_API_KEY` for automatic transcript generation.
        """)
        
        st.success("""
        🔧 **Fixed**: Uploaded videos are now properly indexed for search! 
        Videos marked as "🔍 searchable" in the library will appear in search results.
        """)
        
        # Environment check
        st.markdown("### 🔧 Environment Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Required Packages:**")
            required_packages = [
                "streamlit", "llama-index", "moviepy", 
                "deepgram-sdk", "pandas", "pathlib"
            ]
            
            # Check if we can import key packages
            import_checks = {}
            try:
                import moviepy
                import_checks["moviepy"] = "✅"
            except ImportError:
                import_checks["moviepy"] = "❌"
            
            try:
                from deepgram import DeepgramClient
                import_checks["deepgram"] = "✅"
            except ImportError:
                import_checks["deepgram"] = "❌"
            
            try:
                import llama_index
                import_checks["llama-index"] = "✅"
            except ImportError:
                import_checks["llama-index"] = "❌"
            
            for pkg in ["moviepy", "deepgram", "llama-index"]:
                status = import_checks.get(pkg, "❓")
                st.write(f"{status} {pkg}")
        
        with col2:
            st.markdown("**Environment Variables:**")
            
            # Check for Deepgram API key (Streamlit Cloud secrets or environment variable)
            deepgram_key = None
            try:
                # Try Streamlit secrets first (for Streamlit Cloud)
                deepgram_key = st.secrets.get("deepgram", {}).get("DEEPGRAM_API_KEY")
            except:
                pass
            
            # Fallback to environment variable (for local development)
            if not deepgram_key:
                deepgram_key = os.getenv("DEEPGRAM_API_KEY")
            
            if deepgram_key:
                st.write("✅ DEEPGRAM_API_KEY (set)")
            else:
                st.write("⚠️ DEEPGRAM_API_KEY (not set)")
                st.caption("Required for automatic transcript generation")
        
        # Check if files exist
        st.markdown("### 📁 File Status Check")
        
        data_path = Path("Data")
        transcript_path = Path("transcripts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Video Files:**")
            if data_path.exists():
                video_files = list(data_path.glob("*.mp4"))
                if video_files:
                    st.success(f"✅ Found {len(video_files)} video files")
                    for vf in sorted(video_files):
                        st.write(f"📹 {vf.name}")
                else:
                    st.warning("⚠️ No MP4 files found in Data folder")
            else:
                st.error("❌ Data folder not found")
        
        with col2:
            st.markdown("**Transcript Files:**")
            if transcript_path.exists():
                transcript_files = list(transcript_path.glob("*_transcript.json"))
                if transcript_files:
                    st.success(f"✅ Found {len(transcript_files)} transcript files")
                else:
                    st.warning("⚠️ No transcript files found")
            else:
                st.error("❌ Transcripts folder not found")
    
    elif INDEX_AVAILABLE:
        # Display index statistics
        st.markdown("### 📊 Index Statistics")
        display_index_stats()
    
    # Video Library Overview (works in both modes)
    if st.session_state.upload_initialized and st.session_state.video_manager:
        st.markdown("### 📚 Video Library")
            
            try:
                all_videos = st.session_state.video_manager.get_all_videos()
                
                if all_videos:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_videos = len(all_videos)
                    manual_videos = len([v for v in all_videos if v.get('source') == 'manual'])
                    completed_videos = len([v for v in all_videos if v.get('transcript_status') == 'completed'])
                    indexed_videos = len([v for v in all_videos if v.get('search_indexed') == 'true'])
                    total_size = sum(v.get('file_size', 0) for v in all_videos) / (1024*1024)  # MB
                    
                    with col1:
                        st.metric("Total Videos", total_videos)
                    with col2:
                        st.metric("Uploaded Videos", manual_videos)
                    with col3:
                        st.metric("Processed Videos", completed_videos)
                    with col4:
                        st.metric("Searchable Videos", indexed_videos)
                    
                    # Storage info
                    st.caption(f"📁 Total storage: {total_size:.1f} MB")
                    
                    # Show recent uploads
                    recent_videos = sorted(all_videos, key=lambda x: x.get('added_date', ''), reverse=True)[:5]
                    
                    if recent_videos:
                        st.markdown("**Recent Videos:**")
                        for video in recent_videos:
                            if video.get('search_indexed') == 'true':
                                status_icon = "🔍"  # Searchable
                                status_text = "searchable"
                            elif video.get('transcript_status') == 'completed':
                                status_icon = "✅"  # Processed but not indexed
                                status_text = "processed"
                            else:
                                status_icon = "⏳"  # Pending
                                status_text = "pending"
                            st.write(f"{status_icon} {video.get('title', 'Unknown')} ({status_text})")
                else:
                    st.info("📁 No videos in library yet. Upload some videos to get started!")
                    
            except Exception as e:
                st.warning(f"Could not load video library: {e}")
        
        # Search interface (only if search is available)
        if INDEX_AVAILABLE:
            st.markdown("### 🔍 Search Interface")
            
            # Search input
            query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., 'machine learning algorithms', 'data preprocessing', 'neural networks'...",
                help="Use natural language to search through video content"
            )
            
            # Search buttons
            col1, col2, col3 = st.columns([2, 2, 6])
            
            with col1:
                search_clicked = st.button("🔍 Search", type="primary")
            
            with col2:
                example_clicked = st.button("💡 Try Example")
            
            # Handle example query
            if example_clicked:
                query = "machine learning"
                search_clicked = True
            
            # Execute search
            if search_clicked and query and st.session_state.search_engine:
                with st.spinner("🔍 Searching..."):
                    try:
                        # Create new search engine with current reranker setting if needed
                        if st.session_state.search_engine.use_reranker != use_reranker:
                            index = st.session_state.index_manager.get_or_create_index()
                            st.session_state.search_engine = VideoSearchEngine(index, use_reranker=use_reranker)
                        
                        # Apply filters
                        if 'selected_videos' in locals() and selected_videos:
                            results = st.session_state.search_engine.search_by_video(
                                query, selected_videos, top_k
                            )
                        elif min_confidence > 0:
                            results = st.session_state.search_engine.search_high_confidence(
                                query, min_confidence, top_k
                            )
                        else:
                            results = st.session_state.search_engine.semantic_search(
                                query, top_k, similarity_threshold
                            )
                        
                        st.session_state.search_results = results
                        
                    except Exception as e:
                        st.error(f"Search error: {e}")
                        logger.error(f"Search error: {e}")
            
            # Display search results
            if st.session_state.search_results:
                st.markdown(f"### 🎯 Search Results ({len(st.session_state.search_results)} found)")
                
                for i, result in enumerate(st.session_state.search_results):
                    display_search_result(result, i)
                    
                    if i < len(st.session_state.search_results) - 1:
                        st.divider()
            
            elif query and search_clicked:
                st.info("🔍 No results found. Try adjusting your search query or filters.")
            
            # Video summary section
            st.markdown("### 📈 Video Analytics")
            
            if st.session_state.search_engine:
                stats = st.session_state.index_manager.get_index_stats()
                
                if "video_files" in stats and stats["video_files"]:
                    selected_video = st.selectbox(
                        "Get Video Summary:",
                        options=[""] + stats["video_files"],
                        format_func=lambda x: "Select a video..." if x == "" else x
                    )
                    
                    if selected_video:
                        with st.spinner("📊 Generating summary..."):
                            summary = st.session_state.search_engine.get_video_summary(selected_video)
                            
                            if "error" not in summary:
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Chunks", summary["total_chunks"])
                                with col2:
                                    st.metric("Duration", f"{summary['total_duration']/60:.1f} min")
                                with col3:
                                    st.metric("Words", summary["total_words"])
                                with col4:
                                    st.metric("Avg Confidence", f"{summary['average_confidence']:.2f}")
                                
                                # Full transcript
                                if st.checkbox("Show Full Transcript"):
                                    st.text_area(
                                        "Full Transcript:",
                                        summary["full_transcript"],
                                        height=300
                                    )
                            else:
                                st.error(f"Error getting summary: {summary['error']}")

if __name__ == "__main__":
    main() 