import streamlit as st
import logging
from pathlib import Path
import json
import time
from typing import List, Dict, Any
import pandas as pd

# Import our custom modules
from video_index_manager import VideoIndexManager
from video_search_engine import VideoSearchEngine

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

def initialize_system():
    """Initialize the video search system"""
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
        return False

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
    
    # Header
    st.title("🎥 Video Semantic Search")
    st.markdown("Search through video transcripts using natural language queries powered by LlamaIndex")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # System initialization
        if not st.session_state.index_loaded:
            st.markdown("### Initialize System")
            if st.button("🚀 Load Search Index", type="primary"):
                initialize_system()
        else:
            st.success("✅ System Ready")
            
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
    if not st.session_state.index_loaded:
        st.info("👆 Please initialize the system using the sidebar to start searching")
        
        # Show system requirements
        st.markdown("### 📋 System Requirements")
        st.markdown("""
        - Video files in `Data/` folder (1.mp4, 2.mp4, etc.)
        - Transcript files in `transcripts/` folder
        - Required Python packages (see requirements_streamlit.txt)
        """)
        
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
    
    else:
        # Display index statistics
        st.markdown("### 📊 Index Statistics")
        display_index_stats()
        
        # Search interface
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