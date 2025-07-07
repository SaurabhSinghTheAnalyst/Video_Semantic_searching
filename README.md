# Video Semantic Search with LlamaIndex

This project processes video files into searchable transcripts and provides a powerful semantic search interface powered by LlamaIndex and local embeddings.

## 🌟 Features

- **🎥 Video Processing**: Automatically splits long videos into 1-minute chunks
- **🎵 Audio Extraction**: Extracts high-quality audio from video chunks  
- **🗣️ Speech-to-Text**: Converts audio to text with word-level timestamps using Deepgram
- **🔍 Semantic Search**: Advanced natural language search through video transcripts
- **🚀 Cross-Encoder Reranking**: Enhanced search quality with two-stage retrieval pipeline
- **🏠 Local Embeddings**: Uses local HuggingFace models (no API keys required for search)
- **🖥️ Interactive UI**: Beautiful Streamlit interface with video player integration
- **🎯 Multiple Search Modes**: Search by video, time range, confidence level
- **📊 Video Analytics**: Get summaries and statistics for individual videos
- **⚡ Batch Processing**: Handles multiple video files efficiently

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
python run_app.py
```
This script will automatically install dependencies and launch the app.

### Option 2: Manual Setup

1. **Install Dependencies**:
```bash
pip install -r requirements_streamlit.txt
```

2. **Launch the Application**:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Prerequisites

### For Video Processing & Transcription:
1. Get a Deepgram API key from [https://console.deepgram.com/](https://console.deepgram.com/)
2. Create a `.env` file in the project root:
```
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

### For Semantic Search:
- **No API keys required!** The system uses local HuggingFace embeddings
- Internet connection needed only for initial model download

## 🎯 Usage Guide

### Step 1: Prepare Your Data

1. **Add Video Files**: Place your MP4 files in the `Data/` folder:
   ```
   Data/
   ├── 1.mp4
   ├── 2.mp4
   └── ...
   ```

2. **Process Videos and Generate Transcripts** (if not done already):
   ```bash
   python video_processor.py
   ```
   This automatically:
   - Extracts audio from MP4 files in the `Data/` folder
   - Generates transcripts using Deepgram's speech-to-text API
   - Saves transcripts as JSON files in the `transcripts/` folder

### Step 2: Use the Search Interface

1. **Launch the App**:
   ```bash
   python run_app.py
   ```

2. **Initialize the Search Index**:
   - Click "🚀 Load Search Index" in the sidebar
   - The system will create embeddings from your transcripts (first-time setup takes a few minutes)

3. **Start Searching**:
   - Enter natural language queries like:
     - "machine learning algorithms"
     - "data preprocessing techniques" 
     - "neural network architecture"
   - Click search results to jump to specific video timestamps
   - Use filters to narrow down results by video, confidence, etc.

## 🔍 Search Features

### Basic Search
- **Natural Language**: Search using everyday language
- **Semantic Understanding**: Finds related concepts, not just exact matches
- **Relevance Scoring**: Results ranked by semantic similarity

### 🚀 Advanced Reranking (NEW!)
- **Two-Stage Pipeline**: Semantic retrieval → Cross-encoder reranking → Final results
- **Enhanced Quality**: Up to 0.369 score improvement for relevant results
- **Better Ordering**: Reranker changes top-3 results order for improved relevance
- **Toggle Control**: Enable/disable reranking per query in the UI
- **Performance**: ~0.36s additional overhead for significantly better results

### Advanced Filters
- **Video-Specific**: Search within specific video files
- **Time Range**: Find content within time intervals
- **Confidence Level**: Filter by transcription confidence
- **Result Count**: Adjust number of results shown
- **Reranker Toggle**: Enable/disable cross-encoder reranking

### Search Results Include:
- **Video File**: Which video contains the match
- **Timestamp**: Exact time location (MM:SS format)
- **Text Snippet**: Preview of matching content
- **Relevance Score**: How well it matches your query (with rerank scores if enabled)
- **Confidence**: Transcription quality score
- **Search Method**: Visual indicator (🚀) for reranked results

### Video Player Integration
- **Jump to Timestamp**: Click results to seek to exact moments
- **Inline Player**: Watch videos directly in the interface
- **Context Preservation**: Maintains search context while playing

## 📁 Project Structure

```
SentenceTransformer/
├── Data/                           # Input video files
│   ├── 1.mp4
│   ├── 2.mp4
│   └── ...
├── audio/                          # Generated audio files  
│   ├── 1.mp3                      # Extracted audio files
│   ├── 2.mp3
│   └── ...
├── transcripts/                    # Generated transcripts
│   ├── 1_transcript.json          # Transcript JSON files
│   ├── 2_transcript.json
│   └── ...
├── embeddings/                     # Search index (auto-created)
│   ├── video_index/               # LlamaIndex storage
│   └── chroma_db/                 # ChromaDB vector store
├── app.py                          # Main Streamlit application
├── video_index_manager.py          # Index creation & management
├── video_search_engine.py          # Semantic search engine
├── video_processor.py              # Video processing & transcription script
├── run_app.py                      # Automated startup script
├── requirements_streamlit.txt       # Streamlit app dependencies
├── requirements.txt                # Basic processing dependencies
└── README.md                       # This file
```

## 🔧 Configuration Options

### Embedding Model
You can change the embedding model in `video_index_manager.py`:
```python
# Default: Fast and lightweight
VideoIndexManager(embed_model_name="BAAI/bge-small-en-v1.5")

# Alternative: Better quality, larger size
VideoIndexManager(embed_model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Search Parameters
Adjust search behavior in the Streamlit sidebar:
- **Similarity Threshold**: Higher = more precise matches
- **Number of Results**: How many results to show
- **Confidence Filter**: Minimum transcription quality

### Video Processing
Customize video processing in `video_processor.py`:
```python
transcriber = DeepgramTranscriber(deepgram_api_key=DEEPGRAM_API_KEY)

# Process all videos with custom settings
transcriber.process_all_videos_sync_simple(
    data_folder="Data",
    audio_folder="audio", 
    output_folder="transcripts"
)
```

## 📊 Technical Details

### Embedding & Reranking Models
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (bi-encoder for initial retrieval)
- **Reranker Model**: `BAAI/bge-reranker-base` (cross-encoder for quality reranking)
- **Type**: Local HuggingFace models (no API required)
- **Performance**: Excellent semantic understanding with enhanced relevance
- **Size**: ~120MB embedding model + ~1GB reranker model (one-time download)

### Search Pipeline
```
Query → Semantic Search (Retrieval) → Cross-Encoder Reranking → Final Results
```
- **Stage 1**: Bi-encoder retrieves top candidates (top_k × 5)
- **Stage 2**: Cross-encoder reranks using query-document interaction analysis
- **Result**: Superior relevance ranking and query understanding

### Vector Database
- **ChromaDB**: Efficient similarity search
- **Persistent Storage**: Index saved locally for reuse
- **Incremental Updates**: Add new videos without rebuilding

### Search Engine
- **LlamaIndex**: Advanced retrieval and query processing
- **Cross-Encoder Reranking**: Enhanced relevance with BAAI/bge-reranker-base
- **Flexible Architecture**: Toggle reranking on/off per query
- **Metadata Integration**: Rich video timing and quality data

## 🧪 Reranker Testing & Evaluation

### Test Scripts Available

1. **Simple Evaluation**:
   ```bash
   python simple_rag_evaluation.py
   ```
   Runs comprehensive tests comparing semantic-only vs reranked results

2. **Specific Query Testing**:
   ```bash
   python simple_rag_evaluation.py "your search query here"
   ```

### Example Results

**Query: "contextual retrieval from Anthropic"**

*Without Reranker:*
1. Score: 0.625 | "So you can potentially use a simple prompt..."
2. Score: 0.604 | "In a previous video, we looked at contextual..."
3. Score: 0.601 | "So here's the whole implementation..."

*With Reranker:*
1. Score: 0.998 🚀 | "In a previous video, we looked at contextual..." (+0.393)
2. Score: 0.864 🚀 | "So here's the whole implementation..." (+0.263)
3. Score: 0.496 🚀 | "Now the prompt that you see here..." (-0.101)

**Results**: ✅ Top 3 order changed | 📊 3/3 results reranked | 🎯 Best improvement: +0.393

### Performance Metrics
- **Quality Improvement**: Up to 100% of queries show improved ordering
- **Score Enhancement**: Average improvements of 0.15-0.40 points
- **Time Overhead**: ~0.25s per query for reranking
- **Memory Usage**: Additional ~1GB for reranker model

## 🛠️ Troubleshooting

### Common Issues

1. **"No transcript files found"**
   - Run `python video_processor.py` to generate transcripts
   - Check that `.env` file contains valid Deepgram API key

2. **"Embedding model failed to load"**
   - Ensure stable internet connection for initial download
   - Try restarting the application

3. **"Search returns no results"**
   - Lower the similarity threshold in sidebar
   - Try broader search terms
   - Check if index was properly created

4. **"Video player not working"**
   - Ensure video files are in `Data/` folder
   - Check that video files are in MP4 format

### Performance Tips

- **Index Creation**: First-time setup takes 2-5 minutes depending on content volume
- **Search Speed**: Subsequent searches are very fast (<1 second)
- **Memory Usage**: ~500MB-1GB RAM for typical video collections
- **Storage**: Embeddings require ~10-50MB per hour of video content

## 🚀 Advanced Usage

### Custom Search Modes
The search engine supports multiple query types:

```python
# Search specific videos
search_engine.search_by_video("machine learning", ["1.mp4", "2.mp4"])

# Search time ranges
search_engine.search_by_time_range("algorithms", start_time=300, end_time=900)

# High-confidence only
search_engine.search_high_confidence("deep learning", min_confidence=0.8)
```

### Video Analytics
Get detailed insights for each video:
- Total duration and word count
- Average transcription confidence
- Full transcript text
- Chunk-by-chunk breakdown

### Batch Operations
Process multiple videos efficiently:
- Parallel audio extraction
- Concurrent transcription
- Incremental index updates

---

🎉 **Happy Searching!** Your videos are now semantically searchable with powerful natural language queries. 