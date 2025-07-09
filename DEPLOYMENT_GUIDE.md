# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

### 1. **Required Files ✅**
Your project already includes all necessary files:
- ✅ `app.py` (main Streamlit application)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)
- ✅ `.streamlit/secrets.toml.example` (secrets template)
- ✅ All supporting Python modules (`video_manager.py`, `video_processor.py`, etc.)

### 2. **Required API Keys**
- **Deepgram API Key**: Required for automatic video transcription
  - Sign up at [https://deepgram.com](https://deepgram.com)
  - Get your API key from the dashboard

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Verify `.gitignore`**: 
   - Ensure sensitive files are excluded (already configured)
   - Video files, audio files, and embeddings are ignored for size reasons

### Step 2: Set Up Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [https://share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Choose your branch (usually `main`)

### Step 3: Configure Secrets

1. **In Streamlit Cloud Dashboard**:
   - Go to your app settings
   - Click "Secrets" tab
   - Add the following content:

```toml
[deepgram]
DEEPGRAM_API_KEY = "your_actual_deepgram_api_key_here"
```

2. **Replace with Your Actual Key**:
   - Copy your Deepgram API key
   - Paste it in place of `your_actual_deepgram_api_key_here`

### Step 4: Deploy

1. **Click "Deploy"**
2. **Wait for Build**: First deployment takes 5-10 minutes
3. **Check Logs**: Monitor build progress in the dashboard

## ⚙️ Configuration Details

### Resource Limits
- **Upload Size**: Configured for 1GB max upload
- **Memory Usage**: Optimized for Streamlit Cloud limits
- **Processing**: Videos process one at a time to avoid timeouts

### File Storage
- **Temporary**: Uploaded videos stored temporarily during processing
- **Persistence**: Database and transcripts persist between sessions
- **Cleanup**: Automatic cleanup of temporary files

## 🎯 Key Features for Deployment

### ✅ What Works Out of the Box
- **Video Upload**: Direct file upload interface
- **Transcript Generation**: Automatic transcription via Deepgram
- **Semantic Search**: Full search functionality
- **Video Library**: Complete video management
- **Responsive UI**: Works on desktop and mobile

### ⚠️ Deployment Considerations
- **Cold Starts**: First load may take 30-60 seconds
- **File Size**: Large videos (>100MB) may timeout on upload
- **Processing Time**: Videos process in real-time (no queue system)
- **Storage**: No permanent video storage (files cleaned up periodically)

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. **Build Fails**
```
Error: Package conflicts in requirements.txt
```
**Solution**: The `requirements.txt` has been optimized for Streamlit Cloud

#### 2. **API Key Not Working**
```
Error: DEEPGRAM_API_KEY not found
```
**Solution**: 
- Check secrets configuration in Streamlit Cloud dashboard
- Ensure key is under `[deepgram]` section
- Verify key is valid in Deepgram dashboard

#### 3. **Upload Fails**
```
Error: File too large
```
**Solution**:
- File size limit is 1GB
- Try smaller videos first
- Check internet connection stability

#### 4. **Processing Timeout**
```
Error: Processing timeout
```
**Solution**:
- Use videos shorter than 30 minutes for best results
- Check Deepgram API quota
- Try processing during off-peak hours

### Debug Mode
Add this to your secrets for detailed logging:
```toml
[debug]
ENABLE_LOGGING = "true"
```

## 🔄 Updates & Maintenance

### Updating Your App
1. **Push changes to GitHub**:
   ```bash
   git add .
   git commit -m "Update app"
   git push origin main
   ```

2. **Automatic Redeployment**: Streamlit Cloud auto-deploys on git push

### Managing Storage
- **Cleanup**: App automatically cleans temporary files
- **Database**: Video catalog persists between deployments
- **Transcripts**: Generated transcripts are preserved

## 🎉 Post-Deployment

### Testing Your Deployment
1. **Upload Test Video**: Try a small video file first
2. **Generate Transcript**: Verify Deepgram integration
3. **Search Test**: Confirm search functionality works
4. **Mobile Test**: Check responsive design

### Sharing Your App
- **Public URL**: Streamlit provides a public URL
- **Custom Domain**: Available with Streamlit for Teams
- **Password Protection**: Configure in Streamlit Cloud settings

## 📊 Monitoring

### Usage Analytics
- **Streamlit Cloud Dashboard**: View app usage stats
- **Deepgram Dashboard**: Monitor API usage and costs
- **Error Tracking**: Check logs in Streamlit Cloud

### Performance Optimization
- **Caching**: Enabled by default for search index
- **Lazy Loading**: Components load as needed
- **Cleanup**: Automatic temporary file management

## 🔐 Security Notes

### Data Privacy
- **Temporary Storage**: Videos deleted after processing
- **API Keys**: Securely stored in Streamlit secrets
- **No Persistence**: No permanent user data storage

### Best Practices
- **Regular Updates**: Keep dependencies updated
- **API Monitoring**: Monitor Deepgram usage and costs
- **Access Control**: Use Streamlit for Teams for private apps

---

## 🆘 Support

### Get Help
- **Streamlit Community**: [https://discuss.streamlit.io](https://discuss.streamlit.io)
- **Deepgram Support**: [https://help.deepgram.com](https://help.deepgram.com)
- **GitHub Issues**: Create an issue in your repository

### Common Resources
- **Streamlit Documentation**: [https://docs.streamlit.io](https://docs.streamlit.io)
- **Deployment Guide**: [https://docs.streamlit.io/streamlit-cloud](https://docs.streamlit.io/streamlit-cloud)
- **API Reference**: [https://docs.deepgram.com](https://docs.deepgram.com)

Your Video Semantic Search app is now ready for Streamlit Cloud deployment! 🚀 