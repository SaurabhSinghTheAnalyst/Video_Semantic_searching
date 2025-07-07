from moviepy.editor import VideoFileClip
import os
from pathlib import Path
import json
import time
import random

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

class DeepgramTranscriber:
    def __init__(self, deepgram_api_key=None):
        self.deepgram_api_key = deepgram_api_key
        self.deepgram = DeepgramClient() if deepgram_api_key is None else DeepgramClient(deepgram_api_key)
    
    def _write_file_with_retry(self, file_path, content, max_retries=3, base_delay=1.0, is_json=True):
        """
        Write content to file with retry mechanism for handling timeouts and temporary failures.
        
        Args:
            file_path: Path to the file to write
            content: Content to write (string or JSON-serializable object)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds between retries (with exponential backoff)
            is_json: Whether to treat content as JSON data
        
        Returns:
            bool: True if successful, False if all retries failed
        """
        for attempt in range(max_retries + 1):
            try:
                # Ensure parent directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Write the file
                with open(file_path, "w", encoding='utf-8') as f:
                    if is_json and isinstance(content, str):
                        f.write(content)
                    elif is_json:
                        json.dump(content, f, indent=4, ensure_ascii=False)
                    else:
                        f.write(str(content))
                
                # Verify file was written successfully
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    print(f"✅ Successfully saved: {file_path}")
                    return True
                else:
                    raise IOError(f"File verification failed: {file_path}")
                    
            except (OSError, IOError, PermissionError, TimeoutError) as e:
                if attempt < max_retries:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️  Write attempt {attempt + 1} failed for {file_path}: {e}")
                    print(f"   Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"❌ Failed to write {file_path} after {max_retries + 1} attempts: {e}")
                    return False
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"⚠️  Write timeout (attempt {attempt + 1}) for {file_path}: {e}")
                        print(f"   Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"❌ Write operation timed out for {file_path} after {max_retries + 1} attempts: {e}")
                        return False
                else:
                    print(f"❌ Unexpected error writing {file_path}: {e}")
                    return False
        
        return False

    def has_existing_transcript(self, audio_file, transcript_folder="transcripts"):
        """Check if a transcript already exists for the given audio file."""
        audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
        transcript_filename = f"{audio_filename}_transcript.json"
        
        # Check in the same directory as audio file (for sync transcription)
        local_transcript_path = os.path.splitext(audio_file)[0] + "_transcript.json"
        if os.path.exists(local_transcript_path):
            print(f"Found existing transcript: {local_transcript_path}")
            return True
            
        # Check in transcripts folder (for async transcription)
        transcript_path = os.path.join(transcript_folder, transcript_filename)
        if os.path.exists(transcript_path):
            print(f"Found existing transcript: {transcript_path}")
            return True
            
        return False

    def extract_audio(self, mp4_file, audio_file):
        """Extract audio from an MP4 file and save as an audio file (mp3/wav)."""
        video_clip = VideoFileClip(mp4_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_file, codec='mp3')
        audio_clip.close()
        video_clip.close()

    def transcribe_audio(self, audio_file, model="nova-3", smart_format=True, paragraphs=True, save_json=True, max_retries=3):
        """Transcribe the given audio file using Deepgram with retry logic for API calls."""
        for attempt in range(max_retries + 1):
            try:
                if max_retries > 0:  # Only show attempt number if retries are enabled
                    print(f"Transcription attempt {attempt + 1} for {audio_file}")
                
                with open(audio_file, "rb") as file:
                    buffer_data = file.read()
                payload: FileSource = {"buffer": buffer_data}
                options = PrerecordedOptions(model=model, smart_format=smart_format, paragraphs=paragraphs)
                
                response = self.deepgram.listen.rest.v("1").transcribe_file(payload, options)
                result_json = response.to_json(indent=4)
                
                if save_json:
                    json_filename = os.path.splitext(audio_file)[0] + "_transcript.json"
                    success = self._write_file_with_retry(json_filename, result_json, is_json=True)
                    if not success:
                        print(f"❌ Failed to save transcript for {audio_file}")
                        return None
                        
                return result_json
                
            except (TimeoutError, ConnectionError) as e:
                if attempt < max_retries:
                    delay = 3.0 * (2 ** attempt) + random.uniform(0, 2)
                    print(f"⚠️  Transcription timeout/connection error (attempt {attempt + 1}): {e}")
                    print(f"   Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print(f"❌ Transcription failed for {audio_file} after {max_retries + 1} attempts: {e}")
                    return None
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    if attempt < max_retries:
                        delay = 3.0 * (2 ** attempt) + random.uniform(0, 2)
                        print(f"⚠️  Transcription timeout (attempt {attempt + 1}): {e}")
                        print(f"   Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"❌ Transcription timed out for {audio_file} after {max_retries + 1} attempts: {e}")
                        return None
                else:
                    print(f"❌ Exception during transcription of {audio_file}: {e}")
                    return None
        
        return None

    def process_all_videos_sync_simple(self, data_folder="Data", audio_folder="audio", output_folder="transcripts"):
        """Extract audio from videos and transcribe synchronously using simple approach."""
        # Create folders if they don't exist
        Path(audio_folder).mkdir(exist_ok=True)
        Path(output_folder).mkdir(exist_ok=True)
        
        # Find all video files
        data_path = Path(data_folder)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(data_path.glob(f"*{ext}"))
            video_files.extend(data_path.glob(f"*{ext.upper()}"))
        video_files = sorted(video_files)
        
        if not video_files:
            print(f"No video files found in {data_folder}")
            return
            
        print(f"Found {len(video_files)} video files to process")
        
        # Phase 1: Extract all audio files
        print("\n=== Phase 1: Audio Extraction ===")
        extraction_count = 0
        for video_file in video_files:
            audio_file = Path(audio_folder) / f"{video_file.stem}.mp3"
            
            if not audio_file.exists():
                print(f"Extracting audio from {video_file.name} -> {audio_file.name}")
                try:
                    self.extract_audio(str(video_file), str(audio_file))
                    extraction_count += 1
                except Exception as e:
                    print(f"Error extracting audio from {video_file.name}: {e}")
            else:
                print(f"Audio file {audio_file.name} already exists")
        
        print(f"Audio extraction complete: {extraction_count} new files extracted")
        
        # Phase 2: Transcribe all audio files
        print("\n=== Phase 2: Transcription ===")
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Find all audio files to transcribe
        audio_path = Path(audio_folder)
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_path.glob(f"*{ext}"))
            audio_files.extend(audio_path.glob(f"*{ext.upper()}"))
        audio_files = sorted(audio_files)
        
        for audio_file in audio_files:
            transcript_file = Path(output_folder) / f"{audio_file.stem}_transcript.json"
            
            # Skip if transcript already exists
            if transcript_file.exists():
                print(f"Skipping {audio_file.name} - transcript already exists")
                skipped_count += 1
                continue
                
            print(f"Transcribing {audio_file.name}...")
            # Use the enhanced transcribe_audio method which has built-in retry logic
            result = self.transcribe_audio(
                str(audio_file), 
                save_json=False,  # We'll handle saving ourselves
                max_retries=3
            )
            
            if result:
                # Save transcript with retry mechanism
                success = self._write_file_with_retry(
                    transcript_file, 
                    result, 
                    is_json=True
                )
                
                if success:
                    processed_count += 1
                else:
                    print(f"❌ Failed to save transcript for {audio_file.name}")
                    error_count += 1
            else:
                print(f"❌ Failed to transcribe {audio_file.name}")
                error_count += 1
                
        print(f"\nProcessing complete:")
        print(f"  - Transcribed: {processed_count} audio files")
        print(f"  - Skipped: {skipped_count} audio files (existing transcripts)")
        if error_count > 0:
            print(f"  - Errors: {error_count} audio files")

    def process_all_videos(self, data_folder="Data", audio_folder="audio", skip_existing=True):
        """Extract audio and transcribe all video files in the data folder (sync)."""
        data_path = Path(data_folder)
        audio_path = Path(audio_folder)
        audio_path.mkdir(exist_ok=True)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(data_path.glob(f"*{ext}"))
            video_files.extend(data_path.glob(f"*{ext.upper()}"))
        video_files = sorted(video_files)
        if not video_files:
            print(f"No video files found in {data_folder}")
            return
        print(f"Found {len(video_files)} video files to process.")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for video_file in video_files:
            audio_file = audio_path / (video_file.stem + ".mp3")
            
            # Check if transcript already exists
            if skip_existing and self.has_existing_transcript(str(audio_file)):
                print(f"Skipping {video_file.name} - transcript already exists")
                skipped_count += 1
                continue
                
            print(f"\nProcessing {video_file.name} -> {audio_file.name}")
            
            # Extract audio if it doesn't exist
            if not audio_file.exists():
                self.extract_audio(str(video_file), str(audio_file))
            else:
                print(f"Audio file {audio_file.name} already exists, skipping extraction")
                
            result = self.transcribe_audio(str(audio_file), max_retries=3)
            if result:
                processed_count += 1
            else:
                print(f"❌ Failed to transcribe {audio_file.name}")
                error_count += 1
            
        print(f"\nProcessing complete:")
        print(f"  - Processed: {processed_count} videos")
        print(f"  - Skipped: {skipped_count} videos (existing transcripts)")
        if error_count > 0:
            print(f"  - Errors: {error_count} videos")

    def run(self, mp4_file, audio_file):
        """Extract audio and transcribe it."""
        print(f"Extracting audio from {mp4_file} to {audio_file}...")
        self.extract_audio(mp4_file, audio_file)
        print(f"Transcribing {audio_file}...")
        transcript = self.transcribe_audio(audio_file)
        if transcript:
            print(transcript)
        else:
            print("Transcription failed.")

if __name__ == "__main__":
    # Set your Deepgram API key here
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    transcriber = DeepgramTranscriber(deepgram_api_key=DEEPGRAM_API_KEY)
    
    # Use the simple synchronous method
    transcriber.process_all_videos_sync_simple(
        data_folder="Data", 
        audio_folder="audio", 
        output_folder="transcripts"
    )
                               