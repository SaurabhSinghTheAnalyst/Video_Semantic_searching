import os
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import datetime
import hashlib
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoManager:
    """
    Manages local video uploads with metadata tracking
    """
    
    def __init__(self, base_path: str = "Data"):
        """
        Initialize the VideoManager
        
        Args:
            base_path: Base directory for video storage
        """
        self.base_path = Path(base_path)
        
        # Create directory structure
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.base_path / "video_catalog.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for video catalog"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create videos table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS videos (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        source TEXT NOT NULL,
                        duration INTEGER,
                        file_size INTEGER,
                        added_date TEXT NOT NULL,
                        description TEXT,
                        metadata_path TEXT,
                        processed_status TEXT DEFAULT 'pending',
                        transcript_status TEXT DEFAULT 'pending',
                        search_indexed TEXT DEFAULT 'false'
                    )
                ''')
                
                # Create index for faster searches
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON videos(source)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed ON videos(processed_status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_indexed ON videos(search_indexed)')
                
                conn.commit()
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def add_manual_video(self, video_path: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a manually uploaded video to the catalog
        
        Args:
            video_path: Path to the video file
            title: Optional custom title (uses filename if not provided)
            
        Returns:
            Dictionary with video information
        """
        video_file = Path(video_path)
        
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate unique ID
        video_id = self._generate_video_id(video_path, "manual")
        
        # Use filename as title if not provided
        if not title:
            title = video_file.stem
        
        # Get file info
        file_size = video_file.stat().st_size
        
        # Move video to Data directory if not already there
        if not str(video_file).startswith(str(self.base_path)):
            destination = self.base_path / video_file.name
            # Ensure unique filename
            counter = 1
            while destination.exists():
                name_parts = video_file.stem, counter, video_file.suffix
                destination = self.base_path / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                counter += 1
            
            shutil.copy2(video_file, destination)
            final_path = destination
        else:
            final_path = video_file
        
        # Create metadata
        metadata = {
            'id': video_id,
            'title': title,
            'source': 'manual',
            'original_path': str(video_path),
            'file_path': str(final_path),
            'file_size': file_size,
            'added_date': datetime.datetime.now().isoformat(),
            'filename': final_path.name,
        }
        
        # Save metadata to JSON file
        metadata_path = self.base_path / f"{video_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Add to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO videos 
                    (id, title, file_path, source, file_size, added_date, metadata_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_id, title, str(final_path), 'manual', 
                    file_size, metadata['added_date'], str(metadata_path)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error adding video to database: {e}")
            raise
        
        logger.info(f"✅ Added manual video: {title}")
        return metadata
    
    def get_video_by_id(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video information by ID
        
        Args:
            video_id: Video ID
            
        Returns:
            Video information dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_dict(cursor, row)
                
        except Exception as e:
            logger.error(f"Error getting video by ID {video_id}: {e}")
        
        return None
    
    def get_all_videos(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all videos, optionally filtered by source
        
        Args:
            source: Optional source filter ('manual' or 'youtube')
            
        Returns:
            List of video information dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if source:
                    cursor.execute('SELECT * FROM videos WHERE source = ? ORDER BY added_date DESC', (source,))
                else:
                    cursor.execute('SELECT * FROM videos ORDER BY added_date DESC')
                
                rows = cursor.fetchall()
                return [self._row_to_dict(cursor, row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting all videos: {e}")
            return []
    
    def search_videos(self, query: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search videos by title or description
        
        Args:
            query: Search query
            source: Optional source filter
            
        Returns:
            List of matching video information dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                search_pattern = f'%{query}%'
                
                if source:
                    cursor.execute('''
                        SELECT * FROM videos 
                        WHERE source = ? AND (
                            title LIKE ? OR 
                            description LIKE ?
                        )
                        ORDER BY added_date DESC
                    ''', (source, search_pattern, search_pattern))
                else:
                    cursor.execute('''
                        SELECT * FROM videos 
                        WHERE title LIKE ? OR 
                              description LIKE ?
                        ORDER BY added_date DESC
                    ''', (search_pattern, search_pattern))
                
                rows = cursor.fetchall()
                return [self._row_to_dict(cursor, row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error searching videos: {e}")
            return []
    
    def update_video_status(self, video_id: str, status_type: str, status_value: str) -> bool:
        """
        Update video processing status
        
        Args:
            video_id: Video ID
            status_type: Type of status ('processed_status', 'transcript_status', 'search_indexed')
            status_value: New status value
            
        Returns:
            True if successful, False otherwise
        """
        valid_status_types = ['processed_status', 'transcript_status', 'search_indexed']
        
        if status_type not in valid_status_types:
            logger.error(f"Invalid status type: {status_type}")
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = f'UPDATE videos SET {status_type} = ? WHERE id = ?'
                cursor.execute(query, (status_value, video_id))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating video status: {e}")
            return False
    
    def delete_video(self, video_id: str) -> bool:
        """
        Delete a video and all associated files
        
        Args:
            video_id: Video ID
            
        Returns:
            True if successful, False otherwise
        """
        video_info = self.get_video_by_id(video_id)
        if not video_info:
            logger.error(f"Video not found: {video_id}")
            return False
        
        try:
            # Delete physical files
            file_path = Path(video_info['file_path'])
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted video file: {file_path}")
            
            # Delete metadata file
            metadata_path = video_info.get('metadata_path')
            if metadata_path and Path(metadata_path).exists():
                Path(metadata_path).unlink()
                logger.info(f"Deleted metadata file: {metadata_path}")
            

            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM videos WHERE id = ?', (video_id,))
                conn.commit()
            
            logger.info(f"✅ Successfully deleted video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting video {video_id}: {e}")
            return False
    
    def get_videos_for_processing(self, status: str = 'pending') -> List[Dict[str, Any]]:
        """
        Get videos that need processing
        
        Args:
            status: Status to filter by
            
        Returns:
            List of videos needing processing
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM videos WHERE processed_status = ? ORDER BY added_date ASC',
                    (status,)
                )
                rows = cursor.fetchall()
                return [self._row_to_dict(cursor, row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting videos for processing: {e}")
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Overall stats
                cursor.execute('SELECT COUNT(*), SUM(file_size), SUM(duration) FROM videos')
                total_count, total_size, total_duration = cursor.fetchone()
                
                # Stats by source
                cursor.execute('''
                    SELECT source, COUNT(*), SUM(file_size), SUM(duration) 
                    FROM videos 
                    GROUP BY source
                ''')
                source_stats = cursor.fetchall()
                
                # Processing status stats
                cursor.execute('''
                    SELECT processed_status, COUNT(*) 
                    FROM videos 
                    GROUP BY processed_status
                ''')
                processing_stats = dict(cursor.fetchall())
                
                return {
                    'total_videos': total_count or 0,
                    'total_size_bytes': total_size or 0,
                    'total_size_mb': round((total_size or 0) / (1024 * 1024), 2),
                    'total_size_gb': round((total_size or 0) / (1024 * 1024 * 1024), 2),
                    'total_duration_seconds': total_duration or 0,
                    'total_duration_hours': round((total_duration or 0) / 3600, 2),
                    'source_breakdown': {
                        row[0]: {
                            'count': row[1],
                            'size_mb': round((row[2] or 0) / (1024 * 1024), 2),
                            'duration_hours': round((row[3] or 0) / 3600, 2)
                        }
                        for row in source_stats
                    },
                    'processing_status': processing_stats,
                }
                
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    def cleanup_orphaned_files(self) -> Dict[str, List[str]]:
        """
        Find and optionally remove orphaned files
        
        Returns:
            Dictionary with lists of orphaned files found
        """
        orphaned = {
            'uploaded_videos': [],
            'metadata_files': [],
        }
        
        try:
            # Get all video IDs from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, file_path, metadata_path FROM videos')
                db_videos = cursor.fetchall()
            
            db_video_ids = {row[0] for row in db_videos}
            db_file_paths = {row[1] for row in db_videos}
            db_metadata_paths = {row[2] for row in db_videos if row[2]}
            
            # Check uploaded videos in Data directory
            for video_file in self.base_path.glob('*'):
                if video_file.is_file() and not video_file.name.endswith('_metadata.json'):
                    if str(video_file) not in db_file_paths:
                        orphaned['uploaded_videos'].append(str(video_file))
            
            # Check metadata files
            for metadata_file in self.base_path.glob('*_metadata.json'):
                if str(metadata_file) not in db_metadata_paths:
                    orphaned['metadata_files'].append(str(metadata_file))
            
            return orphaned
            
        except Exception as e:
            logger.error(f"Error checking for orphaned files: {e}")
            return orphaned
    
    def _generate_video_id(self, video_path: str, source: str) -> str:
        """
        Generate a unique video ID
        
        Args:
            video_path: Path to video file
            source: Video source ('manual')
            
        Returns:
            Unique video ID
        """
        # For manual videos, use file hash + timestamp
        if source == 'manual':
            file_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"manual_{file_hash}_{timestamp}"
        
        # Default fallback
        return f"{source}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _row_to_dict(self, cursor: sqlite3.Cursor, row: tuple) -> Dict[str, Any]:
        """
        Convert database row to dictionary
        
        Args:
            cursor: Database cursor
            row: Database row tuple
            
        Returns:
            Dictionary representation of the row
        """
        columns = [description[0] for description in cursor.description]
        result = dict(zip(columns, row))
        
        return result
    
    def export_catalog(self, output_path: str) -> bool:
        """
        Export video catalog to JSON file
        
        Args:
            output_path: Path to save the exported catalog
            
        Returns:
            True if successful, False otherwise
        """
        try:
            videos = self.get_all_videos()
            
            export_data = {
                'export_date': datetime.datetime.now().isoformat(),
                'total_videos': len(videos),
                'videos': videos,
                'storage_stats': self.get_storage_stats(),
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Catalog exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting catalog: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize video manager
    vm = VideoManager()
    
    # Add a manual video (example)
    # vm.add_manual_video("path/to/video.mp4", "My Custom Title")
    
    # Get all videos
    all_videos = vm.get_all_videos()
    print(f"Total videos: {len(all_videos)}")
    
    # Get storage stats
    stats = vm.get_storage_stats()
    print(f"Storage stats: {stats}")
    
    # Search videos
    # results = vm.search_videos("python tutorial")
    # print(f"Search results: {len(results)}") 