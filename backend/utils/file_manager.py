"""
File Manager for handling uploads and file operations
"""

import os
import shutil
import aiofiles
import hashlib
import mimetypes
from typing import Optional
from fastapi import UploadFile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self):
        self.upload_dir = "uploads"
        self.temp_dir = "temp"
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        
        # Create directories
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def save_upload(self, file: UploadFile) -> str:
        """Save uploaded file and return file path"""
        try:
            # Validate file
            await self._validate_file(file)
            
            # Generate unique filename
            file_hash = hashlib.md5(f"{file.filename}{datetime.now()}".encode()).hexdigest()
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{file_hash}{file_extension}"
            
            # Create file path
            file_path = os.path.join(self.upload_dir, unique_filename)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logger.info(f"Saved file: {file.filename} -> {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"File save error: {str(e)}")
            raise
    
    async def _validate_file(self, file: UploadFile):
        """Validate uploaded file"""
        # Check file size
        content = await file.read()
        if len(content) > self.max_file_size:
            raise ValueError(f"File too large. Max size: {self.max_file_size} bytes")
        
        # Reset file pointer
        await file.seek(0)
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file.filename)
        allowed_types = [
            'video/mp4', 'video/avi', 'video/mov', 'video/wmv',
            'audio/wav', 'audio/mp3', 'audio/m4a', 'audio/ogg',
            'image/jpeg', 'image/png', 'image/bmp', 'image/tiff'
        ]
        
        if mime_type not in allowed_types:
            raise ValueError(f"Unsupported file type: {mime_type}")
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"File deletion error: {str(e)}")
            return False
    
    def cleanup_old_files(self, hours: int = 24):
        """Clean up files older than specified hours"""
        try:
            current_time = datetime.now().timestamp()
            
            for directory in [self.upload_dir, self.temp_dir]:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_time = os.path.getmtime(file_path)
                        if (current_time - file_time) > (hours * 3600):
                            self.delete_file(file_path)
                            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
    
    def get_file_info(self, file_path: str) -> dict:
        """Get file information"""
        try:
            if not os.path.exists(file_path):
                return {}
            
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'mime_type': mimetypes.guess_type(file_path)[0]
            }
        except Exception as e:
            logger.error(f"File info error: {str(e)}")
            return {}
