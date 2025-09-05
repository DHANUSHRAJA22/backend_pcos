"""
Input validation utilities for medical image uploads.
Comprehensive validation for security and medical compliance.
"""

import logging
from fastapi import UploadFile, HTTPException

from config import settings

logger = logging.getLogger(__name__)


async def validate_image_file(file: UploadFile) -> bool:
    """
    Comprehensive validation for uploaded medical images.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        bool: True if validation passes
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Check file presence
        if not file or not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file provided or invalid filename"
            )
        
        # Validate file extension
        file_extension = '.' + file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File extension '{file_extension}' not supported. "
                       f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Validate MIME type
        if file.content_type not in settings.ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"MIME type '{file.content_type}' not allowed. "
                       f"Supported: {', '.join(settings.ALLOWED_MIME_TYPES)}"
            )
        
        # Validate file size
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # Reset file pointer
        
        if file_size > settings.MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            limit_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File size ({size_mb:.2f}MB) exceeds limit ({limit_mb}MB)"
            )
        
        # Basic content validation
        if file_size < 1024:  # Minimum 1KB
            raise HTTPException(
                status_code=400,
                detail="File too small - may be corrupted"
            )
        
        # Check image file signatures
        image_signatures = {
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'RIFF': 'WebP'
        }
        
        file_signature = None
        for signature, format_name in image_signatures.items():
            if content.startswith(signature):
                file_signature = format_name
                break
        
        if not file_signature:
            raise HTTPException(
                status_code=400,
                detail="File does not appear to be a valid image"
            )
        
        logger.debug(f"✓ File validation passed: {file.filename} ({file_signature})")
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(
            status_code=500,
            detail="File validation failed"
        )