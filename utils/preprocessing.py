"""
Image preprocessing utilities for consistent model input.
Ensures identical preprocessing between training and inference.
"""

import logging
from typing import Tuple
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def preprocess_face_image(image_data: bytes, target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Preprocess facial image exactly as in VGG16 training pipeline.
    
    Args:
        image_data: Raw image bytes
        target_size: Target dimensions (width, height)
        
    Returns:
        np.ndarray: Preprocessed image array for model input
    """
    try:
        # Load image and convert to RGB
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Resize to target size (100x100 for VGG16)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Optional: Enhance image quality
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)  # Slight contrast boost
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        logger.debug(f"Face image preprocessed to shape: {image_batch.shape}")
        return image_batch
        
    except Exception as e:
        logger.error(f"Face image preprocessing error: {e}")
        raise RuntimeError(f"Failed to preprocess face image: {e}")


def preprocess_xray_image(image_data: bytes) -> Image.Image:
    """
    Preprocess X-ray image for YOLOv8 inference.
    
    YOLOv8 handles most preprocessing internally, so we provide
    a clean PIL Image object.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Image.Image: PIL Image ready for YOLOv8
    """
    try:
        # Load image
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Optional: Enhance contrast for medical images
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        logger.debug(f"X-ray image preprocessed, size: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"X-ray image preprocessing error: {e}")
        raise RuntimeError(f"Failed to preprocess X-ray image: {e}")


def preprocess_image(image_data: bytes, modality: str) -> any:
    """
    Preprocess image based on modality.
    
    Args:
        image_data: Raw image bytes
        modality: Image type ("face" or "xray")
        
    Returns:
        Preprocessed image in appropriate format
    """
    if modality == "face":
        return preprocess_face_image(image_data)
    elif modality == "xray":
        return preprocess_xray_image(image_data)
    else:
        raise ValueError(f"Unknown modality: {modality}")