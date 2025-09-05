"""
Abstract base classes and interfaces for individual AI models.

Provides consistent interfaces for all model types, enabling plug-and-play
architecture for research experimentation and production deployment.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from io import BytesIO
import numpy as np

from config import settings
from schemas import ModelPrediction, RiskLevel, ModelPredictionResult

logger = logging.getLogger(__name__)


class BaseAIModel(ABC):
    """
    Abstract base class for all AI models in the PCOS analysis system.
    
    Provides a consistent interface for model loading, prediction, and resource
    management across different frameworks (PyTorch, TensorFlow, ONNX, etc.).
    """
    
    def __init__(
        self, 
        model_name: str, 
        model_version: str,
        framework: str,
        modality: str
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.framework = framework
        self.modality = modality  # "face" or "xray"
        
        # Model state
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.load_time_ms = 0
        
        # Performance tracking
        self.prediction_count = 0
        self.total_inference_time = 0
        self.last_used = None
        self.peak_memory_mb = 0
        
        # Model configuration
        self.config = self._get_model_config()
        
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration from settings."""
        if self.modality == "face":
            return FACE_MODELS_CONFIG.get(self.model_name, {})
        elif self.modality == "xray":
            return XRAY_MODELS_CONFIG.get(self.model_name, {})
        return {}
    
    @abstractmethod
    async def load_model(self) -> bool:
        """Load the AI model into memory."""
        pass
    
    @abstractmethod
    async def predict(self, image_data: bytes) -> ModelPredictionResult:
        """Generate prediction from image data."""
        pass
    
    async def predict_with_metadata(self, image_data: bytes) -> ModelPrediction:
        """Generate prediction with comprehensive metadata."""
        if not self.is_loaded:
            raise RuntimeError(f"{self.model_name} model not loaded")
        
        start_time = time.time()
        
        try:
            # Get core prediction
            result = await self.predict(image_data)
            
            # Update performance tracking
            self.prediction_count += 1
            self.total_inference_time += result.processing_time_ms
            self.last_used = time.time()
            
            # Determine risk level
            risk_level = self._get_risk_level(result.probability)
            
            # Create comprehensive prediction object
            return ModelPrediction(
                model_name=self.model_name,
                model_version=self.model_version,
                framework=self.framework,
                probability=result.probability,
                predicted_label=risk_level,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                memory_usage_mb=self.peak_memory_mb,
                feature_importance=result.feature_importance,
                input_shape=self.config.get("input_size", [224, 224]),
                preprocessing_applied=self._get_preprocessing_steps()
            )
            
        except Exception as e:
            logger.error(f"Prediction error in {self.model_name}: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")
    
    def _get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level based on clinical thresholds."""
        if probability < settings.RISK_THRESHOLDS["low"]:
            return RiskLevel.LOW
        elif probability < settings.RISK_THRESHOLDS["moderate"]:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
    
    def _get_preprocessing_steps(self) -> List[str]:
        """Get list of preprocessing steps applied to input."""
        return [
            "resize_to_target",
            "normalize_pixels",
            "convert_to_rgb"
        ]
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for monitoring and research."""
        avg_inference_time = (
            self.total_inference_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            "model_name": self.model_name,
            "version": self.model_version,
            "framework": self.framework,
            "modality": self.modality,
            "is_loaded": self.is_loaded,
            "load_time_ms": self.load_time_ms,
            "prediction_count": self.prediction_count,
            "average_inference_time_ms": avg_inference_time,
            "peak_memory_mb": self.peak_memory_mb,
            "last_used": self.last_used,
            "config": self.config
        }
    
    async def cleanup(self):
        """Clean up model resources and free memory."""
        try:
            # Framework-specific cleanup
            if self.framework == "pytorch" and self.model:
                import torch
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            elif self.framework == "tensorflow" and self.model:
                import tensorflow as tf
                del self.model
                tf.keras.backend.clear_session()
            
            elif self.framework == "ultralytics" and self.model:
                del self.model
            
            self.model = None
            self.preprocessor = None
            self.is_loaded = False
            
            logger.info(f"{self.model_name} model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during {self.model_name} cleanup: {e}")


# Import config constants here to avoid circular imports
from config import FACE_MODELS_CONFIG, XRAY_MODELS_CONFIG