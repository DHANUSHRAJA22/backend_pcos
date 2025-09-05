"""
Base AI model interface for consistent model loading and prediction.
All models inherit from this base class for standardized behavior.
"""

import asyncio
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import os

from schemas import ModelPrediction, RiskLevel, ModelStatus

logger = logging.getLogger(__name__)


class BaseAIModel(ABC):
    """
    Abstract base class for all AI models in PCOS Analyzer.
    
    Provides consistent interface for model loading, prediction,
    and resource management across different frameworks.
    """
    
    def __init__(self, model_name: str, model_path: str, framework: str):
        self.model_name = model_name
        self.model_path = model_path
        self.framework = framework
        
        # Model state
        self.model = None
        self.is_loaded = False
        self.load_time_ms = 0
        self.version_hash = None
        
        # Performance tracking
        self.prediction_count = 0
        self.total_inference_time = 0
        self.last_used = None
        
    @abstractmethod
    async def load_model(self) -> bool:
        """Load the AI model into memory."""
        pass
    
    @abstractmethod
    async def predict(self, image_data: bytes) -> ModelPrediction:
        """Generate prediction from image data."""
        pass
    
    async def initialize(self) -> bool:
        """Initialize model and calculate version hash."""
        try:
            start_time = time.time()
            
            # Calculate model file hash for versioning
            if os.path.exists(self.model_path):
                self.version_hash = await self._calculate_file_hash(self.model_path)
            
            # Load the actual model
            success = await self.load_model()
            
            if success:
                self.load_time_ms = (time.time() - start_time) * 1000
                logger.info(f"✓ {self.model_name} loaded in {self.load_time_ms:.2f}ms")
            
            return success
            
        except Exception as e:
            logger.error(f"Model initialization failed for {self.model_name}: {e}")
            return False
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of model file for version tracking."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()[:16]  # Short hash
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        avg_inference_time = (
            self.total_inference_time / self.prediction_count 
            if self.prediction_count > 0 else 0
        )
        
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "framework": self.framework,
            "status": ModelStatus.LOADED if self.is_loaded else ModelStatus.NOT_LOADED,
            "version_hash": self.version_hash,
            "is_loaded": self.is_loaded,
            "load_time_ms": self.load_time_ms,
            "prediction_count": self.prediction_count,
            "average_inference_time_ms": avg_inference_time,
            "last_used": self.last_used
        }
    
    def _get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level."""
        from config import settings
        
        if probability < settings.RISK_THRESHOLDS["low"]:
            return RiskLevel.LOW
        elif probability < settings.RISK_THRESHOLDS["moderate"]:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
    
    async def cleanup(self):
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        self.is_loaded = False
        logger.info(f"{self.model_name} cleanup completed")