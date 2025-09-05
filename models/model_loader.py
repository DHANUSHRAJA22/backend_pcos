"""
Dynamic model loader for hot-swapping and plugin architecture.
Enables loading models from different frameworks with consistent interface.
"""

import asyncio
import logging
import os
from typing import Dict, Optional, Any

from models.face_model import FaceModelManager
from models.xray_model import XrayModelManager
from models.gender_detector import GenderDetector

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Central model loader for all AI models.
    
    Provides unified interface for loading, managing, and hot-swapping
    models across different frameworks and modalities.
    """
    
    def __init__(self):
        self.face_manager = FaceModelManager()
        self.xray_manager = XrayModelManager()
        self.gender_detector = GenderDetector()
        
        self.all_loaded = False
        
    async def initialize_all_models(self, model_registry: Dict[str, Any]) -> bool:
        """
        Initialize all models from registry configuration.
        
        Args:
            model_registry: Model configuration registry
            
        Returns:
            bool: True if all critical models loaded successfully
        """
        try:
            logger.info("Initializing all AI models...")
            
            # Initialize models concurrently
            init_tasks = [
                self.face_manager.initialize(model_registry.get("face_models", {})),
                self.xray_manager.initialize(model_registry.get("xray_models", {})),
                self.gender_detector.initialize()
            ]
            
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            face_loaded = results[0] is True
            xray_loaded = results[1] is True
            gender_loaded = results[2] is True
            
            # Log results
            if face_loaded:
                logger.info("✓ Face models loaded successfully")
            else:
                logger.error("✗ Face models failed to load")
            
            if xray_loaded:
                logger.info("✓ X-ray models loaded successfully")
            else:
                logger.error("✗ X-ray models failed to load")
            
            if gender_loaded:
                logger.info("✓ Gender detector loaded successfully")
            else:
                logger.error("✗ Gender detector failed to load")
            
            # At least one model type must be loaded
            self.all_loaded = face_loaded or xray_loaded
            
            if self.all_loaded:
                logger.info("🎉 Model loader initialization completed successfully")
            else:
                logger.error("❌ Model loader initialization failed - no models loaded")
            
            return self.all_loaded
            
        except Exception as e:
            logger.error(f"Model loader initialization failed: {e}")
            return False
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all loaded models."""
        return {
            "face_models": self.face_manager.get_models_info(),
            "xray_models": self.xray_manager.get_models_info(),
            "gender_detector": {
                "loaded": self.gender_detector.is_loaded,
                "status": "loaded" if self.gender_detector.is_loaded else "not_loaded"
            }
        }
    
    async def cleanup_all(self):
        """Clean up all model resources."""
        await asyncio.gather(
            self.face_manager.cleanup(),
            self.xray_manager.cleanup(),
            return_exceptions=True
        )
        logger.info("All models cleanup completed")