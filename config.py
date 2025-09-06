"""
Production configuration for PCOS Analyzer FastAPI Backend.

All settings, model paths, and configuration constants.
Zero imports from project modules to prevent circular dependencies.
"""

import os
from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic_settings import BaseSettings

# Optional; safe defaults are used if you omit them
TOP_K_ENABLED = True
TOP_K_MODELS = 5

class RiskLevel(str, Enum):
    """PCOS risk assessment levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EnsembleMethod(str, Enum):
    """Available ensemble prediction methods."""
    SOFT_VOTING = "soft_voting"
    WEIGHTED_VOTING = "weighted_voting"
    STACKING = "stacking"
    MAJORITY_VOTING = "majority_voting"


class ModelStatus(str, Enum):
    """AI model loading and operational status."""
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error"
    NOT_LOADED = "not_loaded"


class PCOSSettings(BaseSettings):
    """
    Production settings for PCOS Analyzer API.
    All settings configurable via environment variables.
    """
    
    # API Configuration
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS Configuration - Frontend Integration
    FRONTEND_URL: str = "http://localhost:8080"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8080", 
        "http://127.0.0.1:8080",
        "http://localhost:3000",  # Common React dev port
        "http://127.0.0.1:3000"
    ]
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    # Model File Paths - Configurable via Environment
    FACE_MODEL_PATH: str = "models/pcos_detector_158.h5"
    FACE_LABELS_PATH: str = "models/pcos_detector_158.labels.txt"
    XRAY_MODEL_PATH: str = "models/bestv8.pt"
    
    # Ensemble Configuration
    DEFAULT_ENSEMBLE_METHOD: EnsembleMethod = EnsembleMethod.WEIGHTED_VOTING
    ENABLE_STACKING: bool = True
    
    # Risk Assessment Thresholds
    RISK_THRESHOLDS: Dict[str, float] = {
        "low": 0.3,
        "moderate": 0.7
    }
    
    # Model Weights for Weighted Voting
    FACE_MODEL_WEIGHT: float = 0.6
    XRAY_MODEL_WEIGHT: float = 0.4
    
    # Gender Detection Configuration
    ENABLE_GENDER_DETECTION: bool = True
    GENDER_CONFIDENCE_THRESHOLD: float = 0.8
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_prefix = "PCOS_"
        case_sensitive = False


# Global settings instance
settings = PCOSSettings()


# Model Registry for Easy Plugin Architecture
MODEL_REGISTRY = {
    "face_models": {
        "vgg16_primary": {
            "model_path": settings.FACE_MODEL_PATH,
            "labels_path": settings.FACE_LABELS_PATH,
            "framework": "tensorflow",
            "input_size": [100, 100],
            "preprocessing": "vgg16_standard"
        }
        # TODO: Add more face models here
        # "resnet50": {
        #     "model_path": "models/resnet50_pcos.h5",
        #     "framework": "tensorflow",
        #     "input_size": [224, 224]
        # }
    },
    "xray_models": {
        "yolov8_primary": {
            "model_path": settings.XRAY_MODEL_PATH,
            "framework": "ultralytics",
            "input_size": [640, 640],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45
        }
        # TODO: Add more X-ray models here
        # "efficientnet_xray": {
        #     "model_path": "models/efficientnet_xray.h5",
        #     "framework": "tensorflow"
        # }
    }
}


# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "face": {
        "target_size": (100, 100),
        "color_mode": "RGB",
        "normalization": "0_to_1",
        "enhancement": True
    },
    "xray": {
        "target_size": (640, 640),
        "color_mode": "RGB",
        "enhancement": True,
        "contrast_adjustment": True
    }
}


# Gender Detection Configuration
GENDER_DETECTION_CONFIG = {
    "method": "opencv_cv_analysis",
    "confidence_threshold": 0.8,
    "warning_message": "Warning: Detected a male face. PCOS detection currently applies only to females. Please use a valid input image."
}