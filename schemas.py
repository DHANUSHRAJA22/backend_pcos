"""
Complete Pydantic schemas for PCOS Analyzer API.
All request/response models with exact field definitions for frontend integration.
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum
import time


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
    """AI model loading status."""
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error"
    NOT_LOADED = "not_loaded"


class ModelPrediction(BaseModel):
    """Individual AI model prediction result."""
    model_config = {'protected_namespaces': ()}
    
    model_name: str = Field(..., description="AI model identifier")
    model_version: str = Field(..., description="Model version")
    framework: str = Field(..., description="ML framework used")
    probability: float = Field(..., ge=0.0, le=1.0, description="PCOS probability (0-1)")
    predicted_label: RiskLevel = Field(..., description="Predicted risk level")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")


class GenderDetectionResult(BaseModel):
    """Gender detection result for facial images."""
    model_config = {'protected_namespaces': ()}
    
    predicted_gender: str = Field(..., description="Predicted gender (male/female)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    processing_time_ms: float = Field(..., description="Processing time")
    warning: Optional[str] = Field(None, description="Warning message if male detected")


class FacePredictions(BaseModel):
    """Aggregated predictions from facial analysis models."""
    model_config = {'protected_namespaces': ()}
    
    individual_predictions: List[ModelPrediction] = Field(..., description="Per-model predictions")
    average_probability: float = Field(..., description="Average probability across models")
    consensus_label: RiskLevel = Field(..., description="Consensus prediction")
    model_count: int = Field(..., description="Number of models used")
    gender_detection: Optional[GenderDetectionResult] = Field(None, description="Gender detection result")


class XrayPredictions(BaseModel):
    """Aggregated predictions from X-ray analysis models."""
    model_config = {'protected_namespaces': ()}
    
    individual_predictions: List[ModelPrediction] = Field(..., description="Per-model predictions")
    average_probability: float = Field(..., description="Average probability across models")
    consensus_label: RiskLevel = Field(..., description="Consensus prediction")
    model_count: int = Field(..., description="Number of models used")
    detection_count: Optional[int] = Field(None, description="Number of objects detected")


class EnsembleResult(BaseModel):
    """Ensemble prediction result."""
    model_config = {'protected_namespaces': ()}
    
    ensemble_method: EnsembleMethod = Field(..., description="Ensemble method used")
    final_probability: float = Field(..., ge=0.0, le=1.0, description="Final ensemble probability")
    final_risk_level: RiskLevel = Field(..., description="Final risk assessment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Ensemble confidence")
    model_agreement: float = Field(..., ge=0.0, le=1.0, description="Inter-model agreement")


class PredictionResponse(BaseModel):
    """Complete prediction response for frontend integration."""
    model_config = {'protected_namespaces': ()}
    
    success: bool = Field(True, description="Request success status")
    face_predictions: Optional[FacePredictions] = Field(None, description="Facial analysis results")
    xray_predictions: Optional[XrayPredictions] = Field(None, description="X-ray analysis results")
    ensemble_result: EnsembleResult = Field(..., description="Final ensemble prediction")
    processing_time_ms: float = Field(..., description="Total processing time")
    model_count: int = Field(..., description="Total models used")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Information about a loaded AI model."""
    model_config = {'protected_namespaces': ()}
    
    status: ModelStatus = Field(..., description="Model loading status")
    model_path: str = Field(..., description="Path to model file")
    framework: str = Field(..., description="ML framework")
    version_hash: Optional[str] = Field(None, description="Model file hash for versioning")
    last_used: Optional[str] = Field(None, description="Last usage timestamp")
    total_predictions: int = Field(0, description="Total predictions made")
    average_inference_time_ms: Optional[float] = Field(None, description="Average inference time")


class HealthResponse(BaseModel):
    """API health check response."""
    model_config = {'protected_namespaces': ()}
    
    status: str = Field(..., description="API status (healthy/degraded/unhealthy)")
    version: str = Field(..., description="API version")
    models: Dict[str, ModelInfo] = Field(..., description="Status of all AI models")
    total_models_loaded: int = Field(..., description="Number of loaded models")
    uptime_seconds: float = Field(..., description="API uptime")
    frontend_cors_enabled: bool = Field(..., description="CORS enabled for frontend")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    model_config = {'protected_namespaces': ()}
    
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for debugging")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")


class EvaluationResult(BaseModel):
    """Model evaluation results."""
    model_config = {'protected_namespaces': ()}
    
    model_name: str = Field(..., description="Evaluated model name")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    sample_count: int = Field(..., description="Number of test samples")


class ModelSwapRequest(BaseModel):
    """Request schema for model hot-swapping."""
    model_config = {'protected_namespaces': ()}
    
    model_name: str = Field(..., description="Model to swap")
    new_model_path: str = Field(..., description="Path to new model file")
    modality: str = Field(..., description="Model modality (face/xray)")
    validate_before_swap: bool = Field(True, description="Validate new model before swap")


class ModelSwapResponse(BaseModel):
    """Response schema for model swap operations."""
    model_config = {'protected_namespaces': ()}
    
    success: bool = Field(..., description="Swap operation success")
    model_name: str = Field(..., description="Swapped model name")
    old_model_hash: Optional[str] = Field(None, description="Previous model hash")
    new_model_hash: Optional[str] = Field(None, description="New model hash")
    swap_time_ms: float = Field(..., description="Time taken for swap")
    message: str = Field(..., description="Operation result message")