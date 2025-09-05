"""
Production FastAPI application for PCOS Analyzer.
Ready-to-deploy backend with real AI models and comprehensive API.
"""

import asyncio
import logging
import time
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings, MODEL_REGISTRY
from schemas import (
    PredictionResponse, HealthResponse, ErrorResponse, EnsembleMethod,
    FacePredictions, XrayPredictions, ModelSwapRequest, ModelSwapResponse
)
from models.model_loader import ModelLoader
from ensemble import EnsemblePredictor
from utils.validators import validate_image_file
from utils.preprocessing import preprocess_face_image, preprocess_xray_image

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PCOS ClarityScan API",
    description="Production-ready AI backend for PCOS risk assessment",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Global components
model_loader = ModelLoader()
ensemble_predictor = EnsemblePredictor()

# Application state
app_state = {
    "startup_time": time.time(),
    "total_predictions": 0,
    "models_ready": False
}


@app.on_event("startup")
async def startup_event():
    """Initialize all AI models and systems on startup."""
    try:
        logger.info("🚀 Starting PCOS ClarityScan API...")
        
        # Initialize all models
        models_loaded = await model_loader.initialize_all_models(MODEL_REGISTRY)
        
        # Initialize ensemble predictor
        ensemble_loaded = await ensemble_predictor.initialize()
        
        app_state["models_ready"] = models_loaded and ensemble_loaded
        
        if app_state["models_ready"]:
            logger.info("✅ PCOS ClarityScan API ready for predictions!")
        else:
            logger.warning("⚠️ API started but some models failed to load")
            
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app_state["startup_error"] = str(e)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    try:
        logger.info("Shutting down PCOS ClarityScan API...")
        await model_loader.cleanup_all()
        logger.info("✅ Shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_pcos_risk(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None),
    ensemble_method: EnsembleMethod = Query(EnsembleMethod.WEIGHTED_VOTING)
) -> PredictionResponse:
    """
    Generate PCOS risk prediction using AI models.
    
    Accepts facial and/or X-ray images and returns comprehensive
    risk assessment with ensemble predictions.
    """
    if not app_state["models_ready"]:
        raise HTTPException(
            status_code=503,
            detail="AI models not ready. Please check model files and restart."
        )
    
    if not face_img and not xray_img:
        raise HTTPException(
            status_code=400,
            detail="At least one image (face_img or xray_img) must be provided"
        )
    
    start_time = time.time()
    
    try:
        # Validate uploaded files
        if face_img:
            await validate_image_file(face_img)
        if xray_img:
            await validate_image_file(xray_img)
        
        # Process facial image
        face_predictions = None
        if face_img:
            face_image_data = await face_img.read()
            
            # Run face model predictions
            face_model_predictions = await model_loader.face_manager.predict_all(face_image_data)
            
            # Run gender detection
            gender_result = None
            if model_loader.gender_detector.is_loaded:
                try:
                    gender_result = await model_loader.gender_detector.detect_gender(face_image_data)
                except Exception as e:
                    logger.error(f"Gender detection failed: {e}")
            
            # Create face predictions object
            if face_model_predictions:
                predictions_list = list(face_model_predictions.values())
                avg_prob = np.mean([p.probability for p in predictions_list])
                
                # Determine consensus
                consensus_votes = [p.predicted_label for p in predictions_list]
                consensus_label = max(set(consensus_votes), key=consensus_votes.count)
                
                face_predictions = FacePredictions(
                    individual_predictions=predictions_list,
                    average_probability=avg_prob,
                    consensus_label=consensus_label,
                    model_count=len(predictions_list),
                    gender_detection=gender_result
                )
        
        # Process X-ray image
        xray_predictions = None
        if xray_img:
            xray_image_data = await xray_img.read()
            
            # Run X-ray model predictions
            xray_model_predictions = await model_loader.xray_manager.predict_all(xray_image_data)
            
            # Create X-ray predictions object
            if xray_model_predictions:
                predictions_list = list(xray_model_predictions.values())
                avg_prob = np.mean([p.probability for p in predictions_list])
                
                # Determine consensus
                consensus_votes = [p.predicted_label for p in predictions_list]
                consensus_label = max(set(consensus_votes), key=consensus_votes.count)
                
                # Extract detection count from YOLOv8
                detection_count = None
                for pred in predictions_list:
                    if pred.model_name == "yolov8_xray" and pred.feature_importance:
                        detection_count = int(pred.feature_importance.get("detection_count", 0))
                        break
                
                xray_predictions = XrayPredictions(
                    individual_predictions=predictions_list,
                    average_probability=avg_prob,
                    consensus_label=consensus_label,
                    model_count=len(predictions_list),
                    detection_count=detection_count
                )
        
        # Generate ensemble prediction
        ensemble_result = await ensemble_predictor.predict_ensemble(
            face_predictions=face_predictions,
            xray_predictions=xray_predictions,
            ensemble_method=ensemble_method
        )
        
        # Calculate total processing time
        processing_time = (time.time() - start_time) * 1000
        app_state["total_predictions"] += 1
        
        # Count total models used
        model_count = 0
        if face_predictions:
            model_count += face_predictions.model_count
        if xray_predictions:
            model_count += xray_predictions.model_count
        
        return PredictionResponse(
            success=True,
            face_predictions=face_predictions,
            xray_predictions=xray_predictions,
            ensemble_result=ensemble_result,
            processing_time_ms=processing_time,
            model_count=model_count,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check for all AI models and systems.
    
    Returns detailed status of all models, version hashes,
    and system health for monitoring and debugging.
    """
    try:
        # Get all model information
        all_models_info = model_loader.get_all_models_info()
        
        # Flatten model info for response
        models_status = {}
        total_loaded = 0
        
        # Face models
        for name, info in all_models_info.get("face_models", {}).items():
            models_status[f"face_{name}"] = info
            if info.get("status") == "loaded":
                total_loaded += 1
        
        # X-ray models
        for name, info in all_models_info.get("xray_models", {}).items():
            models_status[f"xray_{name}"] = info
            if info.get("status") == "loaded":
                total_loaded += 1
        
        # Gender detector
        gender_info = all_models_info.get("gender_detector", {})
        models_status["gender_detector"] = gender_info
        if gender_info.get("loaded"):
            total_loaded += 1
        
        # Determine overall status
        if total_loaded == 0:
            status = "unhealthy"
        elif not app_state["models_ready"]:
            status = "degraded"
        else:
            status = "healthy"
        
        # Calculate uptime
        uptime = time.time() - app_state["startup_time"]
        
        return HealthResponse(
            status=status,
            version=settings.API_VERSION,
            models=models_status,
            total_models_loaded=total_loaded,
            uptime_seconds=uptime,
            frontend_cors_enabled=settings.FRONTEND_URL in settings.ALLOWED_ORIGINS
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@app.post("/models/swap", response_model=ModelSwapResponse)
async def swap_model(request: ModelSwapRequest) -> ModelSwapResponse:
    """
    Hot swap a model with zero downtime.
    
    Replaces a model in production without stopping the service.
    """
    try:
        start_time = time.time()
        
        # Determine which manager to use
        if request.modality == "face":
            success = await model_loader.face_manager.hot_swap_model(
                request.model_name, request.new_model_path
            )
        elif request.modality == "xray":
            success = await model_loader.xray_manager.hot_swap_model(
                request.model_name, request.new_model_path
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown modality: {request.modality}"
            )
        
        swap_time = (time.time() - start_time) * 1000
        
        return ModelSwapResponse(
            success=success,
            model_name=request.model_name,
            old_model_hash="previous_hash",  # TODO: Get actual hash
            new_model_hash="new_hash",       # TODO: Calculate new hash
            swap_time_ms=swap_time,
            message="Model swap completed" if success else "Model swap failed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model swap error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model swap failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for comprehensive error logging."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=time.time()
        ).dict()
    )


if __name__ == "__main__":
    """
    Run the PCOS ClarityScan API server.
    
    For development:
        python app.py
    
    For production:
        uvicorn app:app --host 0.0.0.0 --port 8000
    """
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )