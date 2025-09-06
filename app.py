"""
Production FastAPI application for PCOS Analyzer.
Ready-to-deploy backend with real AI models and comprehensive API.
"""

from __future__ import annotations

import logging
import time
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import settings, MODEL_REGISTRY, GENDER_DETECTION_CONFIG
from schemas import (
    PredictionResponse, HealthResponse, ErrorResponse, EnsembleMethod,
    FacePredictions, XrayPredictions, ModelSwapRequest, ModelSwapResponse,
    ModelInfo
)
from models.model_loader import ModelLoader
from ensemble import EnsemblePredictor
from utils.validators import validate_image_file

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")

# ------------------------------------------------------------------------------
# FastAPI app + CORS + Static
# ------------------------------------------------------------------------------
app = FastAPI(
    title="PCOS ClarityScan API",
    description="Production-ready AI backend for PCOS risk assessment",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.getcwd(), "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global components
model_loader = ModelLoader()
ensemble_predictor = EnsemblePredictor()

# Application state
app_state: Dict[str, Any] = {
    "startup_time": time.time(),
    "total_predictions": 0,
    "models_ready": False,
    "startup_error": None
}

# Allowlisted hosts for /img-proxy
ALLOWED_IMG_HOSTS = {
    "as2.ftcdn.net",
    "static.wixstatic.com",
    "resources.ama.uk.com",
    "www.emjreviews.com",
}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _save_upload_bytes(data: bytes, prefix: str, ext_from_name: str) -> str:
    """Save bytes to static/uploads and return a /static url."""
    _, ext = os.path.splitext(ext_from_name or "")
    ext = ext.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{ts}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return f"/static/uploads/{filename}"


def _make_xray_visualization_heuristic(image_bytes: bytes) -> Tuple[Optional[str], int]:
    """
    Heuristic visualization for ultrasound X-ray:
    - enhance contrast (CLAHE)
    - smooth
    - try HoughCircles to find dark circular cavities
    - fallback: contour circularity filter
    Returns: (url, detected_count) or (None, 0)
    """
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, 0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

        h, w = gray.shape[:2]
        min_r = max(8, int(min(h, w) * 0.02))
        max_r = max(min_r + 5, int(min(h, w) * 0.10))

        overlay = img.copy()
        total = 0

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(18, min_r),
            param1=60, param2=22, minRadius=min_r, maxRadius=max_r
        )
        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles[0]))
            for (cx, cy, r) in circles[:30]:
                cv2.circle(overlay, (int(cx), int(cy)), int(r), (0, 140, 255), 2)
                total += 1

        if total == 0:
            thr = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5
            )
            thr = cv2.medianBlur(thr, 5)
            kernel = np.ones((3, 3), np.uint8)
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)

            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 80:
                    continue
                per = cv2.arcLength(cnt, True)
                if per == 0:
                    continue
                circularity = 4 * np.pi * area / (per * per)
                if circularity >= 0.55:
                    (x, y, w2, h2) = cv2.boundingRect(cnt)
                    r = int(0.5 * max(w2, h2))
                    cx, cy = int(x + w2 / 2), int(y + h2 / 2)
                    cv2.circle(overlay, (cx, cy), r, (0, 140, 255), 2)
                    total += 1
                    if total >= 30:
                        break

        if total == 0:
            return None, 0

        out = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
        ts = int(time.time() * 1000)
        out_name = f"xray_vis_{ts}.jpg"
        out_path = os.path.join(UPLOAD_DIR, out_name)
        cv2.imwrite(out_path, out)
        return f"/static/uploads/{out_name}", total
    except Exception as e:
        logger.error(f"Heuristic xray visualization failed: {e}")
        return None, 0


def _as_iso(v):
    """Coerce timestamps or datetimes to ISO strings for Pydantic."""
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(float(v), tz=timezone.utc).isoformat()
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if isinstance(v, str):
            return v
        return str(v)
    except Exception:
        return str(v)


def _parse_gender_result(gr) -> Tuple[Optional[str], Optional[float]]:
    """
    Robustly extract (label, confidence) from many possible detector schemas.
    Works with dicts or objects.
    """
    if gr is None:
        return None, None

    try:
        # Dict-like
        if isinstance(gr, dict):
            label = (
                gr.get("predicted_gender") or gr.get("gender") or
                gr.get("label") or gr.get("class") or gr.get("class_name")
            )
            conf = gr.get("confidence") or gr.get("probability") or gr.get("score") or gr.get("conf")

            if "male_probability" in gr or "female_probability" in gr:
                male_p = float(gr.get("male_probability", 0) or 0)
                female_p = float(gr.get("female_probability", 0) or 0)
                if male_p >= female_p:
                    label, conf = "male", male_p
                else:
                    label, conf = "female", female_p

            return (str(label).lower() if label else None, float(conf) if conf is not None else None)

        # Object-like
        label = (
            getattr(gr, "predicted_gender", None) or getattr(gr, "gender", None) or
            getattr(gr, "label", None) or getattr(gr, "class_name", None)
        )
        conf = (
            getattr(gr, "confidence", None) or getattr(gr, "probability", None) or getattr(gr, "score", None)
        )

        if hasattr(gr, "male_probability") or hasattr(gr, "female_probability"):
            male_p = float(getattr(gr, "male_probability", 0) or 0)
            female_p = float(getattr(gr, "female_probability", 0) or 0)
            if male_p >= female_p:
                label, conf = "male", male_p
            else:
                label, conf = "female", female_p

        return (str(label).lower() if label else None, float(conf) if conf is not None else None)
    except Exception:
        return None, None


def _maybe_gender_warning(label: Optional[str], conf: Optional[float]) -> Optional[str]:
    """
    Returns the configured warning string if label is 'male' and confidence meets threshold.
    If the detector didn't provide a confidence, we still warn.
    """
    min_conf = float(GENDER_DETECTION_CONFIG.get("min_confidence", 0.60))
    warn_msg = GENDER_DETECTION_CONFIG.get(
        "warning_message",
        "Male face detected — facial model is intended for female patients."
    )
    if label == "male" and (conf is None or conf >= min_conf):
        return warn_msg
    return None

# ------------------------------------------------------------------------------
# Root / Startup / Shutdown
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "PCOS ClarityScan API is running",
        "version": settings.API_VERSION,
        "status": "healthy" if app_state["models_ready"] else "starting",
        "docs": "/docs"
    }


@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Starting PCOS ClarityScan API...")
        try:
            models_loaded = await model_loader.initialize_all_models(MODEL_REGISTRY)
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            models_loaded = False
            app_state["startup_error"] = str(e)

        try:
            ensemble_loaded = await ensemble_predictor.initialize()
        except Exception as e:
            logger.error(f"Ensemble initialization failed: {e}")
            ensemble_loaded = False
            if not app_state["startup_error"]:
                app_state["startup_error"] = str(e)

        app_state["models_ready"] = models_loaded and ensemble_loaded
        if app_state["models_ready"]:
            logger.info("✅ PCOS ClarityScan API ready for predictions!")
        else:
            logger.warning("⚠️ API started but models failed to load - serving health endpoint only")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app_state["startup_error"] = str(e)
        app_state["models_ready"] = False


@app.on_event("shutdown")
async def shutdown_event():
    try:
        logger.info("Shutting down PCOS ClarityScan API...")
        await model_loader.cleanup_all()
        logger.info("✅ Shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# ------------------------------------------------------------------------------
# Structured Predict
# ------------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_pcos_risk(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None),
    ensemble_method: EnsembleMethod = Query(EnsembleMethod.WEIGHTED_VOTING)
) -> PredictionResponse:
    if not app_state["models_ready"]:
        raise HTTPException(status_code=503, detail="AI models not ready. Please check model files and restart.")
    if not face_img and not xray_img:
        raise HTTPException(status_code=400, detail="At least one image (face_img or xray_img) must be provided")

    start_time = time.time()

    try:
        if face_img:
            await validate_image_file(face_img)
        if xray_img:
            await validate_image_file(xray_img)

        # Face
        face_predictions = None
        if face_img:
            face_bytes = await face_img.read()
            try:
                face_model_predictions = await model_loader.face_manager.predict_all(face_bytes)
            except Exception as e:
                logger.error(f"Face model prediction failed: {e}")
                face_model_predictions = {}

            gender_result = None
            gender_warning = None
            if hasattr(model_loader, 'gender_detector') and model_loader.gender_detector.is_loaded:
                try:
                    gender_result = await model_loader.gender_detector.detect_gender(face_bytes)
                    glabel, gscore = _parse_gender_result(gender_result)
                    gender_warning = _maybe_gender_warning(glabel, gscore)
                    if gender_warning:
                        logger.info(f"[GenderDetect] label={glabel} conf={gscore} -> warning")
                except Exception as e:
                    logger.error(f"Gender detection failed: {e}")

            if face_model_predictions:
                preds = list(face_model_predictions.values())
                avg = float(np.mean([p.probability for p in preds]))
                votes = [p.predicted_label for p in preds]
                consensus = max(set(votes), key=votes.count)
                face_predictions = FacePredictions(
                    individual_predictions=preds,
                    average_probability=avg,
                    consensus_label=consensus,
                    model_count=len(preds),
                    gender_detection=gender_result
                )

        # X-ray
        xray_predictions = None
        if xray_img:
            xray_bytes = await xray_img.read()
            try:
                xray_model_predictions = await model_loader.xray_manager.predict_all(xray_bytes)
            except Exception as e:
                logger.error(f"X-ray model prediction failed: {e}")
                xray_model_predictions = {}

            if xray_model_predictions:
                preds = list(xray_model_predictions.values())
                avg = float(np.mean([p.probability for p in preds]))
                votes = [p.predicted_label for p in preds]
                consensus = max(set(votes), key=votes.count)
                detection_count = None
                for p in preds:
                    if p.feature_importance and "detection_count" in p.feature_importance:
                        try:
                            detection_count = int(p.feature_importance["detection_count"])
                            break
                        except Exception:
                            pass
                xray_predictions = XrayPredictions(
                    individual_predictions=preds,
                    average_probability=avg,
                    consensus_label=consensus,
                    model_count=len(preds),
                    detection_count=detection_count
                )

        ensemble_result = await ensemble_predictor.predict_ensemble(
            face_predictions=face_predictions,
            xray_predictions=xray_predictions,
            ensemble_method=ensemble_method
        )

        processing_time = (time.time() - start_time) * 1000.0
        app_state["total_predictions"] += 1
        model_count = (face_predictions.model_count if face_predictions else 0) + \
                      (xray_predictions.model_count if xray_predictions else 0)

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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ------------------------------------------------------------------------------
# Health (coerce timestamps to strings)
# ------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        all_models_info = model_loader.get_all_models_info()
        models_status: Dict[str, ModelInfo] = {}
        total_loaded = 0

        for name, info in (all_models_info.get("face_models") or {}).items():
            mi = ModelInfo(
                status=info.get("status", "not_loaded"),
                model_path=str(info.get("model_path", "")),
                framework=str(info.get("framework", "tensorflow")),
                version_hash=(None if info.get("version_hash") is None else str(info.get("version_hash"))),
                last_used=_as_iso(info.get("last_used")),
                total_predictions=int(info.get("total_predictions", 0)),
                average_inference_time_ms=info.get("average_inference_time_ms"),
            )
            models_status[f"face_{name}"] = mi
            if str(mi.status).lower() == "loaded":
                total_loaded += 1

        for name, info in (all_models_info.get("xray_models") or {}).items():
            mi = ModelInfo(
                status=info.get("status", "not_loaded"),
                model_path=str(info.get("model_path", "")),
                framework=str(info.get("framework", "ultralytics")),
                version_hash=(None if info.get("version_hash") is None else str(info.get("version_hash"))),
                last_used=_as_iso(info.get("last_used")),
                total_predictions=int(info.get("total_predictions", 0)),
                average_inference_time_ms=info.get("average_inference_time_ms"),
            )
            models_status[f"xray_{name}"] = mi
            if str(mi.status).lower() == "loaded":
                total_loaded += 1

        ginfo = (all_models_info.get("gender_detector") or {})
        g_loaded = bool(ginfo.get("loaded"))
        mi = ModelInfo(
            status=("loaded" if g_loaded else "not_loaded"),
            model_path=str(ginfo.get("model_path", "opencv_cv_analysis")),
            framework=str(ginfo.get("framework", "opencv")),
            version_hash=(None if ginfo.get("version_hash") is None else str(ginfo.get("version_hash"))),
            last_used=_as_iso(ginfo.get("last_used")),
            total_predictions=int(ginfo.get("total_predictions", 0)),
            average_inference_time_ms=ginfo.get("average_inference_time_ms"),
        )
        models_status["gender_detector"] = mi
        if g_loaded:
            total_loaded += 1

        startup_error = app_state.get("startup_error")
        if startup_error:
            status = "unhealthy"
        elif total_loaded == 0:
            status = "unhealthy"
        elif not app_state["models_ready"]:
            status = "degraded"
        else:
            status = "healthy"

        uptime = time.time() - app_state["startup_time"]

        return HealthResponse(
            status=status,
            version=settings.API_VERSION,
            models=models_status,
            total_models_loaded=total_loaded,
            uptime_seconds=uptime,
            frontend_cors_enabled=settings.FRONTEND_URL in settings.ALLOWED_ORIGINS,
            **({"startup_error": startup_error} if startup_error else {})
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            version=settings.API_VERSION,
            models={},
            total_models_loaded=0,
            uptime_seconds=time.time() - app_state["startup_time"],
            frontend_cors_enabled=settings.FRONTEND_URL in settings.ALLOWED_ORIGINS,
            startup_error=str(e),
        )

# ------------------------------------------------------------------------------
# Image proxy
# ------------------------------------------------------------------------------
from urllib.parse import urlparse
import httpx

@app.get("/img-proxy")
async def img_proxy(url: str):
    host = urlparse(url).netloc.lower()
    if host not in ALLOWED_IMG_HOSTS:
        raise HTTPException(status_code=403, detail=f"host '{host}' not allowed")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("Content-Type", "image/jpeg"),
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Proxy fetch failed: {e}")

# ------------------------------------------------------------------------------
# Hot Swap
# ------------------------------------------------------------------------------
@app.post("/models/swap", response_model=ModelSwapResponse)
async def swap_model(request: ModelSwapRequest) -> ModelSwapResponse:
    try:
        start_time = time.time()
        if request.modality == "face":
            success = await model_loader.face_manager.hot_swap_model(request.model_name, request.new_model_path)
        elif request.modality == "xray":
            success = await model_loader.xray_manager.hot_swap_model(request.model_name, request.new_model_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown modality: {request.modality}")
        swap_time = (time.time() - start_time) * 1000.0
        return ModelSwapResponse(
            success=success,
            model_name=request.model_name,
            old_model_hash="previous_hash",
            new_model_hash="new_hash",
            swap_time_ms=swap_time,
            message="Model swap completed" if success else "Model swap failed"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model swap error: {e}")
        raise HTTPException(status_code=500, detail=f"Model swap failed: {str(e)}")

# ------------------------------------------------------------------------------
# Legacy compatibility mapper + endpoint (adds face_img/xray_img + warnings)
# ------------------------------------------------------------------------------
def _risk_is_positive(risk_level: str) -> bool:
    try:
        return risk_level.lower() in {"moderate", "high"}
    except Exception:
        return False


def _make_combined(face_pos: bool, xray_pos: bool) -> str:
    if face_pos and xray_pos:
        return "High risk: Both modalities indicate PCOS symptoms."
    if face_pos or xray_pos:
        return "Moderate risk: One modality suggests PCOS symptoms."
    return "Low risk: No PCOS detected by either modality."


def _to_legacy_payload(
    pred: "PredictionResponse",
    face_img_url: Optional[str] = None,
    xray_img_url: Optional[str] = None,
    yolo_vis_url: Optional[str] = None,
    found_labels: Optional[List[str]] = None,
    gender_warning: Optional[str] = None,
) -> Dict[str, Any]:
    legacy: Dict[str, Any] = {}
    er = pred.ensemble_result
    face = pred.face_predictions
    xray = pred.xray_predictions

    if face:
        avg = float(face.average_probability or 0.0)
        legacy["face_scores"] = [max(0.0, 1.0 - avg), max(0.0, avg)]
        if _risk_is_positive(getattr(face.consensus_label, "value", str(face.consensus_label))):
            legacy["face_pred"] = "unhealthy"
            legacy["face_risk"] = "high"
        else:
            legacy["face_pred"] = "non_pcos"
            legacy["face_risk"] = "low"
        if face_img_url:
            legacy["face_img"] = face_img_url

    if xray:
        detected = (xray.detection_count or 0) > 0 or \
                   _risk_is_positive(getattr(xray.consensus_label, "value", str(xray.consensus_label)))
        legacy["xray_pred"] = "PCOS symptoms detected in X-ray" if detected else "No PCOS symptoms detected in X-ray"
        legacy["xray_risk"] = "high" if detected else "low"
        if xray_img_url:
            legacy["xray_img"] = xray_img_url
        if yolo_vis_url:
            legacy["yolo_vis"] = yolo_vis_url
        if found_labels:
            legacy["found_labels"] = found_labels
    else:
        legacy["xray_risk"] = "unknown"

    face_pos = legacy.get("face_risk") == "high"
    xray_pos = legacy.get("xray_risk") == "high"
    legacy["combined"] = _make_combined(face_pos, xray_pos)
    legacy["overall_risk"] = getattr(er.final_risk_level, "value", str(er.final_risk_level))
    legacy["ok"] = True
    legacy["model_count"] = pred.model_count
    legacy["processing_time_ms"] = pred.processing_time_ms

    if gender_warning:
        legacy["message"] = gender_warning

    return legacy


from fastapi import Form

@app.post("/predict-legacy")
async def predict_pcos_risk_legacy(
    face_img: Optional[UploadFile] = File(None),
    xray_img: Optional[UploadFile] = File(None),
    ensemble_method: EnsembleMethod = Query(settings.DEFAULT_ENSEMBLE_METHOD),
):
    if not app_state["models_ready"]:
        raise HTTPException(status_code=503, detail="AI models not ready. Please check model files and restart.")
    if not face_img and not xray_img:
        raise HTTPException(status_code=400, detail="At least one image (face_img or xray_img) must be provided")

    t0 = time.time()

    # Validate
    if face_img:
        await validate_image_file(face_img)
    if xray_img:
        await validate_image_file(xray_img)

    # Face
    face_predictions: Optional[FacePredictions] = None
    face_url: Optional[str] = None
    gender_warning: Optional[str] = None
    if face_img:
        face_bytes = await face_img.read()
        face_url = _save_upload_bytes(face_bytes, "face", face_img.filename)
        try:
            face_preds_map = await model_loader.face_manager.predict_all(face_bytes)
        except Exception as e:
            logger.error(f"Face predict failed: {e}")
            face_preds_map = {}

        gender_result = None
        if hasattr(model_loader, "gender_detector") and getattr(model_loader.gender_detector, "is_loaded", False):
            try:
                gender_result = await model_loader.gender_detector.detect_gender(face_bytes)
                glabel, gscore = _parse_gender_result(gender_result)
                gender_warning = _maybe_gender_warning(glabel, gscore)
                logger.info(f"[GenderDetect] label={glabel} conf={gscore}")
            except Exception as e:
                logger.error(f"Gender detection failed: {e}")

        if face_preds_map:
            preds = list(face_preds_map.values())
            avg = float(np.mean([p.probability for p in preds]))
            votes = [p.predicted_label for p in preds]
            consensus = max(set(votes), key=votes.count)
            face_predictions = FacePredictions(
                individual_predictions=preds,
                average_probability=avg,
                consensus_label=consensus,
                model_count=len(preds),
                gender_detection=gender_result,
            )

    # X-ray
    xray_predictions: Optional[XrayPredictions] = None
    xray_url: Optional[str] = None
    yolo_vis_url: Optional[str] = None
    found_labels: List[str] = []
    if xray_img:
        xray_bytes = await xray_img.read()
        xray_url = _save_upload_bytes(xray_bytes, "xray", xray_img.filename)
        try:
            xray_preds_map = await model_loader.xray_manager.predict_all(xray_bytes)
        except Exception as e:
            logger.error(f"X-ray predict failed: {e}")
            xray_preds_map = {}

        detection_count = None
        if xray_preds_map:
            preds = list(xray_preds_map.values())
            avg = float(np.mean([p.probability for p in preds]))
            votes = [p.predicted_label for p in preds]
            consensus = max(set(votes), key=votes.count)

            for p in preds:
                if p.feature_importance:
                    if "detection_count" in p.feature_importance:
                        try:
                            detection_count = int(p.feature_importance["detection_count"])
                        except Exception:
                            pass
                    labels = p.feature_importance.get("labels") if isinstance(p.feature_importance, dict) else None
                    if isinstance(labels, list):
                        found_labels = [str(x) for x in labels]
                    vis_path = p.feature_importance.get("visualization_path") if isinstance(p.feature_importance, dict) else None
                    if isinstance(vis_path, str) and vis_path:
                        yolo_vis_url = vis_path if vis_path.startswith("/static/") else f"/static/{vis_path.lstrip('/')}"
                    if found_labels or yolo_vis_url:
                        break

            # If no vis from model, try heuristic overlay
            if not yolo_vis_url:
                heur_url, heur_count = _make_xray_visualization_heuristic(xray_bytes)
                if heur_url:
                    yolo_vis_url = heur_url
                    if detection_count is None:
                        detection_count = heur_count
                    if not found_labels and heur_count > 0:
                        found_labels = ["cyst-like regions"]

            xray_predictions = XrayPredictions(
                individual_predictions=preds,
                average_probability=avg,
                consensus_label=consensus,
                model_count=len(preds),
                detection_count=detection_count
            )

    # Ensemble
    ensemble_result = await ensemble_predictor.predict_ensemble(
        face_predictions=face_predictions,
        xray_predictions=xray_predictions,
        ensemble_method=ensemble_method
    )

    processing_time = (time.time() - t0) * 1000.0
    app_state["total_predictions"] += 1
    model_count = (face_predictions.model_count if face_predictions else 0) + \
                  (xray_predictions.model_count if xray_predictions else 0)

    structured = PredictionResponse(
        success=True,
        face_predictions=face_predictions,
        xray_predictions=xray_predictions,
        ensemble_result=ensemble_result,
        processing_time_ms=processing_time,
        model_count=model_count,
        timestamp=datetime.now().isoformat()
    )

    return _to_legacy_payload(
        structured,
        face_img_url=face_url,
        xray_img_url=xray_url,
        yolo_vis_url=yolo_vis_url,
        found_labels=found_labels or (["cyst-like regions"] if (xray_predictions and (xray_predictions.detection_count or 0) > 0) else None),
        gender_warning=gender_warning
    )

# ------------------------------------------------------------------------------
# Dev entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
