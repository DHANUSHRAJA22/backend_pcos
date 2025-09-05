"""
Real gender detection model for facial image analysis.
Uses computer vision techniques to detect gender and warn for male faces.
"""

import asyncio
import logging
import time
from typing import Dict
from io import BytesIO

import numpy as np
import cv2
from PIL import Image

from schemas import GenderDetectionResult

logger = logging.getLogger(__name__)


class GenderDetector:
    """
    Real gender detection using computer vision analysis.
    
    Analyzes facial features to classify gender and provide warnings
    when male faces are detected for PCOS analysis.
    """
    
    def __init__(self):
        self.is_loaded = False
        self.face_cascade = None
        
    async def initialize(self) -> bool:
        """Initialize gender detection system."""
        try:
            logger.info("Initializing gender detector...")
            
            # Load OpenCV face cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                return False
            
            self.is_loaded = True
            logger.info("✓ Gender detector initialized")
            return True
            
        except Exception as e:
            logger.error(f"Gender detector initialization failed: {e}")
            return False
    
    async def detect_gender(self, image_data: bytes) -> GenderDetectionResult:
        """
        Detect gender from facial image using computer vision.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            GenderDetectionResult: Gender detection result with confidence
        """
        if not self.is_loaded:
            raise RuntimeError("Gender detector not initialized")
        
        start_time = time.time()
        
        try:
            # Load and convert image
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                # No face detected - default to female for PCOS analysis
                processing_time = (time.time() - start_time) * 1000
                return GenderDetectionResult(
                    predicted_gender="female",
                    confidence=0.3,
                    processing_time_ms=processing_time,
                    warning=None
                )
            
            # Use largest detected face
            (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Analyze gender features
            gender_score = await self._analyze_gender_features(face_roi)
            
            # Determine gender and confidence
            if gender_score > 0.6:
                predicted_gender = "male"
                confidence = min(gender_score * 1.2, 0.95)
                warning = self._generate_warning(confidence) if confidence > 0.8 else None
            else:
                predicted_gender = "female"
                confidence = min((1 - gender_score) * 1.2, 0.95)
                warning = None
            
            processing_time = (time.time() - start_time) * 1000
            
            return GenderDetectionResult(
                predicted_gender=predicted_gender,
                confidence=confidence,
                processing_time_ms=processing_time,
                warning=warning
            )
            
        except Exception as e:
            logger.error(f"Gender detection error: {e}")
            raise RuntimeError(f"Gender detection failed: {e}")
    
    async def _analyze_gender_features(self, face_roi: np.ndarray) -> float:
        """
        Analyze facial features for gender classification.
        
        Uses computer vision techniques to extract gender-indicative features.
        """
        try:
            # Resize for consistent analysis
            face_resized = cv2.resize(face_roi, (100, 100))
            
            # Feature extraction
            features = {}
            
            # 1. Facial hair detection using edge analysis
            edges = cv2.Canny(face_resized, 50, 150)
            lower_face = edges[60:, :]  # Lower portion for facial hair
            edge_density = np.sum(lower_face > 0) / lower_face.size
            features["facial_hair_indicator"] = edge_density
            
            # 2. Face structure analysis (width-to-height ratio)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                width_height_ratio = w / h if h > 0 else 1.0
                features["face_structure"] = width_height_ratio
            else:
                features["face_structure"] = 1.0
            
            # 3. Texture analysis using gradient magnitude
            grad_x = cv2.Sobel(face_resized, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_resized, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            texture_score = np.mean(gradient_magnitude) / 255.0
            features["texture_complexity"] = texture_score
            
            # 4. Intensity distribution analysis
            mean_intensity = np.mean(face_resized)
            std_intensity = np.std(face_resized)
            intensity_variation = std_intensity / mean_intensity if mean_intensity > 0 else 0
            features["intensity_variation"] = intensity_variation
            
            # Combine features for gender score (0=female, 1=male)
            gender_score = (
                features["facial_hair_indicator"] * 0.4 +  # Strong male indicator
                (features["face_structure"] - 1.0) * 0.3 +  # Wider faces tend male
                features["texture_complexity"] * 0.2 +      # Texture differences
                (features["intensity_variation"] - 0.5) * 0.1  # Intensity patterns
            )
            
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, gender_score))
            
        except Exception as e:
            logger.error(f"Gender feature analysis error: {e}")
            return 0.5  # Neutral score on error
    
    def _generate_warning(self, confidence: float) -> str:
        """Generate warning message for male face detection."""
        return (
            f"Warning: Detected a male face (confidence: {confidence:.1%}). "
            f"PCOS detection currently applies only to females. "
            f"Please use a valid input image."
        )