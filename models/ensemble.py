"""
Ensemble prediction logic for combining multiple AI model outputs.

Implements various ensemble methods including soft voting, weighted averaging,
and consensus building for robust PCOS risk assessment.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

from config import settings
from schemas import EnsembleResult, RiskLevel, FacePredictions, XrayPredictions

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor for combining face and X-ray model predictions.
    
    Implements multiple ensemble strategies to create robust final
    predictions from individual model outputs.
    """
    
    def __init__(self):
        self.ensemble_method = settings.ENSEMBLE_METHOD
        self.weights = settings.ENSEMBLE_WEIGHTS
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize the ensemble predictor.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Initializing ensemble predictor...")
            
            # TODO: Load ensemble configuration or meta-models
            # Example:
            # self.meta_model = joblib.load("ensemble_meta_model.pkl")
            # self.feature_selector = joblib.load("feature_selector.pkl")
            
            self.is_initialized = True
            logger.info("Ensemble predictor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Ensemble predictor initialization failed: {e}")
            return False
    
    async def predict(
        self, 
        face_predictions: Optional[FacePredictions] = None,
        xray_predictions: Optional[XrayPredictions] = None
    ) -> EnsembleResult:
        """
        Generate final ensemble prediction from individual model outputs.
        
        Args:
            face_predictions: Aggregated facial analysis results
            xray_predictions: Aggregated X-ray analysis results
            
        Returns:
            EnsembleResult: Final ensemble prediction with metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Ensemble predictor not initialized")
            
        if not face_predictions and not xray_predictions:
            raise ValueError("At least one prediction type must be provided")
        
        try:
            # Collect all individual probabilities
            all_probabilities = []
            model_predictions = []
            
            # Extract face model probabilities
            if face_predictions:
                face_probs = self._extract_face_probabilities(face_predictions)
                all_probabilities.extend(face_probs)
                model_predictions.append(("face", face_predictions.average_probability))
            
            # Extract X-ray model probabilities  
            if xray_predictions:
                xray_probs = self._extract_xray_probabilities(xray_predictions)
                all_probabilities.extend(xray_probs)
                model_predictions.append(("xray", xray_predictions.average_probability))
            
            # Calculate ensemble prediction
            if self.ensemble_method == "weighted_average":
                final_probability = await self._weighted_average_prediction(
                    face_predictions, xray_predictions
                )
            elif self.ensemble_method == "soft_voting":
                final_probability = await self._soft_voting_prediction(all_probabilities)
            else:
                # Default to simple average
                final_probability = np.mean(all_probabilities)
            
            # Determine final risk level
            final_risk_level = self._get_risk_level(final_probability)
            
            # Calculate confidence and agreement metrics
            confidence_score = self._calculate_confidence(all_probabilities)
            model_agreement = self._calculate_agreement(all_probabilities)
            
            # Calculate contributions
            face_contribution = None
            xray_contribution = None
            
            if face_predictions and xray_predictions:
                face_contribution = self.weights["face_models"]
                xray_contribution = self.weights["xray_models"]
            elif face_predictions:
                face_contribution = 1.0
            elif xray_predictions:
                xray_contribution = 1.0
            
            # Generate recommendations and risk factors
            key_risk_factors = self._identify_key_risk_factors(
                face_predictions, xray_predictions
            )
            recommendation = self._generate_recommendation(final_risk_level)
            
            return EnsembleResult(
                final_probability=final_probability,
                final_risk_level=final_risk_level,
                confidence_score=confidence_score,
                ensemble_method=self.ensemble_method,
                model_agreement=model_agreement,
                face_contribution=face_contribution,
                xray_contribution=xray_contribution,
                key_risk_factors=key_risk_factors,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            raise RuntimeError(f"Ensemble prediction failed: {e}")
    
    def _extract_face_probabilities(self, face_predictions: FacePredictions) -> List[float]:
        """Extract probabilities from face model predictions."""
        probabilities = []
        
        if face_predictions.efficientnet:
            probabilities.append(face_predictions.efficientnet.probability)
        if face_predictions.resnet:
            probabilities.append(face_predictions.resnet.probability)
        if face_predictions.vgg:
            probabilities.append(face_predictions.vgg.probability)
            
        return probabilities
    
    def _extract_xray_probabilities(self, xray_predictions: XrayPredictions) -> List[float]:
        """Extract probabilities from X-ray model predictions."""
        probabilities = []
        
        if xray_predictions.yolov8:
            probabilities.append(xray_predictions.yolov8.probability)
        if xray_predictions.vision_transformer:
            probabilities.append(xray_predictions.vision_transformer.probability)
            
        return probabilities
    
    async def _weighted_average_prediction(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> float:
        """
        Calculate weighted average prediction.
        
        Combines face and X-ray predictions using configured weights.
        """
        total_weight = 0
        weighted_sum = 0
        
        if face_predictions:
            weight = self.weights["face_models"]
            weighted_sum += face_predictions.average_probability * weight
            total_weight += weight
        
        if xray_predictions:
            weight = self.weights["xray_models"]
            weighted_sum += xray_predictions.average_probability * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def _soft_voting_prediction(self, probabilities: List[float]) -> float:
        """
        Calculate soft voting ensemble prediction.
        
        Uses simple averaging of all model probabilities.
        """
        return np.mean(probabilities) if probabilities else 0.0
    
    def _calculate_confidence(self, probabilities: List[float]) -> float:
        """
        Calculate overall prediction confidence.
        
        Based on probability distribution and model consensus.
        """
        if not probabilities:
            return 0.0
        
        # High confidence when probabilities are consistent and not near 0.5
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        # Confidence decreases with higher variance and proximity to 0.5
        uncertainty_penalty = abs(mean_prob - 0.5) * 2  # 0 when prob=0.5, 1 when prob=0 or 1
        consistency_bonus = 1 - min(std_prob * 2, 1)    # Higher when std is low
        
        confidence = (uncertainty_penalty + consistency_bonus) / 2
        return max(0.0, min(1.0, confidence))
    
    def _calculate_agreement(self, probabilities: List[float]) -> float:
        """
        Calculate agreement score between models.
        
        Measures how much the models agree on the prediction.
        """
        if len(probabilities) < 2:
            return 1.0
        
        # Agreement is inversely related to standard deviation
        std_dev = np.std(probabilities)
        agreement = 1 - min(std_dev * 2, 1)  # Scale std dev to agreement score
        
        return max(0.0, min(1.0, agreement))
    
    def _get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level based on thresholds."""
        if probability < settings.RISK_THRESHOLDS["low"]:
            return RiskLevel.LOW
        elif probability < settings.RISK_THRESHOLDS["moderate"]:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
    
    def _identify_key_risk_factors(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> List[str]:
        """
        Identify primary risk factors contributing to the prediction.
        
        TODO: Implement actual feature importance analysis
        """
        risk_factors = []
        
        # Mock risk factor identification
        if face_predictions:
            if face_predictions.average_probability > 0.6:
                risk_factors.extend(["Facial characteristics", "Hormonal indicators"])
        
        if xray_predictions:
            if xray_predictions.average_probability > 0.6:
                risk_factors.extend(["Ovarian morphology", "Cystic patterns"])
        
        return risk_factors[:5]  # Limit to top 5 risk factors
    
    def _generate_recommendation(self, risk_level: RiskLevel) -> str:
        """
        Generate clinical recommendation based on risk level.
        
        TODO: Integrate with medical guidelines and expert knowledge
        """
        recommendations = {
            RiskLevel.LOW: "Regular monitoring recommended. Maintain healthy lifestyle.",
            RiskLevel.MODERATE: "Consult healthcare provider for further evaluation and monitoring.",
            RiskLevel.HIGH: "Urgent medical consultation recommended. Consider comprehensive PCOS assessment."
        }
        
        return recommendations.get(risk_level, "Consult healthcare provider for personalized advice.")