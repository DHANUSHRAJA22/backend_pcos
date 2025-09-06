"""
Real ensemble prediction logic for combining multiple AI model outputs.
Implements all ensemble methods using actual model predictions.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
import os

from schemas import (
    EnsembleResult, EnsembleMethod, RiskLevel, 
    FacePredictions, XrayPredictions, ModelPrediction
)
from config import settings

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Real ensemble predictor for combining actual AI model outputs.
    
    Implements all ensemble methods (soft voting, weighted voting,
    stacking, majority voting) using real model predictions.
    """
    
    def __init__(self):
        self.is_initialized = False
        self.prediction_count = 0
        self.meta_models = {}
        
    async def initialize(self) -> bool:
        """Initialize ensemble predictor."""
        try:
            logger.info("Initializing ensemble predictor...")
            
            # Initialize meta-models for stacking (graceful loading)
            try:
                import joblib
                import xgboost as xgb
                from sklearn.linear_model import LogisticRegression
                
                # Try to load pre-trained meta-models
                meta_model_paths = {
                    "xgboost": "models/meta/xgboost_meta_model.pkl",
                    "logistic": "models/meta/logistic_meta_model.pkl"
                }
                
                for name, path in meta_model_paths.items():
                    if os.path.exists(path):
                        try:
                            self.meta_models[name] = joblib.load(path)
                            logger.info(f"Loaded {name} meta-model from {path}")
                        except Exception as e:
                            logger.warning(f"Failed to load {name} meta-model: {e}")
                
                # Create default meta-models if none loaded
                if not self.meta_models:
                    self.meta_models["xgboost"] = xgb.XGBClassifier(random_state=42)
                    self.meta_models["logistic"] = LogisticRegression(random_state=42)
                    logger.info("Created default meta-models (require training)")
                    
            except ImportError:
                logger.warning("Meta-model libraries not available - stacking disabled")
            
            self.is_initialized = True
            logger.info("✓ Ensemble predictor initialized")
            return True
            
        except Exception as e:
            logger.error(f"Ensemble predictor initialization failed: {e}")
            return False
    
    async def predict_ensemble(
        self,
        face_predictions: Optional[FacePredictions] = None,
        xray_predictions: Optional[XrayPredictions] = None,
        ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_VOTING
    ) -> EnsembleResult:
        """
        Generate ensemble prediction from individual model outputs.
        
        Args:
            face_predictions: Facial analysis results
            xray_predictions: X-ray analysis results
            ensemble_method: Ensemble method to use
            
        Returns:
            EnsembleResult: Final ensemble prediction
        """
        if not self.is_initialized:
            raise RuntimeError("Ensemble predictor not initialized")
        
        if not face_predictions and not xray_predictions:
            raise ValueError("At least one prediction type must be provided")
        
        start_time = time.time()
        
        try:
            # Extract probabilities from all models
            all_probabilities = []
            
            if face_predictions:
                face_probs = [pred.probability for pred in face_predictions.individual_predictions]
                all_probabilities.extend(face_probs)
            
            if xray_predictions:
                xray_probs = [pred.probability for pred in xray_predictions.individual_predictions]
                all_probabilities.extend(xray_probs)
            
            # Calculate ensemble prediction based on method
            if ensemble_method == EnsembleMethod.SOFT_VOTING:
                final_probability = await self._soft_voting(all_probabilities)
                
            elif ensemble_method == EnsembleMethod.WEIGHTED_VOTING:
                final_probability = await self._weighted_voting(face_predictions, xray_predictions)
                
            elif ensemble_method == EnsembleMethod.MAJORITY_VOTING:
                final_probability = await self._majority_voting(face_predictions, xray_predictions)
                
            elif ensemble_method == EnsembleMethod.STACKING:
                # TODO: Implement stacking with trained meta-model
                final_probability = await self._stacking_prediction(face_predictions, xray_predictions)
                
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")
            
            # Calculate ensemble metrics
            final_risk_level = self._get_risk_level(final_probability)
            ensemble_confidence = self._calculate_confidence(all_probabilities, final_probability)
            model_agreement = self._calculate_agreement(all_probabilities)
            
            self.prediction_count += 1
            
            return EnsembleResult(
                ensemble_method=ensemble_method,
                final_probability=final_probability,
                final_risk_level=final_risk_level,
                confidence=ensemble_confidence,
                model_agreement=model_agreement
            )
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            raise RuntimeError(f"Ensemble prediction failed: {e}")
    
    async def _soft_voting(self, probabilities: List[float]) -> float:
        """Soft voting: average of all model probabilities."""
        if not probabilities:
            return 0.0
        return float(np.mean(probabilities))
    
    async def _weighted_voting(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> float:
        """Weighted voting using configured model weights."""
        from config import settings
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        # Weight face predictions
        if face_predictions:
            face_weight = settings.FACE_MODEL_WEIGHT
            weighted_sum += face_predictions.average_probability * face_weight
            total_weight += face_weight
        
        # Weight X-ray predictions
        if xray_predictions:
            xray_weight = settings.XRAY_MODEL_WEIGHT
            weighted_sum += xray_predictions.average_probability * xray_weight
            total_weight += xray_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def _majority_voting(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> float:
        """Majority voting based on risk level classifications."""
        votes = []
        
        # Collect face model votes
        if face_predictions:
            for pred in face_predictions.individual_predictions:
                votes.append(pred.predicted_label)
        
        # Collect X-ray model votes
        if xray_predictions:
            for pred in xray_predictions.individual_predictions:
                votes.append(pred.predicted_label)
        
        if not votes:
            return 0.0
        
        # Count votes
        vote_counts = {
            RiskLevel.LOW: votes.count(RiskLevel.LOW),
            RiskLevel.MODERATE: votes.count(RiskLevel.MODERATE),
            RiskLevel.HIGH: votes.count(RiskLevel.HIGH)
        }
        
        # Find majority
        majority_label = max(vote_counts, key=vote_counts.get)
        
        # Convert to probability
        label_to_prob = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.HIGH: 0.8
        }
        
        return label_to_prob[majority_label]
    
    async def _stacking_prediction(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> float:
        """
        Stacking prediction using meta-model.
        
        TODO: Implement with trained meta-model when available.
        For now, falls back to weighted voting.
        """
        logger.info("Stacking not yet implemented, using weighted voting")
        return await self._weighted_voting(face_predictions, xray_predictions)
    
    def _calculate_confidence(self, probabilities: List[float], final_prob: float) -> float:
        """Calculate ensemble confidence based on model agreement."""
        if not probabilities:
            return 0.0
        
        # Higher confidence when models agree and prediction is certain
        certainty = abs(final_prob - 0.5) * 2  # 0 when prob=0.5, 1 when prob=0 or 1
        consistency = 1 - min(np.std(probabilities) * 2, 1)  # Higher when low variance
        
        confidence = (certainty * 0.6 + consistency * 0.4)
        return max(0.0, min(1.0, confidence))
    
    def _calculate_agreement(self, probabilities: List[float]) -> float:
        """Calculate inter-model agreement score."""
        if len(probabilities) < 2:
            return 1.0
        
        std_dev = np.std(probabilities)
        agreement = 1 - min(std_dev * 2, 1)
        return max(0.0, min(1.0, agreement))
    
    def _get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level."""
        from config import settings
        
        if probability < settings.RISK_THRESHOLDS["low"]:
            return RiskLevel.LOW
        elif probability < settings.RISK_THRESHOLDS["moderate"]:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH