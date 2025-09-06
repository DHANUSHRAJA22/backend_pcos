"""
Real ensemble prediction logic for combining multiple AI model outputs.
Implements all ensemble methods using actual model predictions.

This version adds optional Top-K aggregation *within each modality* when using
WEIGHTED_VOTING, without changing your API or schemas.

Config flags (optional, safe if absent):
  - TOP_K_ENABLED: bool = False
  - TOP_K_MODELS: int = 5
"""

import logging
import time
from typing import Dict, List, Optional
import numpy as np

from schemas import (
    EnsembleResult, EnsembleMethod, RiskLevel,
    FacePredictions, XrayPredictions
)

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Real ensemble predictor for combining actual AI model outputs.

    Implements soft voting, weighted voting, majority voting, and a stacking
    placeholder. Weighted voting can optionally use Top-K within each modality
    (face/xray) by reading config flags if present.
    """

    def __init__(self):
        self.is_initialized = False
        self.prediction_count = 0

    async def initialize(self) -> bool:
        """Initialize ensemble predictor."""
        try:
            logger.info("Initializing ensemble predictor...")
            # If you later add meta-models for stacking, load them here.
            self.is_initialized = True
            logger.info("✓ Ensemble predictor initialized")
            return True
        except Exception as e:
            logger.error(f"Ensemble predictor initialization failed: {e}")
            return False

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
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
            # Collect all individual probabilities for metrics
            all_probabilities: List[float] = []
            if face_predictions:
                all_probabilities.extend([float(p.probability) for p in face_predictions.individual_predictions])
            if xray_predictions:
                all_probabilities.extend([float(p.probability) for p in xray_predictions.individual_predictions])

            # Choose method
            if ensemble_method == EnsembleMethod.SOFT_VOTING:
                final_probability = await self._soft_voting(all_probabilities)

            elif ensemble_method == EnsembleMethod.WEIGHTED_VOTING:
                final_probability = await self._weighted_voting(face_predictions, xray_predictions)

            elif ensemble_method == EnsembleMethod.MAJORITY_VOTING:
                final_probability = await self._majority_voting(face_predictions, xray_predictions)

            elif ensemble_method == EnsembleMethod.STACKING:
                final_probability = await self._stacking_prediction(face_predictions, xray_predictions)

            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")

            final_risk_level = self._get_risk_level(final_probability)
            ensemble_confidence = self._calculate_confidence(all_probabilities, final_probability)
            model_agreement = self._calculate_agreement(all_probabilities)

            self.prediction_count += 1

            return EnsembleResult(
                ensemble_method=ensemble_method,
                final_probability=float(final_probability),
                final_risk_level=final_risk_level,
                confidence=float(ensemble_confidence),
                model_agreement=float(model_agreement)
            )

        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            raise RuntimeError(f"Ensemble prediction failed: {e}") from e

    # --------------------------------------------------------------------- #
    # Methods
    # --------------------------------------------------------------------- #
    async def _soft_voting(self, probabilities: List[float]) -> float:
        """Soft voting: mean of all model probabilities across both modalities."""
        if not probabilities:
            return 0.0
        return float(np.mean(probabilities))

    async def _weighted_voting(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> float:
        """
        Weighted voting using configured modality weights.

        Each modality score is computed from its individual predictions:
          - default: mean of all probabilities
          - if TOP_K_ENABLED: mean of the Top-K highest probabilities within the modality

        Then combine:
          score = FACE_MODEL_WEIGHT * face_score + XRAY_MODEL_WEIGHT * xray_score
                  ---------------------------------------------------------------
                                   FACE_MODEL_WEIGHT + XRAY_MODEL_WEIGHT
        """
        from config import settings  # late import to avoid circulars

        # Read weights + Top-K flags safely
        face_w = float(getattr(settings, "FACE_MODEL_WEIGHT", 0.6))
        xray_w = float(getattr(settings, "XRAY_MODEL_WEIGHT", 0.4))
        topk_enabled = bool(getattr(settings, "TOP_K_ENABLED", False))
        topk_k = int(getattr(settings, "TOP_K_MODELS", 5))

        def modality_score(preds: Optional[List[float]]) -> float:
            if not preds:
                return 0.0
            arr = np.asarray(preds, dtype=float)
            if topk_enabled and topk_k > 0:
                k = min(topk_k, arr.size)
                # take largest probabilities (strongest positive indications)
                arr = np.partition(arr, -k)[-k:]
            return float(np.mean(arr)) if arr.size else 0.0

        # Compute per-modality scores from *individual predictions*
        face_score = modality_score(
            [float(p.probability) for p in face_predictions.individual_predictions]
        ) if face_predictions else 0.0

        xray_score = modality_score(
            [float(p.probability) for p in xray_predictions.individual_predictions]
        ) if xray_predictions else 0.0

        total_w = face_w + xray_w
        if total_w <= 0:
            return 0.0

        return float((face_w * face_score + xray_w * xray_score) / total_w)

    async def _majority_voting(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> float:
        """
        Majority voting based on risk level classifications across *individual*
        model predictions. Ties default toward moderate probability.
        """
        votes: List[RiskLevel] = []

        if face_predictions:
            votes.extend([p.predicted_label for p in face_predictions.individual_predictions])

        if xray_predictions:
            votes.extend([p.predicted_label for p in xray_predictions.individual_predictions])

        if not votes:
            return 0.0

        low = sum(1 for v in votes if v == RiskLevel.LOW)
        mod = sum(1 for v in votes if v == RiskLevel.MODERATE)
        high = sum(1 for v in votes if v == RiskLevel.HIGH)

        if high > max(low, mod):
            return 0.8
        if low > max(high, mod):
            return 0.2
        # Tie or moderate majority → neutral/moderate
        return 0.5

    async def _stacking_prediction(
        self,
        face_predictions: Optional[FacePredictions],
        xray_predictions: Optional[XrayPredictions]
    ) -> float:
        """
        Stacking placeholder. When you add a meta-model, extract features like:
          [face_avg, face_std, xray_avg, xray_std, face_topk_avg, xray_topk_avg]
        and feed into the meta-model's predict_proba.
        For now, we fallback to weighted voting.
        """
        logger.info("Stacking not yet implemented, using weighted voting")
        return await self._weighted_voting(face_predictions, xray_predictions)

    # --------------------------------------------------------------------- #
    # Metrics helpers
    # --------------------------------------------------------------------- #
    def _calculate_confidence(self, probabilities: List[float], final_prob: float) -> float:
        """
        Confidence blends:
          - certainty: distance from 0.5 (peaks at 0 or 1)
          - consistency: 1 - normalized std of model probs
        """
        if not probabilities:
            return 0.0

        arr = np.asarray(probabilities, dtype=float)
        certainty = min(1.0, max(0.0, abs(float(final_prob) - 0.5) * 2.0))
        std = float(np.std(arr)) if arr.size > 1 else 0.0
        consistency = 1.0 - min(1.0, std * 2.0)  # map std≈0 → 1 (agree), large std → 0
        conf = 0.6 * certainty + 0.4 * consistency
        return float(max(0.0, min(1.0, conf)))

    def _calculate_agreement(self, probabilities: List[float]) -> float:
        """Agreement = 1 - normalized std deviation (0..1)."""
        if len(probabilities) < 2:
            return 1.0
        std_dev = float(np.std(np.asarray(probabilities, dtype=float)))
        agreement = 1.0 - min(1.0, std_dev * 2.0)
        return float(max(0.0, min(1.0, agreement)))

    def _get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability (0..1) to risk level using config thresholds."""
        from config import settings
        p = float(probability)
        low_thr = float(getattr(settings, "RISK_THRESHOLDS", {}).get("low", 0.30))
        mod_thr = float(getattr(settings, "RISK_THRESHOLDS", {}).get("moderate", 0.70))

        if p < low_thr:
            return RiskLevel.LOW
        elif p < mod_thr:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
