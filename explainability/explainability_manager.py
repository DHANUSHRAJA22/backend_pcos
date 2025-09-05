"""
Centralized explainability manager for coordinating all interpretation methods.

Manages SHAP, LIME, GradCAM, and other explainability techniques
with unified API and configurable explanation generation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np

from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.gradcam_explainer import GradCAMExplainer

logger = logging.getLogger(__name__)


class ExplainabilityManager:
    """
    Unified manager for all explainability methods.
    
    Coordinates different explanation techniques and provides
    comprehensive interpretability analysis for medical AI predictions.
    """
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.gradcam_explainer = GradCAMExplainer()
        
        self.enabled_methods = {
            "shap": True,
            "lime": True,
            "gradcam": True
        }
        
        self.is_initialized = False
    
    async def initialize(self, models: Dict[str, Any]) -> bool:
        """Initialize all explainability methods."""
        try:
            logger.info("Initializing explainability manager...")
            
            # Initialize all explanation methods
            init_tasks = []
            
            if self.enabled_methods["shap"]:
                init_tasks.append(self.shap_explainer.initialize(models))
            
            if self.enabled_methods["lime"]:
                init_tasks.append(self.lime_explainer.initialize(models))
            
            if self.enabled_methods["gradcam"]:
                init_tasks.append(self.gradcam_explainer.initialize(models))
            
            # Wait for all initializations
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            success_count = sum(1 for result in results if result is True)
            total_methods = len(init_tasks)
            
            self.is_initialized = (success_count > 0)
            
            logger.info(f"Explainability manager: {success_count}/{total_methods} methods ready")
            return self.is_initialized
            
        except Exception as e:
            logger.error(f"Explainability manager initialization failed: {e}")
            return False
    
    async def generate_comprehensive_explanation(
        self,
        model_name: str,
        image_data: bytes,
        prediction_result: Dict[str, Any],
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation using multiple methods.
        
        Args:
            model_name: Model to explain
            image_data: Input image
            prediction_result: Model prediction
            methods: Explanation methods to use (default: all enabled)
            
        Returns:
            Dict: Comprehensive explanation from all methods
        """
        if not self.is_initialized:
            raise RuntimeError("Explainability manager not initialized")
        
        if methods is None:
            methods = [method for method, enabled in self.enabled_methods.items() if enabled]
        
        try:
            start_time = time.time()
            explanations = {}
            
            # Generate explanations using requested methods
            explanation_tasks = []
            
            if "shap" in methods and self.enabled_methods["shap"]:
                task = self.shap_explainer.explain_prediction(
                    model_name, image_data, prediction_result
                )
                explanation_tasks.append(("shap", task))
            
            if "lime" in methods and self.enabled_methods["lime"]:
                task = self.lime_explainer.explain_prediction(
                    model_name, image_data, prediction_result
                )
                explanation_tasks.append(("lime", task))
            
            if "gradcam" in methods and self.enabled_methods["gradcam"]:
                task = self.gradcam_explainer.explain_prediction(
                    model_name, image_data, prediction_result
                )
                explanation_tasks.append(("gradcam", task))
            
            # Collect all explanations
            for method_name, task in explanation_tasks:
                try:
                    explanation = await task
                    explanations[method_name] = explanation
                except Exception as e:
                    logger.error(f"{method_name} explanation failed: {e}")
                    explanations[method_name] = {"error": str(e)}
            
            # Generate unified explanation summary
            unified_explanation = await self._create_unified_explanation(
                explanations, prediction_result
            )
            
            total_time = (time.time() - start_time) * 1000
            
            return {
                "model_name": model_name,
                "prediction": prediction_result,
                "explanations": explanations,
                "unified_explanation": unified_explanation,
                "total_explanation_time_ms": total_time,
                "methods_used": list(explanations.keys())
            }
            
        except Exception as e:
            logger.error(f"Comprehensive explanation generation failed: {e}")
            raise RuntimeError(f"Explanation generation failed: {e}")
    
    async def _create_unified_explanation(
        self,
        explanations: Dict[str, Any],
        prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create unified explanation combining insights from all methods.
        
        Synthesizes SHAP, LIME, and GradCAM results into coherent explanation.
        """
        unified = {
            "primary_factors": [],
            "confidence_factors": [],
            "uncertainty_sources": [],
            "clinical_relevance": ""
        }
        
        # Combine feature importance from different methods
        all_features = {}
        
        for method, explanation in explanations.items():
            if "feature_importance" in explanation:
                for feature, importance in explanation["feature_importance"].items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
        
        # Average feature importance across methods
        averaged_features = {
            feature: np.mean(importances)
            for feature, importances in all_features.items()
        }
        
        # Identify primary factors
        sorted_features = sorted(
            averaged_features.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        unified["primary_factors"] = [
            {"feature": feature, "importance": importance}
            for feature, importance in sorted_features[:5]
        ]
        
        # Generate clinical relevance explanation
        unified["clinical_relevance"] = self._generate_clinical_explanation(
            sorted_features, prediction_result
        )
        
        return unified
    
    def _generate_clinical_explanation(
        self,
        sorted_features: List[Tuple[str, float]],
        prediction_result: Dict[str, Any]
    ) -> str:
        """Generate clinical explanation of prediction."""
        probability = prediction_result.get("probability", 0.5)
        top_feature = sorted_features[0][0] if sorted_features else "unknown"
        
        if probability > 0.7:
            return f"High PCOS risk indicated primarily by {top_feature}. Medical consultation recommended."
        elif probability > 0.3:
            return f"Moderate PCOS risk with {top_feature} as key indicator. Monitoring advised."
        else:
            return f"Low PCOS risk. {top_feature} shows minimal concern."
    
    async def get_explainability_status(self) -> Dict[str, Any]:
        """Get status of all explainability methods."""
        return {
            "initialized": self.is_initialized,
            "enabled_methods": self.enabled_methods,
            "available_explainers": list(self.shap_explainer.explainers.keys()),
            "shap_ready": len(self.shap_explainer.explainers) > 0,
            "lime_ready": len(self.lime_explainer.explainers) > 0,
            "gradcam_ready": len(self.gradcam_explainer.explainers) > 0
        }


# Global explainability manager
explainability_manager = ExplainabilityManager()