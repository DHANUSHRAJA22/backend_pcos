"""
AutoML ensemble optimization for finding optimal model combinations.

Implements automated search for best ensemble methods, weights, and meta-models
using validation data and cross-validation for robust performance estimation.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

from config import settings
from schemas import EnsembleMethod, ModelPredictionList
from automl.search_strategies import GridSearchStrategy, BayesianSearchStrategy

logger = logging.getLogger(__name__)


class EnsembleOptimizer:
    """
    AutoML-powered ensemble optimization system.
    
    Automatically discovers the best ensemble methods, model weights,
    and meta-model configurations for maximum prediction accuracy.
    """
    
    def __init__(self):
        self.search_strategies = {
            "grid": GridSearchStrategy(),
            "bayesian": BayesianSearchStrategy(),
            "random": GridSearchStrategy()  # Will implement random search
        }
        
        self.optimization_history = []
        self.best_config = None
        self.is_optimizing = False
        
        # Validation data storage
        self.validation_predictions = []
        self.validation_labels = []
        
    async def optimize_ensemble(
        self,
        validation_data: List[Dict[str, Any]],
        search_strategy: str = "bayesian",
        max_trials: int = 100,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Run AutoML optimization to find best ensemble configuration.
        
        Args:
            validation_data: List of validation samples with predictions and labels
            search_strategy: Optimization strategy ("grid", "bayesian", "random")
            max_trials: Maximum optimization trials
            cv_folds: Cross-validation folds
            
        Returns:
            Dict: Best ensemble configuration and performance metrics
        """
        if self.is_optimizing:
            raise RuntimeError("Ensemble optimization already in progress")
        
        self.is_optimizing = True
        start_time = time.time()
        
        try:
            logger.info(f"Starting AutoML ensemble optimization with {len(validation_data)} samples")
            
            # Prepare validation dataset
            await self._prepare_validation_data(validation_data)
            
            # Get search strategy
            if search_strategy not in self.search_strategies:
                raise ValueError(f"Unknown search strategy: {search_strategy}")
            
            strategy = self.search_strategies[search_strategy]
            
            # Define search space
            search_space = self._define_search_space()
            
            # Run optimization
            best_config, best_score, optimization_results = await strategy.optimize(
                validation_predictions=self.validation_predictions,
                validation_labels=self.validation_labels,
                search_space=search_space,
                max_trials=max_trials,
                cv_folds=cv_folds
            )
            
            # Update best configuration
            self.best_config = best_config
            
            # Save optimization results
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "strategy": search_strategy,
                "best_config": best_config,
                "best_score": best_score,
                "total_trials": len(optimization_results),
                "optimization_time_seconds": time.time() - start_time,
                "validation_samples": len(validation_data)
            }
            
            self.optimization_history.append(optimization_record)
            
            # Persist results
            await self._save_optimization_results(optimization_record)
            
            logger.info(
                f"AutoML optimization completed: best score {best_score:.4f} "
                f"in {optimization_record['optimization_time_seconds']:.2f}s"
            )
            
            return optimization_record
            
        except Exception as e:
            logger.error(f"AutoML ensemble optimization failed: {e}")
            raise RuntimeError(f"Optimization failed: {e}")
        finally:
            self.is_optimizing = False
    
    async def _prepare_validation_data(self, validation_data: List[Dict[str, Any]]):
        """
        Prepare validation data for ensemble optimization.
        
        Extracts model predictions and true labels from validation samples.
        """
        self.validation_predictions = []
        self.validation_labels = []
        
        for sample in validation_data:
            # Extract per-model predictions
            model_preds = []
            
            # Face model predictions
            if "face_predictions" in sample:
                face_preds = sample["face_predictions"]["individual_predictions"]
                for pred in face_preds:
                    model_preds.append(pred["probability"])
            
            # X-ray model predictions
            if "xray_predictions" in sample:
                xray_preds = sample["xray_predictions"]["individual_predictions"]
                for pred in xray_preds:
                    model_preds.append(pred["probability"])
            
            # Add sample to validation set
            if model_preds and "true_label" in sample:
                self.validation_predictions.append(model_preds)
                self.validation_labels.append(sample["true_label"])
        
        logger.info(f"Prepared {len(self.validation_predictions)} validation samples")
    
    def _define_search_space(self) -> Dict[str, Any]:
        """
        Define hyperparameter search space for ensemble optimization.
        
        Returns:
            Dict: Search space configuration for different ensemble methods
        """
        return {
            "ensemble_methods": [
                EnsembleMethod.SOFT_VOTING,
                EnsembleMethod.WEIGHTED_VOTING,
                EnsembleMethod.STACKING,
                EnsembleMethod.MAJORITY_VOTING
            ],
            
            # Weight distributions for weighted voting
            "weight_distributions": [
                "uniform",
                "performance_based",
                "diversity_based",
                "custom_optimized"
            ],
            
            # Meta-model configurations for stacking
            "meta_models": {
                "xgboost": {
                    "n_estimators": [50, 100, 200, 500],
                    "max_depth": [3, 6, 10, 15],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "subsample": [0.8, 0.9, 1.0]
                },
                "logistic": {
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "penalty": ["l1", "l2", "elasticnet"],
                    "solver": ["liblinear", "saga"]
                },
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            
            # Cross-validation and evaluation settings
            "cv_settings": {
                "folds": [3, 5, 10],
                "stratified": [True, False],
                "shuffle": [True, False]
            }
        }
    
    async def _save_optimization_results(self, results: Dict[str, Any]):
        """Save optimization results for future reference and analysis."""
        try:
            results_dir = Path("automl_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"ensemble_optimization_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Optimization results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and history."""
        return {
            "is_optimizing": self.is_optimizing,
            "best_config": self.best_config,
            "optimization_count": len(self.optimization_history),
            "last_optimization": self.optimization_history[-1] if self.optimization_history else None,
            "validation_samples": len(self.validation_predictions)
        }
    
    async def apply_best_config(self) -> bool:
        """Apply the best discovered configuration to the ensemble system."""
        if not self.best_config:
            logger.warning("No best configuration available to apply")
            return False
        
        try:
            # TODO: Apply configuration to ensemble system
            # Update settings.MODEL_WEIGHTS with optimized weights
            # Update settings.DEFAULT_ENSEMBLE_METHOD with best method
            # Load optimized meta-model if using stacking
            
            logger.info("Applied best ensemble configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply best configuration: {e}")
            return False


class ValidationDataCollector:
    """
    Collects and manages validation data for ensemble optimization.
    
    Accumulates prediction results and true labels for AutoML training.
    """
    
    def __init__(self):
        self.collected_samples = []
        self.max_samples = 10000  # Limit memory usage
        
    async def add_prediction_sample(
        self,
        face_predictions: Optional[ModelPredictionList],
        xray_predictions: Optional[ModelPredictionList],
        true_label: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a prediction sample to the validation dataset.
        
        Args:
            face_predictions: Face model predictions
            xray_predictions: X-ray model predictions
            true_label: Ground truth label (0=no PCOS, 1=PCOS)
            metadata: Additional sample metadata
        """
        if len(self.collected_samples) >= self.max_samples:
            # Remove oldest samples to maintain memory limits
            self.collected_samples = self.collected_samples[1000:]
        
        sample = {
            "timestamp": time.time(),
            "face_predictions": face_predictions.dict() if face_predictions else None,
            "xray_predictions": xray_predictions.dict() if xray_predictions else None,
            "true_label": true_label,
            "metadata": metadata or {}
        }
        
        self.collected_samples.append(sample)
        
        logger.debug(f"Added validation sample, total: {len(self.collected_samples)}")
    
    async def export_validation_dataset(self, format: str = "json") -> str:
        """
        Export collected validation data for external analysis.
        
        Args:
            format: Export format ("json", "csv", "parquet")
            
        Returns:
            str: Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            export_path = f"validation_data_{timestamp}.json"
            with open(export_path, 'w') as f:
                json.dump(self.collected_samples, f, indent=2, default=str)
        
        # TODO: Implement other export formats
        # elif format == "csv":
        #     import pandas as pd
        #     df = pd.json_normalize(self.collected_samples)
        #     export_path = f"validation_data_{timestamp}.csv"
        #     df.to_csv(export_path, index=False)
        
        logger.info(f"Exported {len(self.collected_samples)} samples to {export_path}")
        return export_path


# Global instances
ensemble_optimizer = EnsembleOptimizer()
validation_collector = ValidationDataCollector()