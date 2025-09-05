"""
Model management utilities for automated updates and validation.

Provides tools for validating new models, updating configurations,
and managing model lifecycle in production environments.
"""

import asyncio
import logging
import json
import hashlib
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates new model files before deployment.
    
    Performs comprehensive checks including file integrity,
    architecture compatibility, and basic inference testing.
    """
    
    def __init__(self):
        self.validation_history = []
    
    async def validate_face_model(self, model_path: str) -> Dict[str, Any]:
        """
        Validate a new face model before deployment.
        
        Args:
            model_path: Path to new model file
            
        Returns:
            Dict: Validation results and recommendations
        """
        validation_result = {
            "model_path": model_path,
            "valid": False,
            "checks": {},
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check file exists and is readable
            if not os.path.exists(model_path):
                validation_result["errors"].append(f"Model file not found: {model_path}")
                return validation_result
            
            # Check file format
            if model_path.endswith('.h5'):
                validation_result["checks"]["file_format"] = "keras_h5"
                
                # Try to load Keras model
                try:
                    from tensorflow import keras
                    test_model = keras.models.load_model(model_path)
                    validation_result["checks"]["model_loads"] = True
                    
                    # Check model architecture
                    input_shape = test_model.input_shape
                    output_shape = test_model.output_shape
                    
                    validation_result["checks"]["input_shape"] = input_shape
                    validation_result["checks"]["output_shape"] = output_shape
                    
                    # Validate expected dimensions
                    if len(input_shape) == 4 and input_shape[1:3] == (100, 100):
                        validation_result["checks"]["input_dimensions_correct"] = True
                    else:
                        validation_result["warnings"].append(
                            f"Unexpected input shape: {input_shape}, expected: (None, 100, 100, 3)"
                        )
                    
                    # Test inference with dummy data
                    dummy_input = np.random.random((1, 100, 100, 3))
                    test_output = test_model.predict(dummy_input, verbose=0)
                    validation_result["checks"]["inference_test"] = True
                    
                    # Clean up
                    del test_model
                    
                except Exception as e:
                    validation_result["errors"].append(f"Model loading failed: {e}")
                    validation_result["checks"]["model_loads"] = False
            
            else:
                validation_result["errors"].append(f"Unsupported model format: {model_path}")
            
            # Calculate file hash for version tracking
            validation_result["file_hash"] = await self._calculate_file_hash(model_path)
            
            # Determine overall validity
            validation_result["valid"] = (
                len(validation_result["errors"]) == 0 and
                validation_result["checks"].get("model_loads", False) and
                validation_result["checks"].get("inference_test", False)
            )
            
            self.validation_history.append(validation_result)
            return validation_result
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            return validation_result
    
    async def validate_xray_model(self, model_path: str) -> Dict[str, Any]:
        """
        Validate a new X-ray model before deployment.
        
        Args:
            model_path: Path to new model file
            
        Returns:
            Dict: Validation results and recommendations
        """
        validation_result = {
            "model_path": model_path,
            "valid": False,
            "checks": {},
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check file exists
            if not os.path.exists(model_path):
                validation_result["errors"].append(f"Model file not found: {model_path}")
                return validation_result
            
            # Check file format
            if model_path.endswith('.pt'):
                validation_result["checks"]["file_format"] = "pytorch_pt"
                
                # Try to load YOLOv8 model
                try:
                    from ultralytics import YOLO
                    test_model = YOLO(model_path)
                    validation_result["checks"]["model_loads"] = True
                    
                    # Test inference with dummy image
                    dummy_image = Image.new('RGB', (640, 640), color='gray')
                    test_results = test_model(dummy_image, verbose=False)
                    validation_result["checks"]["inference_test"] = True
                    
                    # Check model info
                    model_info = test_model.info(verbose=False)
                    validation_result["checks"]["model_info"] = str(model_info)
                    
                except Exception as e:
                    validation_result["errors"].append(f"YOLOv8 model loading failed: {e}")
                    validation_result["checks"]["model_loads"] = False
            
            else:
                validation_result["errors"].append(f"Unsupported model format: {model_path}")
            
            # Calculate file hash
            validation_result["file_hash"] = await self._calculate_file_hash(model_path)
            
            # Determine validity
            validation_result["valid"] = (
                len(validation_result["errors"]) == 0 and
                validation_result["checks"].get("model_loads", False) and
                validation_result["checks"].get("inference_test", False)
            )
            
            self.validation_history.append(validation_result)
            return validation_result
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            return validation_result
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of model file for version tracking."""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return "hash_calculation_failed"


class ModelVersionManager:
    """
    Manages model versions and deployment history.
    
    Tracks all model deployments, maintains version history,
    and provides rollback capabilities for production safety.
    """
    
    def __init__(self):
        self.version_file = "models/model_versions.json"
        self.versions = self._load_version_history()
    
    def _load_version_history(self) -> Dict[str, Any]:
        """Load version history from JSON file."""
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "version_history": [],
                    "current_version": "1.0.0",
                    "last_updated": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to load version history: {e}")
            return {"version_history": [], "current_version": "1.0.0"}
    
    async def register_new_version(
        self,
        version_tag: str,
        model_updates: Dict[str, Dict[str, Any]],
        ensemble_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new model version in the system.
        
        Args:
            version_tag: Version identifier (e.g., "1.1.0")
            model_updates: Dictionary of model updates
            ensemble_config: Optional ensemble configuration updates
            
        Returns:
            bool: True if registration successful
        """
        try:
            new_version = {
                "version": version_tag,
                "timestamp": datetime.now().isoformat(),
                "models": model_updates,
                "ensemble_config": ensemble_config or {},
                "deployed": False
            }
            
            self.versions["version_history"].append(new_version)
            self.versions["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            await self._save_version_history()
            
            logger.info(f"Registered new model version: {version_tag}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register version {version_tag}: {e}")
            return False
    
    async def _save_version_history(self):
        """Save version history to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")
    
    def get_current_version(self) -> str:
        """Get currently deployed version."""
        return self.versions.get("current_version", "1.0.0")
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get complete version history."""
        return self.versions.get("version_history", [])


class AutomatedModelUpdater:
    """
    Automated model update and deployment system.
    
    Provides CLI and API utilities for seamless model updates
    with validation, testing, and rollback capabilities.
    """
    
    def __init__(self):
        self.validator = ModelValidator()
        self.version_manager = ModelVersionManager()
    
    async def update_model(
        self,
        model_type: str,  # "face" or "xray"
        model_name: str,
        new_model_path: str,
        version_tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update a specific model with validation and version tracking.
        
        Args:
            model_type: Type of model ("face" or "xray")
            model_name: Name of the model to update
            new_model_path: Path to new model file
            version_tag: Optional version tag
            
        Returns:
            Dict: Update results and status
        """
        try:
            logger.info(f"Updating {model_type} model: {model_name}")
            
            # Validate new model
            if model_type == "face":
                validation_result = await self.validator.validate_face_model(new_model_path)
            elif model_type == "xray":
                validation_result = await self.validator.validate_xray_model(new_model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Model validation failed",
                    "validation_result": validation_result
                }
            
            # TODO: Implement hot-swap logic
            # 1. Backup current model
            # 2. Copy new model to production location
            # 3. Trigger model reload via hot-swap API
            # 4. Validate ensemble still works
            # 5. Update version tracking
            
            # For now, just update version tracking
            if not version_tag:
                version_tag = f"auto_{int(time.time())}"
            
            model_update = {
                model_type: {
                    model_name: {
                        "file": os.path.basename(new_model_path),
                        "validation_result": validation_result,
                        "update_timestamp": datetime.now().isoformat()
                    }
                }
            }
            
            await self.version_manager.register_new_version(
                version_tag, model_update
            )
            
            return {
                "success": True,
                "version_tag": version_tag,
                "validation_result": validation_result,
                "next_steps": [
                    "Copy model to production location",
                    "Trigger hot-swap via API",
                    "Validate ensemble performance"
                ]
            }
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instances
model_validator = ModelValidator()
version_manager = ModelVersionManager()
automated_updater = AutomatedModelUpdater()