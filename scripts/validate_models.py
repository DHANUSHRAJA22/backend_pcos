#!/usr/bin/env python3
"""
Model validation script for production deployment.

Validates all models in the system and ensures they're ready
for production inference with proper preprocessing consistency.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from utils.model_management import model_validator
from utils.preprocessing import validate_preprocessing_consistency


async def validate_all_models():
    """Validate all models in the system."""
    print("🔍 PCOS Analyzer Model Validation")
    print("=" * 50)
    
    validation_results = {
        "face_models": {},
        "xray_models": {},
        "preprocessing_check": {},
        "overall_status": "unknown"
    }
    
    # Validate face models
    print("\n📸 Validating Face Models:")
    for model_name, model_path in settings.MODEL_PATHS["face"].items():
        print(f"  Validating {model_name}...")
        result = await model_validator.validate_face_model(model_path)
        validation_results["face_models"][model_name] = result
        
        if result["valid"]:
            print(f"    ✅ {model_name}: VALID")
        else:
            print(f"    ❌ {model_name}: INVALID")
            for error in result["errors"]:
                print(f"       Error: {error}")
    
    # Validate X-ray models
    print("\n🩻 Validating X-ray Models:")
    for model_name, model_path in settings.MODEL_PATHS["xray"].items():
        print(f"  Validating {model_name}...")
        result = await model_validator.validate_xray_model(model_path)
        validation_results["xray_models"][model_name] = result
        
        if result["valid"]:
            print(f"    ✅ {model_name}: VALID")
        else:
            print(f"    ❌ {model_name}: INVALID")
            for error in result["errors"]:
                print(f"       Error: {error}")
    
    # Validate preprocessing consistency
    print("\n🔧 Validating Preprocessing Pipeline:")
    preprocessing_check = validate_preprocessing_consistency()
    validation_results["preprocessing_check"] = preprocessing_check
    
    if preprocessing_check["face_preprocessing_consistent"]:
        print("    ✅ Face preprocessing: CONSISTENT")
    else:
        print("    ❌ Face preprocessing: INCONSISTENT")
    
    if preprocessing_check["xray_preprocessing_consistent"]:
        print("    ✅ X-ray preprocessing: CONSISTENT")
    else:
        print("    ❌ X-ray preprocessing: INCONSISTENT")
    
    # Overall status
    face_valid = all(result["valid"] for result in validation_results["face_models"].values())
    xray_valid = all(result["valid"] for result in validation_results["xray_models"].values())
    preprocessing_valid = (
        preprocessing_check["face_preprocessing_consistent"] and
        preprocessing_check["xray_preprocessing_consistent"]
    )
    
    if face_valid and xray_valid and preprocessing_valid:
        validation_results["overall_status"] = "READY"
        print("\n🎉 Overall Status: READY FOR PRODUCTION")
    else:
        validation_results["overall_status"] = "NOT_READY"
        print("\n⚠️  Overall Status: NOT READY - Fix errors above")
    
    # Save validation report
    report_path = "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n📄 Validation report saved to: {report_path}")
    
    return validation_results["overall_status"] == "READY"


if __name__ == "__main__":
    """
    Run model validation from command line.
    
    Usage:
        python scripts/validate_models.py
    """
    success = asyncio.run(validate_all_models())
    sys.exit(0 if success else 1)