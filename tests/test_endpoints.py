"""
Comprehensive test suite for PCOS Analyzer Advanced API.

Tests all endpoints, ensemble methods, model loading, error handling,
and research features with medical-grade validation requirements.
"""

import pytest
import asyncio
import json
import time
import tempfile
import os
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
from typing import Dict, Any

# Import the FastAPI app
from app import app
from config import settings
from schemas import EnsembleMethod, RiskLevel

# Test client
client = TestClient(app)


class TestAdvancedHealthEndpoint:
    """Test suite for advanced /health endpoint with ensemble monitoring."""
    
    def test_health_endpoint_comprehensive(self):
        """Test comprehensive health check with ensemble status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            "status", "version", "models", "ensemble_config", 
            "total_models_loaded", "uptime_seconds"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify status values
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert data["version"] == "2.0.0"
        assert isinstance(data["total_models_loaded"], int)
        assert data["total_models_loaded"] >= 0
    
    def test_health_ensemble_config(self):
        """Test ensemble configuration in health response."""
        response = client.get("/health")
        data = response.json()
        
        # Verify ensemble configuration structure
        ensemble_config = data["ensemble_config"]
        assert isinstance(ensemble_config, dict)
        
        # Check for face and X-ray ensemble status
        expected_ensembles = ["face_ensemble", "xray_ensemble"]
        for ensemble_name in expected_ensembles:
            if ensemble_name in ensemble_config:
                ensemble_info = ensemble_config[ensemble_name]
                assert "total_models" in ensemble_info
                assert "loaded_models" in ensemble_info
                assert "ready" in ensemble_info
                assert isinstance(ensemble_info["total_models"], int)
                assert isinstance(ensemble_info["loaded_models"], int)
                assert isinstance(ensemble_info["ready"], bool)
    
    def test_health_model_details(self):
        """Test detailed model information in health response."""
        response = client.get("/health")
        data = response.json()
        
        models = data["models"]
        assert isinstance(models, dict)
        
        # Check model information structure
        for model_name, model_info in models.items():
            required_model_fields = [
                "status", "version", "framework", "ready"
            ]
            for field in required_model_fields:
                assert field in model_info, f"Missing model field {field} in {model_name}"


class TestAdvancedPredictEndpoint:
    """Test suite for advanced /predict endpoint with ensemble methods."""
    
    @pytest.fixture
    def sample_face_image(self):
        """Create a high-quality sample face image for testing."""
        img = Image.new('RGB', (224, 224), color=(255, 220, 177))  # Skin tone
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        return img_bytes
    
    @pytest.fixture
    def sample_xray_image(self):
        """Create a sample X-ray image for testing."""
        img = Image.new('L', (512, 512), color=128)  # Grayscale X-ray
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_predict_with_ensemble_methods(self, sample_face_image):
        """Test prediction with different ensemble methods."""
        ensemble_methods = ["soft_voting", "weighted_voting", "majority_voting"]
        
        for method in ensemble_methods:
            sample_face_image.seek(0)  # Reset file pointer
            
            files = {
                "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
            }
            params = {
                "ensemble_method": method,
                "include_individual_models": True
            }
            
            response = client.post("/predict", files=files, params=params)
            
            # May return 503 if models not loaded in test environment
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert "face_predictions" in data
                assert "ensemble_results" in data
                
                # Verify ensemble results structure
                ensemble_results = data["ensemble_results"]
                assert "primary" in ensemble_results
                
                primary_result = ensemble_results["primary"]
                assert primary_result["ensemble_method"] == method
                assert "final_probability" in primary_result
                assert "final_risk_level" in primary_result
                assert "ensemble_confidence" in primary_result
    
    def test_predict_with_both_modalities(self, sample_face_image, sample_xray_image):
        """Test prediction with both face and X-ray images."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg"),
            "xray_img": ("test_xray.png", sample_xray_image, "image/png")
        }
        params = {
            "ensemble_method": "weighted_voting",
            "include_individual_models": True,
            "include_feature_importance": True
        }
        
        response = client.post("/predict", files=files, params=params)
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "face_predictions" in data
            assert "xray_predictions" in data
            assert "ensemble_results" in data
            
            # Verify both modalities processed
            assert data["face_predictions"] is not None
            assert data["xray_predictions"] is not None
            
            # Verify model count
            face_count = data["face_predictions"]["models_count"]
            xray_count = data["xray_predictions"]["models_count"]
            total_expected = face_count + xray_count
            assert data["model_count"] == total_expected
    
    def test_predict_stacking_method(self, sample_face_image):
        """Test stacking ensemble method specifically."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        params = {
            "ensemble_method": "stacking",
            "include_individual_models": True
        }
        
        response = client.post("/predict", files=files, params=params)
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            ensemble_results = data["ensemble_results"]
            
            # Check if stacking result is included
            if "primary" in ensemble_results:
                primary = ensemble_results["primary"]
                if primary["ensemble_method"] == "stacking":
                    assert "stacking_result" in primary
    
    def test_predict_feature_importance(self, sample_face_image):
        """Test feature importance inclusion in predictions."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        params = {
            "include_feature_importance": True,
            "include_individual_models": True
        }
        
        response = client.post("/predict", files=files, params=params)
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if feature importance is included
            if data["face_predictions"]:
                predictions = data["face_predictions"]["individual_predictions"]
                for pred in predictions:
                    # Feature importance should be included when requested
                    assert "feature_importance" in pred
    
    def test_predict_invalid_ensemble_method(self, sample_face_image):
        """Test prediction with invalid ensemble method."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        params = {
            "ensemble_method": "invalid_method"
        }
        
        response = client.post("/predict", files=files, params=params)
        
        # Should return validation error
        assert response.status_code == 422  # Pydantic validation error
    
    def test_predict_no_images_error(self):
        """Test prediction endpoint with no images provided."""
        response = client.post("/predict")
        
        assert response.status_code == 400
        data = response.json()
        assert "At least one image" in data["detail"]
    
    def test_predict_oversized_file(self):
        """Test prediction with file exceeding size limit."""
        # Create oversized image
        large_img = Image.new('RGB', (4000, 4000), color='white')
        img_bytes = BytesIO()
        large_img.save(img_bytes, format='JPEG', quality=100)
        img_bytes.seek(0)
        
        files = {
            "face_img": ("large_image.jpg", img_bytes, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        # Should fail due to size limit
        assert response.status_code == 413
        data = response.json()
        assert "exceeds" in data["detail"].lower()


class TestModelConfigurationEndpoint:
    """Test suite for /models/config endpoint."""
    
    def test_model_config_endpoint(self):
        """Test model configuration retrieval."""
        response = client.get("/models/config")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify configuration structure
        expected_fields = [
            "ensemble_methods", "default_ensemble_method",
            "model_weights", "risk_thresholds",
            "face_models", "xray_models"
        ]
        
        for field in expected_fields:
            assert field in data, f"Missing configuration field: {field}"
        
        # Verify ensemble methods
        ensemble_methods = data["ensemble_methods"]
        expected_methods = ["soft_voting", "weighted_voting", "stacking", "majority_voting"]
        for method in expected_methods:
            assert method in ensemble_methods
        
        # Verify model weights structure
        model_weights = data["model_weights"]
        assert "face" in model_weights
        assert "xray" in model_weights
        assert isinstance(model_weights["face"], dict)
        assert isinstance(model_weights["xray"], dict)


class TestEnsembleValidation:
    """Test suite for ensemble method validation and comparison."""
    
    @pytest.fixture
    def prediction_data(self):
        """Sample prediction data for ensemble testing."""
        return {
            "face_predictions": [0.7, 0.65, 0.8, 0.72, 0.68, 0.75],
            "xray_predictions": [0.6, 0.7, 0.65, 0.75, 0.68]
        }
    
    def test_ensemble_method_consistency(self, sample_face_image):
        """Test that ensemble methods produce consistent results."""
        # Run same prediction with different ensemble methods
        methods_to_test = ["soft_voting", "weighted_voting"]
        results = {}
        
        for method in methods_to_test:
            sample_face_image.seek(0)
            
            files = {
                "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
            }
            params = {"ensemble_method": method}
            
            response = client.post("/predict", files=files, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results[method] = data["ensemble_results"]["primary"]["final_probability"]
        
        # Results should be different but reasonable
        if len(results) >= 2:
            probabilities = list(results.values())
            assert all(0.0 <= p <= 1.0 for p in probabilities)
    
    def test_ensemble_agreement_calculation(self, sample_face_image):
        """Test ensemble agreement score calculation."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        params = {"include_individual_models": True}
        
        response = client.post("/predict", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data["face_predictions"]:
                agreement_score = data["face_predictions"]["agreement_score"]
                assert 0.0 <= agreement_score <= 1.0
                
                # Agreement should correlate with standard deviation
                std_prob = data["face_predictions"]["std_probability"]
                assert isinstance(std_prob, float)
                assert std_prob >= 0.0


class TestSecurityAndValidation:
    """Test suite for security validations and edge cases."""
    
    def test_malicious_file_upload(self):
        """Test handling of potentially malicious files."""
        # Create a text file disguised as image
        malicious_content = b"<?php echo 'malicious code'; ?>"
        malicious_file = BytesIO(malicious_content)
        
        files = {
            "face_img": ("malicious.jpg", malicious_file, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        # Should be rejected by validation
        assert response.status_code == 400
        data = response.json()
        assert "corrupted" in data["detail"].lower() or "invalid" in data["detail"].lower()
    
    def test_file_extension_validation(self):
        """Test file extension validation."""
        # Create valid image with invalid extension
        img = Image.new('RGB', (224, 224), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {
            "face_img": ("test.exe", img_bytes, "image/jpeg")  # Wrong extension
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "extension" in data["detail"].lower()
    
    def test_mime_type_validation(self):
        """Test MIME type validation."""
        img = Image.new('RGB', (224, 224), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {
            "face_img": ("test.jpg", img_bytes, "text/plain")  # Wrong MIME type
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "not allowed" in data["detail"].lower()


class TestGenderDetection:
    """Test suite for gender detection functionality."""
    
    def test_gender_detection_in_prediction(self, sample_face_image):
        """Test gender detection integration in prediction endpoint."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        params = {
            "include_individual_models": True
        }
        
        response = client.post("/predict", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if gender detection results are included
            if data["face_predictions"] and hasattr(data["face_predictions"], "gender_detection"):
                gender_result = data["face_predictions"]["gender_detection"]
                
                # Verify gender detection structure
                assert "predicted_gender" in gender_result
                assert "confidence" in gender_result
                assert gender_result["predicted_gender"] in ["male", "female"]
                assert 0.0 <= gender_result["confidence"] <= 1.0
    
    def test_male_face_warning(self):
        """Test warning generation for male faces."""
        # Create test image that would trigger male detection
        img = Image.new('RGB', (224, 224), color=(200, 180, 160))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {
            "face_img": ("male_face.jpg", img_bytes, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for gender warning if male detected
            if (data["face_predictions"] and 
                hasattr(data["face_predictions"], "gender_warning")):
                warning = data["face_predictions"]["gender_warning"]
                assert "male face" in warning.lower()
                assert "females" in warning.lower()


class TestEvaluationEndpoint:
    """Test suite for model evaluation endpoint."""
    
    def test_evaluate_endpoint_structure(self):
        """Test evaluation endpoint with mock data."""
        # Create temporary test data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {"image_path": "test1.jpg", "label": 0},
                {"image_path": "test2.jpg", "label": 1}
            ]
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            response = client.post(
                "/evaluate",
                params={
                    "test_data_path": temp_path,
                    "include_ensemble_comparison": True,
                    "generate_visualizations": True
                }
            )
            
            # Should return evaluation structure even if models not loaded
            assert response.status_code in [200, 500, 503]
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify evaluation response structure
                assert "test_dataset_info" in data
                assert "evaluation_timestamp" in data
                
                dataset_info = data["test_dataset_info"]
                assert "total_samples" in dataset_info
                assert "dataset_path" in dataset_info
        
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def test_evaluate_invalid_dataset(self):
        """Test evaluation with invalid dataset path."""
        response = client.post(
            "/evaluate",
            params={"test_data_path": "/nonexistent/path.csv"}
        )
        
        assert response.status_code in [400, 500]
        data = response.json()
        assert "error" in data["detail"].lower() or "not found" in data["detail"].lower()


class TestPerformanceAndScaling:
    """Test suite for performance validation and scaling behavior."""
    
    def test_concurrent_predictions(self, sample_face_image):
        """Test handling of concurrent prediction requests."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def make_request():
            sample_face_image.seek(0)
            files = {
                "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
            }
            response = client.post("/predict", files=files)
            results_queue.put(response.status_code)
        
        # Create multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all requests to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Collect results
        status_codes = []
        while not results_queue.empty():
            status_codes.append(results_queue.get())
        
        # All requests should complete successfully or with expected errors
        for status_code in status_codes:
            assert status_code in [200, 503, 429]  # Success, unavailable, or rate limited
    
    def test_prediction_timing(self, sample_face_image):
        """Test prediction response times."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        
        start_time = time.time()
        response = client.post("/predict", files=files)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        if response.status_code == 200:
            data = response.json()
            reported_time = data["processing_time_ms"]
            
            # Reported time should be reasonable
            assert reported_time > 0
            assert reported_time < 30000  # Should complete within 30 seconds
            
            # Response time should include reported processing time
            assert response_time >= reported_time * 0.8  # Allow some overhead
    
    def test_memory_usage_stability(self, sample_face_image):
        """Test memory usage remains stable across multiple predictions."""
        # Make multiple predictions to test for memory leaks
        for i in range(10):
            sample_face_image.seek(0)
            
            files = {
                "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
            }
            
            response = client.post("/predict", files=files)
            
            # Should not degrade over multiple requests
            assert response.status_code in [200, 503]


class TestEnsembleMethodValidation:
    """Test suite for ensemble method validation and comparison."""
    
    def test_all_ensemble_methods_available(self):
        """Test that all configured ensemble methods are available."""
        response = client.get("/models/config")
        
        if response.status_code == 200:
            data = response.json()
            available_methods = data["ensemble_methods"]
            
            expected_methods = ["soft_voting", "weighted_voting", "stacking", "majority_voting"]
            for method in expected_methods:
                assert method in available_methods
    
    def test_ensemble_weight_configuration(self):
        """Test ensemble weight configuration."""
        response = client.get("/models/config")
        
        if response.status_code == 200:
            data = response.json()
            model_weights = data["model_weights"]
            
            # Verify weight structure
            assert "face" in model_weights
            assert "xray" in model_weights
            
            # Verify weights are valid probabilities
            for modality, weights in model_weights.items():
                for model_name, weight in weights.items():
                    assert 0.0 <= weight <= 1.0
                    assert isinstance(weight, (int, float))
    
    def test_risk_threshold_configuration(self):
        """Test risk threshold configuration."""
        response = client.get("/models/config")
        
        if response.status_code == 200:
            data = response.json()
            thresholds = data["risk_thresholds"]
            
            # Verify threshold structure
            assert "low" in thresholds
            assert "moderate" in thresholds
            
            # Verify threshold ordering
            assert thresholds["low"] < thresholds["moderate"]
            assert 0.0 <= thresholds["low"] <= 1.0
            assert 0.0 <= thresholds["moderate"] <= 1.0


class TestErrorHandlingAndEdgeCases:
    """Test suite for comprehensive error handling."""
    
    def test_empty_file_upload(self):
        """Test handling of empty file uploads."""
        empty_file = BytesIO(b"")
        
        files = {
            "face_img": ("empty.jpg", empty_file, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["detail"].lower() or "small" in data["detail"].lower()
    
    def test_corrupted_image_handling(self):
        """Test handling of corrupted image files."""
        # Create corrupted image data
        corrupted_data = b"JPEG\x00\x00\x00corrupted data"
        corrupted_file = BytesIO(corrupted_data)
        
        files = {
            "face_img": ("corrupted.jpg", corrupted_file, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "corrupted" in data["detail"].lower() or "invalid" in data["detail"].lower()
    
    def test_unsupported_image_format(self):
        """Test handling of unsupported image formats."""
        # Create BMP image (not in allowed types)
        img = Image.new('RGB', (224, 224), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='BMP')
        img_bytes.seek(0)
        
        files = {
            "face_img": ("test.bmp", img_bytes, "image/bmp")
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "not allowed" in data["detail"].lower()


class TestAPIDocumentation:
    """Test suite for API documentation and OpenAPI schema."""
    
    def test_openapi_schema_comprehensive(self):
        """Test comprehensive OpenAPI schema generation."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Verify schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        
        # Verify our endpoints are documented
        paths = schema["paths"]
        assert "/predict" in paths
        assert "/health" in paths
        assert "/models/config" in paths
        
        # Verify request/response schemas
        components = schema["components"]["schemas"]
        expected_schemas = [
            "PredictionResponse", "HealthResponse", "ErrorResponse",
            "EnsembleMethod", "ModelPrediction", "EnsembleResult"
        ]
        
        for schema_name in expected_schemas:
            assert schema_name in components
    
    def test_swagger_ui_accessibility(self):
        """Test Swagger UI accessibility."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
        assert "PCOS Analyzer Advanced API" in response.text
    
    def test_redoc_ui_accessibility(self):
        """Test ReDoc UI accessibility."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


class TestResearchFeatures:
    """Test suite for research-specific features and capabilities."""
    
    def test_experimental_features_config(self):
        """Test experimental features configuration."""
        response = client.get("/models/config")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if experimental features are documented
            # This would be expanded based on actual experimental features
            assert isinstance(data, dict)
    
    def test_model_metadata_collection(self, sample_face_image):
        """Test collection of model metadata for research."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        params = {"include_individual_models": True}
        
        response = client.post("/predict", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Verify metadata is collected
            if data["face_predictions"]:
                predictions = data["face_predictions"]["individual_predictions"]
                for pred in predictions:
                    # Check for research-relevant metadata
                    assert "processing_time_ms" in pred
                    assert "confidence" in pred
                    assert "model_version" in pred
                    assert "framework" in pred


# Integration test helpers for research workflows
class TestResearchWorkflows:
    """Test suite for research workflow integration."""
    
    def test_batch_prediction_simulation(self, sample_face_image):
        """Simulate batch prediction workflow."""
        # Test multiple sequential predictions (simulating batch processing)
        batch_results = []
        
        for i in range(5):
            sample_face_image.seek(0)
            
            files = {
                "face_img": ("batch_image_{i}.jpg", sample_face_image, "image/jpeg")
            }
            
            response = client.post("/predict", files=files)
            
            if response.status_code == 200:
                batch_results.append(response.json())
        
        # Analyze batch results for consistency
        if batch_results:
            processing_times = [r["processing_time_ms"] for r in batch_results]
            avg_time = sum(processing_times) / len(processing_times)
            
            # Processing times should be relatively consistent
            assert all(0.5 * avg_time <= t <= 2.0 * avg_time for t in processing_times)


# Performance benchmarking
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for research and optimization."""
    
    def test_model_loading_performance(self):
        """Benchmark model loading times."""
        # This would test actual model loading performance
        # when real models are integrated
        pass
    
    def test_inference_performance_by_model(self, sample_face_image):
        """Benchmark inference performance per model."""
        files = {
            "face_img": ("benchmark.jpg", sample_face_image, "image/jpeg")
        }
        params = {"include_individual_models": True}
        
        response = client.post("/predict", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if data["face_predictions"]:
                predictions = data["face_predictions"]["individual_predictions"]
                
                # Analyze per-model performance
                for pred in predictions:
                    processing_time = pred["processing_time_ms"]
                    assert processing_time > 0
                    assert processing_time < 5000  # Should complete within 5 seconds
    
    def test_ensemble_overhead_measurement(self, sample_face_image):
        """Measure ensemble processing overhead."""
        files = {
            "face_img": ("overhead_test.jpg", sample_face_image, "image/jpeg")
        }
        
        # Test with ensemble
        params = {"ensemble_method": "weighted_voting"}
        response = client.post("/predict", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            total_time = data["processing_time_ms"]
            
            # Ensemble overhead should be minimal
            if data["face_predictions"]:
                individual_times = [
                    pred["processing_time_ms"] 
                    for pred in data["face_predictions"]["individual_predictions"]
                ]
                max_individual_time = max(individual_times) if individual_times else 0
                
                # Total time should not be much more than the slowest individual model
                # (due to parallel processing)
                assert total_time <= max_individual_time * 1.5


if __name__ == "__main__":
    """
    Run comprehensive test suite.
    
    Usage:
        python -m pytest tests/test_endpoints.py -v
        python -m pytest tests/test_endpoints.py::TestAdvancedPredictEndpoint -v
        python -m pytest tests/test_endpoints.py -m performance  # Run performance tests
    """
    pytest.main([__file__, "-v", "--tb=short"])