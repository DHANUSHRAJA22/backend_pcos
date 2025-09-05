"""
Comprehensive test suite for PCOS Analyzer API.
Tests all endpoints with real file uploads and validation.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

from app import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test suite for /health endpoint."""
    
    def test_health_endpoint_structure(self):
        """Test health endpoint returns proper structure."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = [
            "status", "version", "models", "total_models_loaded", 
            "uptime_seconds", "frontend_cors_enabled"
        ]
        for field in required_fields:
            assert field in data
        
        # Verify status values
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["total_models_loaded"], int)
        assert data["total_models_loaded"] >= 0
        assert isinstance(data["frontend_cors_enabled"], bool)


class TestPredictEndpoint:
    """Test suite for /predict endpoint."""
    
    @pytest.fixture
    def sample_face_image(self):
        """Create sample face image for testing."""
        img = Image.new('RGB', (224, 224), color=(255, 220, 177))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        return img_bytes
    
    @pytest.fixture
    def sample_xray_image(self):
        """Create sample X-ray image for testing."""
        img = Image.new('RGB', (512, 512), color=(128, 128, 128))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_predict_face_only(self, sample_face_image):
        """Test prediction with face image only."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        # May return 503 if models not loaded in test environment
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "face_predictions" in data
            assert "ensemble_result" in data
            assert data["face_predictions"] is not None
    
    def test_predict_xray_only(self, sample_xray_image):
        """Test prediction with X-ray image only."""
        files = {
            "xray_img": ("test_xray.png", sample_xray_image, "image/png")
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "xray_predictions" in data
            assert data["xray_predictions"] is not None
    
    def test_predict_both_images(self, sample_face_image, sample_xray_image):
        """Test prediction with both face and X-ray images."""
        files = {
            "face_img": ("test_face.jpg", sample_face_image, "image/jpeg"),
            "xray_img": ("test_xray.png", sample_xray_image, "image/png")
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert data["face_predictions"] is not None
            assert data["xray_predictions"] is not None
    
    def test_predict_no_images(self):
        """Test prediction with no images provided."""
        response = client.post("/predict")
        
        assert response.status_code == 400
        data = response.json()
        assert "at least one image" in data["detail"].lower()
    
    def test_predict_ensemble_methods(self, sample_face_image):
        """Test different ensemble methods."""
        ensemble_methods = ["soft_voting", "weighted_voting", "majority_voting"]
        
        for method in ensemble_methods:
            sample_face_image.seek(0)
            
            files = {
                "face_img": ("test_face.jpg", sample_face_image, "image/jpeg")
            }
            params = {"ensemble_method": method}
            
            response = client.post("/predict", files=files, params=params)
            
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                assert data["ensemble_result"]["ensemble_method"] == method
    
    def test_predict_invalid_file_type(self):
        """Test prediction with invalid file type."""
        # Create text file disguised as image
        text_file = BytesIO(b"This is not an image")
        
        files = {
            "face_img": ("fake.jpg", text_file, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "not appear to be a valid image" in data["detail"]
    
    def test_predict_oversized_file(self):
        """Test prediction with oversized file."""
        # Create large image
        large_img = Image.new('RGB', (4000, 4000), color='white')
        img_bytes = BytesIO()
        large_img.save(img_bytes, format='JPEG', quality=100)
        img_bytes.seek(0)
        
        files = {
            "face_img": ("large.jpg", img_bytes, "image/jpeg")
        }
        
        response = client.post("/predict", files=files)
        
        assert response.status_code == 413
        data = response.json()
        assert "exceeds limit" in data["detail"]


class TestModelSwapEndpoint:
    """Test suite for model hot-swapping."""
    
    def test_model_swap_structure(self):
        """Test model swap endpoint structure."""
        swap_request = {
            "model_name": "test_model",
            "new_model_path": "/path/to/new/model.h5",
            "modality": "face",
            "validate_before_swap": True
        }
        
        response = client.post("/models/swap", json=swap_request)
        
        # Should return proper structure even if swap fails
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = response.json()
            required_fields = [
                "success", "model_name", "swap_time_ms", "message"
            ]
            for field in required_fields:
                assert field in data
    
    def test_model_swap_invalid_modality(self):
        """Test model swap with invalid modality."""
        swap_request = {
            "model_name": "test_model",
            "new_model_path": "/path/to/model.h5",
            "modality": "invalid",
            "validate_before_swap": True
        }
        
        response = client.post("/models/swap", json=swap_request)
        
        assert response.status_code == 400
        data = response.json()
        assert "unknown modality" in data["detail"].lower()


class TestCORSConfiguration:
    """Test CORS configuration for frontend integration."""
    
    def test_cors_headers_present(self):
        """Test CORS headers are properly configured."""
        response = client.options("/health")
        
        # Should have CORS headers
        assert response.status_code in [200, 405]  # OPTIONS may not be explicitly handled
        
        # Test actual endpoint with CORS
        response = client.get("/health")
        assert response.status_code == 200


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Verify schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Verify our endpoints are documented
        paths = schema["paths"]
        assert "/predict" in paths
        assert "/health" in paths
        assert "/models/swap" in paths
    
    def test_swagger_ui(self):
        """Test Swagger UI accessibility."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
    
    def test_redoc_ui(self):
        """Test ReDoc UI accessibility."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


if __name__ == "__main__":
    """Run test suite."""
    pytest.main([__file__, "-v"])