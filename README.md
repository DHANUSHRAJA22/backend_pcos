# PCOS ClarityScan - Production FastAPI Backend

A production-ready FastAPI backend for AI-powered PCOS risk assessment using facial and X-ray image analysis. Features modular architecture, hot-swappable models, and comprehensive API documentation.

## 🏗️ Architecture Overview

```
backend/
├── app.py                    # Main FastAPI application
├── config.py                 # Configuration and settings
├── schemas.py               # Pydantic data models
├── ensemble.py              # Ensemble prediction logic
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── .env.example            # Environment configuration template
├── models/                 # AI model implementations
│   ├── __init__.py
│   ├── base_model.py       # Abstract base class for all models
│   ├── face_model.py       # VGG16 facial analysis
│   ├── xray_model.py       # YOLOv8 X-ray analysis
│   ├── gender_detector.py  # Gender classification
│   └── model_loader.py     # Dynamic model loading
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── validators.py       # File validation
│   └── preprocessing.py    # Image preprocessing
└── tests/                  # Test suite
    ├── __init__.py
    └── test_api.py         # API endpoint tests
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Model Paths

Create `.env` file from template:
```bash
cp .env.example .env
```

Update model paths in `.env`:
```bash
PCOS_FACE_MODEL_PATH=models/pcos_detector_158.h5
PCOS_FACE_LABELS_PATH=models/pcos_detector_158.labels.txt
PCOS_XRAY_MODEL_PATH=models/bestv8.pt
```

### 3. Place Your Model Files

```bash
# Create models directory
mkdir -p models

# Copy your trained models
cp /path/to/pcos_detector_158.h5 models/
cp /path/to/pcos_detector_158.labels.txt models/
cp /path/to/bestv8.pt models/
```

### 4. Run Development Server

```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Verify API is Running

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### POST /predict

Upload images for PCOS risk assessment.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "face_img=@facial_image.jpg" \
  -F "xray_img=@xray_image.png" \
  -G -d "ensemble_method=weighted_voting"
```

**JavaScript/TypeScript:**
```typescript
const formData = new FormData();
formData.append('face_img', faceImageFile);
formData.append('xray_img', xrayImageFile);

const response = await fetch('http://localhost:8000/predict?ensemble_method=weighted_voting', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log('PCOS Risk:', result.ensemble_result.final_probability);
```

**Response:**
```json
{
  "success": true,
  "face_predictions": {
    "individual_predictions": [
      {
        "model_name": "vgg16_face",
        "probability": 0.73,
        "predicted_label": "moderate",
        "confidence": 0.85,
        "processing_time_ms": 45.2
      }
    ],
    "average_probability": 0.73,
    "consensus_label": "moderate",
    "model_count": 1,
    "gender_detection": {
      "predicted_gender": "female",
      "confidence": 0.92,
      "warning": null
    }
  },
  "ensemble_result": {
    "ensemble_method": "weighted_voting",
    "final_probability": 0.71,
    "final_risk_level": "moderate",
    "confidence": 0.88,
    "model_agreement": 0.95
  },
  "processing_time_ms": 245.7,
  "model_count": 2
}
```

### GET /health

Check API and model status.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models": {
    "face_vgg16_primary": {
      "status": "loaded",
      "model_path": "models/pcos_detector_158.h5",
      "framework": "tensorflow",
      "version_hash": "a1b2c3d4e5f6",
      "total_predictions": 127
    },
    "xray_yolov8_primary": {
      "status": "loaded",
      "model_path": "models/bestv8.pt",
      "framework": "ultralytics",
      "version_hash": "f6e5d4c3b2a1"
    }
  },
  "total_models_loaded": 3,
  "uptime_seconds": 3600.5,
  "frontend_cors_enabled": true
}
```

### POST /models/swap

Hot swap models with zero downtime.

**Request:**
```bash
curl -X POST "http://localhost:8000/models/swap" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "vgg16_primary",
    "new_model_path": "models/new_pcos_model.h5",
    "modality": "face",
    "validate_before_swap": true
  }'
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PCOS_HOST` | `0.0.0.0` | API host address |
| `PCOS_PORT` | `8000` | API port |
| `PCOS_DEBUG` | `true` | Debug mode |
| `PCOS_FRONTEND_URL` | `http://localhost:8080` | Frontend URL for CORS |
| `PCOS_FACE_MODEL_PATH` | `models/pcos_detector_158.h5` | VGG16 model path |
| `PCOS_FACE_LABELS_PATH` | `models/pcos_detector_158.labels.txt` | Face model labels |
| `PCOS_XRAY_MODEL_PATH` | `models/bestv8.pt` | YOLOv8 model path |

### Model Configuration

Edit `config.py` to add new models:

```python
MODEL_REGISTRY = {
    "face_models": {
        "vgg16_primary": {
            "model_path": settings.FACE_MODEL_PATH,
            "labels_path": settings.FACE_LABELS_PATH,
            "framework": "tensorflow"
        },
        # Add new face models here
        "resnet50_secondary": {
            "model_path": "models/resnet50_pcos.h5",
            "framework": "tensorflow",
            "input_size": [224, 224]
        }
    },
    "xray_models": {
        "yolov8_primary": {
            "model_path": settings.XRAY_MODEL_PATH,
            "framework": "ultralytics"
        }
        # Add new X-ray models here
    }
}
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build container
docker build -t pcos-analyzer .

# Run with model volume
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e PCOS_DEBUG=false \
  pcos-analyzer
```

### Docker Compose

```yaml
version: '3.8'
services:
  pcos-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - PCOS_DEBUG=false
      - PCOS_FRONTEND_URL=http://localhost:8080
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔄 Adding New Models

### 1. Create Model Class

```python
# In models/face_model.py or models/xray_model.py

class NewFaceModel(BaseAIModel):
    def __init__(self, model_path: str):
        super().__init__("new_face_model", model_path, "tensorflow")
    
    async def load_model(self) -> bool:
        # Load your model here
        self.model = keras.models.load_model(self.model_path)
        self.is_loaded = True
        return True
    
    async def predict(self, image_data: bytes) -> ModelPrediction:
        # Implement prediction logic
        # Return ModelPrediction object
        pass
```

### 2. Register in Configuration

```python
# In config.py
MODEL_REGISTRY = {
    "face_models": {
        "new_face_model": {
            "model_path": "models/new_model.h5",
            "framework": "tensorflow"
        }
    }
}
```

### 3. Update Manager

```python
# In models/face_model.py FaceModelManager.initialize()
if "new_face_model" in model_config:
    new_model = NewFaceModel(model_config["new_face_model"]["model_path"])
    if await new_model.initialize():
        self.models["new_face_model"] = new_model
```

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_api.py::TestPredictEndpoint -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Manual Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction with sample image
curl -X POST "http://localhost:8000/predict" \
  -F "face_img=@sample_face.jpg"
```

## 🔧 Model Hot-Swapping

### Swap Face Model

```bash
curl -X POST "http://localhost:8000/models/swap" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "vgg16_primary",
    "new_model_path": "models/improved_pcos_model.h5",
    "modality": "face",
    "validate_before_swap": true
  }'
```

### Swap X-ray Model

```bash
curl -X POST "http://localhost:8000/models/swap" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "yolov8_primary", 
    "new_model_path": "models/yolov8_improved.pt",
    "modality": "xray",
    "validate_before_swap": true
  }'
```

## 🎯 Frontend Integration

### React/TypeScript Example

```typescript
// api/pcos.ts
interface PredictionRequest {
  faceImage?: File;
  xrayImage?: File;
  ensembleMethod?: 'soft_voting' | 'weighted_voting' | 'majority_voting';
}

export async function predictPCOSRisk(request: PredictionRequest) {
  const formData = new FormData();
  
  if (request.faceImage) {
    formData.append('face_img', request.faceImage);
  }
  
  if (request.xrayImage) {
    formData.append('xray_img', request.xrayImage);
  }
  
  const params = new URLSearchParams();
  if (request.ensembleMethod) {
    params.append('ensemble_method', request.ensembleMethod);
  }
  
  const response = await fetch(`http://localhost:8000/predict?${params}`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.statusText}`);
  }
  
  return response.json();
}

// Usage in React component
const handlePredict = async () => {
  try {
    const result = await predictPCOSRisk({
      faceImage: selectedFaceImage,
      xrayImage: selectedXrayImage,
      ensembleMethod: 'weighted_voting'
    });
    
    console.log('PCOS Risk Level:', result.ensemble_result.final_risk_level);
    console.log('Probability:', result.ensemble_result.final_probability);
    
    // Handle gender warning
    if (result.face_predictions?.gender_detection?.warning) {
      alert(result.face_predictions.gender_detection.warning);
    }
    
  } catch (error) {
    console.error('Prediction failed:', error);
  }
};
```

### Fetch API Example

```javascript
// Simple fetch example
async function analyzePCOS(faceImage, xrayImage) {
  const formData = new FormData();
  formData.append('face_img', faceImage);
  formData.append('xray_img', xrayImage);
  
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  return {
    riskLevel: result.ensemble_result.final_risk_level,
    probability: result.ensemble_result.final_probability,
    confidence: result.ensemble_result.confidence,
    genderWarning: result.face_predictions?.gender_detection?.warning
  };
}
```

## 🔒 Production Deployment

### Environment Setup

```bash
# Production environment variables
export PCOS_DEBUG=false
export PCOS_LOG_LEVEL=INFO
export PCOS_ALLOWED_ORIGINS='["https://yourdomain.com"]'
export PCOS_FACE_MODEL_PATH=/app/models/pcos_detector_158.h5
export PCOS_XRAY_MODEL_PATH=/app/models/bestv8.pt
```

### Production Server

```bash
# Run with Gunicorn for production
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Production

```bash
# Build production image
docker build -t pcos-analyzer-prod .

# Run production container
docker run -d \
  --name pcos-api \
  -p 8000:8000 \
  -v /path/to/models:/app/models \
  -e PCOS_DEBUG=false \
  -e PCOS_ALLOWED_ORIGINS='["https://yourdomain.com"]' \
  pcos-analyzer-prod
```

## 🧪 Model Management

### Check Model Status

```bash
curl http://localhost:8000/health | jq '.models'
```

### Hot Swap Models

```bash
# Backup current model
cp models/pcos_detector_158.h5 models/backup_pcos_detector_158.h5

# Copy new model
cp /path/to/new_model.h5 models/pcos_detector_158_v2.h5

# Hot swap
curl -X POST "http://localhost:8000/models/swap" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "vgg16_primary",
    "new_model_path": "models/pcos_detector_158_v2.h5",
    "modality": "face",
    "validate_before_swap": true
  }'
```

## 🎛️ Ensemble Methods

### Available Methods

1. **Soft Voting**: Simple average of all model probabilities
2. **Weighted Voting**: Weighted average (configurable weights)
3. **Majority Voting**: Consensus based on risk level classifications
4. **Stacking**: Meta-model combination (TODO: implement with trained meta-model)

### Configure Ensemble Weights

```python
# In config.py or via environment variables
FACE_MODEL_WEIGHT = 0.6  # Face models contribute 60%
XRAY_MODEL_WEIGHT = 0.4  # X-ray models contribute 40%
```

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**: Check file paths in `.env` and verify model files exist
2. **Import errors**: Ensure all dependencies installed with `pip install -r requirements.txt`
3. **CORS errors**: Verify `PCOS_FRONTEND_URL` matches your frontend URL
4. **File upload errors**: Check file size limits and supported formats

### Debug Mode

```bash
# Enable detailed logging
export PCOS_DEBUG=true
export PCOS_LOG_LEVEL=DEBUG
python app.py
```

### Check Logs

```bash
# View application logs
tail -f logs/pcos_analyzer.log

# Or check Docker logs
docker logs pcos-api
```

## 📊 Monitoring

### Health Monitoring

```bash
# Check API health
curl http://localhost:8000/health

# Monitor model performance
curl http://localhost:8000/health | jq '.models[] | select(.status == "loaded")'
```

### Performance Metrics

The API tracks:
- Total predictions made
- Average processing time per model
- Model loading status and version hashes
- System uptime and resource usage

## 🚀 Advanced Features (TODO)

### Model Explainability
- SHAP integration for feature importance
- LIME for local explanations
- GradCAM for attention visualization

### AutoML Ensemble Optimization
- Automated ensemble method selection
- Hyperparameter optimization
- Meta-model training

### Advanced Evaluation
- Comprehensive metrics calculation
- ROC curves and confusion matrices
- Model performance comparison

## 📝 License

This project is for medical research and educational purposes. Ensure compliance with medical device regulations and data privacy laws before clinical deployment.

## 🤝 Contributing

1. Follow the modular architecture patterns
2. Add comprehensive tests for new features
3. Update documentation and API schemas
4. Ensure zero-downtime model swapping compatibility

---

**Medical Disclaimer**: This software is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.