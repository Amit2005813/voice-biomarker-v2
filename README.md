# Voice Biomarker Platform v2

## Depression Detection via Voice Analysis

A production-ready platform that uses deep learning to detect depression biomarkers from voice recordings. Built with FastAPI, Streamlit, and state-of-the-art audio analysis pipelines.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Load Balancer (Nginx)                          │
│                    SSL Termination · Rate Limiting · Gzip                │
└──────────┬─────────────────────────────────────────────┬─────────────────┘
           │                                             │
           ▼                                             ▼
┌─────────────────────┐                     ┌─────────────────────────┐
│    API Service      │                     │    Frontend Service     │
│     (FastAPI)       │                     │     (Streamlit)         │
│                     │                     │                         │
│  /api/v1/analyze    │◄───────────────────►│  Upload Audio UI        │
│  /api/v1/models     │                     │  Results Dashboard      │
│  /api/v1/history    │                     │  Model Comparison       │
└──────────┬──────────┘                     └─────────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐ ┌──────────┐
│  Redis   │ │ Database │
│ (Cache)  │ │ (SQLite/ │
│          │ │ Postgres)│
└──────────┘ └──────────┘
```

---

## Features

- **Multi-model inference**: XGBoost, LightGBM, SVM, Random Forest, Neural Network
- **Audio feature extraction**: MFCCs, mel spectrograms, prosodic features, jitter/shimmer
- **Real-time analysis**: Streaming audio processing with WebSocket support
- **Batch processing**: Analyze multiple recordings in parallel
- **REST API**: Full OpenAPI/Swagger documentation at `/docs`
- **Interactive dashboard**: Streamlit-based visualization and comparison tools
- **Model training**: End-to-end training pipeline with MLflow tracking
- **Caching**: Redis-backed result caching for repeat analyses
- **Health monitoring**: Built-in health checks, Prometheus metrics
- **CI/CD**: GitHub Actions pipeline with automated testing and deployment
- **Kubernetes ready**: Full K8s manifests for production deployment
- **Docker support**: Multi-stage builds with optimized images

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| Docker | 24.0+ | Containerization |
| Docker Compose | 2.20+ | Multi-service orchestration |
| Node.js | 18+ | Frontend build (optional) |
| FFmpeg | 5.0+ | Audio processing |
| libsndfile | 1.1+ | Audio file I/O |

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip \
    libsndfile1 ffmpeg liblapack-dev libblas-dev gfortran

# macOS
brew install python@3.11 libsndfile ffmpeg

# Windows (WSL2 recommended)
choco install python@3.11 ffmpeg
```

---

## Installation

### Option 1: Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/voice-biomarker-v2.git
cd voice-biomarker-v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env

# Start Redis (required)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run migrations
python -m src.db.migrate

# Start development server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Streamlit
streamlit run src/frontend/app.py --server.port 8501
```

### Option 2: Docker Compose

```bash
# Clone and configure
git clone https://github.com/your-org/voice-biomarker-v2.git
cd voice-biomarker-v2

# Configure environment
cp .env.example .env
# Edit .env with your settings (at minimum, set SECRET_KEY)

# Build and start all services
docker compose build
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### Option 3: Kubernetes

```bash
# Clone the repository
git clone https://github.com/your-org/voice-biomarker-v2.git
cd voice-biomarker-v2

# Configure secrets
kubectl create namespace voice-biomarker

# Update kubernetes/secrets.yaml with real values, then apply:
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/redis-deployment.yaml
kubectl apply -f kubernetes/api-deployment.yaml
kubectl apply -f kubernetes/api-service.yaml
kubectl apply -f kubernetes/frontend-deployment.yaml
kubectl apply -kubernetes/frontend-service.yaml

# Verify deployment
kubectl get pods -n voice-biomarker
kubectl get svc -n voice-biomarker
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `production` | Application environment |
| `APP_HOST` | `0.0.0.0` | Server bind address |
| `APP_PORT` | `8000` | Server port |
| `APP_WORKERS` | `4` | Uvicorn worker count |
| `LOG_LEVEL` | `info` | Logging level |
| `SECRET_KEY` | *(required)* | Application secret key |
| `DATABASE_URL` | `sqlite:///./data/voice_biomarker.db` | Database connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MODEL_PATH` | `/app/models` | Model storage directory |
| `UPLOAD_PATH` | `/app/uploads` | Upload storage directory |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_UPLOAD_SIZE` | `52428800` | Max upload size in bytes (50MB) |

### Generate Secret Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Configuration Files

- `configs/default.yaml` — Default application configuration
- `configs/production.yaml` — Production overrides
- `configs/training.yaml` — Model training hyperparameters

---

## Usage Guide

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI documentation |
| `GET` | `/redoc` | ReDoc documentation |
| `POST` | `/api/v1/analyze` | Analyze a voice recording |
| `POST` | `/api/v1/analyze/batch` | Batch analysis |
| `GET` | `/api/v1/models` | List available models |
| `GET` | `/api/v1/models/{model_id}` | Get model details |
| `GET` | `/api/v1/history` | Get analysis history |
| `GET` | `/api/v1/history/{analysis_id}` | Get specific analysis |
| `GET` | `/api/v1/metrics` | Prometheus metrics |

### Analyze a Voice Recording

```bash
# Single file analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@recording.wav" \
  -F "model=xgboost" \
  -F "include_features=true"

# Response
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "depression_probability": 0.72,
  "risk_level": "moderate",
  "confidence": 0.89,
  "model_used": "xgboost",
  "features": {
    "mfcc_mean": [...],
    "jitter": 0.0123,
    "shimmer": 0.0456,
    "speaking_rate": 3.2
  },
  "duration_seconds": 45.2,
  "created_at": "2026-07-22T10:30:00Z"
}
```

### Batch Analysis

```bash
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -F "files=@recording1.wav" \
  -F "files=@recording2.wav" \
  -F "files=@recording3.wav" \
  -F "model=ensemble"

# Response
{
  "batch_id": "batch-a1b2c3d4",
  "total_files": 3,
  "status": "processing",
  "estimated_completion": "2026-07-22T10:31:00Z"
}
```

### Check Analysis Status

```bash
curl http://localhost:8000/api/v1/history/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Frontend Dashboard

Access the Streamlit dashboard at `http://localhost:8501`:

1. **Upload**: Drag and drop or select audio files (.wav, .mp3, .flac)
2. **Configure**: Select model, feature extraction options
3. **Analyze**: Click "Analyze" to run the prediction pipeline
4. **Results**: View depression risk score, feature importance, confidence intervals
5. **History**: Browse past analyses, compare models, export results

---

## Training Guide

### Quick Start

```bash
# Prepare your dataset (see Dataset Guide below)
python -m src.training.prepare \
  --data-dir ./data/raw \
  --output-dir ./data/processed

# Train all models
python -m src.training.train \
  --config configs/training.yaml \
  --data-dir ./data/processed \
  --output-dir ./models/checkpoints

# Evaluate models
python -m src.training.evaluate \
  --model-dir ./models/checkpoints \
  --test-data ./data/processed/test \
  --output-dir ./results
```

### Training Configuration

```yaml
# configs/training.yaml
training:
  random_seed: 42
  test_size: 0.2
  validation_size: 0.15
  cross_validation_folds: 5

features:
  sample_rate: 22050
  frame_length: 2048
  hop_length: 512
  n_mfcc: 20
  n_mels: 128
  include_prosodic: true
  include_spectral: true

models:
  xgboost:
    n_estimators: 500
    max_depth: 8
    learning_rate: 0.05
    subsample: 0.8

  lightgbm:
    n_estimators: 500
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.8

  neural_network:
    hidden_layers: [256, 128, 64]
    dropout: 0.3
    learning_rate: 0.001
    batch_size: 32
    epochs: 100
    early_stopping_patience: 10
```

### MLflow Integration

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# Train with MLflow tracking
python -m src.training.train \
  --config configs/training.yaml \
  --mlflow-uri http://localhost:5000 \
  --experiment-name voice-biomarker-v2
```

---

## Dataset Guide

### mPower Dataset (Parkinson's)

The mPower dataset from Sage Bionetworks provides voice recordings for Parkinson's disease research. It can be adapted for depression biomarker research.

1. **Download**: Visit [Synapse](https://www.synapse.org/#!Synapse:syn4993293)
2. **Registration**: Create a Synapse account and accept data use agreement
3. **Extract**: Download voice exercise recordings
4. **Preprocess**:

```bash
python -m src.data.preprocess_mpower \
  --input-dir ./raw/mpower \
  --output-dir ./data/mpower \
  --target-duration 30 \
  --sample-rate 22050
```

### DAIC-WOZ Dataset (Depression)

The DAIC-WOZ dataset from USC ICT contains interviews with depression labels from PHQ-8 questionnaires.

1. **Download**: Visit [Harbor-UCLA Medical Center](https://dcic-world.org/)
2. **Registration**: Complete data use agreement and IRB requirements
3. **Extract**: Download interview audio files and PHQ-8 scores
4. **Preprocess**:

```bash
python -m src.data.preprocess_daicwoz \
  --input-dir ./raw/daicwoz \
  --output-dir ./data/daicwoz \
  --target-duration 60 \
  --sample-rate 22050 \
  --phq8-threshold 10
```

### Custom Datasets

For custom voice datasets, ensure:

- Audio files are in WAV, MP3, or FLAC format
- Minimum 10 seconds of speech per sample
- Sample rate of 22050 Hz (resampled automatically)
- Labels follow the format:

```csv
file_path,label,phq8_score
recordings/s001.wav,depressed,15
recordings/s002.wav,not_depressed,3
```

---

## API Documentation

### Authentication

```bash
# Include API key in requests
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/api/v1/analyze
```

### Rate Limits

| Tier | Requests/min | Burst |
|------|-------------|-------|
| Free | 30 | 5 |
| Pro | 100 | 20 |
| Enterprise | 1000 | 100 |

### Error Responses

```json
{
  "error": {
    "code": "INVALID_AUDIO_FORMAT",
    "message": "Unsupported audio format. Accepted: wav, mp3, flac, ogg",
    "details": {
      "received_format": "pdf"
    }
  }
}
```

| Status Code | Description |
|------------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad request / Invalid input |
| 401 | Unauthorized |
| 413 | File too large |
| 415 | Unsupported media type |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and write tests
4. Run the test suite: `pytest tests/ -v --cov=src`
5. Run linters: `black src/ tests/ && isort src/ tests/ && flake8 src/ tests/`
6. Commit your changes: `git commit -m "feat: add my feature"`
7. Push to the branch: `git push origin feature/my-feature`
8. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Run all checks
make check
```

### Code Standards

- **Formatting**: Black (line length 120)
- **Import sorting**: isort
- **Linting**: Flake8 with E203, W503 ignored
- **Type checking**: MyPy (strict mode)
- **Security**: Bandit

---

## Medical Disclaimer

> **IMPORTANT**: This tool is intended for **research and screening purposes only**. It is NOT a diagnostic tool and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Depression is a serious medical condition that requires evaluation by a qualified healthcare professional.
>
> - Predictions are probabilistic estimates, not diagnoses
> - Results should be interpreted by qualified healthcare providers
> - False positives and false negatives are possible
> - This system has not been validated for clinical use
> - Always consult a licensed mental health professional for diagnosis and treatment
>
> If you or someone you know is experiencing a mental health crisis, contact:
> - **National Suicide Prevention Lifeline**: 988 (US)
> - **Crisis Text Line**: Text HOME to 741741
> - **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Troubleshooting

### Common Issues

**Port already in use**
```bash
# Find and stop the process using the port
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

**Audio file not recognized**
```bash
# Check file format and codec
ffprobe -v error -show_format -show_streams recording.wav

# Convert to supported format
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav
```

**Docker build fails**
```bash
# Clean Docker cache
docker system prune -a
docker compose build --no-cache
```

**Redis connection refused**
```bash
# Check Redis is running
docker ps | grep redis
docker compose logs redis

# Restart Redis
docker compose restart redis
```

**Model files not found**
```bash
# Verify model directory
ls -la models/checkpoints/

# Download pre-trained models
python -m src.models.download --output-dir models/checkpoints
```

**Memory errors during analysis**
```bash
# Reduce worker count
export APP_WORKERS=2

# Or increase Docker memory limit
docker compose up -d --memory=4g api
```

**CUDA/GPU issues**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/voice-biomarker-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/voice-biomarker-v2/discussions)
- **Email**: support@voicebiomarker.com
