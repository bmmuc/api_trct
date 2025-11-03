# Anomaly Detection API

API developed with **FastAPI** for **anomaly detection in time series** of industrial machines.  
Supports multiple models, versioning, real-time metrics, and Docker execution.

---

## Main Features
- **Multiple models:** statistical (μ±3σ) and Isolation Forest (scikit-learn)  
- **Model versioning** by time series  
- **Flexible storage:** file system or Amazon S3  
- **Real-time metrics:** latency, throughput, and system health  
- **Visualization:** plots with detection thresholds  
- **High performance:** inference < 1ms  
- **Docker ready**

---

## Simplified Structure
```
src/
├── api/             # Endpoints and dependencies
├── anomaly_models/  # Models (statistical, sklearn, external)
├── services/        # Business logic
├── storage/         # Filesystem and S3
├── utils/           # Metrics, logging
├── models/          # Pydantic schemas
├── tests/           # Unit tests
└── main.py          # Entry point
```

---

## Technologies
FastAPI · Uvicorn · Pydantic · NumPy · Matplotlib · Pytest · Docker

---

## Installation
```bash
git clone <repo-url>
cd tractian
make install        # or uv sync --all-extras
cp .env.example .env
```

---

## Quick Configuration
Main `.env` variables:
```bash
API_PORT=8000
APP_MODEL_TYPE=statistical     # or sklearn
APP_STORAGE_TYPE=filesystem    # or s3
LOG_LEVEL=INFO
```

---

## Makefile Usage
```bash
make dev           # start local server without docker
make test          # run tests
make lint          # check lint
make performance   # run performance tests
make docker-build  # build Docker image
make docker-run-local
```

---

## Main Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/fit/{series_id}` | Train model for the series |
| `POST` | `/predict/{series_id}` | Predict if value is anomalous |
| `GET`  | `/healthcheck` | Return metrics and status |
| `GET`  | `/plot/{series_id}` | Generate plot |
| `GET`  | `/docs` | Swagger UI |

---

## Supported Models
**1️⃣ Statistical** – μ±kσ (fast and lightweight)  
**2️⃣ Sklearn*** – baseclass

---

## Storage & Metrics
- **Storage:** `filesystem` (local) or `s3`  
- **Metrics:** in-memory or exported via **Prometheus**

---

## Performance Results

### Load Tests (1000 concurrent requests)

#### Model Training
| Metric | Value |
|---------|-------|
| Total Requests | 1000 |
| Successful Requests | **1000 (100%)** |
| Failed Requests | 0 (0%) |
| Total Time | 0.78s |
| Min Latency | 106.96ms |
| **Avg Latency** | **341.97ms** |
| Median Latency | 344.97ms |
| P95 Latency | 567.50ms |
| P99 Latency | 597.70ms |
| Max Latency | 607.36ms |
| **Throughput** | **1285.30 req/s** |

#### Inference (Prediction)
| Metric | Value |
|---------|-------|
| Total Requests | 1000 |
| Successful Requests | **1000 (100%)** |
| Failed Requests | 0 (0%) |
| Total Time | 0.58s |
| Min Latency | 41.10ms |
| **Avg Latency** | **258.20ms** |
| Median Latency | 257.83ms |
| P95 Latency | 470.89ms |
| P99 Latency | 489.54ms |
| Max Latency | 495.00ms |
| **Throughput** | **1723.13 req/s** |

#### System Metrics (After Tests)
| Metric | Value |
|---------|-------|
| Trained Series | 1000 |
| **Avg Inference Latency** | **0.3311542615498582ms** |
| **P95 Inference Latency** | **0.4984855651855469ms** |
| **Avg Training Latency** | **0.7598506605654733ms** |
| **P95 Training Latency** | **1.0830163955688477ms** |

**Run tests:**
```bash
make dev              # start API
make performance      # run performance tests
```

---

## Test Coverage

### Overall Summary
- **Total Coverage:** 83.75%
- **Linting Score:** 10/10
- **Total Statements:** 831
- **Covered Statements:** 696
- **Uncovered Statements:** 135

### Coverage by Module
| Module | Statements | Missing | Coverage |
|--------|-------------|---------|----------|
| `anomaly_models/base_model.py` | 3 | 0 | **100.00%** ✓ |
| `config.py` | 37 | 0 | **100.00%** ✓ |
| `services/anomaly_service.py` | 28 | 0 | **100.00%** ✓ |
| `services/local_training_service.py` | 21 | 0 | **100.00%** ✓ |
| `services/visualization_service.py` | 40 | 0 | **100.00%** ✓ |
| `storage/base_storage.py` | 4 | 0 | **100.00%** ✓ |
| `utils/base_metrics.py` | 3 | 0 | **100.00%** ✓ |
| `models/schemas.py` | 82 | 4 | 95.12% |
| `utils/logger.py` | 18 | 1 | 94.44% |
| `anomaly_models/statistical_model.py` | 36 | 2 | 94.44% |
| `anomaly_models/anomaly_model.py` | 27 | 2 | 92.59% |
| `anomaly_models/model_factory.py` | 11 | 1 | 90.91% |
| `api/dependencies.py` | 49 | 6 | 87.76% |
| `main.py` | 23 | 3 | 86.96% |
| `exceptions.py` | 30 | 5 | 83.33% |
| `utils/memory_metrics.py` | 37 | 8 | 78.38% |
| `storage/model_store.py` | 125 | 29 | 76.80% |
| `utils/metrics_factory.py` | 11 | 3 | 72.73% |
| `storage/storage_factory.py` | 11 | 3 | 72.73% |
| `storage/filesystem_storage.py` | 147 | 43 | 70.75% |
| `api/routes.py` | 85 | 25 | 70.59% |
| `services/base_training_service.py` | 3 | 0 | **100.00%** ✓ |
| **TOTAL** | **831** | **135** | **83.75%** |

### Test Suites
- **`test_anomaly_service.py`** - Anomaly detection service tests  
- **`test_model_store.py`** - Storage and versioning tests  
- **`test_routes.py`** - API routes integration tests  
- **`test_validation.py`** - Data validation tests  

**Run tests:**
```bash
make test             # run all tests
make test-cov         # run with coverage report
make lint             # check code quality
```

**Generated Reports:**
- Performance report: `performance_report.json` and `performance_report.md`

---

## Docker
```bash
make docker-build
make docker-run-local
```
Access:  
📍 API → [http://localhost:8000](http://localhost:8000)  
📘 Docs → [http://localhost:8000/docs](http://localhost:8000/docs)

---