
# iris-mlops

An inference server for iris classification using Flask with MLflow integration, containerized with Docker, designed for ML lifecycle management and deployment.

---

## Features

- Train, evaluate, register, and stage ML models with MLflow
- REST API for inference with live model reloading
- Dockerized app and MLflow tracking server
- Metrics endpoint with latency, status code, and endpoint statistics
- Periodic background model reload for zero downtime

---

## Prerequisites

- Python 3.11+ (for local dev)
- Docker & Docker Compose (for containerized deployment)
- Git (optional, for tracking SHA in MLflow)

---

## Setup Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/iris-mlops.git
   cd iris-mlops
   ```

2. Create and activate virtual environment:

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -U pip setuptools wheel
   pip install -r requirements.txt
   ```

4. Setup environment variables:

   Create `.env` file (optional):

   ```
   MLFLOW_TRACKING_URI=http://localhost:5000
   MLFLOW_MODEL_NAME=iris_classifier
   ```

---

## Setup with Docker

1. Build the Docker image:

   ```bash
   docker compose build
   ```

2. Start containers:

   ```bash
   docker compose up
   ```

3. The MLflow UI is available at [http://localhost:5000](http://localhost:5000)  
   API server available at [http://localhost:5001/api/health](http://localhost:5001/api/health)

---

## Running Locally

### Start MLflow Tracking Server

    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow/mlruns --host 0.0.0.0 --port 5000
    ```

### Run Flask API

    ```bash
    python app.py
    ```

---

## Running with Docker

Start all services with Docker Compose:

```bash
docker compose up
```

---

## Training, Evaluating, Registering & Staging Models

### Locally

Running in host system locally

    ```bash
    python src/train.py
    python src/register.py
    python src/stage.py promote
    python src/evaluate.py <alias>
    ```

### With Docker

Run training inside the app container:

    ```bash
    docker exec -it iris-mlops-app-1 python /app/src/train.py
    docker exec -it iris-mlops-app-1 python /app/src/register.py
    docker exec -it iris-mlops-app-1 python /app/src/stage.py promote
    docker exec -it iris-mlops-app-1 python /app/src/evaluate.py <alias>
    ```

This script will:

- Train models
- Register the best model to MLflow Model Registry
- Stage the best model (to `staging` or `production` depending on config)
- Evaluate models on given stage name (alias)

---

## Model Inference

### Locally

Send a POST request to predict endpoint:

    ```bash
    curl -X POST http://localhost:5001/api/predict \
      -H "Content-Type: application/json" \
      -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}'
    ```

### With Docker

Same as local but using the Docker mapped port.

---

## Troubleshooting & Notes

- **Permission errors with MLflow artifacts:** Ensure volume directories have correct ownership matching container UID/GID.
- **Git warnings in MLflow:** Install git or set `GIT_PYTHON_REFRESH=quiet` environment variable.
- **Model not loaded:** API auto-reloads models periodically; manual reload can be implemented if needed.
