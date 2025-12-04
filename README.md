# ML Inference Service & CI/CD (FastAPI, Docker, GitHub Actions)
Developed a production-ready REST API for model serving with integrated observability metrics (latency, request counts). Built a rigorous CI pipeline including linting (ruff), unit testing (pytest), and contract tests to ensure deployment reliability.


Heart Risk Classification is an end-to-end data project: from data exploration and training, through packaging model artifacts, to a reproducible API service with logging, metrics, tests, Docker, Makefile automation and CI.

The project includes:
- an exploratory notebook (EDA and model training, selection of the best model),
- a Python library in `src/` for preprocessing, training and inference,
- a FastAPI web application in `app/` for batch predictions over CSV and JSON API,
- saved model artifacts in `artifacts/` so that the service can be started immediately,
- Dockerfile, docker compose, Makefile, pytest tests, logging, `/metrics` endpoint and GitHub Actions CI.

---

## Data Engineering and Infrastructure Features

- **Reproducible environment**
  - Pinned Python dependencies in `requirements.txt`.
  - Separate development dependencies (pytest, ruff).
  - `.gitignore` configured for virtual environment and build artifacts.

- **Makefile based one click run**
  - Commands for local development, Docker build and run, linting and tests.
  - Typical scenario: prepare environment and run the service with two commands:
    ```bash
    make init        # bootstrap: venv, dependencies, docker compose pull
    make run-open    # start containers and open the UI in browser
    ```

- **Logging and observability**
  - Centralised logging for the FastAPI service using the standard `logging` module.
  - Information logs for application startup, `/health`, `/predict` and `/metrics` calls.
  - Error logs with full tracebacks for easier debugging.

- **Metrics endpoint `/metrics`**
  - Simple observability layer for the API.
  - The service tracks:
    - total number of processed requests,
    - response time and p95 latency.
  - Metrics are exposed via `/metrics` endpoint in a machine readable format and can be scraped by systems like Prometheus.

- **Automated tests (pytest)**
  - API tests through `TestClient` for `/health`, `/api/predict` and `/download`.
  - Positive and negative tests for inference utilities in `src/`.
  - Basic contract tests that ensure response structure stays stable even when internals change.

- **Docker and docker compose**
  - `Dockerfile` based on a slim Python image.
  - The image installs dependencies, copies code and exposes the FastAPI application on port 8000.
  - `docker-compose.yml` provides a convenient way to run the service with a single command.

- **CI with GitHub Actions**
  - Workflow in `.github/workflows/ci.yml`.
  - On each push or pull request the pipeline:
    - sets up Python,
    - installs runtime and development dependencies,
    - runs linter (`ruff check .`),
    - runs tests (`pytest`).
  - The CI status badge can be added to the top of this README to show build status.
---

## Makefile Commands

The `Makefile` provides a toolbox to work with the project. The most used targets are:

```makefile
make init        # create venv, install deps, pull docker images
make dev         # run uvicorn locally without Docker
make run         # docker compose up -d
make open        # wait for port and open http://localhost:8000/
make run-open    # run + open in browser
make stop        # docker compose down
make lint        # ruff check .
make test        # pytest -q
make ci          # local CI: lint + tests
make build       # docker build -t heart-risk-api:latest .
make doctor      # check Docker and ensure .venv exists
```

For a quick local demonstration you can either use `make dev` to run without Docker or the `make init` and `make run-open` pair to go through the full containerised flow.

The full `help` target also documents additional commands for tagging and pushing an image, viewing logs, cleaning caches and so on.

---

## Machine Learning Features

Even though the project is presented as a Data Engineering showcase, the ML part is still fully functional.

Main capabilities:

- automatic cleaning and type conversion of the input CSV (`EDAAnalyzer`);
- training several models with cross validation and choosing the best one by F beta score (`HeartRiskRunner` and `HeartRiskJob`);
- saving and loading model artifacts (metadata and model file in `artifacts/`);
- local inference in code or notebooks via `HeartRiskInference`;
- FastAPI web interface for batch predictions (CSV upload and download) and JSON API;
- API responses in JSON format that include a compact summary and per patient predictions.

**Output CSV format:** columns `patient_id`, `proba`, `prediction` (0 means low risk, 1 means high risk).

---

## Repository Structure

```text
.
├─ app/                      # FastAPI application
│  ├─ static/                # styles
│  ├─ templates/             # Jinja2 (index.html)
│  └─ main.py                # application entry point
├─ artifacts/                # best model artifacts (best_meta.json, best_model.*)
├─ data/                     # raw data
├─ notebooks/
│  └─ heart_risk_notebook.ipynb
├─ src/                      # library
│  ├─ eda_analyzer.py
│  ├─ heart_runner.py
│  ├─ heart_job.py
│  ├─ inference_utils.py
│  └─ quick_view.py
├─ tests/                    # pytest tests for API and utilities
├─ docker-compose.yml
├─ Dockerfile
├─ Makefile
├─ requirements.txt
└─ README.md
```
---

## Training and saving artifacts 
### In notebook heart_risk_notebook.ipynb

```python
from pathlib import Path
import pandas as pd
from src.heart_job import HeartRiskJob, RunCfg

# data
train_df = pd.read_csv(Path("data") / "heart_train.csv")

# run experiment and save the best model to ./artifacts
job = HeartRiskJob(RunCfg(), artifacts_dir=Path("artifacts"))
meta = job.run_and_save(train_df)
meta
```

After execution the `artifacts/` folder will contain:
- `best_meta.json` with metadata,
- `best_model.cbm` or `best_model.joblib` with the trained model.

---

## Inference in code or notebook

```python
import pandas as pd
from src.inference_utils import HeartRiskInference

inf = HeartRiskInference.from_dir()  # uses ./artifacts by default
df_test = pd.read_csv("data/heart_test.csv")

preds = inf.predict(df_test)         # DataFrame with proba and prediction
print(preds.head())

dist = preds["prediction"].value_counts().to_dict()
print("Low risk (0):", dist.get(0, 0))
print("High risk (1):", dist.get(1, 0))
```

---

## Web application

### Run with Makefile and Docker (recommended for demonstration)

```bash
make init
make run-open
```

The first command prepares the environment and pulls images, the second one brings up the containers and opens the UI in the browser.

### Run from the console without Makefile

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

- http://127.0.0.1:8000 – main page with CSV upload,
- http://127.0.0.1:8000/docs – OpenAPI documentation,
- http://127.0.0.1:8000/metrics – service metrics.

### Run directly from Jupyter Notebook

```python
import nest_asyncio, uvicorn
nest_asyncio.apply()
uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
```

---

## JSON API endpoints

Base URL: `http://127.0.0.1:8000`

### 1) Health check: `GET /health`

Checks service availability.

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{"status": "ok", "version": "<hash>"}
```

### 2) Prediction by CSV path: `POST /api/predict`

Request body:

```json
{ "path": "data/heart_test.csv" }
```

Example (PowerShell or bash):

```bash
curl -X POST http://127.0.0.1:8000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{ \"path\": \"data/heart_test.csv\" }"
```

Example response:

```json
{
  "summary": {
    "n_patients": 1000,
    "n_high": 123, "p_high": 12.3,
    "n_low": 877,  "p_low": 87.7
  },
  "predictions": [
    {"patient_id": 0, "proba": 0.812, "prediction": 1},
    {"patient_id": 1, "proba": 0.132, "prediction": 0}
  ]
}
```

### 3) Download latest results (CSV): `GET /download`

Returns `predictions.csv` with columns `patient_id,proba,prediction`, generated after the last inference (via web form or `/api/predict`).

---

## `artifacts/` folder

Expected files:
- `best_meta.json`
- `best_model.cbm` or `best_model.joblib`

These are enough for the API to start and serve predictions without retraining.

---

## Main classes in `src/`

- `EDAAnalyzer` – cleaning, type casting, imputation and train or test consistency checks.
- `HeartRiskRunner` – cross validation for CatBoost, HistGradientBoosting and RandomForest, F beta threshold tuning, bootstrap confidence intervals and SHAP based feature importance.
- `HeartRiskJob` – orchestration layer that combines training and reporting, and saves artifacts.
- `HeartRiskInference` – loading artifacts, preparing features and running predictions with helper methods such as `predict_proba`, `predict` and `class_distribution`.
- `QuickView` – optional helper for quick plots and summaries during exploration.

---

## Deliverables and what this project demonstrates

- exploratory notebook for data understanding and model selection;
- reusable Python library for training and inference;
- FastAPI service with HTML and JSON interfaces;
- one click like run with Makefile and Docker;
- logging and metrics for basic observability;
- automated tests with pytest;
- CI pipeline in GitHub Actions;
- pinned dependencies and simple environment bootstrap.

The model itself can be retrained or replaced, while the surrounding infrastructure remains the same and can be reused for other data services.
