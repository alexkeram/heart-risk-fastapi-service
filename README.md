# Heart Risk Classification

A web service and Python library for assessing heart attack risk.

This project includes:
- an exploratory notebook (EDA + training, selection of the best model),
- a library (`src/`) with preprocessing/training/inference classes,
- a FastAPI web application (`app/`) for uploading CSV files and obtaining predictions,
- artifacts of the best model in `artifacts/` (for immediate app launch).

---

## ✨ Features

- Automatic cleaning and type conversion of the input CSV (`EDAAnalyzer`);
- Training several models with cross-validation and choosing the best by F-beta score (`HeartRiskRunner` / `HeartRiskJob`);
- Saving and loading model artifacts (metadata + model file);
- Local inference in code/notebook (`HeartRiskInference`);
- Web interface for batch predictions (CSV upload/download) **and** JSON API;
- API responses in JSON format.

**Output format:** CSV with columns
`patient_id`, `proba`, `prediction` (0 — low risk, 1 — high).

---

## 📁 Repository Structure

```
.
├─ app/                      # FastAPI application
│  ├─ static/                # styles
│  ├─ templates/             # Jinja2 (index.html)
│  └─ main.py                # application entry point
├─ artifacts/                # best model artifacts (best_meta.json, best_model.*)
├─ data/                     # (optional) raw data
├─ notebooks/
│  └─ heart_risk_notebook.ipynb
├─ src/                      # library (OOP)
│  ├─ eda_analyzer.py
│  ├─ heart_runner.py
│  ├─ heart_job.py
│  ├─ inference_utils.py
│  └─ quick_view.py
├─ requirements.txt
└─ README.md / README.txt
```

---

## 🔧 Installation

```bash
# 1) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt
```

> Minimum: `pandas`, `numpy`, `scikit-learn`, `catboost`, `shap`,
> `fastapi`, `uvicorn`, `joblib`, `nest-asyncio` (for running from Jupyter).

---

## 🧪 Training and saving artifacts (in notebook)

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
After execution, `artifacts/` will contain:
- `best_meta.json` — metadata,
- `best_model.cbm` **or** `best_model.joblib` — model file.

---

## 🔍 Inference in code / notebook

```python
import pandas as pd
from src.inference_utils import HeartRiskInference

inf = HeartRiskInference.from_dir()  # always uses ./artifacts
df_test = pd.read_csv("data/heart_test.csv")

preds = inf.predict(df_test)         # DataFrame: proba, prediction
print(preds.head())

# class distribution
dist = preds["prediction"].value_counts().to_dict()
print("Low risk (0):", dist.get(0, 0))
print("High risk (1):", dist.get(1, 0))
```

---

## 🌐 Web application (FastAPI)

### Run from console

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Open: http://127.0.0.1:8000
- Upload CSV
- View prediction list
- Download results (CSV)

### Run directly from Jupyter Notebook
```python
import nest_asyncio, uvicorn
nest_asyncio.apply()
uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
```

---

## 🔌 JSON API (endpoints)

Base URL: `http://127.0.0.1:8000`

### 1) Check: `GET /health`
Service availability check.
```bash
curl http://127.0.0.1:8000/health
```
Response:
```json
{"status": "ok", "version": "<hash>"}
```

### 2) Prediction by CSV path: `POST /api/predict`
Request body:
```json
{ "path": "data/heart_test.csv" }
```

Example (bash/PowerShell):
```bash
curl -X POST http://127.0.0.1:8000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{ \"path\": \"data/heart_test.csv\" }"
```
Response (example):
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
Returns `predictions.csv` (columns: `patient_id,proba,prediction`) — generated after the last inference (via web form or `/api/predict`).

---

## 🗂️ `artifacts/` folder

Expected files:
- `best_meta.json`
- `best_model.cbm` **or** `best_model.joblib`

---

## 🧩 Main classes (src/)

- **`EDAAnalyzer`** — cleaning/type casting/imputation + train/test consistency.
- **`HeartRiskRunner`** — CV for CatBoost / HistGB / RF, F-β threshold tuning, bootstrap CI, SHAP top.
- **`HeartRiskJob`** — orchestrator: `run()` (training + report), `save()` (artifacts), `run_and_save()`.
- **`HeartRiskInference`** — load artifacts, prepare features and predictions (`predict_proba`, `predict`, `class_distribution`).
- **`QuickView`** — quick plots/summaries (optional).

---

## ✅ Deliverables

- Notebook with exploration and training;
- Application and library code (`src/`, `app/`);
- Launch instructions (this README);
- Reproducibility: `requirements.txt`.
- Prediction file `predictions.csv` in the submissions folder
