# app/main.py
"""
Heart Risk Inference — FastAPI service
======================================

This module is the entry point of the web application. It connects the web
interface (HTML + CSV upload) and the inference library
(`src/inference_utils.py`), which reads model artifacts from the `artifacts/`
folder and makes predictions.

HOW IT WORKS
------------
1. On application startup (lifespan) we load the inference wrapper once:
   `HeartRiskInference.from_dir(ARTIFACTS_DIR)`.
   ▸ It expects `best_meta.json` and the model file (`best_model.cbm` or
     `best_model.joblib`) in `artifacts/`. The path can be overridden with the
     `ARTIFACTS_DIR` environment variable; otherwise `<project root>/artifacts`
     is used.

2. The web page (`GET /`) serves a form for uploading CSV.
   `POST /` accepts CSV, computes predictions, shows the list in the browser and
   saves the last result in `app.state.last_result` for the download button.

3. JSON API:
   • `POST /api/predict_path` — accepts JSON `{ "path": "<path_to_csv>" }` and
     returns JSON with a summary and list of predictions.
   • `POST /api/predict_file` — accepts a file via multipart/form-data (as on
     the web page) and returns the same JSON.
   • `GET/POST /download` — returns a CSV with the latest predictions (if a
     computation has already been done via web or API).

4. Static files and templates:
   ▸ HTML template in `app/templates/index.html`
   ▸ Styles in `app/static/`
   ▸ The `APP_VERSION` variable is used for cache busting.

EXTENSION POINTS
----------------
• Modify/add preprocessing logic — see `src/eda_analyzer.py`.
• Change training/artifact saving logic — see `src/heart_job.py`,
  `src/heart_runner.py`.
• Inference/artifact loading logic — see `src/inference_utils.py`
  (`HeartRiskInference` class).

EXAMPLE REQUESTS
----------------
• Service health:
  GET http://127.0.0.1:8000/health

• JSON via file path (PowerShell):
  Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/predict_path' `
    -ContentType 'application/json' -Body '{"path":"data/heart_test.csv"}' | ConvertTo-Json -Depth 5

• JSON with file (PowerShell):
  Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/predict_file' `
    -ContentType 'multipart/form-data' -InFile 'data/heart_test.csv' -OutFile 'resp.json'

• Swagger: http://127.0.0.1:8000/docs
"""

import os
from io import BytesIO
from pathlib import Path
import socket
from contextlib import asynccontextmanager
from typing import List, Union

import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---- path settings -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
SRC_DIR = BASE_DIR / "src"
# ---- import modules from src/ ------------------------------------------------
from src.inference_utils import HeartRiskInference
# ---- configuration -----------------------------------------------------------
APP_VERSION = os.environ.get("APP_VERSION", os.urandom(4).hex())  # for cache busting
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", BASE_DIR / "artifacts")).resolve()
# ---- templates/static folders ------------------------------------------------
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
# ---- lifecycle: load model on startup ---------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the inference wrapper once when the service starts.
    If artifact files are missing, fail on startup with a clear error.
    """
    app.state.inf = HeartRiskInference.from_dir(ARTIFACTS_DIR)
    app.state.last_result = None  # store last computed DataFrame (for /download)
    yield
# ---- create app --------------------------------------------------------------
app = FastAPI(title="Heart Risk Inference", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ====================== Pydantic schemas for JSON ============================

class PredictionRow(BaseModel):
    """One prediction row (used in the API JSON response)."""
    patient_id: Union[int, str]
    proba: float
    prediction: int  # 0 — low risk, 1 — high risk

class PredictionSummary(BaseModel):
    """Batch summary: counts and class ratios."""
    n_patients: int
    n_high: int
    n_low: int
    p_high: float
    p_low: float

class PredictResponse(BaseModel):
    """
    Full API JSON response:
      • summary — aggregated summary,
      • predictions — list for each patient.
    """
    summary: PredictionSummary
    predictions: List[PredictionRow]

class PredictPathIn(BaseModel):
    """Input for /api/predict_path — absolute or relative path to CSV."""
    path: str

# ====================== Common prediction logic ==============================

_ID_CANDIDATES = ["patient_id", "PatientID", "patientId", "Patient Id", "id", "ID", "Unnamed: 0", "Unnamed:0"]

def _normalize_patient_id_value(v):
    """
    Normalize patient identifier to a serializable form:
    • keep integers/strings as is,
    • cast numpy types,
    • everything else to string.
    """
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            f = float(v)
            return int(f) if f.is_integer() else f
    except Exception:
        pass
    if isinstance(v, (int, str)):
        return v
    return str(v)

def _make_payload_from_df(df: pd.DataFrame) -> PredictResponse:
    """
    Common inference function for the web form and JSON endpoints:
      1) find patient_id (or use index),
      2) compute predictions via app.state.inf.predict(df),
      3) build DataFrame out (for /download),
      4) compute summary,
      5) assemble JSON response (PredictResponse).
    """
    # 1) patient_id from the input CSV
    id_col = next((c for c in _ID_CANDIDATES if c in df.columns), None)
    patient_id = df[id_col] if id_col is not None else pd.Series(df.index, name="patient_id")

    # 2) inference: returns DataFrame with columns proba, prediction
    pred_df = app.state.inf.predict(df)

    # 3) final table (for download button and possible audit)
    out = pd.DataFrame({
        "patient_id": patient_id.values,
        "proba": pred_df["proba"].values,
        "prediction": pred_df["prediction"].values,
    })
    app.state.last_result = out

    # 4) summary
    n_total = len(out)
    n_high = int((out["prediction"] == 1).sum())
    n_low = n_total - n_high
    p_high = round(100.0 * n_high / n_total, 1) if n_total else 0.0
    p_low = round(100.0 * n_low / n_total, 1) if n_total else 0.0

    # 5) JSON rows
    predictions = [
        PredictionRow(
            patient_id=_normalize_patient_id_value(out.loc[i, "patient_id"]),
            proba=float(out.loc[i, "proba"]),
            prediction=int(out.loc[i, "prediction"]),
        )
        for i in out.index
    ]

    return PredictResponse(
        summary=PredictionSummary(
            n_patients=n_total, n_high=n_high, n_low=n_low, p_high=p_high, p_low=p_low
        ),
        predictions=predictions
    )

# ====================== Routes ==============================================

@app.get("/health")
async def health():
    """
    Service health check.
    Returns: {"status": "ok", "version": "<APP_VERSION>"}.
    """
    return {"status": "ok", "version": APP_VERSION}

# ---------- HTML (web page) --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Return HTML page with CSV upload form.
    Template: app/templates/index.html
    """
    return templates.TemplateResponse("index.html", {"request": request, "summary": None, "version": APP_VERSION})

@app.post("/", response_class=HTMLResponse)
async def predict_page(request: Request, file: UploadFile = File(...)):
    """
    Accept CSV via web form (multipart/form-data),
    compute predictions and render results on the same page.
    """
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))

        payload = _make_payload_from_df(df)

        # Prepare items for the list on the page
        rows = [
            {"idx": pr.patient_id, "pred": pr.prediction, "proba": pr.proba}
            for pr in payload.predictions
        ]
        summary = payload.summary.model_dump()

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "summary": summary, "rows": rows, "version": APP_VERSION}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"File processing error: {e}", "summary": None, "version": APP_VERSION}
        )

# ---------- JSON API: by path to CSV ----------------------------------------
@app.post("/api/predict_path", response_model=PredictResponse)
async def api_predict_path(body: PredictPathIn):
    """
    Predicts from a CSV file on disk.

    Request JSON:
      { "path": "data/heart_test.csv" }  # absolute or project-root-relative path

    Response JSON (PredictResponse schema):
      {
        "summary": { ... },
        "predictions": [ { "patient_id": ..., "proba": ..., "prediction": ... }, ... ]
      }
    """
    csv_path = Path(body.path)
    if not csv_path.is_absolute():
        csv_path = (BASE_DIR / csv_path).resolve()
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail=f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        return _make_payload_from_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

# ---------- JSON API: file upload -------------------------------------------
@app.post("/api/predict_file", response_model=PredictResponse)
async def api_predict_file(file: UploadFile = File(...)):
    """
    Predicts from an uploaded file (multipart/form-data).

    Example (Swagger / Postman):
      form field "file": <your CSV>
    """
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))
        return _make_payload_from_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

# ---------- Download last result (CSV) --------------------------------------
@app.api_route("/download", methods=["GET", "POST"])
async def download():
    out_full = getattr(app.state, "last_result", None)
    if out_full is None or len(out_full) == 0:
        return HTMLResponse("No results to download. Upload a CSV first.", status_code=404)

    # if the input had an id column it is stored in last_result as patient_id.
    # Rename on the fly to "id" and select only two columns.
    out = out_full.rename(columns={"patient_id": "id"})[["id", "prediction"]]

    bio = BytesIO()
    out.to_csv(bio, index=False)
    bio.seek(0)
    return StreamingResponse(
        bio,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'}
    )
# ---- local dev run as `python app/main.py` ----------------------------------
def _choose_free_port(preferred: int = 8000, tries: int = 20) -> int:
    """
    Find a free port starting from preferred. Needed if 8000 is already taken.
    """
    for p in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", _choose_free_port(8000)))
    # enable auto-reload in dev: set RELOAD=1
    reload_flag = os.environ.get("RELOAD", "0") == "1"
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=reload_flag)
