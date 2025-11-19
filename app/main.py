# app/main.py
"""
Heart Risk Inference — FastAPI service
======================================

This module is the entry point of the web application. It connects the web
interface (HTML + CSV upload) and the inference library
(`src/inference_utils.py`), which reads model artifacts from the `artifacts/`
folder and makes predictions.
...
"""

import os
import socket
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import List, Union
from src.heart_job import HeartRiskJob
from src.heart_runner import RunCfg

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ---- import modules from src/ ------------------------------------------------
from src.inference_utils import HeartRiskInference

# ---- logging ------------------------------------------------
from app.logging_config import setup_logging
from app.metrics import Metrics
import logging
from time import perf_counter
from uuid import uuid4

setup_logging()
logger = logging.getLogger("app")

# ---- path settings -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
SRC_DIR = BASE_DIR / "src"

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
    app.state.metrics = Metrics(window_size=500)

    # log startup
    try:
        meta = getattr(app.state.inf, "meta", None)
        info = meta() if callable(meta) else {}
    except Exception:
        info = {}
    logger.info(
        "startup",
        extra={
            "event": "startup",
            "version": APP_VERSION,
            "artifacts_dir": str(ARTIFACTS_DIR),
            "model_key": info.get("model_key"),
            "model_version": info.get("model_version"),
        },
    )
    yield

# ---- create app --------------------------------------------------------------
app = FastAPI(title="Heart Risk Inference", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---- observability: startup log + request logging middleware -----------------
@app.on_event("startup")
async def _on_startup():
    logger.info("startup", extra={"event": "startup", "version": APP_VERSION})

@app.middleware("http")
async def request_logging(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid4())
    request.state.rid = rid
    start = perf_counter()
    try:
        response = await call_next(request)
        dt_ms = int((perf_counter() - start) * 1000)

        # metrics
        try:
            if request.url.path in {"/", "/api/predict_path", "/api/predict_file"} and response.status_code < 500:
                app.state.metrics.observe(dt_ms)
        except Exception:
            logger.exception("metrics_observe_failed", extra={"event": "error", "rid": rid})

        logger.info(
            "request",
            extra={
                "event": "request",
                "rid": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "dt_ms": dt_ms,
            },
        )
        return response
    except Exception:
        logger.exception(
            "unhandled",
            extra={"event": "error", "rid": rid, "path": request.url.path},
        )
        raise


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

_ID_CANDIDATES = ["patient_id", "PatientID",
                  "patientId", "Patient Id",
                  "id", "ID", "Unnamed: 0", "Unnamed:0"]

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
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request, "summary": None, "version": APP_VERSION},
    )

@app.post("/", response_class=HTMLResponse)
async def predict_page(request: Request, file: UploadFile = File(...)):
    """
    Accept CSV via web form (multipart/form-data),
    compute predictions and render results on the same page.
    """
    rid = getattr(request.state, "rid", None)
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

        # observability: aggregate log
        try:
            n = int(summary["n_patients"])
            n_high = int(summary["n_high"])
            p_high = float(summary["p_high"])
            logger.info(
                "predict_done",
                extra={"event": "predict_done", "rid": rid, "n": n, "n_high": n_high, "p_high": p_high},
            )
        except Exception:
            # не мешаем UI даже если в логировании что-то пойдёт не так
            logger.exception("predict_log_failed", extra={"event": "log_error", "rid": rid})

        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "summary": summary, "rows": rows, "version": APP_VERSION},
        )
    except Exception as e:
        logger.exception("predict_failed", extra={"event": "error", "rid": rid})
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request,
             "error": f"File processing error: {e}",
             "summary": None, "version": APP_VERSION},
        )

# ---------- JSON API: by path to CSV ----------------------------------------
@app.post("/api/predict_path", response_model=PredictResponse)
async def api_predict_path(body: PredictPathIn, request: Request):
    """
    Predicts from a CSV file on disk.
    ...
    """
    csv_path = Path(body.path)
    if not csv_path.is_absolute():
        csv_path = (BASE_DIR / csv_path).resolve()
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail=f"CSV file not found: {csv_path}")

    rid = getattr(request.state, "rid", None)
    try:
        df = pd.read_csv(csv_path)
        payload = _make_payload_from_df(df)
        s = payload.summary
        logger.info(
            "predict_done",
            extra={"event": "predict_done", "rid": rid, "n": s.n_patients, "n_high": s.n_high, "p_high": s.p_high},
        )
        return payload
    except Exception as e:
        logger.exception("predict_failed", extra={"event": "error", "rid": rid, "path": str(csv_path)})
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

# ---------- JSON API: file upload -------------------------------------------
@app.post("/api/predict_file", response_model=PredictResponse)
async def api_predict_file(request: Request, file: UploadFile = File(...)):
    """
    Predicts from an uploaded file (multipart/form-data).
    """
    rid = getattr(request.state, "rid", None)
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))
        payload = _make_payload_from_df(df)
        s = payload.summary
        logger.info(
            "predict_done",
            extra={"event": "predict_done", "rid": rid, "n": s.n_patients, "n_high": s.n_high, "p_high": s.p_high},
        )
        return payload
    except Exception as e:
        logger.exception("predict_failed", extra={"event": "error", "rid": rid})
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")


# ---------- HTML: training page ---------------------------------------------
@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse(
        request,
        "train.html",
        {"request": request, "ok": False, "error": None, "version": APP_VERSION},
    )

@app.post("/train", response_class=HTMLResponse)
async def train_upload(request: Request, file: UploadFile = File(...)):
    """
    Accept training CSV, run pipeline, save artifacts into ARTIFACTS_DIR.
    Uses your existing HeartRiskJob(RunCfg()) logic.
    """
    rid = getattr(request.state, "rid", None)
    try:
        raw = await file.read()
        df_train = pd.read_csv(BytesIO(raw))

        # start training and save artifacts
        job = HeartRiskJob(RunCfg(artifacts_dir=str(ARTIFACTS_DIR)))
        meta = job.run_and_save(df_train)  # meta requests dict with columns model_key / model_version, if exist

        rows = int(len(df_train))
        model_key = meta.get("model_key", "unknown")
        model_version = meta.get("model_version", "unknown")

        # event log
        logger.info(
            "train_done",
            extra={
                "event": "train_done",
                "rid": rid,
                "rows": rows,
                "model_key": model_key,
                "model_version": model_version,
            },
        )


        return templates.TemplateResponse(
            request,
            "train.html",
            {
                "request": request,
                "ok": True,
                "rows": rows,
                "model_key": model_key,
                "model_version": model_version,
                "error": None,
                "version": APP_VERSION,
            },
        )
    except Exception as e:
        logger.exception("train_failed", extra={"event": "error", "rid": rid})
        return templates.TemplateResponse(
            request,
            "train.html",
            {"request": request, "ok": False, "error": f"Training error: {e}", "version": APP_VERSION},
        )

@app.get("/metrics")
async def metrics():
    """
    Summary metrics for service performance.
    Format: JSON with requests_total, latency_p95_ms, window_size, n_samples.
    p95=None if observations < 20.
    """
    snap = getattr(app.state, "metrics", None)
    if snap is None:
        return {"requests_total": 0, "latency_p95_ms": None, "window_size": 500, "n_samples": 0}
    return app.state.metrics.snapshot()


# ---------- Download last result (CSV) --------------------------------------
@app.api_route("/download", methods=["GET", "POST"])
async def download():
    out_full = getattr(app.state, "last_result", None)
    if out_full is None or len(out_full) == 0:
        return HTMLResponse("No results to download. Upload a CSV first.", status_code=404)

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
    port = int(os.environ.get("PORT", _choose_free_port(8000)))
    # enable auto-reload in dev: set RELOAD=1
    reload_flag = os.environ.get("RELOAD", "0") == "1"
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=reload_flag)
