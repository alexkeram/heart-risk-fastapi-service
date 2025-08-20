# app/main.py
import os
from io import BytesIO
from pathlib import Path
import sys
import socket
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---- настройки путей ----
BASE_DIR = Path(__file__).resolve().parents[1]  # корень проекта
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---- импорт твоих модулей ----
from inference_utils import HeartRiskInference  # noqa

# ---- конфигурация ----
APP_VERSION = os.environ.get("APP_VERSION", os.urandom(4).hex())  # для кэш-бастинга статики
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", BASE_DIR / "artifacts")).resolve()

# ---- папки шаблонов/статик ----
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---- lifespan-хендлер ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inf = HeartRiskInference.from_dir(ARTIFACTS_DIR)
    app.state.last_result = None
    yield
    # сюда можно добавить освобождение ресурсов при остановке

# ---- создаём приложение ----
app = FastAPI(title="Heart Risk Inference", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---- маршруты ----
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "summary": None, "version": APP_VERSION})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))

        # 1) найдём колонку с ID пациента
        id_candidates = ["patient_id", "PatientID", "patientId", "Patient Id", "id", "ID", "Unnamed: 0", "Unnamed:0"]
        id_col = next((c for c in id_candidates if c in df.columns), None)
        patient_id = df[id_col] if id_col is not None else pd.Series(df.index, name="patient_id")

        # 2) инференс (внутри приведёт признаки через EDAAnalyzer)
        pred_df = app.state.inf.predict(df)  # proba + prediction

        # 3) итоговая таблица
        out = pd.DataFrame({
            "patient_id": patient_id.values,
            "proba": pred_df["proba"].values,
            "prediction": pred_df["prediction"].values,
        })

        # 4) сохраняем последний результат
        app.state.last_result = out

        # 5) сводка по классам
        n_total = len(out)
        n_high = int((out["prediction"] == 1).sum())
        n_low  = n_total - n_high
        p_high = round(100.0 * n_high / n_total, 1) if n_total else 0.0
        p_low  = round(100.0 * n_low  / n_total, 1) if n_total else 0.0

        # 6) элементы для списка
        rows = [
            {"idx": out.loc[i, "patient_id"], "pred": int(out.loc[i, "prediction"]), "proba": float(out.loc[i, "proba"])}
            for i in out.index
        ]
        summary = {"n_patients": n_total, "n_high": n_high, "n_low": n_low, "p_high": p_high, "p_low": p_low}

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "summary": summary, "rows": rows, "version": APP_VERSION}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Ошибка обработки файла: {e}", "summary": None, "version": APP_VERSION}
        )

@app.api_route("/download", methods=["GET", "POST"])
async def download(request: Request):
    out = getattr(app.state, "last_result", None)
    if out is None or len(out) == 0:
        return HTMLResponse("Нет результатов для скачивания. Сначала загрузите CSV.", status_code=404)
    bio = BytesIO()
    out.to_csv(bio, index=False)  # patient_id, proba, prediction
    bio.seek(0)
    return StreamingResponse(bio, media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'})

# ---- локальный dev-запуск как python app/main.py ----
def _choose_free_port(preferred=8000, tries=20):
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
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True)
