# app/main.py
"""
Heart Risk Inference — FastAPI сервис
=====================================

Этот модуль — точка входа веб-приложения. Он связывает веб-интерфейс (HTML + загрузка CSV)
и библиотеку инференса (`src/inference_utils.py`), которая читает артефакты модели из
папки `artifacts/` и делает предсказания.

КАК ЭТО РАБОТАЕТ
--------------------------
1. При старте приложения (lifespan) мы один раз загружаем инференс-обёртку:
   `HeartRiskInference.from_dir(ARTIFACTS_DIR)`.
   ▸ Ожидается, что в `artifacts/` лежат `best_meta.json` и файл модели
     (`best_model.cbm` или `best_model.joblib`). Путь к артефактам можно переопределить
     переменной окружения `ARTIFACTS_DIR`, иначе берём `<корень проекта>/artifacts`.

2. Веб-страница (`GET /`) отдает форму для загрузки CSV.
   На `POST /` принимаем CSV, считаем предсказания и показываем список в браузере,
   а также сохраняем последний результат в `app.state.last_result` для кнопки «скачать».

3. JSON API:
   • `POST /api/predict_path` — принимает JSON `{ "path": "<путь_к_csv>" }` и возвращает
     JSON со сводкой и списком предсказаний.
   • `POST /api/predict_file` — принимает файл как multipart/form-data (как на веб-странице)
     и возвращает тот же JSON.
   • `GET/POST /download` — отдает CSV с последними предсказаниями (если до этого уже
     был выполнен расчёт через веб или API).

4. Статика и шаблоны:
   ▸ HTML-шаблон в `app/templates/index.html`
   ▸ Стили в `app/static/`
   ▸ Переменная `APP_VERSION` используется для cache-busting статики.

ТОЧКИ РАСШИРЕНИЯ
----------------
• Изменить/добавить логику препроцессинга — см. `src/eda_analyzer.py`.
• Изменить логику обучения/сохранения артефактов — см. `src/heart_job.py`, `src/heart_runner.py`.
• Логику инференса/загрузки артефактов — см. `src/inference_utils.py` (класс `HeartRiskInference`).

ПРИМЕРЫ ЗАПРОСОВ
----------------
• Здоровье сервиса:
  GET http://127.0.0.1:8000/health

• JSON по пути к CSV (PowerShell):
  Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/predict_path' `
    -ContentType 'application/json' -Body '{"path":"data/heart_test.csv"}' | ConvertTo-Json -Depth 5

• JSON с файлом (PowerShell):
  Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/predict_file' `
    -ContentType 'multipart/form-data' -InFile 'data/heart_test.csv' -OutFile 'resp.json'

• Swagger: http://127.0.0.1:8000/docs
"""

import os
from io import BytesIO
from pathlib import Path
import sys
import socket
from contextlib import asynccontextmanager
from typing import List, Union

import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---- настройки путей ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # корень проекта
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    # Делаем src/ импортируемым как модуль (EDAAnalyzer, HeartRiskInference и т.д.)
    sys.path.insert(0, str(SRC_DIR))

# ---- импорт модулей из src/ --------------------------------------------------
from inference_utils import HeartRiskInference  # noqa

# ---- конфигурация ------------------------------------------------------------
APP_VERSION = os.environ.get("APP_VERSION", os.urandom(4).hex())  # для cache-busting
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", BASE_DIR / "artifacts")).resolve()

# ---- папки шаблонов/статик ---------------------------------------------------
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ---- lifecycle: загружаем модель на старте -----------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Единожды при старте сервиса загружаем инференс-обёртку.
    Если файлов артефактов нет — упадём на старте с понятной ошибкой.
    """
    app.state.inf = HeartRiskInference.from_dir(ARTIFACTS_DIR)
    app.state.last_result = None  # сюда кладём последний вычисленный DataFrame (для /download)
    yield

# ---- создаём приложение ------------------------------------------------------
app = FastAPI(title="Heart Risk Inference", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ====================== Pydantic-схемы для JSON ==============================

class PredictionRow(BaseModel):
    """Одна строка предсказаний (используется в JSON-ответе API)."""
    patient_id: Union[int, str]
    proba: float
    prediction: int  # 0 — низкий риск, 1 — высокий риск

class PredictionSummary(BaseModel):
    """Сводка по батчу: размеры и доли классов."""
    n_patients: int
    n_high: int
    n_low: int
    p_high: float
    p_low: float

class PredictResponse(BaseModel):
    """
    Полный JSON-ответ API:
      • summary — агрегированная сводка,
      • predictions — список по каждому пациенту.
    """
    summary: PredictionSummary
    predictions: List[PredictionRow]

class PredictPathIn(BaseModel):
    """Вход для /api/predict_path — абсолютный или относительный путь к CSV."""
    path: str

# ====================== Общая логика предсказаний ============================

_ID_CANDIDATES = ["patient_id", "PatientID", "patientId", "Patient Id", "id", "ID", "Unnamed: 0", "Unnamed:0"]

def _normalize_patient_id_value(v):
    """
    Приводим идентификатор пациента к сериализуемому виду:
    • целые/строки оставляем как есть,
    • numpy-типы приводим,
    • прочее — в строку.
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
    Общая функция инференса для веб-формы и JSON-эндпоинтов:
      1) находим patient_id (или берём индекс),
      2) считаем предсказания через app.state.inf.predict(df),
      3) формируем DataFrame out (для /download),
      4) считаем сводку,
      5) собираем JSON-ответ (PredictResponse).
    """
    # 1) patient_id из входного CSV
    id_col = next((c for c in _ID_CANDIDATES if c in df.columns), None)
    patient_id = df[id_col] if id_col is not None else pd.Series(df.index, name="patient_id")

    # 2) инференс: вернёт DataFrame с колонками proba, prediction
    pred_df = app.state.inf.predict(df)

    # 3) итоговая таблица (для кнопки «скачать» и возможного аудита)
    out = pd.DataFrame({
        "patient_id": patient_id.values,
        "proba": pred_df["proba"].values,
        "prediction": pred_df["prediction"].values,
    })
    app.state.last_result = out

    # 4) сводка
    n_total = len(out)
    n_high = int((out["prediction"] == 1).sum())
    n_low = n_total - n_high
    p_high = round(100.0 * n_high / n_total, 1) if n_total else 0.0
    p_low = round(100.0 * n_low / n_total, 1) if n_total else 0.0

    # 5) JSON-строки
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

# ====================== Маршруты ============================================

@app.get("/health")
async def health():
    """
    Проверка сервиса.
    Возвращает: {"status": "ok", "version": "<APP_VERSION>"}.
    """
    return {"status": "ok", "version": APP_VERSION}

# ---------- HTML (веб-страница) ---------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Возвращает HTML-страницу с формой загрузки CSV.
    Шаблон: app/templates/index.html
    """
    return templates.TemplateResponse("index.html", {"request": request, "summary": None, "version": APP_VERSION})

@app.post("/", response_class=HTMLResponse)
async def predict_page(request: Request, file: UploadFile = File(...)):
    """
    Принимает CSV через веб-форму (multipart/form-data),
    считает предсказания и отрисовывает результаты на той же странице.
    """
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))

        payload = _make_payload_from_df(df)

        # Формируем элементы для списка на странице
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
            {"request": request, "error": f"Ошибка обработки файла: {e}", "summary": None, "version": APP_VERSION}
        )

# ---------- JSON API: по пути к CSV -----------------------------------------
@app.post("/api/predict_path", response_model=PredictResponse)
async def api_predict_path(body: PredictPathIn):
    """
    Считает предсказания по файлу на диске.

    Request JSON:
      { "path": "data/heart_test.csv" }  # путь абсолютный или от корня проекта

    Response JSON (схема PredictResponse):
      {
        "summary": { ... },
        "predictions": [ { "patient_id": ..., "proba": ..., "prediction": ... }, ... ]
      }
    """
    csv_path = Path(body.path)
    if not csv_path.is_absolute():
        csv_path = (BASE_DIR / csv_path).resolve()
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail=f"CSV файл не найден: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        return _make_payload_from_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {e}")

# ---------- JSON API: загрузка файла ----------------------------------------
@app.post("/api/predict_file", response_model=PredictResponse)
async def api_predict_file(file: UploadFile = File(...)):
    """
    Считает предсказания из переданного файла (multipart/form-data).

    Пример (Swagger / Postman):
      form field "file": <ваш CSV>
    """
    try:
        raw = await file.read()
        df = pd.read_csv(BytesIO(raw))
        return _make_payload_from_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {e}")

# ---------- Скачивание последнего результата (CSV) ---------------------------
@app.api_route("/download", methods=["GET", "POST"])
async def download():
    out_full = getattr(app.state, "last_result", None)
    if out_full is None or len(out_full) == 0:
        return HTMLResponse("Нет результатов для скачивания. Сначала загрузите CSV.", status_code=404)

    # если во входе была колонка id — она сохранится в last_result как patient_id.
    # Переименуем на лету в "id" и выберем только две колонки.
    out = out_full.rename(columns={"patient_id": "id"})[["id", "prediction"]]

    bio = BytesIO()
    out.to_csv(bio, index=False)
    bio.seek(0)
    return StreamingResponse(
        bio,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'}
    )
# ---- локальный dev-запуск как `python app/main.py` --------------------------
def _choose_free_port(preferred: int = 8000, tries: int = 20) -> int:
    """
    Ищем свободный порт, начиная с preferred. Нужен на случай, если 8000 уже занят.
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
    # включить авто-перезапуск в деве: set RELOAD=1
    reload_flag = os.environ.get("RELOAD", "0") == "1"
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=reload_flag)
