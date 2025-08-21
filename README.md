# Heart Risk Classification

Веб‑сервис и библиотека для скрининга риска сердечного приступа.

Проект включает:
- исследовательский ноутбук (EDA + обучение, выбор лучшей модели),
- библиотеку (`src/`) с классами предобработки/обучения/инференса,
- веб‑приложение на FastAPI (`app/`) для загрузки CSV и получения предсказаний,
- артефакты лучшей модели в `artifacts/` (для немедленного запуска приложения).

---

## ✨ Возможности

- Автоочистка и приведение типов входного CSV (`EDAAnalyzer`);
- Обучение нескольких моделей с кросс‑валидацией и выбор лучшей по F‑β (`HeartRiskRunner` / `HeartRiskJob`);
- Сохранение и загрузка артефактов модели (мета + файл модели);
- Локальный инференс в коде/ноутбуке (`HeartRiskInference`);
- Веб‑интерфейс для массовых предсказаний (загрузка/скачивание CSV) **и** JSON‑API.
- Выдача по API в формате JSON

**Формат выдачи:** CSV с колонками  
`patient_id`, `proba`, `prediction` (0 — низкий риск, 1 — высокий).

---

## 📁 Структура репозитория

```
.
├─ app/                      # FastAPI приложение
│  ├─ static/                # стили
│  ├─ templates/             # Jinja2 (index.html)
│  └─ main.py                # точка входа приложения
├─ artifacts/                # артефакты лучшей модели (best_meta.json, best_model.*)
├─ data/                     # (опционально) исходные данные
├─ notebooks/
│  └─ heart_risk_notebook.ipynb
├─ src/                      # библиотека (ООП)
│  ├─ eda_analyzer.py
│  ├─ heart_runner.py
│  ├─ heart_job.py
│  ├─ inference_utils.py
│  └─ quick_view.py
├─ requirements.txt
└─ README.md / README.txt
```

---

## 🔧 Установка

```bash
# 1) создать и активировать виртуальное окружение
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) установить зависимости
pip install -r requirements.txt
```

> Минимум: `pandas`, `numpy`, `scikit-learn`, `catboost`, `shap`,
> `fastapi`, `uvicorn`, `joblib`, `nest-asyncio` (для запуска из Jupyter).

---

## 🧪 Обучение и сохранение артефактов (в ноутбуке)

```python
from pathlib import Path
import pandas as pd
from src.heart_job import HeartRiskJob, RunCfg

# данные
train_df = pd.read_csv(Path("data") / "heart_train.csv")

# запуск эксперимента и сохранение лучшей модели в ./artifacts
job = HeartRiskJob(RunCfg(), artifacts_dir=Path("artifacts"))
meta = job.run_and_save(train_df)
meta
```
После выполнения в `artifacts/` появятся:
- `best_meta.json` — мета‑информация,
- `best_model.cbm` **или** `best_model.joblib` — файл модели.

---

## 🔍 Инференс в коде / ноутбуке

```python
import pandas as pd
from src.inference_utils import HeartRiskInference

inf = HeartRiskInference.from_dir()  # всегда берёт из ./artifacts
df_test = pd.read_csv("data/heart_test.csv")

preds = inf.predict(df_test)         # DataFrame: proba, prediction
print(preds.head())

# распределение классов
dist = preds["prediction"].value_counts().to_dict()
print("Низкий риск (0):", dist.get(0, 0))
print("Высокий риск (1):", dist.get(1, 0))
```

---

## 🌐 Веб‑приложение (FastAPI)

### Запуск из консоли

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Откройте: http://127.0.0.1:8000  
- Загрузите CSV  
- Посмотрите список предсказаний
- Скачайте результаты (CSV)

### Запуск прямо из Jupyter Notebook
```python
import nest_asyncio, uvicorn
nest_asyncio.apply()
uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
```

---

## 🔌 JSON‑API (эндпоинты)

Базовый URL: `http://127.0.0.1:8000`

### 1) Check: `GET /health`
Проверка доступности сервиса.
```bash
curl http://127.0.0.1:8000/health
```
Ответ:
```json
{"status": "ok", "version": "<hash>"}
```

### 2) Предсказание по пути к CSV: `POST /api/predict`
Request body:
```json
{ "path": "data/heart_test.csv" }
```

Пример (bash/PowerShell):
```bash
curl -X POST http://127.0.0.1:8000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{ \"path\": \"data/heart_test.csv\" }"
```
Ответ (пример):
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

### 3) Скачать последние результаты (CSV): `GET /download`
Отдаёт файл `predictions.csv` (колонки: `patient_id,proba,prediction`) — формируется после последнего инференса (через веб‑форму или `/api/predict`).

---

## 🗂️ Папка `artifacts/`

Ожидаются два файла:
- `best_meta.json`
- `best_model.cbm` **или** `best_model.joblib`

---

## 🧩 Основные классы (src/)

- **`EDAAnalyzer`** — очистка/приведение типов/импутации + консистентность train/test.
- **`HeartRiskRunner`** — CV для CatBoost / HistGB / RF, подбор порога F‑β, бутстреп‑ДИ, SHAP‑топ.
- **`HeartRiskJob`** — оркестратор: `run()` (обучение + отчёт), `save()` (артефакты), `run_and_save()`.
- **`HeartRiskInference`** — загрузка артефактов, подготовка признаков и предсказания (`predict_proba`, `predict`, `class_distribution`).
- **`QuickView`** — быстрые графики/сводки (опционально).

---

## ✅ Что сдаётся

- Ноутбук с исследованием и обучением;
- Код приложения и библиотеки (`src/`, `app/`);
- Инструкция по запуску (этот README);
- Воспроизводимость: `requirements.txt`.
