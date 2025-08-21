# файл: src/inference_utils.py
from __future__ import annotations

"""
HeartRiskInference — продакшен-обёртка для инференса риска инфаркта.

Назначение
----------
Класс загружает лучшую модель и метаданные из папки `artifacts/`, приводит сырые данные
к формату обучения (через EDAAnalyzer) и выдаёт:
  • вероятности положительного класса (predict_proba);
  • таблицу с вероятностью и бинарным решением (predict);
  • готовый к сдаче (ТЗ/соревнование) DataFrame ровно из двух колонок: `id`, `prediction`
    (predict_for_submission) — это минимальный формат, который обычно требуется.

Формат артефактов (лежит в <корень_репозитория>/artifacts)
----------------------------------------------------------
- `best_meta.json`:
    {
      "model_key": "cat | rf | hgb",
      "features": ["..."],          # список признаков (и их порядок) как при обучении
      "cats": ["..."],              # подмножество features — категориальные при обучении
      "threshold": 0.42,            # оптимальный порог (под F_beta) по OoF
      "model_path": "best_model.cbm | best_model.joblib | best_model.pkl"
    }
- файл модели:
    - CatBoost  -> best_model.cbm
    - sklearn   -> best_model.joblib (или .pkl) — пайплайн с OHE внутри

Где искать artifacts/
--------------------
По умолчанию строго `<корень_проекта>/artifacts`, где корень — это папка, содержащая `src/`
(вычисляется как parents[1] от этого файла). Можно передать альтернативный путь в
`HeartRiskInference.from_dir(artifacts_dir=...)`.

Быстрый старт (в ноутбуке/скрипте)
----------------------------------
>>> import pandas as pd
>>> from src.inference_utils import HeartRiskInference
>>> inf = HeartRiskInference.from_dir()                # загрузит артефакты из ./artifacts
>>> df_test = pd.read_csv("data/heart_test.csv")

# 1) Аналитический сценарий: вероятности + метки
>>> preds = inf.predict(df_test)                       # колонки: proba, prediction
>>> preds.head()

# 2) Готовый файл под ТЗ/соревнование: 2 колонки
>>> submit = inf.predict_for_submission(df_test, id_col="id")
>>> submit.to_csv("submission.csv", index=False)       # файл содержит колонки: id,prediction

Примечания
----------
• Для CatBoost передаются индексы категориальных признаков в Pool().
• Для sklearn (RF/HGB) загружается сохранённый пайплайн (preproc + модель) через joblib.
"""

from pathlib import Path
from typing import Optional, Iterable, Dict

import json
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eda_analyzer import EDAAnalyzer


class HeartRiskInference:
    """
    Класс для инференса (получения предсказаний) по сохранённым артефактам.

    Обычно используйте фабрику `from_dir(...)` — она найдёт и загрузит модель/метаавытоматы.

    Args:
        model: Загруженный обученный estimator/пайплайн (CatBoostClassifier или sklearn Pipeline).
        model_key: 'cat' | 'rf' | 'hgb'.
        features: Список имён признаков и их порядок на инференсе (как при обучении).
        cats_train: Имена категориальных признаков (подмножество features).
        threshold: Оптимальный порог (из OoF), применяемый для бинаризации вероятностей.
        artifacts_dir: Путь к папке с артефактами (для отчётности/метаданных).
    """

    def __init__(
        self,
        model,
        model_key: str,
        features: Iterable[str],
        cats_train: Iterable[str],
        threshold: float,
        artifacts_dir: Optional[Path] = None,
    ):
        self.model = model
        self.model_key = str(model_key)         # 'cat' | 'rf' | 'hgb'
        self.features = list(features)          # точный порядок признаков
        self.cats_train = list(cats_train)      # категориальные при обучении
        self.threshold = float(threshold)       # сохранённый порог
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None

    # ---------- Фабрика загрузки из artifacts ----------

    @classmethod
    def from_dir(cls, artifacts_dir: Optional[str | Path] = None) -> "HeartRiskInference":
        """
        Загружает модель и метаданные из папки артефактов.

        По умолчанию ищет в <repo_root>/artifacts (где <repo_root> — папка, содержащая `src/`).

        Args:
            artifacts_dir: альтернативный путь к каталогу артефактов.

        Returns:
            Экземпляр HeartRiskInference.

        Raises:
            FileNotFoundError: если нет best_meta.json или файла модели.
            KeyError: если в best_meta.json отсутствуют нужные поля.
            ValueError: если model_key неизвестен.
        """
        # <repo_root>/src/inference_utils.py -> parents[1] == <repo_root>
        repo_root = Path(__file__).resolve().parents[1]
        if artifacts_dir is None:
            artifacts_dir = (repo_root / "artifacts").resolve()
        else:
            artifacts_dir = Path(artifacts_dir).resolve()

        meta_path = artifacts_dir / "best_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Не найден файл метаданных {meta_path}. "
                f"Сначала обучите и сохраните модель (HeartRiskJob.run_and_save())."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for k in ("model_key", "features", "cats", "threshold", "model_path"):
            if k not in meta:
                raise KeyError(f"В {meta_path} отсутствует ключ '{k}'")

        model_key  = str(meta["model_key"])
        features   = list(meta["features"])
        cats_train = list(meta["cats"])
        threshold  = float(meta["threshold"])

        model_name = Path(meta["model_path"]).name  # в метаданных хранится ТОЛЬКО имя файла
        model_path = artifacts_dir / model_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        if model_key == "cat":
            mdl = CatBoostClassifier()
            mdl.load_model(str(model_path))
        elif model_key in ("rf", "hgb"):
            mdl = joblib.load(model_path)
        else:
            raise ValueError(f"Неизвестный model_key='{model_key}'. Ожидается: 'cat' | 'rf' | 'hgb'.")

        return cls(
            model=mdl,
            model_key=model_key,
            features=features,
            cats_train=cats_train,
            threshold=threshold,
            artifacts_dir=artifacts_dir,
        )

    # ---------- Подготовка данных ----------

    @staticmethod
    def _prepare_X(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        """
        Приводит сырые данные к формату обучения:
          1) прогон через EDAAnalyzer,
          2) удаление таргета, если он случайно есть,
          3) reindex к точному списку features (лишнее отбрасывается, недостающее заполняется 0).

        Returns:
            pd.DataFrame, готовый к подаче в модель.
        """
        proc = EDAAnalyzer(df, target_col=None).process()
        if "Heart Attack Risk (Binary)" in proc.columns:
            proc = proc.drop(columns=["Heart Attack Risk (Binary)"])
        return proc.reindex(columns=list(features), fill_value=0)

    # ---------- Инференс ----------

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Вероятности положительного класса (риск=1) с индексом исходного df.
        Пустой вход -> пустая Series (name='proba').
        """
        if df is None or len(df) == 0:
            return pd.Series(dtype=float, name="proba")

        Xte = self._prepare_X(df, self.features)

        if self.model_key == "cat":
            cat_idx = [Xte.columns.get_loc(c) for c in self.cats_train if c in Xte.columns]
            proba = self.model.predict_proba(Pool(Xte, cat_features=cat_idx))[:, 1]
        else:
            proba = self.model.predict_proba(Xte)[:, 1]

        return pd.Series(proba, index=df.index, name="proba")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Итоговая таблица для анализа:
          • proba — вероятность класса 1,
          • prediction — бинарное решение по сохранённому порогу self.threshold.
        """
        p = self.predict_proba(df)
        if p.empty:
            return pd.DataFrame(columns=["proba", "prediction"])
        pred = (p >= self.threshold).astype(int)
        return pd.DataFrame({"proba": p.values, "prediction": pred.values}, index=p.index)

    def predict_for_submission(self, df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
        """
        Две колонки — `id`, `prediction`.

        Args:
            df: входной DataFrame с колонкой идентификатора (по умолчанию 'id').
            id_col: имя колонки с идентификатором пациента.

        Returns:
            DataFrame с колонками ["id", "prediction"] (или [id_col, "prediction"]).
        """
        preds = self.predict(df)
        if preds.empty:
            return pd.DataFrame(columns=[id_col, "prediction"])

        if id_col not in df.columns:
            raise KeyError(f"Во входном df нет колонки '{id_col}'")

        return pd.DataFrame({
            id_col: df[id_col].values,
            "prediction": preds["prediction"].values,
        })

    # ---------- Утилиты ----------

    def meta(self) -> Dict[str, object]:
        """Короткая сводка метаданных без повторного чтения JSON."""
        return {
            "model_key": self.model_key,
            "n_features": len(self.features),
            "n_cats": len(self.cats_train),
            "threshold": self.threshold,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
        }

    def class_distribution(self, df: pd.DataFrame) -> Dict[int, int]:
        """Распределение предсказанных классов {0: count, 1: count} по результату predict()."""
        preds = self.predict(df)
        if preds.empty:
            return {0: 0, 1: 0}
        return preds["prediction"].value_counts().to_dict()
