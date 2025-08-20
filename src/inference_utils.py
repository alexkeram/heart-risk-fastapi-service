# файл: src/inference_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable, Dict

import json
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eda_analyzer import EDAAnalyzer


class HeartRiskInference:
    """
    Продакшен-обёртка для инференса риска инфаркта.

    Что делает:
      - Загружает лучшую модель и метаданные из <корень проекта>/artifacts.
      - Готовит входные данные (EDAAnalyzer + выравнивание набора признаков).
      - Возвращает вероятности и бинарные предсказания по сохранённому порогу.
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
        self.features = list(features)          # порядок признаков на инференсе
        self.cats_train = list(cats_train)      # имена категориальных, как при обучении
        self.threshold = float(threshold)       # оптимальный порог (из OoF)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None

    # ---------- Загрузка из artifacts ----------

    @classmethod
    def from_dir(cls, _: str | Path = None) -> "HeartRiskInference":
        """
        Загружает модель и метаданные
        """
        # <repo_root>/src/inference_utils.py -> parents[1] == <repo_root>
        repo_root = Path(__file__).resolve().parents[1]
        artifacts_dir = (repo_root / "artifacts").resolve()

        meta_path = artifacts_dir / "best_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Не найден файл метаданных {meta_path}. "
                f"Сначала обучите и сохраните модель (HeartRiskJob.save())."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_key  = str(meta["model_key"])
        features   = list(meta["features"])
        cats_train = list(meta["cats"])
        threshold  = float(meta["threshold"])

        # В мета хранится ТОЛЬКО имя файла — склеиваем с artifacts_dir
        model_name = Path(meta["model_path"]).name
        model_path = artifacts_dir / model_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        # Грузим модель
        if model_key == "cat":
            mdl = CatBoostClassifier()
            mdl.load_model(str(model_path))
        else:
            mdl = joblib.load(model_path)

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
        1) прогон через EDAAnalyzer (те же преобразования, что на train),
        2) удаление таргета, если он случайно есть,
        3) reindex к точному списку признаков обучения (недостающие заполняем нулями).
        """
        proc = EDAAnalyzer(df, target_col=None).process()
        if "Heart Attack Risk (Binary)" in proc.columns:
            proc = proc.drop(columns=["Heart Attack Risk (Binary)"])
        return proc.reindex(columns=list(features), fill_value=0)

    # ---------- Инференс ----------

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Вероятности класса 1 (риск) с индексом входного df."""
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
        DataFrame:
          - proba: вероятность класса 1,
          - prediction: бинарное решение по сохранённому порогу.
        """
        p = self.predict_proba(df)
        if p.empty:
            return pd.DataFrame(columns=["proba", "prediction"])
        pred = (p >= self.threshold).astype(int)
        return pd.DataFrame({"proba": p.values, "prediction": pred.values}, index=p.index)

    # ---------- Утилиты ----------

    def meta(self) -> Dict[str, object]:
        """Короткая сводка метаданных (без повторного чтения JSON)."""
        return {
            "model_key": self.model_key,
            "n_features": len(self.features),
            "n_cats": len(self.cats_train),
            "threshold": self.threshold,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
        }

    def class_distribution(self, df: pd.DataFrame) -> Dict[int, int]:
        """Сколько объектов в каждом классе (0/1) по результату predict()."""
        preds = self.predict(df)
        if preds.empty:
            return {0: 0, 1: 0}
        return preds["prediction"].value_counts().to_dict()
