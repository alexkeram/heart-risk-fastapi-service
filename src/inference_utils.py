# файл: src/inference_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, Tuple

import json
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool

from eda_analyzer import EDAAnalyzer


class HeartRiskInference:
    """
    Класс для загрузки артефактов и инференса.

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
        self.model_key = str(model_key)
        self.features = list(features)
        self.cats_train = list(cats_train)
        self.threshold = float(threshold)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None

    # ---------- фабричный метод ----------

    @classmethod
    def from_dir(cls, artifacts_dir: str | Path) -> "HeartRiskInference":
        """
        Загрузка лучшей модели и метаданных из папки артефактов.
        """
        artifacts_dir = Path(artifacts_dir)
        meta_path = artifacts_dir / "best_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_key = meta["model_key"]
        features = meta["features"]
        cats_train = meta["cats"]
        threshold = float(meta["threshold"])
        model_path = Path(meta["model_path"])

        # подгружаем модель
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

    # ---------- подготовка данных ----------

    @staticmethod
    def _prepare_X(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        """
        EDA + приведение набора к нужным признакам (reindex).
        Таргет, если внезапно есть в тесте, выбрасывается.
        """
        proc = EDAAnalyzer(df, target_col=None).process()
        if "Heart Attack Risk (Binary)" in proc.columns:
            proc = proc.drop(columns=["Heart Attack Risk (Binary)"])
        return proc.reindex(columns=list(features), fill_value=0)

    # ---------- инференс ----------

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Вернуть вероятности класса 1 (риск).
        """
        Xte = self._prepare_X(df, self.features)
        if self.model_key == "cat":
            cat_idx = [Xte.columns.get_loc(c) for c in self.cats_train if c in Xte.columns]
            proba = self.model.predict_proba(Pool(Xte, cat_features=cat_idx))[:, 1]
        else:
            proba = self.model.predict_proba(Xte)[:, 1]
        return pd.Series(proba, index=df.index, name="proba")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вернуть DataFrame с колонками: proba, prediction.
        Порог берётся из meta["threshold"].
        """
        p = self.predict_proba(df)
        pred = (p >= self.threshold).astype(int)
        return pd.DataFrame({"proba": p.values, "prediction": pred.values}, index=df.index)

    # ---------- утилиты (необязательно) ----------

    def meta(self) -> dict:
        """Короткая сводка метаданных """
        return {
            "model_key": self.model_key,
            "n_features": len(self.features),
            "n_cats": len(self.cats_train),
            "threshold": self.threshold,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
        }
