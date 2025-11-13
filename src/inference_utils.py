# file: src/inference_utils.py
from __future__ import annotations

"""
HeartRiskInference — production wrapper for heart attack risk inference.

Purpose
-------
The class loads the best model and metadata from the `artifacts/` folder,
brings raw data to the training format (via EDAAnalyzer) and provides:
  • positive class probabilities (`predict_proba`);
  • a table with probability and binary decision (`predict`);
  • a ready-to-submit DataFrame with exactly two columns: `id`, `prediction`
    (`predict_for_submission`) — the minimal format usually required.

Artifact format (stored in <repository_root>/artifacts)
------------------------------------------------------
- `best_meta.json`:
    {
      "model_key": "cat | rf | hgb",
      "features": ["..."],          # feature list (and order) as during training
      "cats": ["..."],              # subset of features that were categorical
      "threshold": 0.42,            # optimal threshold (F_beta) on OoF
      "model_path": "best_model.cbm | best_model.joblib | best_model.pkl"
    }
- model file:
    - CatBoost  -> best_model.cbm
    - sklearn   -> best_model.joblib (or .pkl) — pipeline with OHE inside

Where to find artifacts/
-----------------------
By default `<project_root>/artifacts`, where project root is the folder containing `src/`
(computed as parents[1] of this file). You can pass an alternative path to
`HeartRiskInference.from_dir(artifacts_dir=...)`.

Quick start (notebook/script)
-----------------------------
>>> import pandas as pd
>>> from src.inference_utils import HeartRiskInference
>>> inf = HeartRiskInference.from_dir()                # load artifacts from ./artifacts
>>> df_test = pd.read_csv("data/heart_test.csv")

# 1) Analytical scenario: probabilities + labels
>>> preds = inf.predict(df_test)                       # columns: proba, prediction
>>> preds.head()

# 2) Ready-to-submit file: 2 columns
>>> submit = inf.predict_for_submission(df_test, id_col="id")
>>> submit.to_csv("submission.csv", index=False)       # file has columns: id,prediction

Notes
-----
• For CatBoost, categorical feature indices are passed to Pool().
• For sklearn (RF/HGB) the saved pipeline (preproc + model) is loaded via joblib.
"""

from pathlib import Path
from typing import Optional, Iterable, Dict

import json
import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.eda_analyzer import EDAAnalyzer


class HeartRiskInference:
    """
    Inference class for generating predictions from saved artifacts.

    Usually use the factory `from_dir(...)` — it will find and load the model and metadata.

    Args:
        model: Loaded trained estimator/pipeline (CatBoostClassifier or sklearn Pipeline).
        model_key: 'cat' | 'rf' | 'hgb'.
        features: List of feature names and their order for inference (as during training).
        cats_train: Names of categorical features (subset of features).
        threshold: Optimal threshold (from OoF) used to binarize probabilities.
        artifacts_dir: Path to the artifacts directory (for reporting/metadata).
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
        self.features = list(features)          # exact feature order
        self.cats_train = list(cats_train)      # categorical features during training
        self.threshold = float(threshold)       # saved threshold
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None

    # ---------- Factory: load from artifacts ----------

    @classmethod
    def from_dir(cls, artifacts_dir: Optional[str | Path] = None) -> "HeartRiskInference":
        """
        Load model and metadata from the artifacts folder.

        By default searches `<repo_root>/artifacts` (where `<repo_root>` is the folder containing `src/`).

        Args:
            artifacts_dir: alternative path to the artifacts directory.

        Returns:
            Instance of HeartRiskInference.

        Raises:
            FileNotFoundError: if best_meta.json or model file is missing.
            KeyError: if required fields are absent in best_meta.json.
            ValueError: if model_key is unknown.
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
                f"Metadata file not found: {meta_path}. "
                f"Train and save the model first (HeartRiskJob.run_and_save())."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        for k in ("model_key", "features", "cats", "threshold", "model_path"):
            if k not in meta:
                raise KeyError(f"Key '{k}' is missing in {meta_path}")

        model_key  = str(meta["model_key"])
        features   = list(meta["features"])
        cats_train = list(meta["cats"])
        threshold  = float(meta["threshold"])

        model_name = Path(meta["model_path"]).name  # meta stores only the file name
        model_path = artifacts_dir / model_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_key == "cat":
            mdl = CatBoostClassifier()
            mdl.load_model(str(model_path))
        elif model_key in ("rf", "hgb"):
            mdl = joblib.load(model_path)
        else:
            raise ValueError(f"Unknown model_key='{model_key}'. Expected: 'cat' | 'rf' | 'hgb'.")

        return cls(
            model=mdl,
            model_key=model_key,
            features=features,
            cats_train=cats_train,
            threshold=threshold,
            artifacts_dir=artifacts_dir,
        )

    # ---------- Data preparation ----------

    @staticmethod
    def _prepare_X(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        """
        Bring raw data to training format:
          1) run through EDAAnalyzer,
          2) drop target column if it appears,
          3) reindex to the exact feature list (extra dropped, missing filled with 0).

        Returns:
            pd.DataFrame ready for the model.
        """
        proc = EDAAnalyzer(df, target_col=None).process()
        if "Heart Attack Risk (Binary)" in proc.columns:
            proc = proc.drop(columns=["Heart Attack Risk (Binary)"])
        return proc.reindex(columns=list(features), fill_value=0)

    # ---------- Inference ----------

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        Probabilities of the positive class (risk=1) with the index of input df.
        Empty input -> empty Series (name='proba').
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
        Final table for analysis:
          • proba — probability of class 1,
          • prediction — binary decision using saved threshold self.threshold.
        """
        p = self.predict_proba(df)
        if p.empty:
            return pd.DataFrame(columns=["proba", "prediction"])
        pred = (p >= self.threshold).astype(int)
        return pd.DataFrame({"proba": p.values, "prediction": pred.values}, index=p.index)

    def predict_for_submission(self, df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
        """
        Two columns — `id`, `prediction`.

        Args:
            df: input DataFrame with identifier column (default 'id').
            id_col: name of the patient identifier column.

        Returns:
            DataFrame with columns ["id", "prediction"] (or [id_col, "prediction"]).
        """
        preds = self.predict(df)
        if preds.empty:
            return pd.DataFrame(columns=[id_col, "prediction"])

        if id_col not in df.columns:
            raise KeyError(f"Column '{id_col}' not found in input dataframe")

        return pd.DataFrame({
            id_col: df[id_col].values,
            "prediction": preds["prediction"].values,
        })

    # ---------- Utilities ----------

    def meta(self) -> Dict[str, object]:
        """Short metadata summary without reading JSON again."""
        return {
            "model_key": self.model_key,
            "n_features": len(self.features),
            "n_cats": len(self.cats_train),
            "threshold": self.threshold,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
        }

    def class_distribution(self, df: pd.DataFrame) -> Dict[int, int]:
        """Distribution of predicted classes {0: count, 1: count} from predict()."""
        preds = self.predict(df)
        if preds.empty:
            return {0: 0, 1: 0}
        return preds["prediction"].value_counts().to_dict()
