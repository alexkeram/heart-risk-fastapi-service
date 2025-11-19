# file: src/heart_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

"""
HeartRiskRunner — core for training and selecting the best model.

Overview:
    The module searches for the best binary classifier of heart attack risk
    via cross-validation, computes metrics and confidence intervals, then
    trains the selected model on the full train set and calculates SHAP values.

Scenarios:
    • "no_leak"   — train without leakage features (columns from cfg.leaks excluded);
    • "with_leak" — train with all features.

Models:
    • CatBoostClassifier     (key "cat") — native categorical support;
    • HistGradientBoosting   (key "hgb") — via OHE;
    • RandomForestClassifier (key "rf")  — via OHE.

Reproducibility:
    • Fixed seed from RunCfg.seed makes splits and training repeatable.
    • Threshold is tuned on OoF predictions (more stable than hold-out).
"""

# =============================== HELPER FUNCTIONS ===============================

def _cat_cols(df: pd.DataFrame) -> List[str]:
    """Return a list of categorical columns (dtype == 'category' or 'object')."""
    return [c for c in df.columns if str(df[c].dtype) in ("category", "object")]

def _class_w(y: pd.Series) -> Dict[int, float]:
    """Compute class weights ~ 0.5 / P(class) to balance class losses."""
    p = y.mean()
    return {0: 0.5 / (1 - p + 1e-12), 1: 0.5 / (p + 1e-12)}

def _make_ohe():
    """
    Create OneHotEncoder with backward compatibility for the sparse_output parameter
    (older sklearn uses `sparse`, newer uses `sparse_output`).
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

def _to_str_matrix(X):
    """Convert matrix/frame to string type (needed for proper OHE)."""
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    X = np.asarray(X)
    return X.astype(str)

def _make_cat_preproc(cats: List[str]) -> ColumnTransformer:
    """
    Preprocessing pipeline for tree models without native categorical support:
      • for categorical: convert to strings and OHE;
      • for others: pass through unchanged.
    """
    to_str = FunctionTransformer(_to_str_matrix, validate=False, feature_names_out="one-to-one")
    return ColumnTransformer(
        [("cat", Pipeline([("to_str", to_str), ("ohe", _make_ohe())]), cats)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

def _tune_thr_fbeta(y_true, p,
                    beta: float,
                    grid=np.linspace(0.01,
                                     0.99,
                                     99)) -> Tuple[float, float]:
    """Select threshold maximizing F_beta on a value grid. Returns (best_threshold, best_score)."""
    best_t, best = -1.0, -1.0
    for t in grid:
        s = fbeta_score(y_true, (p >= t).astype(int), beta=beta, zero_division=0)
        if s > best:
            best, best_t = s, t
    return float(best_t), float(best)

def _metrics(y_true, p, thr: float, beta: float) -> Dict[str, float]:
    """Compute metrics at a fixed threshold thr."""
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(
        F1=float(f1_score(y_true, pred, zero_division=0)),
        F2=float(fbeta_score(y_true, pred, beta=beta, zero_division=0)),
        ROC_AUC=float(roc_auc_score(y_true, p)),
        PR_AUC=float(average_precision_score(y_true, p)),
        Precision=float(precision_score(y_true, pred, zero_division=0)),
        Recall=float(recall_score(y_true, pred, zero_division=0)),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        threshold=float(thr),
    )

def _boot_ci(y_true, p, thr, metric: str, n: int, seed: int, beta: float) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for metrics: evaluates 2.5 and 97.5 percentiles
    over n resamples with replacement. Returns (lo, hi).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        yt, pp = y_true[idx], p[idx]
        if metric == "ROC_AUC":
            vals.append(roc_auc_score(yt, pp))
        elif metric == "PR_AUC":
            vals.append(average_precision_score(yt, pp))
        elif metric == "F1":
            vals.append(f1_score(yt, (pp >= thr).astype(int), zero_division=0))
        elif metric == "F2":
            vals.append(fbeta_score(yt, (pp >= thr).astype(int), beta=beta, zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def _shap_to_2d(shap_vals: np.ndarray | list) -> np.ndarray:
    """
    Convert SHAP values to a 2D matrix [n_samples, n_features].
    For CatBoost/trees shape may be [n, 1/2, d] or [n, d, 1/2] — choose appropriate axis.
    """
    if isinstance(shap_vals, list):
        arr = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
    else:
        arr = np.asarray(shap_vals)
    if arr.ndim == 3:
        n, a1, a2 = arr.shape
        if a1 in (1, 2) and a2 not in (1, 2):
            arr = arr[:, 1 if a1 > 1 else 0, :]
        elif a2 in (1, 2) and a1 not in (1, 2):
            arr = arr[:, :, 1 if a2 > 1 else 0]
        else:
            sl = [slice(None)] * 3
            sl[2] = 1
            arr = arr[tuple(sl)]
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
    return arr

def _make_cat() -> CatBoostClassifier:
    """Basic CatBoostClassifier config for logloss with early stopping."""
    return CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=600,
        l2_leaf_reg=3.0, random_seed=42,
        loss_function="Logloss", eval_metric="AUC", verbose=False,
        use_best_model=True, od_type="Iter", od_wait=80
    )

def _make_rf() -> RandomForestClassifier:
    """RandomForest with class balancing, many trees, and fixed seed."""
    return RandomForestClassifier(n_estimators=500,
                                  class_weight="balanced",
                                  n_jobs=-1,
                                  random_state=42)

def _make_hgb() -> HistGradientBoostingClassifier:
    """HistGradientBoosting — fast histogram-based boosting;
     parameters are moderately conservative."""
    return HistGradientBoostingClassifier(learning_rate=0.05,
                                          max_iter=400,
                                          random_state=42)


# =============================== MAIN CLASS ===============================

@dataclass
class RunCfg:
    """
    Run configuration.

    Parameters:
        target:   name of target column (0/1).
        leaks:    leakage features excluded in the 'no_leak' scenario.
        seed:     fixed seed for reproducibility.
        splits:   number of folds in StratifiedKFold.
        beta:     beta in F-beta (beta>1 penalizes FN more, increases Recall).
        boot_n:   number of bootstrap resamples for metric CIs.
        shap_n:   size of random subset for SHAP calculation (speeds up computation).
    """
    target: str = "Heart Attack Risk (Binary)"
    leaks: Tuple[str, ...] = ('Troponin', 'CK-MB',
                              'Previous Heart Problems',
                              'Medication Use')
    seed: int = 42
    splits: int = 3
    beta: float = 2.0
    boot_n: int = 100
    shap_n: int = 200  # sample size for SHAP

class HeartRiskRunner:
    """
    Experiment launcher: prepares data, runs models across scenarios, selects the best,
    and trains it on all data. Returns dictionaries with OoF predictions, metrics,
    CIs and artifacts.
    """

    def __init__(self, cfg: RunCfg):
        """Store configuration and initialize random generator."""
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def prepare(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare X and y from the input frame:
            • y — target column cfg.target converted to int {0,1};
            • X — remaining features.
        """
        prep = train_df.copy()
        y = pd.to_numeric(prep[self.cfg.target], errors="coerce").round().astype(int)
        X = prep.drop(columns=[self.cfg.target])
        return X, y

    def eval_cv(self, model_key: str, X: pd.DataFrame, y: pd.Series, cats: List[str]) -> Dict:
        """
        Cross-validation for a given model:
            • compute OoF predictions,
            • tune threshold by F_beta,
            • compute OoF metrics and bootstrap CIs.

        Args:
            model_key: 'cat' | 'rf' | 'hgb'
            X, y:      feature matrix and target
            cats:      names of categorical columns

        Returns dict: {oof, thr, m, ci}.
        """
        skf = StratifiedKFold(n_splits=self.cfg.splits, shuffle=True, random_state=self.cfg.seed)
        oof = np.zeros(len(X))
        if model_key in ("rf", "hgb"):
            pre = _make_cat_preproc(cats)

        for tr, va in skf.split(X, y):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]

            if model_key == "cat":
                # CatBoost: native categorical handling + class balancing
                m = _make_cat()
                pos, neg = ytr.sum(), len(ytr) - ytr.sum()
                m.set_params(scale_pos_weight=(neg / (pos + 1e-12)))
                pool_tr = Pool(Xtr, label=ytr, cat_features=[X.columns.get_loc(c) for c in cats])
                pool_va = Pool(Xva, label=yva, cat_features=[X.columns.get_loc(c) for c in cats])
                m.fit(pool_tr, eval_set=pool_va, verbose=False)
                pr = m.predict_proba(pool_va)[:, 1]

            elif model_key == "rf":
                # RandomForest: OHE for categories + positive class probabilities
                m = Pipeline([("pre", pre), ("clf", _make_rf())])
                m.fit(Xtr, ytr)
                pr = m.predict_proba(Xva)[:, 1]

            else:  # hgb
                # HistGB: OHE + sample_weight based on class weights
                m = Pipeline([("pre", pre), ("clf", _make_hgb())])
                sw = ytr.map(_class_w(ytr)).values
                m.fit(Xtr, ytr, clf__sample_weight=sw)
                pr = m.predict_proba(Xva)[:, 1]

            oof[va] = pr

        # Threshold tuning by F_beta and calculation of metrics + CIs
        thr, _ = _tune_thr_fbeta(y, oof, self.cfg.beta)
        m_oof = _metrics(y, oof, thr, self.cfg.beta)
        ci = {k: _boot_ci(y, oof, thr, k, self.cfg.boot_n, self.cfg.seed, self.cfg.beta)
              for k in ("F2", "F1", "ROC_AUC", "PR_AUC")}
        return dict(oof=oof, thr=thr, m=m_oof, ci=ci)

    def run_all(self, train_df: pd.DataFrame) -> Dict:
        """
        Full run across two scenarios (without/with leaks) and three models (CatBoost, HistGB, RF).

        Returns dict:
            {
              "results": [ ... over all combinations ... ],
              "best":    { ... best by F2 ... },
              "artifacts": { ... trained best model, threshold, shap_top ... }
            }
        """
        # 1) data preparation
        X_full, y = self.prepare(train_df)
        scenarios = {
            "no_leak": [c for c in X_full.columns if c not in self.cfg.leaks],
            "with_leak": list(X_full.columns)
        }
        models = {"cat": "CatBoost", "hgb": "HistGB", "rf": "RandomForest"}

        results = []
        for scen, cols in scenarios.items():
            X = X_full[cols].copy()
            cats = _cat_cols(X)
            for mk, mn in models.items():
                r = self.eval_cv(mk, X, y, cats)
                results.append({"model_key": mk, "model_name": mn, "scenario": scen,
                                "features": cols, "cats": cats, **r})

        # 2) select best by F2
        best = max(results, key=lambda z: z["m"]["F2"])

        # 3) train best on full train + SHAP
        artifacts = self.fit_best(X_full, y, best)
        return {"results": results, "best": best, "artifacts": artifacts}

    def fit_best(self, X_full: pd.DataFrame, y: pd.Series, best: Dict) -> Dict:
        """
        Train the best combination on all data and compute top SHAP contributions.

        Returns artifacts:
            {
              "model": trained estimator/pipeline,
              "model_key": model key,
              "features": used features,
              "cats": list of categorical features,
              "threshold": optimal threshold from OoF,
              "shap_top": pd.Series of top features by |SHAP|
            }
        """
        Xb = X_full[best["features"]].copy()
        cats = best["cats"]

        if best["model_key"] == "cat":
            # CatBoost with class balancing and native cat_features
            mdl = _make_cat()
            pos, neg = y.sum(), len(y) - y.sum()
            mdl.set_params(scale_pos_weight=(neg / (pos + 1e-12)))
            pool = Pool(Xb, label=y, cat_features=[Xb.columns.get_loc(c) for c in cats])
            mdl.fit(pool, verbose=False)

            # SHAP for CatBoost: take matrix without last base value column
            idx = self.rng.choice(len(Xb), size=min(self.cfg.shap_n, len(Xb)), replace=False)
            pool_shap = Pool(Xb.iloc[idx],
                             label=y.iloc[idx],
                             cat_features=[Xb.columns.get_loc(c) for c in cats])
            shap_vals = mdl.get_feature_importance(pool_shap, type="ShapValues")[:, :-1]
            mean_abs = np.abs(shap_vals).mean(axis=0)
            s = pd.Series(mean_abs, index=Xb.columns).sort_values(ascending=False)
            shap_top = s.head(20)

        else:
            # For RF/HGB use preprocessor (OHE) + TreeExplainer
            pre = _make_cat_preproc(cats)
            if best["model_key"] == "rf":
                base = _make_rf()
            else:
                base = _make_hgb()
            mdl = Pipeline([("pre", pre), ("clf", base)])

            if best["model_key"] == "hgb":
                sw = y.map(_class_w(y)).values
                mdl.fit(Xb, y, clf__sample_weight=sw)
            else:
                mdl.fit(Xb, y)

            idx = self.rng.choice(len(Xb), size=min(self.cfg.shap_n, len(Xb)), replace=False)
            Xt = mdl.named_steps["pre"].transform(Xb.iloc[idx])
            expl = shap.TreeExplainer(mdl.named_steps["clf"])
            shap_vals = expl.shap_values(Xt, check_additivity=False)
            arr = _shap_to_2d(shap_vals)
            feat_names = mdl.named_steps["pre"].get_feature_names_out()
            if arr.shape[1] != len(feat_names):
                raise ValueError(f"SHAP size mismatch: {arr.shape[1]} vs {len(feat_names)}")
            mean_abs = np.abs(arr).mean(axis=0)
            s = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
            shap_top = s.head(20)

        return {
            "model": mdl,
            "model_key": best["model_key"],
            "features": best["features"],
            "cats": cats,
            "threshold": best["m"]["threshold"],
            "shap_top": shap_top,
        }
