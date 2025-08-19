# файл: src/heart_runner.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import shap
from catboost import CatBoostClassifier, Pool


# =============================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===============================

def _cat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(df[c].dtype) in ("category", "object")]

def _class_w(y: pd.Series) -> Dict[int, float]:
    p = y.mean()
    return {0: 0.5 / (1 - p + 1e-12), 1: 0.5 / (p + 1e-12)}

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

def _to_str_matrix(X):
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    X = np.asarray(X)
    return X.astype(str)

def _make_cat_preproc(cats: List[str]) -> ColumnTransformer:
    to_str = FunctionTransformer(_to_str_matrix, validate=False, feature_names_out="one-to-one")
    return ColumnTransformer(
        [("cat", Pipeline([("to_str", to_str), ("ohe", _make_ohe())]), cats)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

def _tune_thr_fbeta(y_true, p, beta: float, grid=np.linspace(0.01, 0.99, 99)) -> Tuple[float, float]:
    best_t, best = -1.0, -1.0
    for t in grid:
        s = fbeta_score(y_true, (p >= t).astype(int), beta=beta, zero_division=0)
        if s > best:
            best, best_t = s, t
    return float(best_t), float(best)

def _metrics(y_true, p, thr: float, beta: float) -> Dict[str, float]:
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
    return CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=600,
        l2_leaf_reg=3.0, random_seed=42,
        loss_function="Logloss", eval_metric="AUC", verbose=False,
        use_best_model=True, od_type="Iter", od_wait=80
    )

def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42)

def _make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, random_state=42)


# =============================== ОСНОВНОЙ КЛАСС ===============================

@dataclass
class RunCfg:
    target: str = "Heart Attack Risk (Binary)"
    leaks: Tuple[str, ...] = ('Troponin', 'CK-MB', 'Previous Heart Problems', 'Medication Use')
    seed: int = 42
    splits: int = 3
    beta: float = 2.0
    boot_n: int = 100
    shap_n: int = 200  # сэмпл для SHAP

class HeartRiskRunner:
    def __init__(self, cfg: RunCfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def prepare(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        prep = train_df.copy()
        # предполагается, что снаружи ты уже прогнал EDAAnalyzer; если нет — подключи здесь при желании
        y = pd.to_numeric(prep[self.cfg.target], errors="coerce").round().astype(int)
        X = prep.drop(columns=[self.cfg.target])
        return X, y

    def eval_cv(self, model_key: str, X: pd.DataFrame, y: pd.Series, cats: List[str]) -> Dict:
        skf = StratifiedKFold(n_splits=self.cfg.splits, shuffle=True, random_state=self.cfg.seed)
        oof = np.zeros(len(X))
        if model_key in ("rf", "hgb"):
            pre = _make_cat_preproc(cats)
        for tr, va in skf.split(X, y):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            if model_key == "cat":
                m = _make_cat()
                pos, neg = ytr.sum(), len(ytr) - ytr.sum()
                m.set_params(scale_pos_weight=(neg / (pos + 1e-12)))
                pool_tr = Pool(Xtr, label=ytr, cat_features=[X.columns.get_loc(c) for c in cats])
                pool_va = Pool(Xva, label=yva, cat_features=[X.columns.get_loc(c) for c in cats])
                m.fit(pool_tr, eval_set=pool_va, verbose=False)
                pr = m.predict_proba(pool_va)[:, 1]
            elif model_key == "rf":
                m = Pipeline([("pre", pre), ("clf", _make_rf())])
                m.fit(Xtr, ytr)
                pr = m.predict_proba(Xva)[:, 1]
            else:  # hgb
                m = Pipeline([("pre", pre), ("clf", _make_hgb())])
                sw = ytr.map(_class_w(ytr)).values
                m.fit(Xtr, ytr, clf__sample_weight=sw)
                pr = m.predict_proba(Xva)[:, 1]
            oof[va] = pr

        thr, _ = _tune_thr_fbeta(y, oof, self.cfg.beta)
        m_oof = _metrics(y, oof, thr, self.cfg.beta)
        ci = {k: _boot_ci(y, oof, thr, k, self.cfg.boot_n, self.cfg.seed, self.cfg.beta)
              for k in ("F2", "F1", "ROC_AUC", "PR_AUC")}
        return dict(oof=oof, thr=thr, m=m_oof, ci=ci)

    def run_all(self, train_df: pd.DataFrame) -> Dict:
        # 1) подготовка
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

        # 2) выбор лучшего по F2
        best = max(results, key=lambda z: z["m"]["F2"])
        # 3) дообучение лучшей на всём train + SHAP
        artifacts = self.fit_best(X_full, y, best)
        return {"results": results, "best": best, "artifacts": artifacts}

    def fit_best(self, X_full: pd.DataFrame, y: pd.Series, best: Dict) -> Dict:
        Xb = X_full[best["features"]].copy()
        cats = best["cats"]
        if best["model_key"] == "cat":
            mdl = _make_cat()
            pos, neg = y.sum(), len(y) - y.sum()
            mdl.set_params(scale_pos_weight=(neg / (pos + 1e-12)))
            pool = Pool(Xb, label=y, cat_features=[Xb.columns.get_loc(c) for c in cats])
            mdl.fit(pool, verbose=False)

            idx = self.rng.choice(len(Xb), size=min(self.cfg.shap_n, len(Xb)), replace=False)
            pool_shap = Pool(Xb.iloc[idx], label=y.iloc[idx], cat_features=[Xb.columns.get_loc(c) for c in cats])
            shap_vals = mdl.get_feature_importance(pool_shap, type="ShapValues")[:, :-1]
            mean_abs = np.abs(shap_vals).mean(axis=0)
            s = pd.Series(mean_abs, index=Xb.columns).sort_values(ascending=False)
            shap_top = s.head(20)
        else:
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
