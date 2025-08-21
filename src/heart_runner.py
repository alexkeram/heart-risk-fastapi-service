# файл: src/heart_runner.py
from __future__ import annotations
"""
HeartRiskRunner — ядро обучения и отбора лучшей модели

Что это:
    Модуль подбирает лучшую модель бинарной классификации риска инфаркта
    по кросс-валидации и считает метрики/доверительные интервалы, а затем
    дообучает выбранную модель на всём трейне и вычисляет SHAP-вклады.

Сценарии:
    • "no_leak"   — обучаем без «утечек» (колонки из cfg.leaks исключены);
    • "with_leak" — обучаем со всеми признаками.

Модели:
    • CatBoostClassifier     (ключ "cat") — нативные категориальные;
    • HistGradientBoosting   (ключ "hgb") — через OHE;
    • RandomForestClassifier (ключ "rf")  — через OHE.

Как пользоваться (пример):
    >>> import pandas as pd
    >>> from heart_runner import HeartRiskRunner, RunCfg
    >>> df = pd.read_csv("data/heart_train.csv")
    >>> cfg = RunCfg(target="Heart Attack Risk (Binary)", splits=3, beta=2.0)
    >>> runner = HeartRiskRunner(cfg)
    >>> out = runner.run_all(df)
    >>> out.keys()
    dict_keys(['results', 'best', 'artifacts'])
    >>> out["best"]["model_key"], out["best"]["scenario"], out["best"]["m"]["F2"]
    ('hgb', 'no_leak', 0.7)
    >>> # обученная лучшая модель и мета:
    >>> art = out["artifacts"]
    >>> art["model"], art["features"][:5], art["threshold"]

Структура результата:
    out = {
      "results": [                     # список по всем (сценарий × модель)
         {
           "model_key": "cat|hgb|rf",
           "model_name": "CatBoost|HistGB|RandomForest",
           "scenario": "no_leak|with_leak",
           "features": [...],          # какие фичи использовались в этом запуске
           "cats": [...],              # какие фичи трактовались как категориальные
           "oof": np.ndarray,          # OoF-вероятности (длины len(y))
           "thr": float,               # оптимальный порог по F_beta (на OoF)
           "m": {                      # метрики на OoF при thr
             "F1", "F2", "ROC_AUC", "PR_AUC",
             "Precision", "Recall",
             "TN", "FP", "FN", "TP",
             "threshold"
           },
           "ci": {                     # бутстреп-ДИ (2.5%, 97.5%)
             "F1": (lo, hi),
             "F2": (lo, hi),
             "ROC_AUC": (lo, hi),
             "PR_AUC": (lo, hi)
           }
         },
         ...
      ],
      "best": {... как один из results, лучший по F2 ...},
      "artifacts": {                   # обучено на всём трейне
        "model": sklearn/CB модель или Pipeline,
        "model_key": "cat|hgb|rf",
        "features": [...],             # финальный набор колонок
        "cats": [...],                 # финальные категориальные
        "threshold": float,            # оптимальный порог с OoF лучшей
        "shap_top": pd.Series          # топ-20 по |SHAP|
      }
    }

Зависимости:
    numpy, pandas, scikit-learn, catboost, shap

Безопасность/воспроизводимость:
    • Фиксированный seed из RunCfg.seed делает разбиения и обучение повторяемыми.
    • Порог подбирается по OoF-предсказаниям (устойчивее простого hold-out).
"""

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
    """Возвращает список категориальных столбцов (dtype == 'category' или 'object')."""
    return [c for c in df.columns if str(df[c].dtype) in ("category", "object")]

def _class_w(y: pd.Series) -> Dict[int, float]:
    """Считает веса классов ~ 0.5 / P(class), чтобы сбалансировать потери по классам."""
    p = y.mean()
    return {0: 0.5 / (1 - p + 1e-12), 1: 0.5 / (p + 1e-12)}

def _make_ohe():
    """
    Создаёт OneHotEncoder с обратной совместимостью по параметру sparse_output.
    (в старых версиях sklearn используется `sparse`, в новых - `sparse_output`).
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

def _to_str_matrix(X):
    """Приводит матрицу/фрейм к строковому типу (нужно для корректного OHE)."""
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    X = np.asarray(X)
    return X.astype(str)

def _make_cat_preproc(cats: List[str]) -> ColumnTransformer:
    """
    Пайплайн препроцессинга для деревьев без нативной поддержки категорий:
      • для категориальных: преобразовать к строкам и OHE;
      • для остальных: пропускать без изменений.
    """
    to_str = FunctionTransformer(_to_str_matrix, validate=False, feature_names_out="one-to-one")
    return ColumnTransformer(
        [("cat", Pipeline([("to_str", to_str), ("ohe", _make_ohe())]), cats)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

def _tune_thr_fbeta(y_true, p, beta: float, grid=np.linspace(0.01, 0.99, 99)) -> Tuple[float, float]:
    """Подбор порога по максимизации F_beta на сетке значений. Возвращает (best_threshold, best_score)."""
    best_t, best = -1.0, -1.0
    for t in grid:
        s = fbeta_score(y_true, (p >= t).astype(int), beta=beta, zero_division=0)
        if s > best:
            best, best_t = s, t
    return float(best_t), float(best)

def _metrics(y_true, p, thr: float, beta: float) -> Dict[str, float]:
    """Считает набор метрик при фиксированном пороге thr."""
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
    Бутстреп-ДИ для метрик: оценивает 2.5 и 97.5 перцентили по n перезапускам с возвращением.
    Возвращает (lo, hi).
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
    Приводит SHAP-значения к 2D-матрице [n_samples, n_features].
    У CatBoost/деревьев форма может быть [n, 1/2, d] или [n, d, 1/2] — выбираем нужную ось.
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
    """Базовая конфигурация CatBoostClassifier под задачу логлосса + ранняя остановка."""
    return CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=600,
        l2_leaf_reg=3.0, random_seed=42,
        loss_function="Logloss", eval_metric="AUC", verbose=False,
        use_best_model=True, od_type="Iter", od_wait=80
    )

def _make_rf() -> RandomForestClassifier:
    """RandomForest с балансировкой классов, достаточно большим числом деревьев и фиксированным seed."""
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42)

def _make_hgb() -> HistGradientBoostingClassifier:
    """HistGradientBoosting — быстрый бустинг по гистограммам; параметры — умеренно консервативные."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, random_state=42)


# =============================== ОСНОВНОЙ КЛАСС ===============================

@dataclass
class RunCfg:
    """
    Конфигурация прогона.

    Параметры:
        target:   имя целевой колонки (0/1).
        leaks:    признаки-«утечки», которые исключаются в сценарии 'no_leak'.
        seed:     фиксированный seed для воспроизводимости.
        splits:   число фолдов в StratifiedKFold.
        beta:     бета в F-beta (beta>1 — сильнее штрафуем FN, повышаем Recall).
        boot_n:   число бутстреп-перезапусков для доверительных интервалов метрик.
        shap_n:   размер случайного поднабора для расчёта SHAP (ускоряет расчёт).
    """
    target: str = "Heart Attack Risk (Binary)"
    leaks: Tuple[str, ...] = ('Troponin', 'CK-MB', 'Previous Heart Problems', 'Medication Use')
    seed: int = 42
    splits: int = 3
    beta: float = 2.0
    boot_n: int = 100
    shap_n: int = 200  # сэмпл для SHAP

class HeartRiskRunner:
    """
    Запускатель экспериментов: готовит данные, гоняет модели по сценариям, выбирает лучшую и
    дообучает её на всех данных. Возвращает словари с OoF-прогнозами, метриками, ДИ и артефактами.
    """

    def __init__(self, cfg: RunCfg):
        """Сохраняет конфигурацию и инициализирует генератор случайности."""
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def prepare(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Готовит X и y из входного фрейма:
            • y — целевая колонка cfg.target, приведённая к int {0,1};
            • X — остальные признаки.
        """
        prep = train_df.copy()
        y = pd.to_numeric(prep[self.cfg.target], errors="coerce").round().astype(int)
        X = prep.drop(columns=[self.cfg.target])
        return X, y

    def eval_cv(self, model_key: str, X: pd.DataFrame, y: pd.Series, cats: List[str]) -> Dict:
        """
        Кросс-валидация для заданной модели:
            • считает OoF-прогнозы,
            • подбирает порог по F_beta,
            • считает метрики на OoF и бутстреп-ДИ.

        Аргументы:
            model_key: 'cat' | 'rf' | 'hgb'
            X, y:      матрица признаков и целевая
            cats:      имена категориальных колонок

        Возвращает dict: {oof, thr, m, ci}.
        """
        skf = StratifiedKFold(n_splits=self.cfg.splits, shuffle=True, random_state=self.cfg.seed)
        oof = np.zeros(len(X))
        if model_key in ("rf", "hgb"):
            pre = _make_cat_preproc(cats)

        for tr, va in skf.split(X, y):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]

            if model_key == "cat":
                # CatBoost: нативные категориальные + балансировка класса
                m = _make_cat()
                pos, neg = ytr.sum(), len(ytr) - ytr.sum()
                m.set_params(scale_pos_weight=(neg / (pos + 1e-12)))
                pool_tr = Pool(Xtr, label=ytr, cat_features=[X.columns.get_loc(c) for c in cats])
                pool_va = Pool(Xva, label=yva, cat_features=[X.columns.get_loc(c) for c in cats])
                m.fit(pool_tr, eval_set=pool_va, verbose=False)
                pr = m.predict_proba(pool_va)[:, 1]

            elif model_key == "rf":
                # RandomForest: OHE для категорий + вероятности положительного класса
                m = Pipeline([("pre", pre), ("clf", _make_rf())])
                m.fit(Xtr, ytr)
                pr = m.predict_proba(Xva)[:, 1]

            else:  # hgb
                # HistGB: OHE + sample_weight на основе class weights
                m = Pipeline([("pre", pre), ("clf", _make_hgb())])
                sw = ytr.map(_class_w(ytr)).values
                m.fit(Xtr, ytr, clf__sample_weight=sw)
                pr = m.predict_proba(Xva)[:, 1]

            oof[va] = pr

        # Подбор порога по F_beta и расчёт метрик + ДИ
        thr, _ = _tune_thr_fbeta(y, oof, self.cfg.beta)
        m_oof = _metrics(y, oof, thr, self.cfg.beta)
        ci = {k: _boot_ci(y, oof, thr, k, self.cfg.boot_n, self.cfg.seed, self.cfg.beta)
              for k in ("F2", "F1", "ROC_AUC", "PR_AUC")}
        return dict(oof=oof, thr=thr, m=m_oof, ci=ci)

    def run_all(self, train_df: pd.DataFrame) -> Dict:
        """
        Полный прогон по двум сценариям (без утечек / с утечками) и трём моделям (CatBoost, HistGB, RF).

        Возвращает словарь:
            {
              "results": [ ... по всем комбинациям ... ],
              "best":    { ... лучшая по F2 ... },
              "artifacts": { ... обученная лучшая модель, порог, shap_top ... }
            }
        """
        # 1) подготовка данных
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

        # 2) выбор лучшей по F2
        best = max(results, key=lambda z: z["m"]["F2"])

        # 3) дообучение лучшей на полном train + SHAP
        artifacts = self.fit_best(X_full, y, best)
        return {"results": results, "best": best, "artifacts": artifacts}

    def fit_best(self, X_full: pd.DataFrame, y: pd.Series, best: Dict) -> Dict:
        """
        Дообучает лучшую комбинацию на всех данных и считает топ-вкладов по SHAP.

        Возвращает артефакты:
            {
              "model": обученный estimator/пайтлайн,
              "model_key": ключ модели,
              "features": использованные признаки,
              "cats": список категориальных,
              "threshold": оптимальный порог по OoF,
              "shap_top": pd.Series топ признаков по |SHAP|
            }
        """
        Xb = X_full[best["features"]].copy()
        cats = best["cats"]

        if best["model_key"] == "cat":
            # CatBoost с балансировкой и нативными cat_features
            mdl = _make_cat()
            pos, neg = y.sum(), len(y) - y.sum()
            mdl.set_params(scale_pos_weight=(neg / (pos + 1e-12)))
            pool = Pool(Xb, label=y, cat_features=[Xb.columns.get_loc(c) for c in cats])
            mdl.fit(pool, verbose=False)

            # SHAP для CatBoost: забираем матрицу без последнего столбца base value
            idx = self.rng.choice(len(Xb), size=min(self.cfg.shap_n, len(Xb)), replace=False)
            pool_shap = Pool(Xb.iloc[idx], label=y.iloc[idx], cat_features=[Xb.columns.get_loc(c) for c in cats])
            shap_vals = mdl.get_feature_importance(pool_shap, type="ShapValues")[:, :-1]
            mean_abs = np.abs(shap_vals).mean(axis=0)
            s = pd.Series(mean_abs, index=Xb.columns).sort_values(ascending=False)
            shap_top = s.head(20)

        else:
            # Для RF/HGB используем препроцессор (OHE) + TreeExplainer
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
