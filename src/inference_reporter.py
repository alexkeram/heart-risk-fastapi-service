# файл: src/inference_reporter.py
"""
SimpleInferenceSummary — минимальный отчёт по инференсу.

Что делает:
  • Вызывает inf.predict(df), читает inf.threshold.
  • Печатает только базовую сводку в формате:

    === ИНФЕРЕНС: сводка ===
    Объектов: <N>
    Порог   : <threshold>
    Класс '1': <n_pos> (<pct_pos>%)
    Класс '0': <n_neg> (<pct_neg>%)

Где:
  - inf — объект с методами/полями: predict(df) -> DataFrame[c("proba","prediction")], threshold: float
  - df  — DataFrame с сырыми данными для инференса

"""

from __future__ import annotations
from typing import Dict, Any
import pandas as pd


class SimpleInferenceSummary:
    """Печатает и (опционально) возвращает краткую сводку по инференсу."""

    @staticmethod
    def run(inf, df: pd.DataFrame, *, print_summary: bool = True) -> Dict[str, Any]:
        """
        Выполняет инференс и печатает краткую сводку.

        Args:
            inf: объект с .predict(df) и .threshold
            df : DataFrame для инференса
            print_summary: печатать ли сводку (True по умолчанию)

        Returns:
            dict с ключами: n, threshold, n_pos, n_neg, pos_rate
        """
        preds = inf.predict(df)  # ожидаются колонки: proba, prediction
        yhat = preds["prediction"].astype(int)
        thr = float(getattr(inf, "threshold", 0.5))

        n = int(len(preds))
        n_pos = int((yhat == 1).sum())
        n_neg = n - n_pos
        pos_rate = (n_pos / n) if n else 0.0
        neg_rate = 1.0 - pos_rate

        if print_summary:
            print("=== ИНФЕРЕНС: сводка ===")
            print(f"Объектов: {n}")
            print(f"Порог   : {thr:.3f}")
            print(f"Класс '1': {n_pos} ({pos_rate*100:.2f}%)")
            print(f"Класс '0': {n_neg} ({neg_rate*100:.2f}%)")

        return {
            "n": n,
            "threshold": thr,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
        }
