# file: src/inference_reporter.py
"""
SimpleInferenceSummary — minimal inference report.

What it does:
  • Calls inf.predict(df), reads inf.threshold.
  • Prints only a basic summary in the format:

    === INFERENCE: summary ===
    Objects: <N>
    Threshold: <threshold>
    Class '1': <n_pos> (<pct_pos>%)
    Class '0': <n_neg> (<pct_neg>%)

Where:
  - inf — object with methods/fields: predict(df) -> DataFrame[c("proba",
  "prediction")], threshold: float
  - df  — DataFrame with raw data for inference

"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class SimpleInferenceSummary:
    """Print and optionally return a brief inference summary."""

    @staticmethod
    def run(inf, df: pd.DataFrame, *, print_summary: bool = True) -> Dict[str, Any]:
        """
        Run inference and print a short summary.

        Args:
            inf: object with .predict(df) and .threshold
            df : DataFrame for inference
            print_summary: whether to print the summary (True by default)

        Returns:
            dict with keys: n, threshold, n_pos, n_neg, pos_rate
        """
        preds = inf.predict(df)  # expected columns: proba, prediction
        yhat = preds["prediction"].astype(int)
        thr = float(getattr(inf, "threshold", 0.5))

        n = int(len(preds))
        n_pos = int((yhat == 1).sum())
        n_neg = n - n_pos
        pos_rate = (n_pos / n) if n else 0.0
        neg_rate = 1.0 - pos_rate

        if print_summary:
            print("=== INFERENCE: summary ===")
            print(f"Objects : {n}")
            print(f"Threshold: {thr:.3f}")
            print(f"Class '1': {n_pos} ({pos_rate*100:.2f}%)")
            print(f"Class '0': {n_neg} ({neg_rate*100:.2f}%)")

        return {
            "n": n,
            "threshold": thr,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
        }
