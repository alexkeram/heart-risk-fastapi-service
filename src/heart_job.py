# file: src/heart_job.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import json
import joblib
import pandas as pd

# IPython may be absent in some environments
try:
    from IPython.display import display, Markdown  # type: ignore
    _HAVE_IPY = True
except Exception:  # pragma: no cover
    _HAVE_IPY = False
    def display(*_args, **_kwargs):  # noqa: D401
        """No-op if IPython is unavailable."""
        pass
    def Markdown(_text: str) -> str:  # noqa: D401
        """Proxy when IPython is unavailable."""
        return _text

from eda_analyzer import EDAAnalyzer
from heart_runner import HeartRiskRunner, RunCfg


class HeartRiskJob:
    """
    High-level experiment orchestrator.

    Responsible for:
      1) Data preprocessing (via `EDAAnalyzer`).
      2) Iterating/evaluating models with cross-validation and selecting the best one (via `HeartRiskRunner`).
      3) Generating reports (table of all combinations and summary of the best).
      4) Saving the best model artifacts to `<project_root>/artifacts`.

    `artifacts/` folder
    -------------------
    Contains:
      - `best_meta.json` — metadata (model type, features, cats, threshold, **file name** of the model),
      - `best_model.cbm` or `best_model.joblib` — the model file itself.
    `meta["model_path"]` stores only the **file name** without an absolute path.

    Instance attributes
    -------------------
    cfg : RunCfg
        Run configuration.
    artifacts_dir : Path
        Absolute path to the artifacts folder (default `<root>/artifacts`).
    _runner : Optional[HeartRiskRunner]
        Internal runner (available after `run()`).
    _out : Optional[dict]
        Raw results from `run_all()` (metrics, CI, best, artifacts).
    best_artifacts : Optional[dict]
        Artifacts of the best model (model, key, features, cats, threshold, shap_top).
    report_all : Optional[pd.DataFrame]
        Metrics table for all combinations.
    report_best : Optional[pd.DataFrame]
        Summary for the best model.
    trained_list : Optional[List[str]]
        List of trained combinations in the `"Model | Scenario"` format.

    Example
    -------
    .. code-block:: python

        from pathlib import Path
        import pandas as pd
        from src.heart_job import HeartRiskJob, RunCfg

        df_train = pd.read_csv(Path("data") / "heart_train.csv")
        job = HeartRiskJob(RunCfg())
        out = job.run(df_train)
        meta = job.save()

    """

    def __init__(
        self,
        cfg: Optional[RunCfg] = None,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self.cfg: RunCfg = cfg or RunCfg()

        # project root = folder above src/
        project_root = Path(__file__).resolve().parents[2]
        default_artifacts = project_root / "artifacts"

        if artifacts_dir is None:
            self.artifacts_dir = default_artifacts
        else:
            ad = Path(artifacts_dir)
            self.artifacts_dir = ad if ad.is_absolute() else (project_root / ad)

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # internal fields
        self._runner: Optional[HeartRiskRunner] = None
        self._out: Optional[Dict] = None
        self.best_artifacts: Optional[Dict] = None

        # reports
        self.report_all: Optional[pd.DataFrame] = None
        self.report_best: Optional[pd.DataFrame] = None
        self.trained_list: Optional[List[str]] = None

    # --------------------------------------------------------------------- #
    #                            Main methods                              #
    # --------------------------------------------------------------------- #

    def run(
        self,
        train_df_raw: pd.DataFrame,
        target_col: str = "Heart Attack Risk (Binary)",
    ) -> Dict:
        """
        Full run without saving:
        - preprocessing (`EDAAnalyzer`),
        - model iteration (`HeartRiskRunner.run_all`),
        - selecting the best,
        - preparing reports.

        Returns
        -------
        dict
            Raw results from `HeartRiskRunner.run_all`.
        """
        train_df = EDAAnalyzer(train_df_raw, target_col=target_col).process()
        self._runner = HeartRiskRunner(self.cfg)
        self._out = self._runner.run_all(train_df)
        self.best_artifacts = self._out["artifacts"]

        # reports
        self._build_report_frames(self._out)
        self.show_report()
        return self._out

    def save(self) -> Dict:
        """
        Save the best model and metadata to `<project_root>/artifacts`.

        Returns
        -------
        dict
            Saved metadata (with the model file name).
        """
        if not self.best_artifacts:
            raise RuntimeError("Nothing to save: call run(...) before save().")

        ba = self.best_artifacts
        meta = {
            "model_key": ba["model_key"],
            "features": list(ba["features"]),
            "cats": list(ba["cats"]),
            "threshold": float(ba["threshold"]),
        }

        # save model file
        if ba["model_key"] == "cat":
            model_path = self.artifacts_dir / "best_model.cbm"
            ba["model"].save_model(str(model_path))
        else:
            model_path = self.artifacts_dir / "best_model.joblib"
            joblib.dump(ba["model"], model_path)

        # meta contains only the model file name
        meta["model_path"] = model_path.name

        # save metadata alongside
        meta_path = self.artifacts_dir / "best_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self._echo_saved_meta(meta)
        return meta

    def run_and_save(
        self,
        train_df_raw: pd.DataFrame,
        target_col: str = "Heart Attack Risk (Binary)",
    ) -> Dict:
        """
        Run the full pipeline (`run`) and immediately save the model (`save`).

        Returns
        -------
        dict
            Saved metadata.
        """
        self.run(train_df_raw, target_col=target_col)
        return self.save()

    # --------------------------------------------------------------------- #
    #                          Report building                              #
    # --------------------------------------------------------------------- #

    def _build_report_frames(self, out: Dict) -> None:
        """
        Builds:
          - `report_all`: metrics of all combinations,
          - `report_best`: summary for the best,
          - `trained_list`: list of combinations `"Model | Scenario"`.
        """
        rows: List[Dict] = []
        for r in out["results"]:
            m, ci = r["m"], r["ci"]
            N = m["TN"] + m["FP"] + m["FN"] + m["TP"]
            rows.append(
                {
                    "model": r["model_name"],
                    "key": r["model_key"],
                    "scenario": r["scenario"],
                    "F2": m["F2"],
                    "F1": m["F1"],
                    "ROC_AUC": m["ROC_AUC"],
                    "PR_AUC": m["PR_AUC"],
                    "Precision": m["Precision"],
                    "Recall": m["Recall"],
                    "threshold": m["threshold"],
                    "TN": m["TN"],
                    "FP": m["FP"],
                    "FN": m["FN"],
                    "TP": m["TP"],
                    "N": N,
                    "FP/1000": 1000.0 * m["FP"] / max(1, N),
                    "FN/1000": 1000.0 * m["FN"] / max(1, N),
                    "F2_CI": f"[{ci['F2'][0]:.3f}, {ci['F2'][1]:.3f}]",
                }
            )

        rep = pd.DataFrame(rows).sort_values(["model", "scenario"]).reset_index(drop=True)

        def _fmt(df: pd.DataFrame) -> pd.DataFrame:
            for c in ["F2", "F1", "ROC_AUC", "PR_AUC", "Precision", "Recall", "threshold", "FP/1000", "FN/1000"]:
                if c in df.columns:
                    df[c] = df[c].astype(float).round(4)
            return df

        self.report_all = _fmt(rep)

        best = out["best"]
        bm = best["m"]
        ba = out["artifacts"]
        self.report_best = _fmt(
            pd.DataFrame(
                [
                    {
                        "model": best["model_name"],
                        "key": best["model_key"],
                        "scenario": best["scenario"],
                        "F2": bm["F2"],
                        "F1": bm["F1"],
                        "ROC_AUC": bm["ROC_AUC"],
                        "PR_AUC": bm["PR_AUC"],
                        "Precision": bm["Precision"],
                        "Recall": bm["Recall"],
                        "threshold*": bm["threshold"],
                        "#features": len(ba["features"]),
                        "#cats": len(ba["cats"]),
                    }
                ]
            )
        )

        combos = rep[["model", "scenario"]].drop_duplicates().sort_values(["model", "scenario"])
        self.trained_list = [f"{m} | {s}" for m, s in combos.to_records(index=False)]

    def show_report(self) -> None:
        """Display the report in Jupyter (if IPython is available). Silently skip otherwise."""
        if self.report_all is None or self.report_best is None:
            return
        if _HAVE_IPY:
            display(Markdown("### RESULTS (CV, OoF)"))
            display(self.report_all)
            display(Markdown("### Best model"))
            display(self.report_best)

    # --------------------------------------------------------------------- #
    #                               Helpers                                 #
    # --------------------------------------------------------------------- #

    def _echo_saved_meta(self, meta: Dict) -> None:
        """Nicely show saved metadata (via IPython) or print JSON as plain text."""
        text = "**Artifacts saved**:\n\n```json\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```"
        if _HAVE_IPY:
            display(Markdown(text))
        else:  # pragma: no cover
            print(text)
