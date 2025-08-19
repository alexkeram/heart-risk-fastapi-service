# файл: src/heart_job.py
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
from IPython.display import display, Markdown

from eda_analyzer import EDAAnalyzer
from heart_runner import HeartRiskRunner, RunCfg


class HeartRiskJob:
    """
    Оркестратор: EDA -> обучение (CV) -> выбор лучшей -> сохранение -> ОТЧЁТ в Jupyter.
    """

    def __init__(self, cfg: RunCfg | None = None, artifacts_dir: str | Path = "artifacts"):
        self.cfg = cfg or RunCfg()
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._runner: HeartRiskRunner | None = None
        self._out: dict | None = None
        self.best_artifacts: dict | None = None

        # для последующего доступа в ноутбуке
        self.report_all: pd.DataFrame | None = None
        self.report_best: pd.DataFrame | None = None
        self.trained_list: list[str] | None = None

    def run(self, train_df_raw: pd.DataFrame, target_col: str = "Heart Attack Risk (Binary)") -> dict:
        """Полный прогон без сохранения. Формирует отчёт в ноутбуке, возвращает out."""
        train_df = EDAAnalyzer(train_df_raw, target_col=target_col).process()
        self._runner = HeartRiskRunner(self.cfg)
        self._out = self._runner.run_all(train_df)
        self.best_artifacts = self._out["artifacts"]

        # построить фреймы отчёта и вывести
        self._build_report_frames(self._out)
        self.show_report()
        return self._out

    def save(self) -> dict:
        """Сохранение модели и метаданных."""
        if not self.best_artifacts:
            raise RuntimeError("Nothing to save: call run(...) first.")

        ba = self.best_artifacts
        meta = {
            "model_key": ba["model_key"],
            "features": list(ba["features"]),
            "cats": list(ba["cats"]),
            "threshold": float(ba["threshold"]),
        }

        if ba["model_key"] == "cat":
            model_path = self.artifacts_dir / "best_model.cbm"
            ba["model"].save_model(str(model_path))
        else:
            model_path = self.artifacts_dir / "best_model.joblib"
            joblib.dump(ba["model"], model_path)

        meta["model_path"] = str(model_path)
        with open(self.artifacts_dir / "best_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        display(Markdown("**Сохранено артефакты**:\n\n```json\n" +
                         json.dumps(meta, ensure_ascii=False, indent=2) + "\n```"))
        return meta

    def run_and_save(self, train_df_raw: pd.DataFrame, target_col: str = "Heart Attack Risk (Binary)") -> dict:
        """Запустить, вывести отчёт, затем сохранить модель. Возвращает meta."""
        self.run(train_df_raw, target_col=target_col)
        return self.save()

    # ---------- отчёт: сборка и показ ----------

    def _build_report_frames(self, out: dict) -> None:
        rows = []
        for r in out["results"]:
            m, ci = r["m"], r["ci"]
            N = m["TN"] + m["FP"] + m["FN"] + m["TP"]
            rows.append({
                "model": r["model_name"], "key": r["model_key"], "scenario": r["scenario"],
                "F2": m["F2"], "F1": m["F1"], "ROC_AUC": m["ROC_AUC"], "PR_AUC": m["PR_AUC"],
                "Precision": m["Precision"], "Recall": m["Recall"],
                "threshold": m["threshold"],
                "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
                "N": N,
                "FP/1000": 1000.0 * m["FP"] / max(1, N),
                "FN/1000": 1000.0 * m["FN"] / max(1, N),
                "F2_CI": f"[{ci['F2'][0]:.3f}, {ci['F2'][1]:.3f}]",
            })
        rep = pd.DataFrame(rows).sort_values(["model", "scenario"]).reset_index(drop=True)

        def _fmt(df: pd.DataFrame) -> pd.DataFrame:
            for c in ["F2","F1","ROC_AUC","PR_AUC","Precision","Recall","threshold","FP/1000","FN/1000"]:
                if c in df.columns:
                    df[c] = df[c].astype(float).round(4)
            return df

        self.report_all = _fmt(rep)

        best = out["best"]; bm = best["m"]; ba = out["artifacts"]
        self.report_best = _fmt(pd.DataFrame([{
            "model": best["model_name"], "key": best["model_key"], "scenario": best["scenario"],
            "F2": bm["F2"], "F1": bm["F1"], "ROC_AUC": bm["ROC_AUC"], "PR_AUC": bm["PR_AUC"],
            "Precision": bm["Precision"], "Recall": bm["Recall"],
            "threshold*": bm["threshold"],
            "#features": len(ba["features"]), "#cats": len(ba["cats"]),
        }]))

        combos = rep[["model", "scenario"]].drop_duplicates().sort_values(["model","scenario"])
        self.trained_list = [f"{m} | {s}" for m, s in combos.to_records(index=False)]

    def show_report(self) -> None:
        """Показывает отчёт в ноутбуке """
        if self.report_all is None or self.report_best is None:
            return
        display(Markdown("### РЕЗУЛЬТАТЫ (CV, OoF)"))
        display(self.report_all)
        if self.trained_list:
            display(Markdown("### Список обученных комбинаций"))
            display(Markdown(", ".join(self.trained_list)))
        display(Markdown("### Лучшая модель"))
        display(self.report_best)
