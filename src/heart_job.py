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
    Верхнеуровневый оркестратор эксперимента:
    1) Прогоняет EDA (через EDAAnalyzer),
    2) Запускает кросс-валидацию моделей и выбирает лучшую (HeartRiskRunner),
    3) Формирует отчёт в Jupyter (таблицы метрик, лучшая модель, список комбинаций),
    4) При необходимости сохраняет обученную модель и метаданные в папку artifacts (корень проекта).
    """

    def __init__(self, cfg: RunCfg | None = None, artifacts_dir: str | Path | None = None):
        self.cfg = cfg or RunCfg()

        # вычисляем корень проекта (папка выше src)
        project_root = Path(__file__).resolve().parents[2]
        default_artifacts = project_root / "artifacts"

        if artifacts_dir is None:
            self.artifacts_dir = default_artifacts
        else:
            ad = Path(artifacts_dir)
            self.artifacts_dir = ad if ad.is_absolute() else (project_root / ad)

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # служебные поля
        self._runner: HeartRiskRunner | None = None
        self._out: dict | None = None
        self.best_artifacts: dict | None = None

        # отчёт (для последующего доступа в ноутбуке)
        self.report_all: pd.DataFrame | None = None
        self.report_best: pd.DataFrame | None = None
        self.trained_list: list[str] | None = None

    def run(self, train_df_raw: pd.DataFrame, target_col: str = "Heart Attack Risk (Binary)") -> dict:
        """
        Полный прогон без сохранения.
        """
        train_df = EDAAnalyzer(train_df_raw, target_col=target_col).process()
        self._runner = HeartRiskRunner(self.cfg)
        self._out = self._runner.run_all(train_df)
        self.best_artifacts = self._out["artifacts"]

        # отчёты
        self._build_report_frames(self._out)
        self.show_report()
        return self._out

    def save(self) -> dict:
        """
        Сохраняет лучшую модель и метаданные в <корень проекта>/artifacts.
        В метаданных путь к модели — только имя файла.
        """
        if not self.best_artifacts:
            raise RuntimeError("Нечего сохранять: вызовите run(...) перед save().")

        ba = self.best_artifacts
        meta = {
            "model_key": ba["model_key"],
            "features": list(ba["features"]),
            "cats": list(ba["cats"]),
            "threshold": float(ba["threshold"]),
        }

        # сохраняем модель
        if ba["model_key"] == "cat":
            model_path = self.artifacts_dir / "best_model.cbm"
            ba["model"].save_model(str(model_path))
        else:
            model_path = self.artifacts_dir / "best_model.joblib"
            joblib.dump(ba["model"], model_path)

        # в meta только имя файла
        meta["model_path"] = model_path.name

        # сохраняем метаданные
        meta_path = self.artifacts_dir / "best_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        display(Markdown(
            "**Сохранены артефакты**:\n\n```json\n" +
            json.dumps(meta, ensure_ascii=False, indent=2) +
            "\n```"
        ))
        return meta

    def run_and_save(self, train_df_raw: pd.DataFrame, target_col: str = "Heart Attack Risk (Binary)") -> dict:
        """
        Запускает полный прогон и сразу сохраняет модель.
        """
        self.run(train_df_raw, target_col=target_col)
        return self.save()

    # ---------- служебные методы для отчёта ----------

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
        if self.report_all is None or self.report_best is None:
            return
        display(Markdown("### РЕЗУЛЬТАТЫ (CV, OoF)"))
        display(self.report_all)
        display(Markdown("### Лучшая модель"))
        display(self.report_best)
