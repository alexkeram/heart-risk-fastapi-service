# файл: src/heart_job.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List

import json
import joblib
import pandas as pd

# IPython может отсутствовать в некоторых окружениях
try:
    from IPython.display import display, Markdown  # type: ignore
    _HAVE_IPY = True
except Exception:  # pragma: no cover
    _HAVE_IPY = False
    def display(*_args, **_kwargs):  # noqa: D401
        """no-op, если нет IPython."""
        pass
    def Markdown(_text: str) -> str:  # noqa: D401
        """Прокси, если нет IPython."""
        return _text

from eda_analyzer import EDAAnalyzer
from heart_runner import HeartRiskRunner, RunCfg


class HeartRiskJob:
    """
    Верхнеуровневый оркестратор эксперимента.

    Отвечает за:
      1) Предобработку данных (через `EDAAnalyzer`).
      2) Перебор/оценку моделей с кросс-валидацией и выбор лучшей (через `HeartRiskRunner`).
      3) Формирование отчётов (таблица всех комбинаций и сводка по лучшей).
      4) Сохранение артефактов лучшей модели в `<корень проекта>/artifacts`.

    Папка `artifacts/`
    -------------------
    В неё пишутся:
      - `best_meta.json` — метаинформация (тип модели, признаки, cats, threshold, **имя** файла модели),
      - `best_model.cbm` или `best_model.joblib` - сам файл модели.
    В `meta["model_path"]` сохраняется только **имя** файла, без абсолютного пути.

    Атрибуты экземпляра
    -------------------
    cfg : RunCfg
        Конфигурация прогона.
    artifacts_dir : Path
        Абсолютный путь к папке артефактов (по умолчанию `<root>/artifacts`).
    _runner : Optional[HeartRiskRunner]
        Внутренний раннер (после `run()` доступен).
    _out : Optional[dict]
        Сырые результаты `run_all()` (metrics, ci, best, artifacts).
    best_artifacts : Optional[dict]
        Артефакты лучшей модели (модель, ключ, признаки, cats, threshold, shap_top).
    report_all : Optional[pd.DataFrame]
        Таблица метрик по всем комбинациям.
    report_best : Optional[pd.DataFrame]
        Сводка по лучшей модели.
    trained_list : Optional[List[str]]
        Список обученных комбинаций в формате `"Model | Scenario"`.

    Пример
    ------
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

        # корень проекта = папка выше src/
        project_root = Path(__file__).resolve().parents[2]
        default_artifacts = project_root / "artifacts"

        if artifacts_dir is None:
            self.artifacts_dir = default_artifacts
        else:
            ad = Path(artifacts_dir)
            self.artifacts_dir = ad if ad.is_absolute() else (project_root / ad)

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # служебные поля
        self._runner: Optional[HeartRiskRunner] = None
        self._out: Optional[Dict] = None
        self.best_artifacts: Optional[Dict] = None

        # отчёты
        self.report_all: Optional[pd.DataFrame] = None
        self.report_best: Optional[pd.DataFrame] = None
        self.trained_list: Optional[List[str]] = None

    # --------------------------------------------------------------------- #
    #                            Основные методы                            #
    # --------------------------------------------------------------------- #

    def run(
        self,
        train_df_raw: pd.DataFrame,
        target_col: str = "Heart Attack Risk (Binary)",
    ) -> Dict:
        """
        Полный прогон без сохранения:
        - предобработка (`EDAAnalyzer`),
        - перебор моделей (`HeartRiskRunner.run_all`),
        - выбор лучшей,
        - подготовка отчётов.

        Returns
        -------
        dict
            Сырые результаты из `HeartRiskRunner.run_all`.
        """
        train_df = EDAAnalyzer(train_df_raw, target_col=target_col).process()
        self._runner = HeartRiskRunner(self.cfg)
        self._out = self._runner.run_all(train_df)
        self.best_artifacts = self._out["artifacts"]

        # отчёты
        self._build_report_frames(self._out)
        self.show_report()
        return self._out

    def save(self) -> Dict:
        """
        Сохраняет лучшую модель и метаданные в `<корень проекта>/artifacts`.

        Returns
        -------
        dict
            Сохранённая мета (с именем файла модели).
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

        # сохраняем файл модели
        if ba["model_key"] == "cat":
            model_path = self.artifacts_dir / "best_model.cbm"
            ba["model"].save_model(str(model_path))
        else:
            model_path = self.artifacts_dir / "best_model.joblib"
            joblib.dump(ba["model"], model_path)

        # в meta — только имя файла модели
        meta["model_path"] = model_path.name

        # сохраняем метаданные рядом
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
        Запускает полный прогон (`run`) и сразу сохраняет модель (`save`).

        Returns
        -------
        dict
            Сохранённая мета.
        """
        self.run(train_df_raw, target_col=target_col)
        return self.save()

    # --------------------------------------------------------------------- #
    #                          Построение отчётов                            #
    # --------------------------------------------------------------------- #

    def _build_report_frames(self, out: Dict) -> None:
        """
        Формирует:
          - `report_all`: метрики всех комбинаций,
          - `report_best`: сводку по лучшей,
          - `trained_list`: список комбинаций `"Model | Scenario"`.
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
        """Отображает отчёт в Jupyter (если доступен IPython). Вне Jupyter тихо пропускается."""
        if self.report_all is None or self.report_best is None:
            return
        if _HAVE_IPY:
            display(Markdown("### РЕЗУЛЬТАТЫ (CV, OoF)"))
            display(self.report_all)
            display(Markdown("### Лучшая модель"))
            display(self.report_best)

    # --------------------------------------------------------------------- #
    #                               Служебные                               #
    # --------------------------------------------------------------------- #

    def _echo_saved_meta(self, meta: Dict) -> None:
        """Красиво показывает сохранённую мету (через IPython) или печатает JSON строкой."""
        text = "**Сохранены артефакты**:\n\n```json\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```"
        if _HAVE_IPY:
            display(Markdown(text))
        else:  # pragma: no cover
            print(text)
