# файл: src/eda_analyzer.py
from __future__ import annotations

from typing import Optional, List

import pandas as pd


class EDAAnalyzer:
    """
    Пайплайн для нормализации «сырых» данных перед обучением/инференсом.

    Что делает:
      1) Удаляет служебные колонки (`'Unnamed: 0'`, `'Unnamed:0'`, `'id'`).
      2) Нормализует поле `Gender` в Int8 {male:1, female:0} (принимает строки/0/1).
      3) Если присутствуют оба `BMI` и `Obesity` - оставляет только `BMI`.
      4) Базовая расстановка «правильных» типов для некоторых полей:
         - `Diet`: числовая категория (после принудительного числового преобразования);
         - `Stress Level`: целочисленный тип;
         - `Physical Activity Days Per Week`: целочисленный тип.
      5) `missing_values()`:
         - нормализует пропуски (в т.ч. «скрытые» маркеры),
         - добавляет `block_missing`, если *весь блок* социально-медицинских полей пуст.
      6) `cast_dtypes()`:
         - object в бинарный Int8 (если значения из {0/1/yes/no/true/false}), иначе в category;
         - целочисленные/категориальные не трогаем;
         - остальное в числовое (float).
      7) `impute_minimal()`:
         - числовые (кроме таргета) медианой,
         - `Stress Level` целочисленной медианой,
         - `Diet` добавляем категорию `'Unknown'` и заполняем ею пропуски.
      8) `report()` отчёт; `process()` — последовательный запуск шагов и возврат очищенного DataFrame.

    Пример
    -------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({"Gender": ["male", "female", None], "BMI": [24.2, 31.1, None], "id": [1, 2, 3]})
    >>> clean = EDAAnalyzer(raw, target_col=None).process()
    >>> list(clean.columns)
    ['Gender', 'BMI']
    """

    # колонки «блока» для детекции полностью пропущенных анкетных полей
    _BLOCK_COLS: List[str] = [
        "Diabetes", "Family History", "Smoking", "Obesity", "Alcohol Consumption",
        "Previous Heart Problems", "Medication Use", "Stress Level",
        "Physical Activity Days Per Week",
    ]

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        self.df: pd.DataFrame = df.copy()
        self.target_col: Optional[str] = target_col

        # 0) удалить служебные
        drop_cols = [c for c in self.df.columns if c.lower() in ("unnamed: 0", "unnamed:0", "id")]
        self.df.drop(columns=drop_cols, errors="ignore", inplace=True)

        # 1) нормализация пола: строки -> {male:1, female:0}, 1/0 тоже принимаются
        if "Gender" in self.df.columns:
            g = (
                self.df["Gender"]
                .astype("string")
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .str.lower()
            )
            gender_map = {"male": 1, "female": 0, "1": 1, "0": 0}
            self.df["Gender"] = g.map(gender_map).fillna(g)
            # после map могут остаться строки - приведём к Int8 там, где возможно
            self.df["Gender"] = pd.to_numeric(self.df["Gender"], errors="coerce").astype("Int8")

        # защищённые колонки (не трогаем тип при cast_dtypes)
        self._protect = {"Gender"}

        # 2) Если есть BMI и Obesity - оставить только BMI
        if {"BMI", "Obesity"} <= set(self.df.columns):
            self.df.drop(columns=["Obesity"], inplace=True)

        # 3) Базовые типы
        if "Diet" in self.df.columns:
            # нормализация строковых маркеров «неизвестно» в пропуски
            self.df["Diet"] = (
                self.df["Diet"]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"unknown": pd.NA, "n/a": pd.NA, "na": pd.NA, "none": pd.NA})
            )
            # принудительно пытаемся сделать числом
            self.df["Diet"] = pd.to_numeric(self.df["Diet"], errors="coerce")
            # округляем и делаем целочисленным, затем - категориальным
            self.df["Diet"] = self.df["Diet"].round().astype("Int16").astype("category")

        if "Stress Level" in self.df.columns:
            self.df["Stress Level"] = pd.to_numeric(self.df["Stress Level"], errors="coerce").round().astype("Int16")

        if "Physical Activity Days Per Week" in self.df.columns:
            self.df["Physical Activity Days Per Week"] = (
                pd.to_numeric(self.df["Physical Activity Days Per Week"], errors="coerce").round().astype("Int16")
            )

    # ----------------------------- шаги пайплайна -----------------------------

    def missing_values(self) -> None:
        """
        Нормализует пропуски и помечает строки с полным отсутствием значений в «блоке» полей.
        Также пытается заменить «скрытые» маркеры пропусков (одно и то же значение во внегрупповом поле,
        встречающееся ровно столько же раз, сколько NaN в каждом поле блока) на NaN
        *только если эти вхождения строго совпадают со строками, где весь блок пуст*.
        """
        # пустые строки в NaN
        self.df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

        present = [c for c in self._BLOCK_COLS if c in self.df.columns]
        if not present:
            return

        # частота NaN по колонкам блока
        blk_na = self.df[present].isna().sum()
        if (blk_na.nunique() == 1) and (blk_na.iloc[0] > 0):
            n = int(blk_na.iloc[0])
            # проверяем внегрупповые колонки на «особое» значение с той же кратностью
            for c in (x for x in self.df.columns if x not in present):
                vc = self.df[c].value_counts(dropna=False)
                for val, cnt in vc.items():
                    if pd.isna(val) or cnt != n:
                        continue
                    mask_val = self.df[c] == val
                    # заменяем на NaN только там, где весь блок пуст
                    if (self.df.loc[mask_val, present].isna().all(axis=1)).all():
                        self.df.loc[mask_val, c] = pd.NA

        # флаг «весь блок пропущен»
        block_mask = self.df[present].isna().all(axis=1)
        if block_mask.any():
            self.df["block_missing"] = block_mask.astype("Int8")

    def cast_dtypes(self) -> None:
        """
        Приведение типов:
          - object в бинарный Int8 (если значения укладываются в {0,1,yes/no,true/false}) иначе в category;
          - целочисленные/категориальные не трогаем;
          - остальное в числовое (через `to_numeric`).
        """
        for c in self.df.columns:
            if c == self.target_col or c in self._protect:
                continue

            s = self.df[c]

            # object: попытка бинаризации, иначе - category
            if pd.api.types.is_object_dtype(s):
                u = s.astype("string").str.strip().str.lower()
                mapped = u.replace({"true": "1", "false": "0", "yes": "1", "no": "0"})
                t = pd.to_numeric(mapped, errors="coerce")
                if t.dropna().isin([0, 1]).all():
                    self.df[c] = t.astype("Int8")
                else:
                    self.df[c] = u.astype("category")
                continue

            # если это уже int / category - не трогаем
            if s.dtype.kind in "iu" or pd.api.types.is_categorical_dtype(s.dtype):
                continue

            # остальное приводим к числу (float/IntNA)
            self.df[c] = pd.to_numeric(s, errors="coerce")

    def impute_minimal(self) -> None:
        """
        Минимальные заполнения пропусков:
          - все числовые (кроме таргета) - медианой;
          - `Stress Level` - целочисленной медианой (`Int16`);
          - `Diet` - добавляем категорию `'Unknown'` и заполняем ею.
        """
        # числовые - медианой; для Int типов pandas сохранит целочисленный dtype
        num_cols = [
            c for c in self.df.columns
            if (c != self.target_col) and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        for c in num_cols:
            if self.df[c].isna().any():
                med = self.df[c].median()
                self.df[c] = self.df[c].fillna(med)

        if "Stress Level" in self.df.columns and self.df["Stress Level"].isna().any():
            med = int(round(self.df["Stress Level"].median()))
            self.df["Stress Level"] = self.df["Stress Level"].fillna(med).astype("Int16")

        if "Diet" in self.df.columns:
            # на этом этапе Diet - категориальная; добавляем категорию Unknown и заполняем ею
            if self.df["Diet"].isna().any():
                self.df["Diet"] = self.df["Diet"].cat.add_categories(["Unknown"]).fillna("Unknown")

    # ----------------------------- отчёт и запуск -----------------------------

    def report(self) -> None:
        """
        Печатает краткий отчёт:
          - размер,
          - баланс таргета (если есть),
          - *выполняет* missing_values - cast_dtypes - impute_minimal,
          - список колонок и их типы,
          - корреляции числовых с таргетом (если таргет есть).
        """
        print(f"Размер: {self.df.shape}")
        if (self.target_col is not None) and (self.target_col in self.df.columns):
            print("Баланс таргета:\n", self.df[self.target_col].value_counts(normalize=True))

        self.missing_values()
        self.cast_dtypes()
        self.impute_minimal()

        print("Колонки:", list(self.df.columns))
        print("Dtypes:\n", self.df.dtypes)
        if (self.target_col is not None) and (self.target_col in self.df.columns):
            print("\nКорр с таргетом (числовые):")
            print(self.df.corr(numeric_only=True)[self.target_col].sort_values(ascending=False))

    def process(self) -> pd.DataFrame:
        """
        Последовательно запускает:
        `missing_values()` - `cast_dtypes()` - `impute_minimal()` и возвращает очищенный DataFrame.
        """
        self.missing_values()
        self.cast_dtypes()
        self.impute_minimal()
        return self.df
