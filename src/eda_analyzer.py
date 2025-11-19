# file: src/eda_analyzer.py
from __future__ import annotations

from typing import List, Optional

import pandas as pd


class EDAAnalyzer:
    """
    Pipeline for normalizing raw data before training/inference.

    Steps performed:
      1) Remove service columns (`'Unnamed: 0'`, `'Unnamed:0'`, `'id'`).
      2) Normalize `Gender` to Int8 {male:1, female:0} (accepts strings/0/1).
      3) If both `BMI` and `Obesity` are present, keep only `BMI`.
      4) Set basic types for selected fields:
         - `Diet`: numeric category (after coercing to numbers);
         - `Stress Level`: integer type;
         - `Physical Activity Days Per Week`: integer type.
      5) `missing_values()`:
         - normalizes missing values (including hidden markers),
         - adds `block_missing` if the entire block of socio-medical fields is empty.
      6) `cast_dtypes()`:
         - object to binary Int8 (if values are {0/1/yes/no/true/false}), otherwise to category;
         - integer/categorical columns are left untouched;
         - everything else to numeric (float).
      7) `impute_minimal()`:
         - numeric columns (except target) with median,
         - `Stress Level` with integer median,
         - `Diet` adds category `'Unknown'` and fills missing with it.
      8) `report()` gives a summary; `process()` runs the steps sequentially and returns
        the cleaned DataFrame.

    """

    # block columns to detect rows where the entire questionnaire block is missing
    _BLOCK_COLS: List[str] = [
        "Diabetes", "Family History", "Smoking", "Obesity", "Alcohol Consumption",
        "Previous Heart Problems", "Medication Use", "Stress Level",
        "Physical Activity Days Per Week",
    ]

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        self.df: pd.DataFrame = df.copy()
        self.target_col: Optional[str] = target_col

        # 0) remove service columns
        drop_cols = [c for c in self.df.columns if c.lower() in ("unnamed: 0", "unnamed:0", "id")]
        self.df.drop(columns=drop_cols, errors="ignore", inplace=True)

        # 1) gender normalization: strings -> {male:1, female:0}, 1/0 also accepted
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
            # after mapping strings may remain - convert to Int8 where possible
            self.df["Gender"] = pd.to_numeric(self.df["Gender"], errors="coerce").astype("Int8")

        # protected columns (skip type casting)
        self._protect = {"Gender"}

        # 2) If both BMI and Obesity are present - keep only BMI
        if {"BMI", "Obesity"} <= set(self.df.columns):
            self.df.drop(columns=["Obesity"], inplace=True)

        # 3) Basic types
        if "Diet" in self.df.columns:
            # normalize string markers of "unknown" to missing
            self.df["Diet"] = (
                self.df["Diet"]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"unknown": pd.NA, "n/a": pd.NA, "na": pd.NA, "none": pd.NA})
            )
            # force numeric conversion
            self.df["Diet"] = pd.to_numeric(self.df["Diet"], errors="coerce")
            # round to integer and convert to categorical
            self.df["Diet"] = self.df["Diet"].round().astype("Int16").astype("category")

        if "Stress Level" in self.df.columns:
            self.df["Stress Level"] = pd.to_numeric(self.df["Stress Level"],
                                                    errors="coerce").round().astype("Int16")

        if "Physical Activity Days Per Week" in self.df.columns:
            self.df["Physical Activity Days Per Week"] = (
                pd.to_numeric(self.df["Physical Activity Days Per Week"],
                              errors="coerce").round().astype("Int16")
            )

    # ----------------------------- pipeline steps -----------------------------

    def missing_values(self) -> None:
        """
        Normalize missing values and mark rows with all block fields absent.
        Also attempts to replace "hidden" missing markers (same value in an out-of-block column
        appearing exactly as often as NaN in each block column) with NaN
        *only if those occurrences coincide with rows where the entire block is empty*.
        """
        # empty strings to NaN
        self.df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

        present = [c for c in self._BLOCK_COLS if c in self.df.columns]
        if not present:
            return

        # NaN frequency per block column
        blk_na = self.df[present].isna().sum()
        if (blk_na.nunique() == 1) and (blk_na.iloc[0] > 0):
            n = int(blk_na.iloc[0])
            # check out-of-block columns for a special value with the same frequency
            for c in (x for x in self.df.columns if x not in present):
                vc = self.df[c].value_counts(dropna=False)
                for val, cnt in vc.items():
                    if pd.isna(val) or cnt != n:
                        continue
                    mask_val = self.df[c] == val
                    # replace with NaN only where the entire block is empty
                    if (self.df.loc[mask_val, present].isna().all(axis=1)).all():
                        self.df.loc[mask_val, c] = pd.NA
        # flag "entire block missing"
        block_mask = self.df[present].isna().all(axis=1)
        if block_mask.any():
            self.df["block_missing"] = block_mask.astype("Int8")

    def cast_dtypes(self) -> None:
        """
        Type casting:
          - object to binary Int8 (if values are within {0,1,yes/no,true/false}) else to category;
          - integer/categorical columns are untouched;
          - everything else to numeric via `to_numeric`.
        """
        for c in self.df.columns:
            if c == self.target_col or c in self._protect:
                continue

            s = self.df[c]

            # object: attempt binary conversion, otherwise category
            if pd.api.types.is_object_dtype(s):
                u = s.astype("string").str.strip().str.lower()
                mapped = u.replace({"true": "1", "false": "0", "yes": "1", "no": "0"})
                t = pd.to_numeric(mapped, errors="coerce")
                if t.dropna().isin([0, 1]).all():
                    self.df[c] = t.astype("Int8")
                else:
                    self.df[c] = u.astype("category")
                continue

            # if already int / category - leave as is
            if s.dtype.kind in "iu" or pd.api.types.is_categorical_dtype(s.dtype):
                continue

            # convert the rest to numbers (float/IntNA)
            self.df[c] = pd.to_numeric(s, errors="coerce")

    def impute_minimal(self) -> None:
        """
        Minimal missing value imputations:
          - all numeric (except target) with median;
          - `Stress Level` with integer median (`Int16`);
          - `Diet` add category `'Unknown'` and fill with it.
        """
        # numeric columns: median; pandas keeps integer dtype for Int types
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
            # at this stage Diet is categorical; add category Unknown and fill with it
            if self.df["Diet"].isna().any():
                self.df["Diet"] = self.df["Diet"].cat.add_categories(["Unknown"]).fillna("Unknown")

    # ----------------------------- report and run -----------------------------

    def report(self) -> None:
        """
        Print a short report:
          - shape,
          - target balance (if present),
          - executes missing_values - cast_dtypes - impute_minimal,
          - list of columns and their dtypes,
          - correlations of numeric features with target (if target present).
        """
        print(f"Shape: {self.df.shape}")
        if (self.target_col is not None) and (self.target_col in self.df.columns):
            print("Target balance:\n", self.df[self.target_col].value_counts(normalize=True))

        self.missing_values()
        self.cast_dtypes()
        self.impute_minimal()

        print("Columns:", list(self.df.columns))
        print("Dtypes:\n", self.df.dtypes)
        if (self.target_col is not None) and (self.target_col in self.df.columns):
            print("\nCorrelation with target (numeric):")
            print(self.df.corr(numeric_only=True)[self.target_col].sort_values(ascending=False))

    def process(self) -> pd.DataFrame:
        """
        Sequentially runs `missing_values()` - `cast_dtypes()` - `impute_minimal()`
        and returns the cleaned DataFrame.
        """
        self.missing_values()
        self.cast_dtypes()
        self.impute_minimal()
        return self.df
