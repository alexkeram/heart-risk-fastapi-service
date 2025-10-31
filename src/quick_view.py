# file: src/quick_view.py
"""
QuickView — utility for quick DataFrame exploration.

Purpose
-------
Provides a rapid initial understanding of data:
- overall structure and missing values;
- detailed summary (describe, min/max), frequencies of categorical/low-cardinality features,
  correlations with the target;
- basic distribution plots (hist) and boxplots by target classes.

Notes
-----
- All printing methods output to stdout (works for Jupyter cells).
- Plots are drawn with matplotlib; in Jupyter they appear below the cell.
"""
from __future__ import annotations

from typing import Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt


class QuickView:
    """
    Fast "manual EDA" overview of a DataFrame.

    What it does:
      1) quick_overview()  — prints general overview: head, shape, dtypes, missing, duplicates, target balance.
      2) quick_details()   — prints describe, min/max, value frequencies for categorical/low-cardinality features, correlations with target.
      3) quick_plots()     — draws histograms for all numeric features.
      4) quick_boxplots()  — draws boxplots of numeric features grouped by target classes.
      5) report()          — runs all of the above in sequence.

    __init__ parameters:
      df         : pandas.DataFrame, source data.
      target     : name of target (can be None; used for balance/correlations/boxplot).
      max_unique : cardinality threshold below which a feature is treated as "categorical" for frequencies.
    """

    def __init__(self, df: pd.DataFrame, target: Optional[str] = None, max_unique: int = 10) -> None:
        # Store parameters. Original DataFrame is not modified.
        self.df = df
        self.target = target
        self.max_unique = int(max_unique)

    # ============================ TEXT REPORTS ============================

    def quick_overview(self) -> None:
        """
        General dataset overview:
          • first rows (head) transposed (headers in one column),
          • DataFrame shape,
          • data types,
          • number of missing values per column,
          • number of duplicates (and share),
          • target balance (if target is specified and present in df).
        """
        df = self.df
        target = self.target

        print("=== QUICK OVERVIEW ===")
        # head().T — headers become index; handy for eyeballing
        print("\nAppearance (head, T):")
        print(df.head().T)

        print("\nShape (rows, cols):", df.shape)

        print("\nData types (dtypes):")
        print(df.dtypes)

        print("\nMissing values (per column):")
        print(df.isnull().sum())

        dup = df.duplicated().sum()
        dup_pct = (dup / len(df) * 100.0) if len(df) else 0.0
        print(f"\nDuplicates: {dup} ({dup_pct:.2f}%)")

        if target and target in df.columns:
            print("\nTarget balance (class share):")
            print(df[target].value_counts(normalize=True))

    def quick_details(self) -> None:
        """
        Dataset details:
          • describe() for numeric,
          • min/max for numeric,
          • value frequencies for categorical and "low-cardinality" columns (<= max_unique),
          • correlations of numeric features with target (if target is specified and present in df).

        Note: correlation is calculated only for numeric columns (numeric_only=True).
        """
        df = self.df
        target = self.target
        max_unique = self.max_unique

        print("\n=== QUICK DETAILS ===")

        # --- basic stats for numeric
        num_cols = df.select_dtypes(include="number")
        desc = num_cols.describe()  # numeric only
        if not desc.empty:
            print("\n--- Statistics of numeric features (describe) ---")
            print(desc)

            # Output min/max if present
            min_max_rows = [ix for ix in ("min", "max") if ix in desc.index]
            if min_max_rows:
                print("\n--- Min/Max of numeric ---")
                print(desc.loc[min_max_rows])
        else:
            print("\nNo numeric columns for describe().")

        # --- frequencies for categorical and low-cardinality columns
        print("\n--- Frequencies of categorical / low-cardinality features ---")
        any_freq = False
        for col in df.columns:
            # treat as categorical if type object or few unique values
            if df[col].dtype == "object" or df[col].nunique(dropna=False) <= max_unique:
                any_freq = True
                print(f"\nFrequencies in '{col}':")
                print(df[col].value_counts(dropna=False))
        if not any_freq:
            print("(No suitable columns found)")

        # --- correlations with target
        if target and target in df.columns:
            num_corr = df.corr(numeric_only=True)
            if target in num_corr.columns:
                print("\n--- Correlations of numeric features with target ---")
                print(num_corr[target].sort_values(ascending=False))
            else:
                print("\n(Target is non-numeric — correlations not computed.)")

    # ============================ PLOTS ============================

    def quick_plots(self, *, bins: int = 50, ncols: int = 3) -> None:
        """
        Histograms for all numeric columns.

        Parameters:
          bins  : number of histogram bins,
          ncols : number of columns in the subplot grid.

        Calls plt.tight_layout() and plt.show().
        """
        num_cols = self.df.select_dtypes(include="number").columns
        n = len(num_cols)
        if n == 0:
            print("\nNo numeric columns for histograms.")
            return

        ncols = max(1, int(ncols))
        nrows = (n + ncols - 1) // ncols

        print("\n=== HISTOGRAMS ===")
        axes = self.df[num_cols].hist(
            bins=bins,
            figsize=(3.8 * ncols, 3.0 * nrows),
            layout=(nrows, ncols),
            sharex=False, sharey=False,
            grid=False
        )

        # Align labels and titles to avoid overlaps
        # axes may be ndarray or list of lists — normalize via ravel()
        for ax in axes.ravel():
            if ax is None:  # if grid larger than number of plots
                continue
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(ax.get_title(), fontsize=9, pad=4)

        plt.tight_layout()
        plt.show()

    def quick_boxplots(self, cols: Optional[Sequence[str]] = None, *, ncols: int = 3) -> None:
        """
        Boxplots of numeric features grouped by target values.

        Parameters:
          cols  : list of columns (if None — all numeric except target and ID-like columns),
          ncols : number of columns in subplot grid.

        Requires a valid target present in df. If target is missing — method reports and returns.
        """
        y = self.target
        if not y or y not in self.df.columns:
            print("\nquick_boxplots: specify a valid target (self.target) and ensure it exists in df.")
            return

        # Drop possible service ID columns
        base = self.df.drop(columns=["id", "Unnamed: 0", "Unnamed:0"], errors="ignore")

        # Column list: by default all numeric except target
        if cols is None:
            cols = [c for c in base.select_dtypes(include="number").columns if c != y]

        if len(cols) == 0:
            print("\nNo suitable numeric columns for boxplot.")
            return

        ncols = max(1, int(ncols))
        nrows = (len(cols) + ncols - 1) // ncols

        print("\n=== BOXPLOTS by target ===")
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)

        for ax, c in zip(axes.ravel(), cols):
            try:
                base.boxplot(column=c, by=y, ax=ax)
                ax.set_title(c)
                ax.set_xlabel(y)
            except Exception as e:
                ax.set_visible(False)
                print(f"[WARN] Could not build boxplot for '{c}': {e}")

        # Hide extra axes if grid is larger than number of plots
        for ax in axes.ravel()[len(cols):]:
            ax.set_visible(False)

        # remove common title added by pandas.boxplot
        plt.suptitle("")
        fig.tight_layout()
        plt.show()

    # ============================ ORCHESTRATOR ============================

    def report(self) -> None:
        """
        Full report: sequentially calls:
          1) quick_overview
          2) quick_details
          3) quick_plots
          4) quick_boxplots

        If target is not specified, steps 1–3 run and boxplot is skipped.
        """
        self.quick_overview()
        self.quick_details()
        self.quick_plots()
        self.quick_boxplots()


# If run as a script, show a brief usage guide:
if __name__ == "__main__":
    print(
        "QuickView — utility module for quick EDA.\n"
        "Usage (in Python/Jupyter):\n"
        "    from quick_view import QuickView\n"
        "    import pandas as pd\n"
        "    df = pd.read_csv('data.csv')\n"
        "    qv = QuickView(df, target='target_col')\n"
        "    qv.report()\n"
    )
