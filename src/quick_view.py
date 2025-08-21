# файл: src/quick_view.py
"""
QuickView — утилита для быстрого DataFrame.

Назначение
----------
Класс помогает быстро получить первичное представление о данных:
- общий обзор структуры и пропусков;
- детальная сводка (describe, min/max), частоты категориальных/низко-кардинальных признаков,
  корреляции с таргетом;
- базовые графики распределений (hist) и boxplot'ы по классам таргета.

Зависимости
-----------
- pandas (как источник DataFrame)
- matplotlib (для графиков)

Установка (если нужно):
    pip install pandas matplotlib

Быстрый старт (в ноутбуке или скрипте)
--------------------------------------
>>> import pandas as pd
>>> from quick_view import QuickView
>>> df = pd.read_csv("data/heart_train.csv")
>>> qv = QuickView(df, target="Heart Attack Risk (Binary)")
>>> qv.report()        # выведет текстовые отчёты + нарисует hist и boxplot'ы
# или по частям:
>>> qv.quick_overview()
>>> qv.quick_details()
>>> qv.quick_plots(bins=40)
>>> qv.quick_boxplots()

Примечания
----------
- Все печатающие методы выводят информацию в stdout (подходят для ячеек Jupyter).
- Методы рисуют графики с помощью matplotlib; в Jupyter они появятся прямо под ячейкой.
"""

from __future__ import annotations

from typing import Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt


class QuickView:
    """
    Быстрый «ручной EDA»-обзор датафрейма.

    Что делает:
      1) quick_overview()  — печатает общий обзор: head, размер, dtypes, пропуски, дубликаты, баланс таргета.
      2) quick_details()   — печатает describe, min/max, частоты категориальных/низко-кардинальных, корреляции с таргетом.
      3) quick_plots()     — рисует гистограммы для всех числовых признаков.
      4) quick_boxplots()  — рисует boxplot'ы числовых признаков по классам таргета.
      5) report()          — запускает всё по очереди.

    Параметры __init__:
      df          : pandas.DataFrame, исходные данные.
      target      : имя таргета (может быть None; нужен для баланса/корреляций/boxplot).
      max_unique  : порог кардинальности, ниже которого признак считаем «категориальным» для частот.
    """

    def __init__(self, df: pd.DataFrame, target: Optional[str] = None, max_unique: int = 10) -> None:
        # Сохраняем параметры. Исходный DataFrame не модифицируется.
        self.df = df
        self.target = target
        self.max_unique = int(max_unique)

    # ============================ ТЕКСТОВЫЕ ОТЧЁТЫ ============================

    def quick_overview(self) -> None:
        """
        Общий обзор набора:
          • первые строки (head) в транспонированном виде (заголовки в одном столбце),
          • размер датафрейма,
          • типы данных,
          • число пропусков по столбцам,
          • число дубликатов (и доля),
          • баланс таргета (если target указан и присутствует в df).
        """
        df = self.df
        target = self.target

        print("=== QUICK OVERVIEW ===")
        # head().T — заголовки столбцов становятся индексом; удобно «на глаз»
        print("\nВнешний вид (head, T):")
        print(df.head().T)

        print("\nРазмер (rows, cols):", df.shape)

        print("\nТипы данных (dtypes):")
        print(df.dtypes)

        print("\nПропуски (кол-во на столбец):")
        print(df.isnull().sum())

        dup = df.duplicated().sum()
        dup_pct = (dup / len(df) * 100.0) if len(df) else 0.0
        print(f"\nДубликатов: {dup} ({dup_pct:.2f}%)")

        if target and target in df.columns:
            print("\nБаланс таргета (доля классов):")
            print(df[target].value_counts(normalize=True))

    def quick_details(self) -> None:
        """
        Детализация набора:
          • describe() по числовым,
          • min/max по числовым,
          • частоты значений для категориальных и «низко-кардинальных» столбцов (<= max_unique),
          • корреляции числовых признаков с таргетом (если таргет указан и присутствует в df).

        Примечание: корреляция считается только для числовых столбцов (numeric_only=True).
        """
        df = self.df
        target = self.target
        max_unique = self.max_unique

        print("\n=== QUICK DETAILS ===")

        # --- базовая статистика по числовым
        num_cols = df.select_dtypes(include="number")
        desc = num_cols.describe() # только числовые
        if not desc.empty:
            print("\n--- Статистика числовых признаков (describe) ---")
            print(desc)

            # Вывод min/max, если эти строки присутствуют в describe
            min_max_rows = [ix for ix in ("min", "max") if ix in desc.index]
            if min_max_rows:
                print("\n--- Min/Max по числовым ---")
                print(desc.loc[min_max_rows])
        else:
            print("\nНет числовых столбцов для describe().")

        # --- частоты по категориальным и низко-кардинальным столбцам
        print("\n--- Частоты категориальных / низко-кардинальных признаков ---")
        any_freq = False
        for col in df.columns:
            # считаем «категориальным» если тип object или уникальных мало
            if df[col].dtype == "object" or df[col].nunique(dropna=False) <= max_unique:
                any_freq = True
                print(f"\nЧастоты в '{col}':")
                print(df[col].value_counts(dropna=False))
        if not any_freq:
            print("(Подходящих столбцов не найдено)")

        # --- корреляции с таргетом
        if target and target in df.columns:
            num_corr = df.corr(numeric_only=True)
            if target in num_corr.columns:
                print("\n--- Корреляции числовых признаков с таргетом ---")
                print(num_corr[target].sort_values(ascending=False))
            else:
                print("\n(Таргет не числовой — корреляции не посчитаны.)")

    # ============================ ГРАФИКИ ============================

    def quick_plots(self, *, bins: int = 50, ncols: int = 3) -> None:
        """
        Гистограммы распределений для всех числовых столбцов.

        Параметры:
          bins  : число бинов гистограммы,
          ncols : число колонок в сетке субплотов.

        Вызывает plt.tight_layout() и plt.show().
        """
        num_cols = self.df.select_dtypes(include="number").columns
        n = len(num_cols)
        if n == 0:
            print("\nНет числовых столбцов для гистограмм.")
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

        # Выравниваем подписи и заголовки, чтобы не наслаивались
        # axes может быть ndarray либо список списков — нормализуем через ravel()
        for ax in axes.ravel():
            if ax is None:  # если сетка больше числа графиков
                continue
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(ax.get_title(), fontsize=9, pad=4)

        plt.tight_layout()
        plt.show()

    def quick_boxplots(self, cols: Optional[Sequence[str]] = None, *, ncols: int = 3) -> None:
        """
        Boxplot'ы по числовым признакам, сгруппированные по значениям таргета.

        Параметры:
          cols  : список столбцов (если None — все числовые, кроме таргета, и без служебных ID),
          ncols : число колонок в сетке субплотов.

        Требуется валидный target, присутствующий в df. Если target отсутствует — метод сообщает и возвращается.
        """
        y = self.target
        if not y or y not in self.df.columns:
            print("\nquick_boxplots: укажи корректный target (self.target) и убедись, что он есть в df.")
            return

        # Убираем возможные служебные столбцы с идентификаторами, если есть
        base = self.df.drop(columns=["id", "Unnamed: 0", "Unnamed:0"], errors="ignore")

        # Список колонок: по умолчанию все числовые, кроме таргета
        if cols is None:
            cols = [c for c in base.select_dtypes(include="number").columns if c != y]

        if len(cols) == 0:
            print("\nНет подходящих числовых столбцов для boxplot.")
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
                print(f"[WARN] Не удалось построить boxplot для '{c}': {e}")

        # Скрываем «лишние» оси, если сетка больше количества графиков
        for ax in axes.ravel()[len(cols):]:
            ax.set_visible(False)

        # убираем общий заголовок, который pandas.boxplot добавляет автоматически
        plt.suptitle("")
        fig.tight_layout()
        plt.show()

    # ============================ ОРКЕСТРАТОР ============================

    def report(self) -> None:
        """
        Полный отчёт: последовательно вызывает:
          1) quick_overview
          2) quick_details
          3) quick_plots
          4) quick_boxplots

        Если target не указан, шаги 1–3 выполнятся, boxplot пропустится.
        """
        self.quick_overview()
        self.quick_details()
        self.quick_plots()
        self.quick_boxplots()


# Если запустить этот файл как скрипт, покажем краткую справку по использованию:
if __name__ == "__main__":
    print(
        "QuickView — модуль-утилита для быстрого EDA.\n"
        "Использование (в Python/Jupyter):\n"
        "    from quick_view import QuickView\n"
        "    import pandas as pd\n"
        "    df = pd.read_csv('data.csv')\n"
        "    qv = QuickView(df, target='target_col')\n"
        "    qv.report()\n"
    )
