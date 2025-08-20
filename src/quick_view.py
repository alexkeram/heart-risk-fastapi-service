# файл: src/quick_view.py
import matplotlib.pyplot as plt


class QuickView:
    """
    Быстрый «ручной EDA»-обзор датафрейма.

    Что делает:
      1) quick_overview()  — печатает общий обзор: head, размер, dtypes, пропуски, дубликаты, баланс таргета.
      2) quick_details()   — печатает describe, min/max, частоты категориальных/низко-кардинальных, кореляции с таргетом.
      3) quick_plots()     — рисует гистограммы для всех числовых признаков.
      4) quick_boxplots()  — рисует boxplot'ы числовых признаков по классам таргета.
      5) report()          — запускает всё по очереди.

    Параметры:
      df          : входной DataFrame.
      target      : имя таргета (может быть None; нужен для баланса/корреляций/boxplot).
      max_unique  : порог кардинальности, ниже которого признак считаем «категориальным» для частот.
    """

    def __init__(self, df, target=None, max_unique=10):
        # Сохраняем входные параметры (поведение исходных функций не меняем)
        self.df = df
        self.target = target
        self.max_unique = max_unique

    def quick_overview(self):
        """
        Общий обзор:
          • первые строки (head) транспонированно — чтобы заголовки были в одном столбце,
          • размер датафрейма,
          • типы данных,
          • количество пропусков по столбцам,
          • количество дубликатов (и доля),
          • баланс таргета (если задан и присутствует в df).
        """
        df = self.df
        target = self.target
        print("Внешний вид:", df.head().T)
        print("Размер:", df.shape)
        print("\nТипы данных:\n", df.dtypes)
        print("\nПропуски (шт):\n", df.isnull().sum())
        dup = df.duplicated().sum()
        print(f"\nДубликатов: {dup} ({dup / len(df) * 100:.2f}%)")
        if target and target in df.columns:
            print("\nБаланс таргета:")
            print(df[target].value_counts(normalize=True))

    def quick_details(self):
        """
        Детализация:
          • describe() по числовым,
          • min/max по числовым,
          • частоты значений для категориальных и «низко-кардинальных» (<= max_unique),
          • корреляции числовых признаков с таргетом (если таргет задан и в df).
        """
        df = self.df
        target = self.target
        max_unique = self.max_unique

        print("\n--- Статистика числовых признаков ---")
        print(df.describe())

        print("\n--- Min/Max по числовым ---")
        desc = df.describe()
        # сохраняем исходное поведение: доступ по индексам 'min'/'max' если они есть
        print(desc.loc[['min', 'max']] if 'min' in desc.index else desc[['min', 'max']])

        print("\n--- Частоты категориальных ---")
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() <= max_unique:
                # Частоты значений (включая NaN)
                print(f"\nЧастоты в '{col}':")
                print(df[col].value_counts(dropna=False))

        if target and target in df.columns:
            # Корреляции числовых признаков с таргетом (ожидается числовой таргет)
            print("\n--- Корреляции с таргетом ---")
            print(df.corr(numeric_only=True)[target].sort_values(ascending=False))

    def quick_plots(self, bins=50, ncols=3):
        """
        Гистограммы по всем числовым столбцам.
        Параметры:
          bins  : число бинов гистограммы,
          ncols : число колонок в сетке субплотов.
        """
        num_cols = self.df.select_dtypes(include="number").columns
        n = len(num_cols)
        if n == 0:
            print("Нет числовых столбцов для гистограмм.")
            return
        nrows = (n + ncols - 1) // ncols

        axes = self.df[num_cols].hist(
            bins=bins,
            figsize=(3.8 * ncols, 3.0 * nrows),
            layout=(nrows, ncols),
            sharex=False, sharey=False,
            grid=False
        )

        # Выравниваем подписи и заголовки, чтобы не наслаивались
        for ax in axes.ravel():
            if ax is None:  # на случай, если сетка больше числа графиков
                continue
            ax.tick_params(axis='x', labelrotation=30, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_title(ax.get_title(), fontsize=9, pad=4)

        plt.tight_layout()
        plt.show()

    def quick_boxplots(self, cols=None, ncols=3):
        """
        Boxplot'ы по числовым признакам, сгруппированные по значениям таргета.
        Параметры:
          cols  : список столбцов (если None — все числовые, кроме таргета),
          ncols : число колонок в сетке субплотов.
        Требует валидный target, присутствующий в df.
        """
        y = self.target
        if not y or y not in self.df.columns:
            print("Укажи корректный target")
            return

        base = self.df.drop(columns=['id', 'Unnamed: 0', 'Unnamed:0'], errors='ignore')
        cols = cols or [c for c in base.select_dtypes(include="number").columns if c != y]
        nrows = (len(cols) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)
        for ax, c in zip(axes.ravel(), cols):
            base.boxplot(column=c, by=y, ax=ax)
            ax.set_title(c)
            ax.set_xlabel(y)
        for ax in axes.ravel()[len(cols):]:
            ax.set_visible(False)

        plt.suptitle("")
        fig.tight_layout()
        plt.show()

    def report(self):
        """
        Полный отчёт: последовательно вызывает quick_overview, quick_details,
        quick_plots и quick_boxplots.
        """
        self.quick_overview()
        self.quick_details()
        self.quick_plots()
        self.quick_boxplots()
#%%
