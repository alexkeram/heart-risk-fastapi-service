# файл: src/eda_analyzer.py
import pandas as pd


class EDAAnalyzer:
    """
    Пайплайн для нормализации «сырых» данных перед обучением/инференсом.

    Что делает:
      1) Удаляет служебные колонки ('Unnamed: 0', 'Unnamed:0', 'id').
      2) Приводит 'Gender' к бинарному целочисленному признаку (male=1, female=0)
      3) Устранение дублирующего смысла: если есть оба 'BMI' и 'Obesity', оставляет только 'BMI'.
      4) Базовая расстановка «правильных» типов для некоторых известных полей (Diet, Stress Level,
         Physical Activity Days Per Week).
      5) missing_values(): нормализует пропуски (в т.ч. «скрытые» маркеры), ставит флаг block_missing
         при полном пропуске целого «блока» социально-медицинских полей.
      6) cast_dtypes(): пытается привести типы:
           - object в бинарный Int8 (если это «да/нет» и т.п.) или category,
           - уже int/category не трогаем,
           - остальное в числовое (float).
      7) impute_minimal(): минимальные заполнения пропусков:
           - числовые медианой (целочисленные при этом не опускаем в float),
           - Stress Level целочисленной медианой,
           - Diet добавляем категорию 'Unknown' и заполняем ею.
      8) report(): короткий печатный отчёт состояния,
         process(): запускает 5–7 и отдаёт очищенный DataFrame.

    """

    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col

        # 0) удалить служебные
        drop_cols = [c for c in self.df.columns if c.lower() in ('unnamed: 0', 'unnamed:0', 'id')]
        self.df.drop(columns=drop_cols, errors='ignore', inplace=True)

        # 1) нормализация пола: строки -> {male:1, female:0}, значения 1/0 также принимаются
        if 'Gender' in self.df.columns:
            g = (self.df['Gender'].astype('string').str.strip()
                 .str.replace(r'\.0$', '', regex=True).str.lower())
            gender_map = {'male': 1, 'female': 0, '1': 1, '0': 0}
            self.df['Gender'] = g.map(gender_map).fillna(g).astype('Int8')

        # этот столбец защищаем от дальнейшего кастинга типов
        self._protect = {'Gender'}

        # 2) Если есть BMI и Obesity — оставить только BMI
        if {'BMI', 'Obesity'} <= set(self.df.columns):
            self.df.drop(columns=['Obesity'], inplace=True)

        # 3) Базовые типы
        if 'Diet' in self.df.columns:
            # сначала нормализуем строковые маркеры «неизвестно» в пропуски
            self.df['Diet'] = (
                self.df['Diet']
                .astype('string')
                .str.strip()
                .str.lower()
                .replace({'unknown': pd.NA, 'n/a': pd.NA, 'na': pd.NA, 'none': pd.NA})
            )
            # приводим к числу всё нечисловое в NaN
            self.df['Diet'] = pd.to_numeric(self.df['Diet'], errors='coerce')
            # округляем и делаем целочисленным, затем — категориальным
            self.df['Diet'] = self.df['Diet'].round().astype('Int16').astype('category')
        if 'Stress Level' in self.df.columns:
            self.df['Stress Level'] = pd.to_numeric(self.df['Stress Level'], errors='coerce').round().astype('Int16')
        if 'Physical Activity Days Per Week' in self.df.columns:
            self.df['Physical Activity Days Per Week'] = (
                pd.to_numeric(self.df['Physical Activity Days Per Week'], errors='coerce').round().astype('Int16')
            )

    def missing_values(self):
        """
        Нормализует пропуски и помечает строки с полным отсутствием значений в «блоке» полей.
        Также заменяет редкие маркеры пропусков (одно и то же значение во внеблоковом поле,
        встречающееся ровно столько же раз, сколько NaN в каждом поле блока).
        """
        # нормализация текстовых пропусков
        self.df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # колонны «блока» (те самые 9)
        block_cols = [
            'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption',
            'Previous Heart Problems', 'Medication Use', 'Stress Level',
            'Physical Activity Days Per Week'
        ]
        present = [c for c in block_cols if c in self.df.columns]

        # «скрытые» маркеры пропусков: если встречается одно «особое» значение ровно столько же раз,
        # сколько NaN в блоке — заменяем его на NaN при строгом совпадении строк блока
        if present:
            # частоты NaN по блоку
            blk_na = self.df[present].isna().sum()
            if (blk_na.nunique() == 1) and (blk_na.iloc[0] > 0):
                n = int(blk_na.iloc[0])
                for c in [x for x in self.df.columns if x not in present]:
                    vc = self.df[c].value_counts(dropna=False)
                    for val, cnt in vc.items():
                        if pd.isna(val) or cnt != n:
                            continue
                        m = (self.df[c] == val)
                        # строго совпадаем со строками блока: все present = NaN
                        if (self.df.loc[m, present].isna().all(axis=1)).all():
                            self.df.loc[m, c] = pd.NA

            # единый флаг «весь блок пропущен»
            block_mask = self.df[present].isna().all(axis=1)
            if block_mask.any():
                self.df['block_missing'] = block_mask.astype('Int8')

    def cast_dtypes(self):
        """
        Приведение типов:
          - object в бинарный Int8 (если значения укладываются в {0,1,yes/no,true/false}) иначе в category,
          - int/category уже не трогаем,
          - остальное в числовое (через to_numeric).
        """
        for c in self.df.columns:
            if c == self.target_col or c in self._protect:
                continue
            s = self.df[c]

            # object: попытка бинаризации, иначе category
            if pd.api.types.is_object_dtype(s):
                u = s.astype('string').str.strip().str.lower()
                t = pd.to_numeric(u.replace({'true': '1', 'false': '0', 'yes': '1', 'no': '0'}), errors='coerce')
                if t.dropna().isin([0, 1]).all():
                    self.df[c] = t.astype('Int8')
                else:
                    self.df[c] = u.astype('category')
                continue

            # НЕ перекастим уже целочисленные и категориальные обратно во float
            if s.dtype.kind in 'iu' or isinstance(s.dtype, pd.CategoricalDtype):
                continue

            # остальное — попытка привести к числу
            self.df[c] = pd.to_numeric(s, errors='coerce')

    def impute_minimal(self):
        """
        Минимальные заполнения пропусков:
          - все числовые (кроме таргета) медианой;
          - Stress Level целочисленной медианой (Int16);
          - Diet — добавляем категорию 'Unknown' и заполняем ею.
        """
        # Числовые — медиана; для Int типов оставляем целочисленный dtype
        num_cols = [c for c in self.df.columns
                    if (c != self.target_col) and pd.api.types.is_numeric_dtype(self.df[c])]
        for c in num_cols:
            if self.df[c].isna().any():
                med = self.df[c].median()
                self.df[c] = self.df[c].fillna(med)
                # если столбец уже целочисленный (Int8/Int16/...), тип сохранится

        if 'Stress Level' in self.df.columns and self.df['Stress Level'].isna().any():
            med = int(round(self.df['Stress Level'].median()))
            self.df['Stress Level'] = self.df['Stress Level'].fillna(med).astype('Int16')

        if 'Diet' in self.df.columns:
            if self.df['Diet'].isna().any():
                self.df['Diet'] = self.df['Diet'].cat.add_categories(['Unknown']).fillna('Unknown')

    # Отчёт
    def report(self):
        """
        Печатает краткий отчёт о состоянии данных:
          - размер,
          - баланс таргета (если есть),
          - запуски missing_values/cast_dtypes/impute_minimal,
          - список колонок и их типы,
          - корреляции числовых с таргетом (если таргет есть).
        """
        print(f"Размер: {self.df.shape}")
        if self.target_col in self.df.columns:
            print("Баланс таргета:\n", self.df[self.target_col].value_counts(normalize=True))
        self.missing_values()
        self.cast_dtypes()
        self.impute_minimal()
        print("Колонки:", list(self.df.columns))
        print("Dtypes:\n", self.df.dtypes)
        if self.target_col in self.df.columns:
            print("\nКорр с таргетом (числовые):")
            print(self.df.corr(numeric_only=True)[self.target_col].sort_values(ascending=False))

    def process(self):
        """Запускает обработку (missing_values в cast_dtypes в impute_minimal) и возвращает очищенный DataFrame."""
        self.missing_values()
        self.cast_dtypes()
        self.impute_minimal()
        return self.df
