# файл: src/eda_analyzer.py
import pandas as pd


class EDAAnalyzer:
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col

        # 0) удалить служебные
        drop_cols = [c for c in self.df.columns if c.lower() in ('unnamed: 0', 'unnamed:0', 'id')]
        self.df.drop(columns=drop_cols, errors='ignore', inplace=True)

        if 'Gender' in self.df.columns:
            g = (self.df['Gender'].astype('string').str.strip()
                 .str.replace(r'\.0$', '', regex=True).str.lower())
            gender_map = {'male': 1, 'female': 0, '1': 1, '0': 0}
            self.df['Gender'] = g.map(gender_map).fillna(g).astype('Int8')

        self._protect = {'Gender'}  # не трогать в cast_dtypes

        # 2) Если есть BMI и Obesity — оставить только BMI (избежать дублирования смысла)
        if {'BMI', 'Obesity'} <= set(self.df.columns):
            self.df.drop(columns=['Obesity'], inplace=True)

        # 3) Базовые «правильные» типы (минимально и по именам)
        if 'Diet' in self.df.columns:
            self.df['Diet'] = self.df['Diet'].astype('Int16').astype('category')
        if 'Stress Level' in self.df.columns:
            self.df['Stress Level'] = pd.to_numeric(self.df['Stress Level'], errors='coerce').round().astype('Int16')
        if 'Physical Activity Days Per Week' in self.df.columns:
            self.df['Physical Activity Days Per Week'] = (
                pd.to_numeric(self.df['Physical Activity Days Per Week'], errors='coerce').round().astype('Int16')
            )

    def missing_values(self):
        # нормализация текстовых пропусков
        self.df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

        # колонны «блока» (те самые 9)
        block_cols = [
            'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption',
            'Previous Heart Problems', 'Medication Use', 'Stress Level',
            'Physical Activity Days Per Week'
        ]
        present = [c for c in block_cols if c in self.df.columns]

        # скрытые маркеры пропусков: если встречается одно «особое» значение ровно столько же раз,
        # сколько NaN в "блоке" — заменяем его на NaN
        if present:
            # частоты NaN по блоку
            blk_na = self.df[present].isna().sum()
            if (blk_na.nunique() == 1) and (blk_na.iloc[0] > 0):
                n = int(blk_na.iloc[0])
                for c in [x for x in self.df.columns if x not in present]:
                    vc = self.df[c].value_counts(dropna=False)
                    for val, cnt in vc.items():
                        if pd.isna(val) or cnt != n: continue
                        m = (self.df[c] == val)
                        # строго совпадаем со строками блока: все present = NaN
                        if (self.df.loc[m, present].isna().all(axis=1)).all():
                            self.df.loc[m, c] = pd.NA

            # единый флаг блока
            block_mask = self.df[present].isna().all(axis=1)
            if block_mask.any():
                self.df['block_missing'] = block_mask.astype('Int8')

    def cast_dtypes(self):
        for c in self.df.columns:
            if c == self.target_col or c in self._protect:
                continue
            s = self.df[c]

            # object  попытка бинаризации, иначе category
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

            # остальное в числовое
            self.df[c] = pd.to_numeric(s, errors='coerce')

    def impute_minimal(self):
        # Числовые - медиана; Stress Level уже целочисленный — переимпутируем медианой; Diet - "Unknown"
        num_cols = [c for c in self.df.columns
                    if (c != self.target_col) and pd.api.types.is_numeric_dtype(self.df[c])]
        for c in num_cols:
            if self.df[c].isna().any():
                med = self.df[c].median()
                # для ординальных целых — округлим после заполнения
                self.df[c] = self.df[c].fillna(med)
                if pd.api.types.is_integer_dtype(self.df[c]):  # Int16/Int8 уже ок
                    continue
        if 'Stress Level' in self.df.columns and self.df['Stress Level'].isna().any():
            med = int(round(self.df['Stress Level'].median()))
            self.df['Stress Level'] = self.df['Stress Level'].fillna(med).astype('Int16')
        if 'Diet' in self.df.columns:
            if self.df['Diet'].isna().any():
                self.df['Diet'] = self.df['Diet'].cat.add_categories(['Unknown']).fillna('Unknown')

    # Отчёт
    def report(self):
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
        """Обрабатывает данные и возвращает очищенный DataFrame"""
        self.missing_values()
        self.cast_dtypes()
        self.impute_minimal()
        return self.df
