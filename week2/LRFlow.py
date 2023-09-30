import numpy as np
import seaborn as sns
import pandas as pd

from week2.RidgeRegressionModel import RidgeRegressionModel


def default_input_modifier(df: pd.DataFrame):
    x = df.copy()
    x = x.apply(pd.to_numeric, errors='coerce')
    x.fillna(0, inplace=True)
    return x.values


class LRFlow(object):
    def __init__(self, df: pd.DataFrame):
        self.sanitize_inputs(df)
        self.df = df
        self.input_modifier = default_input_modifier
        self.df_train = None

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)

    def set_input_modifier(self, input_modifier: callable):
        self.input_modifier = input_modifier

    def get_validation_rmse(self):
        model, y_predict = self.run_linear_regression()
        result = model.rmse(self.y_validation, y_predict)
        return round(result, 3)

    def run_linear_regression(self):
        self._split_dataframe()
        x_train = self.input_modifier(self.df_train)
        model = RidgeRegressionModel(len(x_train[0]))
        model.train(x_train, self.y_train)
        x_validation = self.input_modifier(self.df_validation)
        y_predict = model.predict(x_validation)
        return model, y_predict

    def plot_validation_predict_vs_actual(self):
        _, y_predict = self.run_linear_regression()
        sns.histplot(self.y_validation)
        sns.histplot(y_predict)

    def _split_dataframe(self):
        if self.df_train is not None:
            return

        n = len(self.df)
        test_ratio = 0.2
        validation_ratio = 0.2
        seed = 22

        n_validation = int(validation_ratio * n)
        n_test = int(test_ratio * n)
        n_train = n - (n_validation + n_test)

        np.random.seed(seed)
        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        self.df_train: pd.DataFrame = df_shuffled.iloc[:n_train].copy()
        self.df_validation: pd.DataFrame = df_shuffled.iloc[n_train:n_train + n_validation].copy()
        self.df_test: pd.DataFrame = df_shuffled.iloc[n_train + n_validation:].copy()

        self.y_train = np.log1p(self.df_train.median_house_value.values)
        self.y_validation = np.log1p(self.df_validation.median_house_value.values)
        self.y_test = np.log1p(self.df_test.median_house_value.values)

    @staticmethod
    def sanitize_inputs(df: pd.DataFrame):
        # normalizing column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # normalizing string values
        string_columns = list(df.dtypes[df.dtypes == 'object'].index)
        for col in string_columns:
            df[col] = df[col].str.lower().str.replace(' ', '_')
