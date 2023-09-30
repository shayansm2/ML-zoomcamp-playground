import numpy as np
import pandas as pd


class CategoricalFeatureEncoder(object):
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, feature_name: str, feature_options: list,
                       number_of_most_frequent_values: int = None) -> tuple[pd.DataFrame, list]:
        assert feature_name in df.columns, f'{feature_name} does not exists'

        options = CategoricalFeatureEncoder._get_options(df, feature_name, feature_options,
                                                         number_of_most_frequent_values)

        for option in options:
            new_feature_name = feature_name + '_' + option
            df[new_feature_name] = (df[feature_name] == option).astype(int)
        return df, feature_options

    @staticmethod
    def _get_options(df, feature_name, feature_options, number_of_most_frequent_values):
        if number_of_most_frequent_values is None or len(feature_options) <= number_of_most_frequent_values:
            return feature_options

        unique_values, counts = np.unique(df[feature_name], return_counts=True)
        for i, val in enumerate(unique_values):
            if val not in feature_options:
                del unique_values[i]
                del counts[i]
        sorted_indices = np.argsort(counts)[::-1]
        sorted_values = unique_values[sorted_indices]
        return sorted_values[:number_of_most_frequent_values]
