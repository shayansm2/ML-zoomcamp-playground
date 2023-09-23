import pandas as pd


class CategoricalFeatureEncoder(object):
    # todo add the number of most frequent values it should consider.
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        assert feature_name in df.columns, f'{feature_name} does not exists'
        options = df[feature_name].unique().tolist()
        for option in options:
            new_feature_name = feature_name + '_' + option
            value = (df[feature_name] == option).astype(int)
            df[new_feature_name] = value
        return df
