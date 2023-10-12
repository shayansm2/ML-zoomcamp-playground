import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


def get_model() -> LogisticRegression:
    with open('model2.bin', 'rb') as f_in:
        model: LogisticRegression = pickle.load(f_in)
    return model


def get_encoder() -> DictVectorizer:
    with open('dv.bin', 'rb') as f_in:
        dv: DictVectorizer = pickle.load(f_in)
    return dv


def predict_probability(client_data: dict):
    dv = get_encoder()
    model = get_model()
    return model.predict_proba(dv.transform(client_data))[0][1]
