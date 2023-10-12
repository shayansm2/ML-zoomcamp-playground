import requests
from ml_model import predict_probability


def question3():
    client = {"job": "retired", "duration": 445, "poutcome": "success"}
    print(predict_probability(client))


def question4():
    url = "http://localhost:9696/credit-prob/"
    client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
    result = requests.post(url, json=client).json()
    print(result)


def question6():
    url = "http://localhost:9696/credit-prob/"
    client = {"job": "retired", "duration": 445, "poutcome": "success"}
    result = requests.post(url, json=client).json()
    print(result)
