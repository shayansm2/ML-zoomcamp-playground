from flask import Flask, request, jsonify
from ml_model import predict_probability

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello, World"


@app.post("/credit-prob/")
def get_credit_prob():
    client = request.get_json()
    prob = predict_probability(client)
    result = {
        'credit_chance': prob,
        'gets_credit': bool(prob >= 0.5)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
