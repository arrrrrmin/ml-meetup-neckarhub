from flask import Flask, request
import numpy as np
import joblib


app = Flask(__name__)
mdl = joblib.load('binary_classification.model')

@app.route('/')
def hello_world():
    return '<h1>Hello, World!</h1>'

@app.route('/predict')
def predict():
    x = request.args.get('x')
    y = request.args.get('y')
    return np.array2string(mdl.predict_proba(np.r_[float(x), float(y)].reshape(1, -1)))

if __name__ == '__main__':
    app.run(host= '0.0.0.0')
