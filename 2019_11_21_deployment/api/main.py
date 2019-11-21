from flask import Flask
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


app = Flask(__name__)
mdl = joblib.load('binary_classification.model')

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict')
def predict():
    return mdl.predict_proba(np.r_[-0.25, 0].reshape(1, -1))

if __name__ == '__main__':
    app.run(host= '0.0.0.0')
