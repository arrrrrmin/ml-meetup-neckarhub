#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:27:39 2019

@author: hh
This is essentially a modified  version of 
https://towardsdatascience.com/deploy-your-machine-learning-model-as-a-rest-api-4fe96bf8ddcc
"""

import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import joblib


app = Flask(__name__)
api = Api(app)

model = joblib.load('binary_classification.model')
classes = ['good', 'bad']

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        #  cs for component score
        cs1 = posted_data['cs1']
        cs2 = posted_data['cs2']
        
        prediction = model.predict([[cs1, cs2]])
        
        predicted_class = classes(prediction)
        
        return jsonify({'Prediction': predicted_class})
    
api.add_resource(MakePrediction, '/predict')    


if __name__ == '__main__':
    app.run(debug=True)