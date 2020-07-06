# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:03:20 2020

@author: Krishna
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from intro_to_flask import app
import os

elecpredapp = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

@elecpredapp.route('/')
def home():
    return render_template('index.html')
  

@elecpredapp.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Winner = 1 Not Winner = 0. The Prediction is  {}'.format(output))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    elecpredapp.run(debug=True, host='0.0.0.0', port=port)


