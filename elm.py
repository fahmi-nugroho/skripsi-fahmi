import numpy as np
from joblib import load

from app import predict

def reshape(pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age):
    test_data = [
        pregnancies, 
        glucose, 
        bloodPressure, 
        skinThickness, 
        insulin, 
        bmi, 
        diabetesPedigreeFunction, 
        age
    ]

    test_data = np.array(test_data)
    test_data = test_data.reshape(1, -1)

    return test_data

def input_to_hidden(data, Win, bias):
    a = np.dot(data, Win)
    a += bias
    a = 1 / (1 + np.exp(-a))
    return a

def elm_predict(data):
    model = load('model/elm.pkl')
    Win = model['bobot']
    bias = model['bias']
    Wout = model['beta']
    x = input_to_hidden(data, Win, bias)
    predicted = np.dot(x, Wout)
    result = np.argmax(predicted)

    return result, predicted
