import numpy as np
import math
from joblib import load

def reshape(pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age):
    total = 0
    penyimpangan = 0
    new_data = []

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

    for data in test_data:
        total += data

    rata2 = total / 8

    for i in range(8):
        penyimpangan += pow(test_data[i]-rata2, 2)
    
    varians = penyimpangan / 8
    std = math.sqrt(varians)
    
    for new in test_data:
        new_value = (new-rata2)/std
        new_data.append(new_value)

    print('=====')
    print(new_data)

    new_data = np.array(new_data)
    new_data = new_data.reshape(1, -1)

    return new_data

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
