from fastapi import FastAPI
import pickle
import numpy as np

with open("/code/app/logistic_regression_model.pkl", 'rb') as f:
    loaded_model = pickle.load(f)

class_names = np.array(['malignant' 'benign'])

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message':'Welcome to the Cancer prediction API'}

@app.post('/predict')
def predict(data:dict):
    features = np.array(data['features']).reshape(1,-1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted result':class_name}
    
