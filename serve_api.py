# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd

import yaml

#parameters file
with open('params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)


# Load the model
with open("absenteeism_model.pkl", "rb") as f:
    model = pickle.load(f)


# Load the target names (class labels)
data = pd.read_csv(config['data']['raw_data_csv'],sep=";")

target_names = data.columns.to_list()
target_names2 = data.columns.to_list()

#eliminamos la etiqueta del valor que vamos a predecir
target_names.remove('Absenteeism time in hours')

# Define the input data format for prediction
class AbsData(BaseModel):
    features: List[float]

# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(abs_data: AbsData):
        
    check = all(e in  model.feature_names_in_.tolist() for e in target_names)
    print('todos')
    print(check)
    # if len(target_names) != model.n_features_in_:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Input must contain {model.n_features_in_} features."
    #     )
    # Predict
    prediction = model.predict([abs_data.features])[0]
    prediction_name = target_names[prediction]
    return {"Predicción horas ausentes": int(prediction)}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Absenteeism model API"}