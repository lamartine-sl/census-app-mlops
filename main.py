"""
Script used to creation a API in FastApi.

Author: Lamartine Santana
Date: September, 2021
"""
import os
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from starter.ml.model import *
from starter.ml.data import *

# FastAPI instance
app = FastAPI()

# Heroku access to DVC data
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Read params.yaml file
CWD = os.getcwd()
with open(os.path.join(CWD, "starter", 'params.yaml'), 'r', encoding="UTF-8") as fp:
    CONFIG = yaml.safe_load(fp)

class Input(BaseModel):
    age : int = 23
    workclass : str = 'Self-emp-inc'
    fnlgt : int = 76516
    education : str = 'Bachelors'
    education_num : int = 13
    marital_status : str ='Married-civ-spouse' 
    occupation : str = 'Exec-managerial'
    relationship : str = 'Husband'
    race : str = 'White'
    sex : str = 'Male'
    capital_gain : int = 0
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = 'United States'

class Output(BaseModel):
    prediction:str

# Load Pickle files: Model, Encoder and LabelBinarizer   
@app.on_event("startup")
async def startup_event(): 
    global model, encoder, binarizer
    model = pickle.load(open("./model/model.pkl", "rb"))
    encoder = pickle.load(open("./model/encoder.pkl", "rb"))
    binarizer = pickle.load(open("./model/lb.pkl", "rb"))

@app.get("/")
def welcome():
    return "Hi, Welcome to Census API"

@app.post("/predict", response_model=Output, status_code=200)
def predict(data: Input):

    # Categorical features for transform model
    cat_features = CONFIG['categorical_features']

    # load predict_data
    request_dict = data.dict(by_alias=True)
    request_data = pd.DataFrame(request_dict, index=[0])
    
    X, _, _, _ = process_data(
                request_data,
                categorical_features=cat_features,
                training=False,
                encoder=encoder,
                lb=binarizer)

    prediction = model.predict(X)
    
    if prediction[0] == 1:
        prediction = "Salary > 50k"
    else:
        prediction = "Salary <= 50k"
    return {"prediction": prediction}