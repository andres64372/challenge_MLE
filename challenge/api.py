import fastapi
from pydantic import BaseModel
from typing import List

from joblib import load
import pandas as pd

from .utils import top_10_features

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class Flights(BaseModel):
    flights: List[Flight]

model = load('./challenge/model.joblib')

app = fastapi.FastAPI()  

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: Flights) -> dict:
    for flight in data.flights:
        if flight.MES not in range(1,13):
            raise fastapi.HTTPException(400)
        if flight.TIPOVUELO not in ['I', 'N']:
            raise fastapi.HTTPException(400)
    df = pd.DataFrame([flight.dict() for flight in data.flights])
    features = pd.concat([
        pd.get_dummies(df['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(df['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(df['MES'], prefix = 'MES')], 
        axis = 1
    )
    for feature in top_10_features:
        if feature not in df:
            features[feature] = False
    return {
        "predict": [int(item) for item in model.predict(features[top_10_features])]
    }