# app.py
from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat
from typing import Literal
import joblib
import pandas as pd
import data_formater

app = FastAPI(
    title="Titanic Model API",
    description="API for predicting survival of Titanic passengers.",
    version="1.0.0"
)

# wczytaj model
bundle = joblib.load("titanic_model.joblib")
model = bundle["model"]
features = bundle["features"]

# schema wejściowa
class Passenger(BaseModel):
    Pclass: conint(ge=1, le=3) = Field(
        ..., 
        description="Ticket class (1 = First, 2 = Second, 3 = Third). Range: 1-3.", 
        example=3
    )
    Sex: Literal["male", "female"] = Field(
        ..., 
        description="Gender of the passenger (male or female)", 
        example="male"
    )
    Age: confloat(ge=0, le=120) = Field(
        ..., 
        description="Age of the passenger in years. Typical range: 0-80.", 
        example=22
    )
    Fare: confloat(ge=0, le=5000) = Field(
        ..., 
        description="Ticket fare price paid by the passenger in British pounds. Typical range: 0-500 (most often under 100).", 
        example=7.25
    )

@app.post("/predict", tags=["Titanic Predictions"])
def predict(passenger: Passenger):
    # zamień na DataFrame
    df = pd.DataFrame([passenger.dict()])

    # preprocessing taki sam jak w treningu
    df = data_formater.format_sex(df)
    df["Sex"] = df["Sex"].astype(int)

    # predykcja
    pred = model.predict(df[features])[0]
    proba = model.predict_proba(df[features])[0][1]

    return {
        "prediction": int(pred),
        "probability_survived": float(proba)
    }