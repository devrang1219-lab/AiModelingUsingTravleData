from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Top20 Spender Precition API")

model = joblib.load("logreg_best_model.joblib")

class PredictInput(BaseModel):
    TRIP_DAYS:int
    TRAVEL_NUM: int
    INCOME: float
    HOUSE_INCOME: float

@app.post("/predict")
def predict(data:PredictInput)
    df = pd.DataFrame([data.dict()])

    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)

    return {
        "pred_label" : pred,
        "pred_proba" : round(float(proba), 4)
    }