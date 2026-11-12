from typing import Optional, Dict, Any

import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

TARGET = "top20_spender"
MODEL_PATH = "logreg_best_model.joblib"
TRAIN_PATH = "결과물/dataset_train.csv"

app = FastAPI(title="Top 20 Spender Predictor API", version="1.0")

class PredictRequest(BaseModel):
    AGE_GRP: str
    JOB: str
    RESIDENCE: str
    DESTINATION: str
    ACCOMPANY: str
    TRIP_DAYS: int
    SIDO: str
    SGG: str

    threshold: Optional[float] = 0.5
    return_input_vector: Optional[bool] = False

try:
    train_df = pd.read_csv(TRAIN_PATH)
    X_cols = [c for c in train_df.columns if c != TARGET]
except Exception as e:
    raise RuntimeError(f"TRAIN PATH 로드 실패: {TRAIN_PATH}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"MODEL_PATH 로드 실패: {TRAIN_PATH}")

def build_input_row(X_cols, req:PredictRequest) -> pd.DataFrame:
    """
    X_cols: 학습에 사용된 전체 feature 컬럼명 리스트
    user_input: 사용자 입력값을 담은 딕셔너리
    return: 사용자 입력값을 기반으로 한 DataFrame (1행) - 컬럼 100% 동일
    """
    x = pd.DataFrame([{c: 0 for c in X_cols}])

    if "TRIP_DAYS" in x.columns:
        x.loc[0, "TRIP_DAYS"] = req.TRIP_DAYS

    pairs = [
        ("AGE_GRP", req.AGE_GRP),
        ("JOB", req.JOB),
        ("TRAVEL_STATUS_RESIDENCE", req.RESIDENCE),
        ("TRAVEL_STATUS_DESTINATION", req.DESTINATION),
        ("TRAVEL_STATUS_ACCOMPANY", req.ACCOMPANY),
        ("TRAVEL_LIKE_SIDO_1", req.SIDO),
        ("TRAVEL_LIKE_SGG_1", req.SGG)
    ]
        
    for col_prefix, value in pairs:
        col_name = f"{col_prefix}_{value}"
        if col_name in x.columns:
            x.loc[0, col_name] = 1
    return x

@app.get("/")
def root():
    return {"status":"ok", "message":"FastAPI server is running"}

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    if req.threshold is None:
        thr = 0.5
    else:
        thr = float(req.threshold)

    if not (0.0 < thr < 1.0):
        raise HTTPException(status_code=400, detail="threshold는 0rhk 1 tkdldudi gkqslek. 예: 0.5")
    
    try:
        X_one = build_input_row(X_cols, req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"입력 벡터 생성 실패: {e}")

    try:
        proba = float(model.predict_proba(X_one)[:,1][0])
        pred = int(proba >= thr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 예측 실패: {e}")

    result = {"proba":proba, "pred":pred}

    if req.return_input_vector:
        result["input_vector"] = X_one.iloc[0].to_dict()

    return result


