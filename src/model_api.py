import json
from typing import Optional, List
from enum import Enum
from loguru import logger
logger.add('logs.log')

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import mlflow

from settings import CREDIT_CARD_MODEL_NAME, TRACKING_URI


# Loading MLFlow Models
mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{CREDIT_CARD_MODEL_NAME}/production"
)


# SettingServing the API
app = FastAPI()
class ModelName(str, Enum):
    """Defining the Models Options.
    """
    creditCardDefault = CREDIT_CARD_MODEL_NAME

class CreditModelObservations(BaseModel):
    """Defining the Data Structure of observations 
    to be predicted with the CreditCardModel.
    """
    V1: List[float]
    V2: List[float]
    V3: List[float]
    V4: List[float]
    V5: List[float]
    V6: List[float]
    V7: List[float]
    V8: List[float]
    V9: List[float]
    V10: List[float]
    V11: List[float]
    V12: List[float]
    V13: List[float]
    V14: List[float]
    V15: List[float]
    V16: List[float]
    V17: List[float]
    V18: List[float]
    V19: List[float]
    V20: List[float]
    V21: List[float]
    V22: List[float]
    V23: List[float]
    V24: List[float]
    V25: List[float]
    V26: List[float]
    V27: List[float]
    V28: List[float]
    Amount: List[float]
    id: List[int]

@app.post("/models/prediction/{model_name}")
def invoke_prediction(model_name: ModelName, observation: CreditModelObservations):
    df_to_be_predicted = pd.read_json(observation.json()).drop(columns=['id'])
    prediction = model.predict(df_to_be_predicted)
    df_with_predictions = df_to_be_predicted
    df_with_predictions['prediction'] = prediction
    df_with_predictions

    return  df_with_predictions.to_dict('list')
