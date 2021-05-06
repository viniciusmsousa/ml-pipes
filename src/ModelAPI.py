from datetime import datetime
import json
from typing import Optional, List
from loguru import logger
logger.add('../logs/logs.log', rotation = '5 MB', level="INFO")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum

import pandas as pd
import mlflow

from settings import CREDIT_CARD_MODEL_NAME, TRACKING_URI, MODEL_LIFE_STAGES    
from dao.CreditCardDefault import write_predictions


# Loading MLFlow Models
mlflow.set_tracking_uri(TRACKING_URI)

## Credit Card Models
try:
    credit_card_production_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{CREDIT_CARD_MODEL_NAME}/production"
    )
except:
    logger.info(f'No {CREDIT_CARD_MODEL_NAME} production model found.')
    pass

try:
    credit_card_staging_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{CREDIT_CARD_MODEL_NAME}/staging"
    )
except:
    logger.info(f'No {CREDIT_CARD_MODEL_NAME} staging model found.')
    pass


# SettingServing the API
app = FastAPI(title='Ml-Pipes')
class ModelName(str, Enum):
    """List of availiable models.
    """
    creditCardDefault = CREDIT_CARD_MODEL_NAME

class ModelLifeStage(str, Enum):
    """List of availiable model life stages.
    """
    production = MODEL_LIFE_STAGES['production']
    staging = MODEL_LIFE_STAGES['staging']

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

@app.post("/models/prediction/{model_name}/{model_life_stage}")
def invoke_prediction(
    model_name: ModelName,
    model_life_stage: ModelLifeStage,
    observation: CreditModelObservations
):
    df_to_be_predicted = pd.read_json(observation.json())

    if model_name == CREDIT_CARD_MODEL_NAME:
        if model_life_stage == MODEL_LIFE_STAGES['production']:
            try:
                prediction = credit_card_production_model.predict(df_to_be_predicted.drop(columns=['id']))
            except:
                raise HTTPException(status_code=404, detail=f'No {model_life_stage} of {model_name} model found.')
        else:
            try:
                prediction = credit_card_staging_model.predict(df_to_be_predicted.drop(columns=['id']))
            except:
                raise HTTPException(status_code=404, detail=f'No {model_life_stage} of {model_name} model found.')
        
        df_with_predictions = df_to_be_predicted
        if type(prediction)==pd.DataFrame:
            df_with_predictions['prediction'] = prediction.predict
        else:
            df_with_predictions['prediction'] = prediction
    else:
        pass

    df_with_predictions['date'] = str(datetime.today())[:19]
    df_with_predictions['model_life_stage'] = model_life_stage
    write_predictions(df_with_predictions)
    
    return  df_with_predictions[['id', 'prediction']].to_dict('list')
