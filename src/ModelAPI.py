from datetime import datetime
from typing import List

from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from enum import Enum

import pandas as pd
import mlflow

from trainers.spark import init_spark_session
from trainers.prediction import predictor
from settings import CREDIT_CARD_MODEL_NAME, \
    TRACKING_URI, MODEL_LIFE_STAGES  # pylint: disable=import-error
from dao.CreditCardDefault \
    import write_predictions  # pylint: disable=import-error

logger.add('../logs/logs.log', rotation='5 MB', level="INFO")

# Seting MLFlow
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Spark Session
spark = init_spark_session()


# Loading Models
try:
    credit_card_production_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{CREDIT_CARD_MODEL_NAME}/production"
    )
except Exception as e:
    logger.error(f'No {CREDIT_CARD_MODEL_NAME} production model found.')
    logger.error(e)

try:
    credit_card_staging_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{CREDIT_CARD_MODEL_NAME}/staging"
    )
except Exception as e:
    logger.error(f'No {CREDIT_CARD_MODEL_NAME} production model found.')
    logger.error(e)

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


# Root End Point
@app.get('/')
def hello():
    return 'Hello! Your ML Api is up and running :)'


# Defining List models end point
@app.get('/models/list')
def list_available_models():
    """Lists availiable models to be invoked.
    """
    dict_models = dict()
    for model in client.list_registered_models():
        d = dict(model)
        d = {key: d[key] for key in ['name', 'description']}
        d = {d['name']: d['description']}
    dict_models.update(d)
    return dict_models


# Defining Prediction end point
@app.post("/models/prediction/{model_name}/{model_life_stage}")
def make_predictions(
    model_name: ModelName, model_life_stage: ModelLifeStage,
    observation: CreditModelObservations
):
    """Makes the predictions using the selected model.

    Raises:
        HTTPException: Status 404 if selected
        model life stage is not availiable.
    """
    df = pd.read_json(observation.json())

    if model_name == CREDIT_CARD_MODEL_NAME:
        if model_life_stage == MODEL_LIFE_STAGES['production']:
            try:
                predictions = predictor(
                    model=credit_card_production_model,
                    spark=spark,
                    df=df.drop(columns=['id'])
                )
            except Exception:
                raise HTTPException(
                    status_code=404,
                    detail=f'{model_life_stage} of {model_name} not found.'
                )
        else:
            try:
                predictions = predictor(
                    model=credit_card_staging_model,
                    spark=spark,
                    df=df.drop(columns=['id'])
                )
            except Exception:
                raise HTTPException(
                    status_code=404,
                    detail=f'{model_life_stage} of {model_name} not found.'
                )

    df['prediction'] = predictions
    df['date'] = str(datetime.today())[:19]
    df['model_life_stage'] = model_life_stage
    write_predictions(df)

    return df[['id', 'prediction']].to_dict('list')
