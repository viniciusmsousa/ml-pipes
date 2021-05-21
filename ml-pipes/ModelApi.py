import os
from datetime import datetime
from typing import List

from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from enum import Enum
from dotenv import load_dotenv

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient



# Credentials Setup
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'minio_access_key'
os.environ['AWS_ACCESS_KEY_ID'] = 'MINIO_SECRET_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'http://127.0.0.1:9000'


# MLFlow Setup
TRACKING_URI = 'http://127.0.0.1:5000'
client = MlflowClient(tracking_uri=TRACKING_URI) 
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)


# Function Definitions (temporaly for now here)
from typeguard import typechecked
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import mlflow

@typechecked
def init_spark_session(max_mem_size: int = 4, n_cores: int = 4):
    """Initialize a local Spark Session.

    Args:
        max_mem_size (int, optional): Max Memory in GB to be allocated
        to Spark. Defaults to 4.
        n_cores (int, optional): Cores to be allocated to Spark. Defaults to 4.

    Raises:
        Exception: All Erros.

    Returns:
        Spark Session Object: Spark Session Object
    """
    try:
        mem_size = f'{max_mem_size}G'

        spark = SparkSession.builder\
            .appName('Ml-Pipes') \
            .master(f'local[{n_cores}]') \
            .config('spark.executor.memory', mem_size) \
            .config('spark.driver.memory', mem_size) \
            .config('spark.memory.offHeap.enabled', 'true') \
            .config('spark.memory.offHeap.size', mem_size) \
            .getOrCreate()
        return spark
    except Exception as e:
        raise Exception(e)

@typechecked
def predictor(
    model: mlflow.pyfunc.PyFuncModel,
    spark: pyspark.sql.session.SparkSession,
    df: pd.DataFrame
):
    """Function that is called in the predoct end-point.
    Here is where the prediction output is normalised to
        be the same from different flavors.

    Args:
        model (mlflow.pyfunc.PyFuncModel): A loaded MLFlow Model
        spark (pyspark.sql.session.SparkSession): SparkSession.
        df (pd.DataFrame): Data frame to make predictions.

    Raises:
        Exception: Errors.

    Returns:
        list|array: List of predictions for each rows from df.
    """
    try:
        flavor = model\
            ._model_meta\
            .to_dict()['flavors']['python_function']['loader_module']

        if flavor == 'mlflow.spark':
            dfs_predict = spark.createDataFrame(df)
            assembler = VectorAssembler(
                inputCols=dfs_predict.columns,
                outputCol='features'
            )
            dfs_predict = assembler.transform(dfs_predict)
            predictions = model.predict(dfs_predict.toPandas())

        elif flavor == 'mlflow.h2o':
            predictions = model.predict(df).predict
        elif flavor == 'mlflow.sklearn':
            predictions = model.predict(df)
        else:
            pass

        return predictions
    except Exception as e:
        raise Exception(e)




# Spark Session init
spark = init_spark_session()

# Loading Models
model_names = list()
for model in client.list_registered_models():
    model_names.append(dict(model)['name'])

models_dict = dict()
for m in model_names:
    models_dict[m] = dict()
    try:
        models_dict[m]['production'] = mlflow.pyfunc.load_model(model_uri=f'models:/{m}/production')
    except Exception as e:
        print(f'No production {m} model found.')
    try:
        models_dict[m]['staging'] = mlflow.pyfunc.load_model(model_uri=f'models:/{m}/staging')
    except Exception as e:
        print(f'No staging {m} model found.') 

# Serving Model Through API
app = FastAPI(title='Ml-Pipes')

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


