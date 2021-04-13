# import libs
import os
from loguru import logger
logger.add('logs.log', rotation = '5 MB', level="INFO")
import mlflow
from mlflow.tracking import MlflowClient

from settings import EXPERIMENT_NAME
from dao.CreditCardDefault import load_creditcard_dataset
from trainers.h2o_automl import H2OClassifier

logger.info('=========================')
logger.info('Setting MLFLOW Experiment')
mlflow.set_tracking_uri("sqlite:///mlruns.db")
try:
    experiment = mlflow.create_experiment(EXPERIMENT_NAME)
except:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

logger.info('-------------------------')
logger.info('Loading Credit Card Dataset')
dataset = load_creditcard_dataset()

logger.info('-------------------------')
logger.info('H2O AutoML Run')
list_max_run_time = [30, 60]
for i in range(len(list_max_run_time)):
    H2OClassifier(
        run_name = f'H2O_CreditCard{i+1}',
        max_mem_size = '3G',
        df = dataset,
        target_col = 'Class',
        sort_metric = 'aucpr',
        max_models = 8,
        max_runtime_secs = list_max_run_time[i],
        nfolds = 5,
        seed = 90
    )
logger.info('=========================')
