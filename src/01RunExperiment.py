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
params_list = [30, 60]
for i in range(len(params_list)):
    H2OClassifier(
        run_name = f'H2O_CreditCard{i+1}',
        max_mem_size = '3G',
        df = dataset,
        target_col = 'Class',
        sort_metric = 'aucpr',
        max_models = 8,
        max_runtime_secs = params_list[i],
        nfolds = 5,
        seed = 90
    )
logger.info('=========================')

# https://varhowto.com/install-miniconda-ubuntu-20-04/
# /home/luana/miniconda3/bin/conda
# export MLFLOW_CONDA_HOME="/home/luana/miniconda3/"
# mlflow models serve -m runs:/bb16aac3de584a5db3beae52dd7bb2ca/model
