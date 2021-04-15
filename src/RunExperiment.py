# Import Libs
import os

from loguru import logger
logger.add('logs.log', rotation = '5 MB', level="INFO")
import mlflow
from mlflow.tracking import MlflowClient

from settings import EXPERIMENT_NAME, FOLDS
from dao.CreditCardDefault import load_creditcard_dataset
from trainers.h2o_automl import H2OClassifier
from trainers.pycaret import PycaretClassifier


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

logger.info('Startting H2O Run')
H2OClassifier(
    run_name = 'H2O',
    max_mem_size = '3G',
    df = dataset,
    target_col = 'Class',
    sort_metric = 'aucpr',
    max_models = 8,
    max_runtime_secs = 60,
    nfolds = FOLDS,
    seed = 90
)
logger.info('-------------------------')

logger.info('Pycaret Run')
PycaretClassifier(
        experiment_name = EXPERIMENT_NAME,
        run_name = 'Pycaret',
        sort_metric = 'precision',
        df = dataset,
        target = 'Class',
        n_best_models = 3,
        data_split_stratify = True,
        nfolds = FOLDS,
        normalize = True,
        transformation = True, 
        ignore_low_variance = True,
        remove_multicollinearity = True,
        multicollinearity_threshold = 0.95,
        session_id = 54321
)

logger.info('=========================')
