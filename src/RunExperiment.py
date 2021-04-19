# Import Libs
from loguru import logger
logger.add('logs.log', rotation = '5 MB', level="INFO")
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType 

from settings import EXPERIMENT_NAME, FOLDS, CREDIT_CARD_MODEL_NAME, CHAMPION_METRIC
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
logger.info('-------------------------')

logger.info('Start Deploying Model')
## Getting The best Model according to AUC Metric
champion = MlflowClient().search_runs(
    experiment_ids=[str(mlflow.get_experiment_by_name(name=EXPERIMENT_NAME).experiment_id)],
    run_view_type=ViewType.ALL,
    order_by=[f"metrics.{CHAMPION_METRIC} DESC"],
    max_results=1
)
run_id = champion[0].info.run_id

## Registering it and setting it to production
model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=CREDIT_CARD_MODEL_NAME
)


MlflowClient().update_model_version(
    name=CREDIT_CARD_MODEL_NAME,
    version=model.version,
    description='Deploying model with model registery'
)

MlflowClient().transition_model_version_stage(
    name=CREDIT_CARD_MODEL_NAME,
    version=model.version,
    stage="Production"
)
logger.info('=========================')
