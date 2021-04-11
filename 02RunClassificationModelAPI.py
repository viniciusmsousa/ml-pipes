import os


from loguru import logger
logger.add('logs/logs.log', rotation = '5 MB')
import pandas as pd
import mlflow
from settings import EXPERIMENT_NAME, CLASSIFICATION_SORT_METRIC_H2O, MLFLOW_CONDA_HOME

# logger.info('Start MLFLOW UI')
# ui = Popen('mlflow ui -p 5000', shell = True)
# ui.wait()

logger.info(f'Getting Best Run from Experiment {EXPERIMENT_NAME}, sorted by {CLASSIFICATION_SORT_METRIC_H2O}')
best_model_id = mlflow.search_runs(experiment_ids=[str(i) for i in range(1,3)], order_by=['metrics.aucpr DESC'])['run_id'][0]

logger.info('Startting API')
os.system(f'export MLFLOW_CONDA_HOME="{MLFLOW_CONDA_HOME}"')
os.system(f'mlflow models serve -m runs:/{best_model_id}/model -p 5001')
