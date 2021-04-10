import os
from loguru import logger
logger.add('logs/deploy.log', rotation = '5 MB')
import pandas as pd
import mlflow


# mlflow models serve -m runs:/bb16aac3de584a5db3beae52dd7bb2ca/model


best_model_id = mlflow.search_runs(experiment_ids=[str(i) for i in range(1,3)], order_by=['metrics.aucpr DESC'])['run_id'][0]


os.system('export MLFLOW_CONDA_HOME="/home/luana/miniconda3/"')

os.system(f'mlflow models serve -m runs:/{best_model_id}/model -p 5001')