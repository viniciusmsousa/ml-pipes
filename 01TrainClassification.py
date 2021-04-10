# import libs
from datetime import datetime
today = f"{datetime.today().year}-{datetime.today().month}-{datetime.today().day}" 

import pandas as pd
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import mlflow
import mlflow.h2o
#from mlflow.tracking import MlflowClient
from loguru import logger
logger.add('logs/logs.log', rotation = '20 MB', level="INFO")

from settings import EXPERIMENT_NAME, H2O_MAX_MEM_SIZE, CLASSIFICATION_SORT_METRIC_H2O

logger.info('=========================')
logger.info('Setting MLFLOW Experiment')
experiment = mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info('-------------------------')


logger.info('=========================')
logger.info('Startting H2O Runs')
h2o.init(max_mem_size=H2O_MAX_MEM_SIZE)

logger.info('Importing Data')
train, valid = h2o.import_file('data/creditcard.csv').split_frame(ratios=[0.7])
train['Class'] = train['Class'].asfactor()
valid['Class'] = valid['Class'].asfactor()

logger.info('Training  AutoML')
# Model Run
for max_runtime_secs in [10, 15]:
    run_name = f'{today}_folds_{max_runtime_secs}'
    log_file = f'data/{run_name}.log'
    logger.add(log_file)
    with mlflow.start_run(run_name=run_name):
        #https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
        model = H2OAutoML(
            max_models = 10, 
            max_runtime_secs = max_runtime_secs,
            seed = 24,
            nfolds = 5,
            sort_metric = CLASSIFICATION_SORT_METRIC_H2O
        )
        model.train(
            x = train.columns[:-1],
            y = train.columns[-1:][0], 
            training_frame = train,
            validation_frame = valid
        )

        mlflow.log_param('max_runtime_secs', max_runtime_secs)
        # https://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html#classification
        # http://h2o-release.s3.amazonaws.com/h2o/master/3259/docs-website/h2o-py/docs/h2o.metrics.html
        mlflow.log_metric("F1", model.leader.F1(valid=True)[0][0])
        mlflow.log_metric("aucpr", model.leader.aucpr(valid=True))
        mlflow.log_metric("ks", model.leader.kolmogorov_smirnov())

        
        try:
            rows = model.leader.model_performance(valid).confusion_matrix(metrics=['f1']).to_list()
            cm = pd.DataFrame(columns=['Pred: 0', 'Pred: 1'], data = rows)\
                .rename(index = {0: 'Act: 0', 1: 'Act: 1'})
            
            logger.info(f'\nConfusion matrix Max F1:\n{cm}')
            mlflow.log_artifact(log_file)
        except:
            pass

        try:
            model.model_correlation_heatmap(valid).savefig('data/model_correlation.png')
            mlflow.log_artifact("data/model_correlation.png")
        except:
            pass

        try:
            model.varimp_plot(valid).savefig('data/varimp_plot.png')
            mlflow.log_artifact("data/varimp_plot.png")
        except:
            pass

        mlflow.h2o.log_model(model.leader, "model")

logger.add('logs/logs.log', rotation = '20 MB', level="INFO")
logger.info('-------------------------')

logger.info('Training Completed')


# https://varhowto.com/install-miniconda-ubuntu-20-04/
# /home/luana/miniconda3/bin/conda
# export MLFLOW_CONDA_HOME="/home/luana/miniconda3/"
# mlflow models serve -m runs:/bb16aac3de584a5db3beae52dd7bb2ca/model
#lsof -i :5000


