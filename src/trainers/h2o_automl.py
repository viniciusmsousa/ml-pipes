from typeguard import typechecked
from loguru import logger
logger.add('logs.log')
import pandas as pd
import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
import h2o
from h2o.automl import H2OAutoML


class H2OClassifier:
    """
    H2O AutoML Classifier
    """
    @typechecked
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        sort_metric: str,
        threshold: float= 0.5,
        run_name: str = 'h2o_automl_classifier',
        max_mem_size: str = '3G',
        max_models: int = 10,
        max_runtime_secs: int = 60,
        nfolds: int = 5,
        seed: int = 90
    ):
        """Wrapper for the H2O AutoMl Algo. It instanciates a H2O enviroment to run AutoMl.

        Args:
            df (pd.DataFrame): Dataset with target and feature.
            target_col (str): Target column to be predicted.
            sort_metric (str): H2O Metric to be used to select the best model.
            run_name (str, optional): MLFlow run name. Defaults to 'h2o_automl_classifier'.
            max_mem_size (str, optional): Max memory to the allocated to H2O. Defaults to '3G'.
            max_models (int, optional): Number max of models o be fitted. Defaults to 10.
            max_runtime_secs (int, optional): Max run time, in seconds. Defaults to 60.
            nfolds (int, optional): Number of folds for cross validation. Defaults to 5.
            seed (int, optional): Seed. Defaults to 90.
        """
        # Params Values
        self.run_name = run_name
        self.max_mem_size = max_mem_size
        self.threshold = threshold
        self.df = df
        self.target_col = target_col
        self.sort_metric = sort_metric
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.nfolds = nfolds
        self.seed = seed

        # 1) Starting H2O
        self.start_h2o()
        
        # 2) Getting Features Cols
        self.feature_cols = self.get_feature_cols()
        
        # 3) Spliting into Train and Valid
        self.train, self.valid = self.train_valid_split()

        # 4) Training Model
        self.run_info = self.mlflow_run()


    def start_h2o(self):
        try:
            h2o.init(max_mem_size=self.max_mem_size)
        except Exception as e:
            logger.error(e)
            raise Exception(e)

    def get_feature_cols(self):
        try:
           return list(self.df.columns[self.df.columns != self.target_col])
        except Exception as e:
            logger.error(e)
            raise Exception(e)

    def train_valid_split(self):
        try:
            train, valid = h2o.H2OFrame(self.df).split_frame(ratios=[0.7])
            train[self.target_col] = train[self.target_col].asfactor()
            valid[self.target_col] = valid[self.target_col].asfactor()
            return train, valid
        except Exception as e:
            logger.error(e)
            raise Exception(e)

    def mlflow_run(self):
        try:
            with mlflow.start_run(run_name=self.run_name) as run:
                ## 1) Training Model
                #https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
                model = H2OAutoML(
                    sort_metric = self.sort_metric,
                    max_models = self.max_models, 
                    max_runtime_secs = self.max_runtime_secs,
                    nfolds = self.nfolds,
                    seed = self.seed,
                )
                model.train(
                    x = self.feature_cols,
                    y = self.target_col, 
                    training_frame = self.train,
                    validation_frame = self.valid
                )

                ## 2) Logging Params
                mlflow.log_param('sort_metric', self.sort_metric)

                ## 3) Logging Metrics
                # https://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html#classification
                # http://h2o-release.s3.amazonaws.com/h2o/master/3259/docs-website/h2o-py/docs/h2o.metrics.html
                mlflow.log_metric('accuracy', model.leader.accuracy(valid=True, thresholds=[self.threshold])[0][1])
                mlflow.log_metric("f1", model.leader.F1(valid=True)[0][0])
                mlflow.log_metric("accuracy", model.leader.accuracy(valid=True)[0][0])
                mlflow.log_metric("aucpr", model.leader.aucpr(valid=True))
                mlflow.log_metric("auc", model.leader.auc(valid=True))
                mlflow.log_metric("ks", model.leader.kolmogorov_smirnov())

                ## 4) Logging Artifacts
                # How to log leaderboard?
                #    get_leaderboard(model.leaderboard, extra_columns='ALL')
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
                
                ## 5) Logging Model
                mlflow.h2o.log_model(model.leader, "model")

                return run.info 

        except Exception as e:
            logger.error(e)
            raise Exception(e)
