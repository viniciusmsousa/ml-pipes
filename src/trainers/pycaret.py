import os
from typeguard import typechecked
from loguru import logger
logger.add('logs.log')
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pycaret.classification as PycaretClassifierModule


class PycaretClassifier:
    """
    Pycaret Classifier within a MLFLOW Run Context
    """
    @typechecked
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        sort_metric: str,
        df: pd.DataFrame,
        target: str,
        n_best_models: int = 3,
        data_split_stratify: bool =True,
        nfolds: int = 5,
        normalize: bool = True,
        transformation: bool = True, 
        ignore_low_variance: bool = True,
        remove_multicollinearity: bool = True,
        multicollinearity_threshold: float = 0.95,
        session_id: int = 54321
    ):
        # Params Values
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.sort_metric = sort_metric
        self.df = df
        self.target = target
        self.n_best_models = n_best_models
        self.data_split_stratify = data_split_stratify
        self.nfolds = nfolds
        self.normalize = normalize
        self.transformation = transformation
        self.ignore_low_variance = ignore_low_variance
        self.remove_multicollinearity = remove_multicollinearity
        self.multicollinearity_threshold = multicollinearity_threshold
        self.session_id = session_id

        # 1) Run Pycaret Classifier
        self.runs_status = self.mlflow_runs()

    def mlflow_runs(self):
        try:
            ## 1) Setup the Experiment
            experiment = PycaretClassifierModule.setup(
                experiment_name = self.experiment_name,
                silent = True,
                data = self.df,
                target = self.target,
                data_split_stratify = self.data_split_stratify,
                # Changes the magnitude of features (features with values in similar order)
                normalize = self.normalize,
                # Changes the format of distribution
                transformation = self.transformation, 
                ignore_low_variance = self.ignore_low_variance,
                remove_multicollinearity = self.remove_multicollinearity, 
                multicollinearity_threshold = self.multicollinearity_threshold,
                session_id = self.session_id,
                verbose = False
            )

            ## 2) Fit Classifiers
            best_models = PycaretClassifierModule.compare_models(
                n_select=self.n_best_models, 
                sort=self.sort_metric,
                fold=self.nfolds
            )
            for i in range(len(best_models)):
                ## 3) Getting Model and Metrics
                model = best_models[i]
                df_predict = PycaretClassifierModule.predict_model(model, verbose=False)
                df_metrics = PycaretClassifierModule.pull()
                df_metrics

                ## 4) MLFlow Logs
                with mlflow.start_run(run_name=f"pycaret_{self.sort_metric}_{i}") as run:
                    mlflow.log_param('sort_metric', self.sort_metric)
                    mlflow.log_param('model', df_metrics['Model'][0])


                    mlflow.log_metric("accuracy", df_metrics['Accuracy'][0])
                    mlflow.log_metric("auc", df_metrics['AUC'][0])
                    mlflow.log_metric("recall", df_metrics['Recall'][0])
                    mlflow.log_metric("precision", df_metrics['Prec.'][0])
                    mlflow.log_metric("f1", df_metrics['F1'][0])
                    mlflow.log_metric("kappa", df_metrics['Kappa'][0])
                    mlflow.log_metric("mcc", df_metrics['MCC'][0])
                    mlflow.sklearn.log_model(model, "model") 
                
                    try:
                        PycaretClassifierModule.plot_model(model,plot='pr', save=True)
                        mlflow.log_artifact("Precision Recall.png")
                        os.remove('Precision Recall.png')
                    except:
                        pass

                    try:
                        PycaretClassifierModule.plot_model(model,plot='confusion_matrix', save=True)
                        mlflow.log_artifact("Confusion Matrix.png")
                        os.remove('Confusion Matrix.png')
                    except:
                        pass

            return 200

        except Exception as e:
            logger.error(e)
            raise Exception(e)
