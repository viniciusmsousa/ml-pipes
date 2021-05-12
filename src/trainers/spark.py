from typeguard import typechecked
import pandas as pd

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier

import mlflow
import mlflow.spark

from ds_toolbox.statistics import ks_test


@typechecked
def classification_metrics(dfs_prediction: pyspark.sql.dataframe.DataFrame, col_target: str, print_metrics: bool = False):
    """Function to compute a few binary classification metrics from a spark df.

    Args:
        dfs_prediction (pyspark.sql.dataframe.DataFrame): SparkDataFrame with label and prediction columns.
        col_target (str): Label colum name.
        print_metrics (bool, optional): Whether or not to print the metrics. Defaults to False.

    Raises:
        Exception: Erros.

    Returns:
        dict: Dict with metrics.
    """
    try:
        # Confusion Matrix
        confusion_matrix = dfs_prediction.groupBy(col_target, "prediction").count()
        TN = dfs_prediction.filter(f'prediction = 0 AND {col_target} = 0').count()
        TP = dfs_prediction.filter(f'prediction = 1 AND {col_target} = 1').count()
        FN = dfs_prediction.filter(f'prediction = 0 AND {col_target} = 1').count()
        FP = dfs_prediction.filter(f'prediction = 1 AND {col_target} = 0').count()

        # Computing Metrics from Confusion Matrix
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall/(precision+recall)) if (precision+recall) > 0 else 0

        # PySpark (2.3.4) Evaluation Metrics
        evaluator = BinaryClassificationEvaluator(labelCol=col_target)
        auroc = evaluator.evaluate(dfs_prediction)

        # digio_ds_toolkit lib
        df_predictions = dfs_prediction.rdd.map(lambda row: (row.prediction, ) + tuple(row.probability.toArray().tolist())).toDF(["prediction"])
        df_predictions = df_predictions.withColumnRenamed('_3','prob_1')
        df_predictions = df_predictions.select('prediction', 'prob_1').toPandas()
        ks_dict = ks_test(df=df_predictions, col_target='prediction', col_probability='prob_1')

        # Results Dict
        out_dict = {
            'confusion_matrix': confusion_matrix.toPandas(),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auroc': auroc,
            'ks': ks_dict
        }
        
        if print_metrics is True:
            print(f'accuracy:  {round(out_dict["accuracy"], 4)}')
            print(f'precision: {round(out_dict["precision"], 4)}')
            print(f'recall:    {round(out_dict["recall"], 4)}')
            print(f'f1:        {round(out_dict["f1"], 4)}')
            print(f'auroc:     {round(out_dict["auroc"], 4)}')
            print(f'max_ks:    {round(out_dict["ks"]["max_ks"], 4)}')
        
        return out_dict
    except Exception as e:
        raise Exception(e)

@typechecked
def init_spark_session(max_mem_size: int = 4, n_cores: int = 4):
    """Initialize a local Spark Session.

    Args:
        max_mem_size (int, optional): Max Memory in GB to be allocated to Spark. Defaults to 4.
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
            .config('spark.memory.offHeap.enabled','true') \
            .config('spark.memory.offHeap.size',mem_size) \
            .getOrCreate()
        return spark
    except Exception as e:
        raise Exception(e)

class SparkClassifier:
    """
    Spark Classifiers. Runs locally.
    """
    @typechecked
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        run_name: str = 'spark_classifier',
        max_mem_size: int = 4,
        n_cores: int = 4,
        seed: int = 90
    ):
        """Wrapper fits DecisionTreeClassifier, 
            RandomForestClassifier, LogisticRegression and GBTClassifier.

        Args:
            df (pd.DataFrame): Dataset with target and feature.
            target_col (str): Target column to be predicted.
            run_name (str, optional): MLFlow run name. Defaults to 'spark_classifier'.
            max_mem_size (str, optional): Max memory to the allocated to Spark. Defaults to '4G'.
            n_cores (str, optional): Number of cores to be used. Defaults to 4.
            seed (int, optional): Seed. Defaults to 90.
        """
        # Params Values
        self.run_name = run_name
        self.max_mem_size = max_mem_size
        self.n_cores = n_cores
        self.df = df
        self.target_col = target_col
        self.seed = seed

        # 1) Starting Spark Session
        self.spark = init_spark_session(n_cores=self.n_cores, max_mem_size=self.max_mem_size)
        
        # 2) Getting Features Cols
        self.feature_cols = self.get_feature_cols()
        
        # 3) Getting spark dataframe
        self.dfs_train, self.dfs_test = self.transform_spark()

        self.classfiers = self.classfiers()

        self.shutdown_spark()

    def get_feature_cols(self):
        try:
           return list(self.df.columns[self.df.columns != self.target_col])
        except Exception as e:
            raise Exception(e)

    def transform_spark(self):
        try:
            dfs = self.spark.createDataFrame(self.df)
            assembler = VectorAssembler(inputCols=self.feature_cols, outputCol='features')
            dfs = assembler.transform(dfs)
            dfs_train, dfs_test = dfs.randomSplit([0.8, 0.2], seed=self.seed)
            return dfs_train, dfs_test
        except Exception as e:
            raise Exception(e)


    def classfiers(self):
        try:
            lr = LogisticRegression(labelCol=self.target_col, featuresCol='features')
            lr_model = lr.fit(self.dfs_train)
            
            dt = DecisionTreeClassifier(labelCol=self.target_col, featuresCol='features')
            dt_model = dt.fit(self.dfs_train)

            rf = RandomForestClassifier(labelCol=self.target_col, featuresCol='features')
            rf_model = rf.fit(self.dfs_train)

            gbt = GBTClassifier(labelCol=self.target_col, featuresCol='features')
            gbt_model = gbt.fit(self.dfs_train)

            names = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GBTClassifier']
            models = [lr_model, dt_model, rf_model, gbt_model]
            for name, model in zip(names, models):
                prediction = model.transform(self.dfs_test)
                metrics = classification_metrics(dfs_prediction=prediction, col_target=self.target_col, print_metrics=False)

                with mlflow.start_run(run_name=f'{self.run_name}_{name}') as run:
                    mlflow.log_metric("f1", metrics["f1"])
                    mlflow.log_metric("auc", metrics["auroc"])
                    mlflow.log_metric('accuracy', metrics["accuracy"])
                    mlflow.log_metric("precision", metrics["precision"])
                    mlflow.log_metric("recall", metrics["recall"]) 
                    mlflow.log_metric("ks", metrics["ks"]["max_ks"])

                    mlflow.spark.log_model(model, "model")

            return 200
        except Exception as e:
            raise Exception(e)

    def shutdown_spark(self):
        self.spark.stop()
