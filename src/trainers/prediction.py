import pandas as pd
import pyspark
from pyspark.ml.feature import VectorAssembler
import mlflow


def predictor(model: mlflow.pyfunc.PyFuncModel, spark: pyspark.sql.session.SparkSession, df: pd.DataFrame):
    try:        
        flavor = model._model_meta.to_dict()['flavors']['python_function']['loader_module']

        if flavor == 'mlflow.spark':
            dfs_predict = spark.createDataFrame(df)
            assembler = VectorAssembler(inputCols=dfs_predict.columns, outputCol='features')
            dfs_predict = assembler.transform(dfs_predict)
            predictions = model.predict(dfs_predict.toPandas())
            
        elif flavor == 'mlflow.h2o':
            predictions = model.predict(df).predict
        elif flavor == 'mlflow.sklearn':
            predictions = model.predict(df)
        else:
            pass
        
        return predictions
    except Exception as e:
        raise Exception(e)
