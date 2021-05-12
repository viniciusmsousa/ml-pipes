import pandas as pd
import pyspark
from pyspark.ml.feature import VectorAssembler
import mlflow


def predictor(model: mlflow.pyfunc.PyFuncModel, spark: pyspark.sql.session.SparkSession, df: pd.DataFrame):
    try:
        df_with_predictions = df.copy()

        flavor = model._model_meta.to_dict()['flavors']['python_function']['loader_module']
        
        if flavor == 'mlflow.spark':
            dfs_predict = spark.createDataFrame(df.drop('Class', axis=1))
            assembler = VectorAssembler(inputCols=dfs_predict.columns, outputCol='features')
            dfs_predict = assembler.transform(dfs_predict)
            df_with_predictions['prediction'] = model.predict(dfs_predict.toPandas())

        elif flavor == 'mlflow.h2o':
            df_with_predictions['prediction'] = model.predict(df.drop('Class', axis=1)).prediction
        
        elif flavor == 'mlflow.sklearn':
            df_with_predictions['prediction'] = model.predict(df.drop('Class', axis=1))
        else:
            pass
        
        return df_with_predictions
    except Exception as e:
        raise Exception(e)
