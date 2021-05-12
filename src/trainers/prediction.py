from typeguard import typechecked
import pandas as pd
import pyspark
from pyspark.ml.feature import VectorAssembler
import mlflow


@typechecked
def predictor(
    model: mlflow.pyfunc.PyFuncModel,
    spark: pyspark.sql.session.SparkSession,
    df: pd.DataFrame
):
    """Function that is called in the predoct end-point.
    Here is where the prediction output is normalised to
        be the same from different flavors.

    Args:
        model (mlflow.pyfunc.PyFuncModel): A loaded MLFlow Model
        spark (pyspark.sql.session.SparkSession): SparkSession.
        df (pd.DataFrame): Data frame to make predictions.

    Raises:
        Exception: Errors.

    Returns:
        list|array: List of predictions for each rows from df.
    """
    try:
        flavor = model\
            ._model_meta\
            .to_dict()['flavors']['python_function']['loader_module']

        if flavor == 'mlflow.spark':
            dfs_predict = spark.createDataFrame(df)
            assembler = VectorAssembler(
                inputCols=dfs_predict.columns,
                outputCol='features'
            )
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
