from loguru import logger
logger.add('logs.log', rotation='2 MB')
import pandas as pd
from settings import PATH_CREDIT_CARD_DATASET

def load_creditcard_dataset():
    try:
        df = pd.read_csv(f'{PATH_CREDIT_CARD_DATASET}creditcard.csv')
        df_nonevent = df.loc[df['Class']==0].sample(1500)
        df_event = df.loc[df['Class']==1].sample(250)
        return df_nonevent.append(df_event)
    except Exception as e:
        logger.error(e)
        raise Exception(e)

def write_predictions(df_predictions: pd.DataFrame):
    try:
        df_predictions.to_csv(
            f'{PATH_CREDIT_CARD_DATASET}creditcard_predictions.csv',
            index=False,
            decimal='.',
            sep=',',
            mode='a'
        )
    except Exception as e:
        logger.error(e)
        raise Exception(e)