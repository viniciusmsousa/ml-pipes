from loguru import logger
logger.add('logs.log', rotation='2 MB')
import pandas as pd
from settings import PATH_CREDIT_CARD_DATASET

def load_creditcard_dataset():
    try:
        return pd.read_csv(PATH_CREDIT_CARD_DATASET)
    except Exception as e:
        logger.error(e)
        raise Exception(e)