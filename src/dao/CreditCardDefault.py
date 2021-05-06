import os
import pandas as pd
from settings import PATH_CREDIT_CARD_DATASET, PATH_SAVE_PREDICTIONS

def load_creditcard_dataset():
    try:
        df = pd.read_csv(f'{PATH_CREDIT_CARD_DATASET}creditcard.csv')
        df_nonevent = df.loc[df['Class']==0].sample(1500)
        df_event = df.loc[df['Class']==1].sample(250)
        return df_nonevent.append(df_event)
    except Exception as e:
        raise Exception(e)

def write_predictions(df_predictions: pd.DataFrame):
    file = f'creditcard_predictions.csv'
    try:
        if os.path.exists(PATH_SAVE_PREDICTIONS) == False:
            os.mkdir(PATH_SAVE_PREDICTIONS)
            df_predictions.to_csv(
                f'{PATH_SAVE_PREDICTIONS}{file}',
                index=False,
                decimal='.',
                sep=',',
                mode='w'
            )
        else:
            df_predictions.to_csv(
                f'{PATH_SAVE_PREDICTIONS}{file}',
                index=False,
                decimal='.',
                sep=',',
                mode='a',
                header=False
            )

    except Exception as e:
        raise Exception(e)