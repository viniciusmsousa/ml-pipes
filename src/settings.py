# Experiment Context
EXPERIMENT_NAME = "CreditCardDefault"
PATH_SAVE_PREDICTIONS = '../prediction_data/'
TRACKING_URI = 'http://127.0.0.1:5000'

# Model Contex
MODEL_LIFE_STAGES = {
    'production': 'production',
    'staging': 'staging'
}
THRESHOLD = 0.5
CHAMPION_METRIC = 'ks'
CREDIT_CARD_MODEL_NAME = EXPERIMENT_NAME
FOLDS = 5
