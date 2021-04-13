# ML-Pipes

An experimental project using MLFLOW in order to create a simple ML in production enviroment. Where the prduction model is automatically retrained once an event/criteria is verified to be true (WIP).


## Current State

Trains a H2O AutoML with the Credit Card dataset. Allows to serve the MLFLOW UI and the trained model as an API. This can be reproduced by:
- You have folder named `dao/data/creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), withou the column `Time`;
- Install the dependencies from the `pyproject.toml` file, see [poetry](https://python-poetry.org/), or using the requirements.txt;
- Also, since it is using the H2O is is necessary to have Java installed.

With that done it is now possible to train the models in an experiment with `cd src/` -> `python 01RunExperiment.py`. This will train the models. Once it is done you can run inside `src/` `mlflow ui --backend-store-uri sqlite:///mlruns.db` and access the localhost:5000 to use the UI. And finally, to serve the model as an API just run `sh DeploySh.sh`, also inside `src/`. Just be aware that before runninf the `DeployModel.sh` you need to stage the model to production.
