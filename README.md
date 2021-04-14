# ML-Pipes

A minimal ML project to deploy a model with MLFLOW using docker. 

## Reproduce

In order to reproduce this project go through the following steps

### 1) Prepare the Enviroment

Make sure that the following bullets are meet:

- The following file `src/dao/data/creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud), withou the column `Time`;
- Install the dependencies from the `pyproject.toml` file, see [poetry](https://python-poetry.org/), or using the requirements.txt;
- Have JRE installed since the example model is using H2O Package.

### 2) Run the Experiment(s)

You can simply execute inside the folder `src/` the following command
```
python RunExperiment.py
```
This will train two models, fell free to check the script and play around. Onde it is finished MLFLOW will have created a `mlruns/` and `mlruns.db` inside src. These are the files needed in order to lauch the tracking UI and to serve the model as an API. 

Make sure that once the experiment is finished you run the UI locally, executing `mlflow ui --host 0.0.0.0 -p 5000 --backend-store-uri sqlite:///mlruns.db` inside `src/`, and [register a model in the model registery](https://www.mlflow.org/docs/latest/model-registry.html#ui-workflow) with the name `TestCreditCard` (or change the name in the `docker-compose.yml` file). 

Now you are all set to deploy it using docker.

### 3) Deploy

Just execute in the root project folder. 
```
docker-compose up
```

If everything was done correctely, then you have acess in port 5000 the UI tracking and in 5001 the production model.

## Current State

Trains a H2O AutoML with the Credit Card dataset. Allows to serve the MLFLOW UI and the trained model as an API. This can be reproduced by:


With that done it is now possible to train the models in an experiment with `cd src/` -> `python 01RunExperiment.py`. This will train the models. Once it is done you can run inside `src/` `mlflow ui --backend-store-uri sqlite:///mlruns.db` and access the localhost:5000 to use the UI. And finally, to serve the model as an API just run `sh DeploySh.sh`, also inside `src/`. Just be aware that before runninf the `DeployModel.sh` you need to stage the model to production.
