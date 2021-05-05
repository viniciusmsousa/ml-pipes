# ML-Pipes

This is a minimal ML project using the MLFlow Tracking and Model Registery Modules. The objetive is to provide an simple and self contained project, but not exhaustedly detailed. Note that this project was built to be executed locally as a way to grasp the main functionalities of MLFlow, the objective is not to provide a production ready framework. 


## Project Structure

The tree bellow displays the project strucuture. Note that the main file where MLFlow is put into action are the `trainers/` where diferente ML framework are use with MLFlow context in order to train models and `RunExperiment.py` where the experiment is configured.  

```
ml-pipes
├─ .dockerignore
├─ .gitignore
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ docker-compose.yml
├─ poetry.lock
├─ pyproject.toml
├─ requirements.txt
└─ src
   ├─ RunExperiment.py
   ├─ dao
   │  └─ CreditCardDefault.py
   ├─ settings.py
   └─ trainers
      ├─ h2o_automl.py
      └─ pycaret.py
```

## Reproduce the Project

The following steps explain how to reproduce this project and by the end of it you should have understand the main funcionalities of MLFlow. In order to do so, the experiment will train different models to predict credit card default and put to production the one with the best performance.


### 1) Prepare the Enviroment

Before start running any code make sure that the following requisites are meet:

- Create the file `src/dao/data/creditcard.csv` by downloading teh file from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and dropping the column `Time`;
- Install the dependencies from the `pyproject.toml` file, see [poetry](https://python-poetry.org/), or using from the requirements.txt;
- Have Java installed since the example model is using H2O Package. If you running on ubuntu 20.04, simply install it by running the command in terminal `sudo apt-get install -y openjdk-11-jre`.

### 2) Runing the Experiment

MLFlow is built upon the concept of experiments. Where each experiment a series os run, each run being a candidate model. The script `src/RunExperiment.py` contain the code to run an H2O AutoML algorithm and the PyCaret classification module algorithms and set to production stage ([MLFlow Model Regitsry Concepts](https://www.mlflow.org/docs/latest/model-registry.html#concepts)). Feel free to go through the script and understand it, the script bassicaly:

1) Setup and MLFLow Experiemnt;
2) Load the Credit Card dataset;
3) Trains an H2O AutoML;
4) Trains sklearn classifiers, using PyCaret interface;
5) Select the best model according to the CHAMPION_METRIC;
6) Set the best model the production stage.

In order to run the experiment you can simply execute inside the folder `src/` the following command:
```shell
python RunExperiment.py
```

Once the experiment finished running you can deploy the [MLFlow Tracking UI](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) locally by running in the terminal inside the `src/` folder the command 
```shell
mlflow ui -p 5000 --backend-store-uri sqlite:///mlruns.db
```

and deploy the best model using the [MLFlow Model Registery](https://www.mlflow.org/docs/latest/model-registry.html) by running

```shell
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db" &&

mlflow models serve -p 5001 --model-uri models:/CreditCardDefault/production --no-conda
```

Now you should be able to access both services in the localhost ports.

### 3) 'Dockering'

In order to have a more robust way to serve the model API and Tracking UI a good ideia is to use docker. This is done in through the `Dockerfile` and `docker-compose.yml`. To start the UI and Api simple execute the following command in the root project directory
```
docker-compose up
```

When you run for the first time this should take a while, since it will build the docker image and then deploy the containers. If everything was done correctly, then you have acess in port 5000 the UI tracking and in 5001 the production model, but now serving each service as a separated container.

