# ML-Pipes

Ml-Pipes is named simply after a pipe, which is a tube of metal, plastic, or other material used to convey water, gas, oil, or other fluid substances. Here the 'substance' that is conveyed is not a fluid matter, but a machile learning model. 

From experience, the steeper part of the Ml-Ops learning curve happens in the transition from running models in a given environment (local machine, jupyter server in a cloud provider, etc.) to keeping tracking of different experiments, deploying it, monitoring it and so on. On top of that, a production system would run on a cloud and that is another layer of complexity to deal with while trying to figure out how to buid a Ml-Ops environment.

With that in mind the main objective of this repository is to provide a simple local (you won't need nothing beyond your computer and internet connection to set it up), but yet complete, example of what a end-to-end machine learning system could look like. The objective of doing it completely in a local machine is to abstract the cloud and infrastructure parts of the a Machine Learning system, but at the same it should be clear that the components used could be configured in a cloud provider. 

## Ml-Pipes Architecture

The image bellow depicts the architecture the this project build.



![Architecture](https://drive.google.com/uc?export=view&id=<1O4E9a0XqAf9Sa9Z6z9TjlaAJW3aqV1ab>)


This is a minimal ML project built with [MLFlow](https://www.mlflow.org/), [FastAPI](https://fastapi.tiangolo.com/) and [Docker](https://docs.docker.com/). The main objective os the project is to provide a self contained example of how to train and register models with MLFlow and deploy it through an API Service that allows us to save the predictions. It is recomends to check each framework documentation to have a detailed explanation of the components used here.


## Project Structure

The tree bellow displays the project strucuture. Note that the main file where MLFlow is put into action are the `trainers/` where diferent ML framework are use within MLFlow context in order to train models and `RunExperiment.py` where the experiment is configured.  

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
   ├─ __init__.py
   ├─ settings.py
   ├─ dao
   │  └─ CreditCardDefault.py
   ├─ RunExperiment.ipynb
   ├─ trainers
   │  ├─ h2o_automl.py
   │  └─ pycaret.py
   └─ ModelAPI.py
```

## Reproduce the Project

The following steps explain how to reproduce this project and by the end of it you should have understand the main funcionalities of MLFlow as well as to have a deployed model through an API Service that saves the predictions every time they are requested. There are three main steps that you need to execute in order to reproduce the project:

1. Prepare the Enviroment;
2. Train the models by executing a MLFlow Experiment;
3. Deploy the Model.

Each subsection bellow depicts how to execute each os the listed steps.

### 1) Prepare the Enviroment

Before start running any code make sure that the following requisites are meet:

- Create the file `src/dao/data/creditcard.csv` by downloading the file from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and dropping the column `Time`. Note that this is done to simplify the execution, in a real-world scenaria the Data Acess Object (dao) would probably come from a databse.;
- Install the dependencies from the `pyproject.toml` file, see [poetry](https://python-poetry.org/), or from the requirements.txt. Feel free to use the tool that you are more confortable with, just be sure to have an isolated enviroment for thi project (;
- Have Java installed. Yep, that's right. One of the trainers in this example is from the H2O framework and requires java to run. If you are running on ubuntu 20.04, simply install it by running the command in terminal `sudo apt-get install -y openjdk-11-jre`. If you don't want to do this, simply delete the H2O part in the `RunExperiment.py` and delete the `src/trainers/h2o_automl.py`.

Once this is done you can now move to the next step.

### 2) Train the models by executing a MLFlow Experiment

MLFlow is built upon the concept of experiments. Where each experiment consists of a series of runs, each run being a candidate model, aka, a fit in the training dataset. 

Here we will try to predict a credit card default using the [Kaggle credit card fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) and we will do that by running a H2O AutoMl model and Pycaret sklearn models as well. Finally, we will set to production the one with the highest [KS Metric](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test). All of this is done by executing the `src/RunExperiment.ipynb` file. Take a few minutes and go through the notebbok to understand how this is done, as you will see the script basically does:

1) Setup and MLFLow Experiemnt;
2) Load the Credit Card dataset;
3) Trains an H2O AutoML;
4) Trains sklearn classifiers, using PyCaret interface;
5) Trains 4 spark classifiers;
6) Select the best model according to the CHAMPION_METRIC;
7) Set the best model the production stage (to better understand this stage make sure to check the [MLFlow Model Registery Documentation](https://www.mlflow.org/docs/latest/model-registry.html)).

Once the experiment finished running you can deploy the [MLFlow Tracking UI](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) locally by running in the terminal (inside the `src/` folder) the following command: 

```shell
mlflow ui -p 5000 --backend-store-uri sqlite:///mlruns.db
```

Now you should be able to access the Traking UI in [http://127.0.0.1:5000](http://127.0.0.1:5000).


### 3)  Deploy the Model

Now that you have trained a few models and set the best one to the production stage is time to deploy it.

Before we start executing the code it is important to highlight one aspect of the deploy. The MLFlow Framework has a few diferent options to deploy the model. However, here it is opted to deploy it through a different way. The main reason for that, is to allow the API that is serving the model to save the predictions that are made, so that they can be used to check for data-drift and build ground truth observations. At the time of the elaboration of this project this was not possible using the MLFLow native API model serve option. 

The model will be deployed through an API Service. The file `src/ModelAPI.py` make use of the [FastAPI](https://fastapi.tiangolo.com/) framework to create a POST route that receives the data, makes the prediction, saves it to a `.csv` file (to simply, this could be saving the prediction into a database) and return the prediction. To do that simply execute the following command inside the `src/` folder:

```shell
uvicorn ModelAPI:app --port 5001 --reload
```

You can now the check the swagger documentation at [http://127.0.0.1:5001/docs](http://127.0.0.1:5001/docs) and try it out. Here is a json with two observation from the dataset that can be used to test the API. You can simply put you DataFrame in this format by ` df.to_dict('list')`.

```
{
  "V1": [-1.54878809850026, -1.80346507043129],
  "V2": [1.80869795041448, 1.94336351248285],
  "V3": [-0.953509033832342, -0.477192108231044],
  "V4": [2.21308539346999, -0.401702597457705],
  "V5": [-2.01572779170327,-0.364354050082577],
  "V6": [-0.913456844516923, -1.36866894683339],
  "V7": [-2.35601298316433, 0.387747076722669],
  "V8": [1.19716896702387, 0.344476113557209],
  "V9": [-1.67837405659509, 0.445387497102268],
  "V10": [-3.53865023182429, 0.731469924962325],
  "V11": [3.1020899271543, -0.685099722598294],
  "V12": [-3.99337305447702, 0.335507199868123],
  "V13": [-1.93741062327519, -0.251346974320909],
  "V14": [-3.82289410599595, 0.397305842955965],
  "V15": [0.830970110708369, -0.0994221366011376],
  "V16": [-2.47535885382925, -0.432827203638135],
  "V17": [-5.21187516766885, 0.182755154865333],
  "V18": [-0.413871678166879, -0.868298803648426],
  "V19": [0.933262164554872, 0.150896667377805],
  "V20": [0.390785963777347, 0.136729912408897],
  "V21": [0.855138263312025, -0.188720269612872],
  "V22": [0.77474482148342, -0.356520984447129],
  "V23": [0.0590371520063436, 0.187783520495935],
  "V24": [0.343199807900813, 0.390557329551228],
  "V25": [-0.468937928609185, -0.933427342866825],
  "V26": [-0.278337986906642, 0.0844267878051226],
  "V27": [0.625922215184372, -0.0667968389304196],
  "V28": [0.395573378256676, 0.0853278684349329],
  "Amount": [76.94, 4.95],
  "id": [116139, 256736]
}
```

Even though this will do the trick, a more robust way to serve the model API and Tracking UI together is using the Docker framework. This is done in through the `Dockerfile` and `docker-compose.yml`. To start the UI and Api simple execute the following command in the **root** project directory
```
docker-compose up
```

When you run for the first time this should take a while, since it will build the docker image and then deploy the containers. If everything was done correctly, then you have acess in port 5000 the UI tracking and in 5001 the production model, but now serving each service as a separated container.


```
docker run --name postgresql-container -p 5432:5432 -e POSTGRES_PASSWORD=test1234 -d postgres

docker run --name teste-pgadmin -p 80:80 -e "PGADMIN_DEFAULT_EMAIL=renatogroff@yahoo.com.br" -e "PGADMIN_DEFAULT_PASSWORD=PgAdmin2018!" -d dpage/pgadmin4

docker run -p 15432:80 \
    -e 'PGADMIN_DEFAULT_EMAIL=user@domain.com' \
    -e 'PGADMIN_DEFAULT_PASSWORD=SuperSecret' \
    -d dpage/pgadmin4


hostname -I

mlflow ui -p 5000 --backend-store-uri postgresql+psycopg2://postgres:test1234@192.168.15.3:5432/test_mlflow
```

```
# 1a) Subir banco para armazenar dados dos experimentos
docker run --name pg-docker -e POSTGRES_PASSWORD=test1234 -d -p 5432:5432 postgres




# 1b) Subir o PGAdmin (e criar um schema para o MLFlow?)
docker run --name pgadmin -p 15432:80 \
    -e 'PGADMIN_DEFAULT_EMAIL=user@domain.com' \
    -e 'PGADMIN_DEFAULT_PASSWORD=SuperSecret' \
    -d dpage/pgadmin4


# 2)  Subir Tracking Server do MLFlow
mlflow server -p 5000 --backend-store-uri postgresql+psycopg2://postgres:test1234@localhost/ --default-artifact-root /media/vinicius/Dados/projects/ml-pipes/mlflow_files



```