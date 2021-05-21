# ML-Pipes

Ml-Pipes is named simply after a pipe, which is a tube of metal, plastic, or other material used to convey water, gas, oil, or other fluid substances. Here the 'substance' that is conveyed is not a fluid matter, but a machile learning model. 

From experience, the steeper part of the Ml-Ops learning curve happens in the transition from running models in a given environment (local machine, jupyter server in a cloud provider, etc.) to keeping tracking of different experiments, deploying it, monitoring it and so on. On top of that, a production system would run on a cloud and that is another layer of complexity to deal with while trying to figure out how to buid a Ml-Ops environment.

With that in mind the main objective of this repository is to provide a simple local (you won't need nothing beyond your computer and internet connection to set it up), but yet complete, example of what a end-to-end machine learning system could look like. The objective of doing it completely in a local machine is to abstract the cloud and infrastructure parts of the a Machine Learning system, but at the same it should be clear that the components used could be configured in a cloud provider. 

## ML-Pipes Architecture

The image bellow depicts the architecture of our Machine Learning System. The directed lines represents information requests and it is presented the 4 main components:
- **1) Model Development**: The enviroments that the model is developed. This part is the only one that is not covered by Ml-Pipes, in other words is assumed that you have an enviroment to develop a model. Ml-Pipes goes as far as to provide an `.ipynb` with an example;
- **2) Tracking Server e Model Registery**: Ml-Pipes makes use of the [MLFlow](https://www.mlflow.org/), which is open source machine learning model management framework;
- **3) Remote Backend Store Tracking**: Is a postgres database that MLFlow will use to record meta data on experiment and runs. [Checkout MLFlow documentation on recording runs to a detailed explanation](https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded);
- **4) Remote Artifact Storage**: This is the place where model and artifacts (images, files) from the models that we develop. This could be an AWS S3 bucket, Google Storage, etc ([Checkout MLFlow documentation on recording runs to a detailed explanation](https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded)). However ML-Pipes creates a [minio](https://min.io/) container to simulate the S3 bucket. This is an workaround to avoid the need to use a cloud account/service and keep the project running localy, but this could be easily changed to a cloud bucket. 

The components inside the blue box are the ones that Ml-Pipes starts using the [Docker](https://docs.docker.com/) framework.

![Architecture](https://drive.google.com/uc?export=view&id=1NpnQcclBGRGNxwiyR7bPDpKT6hDxryME)

## Starting ML-Pipes

This section describes how to setup and start the ML-Pipes containers, so that you can create experiments and log your model runs. By going through the following steps:
- 1) Clonning the ML-Pipes Repository;
- 2) Configure Environment Variables; 
- 3) Start the Containers and Create the `mlflow/` Bucket;
### 1) Clonning the ML-Pipes Repository

Simply run the command in the terminal:
```
git clone https://github.com/viniciusmsousa/ml-pipes.git
```
Or download the zip folder from github UI.

### 2) Configure Environment Variables

It is needed to define a few environment variables to make sure that the containers can communicate with each other. Specifically, it is required to (i) create a user and password Postgres (component 3), so that MLFlow (component 2) can communicate with it. And (ii) a user and password so that components 1 (model development) and 2 (MLFlow) can access the Artifact Storage. The Minio framework works very similar to what it would be with a AWS, we will define a acess and secret key.

Since the services are deployed with docker-compose the environment variables will be declared using [docker compose environment variables](https://docs.docker.com/compose/environment-variables/). Basically, all you have to do is create a `.env` in the same directory of the `docker-compose.yml` and define the variables as following:

```
POSTGRES_DB=<db_name>
POSTGRES_USER=<db_user_name>
POSTGRES_PASSWORD=<db_user_password>

MINIO_ACCESS_KEY=<minio_access_key>
MINIO_SECRET_KEY=<minio_secret_key>
```
Now that the environment variables are configured, the next step is to start the services.
### 3) Start the Containers and Create the `mlflow/` Bucket

The next step is to run the command:

```
docker-compose up
```

The first time that the command is executed will take some sime, since the images will have to be created. Once the images are build you should be able to access the following services:
- MLFlow Server UI in [http://127.0.0.1:5000](http://127.0.0.1:5000);
- Minio Server UI in [http://127.0.0.1:9000](http://127.0.0.1:9000).

In a interface that looks like the image bellow

![uis](https://drive.google.com/uc?export=view&id=1FDUdV_V5bTzAbRLv9rhHDSdK-ZOe6HSH)

The final part of this step is to create a bucket called `mlflow` through the minio UI.

After that ML-Pipes is up and running and if you want to checkout how it was done check the `docker-compose.yml`. Note that this architecture can be 'easily' reproduced in a cloud environment with a (i) relational database, (ii) an instance of the Mlflow server and (iii) a storage bucket.

You can check the `example-model-development/RunExperiment.ipynb` file to see how you can use MLFlow to keep track of your models and decide the best model for the problem.

One final note before moving on is about where the *postgres* and *minio bucket* services will store their data. The project creates [docker volumes](https://docs.docker.com/compose/compose-file/compose-file-v3/#volumes), if you are running in linux, for instance, there should be the folders in `ml-pipes_minio_bucket/` and `ml-pipes_postgres-db/` inside the path `/var/lib/docker/volumes`. On the first folder you will find the artifacts (images, tables and binary models) loggeg from MLFlow and on the second one the postgres files.   
