
FROM python:3.8-slim
LABEL maintainer = "vinisousa04@gmail.com"

## OS Configuration
RUN apt-get update 

# Java Setup
RUN mkdir -p /usr/share/man/man1
RUN apt-get install -y openjdk-11-jre

# MiniConda Setup
ENV MLFLOW_CONDA_HOME="/root/miniconda3/:${PATH}"
ARG MLFLOW_CONDA_HOME="/root/miniconda3/:${PATH}"
RUN apt-get install -y wget\
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 


## App Configuration
# Changing Workdir
RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

# Installing App Dependencies
RUN pip install -r requirements.txt

# Exposing Ports
EXPOSE 5000
EXPOSE 5001

# Deploy
CMD bash deploy.sh
