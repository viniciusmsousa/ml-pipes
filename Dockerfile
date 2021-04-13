# sudo docker build -t viniciusmsousa/ml-pipes:0.1 .
## docker run --restart unless-stopped -d -p 5000:5000 -p 5001:5001 --name ml-pipes viniciusmsousa/ml-pipes:0.1
FROM python:3.8-slim
LABEL maintainer = "vinisousa04@gmail.com"

## OS Configuration
RUN apt-get update 

# Java Setup
RUN mkdir -p /usr/share/man/man1
RUN apt-get install -y openjdk-11-jre

# MiniConda Setup
RUN apt-get install -y wget\
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV MLFLOW_CONDA_HOME="/root/miniconda3/:${PATH}"
ARG MLFLOW_CONDA_HOME="/root/miniconda3/:${PATH}"

# Exposing Ports
EXPOSE 5000
EXPOSE 5001

## App Configuration
# Changing Workdir
RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

# Installing App Dependencies
RUN pip install -r requirements.txt

# Deploy
CMD bash deploy.sh
