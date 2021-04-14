# sudo docker build -t viniciusmsousa/ml-pipes:0.1 .
## docker run --restart unless-stopped -d -p 5000:5000 -p 5001:5001 --name ml-pipes viniciusmsousa/ml-pipes:0.1
FROM python:3.8-slim
LABEL maintainer = "vinisousa04@gmail.com"

## OS Configuration
RUN apt-get update 

# Java Setup
RUN mkdir -p /usr/share/man/man1
RUN apt-get install -y openjdk-11-jre

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

# Export Variable
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db
ARG MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Set WorkDir to Start UI and Serve Model
WORKDIR /usr/src/app/src
