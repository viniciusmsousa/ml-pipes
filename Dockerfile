# sudo docker build -t viniciusmsousa/ml-pipes:0.1 .
## docker run --restart unless-stopped -d -p 5000:5000 -p 5001:5001 --name ml-pipes viniciusmsousa/ml-pipes:0.1
FROM python:3.8-slim
LABEL maintainer = "vinisousa04@gmail.com"

## OS Configuration
RUN apt-get update 

# Java Setup
RUN mkdir -p /usr/share/man/man1
RUN apt-get install -y openjdk-11-jre
RUN apt-get install -y git

# Exposing Ports
EXPOSE 5000

## App Configuration
# Changing Workdir
RUN mkdir /usr/app
WORKDIR /usr/app
COPY . /usr/app

# Installing App Dependencies
RUN pip install -r requirements.txt
