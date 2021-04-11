#!/bin/bash

# Kill process
#lsof -i :5000

# Training Models Candidates
python 01TrainClassification.py

# Serving the UI
mlflow ui -p 5000 &

# Serving the API
python 02RunClassificationModelAPI.py &