#!/bin/bash

# Kill process
#lsof -i :5000

echo "Deploying Production model name=TestCreditCard"
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
mlflow models serve -p 5001 --model-uri models:/TestCreditCard/production --no-conda
