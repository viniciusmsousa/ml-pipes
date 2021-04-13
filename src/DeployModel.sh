#!/bin/bash

# Kill process
#lsof -i :5000

echo "Deploying Production model name=CreditCardDefault"


export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
mlflow models serve -p 5001 --model-uri models:/CreditCardDefault/staging --no-conda
