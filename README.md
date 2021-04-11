# [wip] ML-Pipes

An experimental project in order to create a simple ML in production enviroment. Where the prduction model is automatically retrained once an event/criteria is verified to be true.

## Reproduce

In order to be able to run `deploy.sh` make sure that:
- You have folder named `data/creditcard.csv` from (Kaggle)[https://www.kaggle.com/mlg-ulb/creditcardfraud], withou the column `Time`;
- Install the dependencies from the `pyproject.toml` file, see (poetry)[https://python-poetry.org/];
- And have a Conda installed.

With that done you can change the variables in `settings.py` and run on the terminal `sh deploy.sh`.
