# import libs
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

# start h2o 
h2o.init(max_mem_size='3G')

# set mlflow experiment
experiment_name = 'automl_mlflow'
client = MlflowClient()
try:
    experiment = mlflow.create_experiment(experiment_name)
except:
    experiment = client.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)

# DAO
train, valid = h2o.import_file('data/creditcard.csv').split_frame(ratios=[0.7])
train['Class'] = train['Class'].asfactor()
valid['Class'] = valid['Class'].asfactor()


# Model Run
with mlflow.start_run():
    model = H2OAutoML(max_models=10, max_runtime_secs=150, seed=24, nfolds=3, sort_metric='AUCPR')
    model.train(x=train.columns[:-1], y=train.columns[-1:][0], training_frame=train, validation_frame=valid)

    mlflow.log_metric("auc", model.leader.auc())
    mlflow.log_metric("AUCPR", model.leader.aucpr())

    try:
        model.model_correlation_heatmap(valid).savefig('data/model_correlation.png')
        mlflow.log_artifact("data/model_correlation.png")
    try:
        model.varimp_plot(valid).savefig('data/varimp_plot.png')
        mlflow.log_artifact("data/varimp_plot.png")
    try:
        model.varimp_heatmap().savefig('data/varimp_heatmap.png')
        mlflow.log_artifact("data/varimp_heatmap.png")

    mlflow.h2o.log_model(model.leader, "model")
    



#draft
# lb = model.leaderboard
# lb = get_leaderboard(model, extra_columns='ALL')
# print(lb.head(rows=lb.nrows))
#exa = model.explain(valid, include_explanations = ['confusion_matrix', 'shap_summary', 'varimp','varimp_heatmap','model_correlation_heatmap'])

# all_mlflow_runs = client.list_run_infos(experiment.experiment_id)
# if len(all_mlflow_runs) > 0:
#     run_info = all_mlflow_runs[-1]
#     model = mlflow.h2o.load_model("mlruns/{exp_id}/{run_id}/artifacts/model/".format(exp_id=experiment.experiment_id,run_id=run_info.run_uuid))
#     result = model.predict(valid)
# else:
#     raise Exception('Run the training first')


