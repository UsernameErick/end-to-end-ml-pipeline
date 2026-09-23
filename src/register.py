# регистрация модели в MLFlow по ее run_id. будет вызываться pipeline
import mlflow
from mlflow import MlflowClient
import os

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
client = MlflowClient()

def register_model(run_id, model_uri):
    run = client.get_run(run_id)
    new_cv_accuracy = run.data.metrics["cv_accuracy"]
    print(f"New CV Accuracy: {new_cv_accuracy:.4f}")

    #model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(model_uri=model_uri, name="iris_mlops_pipeline")
    new_version = registered_model.version
    print(f"Registered model Version: {new_version}")

    try:
        champion = client.get_model_version_by_alias("iris_mlops_pipeline", "champion")
        champion_run = client.get_run(champion.run_id)
        champion_cv_accuracy = champion_run.data.metrics["cv_accuracy"]

        print(f"Current champion: V{champion.version}")
        print(f"Champion CV Accuracy: {champion_cv_accuracy:.4f}")
    except:
        champion = None
        champion_cv_accuracy = None
        print("No current champion.")

    if champion is None or new_cv_accuracy > champion_cv_accuracy:
        client.set_registered_model_alias(name = "iris_mlops_pipeline", alias="champion", version=new_version)
        print(f"V{new_version} became champion.")
    else:
        print(
            f"V{new_version} did not become champion. "
            f"Champion remains V{champion.version}."
        )

    return new_version