"""Step 2: register both models under ONE name and set the champion / challenger aliases.

No run IDs are hard-coded: the latest runs are found by their run names, and the
winner is read from the 'winner' tag of the compare_sentiment summary run.
"""
import mlflow
from mlflow import MlflowClient

import config as C


def latest_run(client, experiment_id, run_name):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"No run named {run_name}. Run evaluate.py first.")
    return runs[0]


def main():
    mlflow.set_tracking_uri(C.TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(C.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit("Experiment not found. Run evaluate.py first.")

    compare = latest_run(client, exp.experiment_id, C.RUN_NAME_COMPARE)
    winner = compare.data.tags["winner"]
    print(f"Decision from {C.RUN_NAME_COMPARE}: winner = {winner}")

    versions = {}
    for run_name, spec in [(C.RUN_NAME_A, C.MODEL_A), (C.RUN_NAME_B, C.MODEL_B)]:
        run = latest_run(client, exp.experiment_id, run_name)
        model_uri = run.data.tags.get("logged_model_uri", f"runs:/{run.info.run_id}/model")
        mv = mlflow.register_model(model_uri, C.REGISTERED_MODEL_NAME)
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "hf_model_id", spec["hf_model_id"])
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version, "source_run", run_name)
        client.set_model_version_tag(C.REGISTERED_MODEL_NAME, mv.version,
                                     C.PRIMARY_METRIC, f"{run.data.metrics[C.PRIMARY_METRIC]:.4f}")
        versions[run_name] = mv.version
        print(f"Registered {run_name} -> {C.REGISTERED_MODEL_NAME} version {mv.version}")

    loser = C.RUN_NAME_B if winner == C.RUN_NAME_A else C.RUN_NAME_A
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", versions[winner])
    client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", versions[loser])
    print(f"\nAlias champion   -> version {versions[winner]} ({winner})")
    print(f"Alias challenger -> version {versions[loser]} ({loser})")
    print("Check the UI: Models -> sentiment-model. Next: python predict_from_mlflow.py")


if __name__ == "__main__":
    main()
