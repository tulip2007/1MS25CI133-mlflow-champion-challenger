"""Step 3: load whichever model has the 'champion' alias and predict on NEW data.

This script never imports transformers: everything goes through MLflow.
Usage:
    python predict_from_mlflow.py                 # uses @champion
    python predict_from_mlflow.py challenger      # optional: uses @challenger
"""
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config as C

alias = sys.argv[1] if len(sys.argv) > 1 else "champion"

mlflow.set_tracking_uri(C.TRACKING_URI)
client = MlflowClient()

mv = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, alias)
model_uri = f"models:/{C.REGISTERED_MODEL_NAME}@{alias}"
print(f"Loading {model_uri}")
print(f"  -> registered version : {mv.version}")
print(f"  -> Hugging Face model : {mv.tags.get('hf_model_id')}")
print(f"  -> source run         : {mv.tags.get('source_run')}")

model = mlflow.pyfunc.load_model(model_uri)

new_data = pd.read_csv("data/new_data.csv")
predictions = pd.DataFrame(model.predict(new_data[["text"]]))
predictions.insert(0, "model_version", mv.version)
predictions.insert(1, "alias", alias)

pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 160)
print("\n" + predictions[["text", "predicted_label", "confidence"]].to_string(index=False))

out_file = "outputs/new_data_predictions.csv" if alias == "champion" \
    else f"outputs/new_data_predictions_{alias}.csv"
predictions.to_csv(out_file, index=False)
print(f"\nSaved {out_file}")
