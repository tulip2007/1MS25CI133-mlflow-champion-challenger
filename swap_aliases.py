"""Stretch goal: swap the champion and challenger aliases.
Run this, then rerun predict_from_mlflow.py WITHOUT editing it - the output changes.
Run it again to swap back.
"""
import mlflow
from mlflow import MlflowClient

import config as C

mlflow.set_tracking_uri(C.TRACKING_URI)
client = MlflowClient()
champ = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "champion").version
chall = client.get_model_version_by_alias(C.REGISTERED_MODEL_NAME, "challenger").version
client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "champion", chall)
client.set_registered_model_alias(C.REGISTERED_MODEL_NAME, "challenger", champ)
print(f"champion: version {champ} -> {chall}   challenger: version {chall} -> {champ}")
