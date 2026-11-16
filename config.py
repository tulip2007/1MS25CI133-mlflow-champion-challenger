"""Settings shared by all three scripts. Change values here, not inside the scripts."""
import os

STUDENT_USN = "1MS25CI133"                     # <-- put your USN here

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
TASK = "sentiment"

# Model A = current Champion, Model B = Challenger
MODEL_A = {"role": "champion", "short_name": "distilbert-sst2",
           "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english"}
MODEL_B = {"role": "challenger candidate", "short_name": "roberta-large",
           "hf_model_id": "siebert/sentiment-roberta-large-english"}

RUN_NAME_A = f"A_{TASK}_{MODEL_A['short_name']}"
RUN_NAME_B = f"B_{TASK}_{MODEL_B['short_name']}"
RUN_NAME_COMPARE = f"compare_{TASK}"

# Evaluation set
DATASET_NAME = "stanfordnlp/sst2"
DATASET_SPLIT = "validation"
EVAL_SAMPLES = 200          # use 10 for a first quick test, then 200
SEED = 42

# Inference settings (same for both models)
BATCH_SIZE = 16
MAX_LENGTH = 128
DEVICE = "cpu"

# Champion rule - written BEFORE the final run (also in README.md)
PRIMARY_METRIC = "macro_f1"         # higher is better
MARGIN = 0.02                       # Challenger must win by >= 2 percentage points
MAX_LATENCY_RATIO = 10.0            # reject Challenger if avg latency > 10x Champion's
