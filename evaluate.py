"""Step 1: run Model A and Model B on the SAME evaluation set and log everything to MLflow.

Runs created (in experiment hf-sentiment-champion-challenger):
    compare_sentiment                      <- summary run (parent)
      |- A_sentiment_distilbert-sst2       <- Champion  (nested)
      |- B_sentiment_roberta-large         <- Challenger (nested)

Start the server first:  python -m mlflow server --host 127.0.0.1 --port 5000
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
import transformers
from mlflow.models import infer_signature
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import config as C
from sentiment_wrapper import HFSentimentModel

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
LABELS = ["NEGATIVE", "POSITIVE"]


# ------------------------------------------------------------------
# 1. Build the shared evaluation set and save it
# ------------------------------------------------------------------
def build_eval_set():
    if os.path.exists("data/eval_inputs.csv"):
        df = pd.read_csv("data/eval_inputs.csv")
        if len(df) == C.EVAL_SAMPLES:
            print(f"Reusing the saved evaluation set data/eval_inputs.csv ({len(df)} rows)")
            return df
    try:
        from datasets import load_dataset
        full = load_dataset(C.DATASET_NAME, split=C.DATASET_SPLIT).to_pandas()
    except Exception as err:            # fallback if the datasets package is unavailable
        print("datasets package failed (", err, ") - reading the parquet file directly")
        full = pd.read_parquet(
            "hf://datasets/stanfordnlp/sst2/data/validation-00000-of-00001.parquet")
    sample = full.sample(n=C.EVAL_SAMPLES, random_state=C.SEED).reset_index(drop=True)
    df = pd.DataFrame({
        "source_idx": sample["idx"],
        "text": sample["sentence"].str.strip(),
        "reference": [LABELS[l] for l in sample["label"]],
    })
    df.to_csv("data/eval_inputs.csv", index=False)
    print(f"Saved {len(df)} evaluation sentences to data/eval_inputs.csv")
    return df


# ------------------------------------------------------------------
# 2. Run one model, compute metrics, save files
# ------------------------------------------------------------------
def evaluate_model(spec, prefix, eval_df):
    print(f"\nLoading {spec['hf_model_id']} ...")
    wrapper = HFSentimentModel(spec["hf_model_id"], C.BATCH_SIZE, C.MAX_LENGTH)
    wrapper.load_context(None)
    wrapper.predict_texts(["warm-up sentence"])            # exclude first-call overhead

    start = time.perf_counter()
    preds = wrapper.predict_texts(eval_df["text"].tolist())
    total_s = time.perf_counter() - start

    y_true, y_pred = eval_df["reference"], preds["predicted_label"]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "avg_latency_ms": total_s / len(eval_df) * 1000,
        "total_inference_s": total_s,
    }

    tag = f"{prefix}_{spec['short_name']}"
    pred_file = f"outputs/predictions_{tag}.csv"
    out = pd.DataFrame({
        "source_idx": eval_df["source_idx"], "input": eval_df["text"],
        "reference": y_true, "prediction": y_pred,
        "confidence": preds["confidence"], "correct": (y_true == y_pred).astype(int),
    })
    out.to_csv(pred_file, index=False)

    metrics_file = f"outputs/metrics_summary_{tag}.json"
    with open(metrics_file, "w") as f:
        json.dump({"model": spec["hf_model_id"], **{k: round(v, 4) for k, v in metrics.items()}},
                  f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(f"Confusion matrix - {spec['short_name']}")
    fig.tight_layout()
    cm_file = f"outputs/confusion_matrix_{tag}.png"
    fig.savefig(cm_file, dpi=120)
    plt.close(fig)

    print(f"{tag}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"avg_latency={metrics['avg_latency_ms']:.1f} ms")
    return wrapper, metrics, out, [pred_file, metrics_file, cm_file]


# ------------------------------------------------------------------
# 3. Log one model as a nested MLflow run
# ------------------------------------------------------------------
def log_model_run(run_name, spec, wrapper, metrics, files):
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.log_params({
            "hf_model_id": spec["hf_model_id"], "task": C.TASK, "role": spec["role"],
            "batch_size": C.BATCH_SIZE, "max_length": C.MAX_LENGTH,
            "eval_samples": C.EVAL_SAMPLES, "seed": C.SEED, "device": C.DEVICE,
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "student_usn": C.STUDENT_USN,
            "library_versions": f"transformers={transformers.__version__}; "
                                f"torch={torch.__version__}; mlflow={mlflow.__version__}",
            "dataset_name": f"{C.DATASET_NAME} ({C.DATASET_SPLIT})",
        })
        for f in files + ["data/eval_inputs.csv"]:
            mlflow.log_artifact(f)

        signature = infer_signature(
            pd.DataFrame({"text": ["example"]}),
            pd.DataFrame({"text": ["example"], "predicted_label": ["POSITIVE"],
                          "confidence": [0.99]}))
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=wrapper,
            code_paths=["sentiment_wrapper.py"],
            signature=signature,
            pip_requirements=["mlflow", "transformers", "torch", "pandas"],
        )
        mlflow.set_tag("logged_model_uri", info.model_uri)
        print(f"Logged {run_name} (run_id {run.info.run_id[:8]}...)")


# ------------------------------------------------------------------
# 4. Champion-Challenger decision (rule fixed in config.py / README)
# ------------------------------------------------------------------
def decide(m_a, m_b):
    delta = m_b[C.PRIMARY_METRIC] - m_a[C.PRIMARY_METRIC]
    latency_ratio = m_b["avg_latency_ms"] / m_a["avg_latency_ms"]
    if latency_ratio > C.MAX_LATENCY_RATIO:
        return "A", delta, latency_ratio, "Challenger rejected by latency guardrail"
    if delta >= C.MARGIN:
        return "B", delta, latency_ratio, "Challenger beats Champion by at least the margin"
    return "A", delta, latency_ratio, "Difference below margin - Champion stays"


def comparison_plot(m_a, m_b):
    names = ["accuracy", "macro_f1", "precision_macro", "recall_macro"]
    x = range(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1]})
    ax1.bar([i - 0.2 for i in x], [m_a[n] for n in names], 0.4, label=f"A {C.MODEL_A['short_name']}")
    ax1.bar([i + 0.2 for i in x], [m_b[n] for n in names], 0.4, label=f"B {C.MODEL_B['short_name']}")
    ax1.set_xticks(list(x), names)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_title("Quality metrics (higher is better)")
    ax1.legend()
    ax2.bar(["A", "B"], [m_a["avg_latency_ms"], m_b["avg_latency_ms"]], color=["C0", "C1"])
    ax2.set_title("Avg latency ms (lower is better)")
    fig.tight_layout()
    path = "outputs/comparison_A_vs_B.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    mlflow.set_tracking_uri(C.TRACKING_URI)
    mlflow.set_experiment(C.EXPERIMENT_NAME)
    eval_df = build_eval_set()

    with mlflow.start_run(run_name=C.RUN_NAME_COMPARE):
        w_a, m_a, out_a, files_a = evaluate_model(C.MODEL_A, "A", eval_df)
        log_model_run(C.RUN_NAME_A, C.MODEL_A, w_a, m_a, files_a)
        del w_a

        w_b, m_b, out_b, files_b = evaluate_model(C.MODEL_B, "B", eval_df)
        log_model_run(C.RUN_NAME_B, C.MODEL_B, w_b, m_b, files_b)
        del w_b

        winner, delta, ratio, reason = decide(m_a, m_b)
        winner_name = C.RUN_NAME_A if winner == "A" else C.RUN_NAME_B

        # Combined metrics file
        summary = {
            "primary_metric": C.PRIMARY_METRIC, "margin": C.MARGIN,
            "max_latency_ratio": C.MAX_LATENCY_RATIO,
            "A": {"hf_model_id": C.MODEL_A["hf_model_id"], **{k: round(v, 4) for k, v in m_a.items()}},
            "B": {"hf_model_id": C.MODEL_B["hf_model_id"], **{k: round(v, 4) for k, v in m_b.items()}},
            "delta_primary_metric": round(delta, 4), "latency_ratio_B_over_A": round(ratio, 2),
            "winner": winner_name, "reason": reason,
        }
        with open("outputs/metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Disagreements: add your own comment for 5 of them in the README
        dis = pd.DataFrame({
            "input": out_a["input"], "reference": out_a["reference"],
            "prediction_A": out_a["prediction"], "confidence_A": out_a["confidence"],
            "prediction_B": out_b["prediction"], "confidence_B": out_b["confidence"],
        })
        dis = dis[dis["prediction_A"] != dis["prediction_B"]].copy()
        dis["comment"] = ""
        dis.to_csv("outputs/disagreements.csv", index=False)

        plot = comparison_plot(m_a, m_b)

        mlflow.log_params({"primary_metric": C.PRIMARY_METRIC, "margin": C.MARGIN,
                           "max_latency_ratio": C.MAX_LATENCY_RATIO,
                           "model_A": C.MODEL_A["hf_model_id"], "model_B": C.MODEL_B["hf_model_id"],
                           "eval_samples": C.EVAL_SAMPLES, "seed": C.SEED})
        mlflow.log_metrics({"delta_primary_metric": delta, "latency_ratio_B_over_A": ratio,
                            "disagreements": len(dis)})
        mlflow.set_tags({"winner": winner_name, "decision_reason": reason,
                         "student_usn": C.STUDENT_USN,
                         "dataset_name": f"{C.DATASET_NAME} ({C.DATASET_SPLIT})"})
        for f in [plot, "outputs/metrics_summary.json", "outputs/disagreements.csv",
                  "data/eval_inputs.csv", "README.md"]:
            if os.path.exists(f):
                mlflow.log_artifact(f)

    print("\n================ RESULT ================")
    print(f"{C.PRIMARY_METRIC}: A={m_a[C.PRIMARY_METRIC]:.4f}  B={m_b[C.PRIMARY_METRIC]:.4f}  "
          f"delta(B-A)={delta:+.4f}  margin={C.MARGIN}")
    print(f"Latency: A={m_a['avg_latency_ms']:.1f} ms  B={m_b['avg_latency_ms']:.1f} ms  "
          f"ratio={ratio:.1f}x (limit {C.MAX_LATENCY_RATIO}x)")
    print(f"Disagreements between A and B: {len(dis)}")
    print(f"WINNER: {winner_name}  ({reason})")
    print("Next: python register_champion.py")


if __name__ == "__main__":
    main()
