# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student:** Tulip Parag Pattewar · **USN:** 1MS25CI133 · **Section:** B

Skill Lab (MLOps with MLflow) – Mini Project

---

## 1. Task and models

| Item | Value |
|---|---|
| Task | Binary sentiment classification (POSITIVE / NEGATIVE) |
| Dataset | SST-2 (`stanfordnlp/sst2`), validation split |
| Subset | 200 sentences, random sample with seed 42 → `data/eval_inputs.csv` |
| **Model A – Champion** | `distilbert-base-uncased-finetuned-sst-2-english` (~67M params) |
| **Model B – Challenger** | `siebert/sentiment-roberta-large-english` (~355M params) |
| Inference settings | CPU, batch_size 16, max_length 128, truncation on |

Both models run on the same saved inputs, in the same order. Both are wrapped in one
custom `mlflow.pyfunc.PythonModel` (`sentiment_wrapper.py`, as in Lab 9), which maps every
label to `POSITIVE` / `NEGATIVE` so the outputs are directly comparable.

## 2. How to run

```bash
# 0. one-time setup
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

# 1. terminal 1 – start the tracking server and leave it running
python -m mlflow server --host 127.0.0.1 --port 5000

# 2. terminal 2 – run the project in this order
python evaluate.py               # runs A and B, logs runs, metrics, artifacts, decision
python register_champion.py      # registers both versions, sets champion/challenger aliases
python predict_from_mlflow.py    # loads models:/sentiment-model@champion, predicts on new data

# optional stretch goal
python swap_aliases.py           # swap the aliases
python predict_from_mlflow.py    # same script, no edits -> different model is used
```

MLflow UI: http://127.0.0.1:5000 → experiment **hf-sentiment-champion-challenger**.
Settings (USN, models, sample size, rule) live in `config.py`.

## 3. Champion rule (written before the final run)

1. **Primary metric:** macro-F1 on the 200-sentence evaluation set. **Higher is better.**
2. **Minimum improvement margin:** the Challenger must beat the Champion by **at least
   2 percentage points (0.02) of macro-F1**, otherwise the Champion stays. With only 200
   examples, one sentence is 0.5 points and the standard error of an accuracy around 90%
   is about ±2 points, so a smaller difference could be noise.
3. **Guardrail / tie-breaker:** the Challenger is rejected if its average latency per
   sentence is **more than 10× the Champion's**. If the difference is within the margin,
   the **Champion stays** because it is smaller and faster.
4. **Direction of every metric:** accuracy, macro-F1, precision, recall → higher is better;
   `avg_latency_ms`, `total_inference_s` → lower is better; `delta_primary_metric`
   (Challenger − Champion) → positive means the Challenger is better.

The same rule is coded in `config.py` / `evaluate.py` (`decide()`), and the result is logged
in the `compare_sentiment` run (params `primary_metric`, `margin`; metric
`delta_primary_metric`; tag `winner`).

## 4. Results

*(fill in from `outputs/metrics_summary.json` after running `evaluate.py`)*
| Metric | A – DistilBERT (Champion) | B – RoBERTa-large (Challenger) | B − A |
|---|---|---|---|
| Accuracy ↑ | 0.9350 | 0.9450 | +0.0100 |
| Macro-F1 ↑ (primary) | 0.9343 | 0.9444 | +0.0101 |
| Precision (macro) ↑ | 0.9359 | 0.9460 | +0.0101 |
| Recall (macro) ↑ | 0.9331 | 0.9432 | +0.0101 |
| Avg latency (ms/sentence) ↓ | 7.74 | 60.37 | +52.63 (7.8× slower) |
| Total inference (s) ↓ | 1.55 | 12.07 | +10.53 |

Disagreements between A and B: 20 of 200 sentences.

**Registry:** `sentiment-model` — version 1 = `@champion` (DistilBERT), version 2 = `@challenger` (RoBERTa-large).
Disagreements between A and B: ___ of 200 sentences.

**MLflow runs:** `compare_sentiment` (summary) with nested `A_sentiment_distilbert-sst2`
and `B_sentiment_roberta-large`.
**Registry:** `sentiment-model` — version ___ = `@champion`, version ___ = `@challenger`.

Plots: `outputs/comparison_A_vs_B.png`, `outputs/confusion_matrix_A_distilbert-sst2.png`,
`outputs/confusion_matrix_B_roberta-large.png`.

### Five cases where the models disagree (from `outputs/disagreements.csv`)
| Sentence (short) | Reference | A | B | My comment |
|---|---|---|---|---|
| "oh come on ." | NEGATIVE | POSITIVE | NEGATIVE | Very short with no clear sentiment words. B understood the annoyed tone; A guessed wrong with 0.99 confidence. |
| "movie fans , get ready to take off ... the other direction ." | NEGATIVE | POSITIVE | NEGATIVE | Sarcasm. The sentence starts positive and the joke is at the end; only the larger model B caught it. |
| "it 's a bit disappointing that it only manages to be decent instead of dead brilliant ." | NEGATIVE | NEGATIVE | POSITIVE | B was misled by the positive words "decent" and "brilliant" even though the sentence says the film fails to be brilliant. |
| "all that 's missing is the spontaneity , originality and delight ." | NEGATIVE | NEGATIVE | POSITIVE | Positive words used in a negative frame ("all that's missing is…"). A got it right; B focused on the positive words. |
| "mcconaughey 's fun to watch , the dragons are okay , not much fire in the script ." | POSITIVE | POSITIVE | NEGATIVE | Mixed review. A was right but unsure (0.71); B was confidently wrong (0.998). Even the reference label is debatable here. |
## 5. Decision

*(150–250 words, written after the run. Answer: does the Challenger improve on the
Champion? By how much? Is it meaningful with 200 samples? What are the trade-offs in
speed, size and errors? State which rule step decided the outcome.)*

## 6. Limitations and declaration

- Only 200 sentences from one dataset (short movie-review sentences), so small metric
  differences are not reliable and the results may not carry over to other text such as
  tweets or product reviews.
- Model A was fine-tuned on SST-2 itself, which may favour it on this dataset.
- Latency was measured once on my laptop CPU; it depends on hardware and background load.
- The new examples in `data/new_data.csv` have no reference labels, so they show
  behaviour rather than accuracy.
- **AI assistance:** I used Claude (Anthropic) to help structure the project and write the
  scripts. I ran every step myself and can explain every line of the code.
- The models have no neutral class, so neutral new sentences (e.g. "I'm in the music and fashion team") were labelled POSITIVE with over 94% confidence.
## Repository structure

```
README.md  requirements.txt  config.py  sentiment_wrapper.py
evaluate.py  register_champion.py  predict_from_mlflow.py  swap_aliases.py
data/         eval_inputs.csv, new_data.csv
outputs/      predictions_*.csv, metrics_summary*.json, disagreements.csv, plots,
              new_data_predictions.csv
screenshots/  mlflow_compare.png, registry_aliases.png, predict_output.png
```
`mlruns/`, `mlartifacts/` and `mlflow.db` are excluded through `.gitignore`.
