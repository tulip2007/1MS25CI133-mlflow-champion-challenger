"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (as in Lab 9).

Every model returns the same labels, POSITIVE or NEGATIVE, so models with
LABEL_0 / LABEL_1 outputs can be compared fairly.
"""
import mlflow
import pandas as pd

LABEL_MAP = {
    "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE",
    "LABEL_1": "POSITIVE", "LABEL_0": "NEGATIVE",
    "POS": "POSITIVE", "NEG": "NEGATIVE",
}


def normalise_label(label):
    return LABEL_MAP.get(str(label).upper(), str(label).upper())


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id, batch_size=16, max_length=128):
        self.hf_model_id = hf_model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.classifier = None

    def load_context(self, context):
        # Hugging Face is only used here, inside the wrapper.
        from transformers import pipeline
        self.classifier = pipeline("sentiment-analysis", model=self.hf_model_id, device=-1)

    def predict_texts(self, texts):
        if self.classifier is None:
            self.load_context(None)
        outputs = self.classifier(list(texts), batch_size=self.batch_size,
                                  truncation=True, max_length=self.max_length)
        return pd.DataFrame({
            "text": list(texts),
            "predicted_label": [normalise_label(o["label"]) for o in outputs],
            "confidence": [round(float(o["score"]), 4) for o in outputs],
        })

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        return self.predict_texts(texts)

    def __getstate__(self):
        # Do not pickle the downloaded pipeline into MLflow; it is rebuilt on load.
        state = self.__dict__.copy()
        state["classifier"] = None
        return state
