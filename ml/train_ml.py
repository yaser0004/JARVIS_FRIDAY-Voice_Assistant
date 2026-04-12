from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

try:
    from ml.split_utils import load_intent_dataframe, split_intent_dataframe
except ModuleNotFoundError:
    from split_utils import load_intent_dataframe, split_intent_dataframe


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "intents_augmented.csv"
MODELS_DIR = ROOT / "ml" / "models"
RESULTS_DIR = ROOT / "ml" / "results"


def measure_latency(model: Pipeline, sample_texts: pd.Series, runs: int = 1000) -> Dict[str, float]:
    latencies = []
    values = sample_texts.tolist()
    if not values:
        return {"median_ms": 0.0, "p95_ms": 0.0}

    for i in range(runs):
        text = values[i % len(values)]
        start = time.perf_counter()
        model.predict([text])
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "median_ms": float(median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_intent_dataframe(DATA_PATH)
    train_df, test_df = split_intent_dataframe(df, test_size=0.2, random_state=42)

    X_train = train_df["text"]
    X_test = test_df["text"]
    y_train = train_df["intent"]
    y_test = test_df["intent"]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    linearsvc = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_features=50000,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")),
        ]
    )

    logreg = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=50000)),
            (
                "clf",
                LogisticRegression(
                    C=5,
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )

    linearsvc.fit(X_train, y_train_encoded)
    logreg.fit(X_train, y_train_encoded)

    y_pred_svc = linearsvc.predict(X_test)
    y_pred_log = logreg.predict(X_test)

    svc_metrics = evaluate(y_test_encoded, y_pred_svc)
    log_metrics = evaluate(y_test_encoded, y_pred_log)

    svc_latency = measure_latency(linearsvc, X_test, runs=1000)
    log_latency = measure_latency(logreg, X_test, runs=1000)

    results = {
        "linearsvc": {
            **svc_metrics,
            "latency_ms": svc_latency,
        },
        "logistic_regression": {
            **log_metrics,
            "latency_ms": log_latency,
        },
    }

    with (RESULTS_DIR / "ml_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(linearsvc, MODELS_DIR / "linearsvc.pkl")
    joblib.dump(linearsvc.named_steps["tfidf"], MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(logreg, MODELS_DIR / "logreg.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    print("LinearSVC accuracy:", svc_metrics["accuracy"])
    print("LogReg accuracy:", log_metrics["accuracy"])
    print("Results saved to ml/results/ml_results.json")


if __name__ == "__main__":
    main()
