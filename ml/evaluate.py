from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "intents_augmented.csv"
MODELS_DIR = ROOT / "ml" / "models"
RESULTS_DIR = ROOT / "ml" / "results"


def _preferred_onnx_providers() -> List[str]:
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    selected = [provider for provider in preferred if provider in available]
    return selected or available


def _onnx_tensor_dtype(tensor_type: str):
    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32,
        "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return dtype_map.get(tensor_type, np.float32)


def _load_test_split() -> Tuple[pd.Series, np.ndarray, joblib.Memory]:
    df = pd.read_csv(DATA_PATH)
    X = df["text"].astype(str)
    y = df["intent"].astype(str)
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    y_enc = le.transform(y)
    _, X_test, _, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42,
    )
    return X_test, y_test, le


def _model_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total / (1024 * 1024)


def _latency_stats(fn, inputs: List[str], runs: int = 1000) -> Dict[str, float]:
    if not inputs:
        return {"median_ms": 0.0, "p95_ms": 0.0}
    latencies = []
    for i in range(runs):
        sample = inputs[i % len(inputs)]
        start = time.perf_counter()
        fn(sample)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], matrix: np.ndarray) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": per_class,
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(),
    }


def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], output_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 9))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_loss_curve(history_path: Path, output_path: Path, title: str) -> None:
    if not history_path.exists():
        return

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    plt.figure(figsize=(8, 4))
    if isinstance(history, dict):
        train_loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
        if train_loss:
            plt.plot(train_loss, label="train_loss")
        if val_loss:
            plt.plot(val_loss, label="val_loss")
    elif isinstance(history, list):
        train_loss = [item["loss"] for item in history if isinstance(item, dict) and "loss" in item]
        eval_loss = [item["eval_loss"] for item in history if isinstance(item, dict) and "eval_loss" in item]
        if train_loss:
            plt.plot(train_loss, label="train_loss")
        if eval_loss:
            plt.plot(eval_loss, label="eval_loss")

    plt.title(title)
    plt.xlabel("Step/Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _run_linearsvc(X_test: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    model = joblib.load(MODELS_DIR / "linearsvc.pkl")
    preds = model.predict(X_test)
    latency = _latency_stats(lambda txt: model.predict([txt]), X_test, runs=1000)
    return preds, latency


def _run_bilstm(X_test: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    onnx_path = MODELS_DIR / "bilstm.onnx"
    with (MODELS_DIR / "tokenizer.pkl").open("rb") as f:
        tokenizer = pickle.load(f)

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    session = ort.InferenceSession(
        str(onnx_path),
        providers=_preferred_onnx_providers(),
    )

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_dtype = _onnx_tensor_dtype(input_meta.type)

    def infer(text: str) -> np.ndarray:
        seq = tokenizer.texts_to_sequences([text])
        x = pad_sequences(seq, maxlen=32, padding="post", truncating="post").astype(input_dtype)
        output = session.run(None, {input_name: x})[0]
        return np.argmax(output, axis=1)

    preds = np.concatenate([infer(text) for text in X_test]).astype(int)
    latency = _latency_stats(lambda txt: infer(txt), X_test, runs=1000)
    return preds, latency


def _run_distilbert(X_test: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
    model_path = MODELS_DIR / "distilbert_onnx" / "model.onnx"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    session = ort.InferenceSession(
        str(model_path),
        providers=_preferred_onnx_providers(),
    )

    input_dtypes = {node.name: _onnx_tensor_dtype(node.type) for node in session.get_inputs()}
    inputs = list(input_dtypes)

    def infer(text: str) -> np.ndarray:
        encoded = tokenizer(
            [text],
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        feed = {}
        for name in inputs:
            if name in encoded:
                feed[name] = encoded[name].astype(input_dtypes[name])
        output = session.run(None, feed)[0]
        return np.argmax(output, axis=1)

    preds = np.concatenate([infer(text) for text in X_test]).astype(int)
    latency = _latency_stats(lambda txt: infer(txt), X_test, runs=1000)
    return preds, latency


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test, label_encoder = _load_test_split()
    labels = list(label_encoder.classes_)
    test_list = X_test.tolist()

    models = {
        "linearsvc": {
            "runner": _run_linearsvc,
            "artifact": MODELS_DIR / "linearsvc.pkl",
        },
        "bilstm": {
            "runner": _run_bilstm,
            "artifact": MODELS_DIR / "bilstm.onnx",
        },
        "distilbert": {
            "runner": _run_distilbert,
            "artifact": MODELS_DIR / "distilbert_onnx",
        },
    }

    report = {}
    for name, spec in models.items():
        preds, latency = spec["runner"](test_list)
        cm = confusion_matrix(y_test, preds)
        metrics = _compute_metrics(y_test, preds, labels, cm)
        report[name] = {
            **metrics,
            "latency_ms": latency,
            "model_size_mb": _model_size_mb(spec["artifact"]),
        }

        _plot_confusion_matrix(
            cm,
            labels,
            RESULTS_DIR / f"confusion_matrix_{name}.png",
            f"Confusion Matrix - {name}",
        )

    _plot_loss_curve(
        RESULTS_DIR / "bilstm_history.json",
        RESULTS_DIR / "loss_curve_bilstm.png",
        "BiLSTM Loss Curve",
    )
    _plot_loss_curve(
        RESULTS_DIR / "distilbert_history.json",
        RESULTS_DIR / "loss_curve_distilbert.png",
        "DistilBERT Loss Curve",
    )

    with (RESULTS_DIR / "comparison_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nModel Comparison")
    print("-" * 96)
    print(f"{'Model':<16} {'Accuracy':<10} {'MacroF1':<10} {'Median(ms)':<12} {'P95(ms)':<10} {'Size(MB)':<10}")
    print("-" * 96)
    for name, data in report.items():
        print(
            f"{name:<16} "
            f"{data['accuracy']:<10.4f} "
            f"{data['macro_f1']:<10.4f} "
            f"{data['latency_ms']['median_ms']:<12.3f} "
            f"{data['latency_ms']['p95_ms']:<10.3f} "
            f"{data['model_size_mb']:<10.2f}"
        )


if __name__ == "__main__":
    main()
