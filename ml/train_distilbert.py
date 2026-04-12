from __future__ import annotations

import atexit
import json
import inspect
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# Keep linear-algebra thread fan-out low to avoid OpenBLAS allocation failures.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Force transformers to stay on the PyTorch backend for this training script.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
try:
    import psutil
except Exception:  # pragma: no cover - optional guard
    psutil = None
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from ml.split_utils import load_intent_dataframe, split_intent_dataframe
except ModuleNotFoundError:
    from split_utils import load_intent_dataframe, split_intent_dataframe


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "intents_augmented.csv"
MODELS_DIR = ROOT / "ml" / "models"
RESULTS_DIR = ROOT / "ml" / "results"


@dataclass
class EncodedBatch:
    input_ids: list
    attention_mask: list
    labels: list


def build_dataset(df: pd.DataFrame, tokenizer: DistilBertTokenizer) -> Dataset:
    encoded = tokenizer(
        df["text"].astype(str).tolist(),
        max_length=64,
        padding="max_length",
        truncation=True,
    )
    encoded["labels"] = df["label"].astype(int).tolist()
    return Dataset.from_dict(encoded)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


def _export_and_optimize_onnx(model_dir: Path, onnx_dir: Path) -> None:
    onnx_dir.mkdir(parents=True, exist_ok=True)
    module_cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        str(model_dir),
        "--task",
        "text-classification",
        str(onnx_dir),
    ]

    cli_path = shutil.which("optimum-cli")
    cli_cmd = [
        cli_path,
        "export",
        "onnx",
        "--model",
        str(model_dir),
        "--task",
        "text-classification",
        str(onnx_dir),
    ] if cli_path else None

    try:
        subprocess.run(module_cmd, check=True)
    except subprocess.CalledProcessError:
        if cli_cmd is None:
            raise
        subprocess.run(cli_cmd, check=True)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        model_path = onnx_dir / "model.onnx"
        quantized_path = onnx_dir / "model.int8.onnx"
        if model_path.exists():
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
            )
    except Exception:
        pass


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is not None:
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _is_expected_trainer_process(pid: int) -> bool:
    if pid <= 0:
        return False
    if psutil is None:
        return True
    try:
        cmdline = " ".join(psutil.Process(pid).cmdline()).lower()
    except Exception:
        return False
    return "train_distilbert.py" in cmdline


def _release_single_run_lock(lock_path: Path) -> None:
    if not lock_path.exists():
        return
    try:
        owner_pid = int(lock_path.read_text(encoding="utf-8").strip())
    except Exception:
        owner_pid = -1
    if owner_pid not in (-1, os.getpid()):
        return
    try:
        lock_path.unlink()
    except OSError:
        pass


def _acquire_single_run_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except Exception:
            existing_pid = -1
        if (
            existing_pid > 0
            and existing_pid != current_pid
            and _pid_is_alive(existing_pid)
            and _is_expected_trainer_process(existing_pid)
        ):
            raise RuntimeError(
                f"Another DistilBERT training run is already active (pid={existing_pid}). "
                "Stop it before starting a new run."
            )
        try:
            lock_path.unlink()
        except OSError:
            pass
    lock_path.write_text(str(current_pid), encoding="utf-8")
    atexit.register(_release_single_run_lock, lock_path)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lock_path = MODELS_DIR / "distilbert_intent.train.lock"
    try:
        _acquire_single_run_lock(lock_path)
    except RuntimeError as exc:
        print(f"[distilbert] {exc}")
        return

    df = load_intent_dataframe(DATA_PATH)
    train_df, test_df = split_intent_dataframe(df, test_size=0.2, random_state=42)

    le = LabelEncoder()
    le.fit(df["intent"].astype(str))
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["label"] = le.transform(train_df["intent"].astype(str))
    test_df["label"] = le.transform(test_df["intent"].astype(str))

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = build_dataset(train_df, tokenizer)
    test_ds = build_dataset(test_df, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(le.classes_),
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        model = model.to("cuda")
        print(f"[distilbert] cuda_available=True device={torch.cuda.get_device_name(0)}")
    else:
        print("[distilbert] cuda_available=False training will run on CPU")

    output_dir = MODELS_DIR / "distilbert_intent"
    dataloader_workers = 0 if os.name == "nt" else 4

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": 5,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "learning_rate": 2e-5,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "fp16": use_cuda,
        "dataloader_pin_memory": use_cuda,
        "dataloader_num_workers": dataloader_workers,
        "logging_steps": 50,
        "report_to": [],
    }

    param_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in param_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    if "use_cpu" in param_names:
        training_kwargs["use_cpu"] = False
    if "no_cuda" in param_names:
        training_kwargs["no_cuda"] = False

    args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"[distilbert] trainer_device={trainer.args.device}")

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    preds_output = trainer.predict(test_ds)
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=1)

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    with (RESULTS_DIR / "distilbert_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with (RESULTS_DIR / "distilbert_history.json").open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    onnx_dir = MODELS_DIR / "distilbert_onnx"
    _export_and_optimize_onnx(output_dir, onnx_dir)

    import joblib

    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    print("DistilBERT fine-tuning complete.")
    print("ONNX export available at ml/models/distilbert_onnx")


if __name__ == "__main__":
    main()
