from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    from ml.split_utils import load_intent_dataframe, split_intent_dataframe
except ModuleNotFoundError:
    from split_utils import load_intent_dataframe, split_intent_dataframe


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "intents_augmented.csv"
GLOVE_PATH = ROOT / "data" / "glove" / "glove.6B.100d.txt"
MODELS_DIR = ROOT / "ml" / "models"
RESULTS_DIR = ROOT / "ml" / "results"

MAX_LEN = 32
EMBED_DIM = 100
MAX_WORDS = 30000


def export_model_to_onnx(model: tf.keras.Model, onnx_output: Path) -> None:
    try:
        import tf2onnx
    except Exception as exc:
        raise RuntimeError(
            "tf2onnx is required for BiLSTM ONNX export. Install it with: python -m pip install tf2onnx"
        ) from exc

    input_signature = [tf.TensorSpec([None, MAX_LEN], tf.int32, name="input_ids")]

    @tf.function(input_signature=input_signature)
    def serving_fn(input_ids: tf.Tensor) -> Dict[str, tf.Tensor]:
        return {"logits": model(input_ids, training=False)}

    def _convert() -> None:
        tf2onnx.convert.from_function(
            serving_fn,
            input_signature=input_signature,
            output_path=str(onnx_output),
            opset=17,
        )

    try:
        _convert()
    except Exception as exc:
        if "np.cast" not in str(exc):
            raise

        # tf2onnx<2.0 expects np.cast. Provide a tiny shim for NumPy 2.x.
        if not hasattr(np, "cast"):
            class _CompatCast(dict):
                def __getitem__(self, dtype):
                    return lambda arr: np.asarray(arr, dtype=dtype)

            np.cast = _CompatCast()  # type: ignore[attr-defined]

        _convert()


def load_glove_embeddings() -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    if not GLOVE_PATH.exists() or GLOVE_PATH.stat().st_size == 0:
        return embeddings

    with GLOVE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")
            if len(values) != EMBED_DIM + 1:
                continue
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def build_embedding_matrix(tokenizer: Tokenizer, glove_index: Dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    rng = np.random.default_rng(42)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(vocab_size, EMBED_DIM)).astype(np.float32)
    matrix[0] = 0.0
    covered = 0

    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        vec = glove_index.get(word)
        if vec is not None:
            matrix[idx] = vec
            covered += 1

    return matrix, covered


def build_model(vocab_size: int, num_classes: int, embedding_matrix: np.ndarray, embedding_trainable: bool) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(MAX_LEN,), dtype="int32"),
            Embedding(
                input_dim=vocab_size,
                output_dim=EMBED_DIM,
                weights=[embedding_matrix],
                trainable=embedding_trainable,
            ),
            Bidirectional(LSTM(128, return_sequences=False)),
            Dropout(0.4),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_intent_dataframe(DATA_PATH)
    train_df, test_df = split_intent_dataframe(df, test_size=0.2, random_state=42)

    X_train = train_df["text"].astype(str)
    X_test = test_df["text"].astype(str)
    y_train_labels = train_df["intent"].astype(str)
    y_test_labels = test_df["intent"].astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(df["intent"].astype(str))
    y_train = label_encoder.transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    glove = load_glove_embeddings()
    embedding_matrix, covered_vocab = build_embedding_matrix(tokenizer, glove)
    vocab_size = embedding_matrix.shape[0]
    embedding_trainable = True

    if glove and covered_vocab:
        coverage = covered_vocab / max(1, vocab_size - 1)
        print(f"Loaded GloVe vectors for {covered_vocab} tokens ({coverage:.1%} of tokenizer vocab).")
    else:
        print("GloVe vectors missing or invalid; using random-initialized trainable embeddings.")

    model = build_model(
        vocab_size=vocab_size,
        num_classes=len(label_encoder.classes_),
        embedding_matrix=embedding_matrix,
        embedding_trainable=embedding_trainable,
    )

    present_classes = np.unique(y_train)
    class_weights_array = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
    class_weights = {int(label): float(weight) for label, weight in zip(present_classes, class_weights_array)}

    callbacks = [
        EarlyStopping(monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    y_probs = model.predict(X_test_pad, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    keras_model_path = MODELS_DIR / "bilstm.keras"
    model.save(keras_model_path)

    onnx_output = MODELS_DIR / "bilstm.onnx"
    export_model_to_onnx(model=model, onnx_output=onnx_output)

    with (MODELS_DIR / "tokenizer.pkl").open("wb") as f:
        pickle.dump(tokenizer, f)

    with (RESULTS_DIR / "bilstm_results.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (RESULTS_DIR / "bilstm_history.json").open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    print("BiLSTM training complete.")
    print("Keras model saved to ml/models/bilstm.keras")
    print("ONNX saved to ml/models/bilstm.onnx")


if __name__ == "__main__":
    main()
