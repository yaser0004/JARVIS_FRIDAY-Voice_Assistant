from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from core.config import CNN_VISION_LABELS_PATH, CNN_VISION_WEIGHTS_PATH
from vision.cnn_scratch import ScratchVisionCNN, build_eval_transform


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


class CNNImageClassifier:
    def __init__(self) -> None:
        self.model = None
        self.transforms = None
        self.labels: List[str] = []
        self._load_error = ""
        self._device = "cpu"
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch

            if not CNN_VISION_WEIGHTS_PATH.exists() or not CNN_VISION_LABELS_PATH.exists():
                self._load_error = (
                    "Scratch CNN artifacts were not found. Train a model first with "
                    "`python ml/train_cnn_vision.py` and place outputs under ml/models/cnn_vision/."
                )
                self.model = None
                self.transforms = None
                self.labels = []
                return

            labels_raw = json.loads(CNN_VISION_LABELS_PATH.read_text(encoding="utf-8"))
            labels = [str(item) for item in labels_raw if str(item).strip()]
            if not labels:
                raise RuntimeError("labels.json is empty or invalid.")

            checkpoint = torch.load(CNN_VISION_WEIGHTS_PATH, map_location="cpu")
            state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state_dict, dict):
                raise RuntimeError("Model checkpoint does not contain a valid state_dict.")

            self.model = ScratchVisionCNN(num_classes=len(labels))
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.transforms = build_eval_transform()
            self.labels = labels

            with torch.no_grad():
                dummy = torch.zeros((1, 3, 128, 128))
                _ = self.model(dummy)

            self._device = "cpu"
            self._load_error = ""
        except Exception as exc:
            self.model = None
            self.transforms = None
            self.labels = []
            self._load_error = str(exc)

    def is_ready(self) -> bool:
        return self.model is not None and self.transforms is not None and bool(self.labels)

    def classify_image(self, image_path: str | Path, top_k: int = 3) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            return _response(False, "Image file was not found.", {"image_path": str(path)})

        if not self.is_ready():
            details = f" ({self._load_error})" if self._load_error else ""
            return _response(False, f"CNN classifier is unavailable{details}.")

        try:
            import torch
            from PIL import Image

            assert self.model is not None
            assert self.transforms is not None

            with Image.open(path).convert("RGB") as img:
                tensor = self.transforms(img).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                top_probs, top_indices = torch.topk(probs, k=max(1, min(top_k, len(self.labels))))

            predictions = []
            for score, idx in zip(top_probs.tolist(), top_indices.tolist()):
                label = self.labels[int(idx)] if int(idx) < len(self.labels) else f"class_{idx}"
                predictions.append({"label": label, "confidence": float(score)})

            top = predictions[0]
            return _response(
                True,
                f"Top match: {top['label']} ({top['confidence'] * 100:.1f}% confidence).",
                {"image_path": str(path), "predictions": predictions},
            )
        except Exception as exc:
            return _response(False, f"I could not classify the image: {exc}", {"error": str(exc), "image_path": str(path)})
