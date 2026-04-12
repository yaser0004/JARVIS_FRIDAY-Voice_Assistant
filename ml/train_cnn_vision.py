from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.datasets import ImageFolder

try:
    from core.config import CNN_VISION_LABELS_PATH, CNN_VISION_WEIGHTS_PATH, ML_RESULTS_DIR, ensure_runtime_dirs
except ModuleNotFoundError:
    import sys

    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from core.config import CNN_VISION_LABELS_PATH, CNN_VISION_WEIGHTS_PATH, ML_RESULTS_DIR, ensure_runtime_dirs

from vision.cnn_scratch import IMAGE_SIZE, ScratchVisionCNN, build_eval_transform, build_train_transform


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = ROOT / "data" / "vision_dataset"
RESULT_PATH = ML_RESULTS_DIR / "cnn_vision_results.json"
CIFAR_BATCH_DIR_NAME = "cifar-10-batches-py"
CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _resolve_training_device(device_preference: str) -> tuple[torch.device, str]:
    choice = (device_preference or "auto").strip().lower()

    if choice == "cpu":
        return torch.device("cpu"), "cpu"

    if choice in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), "gpu"
        print("[Device] GPU requested but CUDA is unavailable. Falling back to CPU.", flush=True)
        return torch.device("cpu"), "cpu-fallback"

    if torch.cuda.is_available():
        return torch.device("cuda"), "auto-gpu"
    return torch.device("cpu"), "auto-cpu"


def _describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"

    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    return f"cuda:{index} ({name})"


def _load_cifar_batch(batch_path: Path) -> tuple[np.ndarray, List[int]]:
    with batch_path.open("rb") as f:
        payload = pickle.load(f, encoding="bytes")

    data = payload.get(b"data")
    labels = payload.get(b"labels")
    if data is None or labels is None:
        raise RuntimeError(f"Invalid CIFAR-10 batch file: {batch_path}")
    return data, list(labels)


def _write_cifar_sample_png(flat_image: np.ndarray, output_path: Path) -> None:
    array = np.asarray(flat_image, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
    Image.fromarray(array, mode="RGB").save(output_path)


def _ensure_cifar10_imagefolder(dataset_root: Path, val_ratio: float, seed: int) -> bool:
    train_dir = dataset_root / "train"
    if train_dir.exists():
        existing_train_images = sum(1 for path in train_dir.rglob("*") if path.is_file())
        if existing_train_images > 0:
            return False

    cifar_dir = dataset_root / CIFAR_BATCH_DIR_NAME
    required = [cifar_dir / f"data_batch_{i}" for i in range(1, 6)] + [cifar_dir / "test_batch"]
    if not all(path.exists() for path in required):
        return False

    print(f"[Dataset] CIFAR-10 raw batches detected at {cifar_dir}", flush=True)
    print("[Dataset] Converting to ImageFolder format under data/vision_dataset/...", flush=True)

    val_ratio = max(0.05, min(float(val_ratio), 0.4))
    splitter = random.Random(seed)

    for split in ("train", "val", "test"):
        for label in CIFAR_CLASSES:
            (dataset_root / split / label).mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0
    test_count = 0

    for batch_idx in range(1, 6):
        data, labels = _load_cifar_batch(cifar_dir / f"data_batch_{batch_idx}")
        pbar_train = tqdm(
            total=len(data),
            desc=f"  data_batch_{batch_idx}/5",
            unit="img",
            leave=False,
            dynamic_ncols=True,
        )
        for item_idx, (flat_image, label_idx) in enumerate(zip(data, labels)):
            label_name = CIFAR_CLASSES[int(label_idx)]
            is_val = splitter.random() < val_ratio
            split = "val" if is_val else "train"
            out_name = f"cifar10_train_{batch_idx}_{item_idx:05d}.png"
            output_path = dataset_root / split / label_name / out_name
            _write_cifar_sample_png(flat_image, output_path)
            if is_val:
                val_count += 1
            else:
                train_count += 1
            pbar_train.update(1)
        pbar_train.close()
        print(f"[Dataset] Converted data_batch_{batch_idx}/5", flush=True)

    test_data, test_labels = _load_cifar_batch(cifar_dir / "test_batch")
    pbar_test = tqdm(
        total=len(test_data),
        desc="  test_batch",
        unit="img",
        leave=False,
        dynamic_ncols=True,
    )
    for item_idx, (flat_image, label_idx) in enumerate(zip(test_data, test_labels)):
        label_name = CIFAR_CLASSES[int(label_idx)]
        out_name = f"cifar10_test_{item_idx:05d}.png"
        output_path = dataset_root / "test" / label_name / out_name
        _write_cifar_sample_png(flat_image, output_path)
        test_count += 1
        pbar_test.update(1)
    pbar_test.close()

    print(
        "[Dataset] CIFAR-10 conversion complete: "
        f"train={train_count}, val={val_count}, test={test_count}",
        flush=True,
    )
    return True


class IndexedImageFolderDataset(Dataset):
    def __init__(self, base: ImageFolder, indices: Sequence[int], transform) -> None:
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source_idx = self.indices[item]
        path, target = self.base.samples[source_idx]
        image = self.base.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _split_indices(total: int, val_ratio: float, seed: int) -> tuple[List[int], List[int]]:
    all_indices = list(range(total))
    random.Random(seed).shuffle(all_indices)

    val_size = max(1, int(total * val_ratio))
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]
    if not train_indices:
        raise RuntimeError("Validation split consumed all samples. Provide more training images.")
    return train_indices, val_indices


def _accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    progress_desc: str | None = None,
) -> float:
    model.eval()
    total = 0
    correct = 0
    batches = dataloader
    progress = None
    if progress_desc:
        progress = tqdm(
            dataloader,
            desc=progress_desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        batches = progress
    with torch.no_grad():
        for images, targets in batches:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total += int(targets.numel())
            correct += int((preds == targets).sum().item())
    if progress is not None:
        progress.close()
    if total == 0:
        return 0.0
    return correct / total


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    progress = tqdm(
        dataloader,
        desc=f"  Epoch {epoch:02d}/{total_epochs} [train]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.size(0))
        total += batch_size
        running_loss += float(loss.item()) * batch_size
        progress.set_postfix(loss=f"{float(loss.item()):.4f}")

    progress.close()

    if total == 0:
        return 0.0
    return running_loss / total


def _build_datasets(dataset_root: Path, val_ratio: float, seed: int) -> tuple[Dataset, Dataset, Dataset | None, List[str]]:
    _ensure_cifar10_imagefolder(dataset_root=dataset_root, val_ratio=val_ratio, seed=seed)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    if not train_dir.exists():
        raise RuntimeError(
            f"Training dataset directory not found: {train_dir}. "
            "Expected format: data/vision_dataset/train/<class_name>/*.(jpg|jpeg|png)"
        )

    base_train = ImageFolder(str(train_dir), transform=None)
    if len(base_train.samples) < 8:
        raise RuntimeError("Need at least 8 training images to train a scratch CNN reliably.")

    labels = list(base_train.classes)
    print(f"[Dataset] Classes: {', '.join(labels)}", flush=True)

    if val_dir.exists():
        print(f"[Dataset] Using explicit validation split: {val_dir}", flush=True)
        train_dataset = IndexedImageFolderDataset(base_train, list(range(len(base_train.samples))), build_train_transform())
        base_val = ImageFolder(str(val_dir), transform=build_eval_transform())
        if list(base_val.classes) != labels:
            raise RuntimeError("Validation classes must match training classes exactly.")
        val_dataset: Dataset = base_val
    else:
        train_indices, val_indices = _split_indices(len(base_train.samples), val_ratio=val_ratio, seed=seed)
        train_dataset = IndexedImageFolderDataset(base_train, train_indices, build_train_transform())
        val_dataset = IndexedImageFolderDataset(base_train, val_indices, build_eval_transform())

    test_dataset: Dataset | None = None
    if test_dir.exists():
        base_test = ImageFolder(str(test_dir), transform=build_eval_transform())
        if list(base_test.classes) != labels:
            raise RuntimeError("Test classes must match training classes exactly.")
        test_dataset = base_test

    print(
        f"[Dataset] Split sizes: train={len(train_dataset)} val={len(val_dataset)} "
        f"test={len(test_dataset) if test_dataset is not None else 0}",
        flush=True,
    )

    return train_dataset, val_dataset, test_dataset, labels


def train_cnn_from_scratch(
    dataset_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    seed: int,
    device_preference: str = "auto",
) -> Dict[str, object]:
    ensure_runtime_dirs()
    torch.manual_seed(seed)
    random.seed(seed)

    device, device_mode = _resolve_training_device(device_preference)
    print(f"[Device] Using {_describe_device(device)} (mode={device_mode})", flush=True)
    train_dataset, val_dataset, test_dataset, labels = _build_datasets(dataset_root, val_ratio, seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        if test_dataset is not None
        else None
    )

    model = ScratchVisionCNN(num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history: List[Dict[str, float]] = []
    best_val_acc = 0.0
    best_state_dict = None
    start = time.perf_counter()

    print(
        f"[Train] Starting {epochs} epochs with batch_size={batch_size} over {len(train_dataset)} samples.",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            epoch=epoch,
            total_epochs=epochs,
        )
        val_acc = _accuracy(model, val_loader, device, progress_desc=f"  Epoch {epoch:02d}/{epochs} [val]")
        scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_accuracy": float(val_acc),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"loss={train_loss:.4f} "
            f"val_acc={val_acc * 100:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"  [Checkpoint] Best so far: val_acc={best_val_acc * 100:.2f}%", flush=True)

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    CNN_VISION_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state_dict,
            "labels": labels,
            "image_size": IMAGE_SIZE,
            "best_val_accuracy": float(best_val_acc),
        },
        CNN_VISION_WEIGHTS_PATH,
    )
    CNN_VISION_LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=True, indent=2), encoding="utf-8")

    model.load_state_dict(best_state_dict, strict=True)
    model.to(device)
    model.eval()

    test_acc = None
    if test_loader is not None:
        print("[Eval] Running final test-set evaluation...", flush=True)
        test_acc = _accuracy(model, test_loader, device, progress_desc="  Final [test]")

    elapsed_s = time.perf_counter() - start
    summary: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "num_classes": len(labels),
        "classes": labels,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device_mode": device_mode,
        "device": str(device),
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc) if test_acc is not None else None,
        "elapsed_seconds": float(elapsed_s),
        "history": history,
        "artifacts": {
            "weights": str(CNN_VISION_WEIGHTS_PATH),
            "labels": str(CNN_VISION_LABELS_PATH),
        },
    }

    ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JARVIS's scratch CNN image classifier.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Training device selection (default: auto).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train_cnn_from_scratch(
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device_preference=args.device,
    )
    print(
        "Training complete: "
        f"val_acc={float(summary['best_val_accuracy']) * 100:.2f}% "
        f"weights={summary['artifacts']['weights']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

