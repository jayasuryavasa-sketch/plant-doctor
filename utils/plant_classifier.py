"""Optional local crop-disease classifier used as a cautious Gemini hint.

The lightweight public model is trained on PlantVillage images. It is never
shown as a diagnosis and it never replaces the Gemini explanation. A failed
download or inference simply returns ``None`` so the normal scan continues.
"""
from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from urllib.request import Request, urlopen


MODEL_REPOSITORY = "imaflower/plantvillage-mobilenetv3"
MODEL_BASE_URL = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/main"
MINIMUM_CONFIDENCE = 0.90


def classify_leaf_photo(image_bytes: bytes) -> dict[str, Any] | None:
    """Return a high-confidence supporting label, or None when unavailable."""
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError:
        return None

    try:
        model_path, labels_path = _ensure_model_files()
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            return None
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_spec = session.get_inputs()[0]
        image_size = _image_size(input_spec.shape)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = _center_crop(image, image_size)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
        logits = np.asarray(session.run(None, {input_spec.name: tensor})[0])[0]
        probabilities = _softmax(logits, np)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        if confidence < MINIMUM_CONFIDENCE or index >= len(labels):
            return None
        return {"label": labels[index], "confidence": confidence, "source": "PlantVillage classifier"}
    except Exception:
        return None


def _ensure_model_files() -> tuple[Path, Path]:
    cache = Path(os.environ.get("PLANT_CLASSIFIER_CACHE", Path(gettempdir()) / "plant-doctor-classifier"))
    cache.mkdir(parents=True, exist_ok=True)
    model_path = cache / "model.onnx"
    labels_path = cache / "class_names.json"
    if not model_path.exists():
        _download("model.onnx", model_path)
    if not labels_path.exists():
        _download("class_names.json", labels_path)
    return model_path, labels_path


def _download(filename: str, destination: Path) -> None:
    headers = {"User-Agent": "Plant-Doctor/1.0"}
    request = Request(f"{MODEL_BASE_URL}/{filename}", headers=headers)
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urlopen(request, timeout=20) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)


def _image_size(shape: list[Any]) -> int:
    dimensions = [value for value in shape if isinstance(value, int) and value > 16]
    return int(dimensions[-1]) if dimensions else 224


def _center_crop(image: Any, size: int) -> Any:
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)))
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def _softmax(logits: Any, np: Any) -> Any:
    values = logits - np.max(logits)
    exponentials = np.exp(values)
    return exponentials / np.sum(exponentials)
