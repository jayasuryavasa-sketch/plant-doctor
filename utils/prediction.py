from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prediction:
    class_name: str
    plant: str
    confidence: float
    is_uncertain: bool
    mode: str


class PlantPredictor:
    """Runs a local, lightweight plant-disease classifier when no project model exists."""

    def __init__(self, model_path: Path, classes_path: Path, confidence_threshold: float = 0.65,
                 hf_token: str = "", hf_model: str = "", use_local_model: bool = False,
                 local_hf_model: str = ""):
        self.model_path = model_path
        self.classes_path = classes_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.classes = []
        self.mode = "unavailable"
        self.hf_token = hf_token
        self.hf_model = hf_model
        self.use_local_model = use_local_model
        self.local_hf_model = local_hf_model
        self.processor = None
        self.onnx_session = None
        self._load_real_model()
        if self.model is None and self.use_local_model:
            self._load_free_local_model()

    def _load_real_model(self) -> None:
        if not (self.model_path.exists() and self.classes_path.exists()):
            return
        try:
            import torch
            from torchvision import models
            self.classes = json.loads(self.classes_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(self.classes))
            model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
            model.eval()
            self.model = model
            self.mode = "trained"
        except Exception:
            # A broken or incompatible checkpoint should never prevent the web app from running.
            self.model = None
            self.mode = "unavailable"

    def _load_free_local_model(self) -> None:
        """Load a small public ONNX model locally; no API key or paid service.

        The model files are cached by Hugging Face on the service.  Its label set
        is intentionally limited to the crops it was trained on; this method
        never invents coverage for a different crop.
        """
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            model_file = hf_hub_download(repo_id=self.local_hf_model, filename="model.onnx")
            classes_file = hf_hub_download(repo_id=self.local_hf_model, filename="class_names.json")
            classes = json.loads(Path(classes_file).read_text(encoding="utf-8"))
            if not isinstance(classes, list) or not classes:
                raise ValueError("The local model has no usable class list")
            self.onnx_session = ort.InferenceSession(
                model_file, providers=["CPUExecutionProvider"]
            )
            self.classes = classes
            self.mode = "local PlantVillage model"
        except Exception:
            self.model = None
            self.processor = None
            self.onnx_session = None
            self.mode = "unavailable"

    def predict(self, image_path: Path) -> Prediction:
        result: Prediction
        if self.onnx_session is not None:
            result = self._predict_local_onnx(image_path)
        elif self.model is not None:
            if self.processor is not None:
                result = self._predict_local_transformers(image_path)
            else:
                result = self._predict_real(image_path)
        else:
            raise RuntimeError(
                "The local plant model could not start. Please try again after the service finishes starting."
            )
        return self._apply_visible_damage_guard(image_path, result)

    def _predict_local_onnx(self, image_path: Path) -> Prediction:
        """Match the published PlantVillage ONNX preprocessing exactly."""
        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # Resize the shorter side then centre crop, as used by the model card.
        width, height = image.size
        scale = 256 / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        left = (resized.width - 256) // 2
        top = (resized.height - 256) // 2
        image = resized.crop((left, top, left + 256, top + 256))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        batch = np.transpose(pixels, (2, 0, 1))[None, :, :, :]
        input_name = self.onnx_session.get_inputs()[0].name
        logits = self.onnx_session.run(None, {input_name: batch})[0][0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        index = int(np.argmax(probabilities))
        label = str(self.classes[index])
        score = float(probabilities[index])
        plant = label.split("___", 1)[0].replace("_", " ").title()
        return Prediction(label, plant, score, score < self.confidence_threshold, self.mode)

    def _apply_visible_damage_guard(self, image_path: Path, result: Prediction) -> Prediction:
        """Do not present visibly yellow/brown foliage as healthy.

        PlantVillage-style classifiers can assign a healthy class to real-world
        photos outside their training data. This is intentionally a conservative
        guard: it does not diagnose a disease; it only replaces an implausible
        healthy result with a request for further assessment.
        """
        if "healthy" not in result.class_name.lower():
            return result
        try:
            from PIL import Image

            image = Image.open(image_path).convert("HSV").resize((160, 160))
            pixels = list(image.getdata())
            # Saturated yellow/orange/brown pixels. Very dark pixels are ignored
            # so shadows and black backgrounds do not incorrectly trigger it.
            damaged = sum(
                1 for hue, saturation, value in pixels
                if saturation >= 65 and value >= 45 and 8 <= hue <= 48
            )
            damaged_share = damaged / len(pixels)
            if damaged_share >= 0.12:
                return Prediction(
                    "Unknown___Visible_leaf_damage",
                    "Plant (unconfirmed)",
                    result.confidence,
                    True,
                    f"{result.mode} safety check",
                )
        except Exception:
            # A safety enhancement must never make an otherwise usable scan fail.
            pass
        return result

    def _predict_local_transformers(self, image_path: Path) -> Prediction:
        import torch
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            probabilities = torch.softmax(self.model(**inputs).logits[0], dim=0)
        confidence, index = probabilities.max(0)
        label = self.model.config.id2label.get(str(index.item()), str(index.item()))
        plant = label.split("___", 1)[0].replace("_", " ").title()
        score = float(confidence.item())
        return Prediction(label, plant, score, score < self.confidence_threshold, self.mode)

    def _predict_real(self, image_path: Path) -> Prediction:
        import torch
        from PIL import Image
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            probabilities = torch.softmax(self.model(transform(image).unsqueeze(0)), dim=1)[0]
        confidence, index = probabilities.max(0)
        label = self.classes[index.item()]
        plant = label.split("___", 1)[0].replace("_", " ").title()
        score = float(confidence.item())
        return Prediction(label, plant, score, score < self.confidence_threshold, "trained")
