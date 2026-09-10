from __future__ import annotations

import hashlib
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
    """Uses a trained MobileNetV3 checkpoint when available; otherwise a deterministic demo."""

    def __init__(self, model_path: Path, classes_path: Path, confidence_threshold: float = 0.65,
                 hf_token: str = "", hf_model: str = "", use_local_model: bool = False,
                 local_hf_model: str = ""):
        self.model_path = model_path
        self.classes_path = classes_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.classes = []
        self.mode = "demo"
        self.hf_token = hf_token
        self.hf_model = hf_model
        self.use_local_model = use_local_model
        self.local_hf_model = local_hf_model
        self.processor = None
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
            self.mode = "demo"

    def _load_free_local_model(self) -> None:
        """Load a public PlantVillage model locally; no API key or paid service."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            self.processor = AutoImageProcessor.from_pretrained(self.local_hf_model)
            self.model = AutoModelForImageClassification.from_pretrained(self.local_hf_model)
            self.model.eval()
            self.mode = "local PlantVillage model"
        except Exception:
            self.model = None
            self.processor = None
            self.mode = "demo"

    def predict(self, image_path: Path) -> Prediction:
        if self.model is not None:
            if self.processor is not None:
                return self._predict_local_transformers(image_path)
            return self._predict_real(image_path)
        if self.hf_token and self.hf_model:
            try:
                return self._predict_huggingface(image_path)
            except Exception:
                # Network/API availability must not make the core app unusable.
                pass
        return self._predict_demo(image_path)

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

    def _predict_huggingface(self, image_path: Path) -> Prediction:
        """Optional hosted inference. Token remains server-side in .env, never JavaScript."""
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", api_key=self.hf_token)
        responses = client.image_classification(str(image_path), model=self.hf_model)
        if not responses:
            raise RuntimeError("Hosted model returned no classifications")
        best = responses[0]
        if isinstance(best, dict):
            label = best.get("label", "Unknown")
            score = float(best.get("score", 0.0))
        else:
            label = best.label
            score = float(best.score)
        plant = label.split("___", 1)[0].replace("_", " ").title()
        return Prediction(label, plant, score, score < self.confidence_threshold, "hosted API")

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

    def _predict_demo(self, image_path: Path) -> Prediction:
        # This intentionally makes no claim to recognize the image. It permits full UI testing
        # until a trained checkpoint is added.
        labels = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight", "Pepper_bell___healthy"]
        digest = hashlib.sha256(image_path.read_bytes()).digest()
        label = labels[digest[0] % len(labels)]
        confidence = 0.42 + (digest[1] / 255) * 0.18  # always below default threshold
        plant = label.split("___", 1)[0].replace("_", " ").title()
        return Prediction(label, plant, round(confidence, 3), True, "demo")
