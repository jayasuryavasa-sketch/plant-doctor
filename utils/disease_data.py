from __future__ import annotations
import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "diseases.json"


def all_diseases() -> list[dict]:
    with DATA_FILE.open(encoding="utf-8") as f:
        return json.load(f)["diseases"]


def get_disease(class_name: str) -> dict | None:
    normalized = class_name.lower().replace(" ", "_")
    for item in all_diseases():
        if item["slug"] == normalized or item.get("model_class", "").lower() == class_name.lower():
            return item
    if "___" in class_name:
        plant, condition = class_name.split("___", 1)
        healthy = "healthy" in condition.lower()
        return {"slug": normalized, "model_class": class_name, "plant": plant.replace("_", " ").title(),
                "name": "Healthy foliage" if healthy else condition.replace("_", " ").title(),
                "status": "Healthy" if healthy else "Possible condition", "severity": "Not estimated",
                "description": "The local model assigned this PlantVillage class. Compare it with symptoms on the whole plant before acting.",
                "symptoms":["Inspect nearby leaves and new growth", "Compare the visible pattern with trusted local guidance"],
                "causes":["A photo-only model cannot confirm the cause", "Pests, weather, and nutrient issues can look similar"],
                "spread":"Spread risk depends on the confirmed condition; avoid moving affected material until assessed.",
                "conditions":["Wet foliage", "Poor airflow", "Plant stress"],
                "immediate_steps":["Photograph more leaves and note when symptoms began", "Clean tools and avoid handling wet foliage"],
                "treatment":["Confirm the condition with a local agricultural extension service before using treatments", "Follow product labels and local regulations"],
                "organic":["Improve airflow and watering practices", "Remove clearly dead material with clean tools where appropriate"],
                "avoid":["Unverified pesticide recipes", "Treating a model label as a guaranteed diagnosis"],
                "prevention":["Monitor plants regularly", "Avoid prolonged leaf wetness", "Keep tools clean"],
                "care":["Water at soil level", "Provide suitable spacing and sunlight"],
                "notes":"This class has conservative general guidance; the result is not a confirmed field diagnosis."}
    return None
