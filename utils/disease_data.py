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
    return None
