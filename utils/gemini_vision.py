"""Conservative plant-photo analysis using the Gemini vision API.

The API key is read only from the server environment.  No key is exposed to
the browser or saved in the database.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils.prediction import Prediction


class GeminiVisionError(RuntimeError):
    """A configured Gemini service could not complete an analysis."""


def analyze_plant_photo(image_path: Path, api_key: str, model: str) -> tuple[Prediction, dict] | None:
    """Return an AI result, or None so the caller can safely use its fallback."""
    if not api_key:
        return None
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    prompt = """You are a careful plant-health assistant. Analyze the attached plant photo.
Return ONLY valid JSON. Do not pretend certainty. Do not identify a plant or disease
unless visual evidence supports it. Do not recommend pesticides, fungicides, or
fertilizer products. Use concise plain language.

Required JSON shape:
{
  "plant": "common plant name or Plant unconfirmed",
  "condition": "possible issue or Further assessment needed",
  "status": "Possible issue" or "Healthy only if clearly supported",
  "severity": "Low" or "Moderate" or "High" or "Needs review",
  "confidence": 0.0 to 1.0,
  "description": "one or two careful sentences",
  "symptoms": ["up to 3 visible observations"],
  "causes": ["up to 3 plausible causes"],
  "spread": "careful note about spread risk",
  "conditions": ["up to 3 relevant factors"],
  "immediate_steps": ["up to 3 safe next steps"],
  "treatment": ["up to 2 non-prescriptive next steps"],
  "organic": ["up to 2 safe care practices"],
  "avoid": ["up to 2 things not to do"],
  "prevention": ["up to 3 prevention steps"],
  "care": ["up to 2 ongoing care steps"],
  "notes": "This is educational guidance, not a confirmed diagnosis."
}"""
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_path.read_bytes()).decode("ascii")}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.15, "maxOutputTokens": 1400},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            raw = json.loads(text)
            break
        except HTTPError as error:
            # A 503 is a temporary Google service overload. Retry twice before
            # asking the user to try again; no scan quota is consumed on failure.
            if error.code == 503:
                if attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                # Free shared capacity can be temporarily unavailable for the
                # primary model. Try the available Flash-Lite vision model once
                # rather than returning a demo prediction or a fake result.
                if model != "gemini-2.5-flash-lite":
                    return analyze_plant_photo(image_path, api_key, "gemini-2.5-flash-lite")
            raise GeminiVisionError(f"Gemini API returned HTTP {error.code}") from None
        except URLError:
            if attempt < 2:
                time.sleep(attempt + 1)
                continue
            raise GeminiVisionError("Could not connect to Gemini API") from None
        except (KeyError, IndexError, TypeError, ValueError, OSError):
            raise GeminiVisionError("Gemini returned an unreadable analysis") from None
    else:  # Defensive guard: the loop always exits by success or exception.
        raise GeminiVisionError("Gemini API did not return an analysis")

    disease = _normalise(raw)
    prediction = Prediction(
        class_name=f"AI___{_slug(disease['condition'])}",
        plant=disease["plant"],
        confidence=disease["confidence"],
        is_uncertain=disease["status"] != "Healthy" or disease["confidence"] < 0.8,
        mode="Gemini vision AI",
    )
    return prediction, disease


def _normalise(raw: object) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    def text(key: str, fallback: str) -> str:
        value = str(raw.get(key, fallback)).strip()
        return value[:600] or fallback
    def items(key: str, fallback: list[str]) -> list[str]:
        value = raw.get(key, fallback)
        if not isinstance(value, list):
            value = fallback
        cleaned = [str(item).strip()[:220] for item in value if str(item).strip()]
        return cleaned[:3] or fallback
    try:
        confidence = float(raw.get("confidence", 0.45))
    except (TypeError, ValueError):
        confidence = 0.45
    return {
        "plant": text("plant", "Plant unconfirmed"),
        "name": text("condition", "Further assessment needed"),
        "status": text("status", "Possible issue"),
        "severity": text("severity", "Needs review"),
        "confidence": max(0.0, min(1.0, confidence)),
        "description": text("description", "The photo needs further assessment before a plant condition can be confirmed."),
        "symptoms": items("symptoms", ["Inspect multiple leaves and both leaf surfaces"]),
        "causes": items("causes", ["A photo alone cannot confirm the cause"]),
        "spread": text("spread", "Spread risk is unknown until the issue is identified."),
        "conditions": items("conditions", ["Recent weather, watering, pests, and nutrition can affect leaves"]),
        "immediate_steps": items("immediate_steps", ["Inspect the whole plant and photograph additional leaves"]),
        "treatment": items("treatment", ["Confirm the issue before applying any treatment"]),
        "organic": items("organic", ["Water at soil level and improve airflow where appropriate"]),
        "avoid": items("avoid", ["Avoid applying treatments based on one photo alone"]),
        "prevention": items("prevention", ["Keep tools clean and monitor nearby plants"]),
        "care": items("care", ["Check soil moisture before watering"]),
        "notes": text("notes", "This is educational guidance, not a confirmed diagnosis."),
    }


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")[:100] or "assessment_needed"
