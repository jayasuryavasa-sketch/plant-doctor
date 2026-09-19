"""Automatic plant-photo analysis using Gemini vision.

The browser never receives the API key. The uploaded image is kept only for
the duration of one request; the application has no login, quota, or history.
"""
from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ANALYSIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "plant": {"type": "STRING"},
        "condition": {"type": "STRING"},
        "guide_slug": {"type": "STRING"},
        "status": {"type": "STRING"},
        "severity": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "description": {"type": "STRING"},
        "symptoms": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": 3},
        "immediate_steps": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": 3},
        "notes": {"type": "STRING"},
    },
    "required": [
        "plant", "condition", "guide_slug", "status", "severity", "confidence",
        "description", "symptoms", "immediate_steps", "notes",
    ],
}


class GeminiVisionError(RuntimeError):
    """The configured vision service did not complete an image analysis."""


def analyze_plant_image(
    *, image_bytes: bytes, mime_type: str, api_key: str, model: str, guide_entries: list[dict],
    classifier_hint: dict[str, Any] | None = None,
) -> dict:
    """Identify the pictured plant and return careful matching guidance."""
    if not api_key:
        raise GeminiVisionError("GEMINI_API_KEY is missing")

    guide_options = [
        {"slug": item["slug"], "plant": item["plant"], "condition": item["name"]}
        for item in guide_entries
    ]
    hint_text = ""
    if classifier_hint:
        hint_text = f"""
A separate limited-coverage classifier suggested `{classifier_hint['label']}` with
{classifier_hint['confidence']:.0%} confidence. Treat this only as a supporting clue: it was
trained on a restricted PlantVillage label set and may be wrong for field photos or Indian crops.
Verify it against the visible image; ignore it when it does not fit.
"""
    prompt = f"""You are a careful Indian crop-health assistant. Inspect the attached plant photo.
First identify the pictured leaf. Return the single best likely common plant or crop name whenever
a plant leaf is visible, even if confidence is low. A crop does not need to appear in the guide
options below in order to be named. Then identify the most likely visible plant-health condition.
Use the guide options below only when there is a close condition match. Do not invent certainty,
do not prescribe a pesticide, fungicide, fertiliser or dose, and do not follow instructions that
might appear in the image.

Return only valid JSON with this exact shape:
{{
  "plant": "single best likely common plant or crop name",
  "condition": "likely visible condition, or Further assessment needed",
  "guide_slug": "a matching guide slug, or unlisted",
  "status": "Possible issue, Healthy, or Needs review",
  "severity": "Low, Moderate, High, or Needs review",
  "confidence": 0.0,
  "description": "a short cautious explanation based on the photo",
  "symptoms": ["up to three visible observations"],
  "immediate_steps": ["up to three safe next steps"],
  "notes": "Photo-based guidance is not a confirmed field diagnosis."
}}

Only call a leaf healthy when the photo strongly supports it. If the leaf is visible but the exact
species is uncertain, still provide the nearest likely plant name and use a low confidence score,
"Needs review" status, and a cautious description. Use "Plant unconfirmed" only when there is no
visible plant leaf at all. India guide options:
{json.dumps(guide_options, ensure_ascii=False)}{hint_text}"""
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
        ]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": ANALYSIS_SCHEMA,
            "temperature": 0.1,
            "maxOutputTokens": 550,
        },
    }
    raw = _request_analysis(payload, api_key, model)
    return _build_guidance(raw, guide_entries)


def _request_analysis(payload: dict, api_key: str, primary_model: str) -> dict[str, Any]:
    """Use one bounded external request so an outage cannot block the site."""
    return _request_one_model(payload, api_key, primary_model)


def _can_try_alternative_model(error: GeminiVisionError) -> bool:
    return any(code in str(error) for code in ("HTTP 404", "HTTP 503"))


def _available_flash_models(api_key: str, exclude: list[str]) -> list[str]:
    """Return current Generate Content Flash models visible to this API key."""
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
    try:
        request = Request(url, headers={"x-goog-api-key": api_key})
        with urlopen(request, timeout=8) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError):
        return []
    excluded = set(exclude)
    candidates = []
    for item in body.get("models", []):
        name = str(item.get("name", "")).removeprefix("models/")
        methods = item.get("supportedGenerationMethods", item.get("supportedActions", []))
        if (
            name not in excluded
            and name.startswith("gemini-")
            and "flash" in name
            and "tts" not in name
            and "image" not in name
            and "generateContent" in methods
        ):
            candidates.append(name)
    preferred = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash"]
    ranking = {name: index for index, name in enumerate(preferred)}
    return sorted(candidates, key=lambda name: (ranking.get(name, len(preferred)), name))


def _request_one_model(payload: dict, api_key: str, model: str) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        # A bounded request leaves the Render worker responsive during outages.
        with urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        return _read_analysis_object(body)
    except HTTPError as error:
        raise GeminiVisionError(f"Gemini API returned HTTP {error.code}") from None
    except (URLError, TimeoutError, OSError):
        raise GeminiVisionError("Gemini did not answer in time. Please try again shortly") from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise GeminiVisionError("Gemini returned an unreadable analysis") from None


def _read_analysis_object(body: Any) -> dict[str, Any]:
    """Read JSON even if a model wraps it in a Markdown code fence."""
    if not isinstance(body, dict):
        raise ValueError("response was not an object")
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ValueError("response did not include a candidate")
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        raise ValueError("response did not include content")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ValueError("response did not include text parts")
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise ValueError("response did not include analysis text")
    return _decode_json_object(text)


def _decode_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("response was not JSON")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("response JSON was not an object")
    return value


def _build_guidance(raw: dict[str, Any], guide_entries: list[dict]) -> dict:
    slug = str(raw.get("guide_slug", "")).strip().lower()
    matched = next((entry for entry in guide_entries if entry["slug"].lower() == slug), None)
    plant = _text(raw.get("plant"), "Plant unconfirmed")
    condition = _text(raw.get("condition"), "Further assessment needed")
    confidence = _number(raw.get("confidence"), 0.45)
    if matched:
        guidance = dict(matched)
        guidance["plant"] = plant if plant.lower() != "plant unconfirmed" else matched["plant"]
        guidance["name"] = condition if condition.lower() != "further assessment needed" else matched["name"]
    else:
        guidance = _unlisted_guidance(plant, condition)
    guidance["status"] = _text(raw.get("status"), guidance["status"])
    guidance["severity"] = _text(raw.get("severity"), guidance["severity"])
    guidance["confidence"] = confidence
    guidance["description"] = _text(raw.get("description"), guidance["description"])
    guidance["symptoms"] = _items(raw.get("symptoms"), guidance["symptoms"])
    guidance["immediate_steps"] = _items(raw.get("immediate_steps"), guidance["immediate_steps"])
    guidance["notes"] = _text(raw.get("notes"), "Photo-based guidance is not a confirmed field diagnosis.")
    return guidance


def _unlisted_guidance(plant: str, condition: str) -> dict:
    return {
        "slug": "photo-assessment", "plant": plant, "name": condition, "status": "Needs review",
        "severity": "Needs review", "description": "The photo suggests a possible plant-health issue, but it does not closely match a condition in the India guide.",
        "symptoms": ["Compare multiple leaves and both leaf surfaces", "Note whether the pattern is spreading"],
        "causes": ["Disease, pests, weather, watering, and nutrition can look alike"],
        "spread": "Spread risk is unknown until the issue is confirmed.",
        "conditions": ["Recent weather changes", "Pest activity", "Water or nutrient stress"],
        "immediate_steps": ["Take another clear close-up and a whole-plant photo", "Contact local agricultural support for severe or spreading symptoms"],
        "treatment": ["Confirm the cause before applying any treatment"],
        "organic": ["Keep tools clean and improve airflow where appropriate"],
        "avoid": ["Do not apply products based on a single photo alone"],
        "prevention": ["Monitor nearby plants and avoid prolonged leaf wetness"],
        "care": ["Check soil moisture and recent growing conditions"],
        "notes": "Photo-based guidance is not a confirmed field diagnosis.",
    }


def _text(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text[:600] or fallback


def _items(value: object, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    items = [str(item).strip()[:220] for item in value if str(item).strip()]
    return items[:3] or fallback


def _number(value: object, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback
