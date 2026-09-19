from __future__ import annotations

import base64
import json
import os
from secrets import token_urlsafe
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, redirect, render_template, request, url_for
from flask import session

from utils.disease_data import all_diseases, get_disease
from utils.gemini_vision import GeminiVisionError, analyze_plant_image
from utils.image_quality import photo_quality_error
from utils.plant_classifier import classify_leaf_photo


ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def create_app(test_config: dict | None = None) -> Flask:
    """Create a plant-photo analysis site with optional Google sign-in."""
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "replace-this-before-production"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY", ""),
        # Current multimodal Flash model. The vision helper also checks which
        # models are visible to this specific API key if Google retires a name.
        GEMINI_MODEL=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        GOOGLE_CLIENT_ID=os.environ.get("GOOGLE_CLIENT_ID", ""),
        GOOGLE_CLIENT_SECRET=os.environ.get("GOOGLE_CLIENT_SECRET", ""),
        GOOGLE_REDIRECT_URI=os.environ.get("GOOGLE_REDIRECT_URI", ""),
        # Keep the optional local model off unless the service owner explicitly
        # enables it. Downloading model weights during a free web request can
        # exceed the worker timeout; Gemini remains the reliable default.
        PLANT_CLASSIFIER_ENABLED=os.environ.get("PLANT_CLASSIFIER_ENABLED", "false").lower() == "true",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=True,
    )
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.context_processor
    def add_sign_in_context():
        return {
            "current_user": session.get("current_user"),
            "google_sign_in_enabled": bool(
                app.config["GOOGLE_CLIENT_ID"] and app.config["GOOGLE_CLIENT_SECRET"]
            ),
        }

    @app.get("/auth/google")
    def google_sign_in():
        if not app.config["GOOGLE_CLIENT_ID"] or not app.config["GOOGLE_CLIENT_SECRET"]:
            return _auth_error("Google sign-in has not been connected by the site owner yet.")
        state = token_urlsafe(32)
        session["google_oauth_state"] = state
        parameters = {
            "client_id": app.config["GOOGLE_CLIENT_ID"],
            "redirect_uri": _google_redirect_uri(app),
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "prompt": "select_account",
        }
        return redirect(f"{GOOGLE_AUTHORIZE_URL}?{urlencode(parameters)}")

    @app.get("/auth/google/callback")
    def google_callback():
        if request.args.get("state") != session.pop("google_oauth_state", None):
            return _auth_error("Google sign-in could not be verified. Please try again.")
        code = request.args.get("code")
        if not code:
            return _auth_error("Google sign-in was cancelled or did not return a code.")
        try:
            user = _get_google_user(app, code)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError):
            return _auth_error("Google sign-in could not be completed. Please try again.")
        if not user.get("email") or not user.get("email_verified"):
            return _auth_error("Please choose a Google account with a verified email address.")
        session["current_user"] = {
            "name": str(user.get("name") or user["email"]).strip(),
            "email": str(user["email"]).strip(),
        }
        return redirect(url_for("index"))

    @app.post("/auth/sign-out")
    def sign_out():
        session.pop("current_user", None)
        session.pop("google_oauth_state", None)
        return redirect(url_for("index"))

    @app.route("/scan", methods=["GET", "POST"])
    def scan():
        if request.method == "GET":
            return render_template("scan.html")

        image = request.files.get("image")
        if not image or not image.filename:
            return _scan_error("Choose a JPG, PNG, or WEBP leaf photo first.")
        if image.mimetype not in ALLOWED_MIME_TYPES:
            return _scan_error("Please choose a JPG, PNG, or WEBP image.")
        image_bytes = image.read()
        if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
            return _scan_error("Choose an image smaller than 8 MB.")
        if not _looks_like_image(image_bytes, image.mimetype):
            return _scan_error("That file is not a readable image. Please choose another photo.")
        quality_error = photo_quality_error(image_bytes, image.mimetype)
        if quality_error:
            return _scan_error(quality_error)
        classifier_hint = None
        if app.config["PLANT_CLASSIFIER_ENABLED"]:
            classifier_hint = classify_leaf_photo(image_bytes)

        try:
            disease = analyze_plant_image(
                image_bytes=image_bytes,
                mime_type=image.mimetype,
                api_key=app.config["GEMINI_API_KEY"],
                model=app.config["GEMINI_MODEL"],
                guide_entries=all_diseases(),
                classifier_hint=classifier_hint,
            )
        except GeminiVisionError as exc:
            app.logger.warning("Plant photo analysis unavailable: %s", exc)
            return _scan_error(f"Automatic photo analysis is unavailable: {exc}")

        image_data = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{image.mimetype};base64,{image_data}"
        return render_template("result.html", disease=disease, image_url=image_url)

    @app.get("/guide")
    def guide():
        plant = request.args.get("plant", "").strip()
        query = request.args.get("q", "").strip().lower()
        diseases = all_diseases()
        if plant:
            diseases = [item for item in diseases if item["plant"].lower() == plant.lower()]
        if query:
            diseases = [item for item in diseases if query in (item["name"] + " " + item["plant"]).lower()]
        plants = sorted({item["plant"] for item in all_diseases()})
        return render_template("disease_guide.html", diseases=diseases, plants=plants, selected_plant=plant, query=query)

    @app.get("/guide/<slug>")
    def disease_detail(slug: str):
        disease = get_disease(slug)
        if disease is None:
            return redirect(url_for("guide"))
        return render_template("disease_detail.html", disease=disease)

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.errorhandler(413)
    def too_large(_error):
        return _scan_error("That image is too large. Please choose one under 8 MB.")

    @app.errorhandler(Exception)
    def unexpected_error(error):
        """Show the deployment owner a safe error category, never secret values."""
        app.logger.exception("Unexpected application error")
        safe_reason = type(error).__name__
        return render_template("service_error.html", reason=safe_reason), 500

    return app


def _scan_error(message: str):
    """Render one upload warning directly, without adding session messages."""
    return render_template("scan.html", form_error=message), 400


def _auth_error(message: str):
    return render_template("auth_error.html", message=message), 400


def _google_redirect_uri(app: Flask) -> str:
    configured = app.config["GOOGLE_REDIRECT_URI"].strip()
    if configured:
        return configured
    return url_for("google_callback", _external=True, _scheme="https")


def _get_google_user(app: Flask, code: str) -> dict:
    token_payload = urlencode({
        "code": code,
        "client_id": app.config["GOOGLE_CLIENT_ID"],
        "client_secret": app.config["GOOGLE_CLIENT_SECRET"],
        "redirect_uri": _google_redirect_uri(app),
        "grant_type": "authorization_code",
    }).encode("utf-8")
    token_request = Request(
        GOOGLE_TOKEN_URL,
        data=token_payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(token_request, timeout=20) as response:
        token = json.loads(response.read().decode("utf-8"))
    access_token = token["access_token"]
    user_request = Request(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(user_request, timeout=20) as response:
        user = json.loads(response.read().decode("utf-8"))
    if not isinstance(user, dict):
        raise ValueError("Google returned an invalid user profile")
    return user


def _looks_like_image(data: bytes, mime_type: str) -> bool:
    """Verify common image signatures without another deployment dependency."""
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
