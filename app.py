from __future__ import annotations

import base64
import os

from flask import Flask, redirect, render_template, request, url_for

from utils.disease_data import all_diseases, get_disease
from utils.gemini_vision import GeminiVisionError, analyze_plant_image


ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def create_app(test_config: dict | None = None) -> Flask:
    """Create a no-account plant-photo analysis and India crop guide site."""
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "replace-this-before-production"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY", ""),
        # Use Google's maintained Flash alias. Fixed model names can disappear
        # from an individual free-tier key even while the API key remains valid.
        GEMINI_MODEL="gemini-flash-latest",
    )
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        return render_template("index.html")

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

        try:
            disease = analyze_plant_image(
                image_bytes=image_bytes,
                mime_type=image.mimetype,
                api_key=app.config["GEMINI_API_KEY"],
                model=app.config["GEMINI_MODEL"],
                guide_entries=all_diseases(),
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
