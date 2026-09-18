from __future__ import annotations

import base64
import os

from flask import Flask, flash, redirect, render_template, request, url_for

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
        GEMINI_MODEL=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
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
            flash("Choose a JPG, PNG, or WEBP leaf photo first.", "error")
            return redirect(url_for("scan"))
        if image.mimetype not in ALLOWED_MIME_TYPES:
            flash("Please choose a JPG, PNG, or WEBP image.", "error")
            return redirect(url_for("scan"))
        image_bytes = image.read()
        if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
            flash("Choose an image smaller than 8 MB.", "error")
            return redirect(url_for("scan"))
        if not _looks_like_image(image_bytes, image.mimetype):
            flash("That file is not a readable image. Please choose another photo.", "error")
            return redirect(url_for("scan"))

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
            # The message contains only a safe error category/status, never the
            # API key or uploaded image. It lets the owner fix Render settings
            # without having to guess why automatic analysis was unavailable.
            flash(f"Automatic photo analysis is unavailable: {exc}", "error")
            return redirect(url_for("scan"))

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
        flash("That image is too large. Please choose one under 8 MB.", "error")
        return redirect(url_for("scan"))

    return app


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
