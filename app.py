from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils.disease_data import all_diseases, get_disease
from utils.advice import answer_question
from utils.image_validation import ImageValidationError, validate_image
from utils.prediction import PlantPredictor


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
DATABASE_PATH = BASE_DIR / "database" / "plant_doctor.db"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_DAILY_SCAN_LIMIT = 20


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "change-this-before-production"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        DATABASE=str(DATABASE_PATH),
        UPLOAD_FOLDER=str(UPLOAD_DIR),
        CONFIDENCE_THRESHOLD=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.65")),
        HF_TOKEN=os.environ.get("HF_TOKEN", ""),
        HF_MODEL=os.environ.get("HF_MODEL", "imaflower/plantvillage-mobilenetv3"),
        USE_LOCAL_MODEL=os.environ.get("USE_LOCAL_MODEL", "false").lower() == "true",
        LOCAL_HF_MODEL=os.environ.get("LOCAL_HF_MODEL", "VaigandlaHemanth/leaf-disease-clip-vit"),
        DAILY_SCAN_LIMIT=int(os.environ.get("DAILY_SCAN_LIMIT", str(DEFAULT_DAILY_SCAN_LIMIT))),
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    init_db(app)
    app.extensions["predictor"] = PlantPredictor(
        model_path=BASE_DIR / "model" / "plant_disease_model.pth",
        classes_path=BASE_DIR / "model" / "class_names.json",
        confidence_threshold=app.config["CONFIDENCE_THRESHOLD"],
        hf_token=app.config["HF_TOKEN"],
        hf_model=app.config["HF_MODEL"],
        use_local_model=app.config["USE_LOCAL_MODEL"],
        local_hf_model=app.config["LOCAL_HF_MODEL"],
    )

    @app.context_processor
    def globals_for_templates():
        return {"model_mode": app.extensions["predictor"].mode}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.route("/scan", methods=["GET", "POST"])
    def scan():
        if request.method == "GET":
            status = get_scan_quota_status(app, get_quota_client_id())
            return render_template("scan.html", quota=status)
        file = request.files.get("image")
        if not file or not file.filename:
            flash("Choose a JPG, PNG, or WEBP photo of a leaf first.", "error")
            return redirect(url_for("scan"))
        extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            flash("Please upload a JPG, JPEG, PNG, or WEBP image.", "error")
            return redirect(url_for("scan"))
        stored_name = f"{uuid4().hex}_{secure_filename(file.filename)}"
        image_path = Path(app.config["UPLOAD_FOLDER"]) / stored_name
        try:
            file.save(image_path)
            validate_image(image_path, max_bytes=MAX_UPLOAD_BYTES)
            if not consume_scan_quota(app, get_quota_client_id()):
                image_path.unlink(missing_ok=True)
                flash(f"You have used all {app.config['DAILY_SCAN_LIMIT']} scans for today. Please try again tomorrow.", "error")
                return redirect(url_for("scan"))
            result = app.extensions["predictor"].predict(image_path)
        except ImageValidationError as exc:
            image_path.unlink(missing_ok=True)
            flash(str(exc), "error")
            return redirect(url_for("scan"))
        except Exception:
            image_path.unlink(missing_ok=True)
            app.logger.exception("Prediction failed")
            flash("We couldn't analyze this image right now. Please try again.", "error")
            return redirect(url_for("scan"))

        disease = get_disease(result.class_name)
        scan_id = save_scan(app, stored_name, result, disease)
        return redirect(url_for("result", scan_id=scan_id))

    @app.get("/result/<int:scan_id>")
    def result(scan_id: int):
        scan = get_scan(app, scan_id)
        if scan is None:
            abort(404)
        disease = get_disease(scan["class_name"])
        return render_template("result.html", scan=scan, disease=disease)

    @app.post("/result/<int:scan_id>/ask")
    def ask_plant_doctor(scan_id: int):
        scan = get_scan(app, scan_id)
        if scan is None:
            return jsonify({"error": "This scan could not be found."}), 404
        question = str((request.get_json(silent=True) or {}).get("question", ""))[:500]
        return jsonify({"answer": answer_question(question, get_disease(scan["class_name"]))})

    @app.get("/history")
    def history():
        with connect(app) as db:
            scans = db.execute("SELECT * FROM scans ORDER BY created_at DESC").fetchall()
        return render_template("history.html", scans=scans)

    @app.post("/history/<int:scan_id>/delete")
    def delete_scan(scan_id: int):
        scan = get_scan(app, scan_id)
        if scan:
            (Path(app.config["UPLOAD_FOLDER"]) / scan["image_name"]).unlink(missing_ok=True)
            with connect(app) as db:
                db.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
                db.commit()
            flash("Scan removed from history.", "success")
        return redirect(url_for("history"))

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

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
            abort(404)
        return render_template("disease_detail.html", disease=disease)

    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.errorhandler(413)
    def too_large(_error):
        flash("That image is too large. Please choose one under 8 MB.", "error")
        return redirect(url_for("scan"))

    return app


def connect(app: Flask):
    db = sqlite3.connect(app.config["DATABASE"])
    db.row_factory = sqlite3.Row
    return db


def init_db(app: Flask) -> None:
    with connect(app) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            plant_name TEXT,
            confidence REAL NOT NULL,
            severity TEXT,
            is_uncertain INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS scan_usage (
            client_id TEXT PRIMARY KEY,
            usage_date TEXT NOT NULL,
            scan_count INTEGER NOT NULL DEFAULT 0
        )""")
        db.commit()


def get_quota_client_id() -> str:
    """Return a signed browser-session identifier for the daily scan quota."""
    client_id = session.get("quota_client_id")
    if not client_id:
        client_id = uuid4().hex
        session["quota_client_id"] = client_id
    return client_id


def _quota_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def get_scan_quota_status(app: Flask, client_id: str) -> dict:
    with connect(app) as db:
        row = db.execute(
            "SELECT usage_date, scan_count FROM scan_usage WHERE client_id = ?",
            (client_id,),
        ).fetchone()
    used = row["scan_count"] if row and row["usage_date"] == _quota_today() else 0
    limit = app.config["DAILY_SCAN_LIMIT"]
    return {"limit": limit, "used": used, "remaining": max(0, limit - used)}


def consume_scan_quota(app: Flask, client_id: str) -> bool:
    """Consume one daily scan slot; return False when the browser has reached its limit."""
    today = _quota_today()
    limit = app.config["DAILY_SCAN_LIMIT"]
    with connect(app) as db:
        row = db.execute(
            "SELECT usage_date, scan_count FROM scan_usage WHERE client_id = ?",
            (client_id,),
        ).fetchone()
        if row is None:
            db.execute(
                "INSERT INTO scan_usage (client_id, usage_date, scan_count) VALUES (?, ?, 1)",
                (client_id, today),
            )
            return True
        if row["usage_date"] != today:
            db.execute(
                "UPDATE scan_usage SET usage_date = ?, scan_count = 1 WHERE client_id = ?",
                (today, client_id),
            )
            return True
        if row["scan_count"] >= limit:
            return False
        db.execute(
            "UPDATE scan_usage SET scan_count = scan_count + 1 WHERE client_id = ?",
            (client_id,),
        )
        return True


def save_scan(app: Flask, image_name: str, prediction, disease: dict | None) -> int:
    with connect(app) as db:
        cursor = db.execute(
            """INSERT INTO scans (image_name, class_name, plant_name, confidence, severity, is_uncertain, mode)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (image_name, prediction.class_name, prediction.plant, prediction.confidence,
             disease.get("severity") if disease else None, int(prediction.is_uncertain), prediction.mode),
        )
        db.commit()
        return cursor.lastrowid


def get_scan(app: Flask, scan_id: int):
    with connect(app) as db:
        return db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
