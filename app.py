from __future__ import annotations

from flask import Flask, abort, redirect, render_template, request, url_for

from utils.disease_data import all_diseases, get_disease


def create_app(test_config: dict | None = None) -> Flask:
    """Create the India crop guidance site without AI or account features."""
    app = Flask(__name__)
    app.config.from_mapping(SECRET_KEY="not-used-for-guide-only-site")
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/scan")
    def scan():
        crops = {item["plant"] for item in all_diseases()}
        return render_template("scan.html", crop_count=len(crops))

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

    @app.get("/history")
    def history():
        return redirect(url_for("guide"))

    @app.get("/about")
    def about():
        return render_template("about.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
