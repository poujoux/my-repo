import logging

from flask import Flask, jsonify, render_template, request
from modules.model_service import predict_all

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/educateit", methods=["POST"])
def educate():
    uploaded = request.files.get("efiles")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Please choose a plant image first."}), 400
    try:
        image_bytes = uploaded.read()
        if not image_bytes:
            return jsonify({"error": "The uploaded file is empty."}), 400
        return jsonify({"predictions": predict_all(image_bytes)}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Plant prediction failed")
        return jsonify({"error": "Could not process image. Ensure models are trained."}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 8 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
