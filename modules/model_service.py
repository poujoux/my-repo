"""Load trained plant classifiers and predict with all three models."""
import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SMALL_SIZE = (32, 32)
MOBILE_SIZE = (160, 160)
MAX_IMAGE_PIXELS = 25_000_000

_REQUIRED = [
    MODEL_DIR / "knn.joblib",
    MODEL_DIR / "svm.joblib",
    MODEL_DIR / "mobilenet.keras",
    MODEL_DIR / "classes.json",
]
if not all(path.is_file() for path in _REQUIRED):
    raise RuntimeError(
        "Trained models were not found. Run 'python train_models.py' first."
    )

# Only load model files you created and trust. Joblib files are not safe from unknown sources.
knn = joblib.load(MODEL_DIR / "knn.joblib")
svm = joblib.load(MODEL_DIR / "svm.joblib")
mobile = tf.keras.models.load_model(MODEL_DIR / "mobilenet.keras")
classes = json.loads((MODEL_DIR / "classes.json").read_text(encoding="utf-8"))


def predict_all(image_bytes: bytes) -> dict:
    """Return predictions for KNN, SVM, and MobileNetV2."""
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large.")
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Please upload a valid JPG, PNG, or WebP image.") from exc

    small = np.asarray(image.resize(SMALL_SIZE), dtype=np.float32)
    small = (small / 255.0).reshape(1, -1)

    large = np.asarray(image.resize(MOBILE_SIZE), dtype=np.float32)
    large = preprocess_input(large)
    large = np.expand_dims(large, axis=0)

    results = {}
    for name, model, features in (
        ("KNN", knn, small),
        ("SVM", svm, small),
    ):
        label = int(model.predict(features)[0])
        scores = model.predict_proba(features)[0]
        results[name] = {
            "species": classes[label],
            "score": round(float(scores[label]), 4),
        }

    scores = mobile.predict(large, verbose=0)[0]
    label = int(np.argmax(scores))
    results["MobileNetV2"] = {
        "species": classes[label],
        "score": round(float(scores[label]), 4),
    }
    return results
