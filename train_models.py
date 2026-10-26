"""Train KNN, SVM, and MobileNetV2 on a folder-based plant dataset.

Expected layout:
dataset/
  Rose/
    rose_001.jpg
  Tulip/
    tulip_001.jpg

Run: python train_models.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SMALL_SIZE = (32, 32)
MOBILE_SIZE = (160, 160)
SEED = 42


def load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size)
        return np.asarray(image, dtype=np.float32)


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(
            "Dataset folder not found. Create dataset/<species_name>/ and add labeled images."
        )

    class_names = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    if len(class_names) < 2:
        raise SystemExit("Add at least two species folders inside dataset/.")

    paths, labels = [], []
    for label, class_name in enumerate(class_names):
        class_paths = [
            p for p in (DATA_DIR / class_name).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(class_paths) < 5:
            raise SystemExit(
                f"Class '{class_name}' has {len(class_paths)} images. "
                "For this starter split, add at least 5 images per class."
            )
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))

    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_paths)} | Test images: {len(test_paths)}")

    # KNN and SVM use the same flattened 32x32 RGB features.
    X_train_small = np.stack([
        load_rgb(p, SMALL_SIZE).reshape(-1) / 255.0 for p in train_paths
    ])
    X_test_small = np.stack([
        load_rgb(p, SMALL_SIZE).reshape(-1) / 255.0 for p in test_paths
    ])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    knn = KNeighborsClassifier(n_neighbors=min(5, len(train_paths)))
    knn.fit(X_train_small, y_train)
    joblib.dump(knn, MODEL_DIR / "knn.joblib")

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
    )
    svm.fit(X_train_small, y_train)
    joblib.dump(svm, MODEL_DIR / "svm.joblib")

    # Transfer learning: freeze ImageNet-pretrained MobileNetV2 and train a new head.
    X_train_mobile = np.stack([
        preprocess_input(load_rgb(p, MOBILE_SIZE)) for p in train_paths
    ])
    X_test_mobile = np.stack([
        preprocess_input(load_rgb(p, MOBILE_SIZE)) for p in test_paths
    ])
    y_train_array = np.asarray(y_train, dtype=np.int64)
    y_test_array = np.asarray(y_test, dtype=np.int64)

    base = MobileNetV2(
        input_shape=(MOBILE_SIZE[1], MOBILE_SIZE[0], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(MOBILE_SIZE[1], MOBILE_SIZE[0], 3))
    features = base(inputs, training=False)
    features = GlobalAveragePooling2D()(features)
    features = Dropout(0.2)(features)
    outputs = Dense(len(class_names), activation="softmax")(features)
    mobile = Model(inputs, outputs)
    mobile.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    mobile.fit(
        X_train_mobile,
        y_train_array,
        epochs=5,
        batch_size=16,
        verbose=1,
    )
    mobile.save(MODEL_DIR / "mobilenet.keras")

    (MODEL_DIR / "classes.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )

    predictions = {
        "KNN": knn.predict(X_test_small),
        "SVM": svm.predict(X_test_small),
        "MobileNetV2": mobile.predict(X_test_mobile, verbose=0).argmax(axis=1),
    }
    for name, predicted in predictions.items():
        print(f"\n{name} accuracy: {accuracy_score(y_test_array, predicted):.3f}")
        print(classification_report(
            y_test_array,
            predicted,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        ))

    print(f"\nSaved all models to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
