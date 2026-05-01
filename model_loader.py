"""
ModelLoader — loads a Keras ResNet model (model.h5) and runs inference.

Label mapping covers 36 common fruit/vegetable classes trained with
the Fruits-360 dataset convention. Adjust LABELS and QUALITY_MAP to
match your own model.
"""

import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Label mapping ──────────────────────────────────────────────────────────────
# Keys are zero-based class indices; values are human-readable fruit names.
# Extend / replace with the exact labels your model was trained on.
LABELS = {
    0:  "Apple",
    1:  "Avocado",
    2:  "Banana",
    3:  "Cherry",
    4:  "Coconut",
    5:  "Fig",
    6:  "Grape",
    7:  "Guava",
    8:  "Jackfruit",
    9:  "Kiwi",
    10: "Lemon",
    11: "Lime",
    12: "Lychee",
    13: "Mango",
    14: "Melon",
    15: "Orange",
    16: "Papaya",
    17: "Peach",
    18: "Pear",
    19: "Pineapple",
    20: "Plum",
    21: "Pomegranate",
    22: "Raspberry",
    23: "Strawberry",
    24: "Tomato",
    25: "Watermelon",
    26: "Apricot",
    27: "Blueberry",
    28: "Cantaloupe",
    29: "Dragonfruit",
    30: "Durian",
    31: "Elderberry",
    32: "Gooseberry",
    33: "Mulberry",
    34: "Passion Fruit",
    35: "Sapodilla",
}

# ── Quality inference based on confidence ─────────────────────────────────────
# This is a simple heuristic; replace with a dedicated quality model if needed.
QUALITY_THRESHOLDS = {
    "Fresh":    0.80,   # confidence >= 0.80 → Fresh
    "Good":     0.65,   # confidence >= 0.65 → Good
    "Ripe":     0.50,   # confidence >= 0.50 → Ripe
    "Overripe": 0.30,   # confidence >= 0.30 → Overripe
    # Below 0.30 → Rotten
}


def infer_quality(confidence: float) -> str:
    if confidence >= QUALITY_THRESHOLDS["Fresh"]:
        return "Fresh"
    elif confidence >= QUALITY_THRESHOLDS["Good"]:
        return "Good"
    elif confidence >= QUALITY_THRESHOLDS["Ripe"]:
        return "Ripe"
    elif confidence >= QUALITY_THRESHOLDS["Overripe"]:
        return "Overripe"
    else:
        return "Rotten"


class ModelLoader:
    """
    Wraps loading and inference for a Keras / TF SavedModel.

    Place model.h5 (or model/ directory for SavedModel format) in the same
    folder as this file, or set the MODEL_PATH environment variable.
    """

    DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
    IMG_SIZE = (224, 224)

    def __init__(self):
        self._model = None
        self._loaded = False

    def load(self):
        """Load the model from disk. Called once at startup."""
        try:
            # Import here so that the app can still start even if TF is not
            # installed (useful for local testing without GPU).
            import tensorflow as tf
            from tensorflow import keras  # noqa

            model_path = self.DEFAULT_MODEL_PATH

            if not os.path.exists(model_path):
                logger.warning(
                    f"Model file not found at '{model_path}'. "
                    "Predictions will use mock data."
                )
                self._loaded = False
                return

            logger.info(f"Loading model from {model_path} …")
            self._model = tf.keras.models.load_model(model_path, compile=False)
            self._loaded = True
            logger.info("Model loaded. Input shape: %s", self._model.input_shape)

        except ImportError:
            logger.warning("TensorFlow not installed — running in MOCK mode.")
            self._loaded = False
        except Exception as exc:
            logger.error("Failed to load model: %s", exc, exc_info=True)
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL Image and return a prediction dict:
            {"fruit": str, "confidence": float, "quality": str}
        """
        if not self._loaded or self._model is None:
            return self._mock_predict(image)

        try:
            preprocessed = self._preprocess(image)
            predictions = self._model.predict(preprocessed, verbose=0)
            class_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][class_index])

            fruit_name = LABELS.get(class_index, f"Unknown ({class_index})")
            quality = infer_quality(confidence)

            return {
                "fruit": fruit_name,
                "confidence": round(confidence, 4),
                "quality": quality,
            }

        except Exception as exc:
            logger.error("Inference error: %s", exc, exc_info=True)
            raise

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize to 224×224 and apply ResNet50 preprocessing."""
        import tensorflow as tf

        image = image.resize(self.IMG_SIZE, Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32)

        # ResNet50 expects images preprocessed with preprocess_input
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        # Add batch dimension
        return np.expand_dims(img_array, axis=0)

    # ── Mock mode (no model file) ──────────────────────────────────────────────
    def _mock_predict(self, image: Image.Image) -> dict:
        """
        Returns a deterministic-looking fake prediction when the model is not
        available. Useful for testing the Android app without a trained model.
        """
        import hashlib

        # Use image pixel average as a cheap seed for a reproducible mock result
        arr = np.array(image.resize((32, 32)))
        seed_bytes = arr.tobytes()
        digest = int(hashlib.md5(seed_bytes).hexdigest(), 16)

        class_index = digest % len(LABELS)
        confidence = 0.50 + (digest % 50) / 100.0   # 0.50 – 0.99
        fruit_name = LABELS[class_index]
        quality = infer_quality(confidence)

        logger.info("MOCK prediction: %s %.2f %s", fruit_name, confidence, quality)
        return {
            "fruit": fruit_name,
            "confidence": round(confidence, 4),
            "quality": quality,
        }
