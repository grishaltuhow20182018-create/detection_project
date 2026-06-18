from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
FAST_MODEL_PATH = ROOT / "fast_model.pt"
THINKING_MODEL_PATH = ROOT / "thinking_model.pt"

app = Flask(__name__)

FAST_MODEL = YOLO(FAST_MODEL_PATH)
THINKING_MODEL = YOLO(THINKING_MODEL_PATH)


def choose_model(request_text: str, image: Image.Image) -> tuple[YOLO, str, str]:
    normalized = request_text.lower().strip()
    blurry_keywords = {
        "blurry",
        "blur",
        "low quality",
        "hard",
        "difficult",
        "small objects",
        "tiny objects",
        "accurate",
        "quality",
        "maximum",
        "best",
        "smart",
        "complex",
        "размытая",
        "точная",
        "максимум",
        "лучше",
    }
    fast_keywords = {
        "fast",
        "quick",
        "speed",
        "rapid",
        "simple",
        "clear",
        "easy",
        "быстро",
        "простой",
        "четкая",
    }

    if any(keyword in normalized for keyword in blurry_keywords):
        return THINKING_MODEL, "thinking", "The request suggests a harder or lower-quality image, so I picked the more accurate model."

    if any(keyword in normalized for keyword in fast_keywords):
        return FAST_MODEL, "fast", "The request asks for speed or a simple image, so I picked the faster model."

    width, height = image.size
    if min(width, height) < 900:
        return THINKING_MODEL, "thinking", "The image is relatively small, so I picked the more accurate model to preserve detail."

    return FAST_MODEL, "fast", "No strong quality warning was given, so I picked the faster model for a quick result."


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/")
def index() -> str:
    return send_file(ROOT / "index.html")


@app.post("/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Please choose an image file."}), 400

    request_text = request.form.get("request_text", "")
    confidence = float(request.form.get("confidence", 0.25) or 0.25)

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    model, model_name, reason = choose_model(request_text, image)
    results = model.predict(source=image, conf=confidence, verbose=False)
    result = results[0]

    detections = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "label": names.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf.item()), 4),
                    "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )

    annotated_image = Image.fromarray(result.plot()[:, :, ::-1])

    return jsonify(
        {
            "model": model_name,
            "reason": reason,
            "detections": detections,
            "count": len(detections),
            "image": image_to_base64(annotated_image),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)