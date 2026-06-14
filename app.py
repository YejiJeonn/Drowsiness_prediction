from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from PIL import Image

from src.modeling import build_transforms, create_model

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
TRANSFORM = None
IDX_TO_CLASS = None


def load_once():
    global MODEL, TRANSFORM, IDX_TO_CLASS
    if MODEL is not None:
        return
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = checkpoint["class_to_idx"]
    IDX_TO_CLASS = {v: k for k, v in class_to_idx.items()}
    MODEL = create_model(checkpoint["model_name"], num_classes=len(class_to_idx), pretrained=False)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()
    _, TRANSFORM = build_transforms(checkpoint.get("image_size", 224))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_path": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict():
    load_once()
    if "file" not in request.files:
        return jsonify({"error": "upload a file field named 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = MODEL(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_label = IDX_TO_CLASS[pred_idx]
        probabilities = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))}
        return jsonify({
            "prediction": pred_label,
            "probabilities": probabilities,
            "drowsy_probability": probabilities.get("drowsy"),
            "normal_probability": probabilities.get("normal"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_once()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
