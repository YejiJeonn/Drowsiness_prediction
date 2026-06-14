from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from modeling import build_transforms, create_model


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = create_model(checkpoint["model_name"], num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    _, eval_tf = build_transforms(checkpoint.get("image_size", 224))
    return model, eval_tf, idx_to_class


@torch.no_grad()
def predict_image(model, transform, idx_to_class, image_path: str, device: torch.device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = idx_to_class[pred_idx]
    result = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    return pred_label, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/best_model.pt")
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, idx_to_class = load_model(args.model_path, device)
    label, probs = predict_image(model, tf, idx_to_class, args.image_path, device)

    print(f"prediction: {label}")
    print(probs)


if __name__ == "__main__":
    main()
