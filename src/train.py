from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from modeling import MODEL_PRESETS, build_transforms, create_model


# 기본 실행 모델입니다. 다른 모델 비교 시 아래 한 줄만 바꾸면 됩니다.
DEFAULT_MODEL = "efficientnet_b0"
# DEFAULT_MODEL = "resnet18"
# DEFAULT_MODEL = "mobilenet_v3_small"


def get_loaders(data_dir: Path, batch_size: int, num_workers: int, image_size: int) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    train_tf, eval_tf = build_transforms(image_size=image_size)

    datasets = {
        "train": ImageFolder(data_dir / "train", transform=train_tf),
        "val": ImageFolder(data_dir / "val", transform=eval_tf),
        "test": ImageFolder(data_dir / "test", transform=eval_tf),
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }
    return loaders, datasets["train"].class_to_idx


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool, use_amp: bool) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    preds, targets = [], []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and train)

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = accuracy_score(targets, preds)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    preds, targets = [], []
    inference_times = []

    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device)
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed / images.size(0))

        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(targets, preds),
        "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, preds).tolist(),
        "avg_inference_time_sec_per_image": float(sum(inference_times) / max(len(inference_times), 1)),
        "fps": float(1.0 / (sum(inference_times) / max(len(inference_times), 1))),
    }


def plot_confusion_matrix(cm, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "outputs"))
    parser.add_argument("--model_name", default=DEFAULT_MODEL, choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--feature_extract", action="store_true", help="Freeze backbone and train only classifier head")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model_name]
    epochs = args.epochs or preset["epochs"]
    batch_size = args.batch_size or preset["batch_size"]
    lr = args.lr or preset["lr"]
    weight_decay = args.weight_decay or preset["weight_decay"]

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    loaders, class_to_idx = get_loaders(data_dir, batch_size, args.num_workers, args.image_size)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = create_model(
        model_name=args.model_name,
        num_classes=len(class_to_idx),
        pretrained=not args.no_pretrained,
        feature_extract=args.feature_extract,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    bad_epochs = 0
    history = []
    best_path = output_dir / "best_model.pt"

    print(f"device={device}, model={args.model_name}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"class_to_idx={class_to_idx}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(model, loaders["train"], criterion, optimizer, device, True, use_amp)
        val_loss, val_acc = run_one_epoch(model, loaders["val"], criterion, optimizer, device, False, False)
        scheduler.step(val_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "class_to_idx": class_to_idx,
                    "image_size": args.image_size,
                    "pretrained": not args.no_pretrained,
                },
                best_path,
            )
            print(f"saved best model: {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)

    checkpoint = torch.load(best_path, map_location=device)
    model = create_model(checkpoint["model_name"], num_classes=len(checkpoint["class_to_idx"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, loaders["test"], device)
    metrics["model_name"] = args.model_name
    metrics["class_to_idx"] = class_to_idx
    metrics["idx_to_class"] = idx_to_class
    metrics["hyperparameters"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "image_size": args.image_size,
        "pretrained": not args.no_pretrained,
        "feature_extract": args.feature_extract,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(metrics["confusion_matrix"], [idx_to_class[i] for i in range(len(idx_to_class))], output_dir / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
