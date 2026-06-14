from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

NORMAL_KEYWORDS = [
    "no_yawn", "no-yawn", "noyawn", "no yawn", "not_yawn", "non_yawn", "non-yawn",
    "normal", "awake", "open", "open_eye", "open-eyes", "openeyes", "opened",
]
DROWSY_KEYWORDS = [
    "closed", "close", "closed_eye", "closed-eyes", "closedeyes", "sleep", "sleepy",
    "drowsy", "yawn", "yawning",
]


def label_from_path(path: Path) -> Optional[str]:
    """Infer binary label from a file path.

    Final labels:
    - drowsy: closed eye or yawn
    - normal: open eye or no-yawn/normal

    If your downloaded Kaggle folder has different names, rename the folders to include
    open/closed/yawn/no_yawn or make a custom CSV and modify this function.
    """
    text = path.as_posix().lower().replace("\\", "/")

    # no_yawn contains the word yawn, so normal keywords must be checked first.
    if any(k in text for k in NORMAL_KEYWORDS):
        return "normal"
    if any(k in text for k in DROWSY_KEYWORDS):
        return "drowsy"
    return None


def iter_files(raw_dir: Path) -> Iterable[Path]:
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS):
            yield p


def save_image(src: Path, dst: Path, image_size: int) -> bool:
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((image_size, image_size))
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=95)
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def extract_video_frames(src: Path, temp_dir: Path, label: str, image_size: int, every_n_frames: int) -> list[Path]:
    cap = cv2.VideoCapture(str(src))
    saved = []
    frame_idx = 0
    saved_idx = 0
    safe_stem = src.stem.replace(" ", "_").replace("/", "_")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((image_size, image_size))
            out_path = temp_dir / label / f"{safe_stem}_frame_{saved_idx:06d}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, quality=90)
            saved.append(out_path)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return saved


def copy_split(samples: list[tuple[Path, str]], out_dir: Path, split: str, image_size: int) -> list[dict]:
    rows = []
    for i, (src, label) in enumerate(tqdm(samples, desc=f"copy {split}")):
        dst = out_dir / split / label / f"{label}_{i:07d}.jpg"
        if save_image(src, dst, image_size):
            rows.append({"split": split, "label": label, "path": str(dst), "source": str(src)})
    return rows


def balance_samples(samples: list[tuple[Path, str]], seed: int) -> list[tuple[Path, str]]:
    by_label = {"normal": [], "drowsy": []}
    for item in samples:
        by_label[item[1]].append(item)

    n = min(len(by_label["normal"]), len(by_label["drowsy"]))
    rng = random.Random(seed)
    balanced = rng.sample(by_label["normal"], n) + rng.sample(by_label["drowsy"], n)
    rng.shuffle(balanced)
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw", help="Folder containing downloaded Kaggle datasets")
    parser.add_argument("--out_dir", default="data/processed", help="Output ImageFolder dataset")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--video_every_n_frames", type=int, default=15, help="Frame interval for YawDD videos")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_balance", action="store_true", help="Disable 50:50 class balancing")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    temp_frames = out_dir / "_temp_video_frames"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    image_samples: list[tuple[Path, str]] = []
    skipped = []

    for path in tqdm(list(iter_files(raw_dir)), desc="scan raw files"):
        label = label_from_path(path)
        if label is None:
            skipped.append(str(path))
            continue

        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            image_samples.append((path, label))
        elif suffix in VIDEO_EXTS:
            frames = extract_video_frames(path, temp_frames, label, args.image_size, args.video_every_n_frames)
            image_samples.extend((frame, label) for frame in frames)

    if not image_samples:
        raise RuntimeError(
            "No labeled samples found. Check folder names. They should include open/closed/yawn/no_yawn/normal."
        )

    if not args.no_balance:
        image_samples = balance_samples(image_samples, args.seed)

    labels = [label for _, label in image_samples]
    train_samples, temp_samples = train_test_split(
        image_samples,
        test_size=args.val_ratio + args.test_ratio,
        stratify=labels,
        random_state=args.seed,
    )
    temp_labels = [label for _, label in temp_samples]
    val_relative = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=1 - val_relative,
        stratify=temp_labels,
        random_state=args.seed,
    )

    metadata = []
    metadata.extend(copy_split(train_samples, out_dir, "train", args.image_size))
    metadata.extend(copy_split(val_samples, out_dir, "val", args.image_size))
    metadata.extend(copy_split(test_samples, out_dir, "test", args.image_size))

    if temp_frames.exists():
        shutil.rmtree(temp_frames)

    with open(out_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "path", "source"])
        writer.writeheader()
        writer.writerows(metadata)

    counts = {}
    for row in metadata:
        key = (row["split"], row["label"])
        counts[key] = counts.get(key, 0) + 1

    print("\nDataset prepared:")
    for key, value in sorted(counts.items()):
        print(f"  {key[0]:5s} / {key[1]:7s}: {value}")
    print(f"\nmetadata: {out_dir / 'metadata.csv'}")
    print(f"skipped unlabeled files: {len(skipped)}")


if __name__ == "__main__":
    main()
