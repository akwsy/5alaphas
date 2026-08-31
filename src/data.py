"""Dataset assembly and reproducible splits.

Splits are assigned by hashing the file's basename, not by random shuffling, so
the train/val partition is identical across machines and reruns without a
shared seed file (spec SS2.3).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

from PIL import Image

VAL_FRACTION = 0.20
CLASSES = {"real": 0, "fake": 1}  # 0 = real, 1 = AI-generated (matches predict.py's `pred` semantics)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # 0 = real, 1 = fake


def _split_for(name: str) -> Literal["train", "val"]:
    """Deterministic split assignment: MD5(basename) mod 100 < 20 => val."""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < int(VAL_FRACTION * 100) else "train"


def load_samples(data_dir: Path) -> Tuple[List[Sample], List[Sample]]:
    """Scan data_dir/{real,fake}/*.png and return (train, val) sample lists."""
    train, val = [], []
    for cls_name, label in CLASSES.items():
        cls_dir = data_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*.png")):
            sample = Sample(path=p, label=label)
            (val if _split_for(p.name) == "val" else train).append(sample)
    return train, val


def load_image(path: Path) -> Image.Image | None:
    """Robust image load. Returns None on any failure rather than raising --
    callers decide whether a missing image is fatal (training) or should be
    skipped with a warning (inference on arbitrary user directories)."""
    try:
        img = Image.open(path)
        img.load()  # force decode now, so truncated files fail here, not later
        return img.convert("RGB")
    except Exception:
        return None
