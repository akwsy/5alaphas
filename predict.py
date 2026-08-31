#!/usr/bin/env python3
"""Scored deliverable (spec SS4): image directory -> JSON of {image_path, pred}.

    python predict.py --input-dir path/to/images --output preds.json

`pred` is P(image is AI-generated) in [0,1], not a hard label. image_path is
relative to --input-dir so the output is portable across machines.

Never crashes on a bad file: a corrupt/unreadable image gets pred=0.5 and a
stderr warning rather than aborting the whole run (spec SS4 -- a crash on
image 4,000 of 5,000 must not lose the other 3,999 predictions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
from PIL import Image, ImageFile

# Tolerate mildly truncated JPEGs rather than raising on load -- common in
# real-world image directories (partial downloads, interrupted uploads).
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import FeatureExtractor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_HEAD = Path("artifacts/head_aug.joblib")


def find_images(input_dir: Path) -> List[Path]:
    """Recurse input_dir for image files. p.is_file() already follows symlinks
    and is False for a broken link, so this naturally includes valid symlinks
    and skips dangling ones without special-casing them."""
    return [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def safe_load(path: Path) -> Image.Image | None:
    """Load and normalise to RGB. Handles the edge cases enumerated in spec
    SS4: EXIF rotation, greyscale/CMYK, alpha channels, animated formats
    (first frame only), truncated files, tiny images."""
    try:
        img = Image.open(path)
        img.seek(0)  # first frame of animated GIF/WebP
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)  # honour EXIF rotation before it's lost
        if img is None:
            return None
        img = img.convert("RGB")  # handles L, CMYK, RGBA, P, etc. uniformly
        img.load()
        return img
    except Exception as e:
        print(f"warning: failed to load {path}: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--head", type=Path, default=DEFAULT_HEAD,
                     help="fitted LinearSVC head (default: artifacts/head_aug.joblib)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda/cpu; auto-detected if omitted")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"error: --input-dir {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not args.head.exists():
        print(f"error: head checkpoint not found at {args.head}. Run `python -m src.train` first.",
              file=sys.stderr)
        sys.exit(1)

    clf = joblib.load(args.head)
    extractor = FeatureExtractor(device=args.device)
    print(f"backbone: {extractor.n_params:,} params, device={extractor.device}", file=sys.stderr)

    paths = find_images(args.input_dir)
    print(f"found {len(paths)} candidate images under {args.input_dir}", file=sys.stderr)

    results = []
    batch_imgs: List[Image.Image] = []
    batch_paths: List[Path] = []
    n_failed = 0

    def flush():
        nonlocal batch_imgs, batch_paths
        if not batch_imgs:
            return
        X = extractor.embed_batch(batch_imgs)
        # decision_function is an unbounded margin; squash to [0,1] with a
        # logistic so `pred` is a genuine confidence, not a raw SVM score
        margins = clf.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-margins))
        for p, prob in zip(batch_paths, probs):
            rel = p.relative_to(args.input_dir).as_posix()
            results.append({"image_path": rel, "pred": round(float(prob), 4)})
        batch_imgs, batch_paths = [], []

    for i, path in enumerate(paths):
        img = safe_load(path)
        if img is None:
            n_failed += 1
            rel = path.relative_to(args.input_dir).as_posix()
            results.append({"image_path": rel, "pred": 0.5})
            continue
        batch_imgs.append(img)
        batch_paths.append(path)
        if len(batch_imgs) >= args.batch_size:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(paths)}", file=sys.stderr)
    flush()

    # deterministic output order regardless of batching/flush timing
    results.sort(key=lambda r: r["image_path"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {len(results)} predictions ({n_failed} failed loads -> pred=0.5) to {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
