"""Extract CLIP features and fit the linear head.

Usage:
    python -m src.train --data-dir data/raw --out-dir artifacts

Produces two fitted heads (spec SS5.2's augmentation ablation):
  artifacts/head_noaug.joblib   -- trained on clean, unaugmented features
  artifacts/head_aug.joblib     -- trained on src.transforms.train_augment features

and caches every embedding set to artifacts/features/*.npz so re-runs and
evaluate.py never recompute a CLIP forward pass.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_image, load_samples
from src.features import FeatureExtractor, extract_to_disk, load_features
from src.transforms import train_augment
import numpy.random as _npr


def fit_head(X: np.ndarray, y: np.ndarray) -> LinearSVC:
    """LinearSVC per spec SS3.3 -- ablated by Cozzolino et al. against LR,
    Mahalanobis, GNB and soft-NN and found the strongest single head."""
    clf = LinearSVC(C=1.0, max_iter=20000, dual="auto")
    clf.fit(X, y)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    feat_dir = args.out_dir / "features"
    train_samples, val_samples = load_samples(args.data_dir)
    print(f"train={len(train_samples)} val={len(val_samples)}", file=sys.stderr)

    extractor = FeatureExtractor(device=args.device)
    print(f"backbone params={extractor.n_params:,} device={extractor.device}", file=sys.stderr)

    # -- train features, two variants for the ablation -----------------------
    for tag, augment in [("noaug", False), ("aug", True)]:
        out = feat_dir / f"train_{tag}.npz"
        if out.exists():
            print(f"skip existing {out}", file=sys.stderr)
            continue
        t0 = time.time()
        extract_to_disk(
            _paired_stream(train_samples, augment),
            out_path=out,
            extractor=extractor,
            batch_size=args.batch_size,
        )
        print(f"{tag}: {time.time()-t0:.1f}s", file=sys.stderr)

    # -- val features, clean only (used for threshold calibration / sanity) --
    val_out = feat_dir / "val_clean.npz"
    if not val_out.exists():
        extract_to_disk(
            _paired_stream(val_samples, augment=False),
            out_path=val_out,
            extractor=extractor,
            batch_size=args.batch_size,
        )

    # -- fit both heads --------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag in ("noaug", "aug"):
        X, y = load_features(feat_dir / f"train_{tag}.npz")
        clf = fit_head(X, y)
        train_acc = clf.score(X, y)
        head_path = args.out_dir / f"head_{tag}.joblib"
        joblib.dump(clf, head_path)
        print(f"head_{tag}: train_acc={train_acc:.4f} -> {head_path}", file=sys.stderr)

        Xv, yv = load_features(val_out)
        val_acc = clf.score(Xv, yv)
        print(f"head_{tag}: val_acc={val_acc:.4f}", file=sys.stderr)


def _paired_stream(samples, augment: bool, seed: int = 0):
    """Yield (image, label) pairs, loading each file exactly once. A skipped
    (unreadable) file simply contributes no pair -- there is no separate label
    stream to fall out of sync with."""
    rng = _npr.default_rng(seed)
    for s in samples:
        img = load_image(s.path)
        if img is None:
            continue
        yield (train_augment(img, rng) if augment else img), s.label


if __name__ == "__main__":
    main()
