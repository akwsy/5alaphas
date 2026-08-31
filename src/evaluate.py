"""Robustness grid: score a fitted head across every judged transform and every
evaluation dataset, per spec SS5.1-5.2.

Usage:
    python -m src.evaluate --artifacts-dir artifacts --out docs/RESULTS.md

Emits AUROC, accuracy @ a threshold calibrated once on clean val, TPR@1%FPR and
TPR@5%FPR, for both fitted heads (noaug/aug -- the ablation) across:
  - the 16-cell transform grid on SID_Set val (in-distribution robustness)
  - the `laion_matched` config of the organisers' eval subset (headline OOD)
  - the `cross_generator` config (multi-generator generalisation)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_image, load_samples
from src.features import FeatureExtractor
from src.transforms import EVAL_TRANSFORMS


def calibrate_threshold(clf, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Youden's J threshold on clean val, held fixed across every transform
    and every dataset (spec SS5.1: re-tuning per cell would hide exactly the
    threshold-drift failure this project is about)."""
    scores = clf.decision_function(X_val)
    fpr, tpr, thr = roc_curve(y_val, scores)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def score_cell(clf, threshold: float, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    scores = clf.decision_function(X)
    preds = (scores >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y, scores) if len(set(y.tolist())) > 1 else float("nan"),
        "accuracy": float((preds == y).mean()),
        "tpr@1%fpr": tpr_at_fpr(y, scores, 0.01),
        "tpr@5%fpr": tpr_at_fpr(y, scores, 0.05),
    }


def embed_samples(
    samples,
    extractor: FeatureExtractor,
    transform: Callable[[Image.Image], Image.Image] | None = None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    imgs, labs, feats = [], [], []

    def flush():
        nonlocal imgs, labs
        if imgs:
            feats.append(extractor.embed_batch(imgs))
        imgs = []

    for s in samples:
        img = load_image(s.path)
        if img is None:
            continue
        imgs.append(transform(img) if transform else img)
        labs.append(s.label)
        if len(imgs) >= batch_size:
            flush()
    flush()
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, extractor.model.config.hidden_size))
    return X, np.array(labs, dtype=np.int64)


def run_grid_on_val(extractor, val_samples, heads, out_lines, batch_size):
    for cell_name, tfm in EVAL_TRANSFORMS.items():
        X, y = embed_samples(val_samples, extractor, transform=tfm, batch_size=batch_size)
        if len(y) == 0:
            continue
        for tag, (clf, thr) in heads.items():
            m = score_cell(clf, thr, X, y)
            out_lines.append(
                f"| val-ID | {cell_name} | {tag} | {m['auroc']:.3f} | {m['accuracy']:.3f} "
                f"| {m['tpr@1%fpr']:.3f} | {m['tpr@5%fpr']:.3f} |"
            )
        print(f"  grid cell '{cell_name}' done (n={len(y)})", file=sys.stderr)


def run_ood_config(extractor, config: str, heads, out_lines, batch_size):
    from datasets import load_dataset

    try:
        ds = load_dataset("techjam-aigc/wildfake-eval-subset", config, split="validation")
    except Exception as e:
        print(f"  SKIP {config}: could not load ({e})", file=sys.stderr)
        return

    imgs, labs, feats = [], [], []

    def flush():
        nonlocal imgs
        if imgs:
            feats.append(extractor.embed_batch(imgs))
        imgs = []

    for row in ds:
        imgs.append(row["image"].convert("RGB"))
        labs.append(int(row["label"]))
        if len(imgs) >= batch_size:
            flush()
    flush()
    X = np.concatenate(feats, axis=0)
    y = np.array(labs, dtype=np.int64)

    for tag, (clf, thr) in heads.items():
        m = score_cell(clf, thr, X, y)
        out_lines.append(
            f"| {config} | clean | {tag} | {m['auroc']:.3f} | {m['accuracy']:.3f} "
            f"| {m['tpr@1%fpr']:.3f} | {m['tpr@5%fpr']:.3f} |"
        )
    print(f"  OOD config '{config}' done (n={len(y)})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-ood", action="store_true", help="skip the HF download for a fast local check")
    args = ap.parse_args()

    _, val_samples = load_samples(args.data_dir)
    extractor = FeatureExtractor(device=args.device)

    # load both heads, calibrate each independently on clean val (spec SS5.1)
    from src.features import load_features

    Xv_clean, yv_clean = load_features(args.artifacts_dir / "features" / "val_clean.npz")
    heads = {}
    for tag in ("noaug", "aug"):
        clf = joblib.load(args.artifacts_dir / f"head_{tag}.joblib")
        thr = calibrate_threshold(clf, Xv_clean, yv_clean)
        heads[tag] = (clf, thr)
        print(f"head_{tag}: calibrated threshold={thr:.4f}", file=sys.stderr)

    lines: List[str] = [
        "# Robustness Results",
        "",
        "Auto-generated by `src/evaluate.py`. Threshold calibrated once on clean",
        "SID_Set val per head (Youden's J), held fixed across every row.",
        "",
        "| dataset | cell | head | AUROC | Acc@thr | TPR@1%FPR | TPR@5%FPR |",
        "|---|---|---|---|---|---|---|",
    ]

    print("=== 16-cell grid on SID_Set val ===", file=sys.stderr)
    run_grid_on_val(extractor, val_samples, heads, lines, args.batch_size)

    if not args.skip_ood:
        print("=== OOD: laion_matched (headline) ===", file=sys.stderr)
        run_ood_config(extractor, "laion_matched", heads, lines, args.batch_size)
        print("=== OOD: cross_generator ===", file=sys.stderr)
        run_ood_config(extractor, "cross_generator", heads, lines, args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
