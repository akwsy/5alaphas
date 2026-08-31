"""Sample and render the highest-confidence false positives / false negatives
(spec SS5.3), testing the specific hypotheses the spec calls out:

  - FP on smooth/denoised real photos (the SS1.1 confound)
  - FP on heavily compressed or upscaled reals
  - FN on synthetic images that have been hard-compressed
  - FN on generators unlike SID_Set's training distribution

Usage:
    python -m src.error_analysis --artifacts-dir artifacts --out docs/error_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_image, load_samples
from src.features import FeatureExtractor


def make_contact_sheet(items: List[tuple[Path, float, int]], title: str, out_path: Path,
                        thumb: int = 200, cols: int = 5) -> None:
    """items: list of (path, pred_prob, true_label). Renders a grid with the
    predicted probability and true label burned into each thumbnail."""
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black")

    for i, (path, prob, true_label) in enumerate(items):
        img = Image.open(path).convert("RGB").resize((thumb, thumb))
        x, y = (i % cols) * thumb, 40 + (i // cols) * thumb
        sheet.paste(img, (x, y))
        label_str = "real" if true_label == 0 else "fake"
        draw.rectangle([x, y, x + thumb, y + 18], fill=(0, 0, 0))
        draw.text((x + 3, y + 2), f"pred={prob:.2f} true={label_str}", fill="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"wrote {out_path} ({len(items)} images)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--head", default="aug", choices=["aug", "noaug"])
    ap.add_argument("--out", type=Path, default=Path("docs/error_analysis"))
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    _, val_samples = load_samples(args.data_dir)
    clf = joblib.load(args.artifacts_dir / f"head_{args.head}.joblib")
    extractor = FeatureExtractor(device=args.device)

    paths, labels, probs = [], [], []
    batch_imgs, batch_paths, batch_labels = [], [], []

    def flush():
        nonlocal batch_imgs, batch_paths, batch_labels
        if not batch_imgs:
            return
        X = extractor.embed_batch(batch_imgs)
        margins = clf.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-margins))
        paths.extend(batch_paths)
        labels.extend(batch_labels)
        probs.extend(p.tolist())
        batch_imgs, batch_paths, batch_labels = [], [], []

    for s in val_samples:
        img = load_image(s.path)
        if img is None:
            continue
        batch_imgs.append(img)
        batch_paths.append(s.path)
        batch_labels.append(s.label)
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()

    labels_arr = np.array(labels)
    probs_arr = np.array(probs)

    # false positives: true=real (0), predicted confidently fake (high prob)
    fp_idx = np.where(labels_arr == 0)[0]
    fp_sorted = fp_idx[np.argsort(-probs_arr[fp_idx])][: args.top_n]

    # false negatives: true=fake (1), predicted confidently real (low prob)
    fn_idx = np.where(labels_arr == 1)[0]
    fn_sorted = fn_idx[np.argsort(probs_arr[fn_idx])][: args.top_n]

    fp_items = [(paths[i], probs_arr[i], labels_arr[i]) for i in fp_sorted]
    fn_items = [(paths[i], probs_arr[i], labels_arr[i]) for i in fn_sorted]

    make_contact_sheet(fp_items, f"Top false positives (head={args.head})",
                        args.out / "false_positives.png")
    make_contact_sheet(fn_items, f"Top false negatives (head={args.head})",
                        args.out / "false_negatives.png")

    # also dump a small text summary with the raw scores, for citing in the README
    summary = [f"# Error analysis (head={args.head})\n"]
    summary.append(f"Val set: {len(val_samples)} images, {int((labels_arr==0).sum())} real / "
                    f"{int((labels_arr==1).sum())} fake\n")
    summary.append(f"\n## Top {args.top_n} false positives (real misclassified as fake)\n")
    for p, prob, _ in fp_items:
        summary.append(f"- `{p.name}`  pred={prob:.4f}")
    summary.append(f"\n## Top {args.top_n} false negatives (fake misclassified as real)\n")
    for p, prob, _ in fn_items:
        summary.append(f"- `{p.name}`  pred={prob:.4f}")
    (args.out / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    print(f"wrote {args.out / 'SUMMARY.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
