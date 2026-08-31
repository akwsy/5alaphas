"""Demo UI for the submission video (spec SS9 task 9).

    python -m src.app --head artifacts/head_aug.joblib

Single-image upload -> P(AI-generated), plus one-click views of the same image
under each judged transform so the video can show robustness directly rather
than just quoting the results table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import FeatureExtractor
from src.transforms import EVAL_TRANSFORMS

_state = {}


def predict(image: Image.Image, transform_name: str):
    if image is None:
        return None, "Upload an image first."
    img = image.convert("RGB")
    tfm = EVAL_TRANSFORMS[transform_name]
    transformed = tfm(img)

    X = _state["extractor"].embed_batch([transformed])
    margin = _state["clf"].decision_function(X)[0]
    prob = 1.0 / (1.0 + np.exp(-margin))

    verdict = "AI-GENERATED" if prob >= 0.5 else "REAL"
    label = f"{verdict}  (P(fake) = {prob:.3f})  [after: {transform_name}]"
    return transformed, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=Path("artifacts/head_aug.joblib"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    if not args.head.exists():
        print(f"error: {args.head} not found. Run `python -m src.train` first.", file=sys.stderr)
        sys.exit(1)

    _state["clf"] = joblib.load(args.head)
    _state["extractor"] = FeatureExtractor(device=args.device)

    with gr.Blocks(title="AIGC Image Detector") as demo:
        gr.Markdown(
            "# Robust AI-Generated Image Detection\n"
            "Upload an image and optionally apply one of the judged robustness "
            "transforms to see whether the prediction holds up."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input image")
                transform_dd = gr.Dropdown(
                    choices=list(EVAL_TRANSFORMS.keys()), value="clean",
                    label="Apply transform before scoring",
                )
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Image as scored (post-transform)")
                out_label = gr.Textbox(label="Prediction")

        btn.click(predict, inputs=[inp, transform_dd], outputs=[out_img, out_label])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
