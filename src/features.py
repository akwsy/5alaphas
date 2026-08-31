"""Frozen CLIP ViT-L/14 feature extraction.

This is the only GPU-expensive step in the whole pipeline (spec SS8b). The
backbone never receives a gradient; every downstream operation -- training,
evaluation, inference -- runs on the 768-d vectors this module produces.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-large-patch14"
EMBED_DIM = 1024  # ViT-L/14 hidden size (verified by instantiation, not the 768 of ViT-B)
MAX_PARAMS = 2_000_000_000  # problem statement SS5.3 hard limit


class FeatureExtractor:
    """Wraps frozen CLIP. Construct once, reuse across many `embed_batch` calls."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(MODEL_ID).vision_model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params < MAX_PARAMS, (
            f"backbone has {n_params:,} params, exceeds the {MAX_PARAMS:,} limit"
        )
        self.n_params = n_params

        # fp16 + autocast on CUDA; T4 has no bf16 support (spec SS8) so we pin fp16
        # explicitly rather than letting autocast pick a dtype.
        self._amp_dtype = torch.float16 if self.device.startswith("cuda") else None

    @torch.no_grad()
    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """images -> (N, 768) float32 array of penultimate-layer CLIP features."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        if self._amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                out = self.model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
        else:
            out = self.model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
        # penultimate hidden state, CLS token, pre-projection -- matches the
        # Ojha/Cozzolino protocol (spec SS3.2)
        penultimate = out.hidden_states[-2][:, 0, :]
        return penultimate.float().cpu().numpy()


def extract_to_disk(
    pairs: Iterable[tuple[Image.Image, int]],
    out_path: Path,
    extractor: FeatureExtractor,
    batch_size: int = 32,
    transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    log_every: int = 500,
) -> None:
    """Stream (image, label) pairs through the extractor and save to out_path.

    A single paired stream -- rather than separate images/labels iterables --
    so a caller filtering out unreadable files (see src/data.py's load_image)
    cannot desync image N from label N.
    """
    feats: List[np.ndarray] = []
    labs: List[int] = []
    batch_imgs: List[Image.Image] = []
    batch_labs: List[int] = []
    n = 0

    def flush():
        nonlocal batch_imgs, batch_labs
        if not batch_imgs:
            return
        feats.append(extractor.embed_batch(batch_imgs))
        labs.extend(batch_labs)
        batch_imgs, batch_labs = [], []

    for img, lab in pairs:
        im = transform(img) if transform is not None else img
        batch_imgs.append(im)
        batch_labs.append(lab)
        if len(batch_imgs) >= batch_size:
            flush()
            n += batch_size
            if n % log_every == 0:
                print(f"  extracted {n}", file=sys.stderr, flush=True)
    flush()

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, EMBED_DIM), dtype=np.float32)
    y = np.array(labs, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)
    print(f"saved {out_path} X={X.shape} y={y.shape}", file=sys.stderr, flush=True)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["X"], d["y"]
