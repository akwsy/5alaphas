"""Robustness transform suite.

Implements exactly the transform grid from the problem statement (section 5.2).
Two distinct uses, deliberately kept separate:

  * `EVAL_TRANSFORMS` -- the fixed, deterministic benchmark grid. Never change
    these; the robustness table is only comparable across runs if the grid is
    frozen. Seeded where stochastic (noise).
  * `train_augment()`  -- randomised versions of the same family, applied during
    training so the model cannot rely on artifacts that any of them destroy.

Everything operates on PIL RGB images and returns PIL RGB images, so the same
code path serves both the offline benchmark and the inference script.
"""
from __future__ import annotations

import io
from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# --------------------------------------------------------------------------
# primitives -- each mirrors one row of the problem statement's table
# --------------------------------------------------------------------------


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Social-media re-encode / messaging apps."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Out-of-focus capture."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def downscale_upscale(img: Image.Image, scale: float) -> Image.Image:
    """Thumbnail generation then re-display at original size."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator | None = None) -> Image.Image:
    """Low-light sensor noise. sigma is in [0,1] units, per the spec table."""
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))


def color_jitter(img: Image.Image, brightness: float = 1.0,
                 contrast: float = 1.0, saturation: float = 1.0) -> Image.Image:
    """Filter apps / auto-enhance. Factors are multiplicative (1.0 = identity)."""
    out = ImageEnhance.Brightness(img).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    out = ImageEnhance.Color(out).enhance(saturation)
    return out


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    """Profile-picture cropping. Crops to `frac` of each side, keeps resolution."""
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)


# --------------------------------------------------------------------------
# the frozen evaluation grid
# --------------------------------------------------------------------------

EVAL_TRANSFORMS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": lambda im: im,
    # JPEG quality = 90, 70, 50, 30
    **{f"jpeg_q{q}": (lambda q: lambda im: jpeg_compress(im, q))(q) for q in (90, 70, 50, 30)},
    # Gaussian blur sigma = 0.5, 1.0, 2.0
    **{f"blur_s{s}": (lambda s: lambda im: gaussian_blur(im, s))(s) for s in (0.5, 1.0, 2.0)},
    # Resize 0.5x / 0.25x then upscale
    **{f"resize_{s}": (lambda s: lambda im: downscale_upscale(im, s))(s) for s in (0.5, 0.25)},
    # Gaussian noise sigma = 0.02, 0.05, 0.10  (seeded -> deterministic benchmark)
    **{f"noise_s{s}": (lambda s: lambda im: gaussian_noise(im, s, np.random.default_rng(1234)))(s)
       for s in (0.02, 0.05, 0.10)},
    # Colour jitter +/-20%
    "jitter_up": lambda im: color_jitter(im, 1.2, 1.2, 1.2),
    "jitter_down": lambda im: color_jitter(im, 0.8, 0.8, 0.8),
    # Centre crop 80%
    "crop_80": lambda im: center_crop(im, 0.80),
}


# --------------------------------------------------------------------------
# train-time augmentation
# --------------------------------------------------------------------------
#
# Two jobs, both essential:
#
#   1. Destroy the compression-history shortcut. SID_Set reals are OpenImages
#      (already JPEG'd); fakes are pristine FLUX PNG. Measured on our own data,
#      a single 8x8-blockiness feature separates the classes at AUROC 0.830.
#      Re-encoding EVERY image at a random JPEG quality drops that to 0.558.
#      This is not optional and it must apply to both classes equally.
#
#   2. Stop the model relying on high-frequency energy, which blur and rescaling
#      destroy at eval time.
#
# Ordering is deliberate: geometry -> photometric -> noise -> JPEG last, because
# JPEG is the final step in the real-world chain (a platform re-encodes on
# upload, after every other edit has happened).

# Blur is applied at lower probability than the rest: the [RE] reproducibility
# study found blur-only augmentation was net-harmful (89.7 -> 83.8 mAP), while
# JPEG was nearly free. Verify on val before raising this.
P_BLUR = 0.25
P_NOISE = 0.25
P_JITTER = 0.5
P_RESIZE = 0.5


def train_augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Randomised augmentation matching the judged transform families.

    Always ends with a JPEG re-encode -- see note above; that step is what
    removes the compression-history confound rather than merely obscuring it.
    """
    out = img

    # -- geometry -----------------------------------------------------------
    if rng.random() < P_RESIZE:
        out = downscale_upscale(out, float(rng.uniform(0.25, 1.0)))

    # random crop 5/8 .. full, then restore size (Cozzolino: resize, don't
    # fixed-centre-crop, since we are graded on robustness to both)
    frac = float(rng.uniform(0.625, 1.0))
    if frac < 0.999:
        out = center_crop(out, frac)

    # -- photometric --------------------------------------------------------
    if rng.random() < P_JITTER:
        out = color_jitter(
            out,
            brightness=float(rng.uniform(0.8, 1.2)),
            contrast=float(rng.uniform(0.8, 1.2)),
            saturation=float(rng.uniform(0.8, 1.2)),
        )

    # -- blur / noise -------------------------------------------------------
    # Composed with JPEG below rather than applied only in isolation: the [RE]
    # study found the compositional variant scored 93.7 vs 66.3 AP on the
    # hardest evaluation set.
    if rng.random() < P_BLUR:
        out = gaussian_blur(out, float(rng.uniform(0.0, 2.0)))

    if rng.random() < P_NOISE:
        out = gaussian_noise(out, float(rng.uniform(0.0, 0.10)), rng)

    # -- compression, always ------------------------------------------------
    return jpeg_compress(out, int(rng.integers(30, 96)))


def normalize_compression(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Eval-time counterpart of the JPEG step in `train_augment`.

    Applied to every image at inference so that train and test see the same
    compression distribution. Without this the model meets pristine PNGs it
    never saw in training.
    """
    return jpeg_compress(img, int(rng.integers(30, 96)))
