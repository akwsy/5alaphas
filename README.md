# AI-Generated Image Detection — TikTok TechJam

Binary classifier distinguishing real photographs from fully AI-generated images.

## Data

[`fetch_data.py`](fetch_data.py) streams [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
and writes a balanced subset to `data/raw/{real,fake}/`.

Label mapping (verified against `img_id` prefixes and mask presence):

| label | meaning | kept |
|---|---|---|
| 0 | real photograph | yes |
| 1 | fully synthetic | yes |
| 2 | tampered region | no — localization task, not our problem |

Two decisions in the loader are deliberate and should not be "cleaned up":

- **Everything saves as PNG.** Mixed formats let the model learn JPEG artifacts as
  a class signal — it scores ~99% locally and collapses on the real test set.
- **Centre-crop, never resize.** Resampling is a low-pass filter that smears the
  high-frequency generator fingerprints the detector depends on. Cropping leaves
  surviving pixels bit-exact.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/hf auth login          # HuggingFace read token
.venv/bin/python fetch_data.py   # resumable; re-run to top up
```

## Layout

| path | role | owner |
|---|---|---|
| `fetch_data.py` | dataset download | — |
| `src/transforms.py` | augmentation / preprocessing | |
| `src/data.py` | dataset + dataloaders | |
| `src/train.py` | training loop | |
| `src/evaluate.py` | metrics, confusion matrix | |
| `src/app.py` | demo UI | |
