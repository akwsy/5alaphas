"""Download a balanced real/synthetic subset of SID_Set to data/raw/.

Labels (verified against img_id prefixes and mask presence):
    0 = real, 1 = fully synthetic, 2 = tampered (dropped -- localization task)
Everything is saved as PNG so no class carries JPEG artifacts the other lacks.
"""
import sys
from pathlib import Path

from datasets import load_dataset

IMG_FIELD, LABEL_FIELD = "image", "label"
KEEP = {0: "real", 1: "fake"}
N_PER_CLASS = 6000
CROP = 512              # centre-crop size; originals are mostly 1024x1024

OUT = Path("data/raw")
for name in KEEP.values():
    (OUT / name).mkdir(parents=True, exist_ok=True)

# resume: count what's already on disk so a re-run tops up instead of restarting
counts = {name: len(list((OUT / name).glob("*.png"))) for name in KEEP.values()}
print(f"resuming from {counts}", flush=True)

ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)

seen = 0
try:
    for ex in ds:
        seen += 1
        lab = ex[LABEL_FIELD]
        if lab not in KEEP:
            continue
        name = KEEP[lab]
        if counts[name] >= N_PER_CLASS:
            if all(c >= N_PER_CLASS for c in counts.values()):
                break
            continue

        dest = OUT / name / f"{ex['img_id']}.png"
        if dest.exists():
            continue
        try:
            img = ex[IMG_FIELD].convert("RGB")
            # centre-crop, never resize: resampling low-pass-filters away the
            # high-frequency generator artifacts the detector relies on.
            w, h = img.size
            s = min(w, h, CROP)
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
            img.save(dest)
        except Exception as e:                    # one bad row must not kill the night
            print(f"skip {ex.get('img_id')}: {e}", file=sys.stderr, flush=True)
            continue

        counts[name] += 1
        if seen % 500 == 0:
            print(f"seen={seen} {counts}", flush=True)
except KeyboardInterrupt:
    print("interrupted", flush=True)

complete = all(c >= N_PER_CLASS for c in counts.values())
print(f"{'done' if complete else 'PARTIAL'} seen={seen} {counts}", flush=True)
sys.exit(0 if complete else 1)
