# Project Specification — Robust AIGC Image Detection

**TikTok TechJam 2026, Problem Statement 5**

Target: an image-level AI-generated-content detector that holds accuracy under
real-world post-processing, evaluated on a generator it never saw in training.

---

## 0. Non-negotiables (from the problem statement)

| Requirement | Source | Status |
|---|---|---|
| Model < 2B parameters | §5.3 Limits | Backbone chosen to comply; asserted in code |
| Script: image dir → JSON of `{image_path, pred}` | §5.5.2 | `predict.py`, the primary scored artifact |
| Public GitHub repo, commented, with README | §5.5.2 | — |
| Robustness table: clean vs transformed | §5.5.4 | `evaluate.py` emits it |
| Error analysis: representative FP/FN | §5.5.5 | `error_analysis.py` |
| Demo video, public on YouTube | §5.5.3 | — |
| Do **not** train on the WildFake validation subset | §5.4 | Enforced by dataset separation |

`pred` is a **confidence score in [0,1]** = P(image is AI-generated). Not a
hard label. The JSON schema is fixed:

```json
[{"image_path": "imgs/a.jpg", "pred": 0.9213}, {"image_path": "imgs/b.jpg", "pred": 0.0142}]
```

---

## 1. The core insight (measured on our own data)

There are **two** exploitable shortcuts in this dataset. Both were measured, not
assumed, and the second one is the project's biggest risk.

### 1.1 Shortcut A — synthetic images are smoother

High-frequency energy (mean Laplacian response), 150 images/class:

| Class | High-freq energy | Contrast | Brightness |
|---|---|---|---|
| Real | 13.69 +/- 10.42 | 63.53 | 107.65 |
| Fake | 8.03 +/- 4.46 | 63.01 | 95.01 |

Synthetic images carry **41% less high-frequency energy**. Under the judged
transforms the absolute gap collapses but Cohen's d stays flat:

| Transform | Real HF | Fake HF | Gap | Cohen's d |
|---|---|---|---|---|
| clean | 14.35 | 7.88 | 6.47 | 0.77 |
| jpeg q90 | 14.64 | 8.26 | 6.38 | 0.72 |
| jpeg q30 | 11.98 | 7.20 | 4.78 | 0.70 |
| blur s1.0 | 4.51 | 3.00 | 1.51 | 0.85 |
| blur s2.0 | 1.65 | 1.27 | 0.38 | 0.75 |
| resize 0.25x | 2.15 | 1.63 | 0.52 | 0.71 |

d ~= 0.8 is only ~65% accuracy, so this is **not a detector** -- but it is the
cheapest signal available on clean data, so a model will anchor on it and then
mis-threshold once blur compresses its dynamic range.

### 1.2 Shortcut B — compression history (CRITICAL)

SID_Set reals are OpenImages (Flickr-sourced, **already JPEG-compressed**);
fakes are FLUX output (**pristine 1024px PNG**). Saving everything as PNG does
**not** fix this -- JPEG blocking is baked into pixel values, not the container.

Measured 8x8 block-boundary energy ratio, 200 images/class:

| Preprocessing | Real | Fake | **AUROC from this single feature** |
|---|---|---|---|
| As-downloaded PNG | 1.091 | 0.991 | **0.830** |
| All re-encoded JPEG q75 | 1.300 | 1.227 | 0.606 |
| All re-encoded JPEG q50 | 1.514 | 1.426 | 0.578 |
| **Random QF per image [30,95]** | 1.396 | 1.322 | **0.558** |

**A single hand-crafted feature separates the classes at AUROC 0.830.** A network
will find this in one epoch, report ~99% val accuracy, and have learned
"JPEG artifacts => real". On the organisers' eval (COCO reals, which *are* JPEG;
DALL-E fakes, which may be clean PNG) that model would still look plausible for
entirely the wrong reason -- and would invert the moment anyone compresses a fake.

**Mandatory mitigation, applied identically to both classes, at train AND eval
time: re-encode every image at a random JPEG quality in [30, 95] before it
reaches the model.** This is the single highest-value change in the pipeline; our
measurement shows it takes the shortcut from 0.830 to 0.558 (near chance).

Corollary: geometry must be normalised too. FLUX fakes are fixed-square;
OpenImages reals are variable aspect. Random-resized-crop both classes to
identical geometry so size cannot leak the label.

## 2. Data

### 2.1 Training (in hand)

`data/raw/` — 12,000 images, 6,000 real + 6,000 synthetic, from SID_Set.

**Provenance (corrects an earlier assumption): SID_Set fakes are FLUX, not
Stable Diffusion; reals are OpenImages V7 (Flickr-sourced).** So our train->test
gap is **FLUX -> DALL-E 3**, i.e. rectified-flow to an unknown commercial model.
That is wider than a same-family gap and is the project's main generalisation
risk.

- All PNG, RGB, 512×512 (56 slightly smaller, source-limited)
- 0 corrupt, 0 cross-class filename collisions
- Centre-cropped, never resampled, to preserve generator artifacts
- SID_Set label 2 (locally tampered) is **excluded** — that is a localization
  task, not the binary problem

Two loader decisions are load-bearing and must not be "cleaned up":

- **PNG everywhere.** Mixed formats let the model learn JPEG artifacts as a class
  signal; it scores ~99% locally and collapses on the real test set.
- **Centre-crop, never resize, at download time.** Resampling is a low-pass
  filter that smears exactly the high-frequency fingerprints the task depends on.

### 2.2 Held-out evaluation — ALREADY AVAILABLE, verified

The organisers' reference benchmark is published as parquet and loads in one line.
**Verified live**: repo is public, row counts match the problem statement exactly
(13,841 = 4,998 COCO val2017 + 8,843 DALL-E 3).

```python
from datasets import load_dataset
ds = load_dataset("techjam-aigc/wildfake-eval-subset", "laion_matched", split="validation")
```

| config | rows | contents | resolution | size |
|---|---|---|---|---|
| `default` | 13,841 | 4,998 COCO val2017 + 8,843 DALL-E 3 | native | 2.93 GB |
| `normalized` | 13,841 | same, centre-cropped | 200x200 | 229 MB |
| `laion_matched` | 7,652 | 3,826 LAION + 3,826 DALL-E 3 | 512x512 | 596 MB |
| `cross_generator` | 5,494 | LAION vs DALL-E3/MJv5/SDXL/GigaGAN | 256x256 | 129 MB |

**DO NOT TRAIN ON ANY CONFIG.** The README states the final test set is drawn
from the same corpus; training on it leaks.

#### The `default` config is trivially gameable -- we verified this ourselves

Streaming 600 rows and inspecting image sizes:

- **Every** label-0 (real) image is exactly **200x200**
- **No** label-1 (fake) image is -- they are 1024x1024, 1792x1024, etc.
- `lambda img: 0 if img.size == (200,200) else 1` scores **AUROC 1.000, no model**
- Class balance is **36% real / 64% fake**, so always-"fake" scores 64% accuracy

Consequences, and these are not optional:

1. **Report `laion_matched` as the headline number.** It is resolution-matched
   and genuinely hard. If we score ~1.00 on `default` and ~0.75 on
   `laion_matched`, we learned resolution and nothing else.
2. **Never report raw accuracy on `default`** -- use AUROC or balanced accuracy.
3. Report `cross_generator` too: it is the only config testing generalisation
   across four distinct generators, which is the actual research question.
4. Worth raising with the organisers whether the final test set shares this
   defect. Noticing it is itself a strong Problem Insight signal (20% of score).

Documented single-feature leakage in the other configs (AUROC, no learning):

| feature | default | normalized | laion_matched | cross_generator |
|---|---|---|---|---|
| image size | **1.000** | 0.500 | 0.500 | 0.500 |
| mean luminance | - | 0.529 | 0.734 | 0.701 |
| recompressed bytes | - | 0.602 | 0.696 | 0.565 |

Note luminance still leaks at 0.734 on `laion_matched` -- our colour-jitter
augmentation is what addresses that, and we should verify it does.

### 2.3 Splits

| Split | Source | Size | Use |
|---|---|---|---|
| train | SID_Set | 80% (9,600) | gradient updates |
| val | SID_Set | 20% (2,400) | early stopping, threshold calibration |
| test-ID | SID_Set val, transformed | 2,400 × 16 | robustness grid, in-distribution |
| test-OOD | COCO + DALL·E | acquired | generalization, headline number |

Split by **file hash, not random shuffle**, so it is reproducible across machines
and reruns without a shared seed file.

---

## 3. Model

Backbone: **`openai/clip-vit-large-patch14`, frozen, 427.6M params** (verified
against HuggingFace safetensors metadata; comfortably under the 2B limit).
Features from the **penultimate layer, 768-d**.

### 3.1 Why frozen CLIP rather than a fine-tuned CNN

A CNN trained from scratch becomes *asymmetrically tuned*: it learns "what makes
this fake," so the **real class becomes a sink** that absorbs anything not
matching the training generator's artifacts. Fakes from an unseen generator land
in the sink. A frozen feature space never learned that asymmetry, so real and
fake stay linearly separable across generators.

Reported (Ojha et al., CVPR 2023): unseen diffusion/autoregressive generators
**~82% acc / 95.00 mAP vs 53-58% / 75.51** for trained CNNs.

### 3.2 The critical correction — plain CLIP probing FAILS on our eval condition

Our train->test gap is **FLUX -> DALL-E 3**, and on post-processed commercial
images Ojha's method scores **34.4 AUC / 45.6 acc on DALL-E 3 -- worse than
chance.** Do not ship plain Ojha.

The fix is Cozzolino et al. (CVPRW 2024), which uses the same backbone with a
better recipe and is evaluated on exactly this condition:

| Method | Avg AUC (post-processed, 18 generators) | DALL-E 3 (AUC / Acc) |
|---|---|---|
| Ojha et al. | 71.8 | 34.4 / 45.6 |
| Corvi et al. (low-level) | 70.8 | - |
| LGrad / DIRE / NPR | 49-51 (chance) | - |
| Cozzolino 1k, no aug | 77.5 | 69 / 51 |
| **Cozzolino 1k + aug** | **83.2** | **82.1 / 73.3** |
| Cozzolino 10k + aug | 85.2 | - |

Two findings to internalise: augmentation is worth **~13 AUC and ~22 acc** on
DALL-E 3, and **1k+aug beats 10k+no-aug** -- more data does not substitute for
augmentation. We already have far more images than we need.

### 3.3 Head

**Linear SVM** (`sklearn.svm.LinearSVC`), which Cozzolino ablated against
logistic regression, Mahalanobis, GNB and soft-NN and found best. Fit a torch
linear layer alongside for comparison; both take seconds on cached features.

### 3.4 Augmentation — the highest-leverage component

Applied **identically to both classes**, at **pixel level before the CLIP
preprocessor**:

| Augmentation | Range | Evidence |
|---|---|---|
| JPEG re-encode | QF [30, 95] | Mandatory -- also fixes Shortcut B (SS1.2) |
| Random resize | down to 0.25x | Judged transform |
| Random crop | 5/8 - full | Judged transform |
| Colour jitter | +/-20% | Judged transform |
| Gaussian noise | sigma [0, 0.10] | Judged transform |
| Gaussian blur | sigma [0, 2.0] | **Apply cautiously -- verify on val** |

Two evidence-based details:

- **Compose blur and JPEG together**, not only independently. The [RE]
  reproducibility study found the compositional variant scored **93.7 vs 66.3 AP**
  on the hardest set. Cheap to implement, large effect.
- **Blur alone was net-harmful** in that same ablation (89.7 -> 83.8 mAP), and
  augmentation is dataset-specific. JPEG/resize/crop aggressively; add blur only
  if it helps on our own val split. Do not assume.

On the clean-accuracy tradeoff: in-distribution AP held at **100.0** across
No-Aug / Blur / JPEG / Blur+JPEG while mAP moved 89.7 -> 93.8. **JPEG
augmentation is close to free.**

### 3.5 Preprocessing decision

Ojha crops; Cozzolino resizes. We are graded on robustness to *both* crop and
resize, so **follow Cozzolino (resize)** and let augmentation cover geometry,
rather than betting that a fixed 224 centre crop survives an adversarial crop.

### 3.6 What we are explicitly NOT doing

- **No from-scratch frequency/DCT branch.** Every low-level method sits at chance
  under crop+resize+JPEG (LGrad 49.4, DIRE 49.9, NPR 51.0). The artifacts are a
  GAN/UNet-decoder phenomenon that FLUX and DALL-E 3 do not share, they are
  forgeable in both directions, and JPEG's own 8x8 DCT overwrites the exact band
  they read. Legitimate only as late fusion via a *pretrained* checkpoint (+3.6
  AUC), and only if the core is already working.
- **No ensemble** until a single model runs end-to-end. Ensembles are where
  hackathon projects die at 4am.
- **SID_Set label 2 (tampered) excluded.** Those are mostly-real pixels; mapping
  them to "fake" drags the decision boundary. Optional separate experiment only.

### 3.7 Optional, if time permits

`laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (986.1M) or
`google/siglip-so400m-patch14-384` (878.0M) -- both under budget. Larger CLIP
pretraining corpus is worth ~10 points. **DINOv2 is NOT recommended as primary**:
evidence is mixed, and its self-supervised objective explicitly optimises
invariance to local perturbations, which suppresses the sensitivity we need.

## 4. Deliverable: `predict.py`

The scored artifact. Requirements beyond the obvious:

```bash
python predict.py --input-dir path/to/images --output preds.json
```

- Recurses the directory; accepts jpg/jpeg/png/webp/bmp, case-insensitive
- **Never crashes on a bad image.** Corrupt/unreadable → `pred: 0.5` plus a
  stderr warning. A crash on image 4,000 of 5,000 loses everything.
- Batched inference with a progress line; `--batch-size` and `--device` flags
- `image_path` in the output is **relative to `--input-dir`**, stable across machines
- Deterministic: same input → byte-identical JSON
- Runs on CPU if no GPU is present (slower, but it runs)

Edge cases that must be handled because they occur in real image dirs: EXIF
rotation, greyscale and CMYK images, alpha channels, animated GIF/WebP (first
frame), images under 32px, truncated files, symlinks, and non-image files with
image extensions.

---

## 5. Evaluation

### 5.1 Robustness grid — the headline table

For every cell of the 16-transform grid (`src/transforms.py`, frozen), report on
both test-ID and test-OOD:

**AUROC, accuracy @ calibrated threshold, TPR @ 1% FPR, TPR @ 5% FPR.**

AUROC is threshold-free and the honest primary metric. Accuracy is included
because it is what non-specialists read. **TPR at low fixed FPR is the one that
matters operationally**: a platform moderating at scale cannot tolerate flagging
1-in-20 real photos, so "accuracy" alone hides the deployment-relevant failure.

The threshold is calibrated **once on clean val** and then held fixed across all
transforms. Re-tuning per transform would be cheating, and would hide precisely
the threshold-drift failure described in §1.

### 5.2 Quantified claims — targets calibrated against published numbers

Each is a specific measurement with a pass condition stated in advance. Targets
are anchored to Cozzolino et al. (CVPRW 2024) on the same eval condition, so
they are neither sandbagged nor fantasy.

| Claim | Metric | Target | Reference |
|---|---|---|---|
| Clean in-distribution | AUROC, SID_Set val, clean | > 0.95 | — |
| Robustness floor | worst cell of 16, SID_Set val | > 0.85 | — |
| Robustness spread | mean AUROC drop clean->transformed | < 0.10 | — |
| **Cross-generator (headline)** | **AUROC, `laion_matched`** | **> 0.80** | Cozzolino 1k+aug: 82.1 on DALL-E 3 |
| Multi-generator | AUROC, `cross_generator` | > 0.75 | — |
| Operational precision | TPR @ 1% FPR, `laion_matched` | reported | — |
| **Augmentation ablation** | delta worst-cell AUROC, aug vs no-aug | **> +0.05** | Cozzolino: +13 AUC on DALL-E 3 |
| Shortcut neutralisation | blockiness-only AUROC after aug | ~0.50 | **measured: 0.830 -> 0.498** |
| Latency | ms/image, batch 32, T4 | reported | — |
| Throughput | images/sec, single T4 | reported | — |

Sanity anchors from the literature — if we land outside these, something is wrong:

- Plain Ojha-style CLIP probing scores **34.4 AUC on DALL-E 3** (worse than
  chance). If our no-augmentation ablation lands near chance on `laion_matched`,
  that is the expected, publishable result — not a bug.
- Every low-level/frequency method sits at 49-51 AUC under crop+resize+JPEG.
- Consumer detectors report 5-15% FPR on real photos; Bellingcat found one
  flagged 6/20 genuine photojournalism images. Our FPR should be stated plainly
  against that backdrop.

**The augmentation ablation is the most important experiment in the project.** It
is the quantified evidence that robustness is engineered rather than incidental.
Train two identical heads, one with `train_augment` and one without, report both
rows. We already have the pre-registered prediction that it helps, and the
shortcut measurement (0.830 -> 0.498) explaining *why*.

### 5.3 Error analysis

Sample the highest-confidence false positives and false negatives, per transform
cell. Specific hypotheses to test, each a plausible real-world failure:

- FP on heavily denoised smartphone photos (computational photography smooths in
  a way that mimics generative smoothness — the exact §1 confound)
- FP on legitimately upscaled or heavily compressed real images
- FN on synthetic images that have been re-photographed or hard-compressed
- FN on generators unlike SID_Set's training distribution

---

## 6. Real-time / production framing

The problem statement scopes out full deployment, but Feasibility is 15% and past
winners emphasised deployability. We address it with measurements, not claims:

- **Measured throughput and p50/p95 latency** on a single T4, batch 1 and 32
- Embedding cache: CLIP features computed once, reused — the design that makes
  the 16-cell grid affordable is the same one that makes serving cheap
- A stated deployment sketch: which component runs per-upload, what is batched,
  where the threshold is configured per-surface (a strict threshold for
  recommendation ranking, a lenient one for user-facing warnings)
- Honest statement of what we did *not* build (no video, no serving
  infrastructure, no A/B framework)

---

## 7. Repository layout

```
fetch_data.py        SID_Set downloader (done, resumable)
predict.py           SCORED DELIVERABLE: image dir -> JSON
run.sh               entrypoint
src/
  transforms.py      frozen 16-cell eval grid + train-time augmentation (done)
  data.py            dataset, hash-based splits, loaders
  features.py        CLIP embedding extraction + on-disk cache
  train.py           linear probe + fine-tune paths
  evaluate.py        robustness grid -> results table
  error_analysis.py  FP/FN sampling with rendered contact sheets
  app.py             demo UI for the video
docs/
  SPEC.md            this file
  RESULTS.md         generated benchmark tables
```

---

## 8. Compute — Kaggle (verified specs)

No local GPU; training happens on Kaggle. Key constraints:

| Resource | Limit |
|---|---|
| Session runtime | ~12 h |
| Weekly GPU quota | ~30 h, shared across accelerator types |
| `/kaggle/working` | ~20 GB, persists between runs of the same notebook |
| `/kaggle/temp` | ephemeral scratch, does **not** count against the 20 GB |

**Use T4 x2, not P100** — real fp16 tensor cores (~65 TFLOPS) and 32 GB
aggregate. Critical gotcha: **T4 does not support CUDA bfloat16.** Use
`torch.float16` with `torch.amp.autocast` + `GradScaler`, never bf16.

Set `HF_HOME=/kaggle/temp/hf` before any `load_dataset` call, or a multi-GB
cache eats the 20 GB output quota.

Because the backbone is frozen, we extract features **once** and fit the head in
seconds. We will not approach either the 12 h session limit or the 30 h weekly
quota, and 16 GB VRAM is never a constraint.

Checkpoint persistence: write to `/kaggle/working/`, "Save & Run All (Commit)",
then mount into the next notebook via Add Data -> Kernel Output Files. For
anything durable, push to HF Hub — it survives quota exhaustion and teammates
can pull it.

## 9. Sequencing and ownership

Dependency-ordered. Critical path: features -> train -> evaluate.

| # | Task | Depends on | Owner |
|---|---|---|---|
| 1 | `data.py` — hash splits, edge-case-tolerant loader | — | |
| 2 | `features.py` — CLIP ViT-L/14 embeddings + on-disk cache | 1 | |
| 3 | `train.py` — LinearSVC head, with/without aug | 2 | |
| 4 | `evaluate.py` — 16-cell grid x 4 metrics, all 3 eval configs | 3 | |
| 5 | `predict.py` — **the scored deliverable** | 3 | |
| 6 | Eval set wiring — `techjam-aigc/wildfake-eval-subset` | — | |
| 7 | Augmentation ablation (the headline experiment) | 3 | |
| 8 | `error_analysis.py` + contact sheets | 4 | |
| 9 | `app.py` demo + video | 5 | |
| 10 | README, Devpost writeup | 4, 7, 8 | |

Tasks 1 and 6 start immediately in parallel. Task 5 depends only on a trained
head, so write it against a stub and wire it up later — **do not leave the
scored deliverable until last.**

Already done: `fetch_data.py`, `src/transforms.py` (16-cell frozen grid +
`train_augment` + `normalize_compression`), this spec.

## 10. Definition of done

- [ ] `predict.py` runs on an arbitrary image directory and emits valid JSON
- [ ] Robustness table populated for all 16 cells, both test-ID and test-OOD
- [ ] Augmentation ablation reported with a signed delta
- [ ] Parameter count asserted < 2B
- [ ] Error analysis with real sampled images, not prose
- [ ] README with setup, reproduction steps, limitations, contributions
- [ ] Demo video public on YouTube, linked from Devpost
- [ ] Devpost submission finalised
