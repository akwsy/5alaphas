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

## 1. The core insight (measured, not assumed)

We measured high-frequency energy (mean Laplacian response) over 150 images per
class from our SID_Set subset:

| Class | High-freq energy | Contrast | Brightness |
|---|---|---|---|
| Real | 13.69 ± 10.42 | 63.53 | 107.65 |
| Fake | 8.03 ± 4.46 | 63.01 | 95.01 |

**Synthetic images are measurably smoother — 41% less high-frequency energy.**
This is a shortcut a model will learn within one epoch, and it is *fragile*:

| Transform | Real HF | Fake HF | Absolute gap | Cohen's d |
|---|---|---|---|---|
| clean | 14.35 | 7.88 | 6.47 | 0.77 |
| jpeg q90 | 14.64 | 8.26 | 6.38 | 0.72 |
| jpeg q30 | 11.98 | 7.20 | 4.78 | 0.70 |
| blur σ1.0 | 4.51 | 3.00 | 1.51 | 0.85 |
| blur σ2.0 | 1.65 | 1.27 | 0.38 | 0.75 |
| resize 0.25× | 2.15 | 1.63 | 0.52 | 0.71 |

Read this carefully, because the naive reading is wrong. The *absolute* gap
collapses under blur (6.47 → 0.38), but Cohen's d stays ~0.7–0.85 throughout.
Two consequences:

1. **A hand-crafted frequency feature is not a detector.** d ≈ 0.8 corresponds to
   roughly 65% accuracy — far below competitive. Anyone proposing "just use the
   FFT" should be shown this table.
2. **A CNN trained on clean data will anchor on this axis anyway**, because on
   clean data it is the cheapest available signal. When the input is blurred, the
   feature's dynamic range compresses to near zero and the decision threshold —
   calibrated on clean data — lands in the wrong place. That is the failure mode
   the judges are testing for.

**Therefore the architecture is chosen to avoid learning this shortcut in the
first place**, not to exploit it. See §3.

---

## 2. Data

### 2.1 Training (in hand)

`data/raw/` — 12,000 images, 6,000 real + 6,000 synthetic, from SID_Set.

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

### 2.2 Held-out evaluation (must acquire)

The organisers evaluate on **COCO val2017 reals + DALL·E fakes** — a different
real-image distribution *and* a different generator from SID_Set. This is the
hard part of the problem, and clean SID_Set accuracy will overstate our
performance on it.

We build our own proxy for this and report it honestly:

- `data/eval_ood/real/` — COCO val2017 sample
- `data/eval_ood/fake/` — a DALL·E-family set we did not train on

**Never train, tune, or early-stop on this.** It is touched once per checkpoint.

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

### 3.1 Primary: frozen CLIP + linear probe

```
image → CLIP ViT-L/14 (frozen, no grad) → 768-d embedding → LogisticRegression → P(fake)
```

Rationale:

- A frozen, general-purpose feature space cannot overfit to SID_Set's specific
  generator, because it was never trained on it. Only the linear head is fitted,
  and a linear head on frozen features has very limited capacity to memorise the
  smoothness shortcut.
- Trains in **minutes on CPU** once embeddings are cached. Fits the deadline.
- Embeddings are computed once and reused across every experiment, including all
  16 transform cells — which makes the full robustness grid cheap.

Parameter count must be asserted < 2B in code, not assumed.

### 3.2 Secondary: fine-tuned backbone with robustness augmentation

A small ConvNeXt/ViT fine-tuned end-to-end **with the transform suite applied as
training augmentation**. This directly targets the §1 failure mode: if the model
sees blurred and JPEG-crushed images during training, it cannot rely on a feature
that those operations destroy.

Ship whichever wins on **test-OOD**, not on clean accuracy. Report both.

### 3.3 What we are explicitly not doing

- No hand-crafted FFT/DCT feature as the primary signal — §1 shows it is weak and
  fragile. It may appear in error analysis as a baseline to beat.
- No ensemble unless a single model is already working end-to-end and there is
  time left. Ensembles are where hackathon projects go to die at 4am.

---

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

### 5.2 Quantified claims we intend to be able to make

These are the numbers that replace fluff. Each is a specific measurement, and
each has a pass condition we state in advance:

| Claim | Metric | Target |
|---|---|---|
| Clean in-distribution performance | AUROC, test-ID clean | > 0.95 |
| Robustness floor | worst-cell AUROC over all 16, test-ID | > 0.85 |
| Robustness spread | mean AUROC drop, clean → transformed | < 0.10 |
| Cross-generator generalization | AUROC, test-OOD clean | > 0.80 |
| Operational precision | TPR @ 1% FPR, test-OOD | reported, no target |
| Augmentation ablation | Δ worst-cell AUROC, with vs without aug | > +0.05 |
| Latency | ms/image, batch 32, T4 | reported |
| Throughput | images/sec, single T4 | reported |

The **augmentation ablation is the most important experiment in the project**: it
is the direct, quantified evidence that our robustness is engineered rather than
incidental. Train two identical models, one with the transform augmentation and
one without, and report both rows. If the delta is near zero, we learned
something and we say so.

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

## 8. Sequencing and ownership

Dependency-ordered. The critical path is features → train → evaluate.

| # | Task | Depends on | Owner |
|---|---|---|---|
| 1 | `data.py` — hash splits, PIL loader, edge-case handling | — | |
| 2 | `features.py` — CLIP embeddings + cache | 1 | |
| 3 | `train.py` — linear probe | 2 | |
| 4 | `evaluate.py` — 16-cell grid, 4 metrics | 3 | |
| 5 | `predict.py` — the scored deliverable | 3 | |
| 6 | OOD set — COCO + DALL·E acquisition | — | |
| 7 | Fine-tune + augmentation ablation | 3 | |
| 8 | `error_analysis.py` + contact sheets | 4 | |
| 9 | `app.py` demo + video | 5 | |
| 10 | README, Devpost writeup | 4, 8 | |

Tasks 1 and 6 are independent and start immediately in parallel. Task 5 depends
only on a trained head, so it can be written against a stub and wired up later —
**do not leave the scored deliverable until last.**

## 9. Definition of done

- [ ] `predict.py` runs on an arbitrary image directory and emits valid JSON
- [ ] Robustness table populated for all 16 cells, both test-ID and test-OOD
- [ ] Augmentation ablation reported with a signed delta
- [ ] Parameter count asserted < 2B
- [ ] Error analysis with real sampled images, not prose
- [ ] README with setup, reproduction steps, limitations, contributions
- [ ] Demo video public on YouTube, linked from Devpost
- [ ] Devpost submission finalised
