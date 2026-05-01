---
date: 2026-05-01
status: draft
description: Final-stretch experiment runs and analysis pipeline for the DS6050 G8 final report
---

# Final Runs — Plan

## Goal

Produce all results that teammates need to write the final report and presentation. Per M2 instructor feedback, three things are missing:

1. **Data-scaling experiment** (RQ1, RQ3): full 4×2×5×3 = 120-run grid → centerpiece mAP-vs-fraction figure.
2. **Augmentation ablation** (RQ2): pretrained ResNet-50 + pretrained ViT-B/16 × {none, standard, strong} × 3 seeds = 18 runs.
3. **Per-class AP analysis**: aggregate from existing `eval_results.json` files — no new runs.

Code already supports everything (timm/torchvision models, three augmentation modes in `transforms.py`, per-class AP in eval output, sweep launcher + SLURM array). The remaining work is a small launcher extension, executing the runs on Rivanna, and reworking the figure script.

## Current state

- `configs/sweep.yaml` defines the 120-run main sweep with `augmentation: standard` (single string).
- `scripts/launch_sweep.py` reads sweep yaml → writes per-run config files + `sweep_configs.txt` manifest.
- `slurm/sweep_array.sh` reads `MANIFEST=configs/generated/sweep_configs.txt` and runs one job per line.
- `src/voc_bench/config.py:72` builds output dir as `{model}_{pretrain}_frac{X.XX}_aug-{mode}_seed{N}` — augmentation is already in the path, so different aug runs at same (model, frac, seed) won't collide.
- `scripts/make_figures.py` is hardcoded to the 4 M2 baselines on full data — needs extension for the centerpiece figure and augmentation chart.
- 4 single-seed M2 baseline runs are in `results/` (ResNet-50 / ViT-B/16, pretrained / scratch, frac=1.00). These can stay; the new sweep will overwrite seed=42 entries cleanly because dir names are deterministic.

## Design decisions

### 1. Two separate sweep configs, one shared launcher

Keep `sweep.yaml` unchanged for the data-scaling sweep. Add `sweep_aug.yaml` for the ablation. Extend `launch_sweep.py` so the `augmentation` field accepts **either a string or a list**. This avoids exploding the main 120-run sweep into 360 runs (which would happen if we iterated all augmentations × all fractions × all models).

**Why:** the augmentation ablation only varies along one extra axis but restricts the model/fraction axes, so a separate config file is the natural shape. The launcher change is one for-loop.

### 2. Submit the two arrays as parallel sbatch jobs

Submit `sweep_array.sh` (120 jobs) and a sibling `sweep_aug_array.sh` (18 jobs) at the same time. They share GPU partition slots but don't depend on each other, so total wall clock = max(longer queue) instead of sum.

**Why:** Rivanna queue wait is the dominant cost, not compute (~12 GPU-hr serial / 20 concurrent ≈ 35 min compute). Two arrays double our queue presence.

### 3. Output manifest and array script per sweep

Use `--output-dir configs/generated_aug` for the ablation so its manifest doesn't overwrite the main one. New SLURM script reads from that manifest. No changes to the main array script.

### 4. Figure script: keep existing functions, add new ones

`make_figures.py` already has training curves + mAP comparison + per-class AP for the M2 baselines. Add three new functions:
- `fig_data_scaling()` — centerpiece: mAP (mean ± std across seeds) vs fraction, 8 lines (4 archs × 2 pretrain), log-x or linear-x.
- `fig_augmentation_ablation()` — grouped bar chart, x=architecture, hue=augmentation mode.
- `fig_per_class_ap_full()` — extend existing heatmap to all 8 architecture×pretrain cells at frac=1.00.

Don't refactor the M2 figures; they're cited in `m2.tex` already.

## Steps

### Step 1 — Commit working-tree state (clean baseline)

```bash
git add reports/m1.tex reports/m2.tex reports/G08_M2.pdf feedback/G08-MII.pdf
git rm reports/1.tex
git commit -m "M2 deliverable + instructor feedback"
```

### Step 2 — Generate and submit the data-scaling sweep (120 runs)

```bash
python scripts/launch_sweep.py
# → writes 120 configs to configs/generated/, manifest at configs/generated/sweep_configs.txt
```

On Rivanna:
```bash
sbatch --array=0-119%20 slurm/sweep_array.sh
```

### Step 3 — Add augmentation ablation infrastructure

**3a.** Extend `scripts/launch_sweep.py` to accept `augmentation` as either str or list:

```python
augmentations = sweep["augmentation"]
if isinstance(augmentations, str):
    augmentations = [augmentations]
# add a 4th nested loop over augmentations
```

**3b.** Add `configs/sweep_aug.yaml`:
```yaml
models:
  - resnet50_pretrained
  - vit_b16_pretrained
fractions:
  - 1.00
seeds:
  - 42
  - 123
  - 456
augmentation:
  - none
  - standard
  - strong
```

**3c.** Add `slurm/sweep_aug_array.sh` (copy of `sweep_array.sh` with `MANIFEST=configs/generated_aug/sweep_configs.txt`).

**3d.** Generate and submit:
```bash
python scripts/launch_sweep.py --sweep-config configs/sweep_aug.yaml --output-dir configs/generated_aug
sbatch --array=0-17%20 slurm/sweep_aug_array.sh
```

### Step 4 — Monitor and wait

Periodic `squeue -u $USER` until both arrays drain. SLURM walltime is 4hr/job; no individual run will hit it. Expect 1.5–6 hr wall clock depending on queue.

### Step 5 — Pull results back to local

```bash
rsync -av rivanna:image-cnn-transformers/results/ ./results/
```

### Step 6 — Aggregate

```bash
python scripts/aggregate_results.py --results-dir results
# → results/summary.csv with mean ± std across 3 seeds, grouped by (model, pretrained, fraction, augmentation)
# → results/summary_full.csv with one row per run
```

### Step 7 — Extend `make_figures.py`

Add the three new functions described above. Keep the M2 figures untouched. Verify each renders without error against the new aggregated data.

```bash
python scripts/make_figures.py
# → reports/figures/*.png/.pdf
```

### Step 8 — Hand-off to teammates

Push the new figures + summary CSVs and post in the team channel. Provide:
- `results/summary.csv` for the headline numbers table
- `reports/figures/data_scaling.{png,pdf}` for the centerpiece
- `reports/figures/augmentation_ablation.{png,pdf}` for RQ2
- `reports/figures/per_class_ap_full.{png,pdf}` for the per-class analysis
- A short summary message: which architecture wins where, where ViT crosses CNN, what the augmentation gap looks like

## Risks and unknowns

- **Rivanna queue contention** at end-of-semester is the wildcard. If the main array hasn't started moving after ~1 hr, consider lowering `%20` → `%10` so jobs start individually rather than waiting for a fat allocation.
- **Scratch ViT-B/16 at low fractions** may behave numerically badly (loss explodes, NaN). The training loop catches early stop but doesn't catch NaN — check first 10 runs land sane mAP in `metrics.jsonl` before walking away.
- **DataLoader workers + Rivanna**: 4 workers is fine for ImageNet-sized images, but if any job mysteriously hangs, suspect dataloader → drop to `num_workers=2`.
- **Storage**: 138 result dirs × ~10MB checkpoint each = ~1.4GB. Well under quota. No checkpoint pruning needed.

## Out of scope (deliberately)

- No model code changes (don't switch from timm/torchvision to HuggingFace — see prior conversation).
- No retraining of M2 baselines for "consistency" — the new sweep includes seed=42 runs that will overwrite them with identical configs anyway.
- No new architectures or augmentation strategies beyond what feedback called for.
- No hyperparameter tuning; learning rates already chosen per model in their YAML configs.
