# NRC-Cal — "Confidence Has a Shape. It's Already There."

Reproducible research repository for **NRC-Cal**: predicting probabilistic
calibratability of deep regression models via Neural Regression Collapse (NRC)
geometry, entirely post-hoc and training-free.

This repo builds on and reuses artifacts from:

- Dheur & Ben Taieb, *Probabilistic Calibration by Design for Neural Network
  Regression* (AISTATS 2024) — code: https://github.com/Vekteur/quantile-recalibration-training
- Dheur & Ben Taieb, *A Large-Scale Study of Probabilistic Calibration in
  Neural Network Regression* (ICML 2023) — code: https://github.com/Vekteur/probabilistic-calibration-study

## Three findings from actually running code against the real repos (not assumed from the papers alone)

1. **No pretrained checkpoints are shipped by either repo.** Their main
   experiment command even runs with `remove_checkpoints=True`. `03` trains
   small pilot models itself — it does not download weights.
2. **`demo/datamodule.py`'s train/val/calib/test split (50/10/15/25) is NOT
   the split used for the paper's actual reported numbers.** The real split,
   confirmed in `uq/configs/dataset_groups.py`, is `[0.65, 0.0, 0.1, 0.15, 0.1]`
   (65/10/15/10, matching the paper text). `configs/config.yaml` and every
   notebook here use the verified real split, not the demo's.
3. **`openml==0.13.1`, pinned exactly by both upstream repos, is broken by
   NumPy 2.0** (uses the since-removed `np.sctypes`) — confirmed by actually
   running the download path in the sandbox this project was built in.
   `requirements.txt` here installs a modern `openml` instead.

## Verified model architecture (03/04) — from source, not the paper text alone

Read directly from `uq/models/general/mlp.py`, `uq/models/general/nn_module.py`,
`uq/models/pred_type/mixture_dist.py`, `uq/models/base_module.py`:

- `MLP`: 3 hidden layers × 128 units (ReLU), dropout 0.2, matching the paper's
  "3-layer MLP with 128 hidden units" exactly.
- `MixturePrediction(event_size=1, mixture_size=K)`: outputs `(means, rhos,
  mix_logits)`, each width `K`; `stds = softplus(rhos) + 1e-3`.
- Distribution: `NormalMixtureDist` — a thin subclass of
  `torch.distributions.MixtureSameFamily`, standard `log_prob`.
- Optimizer: `AdamW(lr=1e-3)`. Batch size 512 (capped to split size on the
  smallest datasets). Early stopping: patience=30 on validation NLL,
  restoring best-epoch weights (matching the paper's own protocol).

**The architecture classes above are imported directly from the cloned repo,
unchanged.** The training *loop* around them (`src/models/pilot_mixture_model.py`)
is this project's own short, auditable code, not upstream's Hydra-driven
grid-search runner — see that file's module docstring for exactly why
(short version: upstream's runner sweeps every method in the paper's full
comparison table via a custom grid DSL; isolating just "BASE" from that grid
without extensive further reverse-engineering was judged riskier than a
~40-line from-scratch loop around the *real* model class).

**Pilot run uses `mixture_size=1`** (single Gaussian) — the branch whose
assumptions match Neural Regression Collapse theory (see go/no-go plan).
Switching to `mixture_size=3` (the paper's main default) later is a
one-line change in `03`.

All of 01-04 were executed end-to-end for real in this project's own build
(real cloning, real training with real gradient descent using the real
upstream architecture, real checkpoint save/load round-trips) — the one
exception is `02`'s actual UCI download call, which needs a network reach
this project was authored without; that specific call was still verified
to reach the correct real network boundary before failing on that
restriction (see `02`'s notebook markdown for detail). 53/53 unit tests
pass (`pytest tests/`).

## Target compute: Colab, T4 GPU, reading files that live on your Mac

- **Compute** runs on Google's cloud T4 — see `configs/config.yaml` ->
  `compute.target_gpu`. Notebook 00 verifies the attached GPU and warns
  (does not hard-fail) if it isn't a T4. Mixed precision on T4 must use
  `float16` (Turing has no native bf16 tensor cores).
- **Your files** stay on your Mac and are still visible to Colab via
  **Google Drive for Desktop** (Mirror or Stream mode) syncing this
  `project/` folder to `My Drive`; Colab mounts Drive (Option A, notebook
  00) and sees the identical files. This is a networking fact, not a
  preference: Colab's cloud runtime cannot reach into a MacBook's
  filesystem any other way without a reachable tunnel (see Option C below).

## Build plan (one notebook at a time, gated by approval)

| # | Notebook | Status |
|---|---|---|
| 00 | `00_environment.ipynb` | ✅ delivered, tested (T4-aware, local-file-access via Drive) |
| 01 | `01_download_repositories.ipynb` | ✅ delivered, tested (clones + verifies both real repos, confirms no checkpoints) |
| 02 | `02_prepare_datasets.ipynb` | ✅ delivered, tested (real download wrapper + hand-verified split arithmetic, 16 unit tests) |
| 03 | `03_train_pilot_models.ipynb` | ✅ delivered, tested — **real architecture, real gradient descent** (see below) |
| 04 | `04_extract_features.ipynb` | ✅ delivered, tested end-to-end (real checkpoint round-trip, real feature extraction) |
| 05 | `05_compute_NRC.ipynb` | ✅ delivered, tested — **formula gate cleared** (see below) |
| 06 | `06_correlation_analysis.ipynb` | ✅ delivered, tested — **the actual go/no-go notebook** |
| 07 | `07_visualization.ipynb` | ✅ delivered, tested |
| 08 | `08_closed_form_calibration.ipynb` | ⏳ **paused here on purpose** — see "What to do next" below |
| 09 | `09_full_experiment.ipynb` | ⏳ full 57-dataset scale-up |
| 10 | `10_ablation.ipynb` | ⏳ |
| 11 | `11_multivariate.ipynb` | ⏳ needs a genuinely multivariate-target benchmark — NRC3 is inapplicable to every `uci`-group dataset (all univariate, see below) |
| 12 | `12_reproduce_tables.ipynb` | ⏳ |
| 13 | `13_export_results.ipynb` | ⏳ |

## The formula gate (05) — cleared, with a citation trail

Every formula in `src/geometry/nrc.py` is transcribed from Andriopoulos,
Dong, Guo, Zhao & Ross, *The Prevalence of Neural Collapse in Neural
Multivariate Regression* (NeurIPS 2024, arXiv:2409.04180) — fetched and
read in full (the NeurIPS proceedings PDF) before any code was written.

**A finding from reading the paper that changes this pilot's scope:** the
paper states directly (Appendix A.3) that **NRC3 is trivially zero and not
meaningful for univariate regression (n=1)** — every dataset in the `uci`
pilot group has a scalar target, so NRC3 is not computed here at all
(`compute_nrc3` returns `None` for n=1 by design). A genuinely multivariate
dataset is needed to ever compute NRC3, which is why `11_multivariate` is
its own separate, not-yet-started notebook rather than folded into this pilot.

**Verification, not just citation:** `tests/test_nrc.py` includes a test
that constructs features/weights/targets using the paper's own Theorem 4.1
/ Corollary 4.2 closed-form global-minimum solution (worked by hand for the
n=2, uncorrelated-target case, in the test's own docstring) and checks that
NRC1, NRC2, and NRC3 as implemented here all independently evaluate to ~0
on it. 21/21 geometry tests pass, including this one.

**Also flagged (05's own markdown, worth reading before trusting a NO-GO):**
a synthetic-fixture test run showed only moderate NRC1/NRC2 collapse, which
may mean early stopping (correct for calibration quality) doesn't run long
enough to reach the "terminal phase" NRC theory's own experiments use (up
to 1.5M epochs in the source paper). Re-run `03` without early stopping on
a couple of datasets before concluding a weak `06` correlation kills the
whole direction.

## The actual go/no-go (06) — what it computes, and what to do with the result

`06` correlates NRC-distance (`05`) against two PCE numbers (`src/metrics/pce.py`,
formulas from the QRT paper, Section 2 — already fully read earlier in this
project, no separate gate needed): PCE(BASE) and PCE(BASE + Quantile
Recalibration). The second is the more valuable question — does NRC predict
what standard post-hoc recalibration *doesn't already fix*?

**This synthetic test run's numbers are not a research result** — they
exist only to prove the pipeline computes something sane end-to-end (which
it does: 86/86 tests pass across the whole project, and QR correctly
reduced PCE on 11/12 synthetic pilot datasets in `07`'s bar chart). The
real go/no-go verdict requires running `02` against live UCI data, which
needs network access this project was built without (see below).

## What to do next

1. Push this repo to your own GitHub, or place it under Drive via Drive for
   Desktop (see "Target compute" below), and open it in Colab.
2. Run `00` through `07` in order, for real, against live data.
3. Look at `06`'s printed correlations and `07`'s scatter plot (`figures/nrc_vs_pce_scatter_mixture_1.png`).
   Apply the decision rule `06` prints.
4. Bring the real numbers back — `08`'s design (which NRC component to
   weight by, which PCE target to aim at) should be chosen based on which
   correlation actually turned out strongest, not decided in advance from
   a synthetic test run.

## Directory structure

```
project/
├── notebooks/      # one notebook per pipeline stage; each is independently runnable
├── src/            # helper modules only — notebooks call into these, not the reverse
│   ├── utils/      # environment, I/O, generic helpers
│   ├── datasets/    # dataset loading/splitting (from notebook 02 onward)
│   ├── models/      # model loading wrappers (from notebook 03 onward)
│   ├── metrics/      # PCE / NLL / CRPS / RMSE etc. (from notebook 06/10 onward)
│   ├── geometry/     # NRC1/NRC2/NRC3 implementations (from notebook 05 onward — gated, see above)
│   ├── calibration/   # closed-form NRC-weighted calibration map (from notebook 08 onward)
│   └── plotting/     # shared plotting utilities (from notebook 07 onward)
├── configs/        # Hydra configs
├── outputs/        # CSV / LaTeX tables
├── figures/        # generated figures
├── checkpoints/     # model checkpoints (synced to Google Drive when on Colab)
├── logs/           # environment reports, run logs
└── tests/          # unit tests for src/
```

## Three sync options (Colab ⇄ your MacBook)

**Option A — Google Drive (preferred, implemented, default).** Mount Drive,
place this folder under `MyDrive/`, and every notebook auto-locates it.

**Option B — GitHub.** Push this repo to your own GitHub remote, then set
`GIT_REPO_URL` in `configs/config.yaml`; notebook 00 will `git clone`/`git pull`
it automatically on a fresh Colab runtime. Empty by default until you've
pushed the repo somewhere.

**Option C — SSH / rsync — implemented, with an honest caveat.** Colab runs on
Google's cloud and **cannot reach a MacBook behind home NAT directly**. The
`rsync` helper in `src/utils/env_utils.py` is fully implemented and will work
*if* your Mac is reachable — e.g. via Tailscale, a reverse SSH tunnel, or a
static IP with port-forwarding. Without one of those, this option will time
out; that's a networking fact, not a bug in the code. Option A or B are the
practical defaults on a fresh Colab runtime.

## Running notebook 00

Open `notebooks/00_environment.ipynb` in Colab. It will mount Drive, locate
or create the project root, install dependencies from `requirements.txt`,
verify PyTorch/CUDA, detect system resources, and save an environment report
to `logs/environment_report.json`. No GPU is required for this notebook.
