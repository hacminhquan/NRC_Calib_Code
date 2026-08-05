# NRC-Cal Colab Run Guide

This is the exact execution order for a Colab Pro T4 runtime. Run notebooks from top to bottom; each notebook is independently bootstrapped, but later notebooks consume artifacts produced by earlier ones.

## 1. Start the Colab runtime

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Open `project/notebooks/00_environment.ipynb`.
4. Run every cell in order. When prompted, authorize Google Drive.
5. In Cell 3A, set `DRIVE_PROJECT_ROOT` if your folder is not `MyDrive/NRC_CALIB_CODE`.
6. Confirm Cell 5 prints `CUDA available: True` and a T4 GPU name.

The preferred layout in Drive is:

```text
MyDrive/NRC_CALIB_CODE/
```

The notebook also supports SSH/rsync (Cell 3B). Configure only the option you use; leave the others unchanged.

## 2. Run the numbered notebooks

Run all cells in this order:

| Order | Notebook | Main output |
|---:|---|---|
| 1 | `00_environment.ipynb` | `outputs/environment.json` |
| 2 | `01_download_repositories.ipynb` | `external/` and checkpoint manifest |
| 3 | `02_prepare_datasets.ipynb` | dataset manifest and split metadata |
| 4 | `03_load_pretrained_models.ipynb` | model manifest/checkpoint cache |
| 5 | `04_extract_features.ipynb` | frozen feature and prediction caches |
| 6 | `05_compute_NRC.ipynb` | NRC1--NRC3 and NRC-distance rows |
| 7 | `06_correlation_analysis.ipynb` | Pearson/Spearman/Kendall and uncertainty results |
| 8 | `07_visualization.ipynb` | figures under `figures/` |
| 9 | `08_closed_form_calibration.ipynb` | fitted NRC-Cal calibration artifacts |
| 10 | `09_full_experiment.ipynb` | complete experiment results |
| 11 | `10_ablation.ipynb` | ablation results |
| 12 | `11_multivariate.ipynb` | multivariate results |
| 13 | `12_reproduce_tables.ipynb` | tables and statistical tests |
| 14 | `13_export_results.ipynb` | final CSV, LaTeX, and publication figures |

Do not skip a notebook unless its required input artifact already exists and is visibly present in `project/outputs/` or `project/checkpoints/`.

## 3. What to check after each stage

- After notebook 01: both upstream repositories exist below `project/external/`.
- After notebook 02: the dataset manifest reports the expected datasets/splits.
- After notebook 03: the model manifest identifies available Gaussian and mixture checkpoints.
- After notebook 04: feature cache files are non-empty.
- After notebook 05: NRC metric output contains NRC1, NRC2, NRC3, and distance columns.
- After notebook 08: calibration output contains fitted coefficients and validation metrics.
- After notebook 13: inspect `project/outputs/tables/`, `project/outputs/csv/`, and `project/figures/`.

## 4. If Colab reports `ModuleNotFoundError: utils.runtime`

This usually means Colab is executing an old notebook copy. Close the notebook tab, reopen the current file from Drive, and run **Runtime -> Restart session**, then rerun notebook 00 first. The current notebooks load `project/src/utils/runtime.py` by file path and do not depend on a fragile package import.

You can verify the notebook is current by searching the first code cell for `spec_from_file_location`. If it is absent, replace the Drive copy with the latest `project/notebooks/01_download_repositories.ipynb` from this repository.

## 5. Resume after disconnects

Reconnect to the same Drive project and rerun notebook 00. Existing manifests, caches, checkpoints, and figures are reused when present. Never delete `outputs/` or `checkpoints/` when resuming.

## 6. Local validation before Colab (optional)

From the repository root:

```bash
python3 -m pytest project/tests -q
```

The notebooks are the experiment entry points; `project/src/` contains reusable helpers only.
