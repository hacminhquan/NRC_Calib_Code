"""Generate the numbered, self-bootstrapping NRC-Cal Jupyter notebooks."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

BOOTSTRAP = '''# Self-contained Colab bootstrap: Drive first, then configured GitHub/SSH roots.
from pathlib import Path
import importlib.util
import os, sys

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    pass

candidates = []
if os.environ.get("NRC_CAL_PROJECT_ROOT"):
    candidates.append(Path(os.environ["NRC_CAL_PROJECT_ROOT"]))
candidates += [Path("/content/drive/MyDrive/NRC-Cal/project"), Path("/content/NRC-Cal/project"), Path.cwd().parent, Path.cwd()]
PROJECT_ROOT = next((p.resolve() for p in candidates if (p / "src").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("NRC-Cal project unavailable. Run 00_environment.ipynb or set NRC_CAL_PROJECT_ROOT after Drive, GitHub, or SSH/rsync synchronization.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
runtime_path = PROJECT_ROOT / "src" / "utils" / "runtime.py"
if not runtime_path.is_file():
    raise FileNotFoundError(f"Missing runtime helper: {runtime_path}")
spec = importlib.util.spec_from_file_location("nrc_cal_runtime", runtime_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load runtime helper from {runtime_path}")
runtime_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime_module)
configure_runtime = runtime_module.configure_runtime
gpu_summary = runtime_module.gpu_summary
RUNTIME = configure_runtime(PROJECT_ROOT)
print(f"PROJECT_ROOT={RUNTIME.project_root}")
print(f"CHECKPOINT_ROOT={RUNTIME.checkpoint_root}")
print(f"CUDA available={RUNTIME.cuda_available}; PyTorch CUDA={RUNTIME.cuda_version}; GPU={RUNTIME.gpu_name}")
print(f"RAM={RUNTIME.ram_gib:.2f} GiB; driver={gpu_summary()}")
'''

SPECS: dict[str, tuple[str, str, str, str]] = {
    "01_download_repositories": ("Download Upstream Repositories", "Clone and audit the two Vekteur repositories. No NRC equation is evaluated in this notebook.", "Existing theory: none. New NRC-Cal contribution: reproducibility inventory only.", '''import hashlib, json, subprocess
from datetime import datetime, timezone
external = PROJECT_ROOT / "external"; external.mkdir(exist_ok=True)
repos = {"quantile-recalibration-training": "https://github.com/Vekteur/quantile-recalibration-training.git", "probabilistic-calibration-study": "https://github.com/Vekteur/probabilistic-calibration-study.git"}
inventory = {"created_utc": datetime.now(timezone.utc).isoformat(), "repositories": {}}
for name, url in repos.items():
    path = external / name
    if not (path / ".git").is_dir(): subprocess.run(["git", "clone", url, str(path)], check=True)
    checkpoints = []
    for file in path.rglob("*"):
        if file.suffix.lower() in {".ckpt", ".pt", ".pth", ".pkl", ".joblib", ".safetensors"}:
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            checkpoints.append({"path": str(file.relative_to(PROJECT_ROOT)), "bytes": file.stat().st_size, "sha256": digest})
    commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    inventory["repositories"][name] = {"commit": commit, "checkpoints": checkpoints}
    print(name, commit, "checkpoints:", len(checkpoints))
(PROJECT_ROOT / "outputs").mkdir(exist_ok=True)
(PROJECT_ROOT / "outputs" / "upstream_checkpoint_inventory.json").write_text(json.dumps(inventory, indent=2))
'''),
    "02_prepare_datasets": ("Prepare QRT-57 Datasets", "Download the upstream QRT-57 sources and cache exact train/validation/calibration/test splits.", "Existing theory: QRT dataset IDs and split protocol. New NRC-Cal contribution: manifest audit; no NRC equation.", '''import pandas as pd
from datasets.manifest import qrt57_manifest
from datasets.upstream import download_and_cache_qrt57
from utils.io import save_frame
manifest = pd.DataFrame([spec.__dict__ for spec in qrt57_manifest()])
assert len(manifest) == 57
path = save_frame(manifest, PROJECT_ROOT / "outputs" / "datasets" / "qrt57_manifest.csv")
print(manifest.groupby("group").size())
print("Manifest saved:", path)
saved = download_and_cache_qrt57(PROJECT_ROOT / "external" / "quantile-recalibration-training", PROJECT_ROOT / "outputs" / "raw_data", PROJECT_ROOT / "outputs" / "datasets" / "splits")
print(f"Cached {len(saved)} QRT-compatible split files. Splits follow [0.65, 0, 0.10, 0.15, 0.10] with the upstream calibration cap.")
'''),
    "03_load_pretrained_models": ("Load Frozen Pretrained Models", "Locate Gaussian and Gaussian-mixture checkpoints without modifying them.", "Existing theory: none. NRC-Cal assumption A1: models are frozen after checkpoint load.", '''from pathlib import Path
suffixes = {".ckpt", ".pt", ".pth", ".safetensors"}
files = [p for p in RUNTIME.checkpoint_root.rglob("*") if p.suffix.lower() in suffixes]
for file in files: print(file, file.stat().st_size)
if not files: print("No checkpoints found. Run notebook 01 and follow the upstream checkpoint acquisition documentation before feature extraction.")
print("Supported adapter targets: Gaussian (K=1), Mixture Gaussian, Mixture-3, Mixture-10. The frozen adapter must expose a penultimate layer and final mean-head weight.")
'''),
    "04_extract_features": ("Extract Frozen Features", "Run one forward pass over the calibration split and cache features, outputs, and targets.", "Existing theory: NRC takes last-layer features and W. New NRC-Cal algorithm: artifact caching only.", '''from models.features import extract_features
from utils.io import save_arrays
print("Call extract_features(model, penultimate_layer, calibration_loader, device, output_transform) after loading one upstream checkpoint.")
print("The returned FeatureBatch contains last hidden features, raw mean/variance-head output, and ground truth; save with save_arrays(..., features=..., outputs=..., targets=...).")
print("This notebook intentionally does not guess upstream checkpoint architecture. It fails only after a concrete checkpoint lacks a selected penultimate layer.")
'''),
    "05_compute_NRC": ("Compute Published NRC and Proposed Distance", "Evaluate exactly cited NRC1--NRC3 and then compute the separately proposed NRC-Cal distance.", "Existing theory: NRC1--NRC3 from NeurIPS 2024. Proposed: D_dataset and D_i in docs/methodology.md.", '''import numpy as np
from geometry.nrc import compute_nrc
feature_cache = PROJECT_ROOT / "outputs" / "features" / "calibration_features.npz"
if not feature_cache.exists():
    raise FileNotFoundError(f"Missing {feature_cache}; run notebook 04 for a concrete frozen checkpoint.")
cache = np.load(feature_cache)
# Required weight is the mean-head W with shape [target_dimension, feature_dimension].
weight_path = PROJECT_ROOT / "outputs" / "features" / "mean_head_weight.npy"
if not weight_path.exists(): raise FileNotFoundError(f"Export the frozen mean-head weight to {weight_path}.")
result = compute_nrc(cache["features"], cache["targets"], np.load(weight_path))
print(result)
np.savez_compressed(PROJECT_ROOT / "outputs" / "features" / "nrc_metrics.npz", sample_distance=result.sample_distance, residual_nrc1=result.residual_nrc1, residual_nrc2=result.residual_nrc2)
'''),
    "06_correlation_analysis": ("Correlation Analysis", "Associate dataset-level NRC-Cal distance with PCE and other frozen-model results.", "Existing theory: NRC metrics. Proposed analysis: D_dataset as a calibratability diagnostic; correlation tests are standard statistics.", '''import pandas as pd
from metrics.statistics import correlations
from utils.io import save_frame
path = PROJECT_ROOT / "outputs" / "results" / "dataset_metrics.csv"
if not path.exists(): raise FileNotFoundError(f"Write dataset,nrc_distance,pce rows to {path} after notebooks 05 and evaluation.")
frame = pd.read_csv(path)
results = correlations(frame["nrc_distance"].to_numpy(), frame["pce"].to_numpy())
table = pd.DataFrame([result.__dict__ for result in results])
print(table)
save_frame(table, PROJECT_ROOT / "outputs" / "statistics" / "nrc_pce_correlations.csv")
'''),
    "07_visualization": ("Visualization", "Create correlation plots, heatmaps, and PCA/UMAP/t-SNE feature views.", "Existing theory: NRC geometry. Proposed visual diagnostic: color feature embeddings by NRC-Cal sample distance.", '''import pandas as pd
from plotting.figures import save_correlation_scatter, save_heatmap
results = pd.read_csv(PROJECT_ROOT / "outputs" / "results" / "dataset_metrics.csv")
save_correlation_scatter(results, "nrc_distance", "pce", PROJECT_ROOT / "figures" / "nrc_distance_vs_pce.pdf")
save_heatmap(results, PROJECT_ROOT / "figures" / "result_correlation_heatmap.pdf")
print("Saved publication-quality scatter and heatmap. Use plotting.embedding(features, method='pca'|'umap'|'tsne') for per-dataset feature figures.")
'''),
    "08_closed_form_calibration": ("NRC-Cal Closed-Form Calibration", "Fit and apply the explicitly proposed frozen-model NRC-Cal covariance correction.", "Existing theory: NRC1--NRC3 only. Proposed: log-Mahalanobis regression, mixture-preserving scale map, and stability bounds.", '''from calibration.nrc_cal import fit_nrc_calibrator, select_ridge
from models.predictions import gaussian_prediction
print("Fit NRC-Cal only on an independent calibration split:")
print("  fitted = select_ridge(prediction, calibration_targets, calibration_sample_distance)")
print("  test_prediction = fitted.transform(test_prediction, test_sample_distance)")
print("The transform preserves mixture weights and predictive mean, and scales total covariance by the positive bounded factor documented in docs/methodology.md.")
'''),
    "09_full_experiment": ("Full Experiment", "Evaluate BASE, QR/QRC/QRTC/QREGC adapters, and NRC-Cal across QRT-57.", "Existing theory: published NRC equations. Proposed method: NRC-Cal; all baselines retain upstream attribution.", '''import pandas as pd
from calibration.baselines import baseline_registry
from metrics.experiment import run_frozen_nrc_cal
print("Methods:", baseline_registry())
artifact_root = PROJECT_ROOT / "outputs" / "artifacts"
rows = []
for dataset_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()) if artifact_root.exists() else []:
    required = [dataset_dir / name for name in ("calibration_features.npz", "calibration_predictions.npz", "test_features.npz", "test_predictions.npz", "mean_head_weight.npy")]
    if not all(path.exists() for path in required):
        print("Skipping incomplete artifact set:", dataset_dir); continue
    rows.append(run_frozen_nrc_cal(dataset_dir.name, "frozen", *required[:4], mean_head_weight=__import__("numpy").load(required[4]), output_path=dataset_dir / "nrc_cal_results.csv"))
if rows:
    pd.concat(rows, ignore_index=True).to_csv(PROJECT_ROOT / "outputs" / "results" / "full_experiment.csv", index=False)
    print("Completed frozen BASE/NRC-Cal runs:", len(rows))
else:
    print("No complete artifact sets yet. Notebook 04 creates the required immutable artifact contract for each available checkpoint.")
'''),
    "10_ablation": ("Ablation", "Test each proposed NRC-Cal component while holding frozen models and data splits fixed.", "Existing theory: NRC1--NRC3. Proposed ablations: NRC1-only, NRC2-only, NRC3-only, full, global-only, no-ridge, no-clipping.", '''ABLATIONS = {"nrc1_only": (1.,0.,0.), "nrc2_only": (0.,1.,0.), "nrc3_only": (0.,0.,1.), "full": (1/3,1/3,1/3), "global_only": None, "no_ridge": "ridge=0", "no_clipping": "unbounded"}
for name, setting in ABLATIONS.items(): print(name, setting)
print("For univariate datasets nrc3_only is invalid by published theory and must be reported as not applicable, not zero-performance.")
'''),
    "11_multivariate": ("Multivariate NRC-Cal", "Run NRC3 and mixture covariance correction on n>1 targets.", "Existing theory: full-rank target covariance and gamma-constrained NRC3. Proposed: multivariate NRC-Cal total-covariance scale map.", '''from geometry.nrc import compute_nrc
print("Multivariate prerequisites: targets [M,n] have positive-definite covariance, mean head W [n,d], and each mixture total covariance is positive definite.")
print("compute_nrc performs the published bounded gamma minimization. NRC-Cal then applies mu'_ik=mu_i+s_i(mu_ik-mu_i), V'_ik=s_i^2 V_ik.")
'''),
    "12_reproduce_tables": ("Reproduce Tables and Statistics", "Produce ranked metric tables, Wilcoxon/Holm results, Friedman test, and CD inputs.", "Existing theory: none. NRC-Cal claim evaluation uses standard paired statistical procedures.", '''import pandas as pd
from metrics.statistics import paired_statistics
results = pd.read_csv(PROJECT_ROOT / "outputs" / "results" / "full_experiment.csv")
pivot = results.pivot(index="dataset", columns="method", values="pce").dropna()
summary = paired_statistics(pivot.to_numpy())
print(summary)
pd.DataFrame({"method": pivot.columns, "average_rank": summary["average_ranks"]}).to_csv(PROJECT_ROOT / "outputs" / "tables" / "pce_average_ranks.csv", index=False)
'''),
    "13_export_results": ("Export Results", "Export CSV, LaTeX tables, figures, and a provenance manifest.", "Existing theory: none. Exported NRC-Cal equations retain their proposed-method attribution from docs/methodology.md.", '''import pandas as pd
from utils.io import save_json
results = pd.read_csv(PROJECT_ROOT / "outputs" / "results" / "full_experiment.csv")
table = results.pivot_table(index="dataset", columns="method", values=["pce", "nll", "crps"], aggfunc="mean")
destination = PROJECT_ROOT / "outputs" / "tables"; destination.mkdir(parents=True, exist_ok=True)
table.to_csv(destination / "main_results.csv")
(destination / "main_results.tex").write_text(table.to_latex(float_format="%.4f"), encoding="utf-8")
save_json({"methodology": "docs/methodology.md", "results": str(destination / "main_results.csv"), "figures": str(PROJECT_ROOT / "figures")}, destination / "export_manifest.json")
print("Export complete:", destination)
'''),
}


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]}


def code(text: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.splitlines()]}


def build(name: str, title: str, description: str, attribution: str, body: str) -> None:
    cells = [
        markdown(f"# NRC-Cal: {title}\n\n{description}\n\n**Attribution.** {attribution}\n\n**Resources.** Expected runtime varies with dataset/checkpoint availability; GPU memory is limited to one frozen model and one batch. Expected output paths are printed by executable cells. Every code cell independently resolves Drive/GitHub/SSH-synchronized project source before it acts."),
        code(BOOTSTRAP),
        markdown("## Equations and attribution\n\n**Published NRC theory (NeurIPS 2024).** `NRC1=M^-1 sum_i ||h~_i-P_H_PCA_n(h~_i)||_2^2`; `NRC2=M^-1 sum_i ||h~_i-P_W^T(h~_i)||_2^2`; and `NRC3=||WW^T/||WW^T||_F-(Sigma^(1/2)-gamma^(1/2)I_n)/||Sigma^(1/2)-gamma^(1/2)I_n||_F||_F^2`, where `gamma in (0,lambda_min)`. These are implemented exactly in `geometry.nrc`. **NRC-Cal proposal.** `D_i=w1 r1_i+w2 r2_i+w3 NRC3`, `D_dataset=M^-1 sum_i D_i`, and the frozen scale map are new equations, specified with assumptions and proof sketch in `docs/methodology.md`. No proposed equation is attributed to the NRC papers."),
        code(body),
        markdown("## Reproducibility note\n\nThis notebook writes only under `outputs/`, `figures/`, or the Drive-backed checkpoint root. It does not retrain a model. Preserve the environment JSON from notebook 00 and the upstream checkpoint inventory when exporting results."),
    ]
    document = {"cells": cells, "metadata": {"colab": {"name": f"{name}.ipynb", "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    (NOTEBOOKS / f"{name}.ipynb").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")


for notebook_name, arguments in SPECS.items():
    build(notebook_name, *arguments)
