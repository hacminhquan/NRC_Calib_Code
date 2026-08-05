# NRC-Cal

NRC-Cal is a notebook-first, frozen-model study of whether Neural Regression
Collapse (NRC) geometry predicts probabilistic calibratability. The project
reproduces the NRC1--NRC3 metrics from the primary literature and explicitly
labels the NRC-Cal distance and calibration map as new proposed methodology.

Start in `notebooks/00_environment.ipynb`, then run the numbered notebooks.
Each notebook has Drive and SSH/rsync project-access instructions.

## Method attribution

`docs/methodology.md` is the normative specification. It separates published
NRC definitions from NRC-Cal's new assumptions, derivations, algorithms,
theorems, and ablations. Do not cite NRC-Cal equations as results from the
NRC papers.

## Reproducibility

The exact upstream repositories are stored under `external/`. The paper
sources used to transcribe published equations are stored under `references/`.
Every experiment writes feature caches, metric rows, environment provenance,
and figures below `outputs/` or `figures/`.
