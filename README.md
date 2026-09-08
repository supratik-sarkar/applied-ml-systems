# applied-ml-systems

A historical applied machine learning engineering archive consolidating foundational systems and pipelines developed across 2025.

---

## Architectural Scope & Archive Character

This repository is deliberately preserved as an archive of earlier applied machine learning engineering projects. It is structured under `legacy/` to maintain clean historical fidelity and provenance:

```
applied-ml-systems/
├── legacy/
│   ├── anomaly-detection/        Statistical & unsupervised tabular/time-series anomaly detection
│   ├── optimization-hpo/         Convex optimization, Bayesian optimization & natural gradient HPO
│   └── recommenders/             Field-aware factorization machines (FFM) & neural collaborative filtering
├── scripts/
│   └── verify_all.sh             Archive verification suite
├── LICENSE                       MIT License
├── LICENSES.md                   License mapping
└── README.md
```

> **Note on Provenance**: Historical commit timestamps document the actual development history of these components in 2025. This repository collects prior applied ML workflows and does not represent 2026 agentic or generative AI systems.

---

## Consolidated Engineering Modules

### 1. Anomaly Detection (`legacy/anomaly-detection/`)
Statistical, distance-based, and unsupervised algorithms for detecting outliers in temporal and multi-attribute tabular data:
* **Time-Series Anomaly Detection (`time_series_anomaly.py`)**: Rolling statistical bounds, EWMA smoothing, and seasonal decomposition for outlier identification.
* **Non-Time-Series Anomaly Detection (`non_time_series_anomaly.py`)**: Isolation Forest, Local Outlier Factor (LOF), and Mahalanobis distance metrics.
* **Exploratory Analysis Notebook (`anomaly_runner_v2.ipynb`)**: End-to-end evaluation and visualization workflows.

### 2. Hyperparameter Optimization (`legacy/optimization-hpo/`)
Scalable tuning and search algorithms for machine learning parameter landscapes:
* **Algorithms**: Bayesian optimization with Gaussian Processes, natural gradient search, and convex relaxation routines (`src/train_hpo.py`).
* **Pipelines**: Automated search loop orchestration (`src/run.py`).

### 3. Recommendation Systems (`legacy/recommenders/`)
High-dimensional sparse collaborative filtering and factorization models:
* **Architectures**: Field-aware Factorization Machines (FFM) and Neural Collaborative Filtering (NCF) (`src/train_reco.py`).
* **Optimizers**: Shampoo second-order matrix preconditioner integration and Adam baselines.
* **Datasets**: MovieLens preprocessing and sparse batching pipelines (`src/data_movielens.py`).

---

## Verification & Reproducibility

Each legacy subsystem preserves its genuine reproducibility paths and tests:

```bash
./scripts/verify_all.sh
```

The script executes:
1. Strict Python 3.12.13 version confirmation (fails closed on mismatch);
2. Syntax and static compilation across anomaly detection modules (`non_time_series_anomaly.py`, `time_series_anomaly.py`);
3. Unit test execution in `legacy/optimization-hpo/tests/` (1 test passed);
4. Unit test execution in `legacy/recommenders/tests/` (1 test passed).

Across the four engineering umbrellas, 417 modern-component tests pass, plus this archive's 2 historical unit tests and 2 anomaly-module syntax checks.

---

## Historical Lineage & Provenance

This repository is anchored on the genuine 2025 `anomaly_detection` GitHub repository object. Historical constituent projects are preserved under `legacy/`:
* `legacy/anomaly-detection/`: Statistical & unsupervised tabular/time-series anomaly detection.
* `legacy/optimization-hpo/`: Convex optimization, Bayesian optimization & natural gradient HPO.
* `legacy/recommenders/`: Field-aware factorization machines (FFM) & neural collaborative filtering.

In September 2026, the repository was transitioned into the `applied-ml-systems` umbrella with un-squashed Git commit histories.

See [HISTORY.md](HISTORY.md) for full lineage proofs, earliest commit timestamps, component tags, and commit links.

### Git Tags & Provenance Anchor

Original historical repository commit tips are preserved via tags:
* `legacy/anomaly-detection-2025`
* `legacy/optimization-hpo-2025`
* `legacy/recommenders-2025`

## License

This archive and all constituent modules are licensed under the MIT License. See [LICENSE](LICENSE) and [LICENSES.md](LICENSES.md).
