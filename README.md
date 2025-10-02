# Anomaly Detection (Time Series & Non Time Series)

This repo provides a minimal, CPU-friendly setup to run anomaly detection from a single notebook with two supporting modules.

## Install

```bash
(python -m) pip install -r requirements.txt
```

```bash
# Optional: use a venv first
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate                              # Windows
```

## File Structure

```
anomaly_detection/
├── time_series_anomaly.py
├── non_time_series_anomaly.py
├── anomaly_runner.ipynb
└── requirements.txt
```

- **time_series_anomaly.py**: Univariate & multivariate **time series** anomaly detectors, loaders (URL/Excel/path), synthetic & public fallbacks, and plotting helpers (IQR band + red "prophet" markers).
- **non_time_series_anomaly.py**: Univariate & multivariate **tabular** anomaly detectors, loaders (URL/Excel/path), synthetic & public fallbacks, and plotting helpers.
- **anomaly_runner.ipynb**: One-click notebook that fetches **3 public + 1 synthetic** datasets for **both** time series and non-time-series, renders **modern IQR plots**, and prints **separate benchmark tables**.
- **requirements.txt**: Minimal dependencies; optional extras are commented.

## Quick Start

1. Install dependencies (optionally, in a virtual environment).
2. Open `anomaly_runner.ipynb` in Jupyter (or Colab) and **Run All**.
3. Results:
   - **Table-1**: Time series benchmarks (3 public + 1 synthetic).
   - **Table-2**: Non time series benchmarks (3 public + 1 synthetic).
   - IQR plots with shaded Q1–Q3 band, dashed whiskers, blue series line, and red anomaly markers labeled **"prophet"**.

## Notes

- Public datasets are fetched from stable, well-known GitHub sources.
- If external URLs are blocked, the notebook falls back to synthetic data so everything still runs on CPU.
