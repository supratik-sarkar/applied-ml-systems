
# Recommendation — FFM + NCF (+ Shampoo) on MovieLens

**Goal.** Production-grade ranking framework with **FastFM/FFM-style features** + **Neural Collaborative Filtering** and
**Shampoo optimizer**. Benchmarks on **MovieLens 20M** (fallback: 1M). Includes GPU-accelerated matrix factorization.

## Quickstart
```bash
make init
python -m src.data.movielens --download 20m
python -m src.train_ffm --epochs 3
python -m src.train_ncf --epochs 3 --optimizer shampoo
python -m src.eval --k 10
```
