
# HPO: Convex Relaxations + Bayesian Optimization + Natural Gradients (Kappa)

**Goal.** Showcase a hybrid HPO strategy combining convex relaxations (for temperature-scaled surrogates),
Bayesian Optimization (Optuna), and natural-gradient training (K-FAC/JAX or PyTorch approximation).
Target metric: **Cohen's Kappa** on **Fashion-MNIST** and **Covertype**.

## Quickstart
```bash
make init
python -m src.hpo.run --dataset fashion_mnist --trials 20
```
