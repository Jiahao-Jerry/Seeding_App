"""
SAE (Sparse Autoencoder) module — discovery instrument for delivery axes.

Trains a small SAE on within-subcluster BGE-M3 embedding residuals to surface
candidate delivery dimensions that may extend or validate the 7 LLM-annotated
axes. Runs offline; does not participate in the serving loop.

Modules:
- residuals.py    : compute residual = emb - subcluster_mean for each post
- model.py        : PyTorch SparseAutoencoder
- train.py        : training loop + per-post activation extraction
- interpret.py    : top/bottom activating posts per feature
- correlate.py    : feature ↔ LLM-axis lift (mean axis when active − mean when not)

Entry point: backend/m4_sae.py (orchestrator mirroring M1/M2/M3 style).
"""
