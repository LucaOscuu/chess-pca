# Chess PCA — Morphological Framework

Replication code for the Master thesis *"Morphological Modelling for Strategic Intelligence"*.  
The pipeline extracts per-move positional features from PGN game files and applies a unified Principal Component Analysis to study game trajectories across outcomes and player perspectives.

> **Abstract** — This thesis studies whether strategic decision-making in high-dimensional environments can be represented through reduced geometric structure, complementing classical equilibrium analysis under practical computational limits. Using Stockfish self-play chess as an empirical laboratory, the results show that strategic dynamics admit a stable low-dimensional morphology: variance concentrates in a small set of latent directions that remain robust under substantial resampling. Although the latent space is estimated without outcome labels, projected trajectories display systematic regional organization across outcomes (Win/Draw/Loss), consistent with Empirical Attractor Basins.

---

## Project structure

```
chess-pca/
├── data/
│   ├── raw/                   # Input PGN / PGN.GZ files (not tracked by git)
│   ├── cache/                 # Cached parquet datasets (not tracked by git)
│   └── outputs/
│       └── final_figures/     # Final figures committed to the repo
├── scripts/
│   └── run_pipeline.py        # Entry point to run the full pipeline
└── src/
    └── chess_pca/
        ├── pipeline.py        # End-to-end orchestration
        ├── io_pgn.py          # PGN / PGN.GZ streaming ingestion
        ├── features.py        # Feature extraction (python-chess)
        ├── analysis.py        # PCA + trajectory utilities
        ├── plotting.py        # Shared plotting utilities
        ├── figure_loadings.py # Loading tables and profiles (PC1/PC2)
        ├── figure_h1.py       # Scree plot + null models for EVR(PC1+PC2)
        ├── figure_h2.py       # Bootstrap stability of PCA components
        ├── figure_h3.py       # Geometric drift on PC1 (Δz1)
        └── figure_h4_residence.py  # Residence time in attractor basins
```

---

## Setup

Requires **Python 3.9+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode
pip install -e .
```

---

## Usage

Place your PGN or PGN.GZ files in `data/raw/`, then run:

```bash
python scripts/run_pipeline.py
```

Figures will be saved to `data/outputs/`. Final figures intended for the thesis are stored in `data/outputs/final_figures/`.

---

## Hypotheses tested

| Figure | Hypothesis | Description |
|--------|-----------|-------------|
| H1 | Structured Dimensional Reduction | EVR explained by PC1+PC2 vs. null models |
| H2 | Stability of Latent Directions | Bootstrap cosine similarity of PCA components |
| H3 | Behavioral Relevance | Geometric drift on PC1 (Δz1) across outcomes |
| H4 | Empirical Attractor Basins | Residence time in outcome-aligned regions of latent space |

---

## Data

The pipeline was developed using the [Fishtest 2018 dataset](https://database.lichess.org/) — 9,999 Stockfish self-play games (LTC, classical chess).  
Raw data files are not tracked by git. Place them in `data/raw/` before running the pipeline.

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full list. Main dependencies:

- `python-chess` — PGN parsing and position evaluation
- `scikit-learn` — PCA
- `pandas` / `numpy` — data manipulation
- `matplotlib` — figures
- `pyarrow` — parquet caching

---

## License

This code is released for academic replication purposes.  
© 2026 Luca Osculati — ESCP Business School
