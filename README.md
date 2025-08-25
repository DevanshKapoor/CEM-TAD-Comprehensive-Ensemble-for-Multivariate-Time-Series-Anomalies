# CEM-TAD+ — Comprehensive Ensemble for Multivariate Time-Series Anomaly Detection

**Live demo:** https://cem-tad-comprehensive-ensemble-for-multivariate-time-series-an.streamlit.app/  
**Tech:** Python • scikit-learn • (optional) PyTorch • Plotly • Streamlit

CEM-TAD+ is an explainable ensemble for **multivariate** time-series anomaly detection.  
It outputs a single **0–100 Abnormality_score** per row **and** the **Top-7 contributing features** (`top_feature_1..7`) so operators can see *what* triggered an alert.

---

## Key features

- **Five complementary experts** catch different failure modes:  
  1) **Temporal forecaster (Patch-Transformer style)** → per-feature residuals `|ŷ−y|`  
  2) **PCA reconstruction** → low-rank manifold deviation  
  3) **Robust Mahalanobis (Ledoit–Wolf)** → multivariate distance, per-feature z² contributions  
  4) **Isolation Forest** → partition-based outlier score (LOFO-style per-feature proxy)  
  5) **Correlation-shift graph** → rolling correlation vs training; L1 matrix Δ; per-feature = row-sum

- **Ensemble & scoring:** rank-average expert scores → **robust calibration** (median/MAD + percentile anchor) → **0–100** → **EWMA** smoothing  
- **Explainability:** `top_feature_1..7` per row (>\~1% rule; blanks allowed if fewer than 7)  
- **Two modes:**  
  - **Raw CSV** → run full pipeline and produce scored CSV  
  - **Scored CSV** → load and analyze instantly in the UI  
- **Operator-friendly dashboard:** threshold suggestions (Youden-J / train-FPR / legacy), event segmentation, contributors frequency, train-vs-rest histogram, time-of-day heatmap, correlation-Δ heatmap, raw overlays  
- **GPU-first forecaster** (A100-ready), other experts are lightweight on CPU

---

## Try it now (Streamlit)

**Live app:** https://cem-tad-comprehensive-ensemble-for-multivariate-time-series-an.streamlit.app/

- Use **“Scored file”** mode for an instant tour, or **“Raw file”** to run the full pipeline in-app (CPU-friendly defaults).  
- Upload a CSV with a timestamp column (e.g., `Time`) and numeric features.

---

## 📦 Installation (local)

```bash
# create & activate a venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt
# Optional (GPU forecaster): install PyTorch per your CUDA setup at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
```

## Usage
1) Run the pipeline (CLI)
From the project root:

```bash
export PYTHONPATH=.
python scripts/run_cemtad.py \
  --input TEP_Train_Test.csv \
  --output results/TEP_scored_cemtad.csv \
  --timestamp_col Time \
  --normal_start "2004-01-01 00:00" \
  --normal_end   "2004-01-05 23:59" \
  --window 60 --epochs 30 --corr_window 120 --patch 6 \
  --smoothing_alpha 0.15 \
  --make_plots --plots_dir results/figs
```
# Outputs
results/TEP_scored_cemtad.csv → original columns + Abnormality_score + top_feature_1..7

results/figs/ → scores_over_time.png, score_hist_train_vs_rest.png, top_features_at_<idx>.png

results/TEP_scored_cemtad_report.json → rows, features, calibration params


## Project Layout

repo/
├─ cemtad/
│  ├─ init.py
│  ├─ data.py           # load/validate/resample/scale + normal mask
│  ├─ experts.py        # 5 experts (temporal, PCA, Mahalanobis, IF, corr-shift)
│  ├─ ensemble.py       # rank-average fuse
│  ├─ scoring.py        # robust calibration → 0–100 + EWMA
│  ├─ attribution.py    # Top-7 contributors
│  ├─ visualize.py      # timeline, hist, top-features bar
│  └─ streamlit_app.py  # dashboard (raw/scored modes)
├─ scripts/
│  └─ run_cemtad.py     # CLI entry (run_pipeline)
├─ results/             # sample outputs & figures
├─ requirements.txt
├─ README.md
└─ .streamlit/config.toml


## Results
Results
(attained on TEP slice; proxy labels = train vs rest)

Method	          AUROC	 AP
CEM-TAD+ (0–100)	0.934	0.976
Mahalanobis (LW)	0.848	0.947

Export to Sheets
Training window KPIs: mean 3.87, max 19.35, p99 15.88 (target: mean <10, max <25)

Suggested threshold: ≈ 60 (Youden-J / train-FPR 0.5% / legacy)

Plots: see results/figs/ — timeline with training band, train-vs-rest histogram, Top-7 at peak.


## How It Works

flowchart LR
    A[Raw CSV] --> B[Preprocess\nparse → 1-min → impute → scale]
    B --> E1[Forecaster residuals]
    B --> E2[PCA recon error]
    B --> E3[Mahalanobis (LW)]
    B --> E4[Isolation Forest]
    B --> E5[Correlation shift]
    E1 --> F[Rank-average]
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F
    F --> C[Calibrate median/MAD + q99 → 0–100]
    C --> S[EWMA smoothing]
    S --> T{Threshold suggest}
    T --> G[Event segmentation]
    S --> O1[Scored CSV (+Top-7)]
    C --> O2[Report JSON]
    S --> O3[Plots]


##  Data Assumptions

Timestamp column (e.g., Time) parseable by pandas.to_datetime.

Minute-level alignment (the pipeline resamples to 1-min using "min").

Missing values handled via ffill/bfill + time-interpolation.

A normal training window must be provided for robust calibration (e.g., 2004-01-01 00:00 → 2004-01-05 23:59).


## References

Nie, Y. et al. PatchTST: A Time Series is Worth 1D Patches (ICLR 2023)

Liu, F. T. et al. Isolation Forest (ICDM 2008)

Ledoit, O. & Wolf, M. A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices (J Multivar Anal, 2004)

scikit-learn, PyTorch, Plotly, Streamlit documentation

Downs & Vogel. Tennessee Eastman Process (Computers & Chemical Engineering, 1993)
