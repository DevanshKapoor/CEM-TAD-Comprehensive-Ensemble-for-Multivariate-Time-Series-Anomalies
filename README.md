CEM-TAD+ — Comprehensive Ensemble for Multivariate Time-Series Anomaly Detection
CEM-TAD+ is a GPU-ready and explainable anomaly detection system designed for multivariate time-series data. It processes your data and outputs a scored CSV file enriched with an Abnormality_score (ranging from 0 to 100) and the top contributing features (top_feature_1 through top_feature_7) for each timestamp.

An interactive Streamlit dashboard is also included to explore either the raw input data or the final scored results.

✨ Features
Five-Expert Ensemble: The core of the system is an ensemble model combining a Patch-Transformer forecaster, PCA reconstruction, Robust Mahalanobis distance (with Ledoit–Wolf covariance), an Isolation Forest, and a Correlation-Shift graph.

Robust Scoring: Anomaly scores are generated using a rank-average of the experts' outputs, calibrated to a 0–100 scale based on a training window, and smoothed with an Exponentially Weighted Moving Average (EWMA).

Built-in Explainability: The model identifies per-feature contributions to the anomaly score, listing the top 7 features that contribute more than 1% to the score.

Interactive Dashboard: The Streamlit app provides powerful visualizations, including a suggested anomaly threshold, event segmentation with KPIs, a feature-contribution frequency plot, time-of-day/weekday heatmaps, a correlation-change heatmap, and raw signal overlays.

📂 Project Layout
The project is organized as follows:

project/
├─ cemtad/
│  └─ streamlit_app.py
├─ scripts/
│  ├─ run_cemtad.py          # Main scoring pipeline
│  └─ eval_baselines.py      # (Optional) AUROC/AP vs baselines
├─ results/
├─ .streamlit/config.toml
├─ requirements.txt
└─ README.md
🚀 Quickstart
Follow these steps to get up and running.

1. Environment Setup
First, create and activate a virtual environment.

Bash

# Create and activate the environment
python -m venv .venv && source .venv/bin/activate
# For Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Install base requirements
pip install -r requirements.txt
For GPU acceleration with the forecaster model, install the appropriate PyTorch version.

Bash

# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only execution
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
2. Score a Dataset
Run the main pipeline script to process your time-series data. The output CSV will contain the original columns plus the Abnormality_score and top_feature_1..7.

Bash

PYTHONPATH=. python scripts/run_cemtad.py \
  --input TEP_Train_Test.csv \
  --output results/TEP_scored_cemtad.csv \
  --timestamp_col Time \
  --normal_start "2004-01-01 00:00" --normal_end "2004-01-05 23:59" \
  --window 60 --epochs 30 --corr_window 120 --patch 6 \
  --make_plots --plots_dir results/figs
3. Explore in the Dashboard
Launch the Streamlit application to visualize your raw or scored dataset.

Bash

streamlit run cemtad/streamlit_app.py
📊 (Optional) Evaluate Baselines
You can evaluate the ensemble's performance (AUROC/AP) against individual expert models using the eval_baselines.py script.

Bash

PYTHONPATH=. python scripts/eval_baselines.py \
  --input results/TEP_scored_cemtad.csv \
  --timestamp_col Time \
  --normal_start "2004-01-01 00:00" \
  --normal_end "2004-01-05 23:59"
🛡️ Edge-Case & Data-Quality Handling
The pipeline includes several checks to ensure robustness:

Missing Values: Handled using forward-fill (ffill) followed by linear interpolation.

Constant Features: A guard is in place to manage features that do not vary.

Training Window: The system warns if the training window is shorter than 72 hours or if high anomaly scores are detected within it.

Explainability: If fewer than 7 features contribute significantly to an anomaly, the remaining top_feature columns are left blank.

Numerical Stability: An optional tiny epsilon can be added to residuals to avoid issues with exact zeros.

⚙️ For Developers
To maintain code quality, we use black for formatting and ruff for linting.

Bash

# Install development tools
pip install black ruff

# Format and lint the codebase
black . -l 100
ruff check . --fix
🚢 Deployment
Local Deployment
Run the Streamlit app directly for local use.

Bash

streamlit run cemtad/streamlit_app.py