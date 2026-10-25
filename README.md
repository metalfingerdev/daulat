# Daulat

Daulat is an explainable credit-risk demo. It generates a synthetic loan dataset, trains an XGBoost approval model, and serves an interactive Streamlit dashboard with local LIME explanations.

## Features

- Synthetic applicant and loan data generation with a reproducible seed
- XGBoost binary classifier for loan approval
- Saved model and metadata artifacts for repeatable inference
- Interactive applicant form with approval probability and test metrics
- LIME feature contributions explaining each individual prediction

## Project layout

```text
.
├── app.py                         # Streamlit dashboard
├── data/credit_data.csv           # Generated training data
├── models/
│   ├── artifacts.joblib           # Encodings, LIME data, input ranges, metrics
│   └── xgb_model.joblib           # Trained XGBoost model
├── src/
│   ├── generate_data.py           # Synthetic data generator
│   └── train_model.py             # Training and artifact export
├── docs/                          # Development notes
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the dashboard

The repository includes generated data and model artifacts. Start the app with:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

The app can also be launched with `python app.py`; it re-executes itself through Streamlit.

## Rebuild data and model

Generate a new deterministic dataset, then train the model:

```bash
python src/generate_data.py --rows 1000 --seed 42
python src/train_model.py
```

The training script writes `models/xgb_model.joblib` and `models/artifacts.joblib`. Restart or refresh Streamlit after replacing these files so its cached resources are reloaded.

## Important behavior

- The target column is `loan_approved`.
- Categorical values are integer-encoded using the category lists saved in the artifacts.
- Approval is displayed when the predicted approved probability is at least `0.5`.
- LIME explanations describe local surrogate contributions; they are not causal explanations.
- The dataset is synthetic and the model is for demonstration only, not real lending decisions.

## Development notes

See [docs/development.md](docs/development.md) for the training pipeline, conventions, validation checks, and common troubleshooting steps.