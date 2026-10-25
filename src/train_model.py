"""Train an XGBoost credit approval model and save it with LIME baseline data.

Outputs (in models/):
    xgb_model.joblib   - the fitted XGBClassifier
    artifacts.joblib   - feature metadata, category encodings, training data
                         for the LIME explainer, per-feature stats and metrics

Usage:
    python src/train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "credit_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"
ARTIFACTS_PATH = MODEL_DIR / "artifacts.joblib"

TARGET = "loan_approved"
CATEGORICAL = ["home_ownership", "loan_purpose"]
CLASS_NAMES = ["Rejected", "Approved"]


def preprocess(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Integer-encode categorical columns using fixed category lists.

    Integer codes (rather than one-hot) keep one column per feature, which
    LIME needs in order to treat each categorical as a single feature.
    """
    df = df.copy()
    for col, values in categories.items():
        df[col] = pd.Categorical(df[col], categories=values).codes
        if (df[col] < 0).any():
            raise ValueError(f"Unknown category in column '{col}'")
    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    categories = {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL}
    X = preprocess(df.drop(columns=[TARGET]), categories)
    y = df[TARGET]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")

    # Summary of the raw (un-encoded) training data, used for sidebar ranges/defaults.
    raw_train = df.loc[X_train.index]
    numeric_stats = {
        col: {
            "min": float(raw_train[col].min()),
            "max": float(raw_train[col].max()),
            "median": float(raw_train[col].median()),
            "is_int": bool(np.issubdtype(raw_train[col].dtype, np.integer)),
        }
        for col in feature_names
        if col not in CATEGORICAL
    }

    artifacts = {
        "feature_names": feature_names,
        "categorical_features": [feature_names.index(c) for c in CATEGORICAL],
        "categories": categories,
        "class_names": CLASS_NAMES,
        # LIME samples perturbations around the distribution of this data.
        "lime_training_data": X_train.to_numpy(dtype=float),
        "numeric_stats": numeric_stats,
        "metrics": metrics,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved artifacts to {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
