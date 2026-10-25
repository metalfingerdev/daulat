# Development Notes

## Pipeline

1. `src/generate_data.py` creates `data/credit_data.csv` from a seeded NumPy random generator.
2. `src/train_model.py` reads the CSV, removes missing rows, encodes the two categorical columns, and performs a stratified 80/20 train-test split.
3. The script trains an `XGBClassifier`, prints a classification report and ROC AUC, and saves the model plus inference metadata under `models/`.
4. `app.py` loads both joblib files, applies the saved category encodings to sidebar input, predicts approval probability, and creates a LIME explanation.

## Data contract

The generated dataset contains these feature columns:

```text
age, annual_income, employment_years, loan_amount,
loan_term_months, credit_score, debt_to_income,
num_credit_lines, past_defaults, home_ownership, loan_purpose
```

The target is `loan_approved`. Keep the feature names and their order aligned with the artifacts. The app uses the saved `feature_names` list when constructing the prediction dataframe.

Categorical columns are currently:

- `home_ownership`
- `loan_purpose`

If categories or features change, regenerate the dataset and retrain the model before starting the app.

## Reproducibility

The default generator seed and the model split/model seeds are both `42`. To compare model changes fairly, keep the row count and seed fixed. Changing generated data invalidates the existing model artifacts until training is run again.

## Validation checklist

Before committing a change:

```bash
python src/generate_data.py --rows 1000 --seed 42
python src/train_model.py
streamlit run app.py
```

Confirm that:

- training completes and writes both files in `models/`;
- the dashboard loads without a missing-model message;
- the sidebar accepts the generated feature ranges;
- the approval decision, probability, metrics, chart, and contribution table render;
- changing an applicant input changes the prediction or explanation where expected.

For a quick import check without starting the dashboard, run:

```bash
python -c "import app; print('app import ok')"
```

## Design decisions

### Integer encoding

The categorical features use fixed integer codes rather than one-hot encoding. This keeps one column per logical feature, which lets `LimeTabularExplainer` treat each categorical feature as a single input dimension.

### Cached resources

The model and LIME explainer are cached with Streamlit resource caching. After retraining, use the Streamlit menu to clear the cache or restart the process if the dashboard appears to use stale metrics or predictions.

### Synthetic data limitations

The target is sampled from a hand-authored logistic relationship. It is useful for demonstrating the pipeline, but it does not represent a validated underwriting policy, real-world population, or production fairness assessment.

## Common troubleshooting

### Missing model files

Run the generator and trainer from the repository root:

```bash
python src/generate_data.py
python src/train_model.py
```

### Dependency errors

Make sure the active Python environment is the one where `requirements.txt` was installed:

```bash
python -m pip install -r requirements.txt
```

### Stale dashboard output

Restart Streamlit or clear its cache after changing `data/credit_data.csv` or either file in `models/`.