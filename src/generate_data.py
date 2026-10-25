"""Generate a realistic mock dataset for credit risk assessment.

The target `loan_approved` is drawn from a logistic model over the applicant's
financial profile, so the relationships are learnable but not deterministic.

Usage:
    python src/generate_data.py [--rows 1000] [--seed 42]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "credit_data.csv"

HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN"]
LOAN_PURPOSE = ["DEBT_CONSOLIDATION", "HOME_IMPROVEMENT", "EDUCATION", "MEDICAL", "BUSINESS", "PERSONAL"]


def generate(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n_rows)

    # Experience grows with age, capped by working years.
    employment_years = np.clip(rng.gamma(2.0, 3.0, n_rows), 0, age - 18).round(1)

    # Log-normal income, rising modestly with experience.
    annual_income = (rng.lognormal(mean=10.8, sigma=0.45, size=n_rows) * (1 + 0.02 * employment_years)).round(-2)
    annual_income = np.clip(annual_income, 15_000, 400_000)

    # Loan requests scale with income.
    loan_amount = (annual_income * rng.uniform(0.05, 0.6, n_rows)).round(-2)
    loan_amount = np.clip(loan_amount, 1_000, 150_000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_rows, p=[0.1, 0.2, 0.35, 0.15, 0.2])

    credit_score = np.clip(rng.normal(680, 70, n_rows) + 1.5 * employment_years, 300, 850).round().astype(int)

    # Existing debt load (excluding the new loan) as fraction of income.
    debt_to_income = np.clip(rng.beta(2.2, 6.0, n_rows) + (720 - credit_score) / 3000, 0.0, 0.95).round(3)

    num_credit_lines = rng.poisson(5, n_rows)
    past_defaults = rng.choice([0, 1, 2, 3], n_rows, p=[0.78, 0.14, 0.06, 0.02])

    home_ownership = rng.choice(HOME_OWNERSHIP, n_rows, p=[0.4, 0.4, 0.2])
    loan_purpose = rng.choice(LOAN_PURPOSE, n_rows)

    loan_to_income = loan_amount / annual_income

    # Latent creditworthiness score -> approval probability.
    logit = (
        -0.5
        + 0.018 * (credit_score - 680)
        - 6.0 * (debt_to_income - 0.3)
        - 3.0 * (loan_to_income - 0.3)
        + 0.08 * np.minimum(employment_years, 15)
        - 1.3 * past_defaults
        + 0.4 * (home_ownership == "OWN")
        + 0.2 * (home_ownership == "MORTGAGE")
        - 0.3 * (loan_purpose == "BUSINESS")
        + 0.15 * (loan_purpose == "DEBT_CONSOLIDATION")
        - 0.01 * np.abs(age - 45)
    )
    prob = 1 / (1 + np.exp(-logit))
    loan_approved = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "num_credit_lines": num_credit_lines,
            "past_defaults": past_defaults,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "loan_approved": loan_approved,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"Saved {len(df)} rows to {DATA_PATH}")
    print(f"Approval rate: {df['loan_approved'].mean():.1%}")


if __name__ == "__main__":
    main()
