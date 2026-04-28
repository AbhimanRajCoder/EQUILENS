"""
pretrain_agent.py — Pre-train the PPO bias-mitigation agent on the Adult Income dataset.

Usage:
    cd backend/
    python -m scripts.pretrain_agent

Target column : income  (binary: <=50K / >50K)
Sensitive attr : sex     (Male / Female)
Timesteps      : 10,000
Output         : ppo_bias_agent.zip  (saved in the backend/ directory)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Resolve paths so the script works whether run from backend/ or scripts/
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from services.rl_agent import BiasEnv

# ---------------------------------------------------------------------------
# 1. Load the Adult Income dataset
# ---------------------------------------------------------------------------
ADULT_CSV_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

TARGET_COL = "income"
SENSITIVE_COL = "sex"


def load_adult_income() -> pd.DataFrame:
    """Download (or read cached) Adult Income CSV and clean it."""
    print("Loading Adult Income dataset …")
    df = pd.read_csv(
        ADULT_CSV_URL,
        header=None,
        names=ADULT_COLUMNS,
        na_values=" ?",
        skipinitialspace=True,
    )
    df.dropna(inplace=True)

    # Binarise target: 1 = >50K, 0 = <=50K
    df[TARGET_COL] = (df[TARGET_COL].str.strip().str.startswith(">50K")).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Compute baseline bias & accuracy from a quick logistic-regression model
# ---------------------------------------------------------------------------
def compute_baseline_metrics(df: pd.DataFrame):
    """Return (bias_score, accuracy) derived from the dataset."""
    X = pd.get_dummies(df.drop(columns=[TARGET_COL]))
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))

    # Demographic-parity gap across sex groups  →  used as bias_score
    test_df = X_test.copy()
    test_df["prediction"] = y_pred
    test_df[SENSITIVE_COL] = df.loc[X_test.index, SENSITIVE_COL].values

    approval_rates = (
        test_df.groupby(SENSITIVE_COL)["prediction"].mean()
    )
    bias_score = float(approval_rates.max() - approval_rates.min())

    print(f"  Baseline accuracy : {accuracy:.4f}")
    print(f"  Baseline bias     : {bias_score:.4f}  (demographic-parity gap on '{SENSITIVE_COL}')")
    return bias_score, accuracy


# ---------------------------------------------------------------------------
# 3. Pre-train PPO agent
# ---------------------------------------------------------------------------
def pretrain():
    df = load_adult_income()
    bias_score, accuracy = compute_baseline_metrics(df)

    print(f"\nCreating BiasEnv(initial_bias={bias_score:.4f}, initial_accuracy={accuracy:.4f})")
    env = BiasEnv(initial_bias=bias_score, initial_accuracy=accuracy)

    model = PPO("MlpPolicy", env, verbose=1)

    print("Training PPO for 10,000 timesteps …")
    model.learn(total_timesteps=10_000)

    # Save into the backend/ directory so the server can find it
    save_path = os.path.join(BACKEND_DIR, "ppo_bias_agent")
    model.save(save_path)
    print(f"\n✓ Model saved to {save_path}.zip")


if __name__ == "__main__":
    pretrain()
