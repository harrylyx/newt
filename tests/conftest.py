import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def german_credit_data():
    """
    Load German Credit Data and train a simple model to get realistic scores.
    Returns dictionary with y_true, y_prob (train and test).
    """
    # Locating the file relative to this test file or project root
    # Assuming tests are run from project root, but let's be robust
    base_path = os.path.dirname(os.path.dirname(__file__))  # risk_toolkit/
    data_path = os.path.join(
        base_path,
        "examples",
        "data",
        "statlog+german+credit+data",
        "german.data-numeric",
    )

    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found at {data_path}")

    # Load data
    # german.data-numeric has no headers, whitespace separated.
    # Last column (25th usually, or check doc) is target.
    # Doc says 24 numerical attributes. So 25th column is class.
    df = pd.read_csv(data_path, delim_whitespace=True, header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Target: 1 = Good, 2 = Bad. Remap to 0 = Good, 1 = Bad.
    y = (y == 2).astype(int)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    # Get probabilities
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    return {
        "train": {"y_true": y_train.values, "y_prob": y_prob_train},
        "test": {"y_true": y_test.values, "y_prob": y_prob_test},
        "all": {
            "y_true": y,
            "y_prob": np.concatenate(
                [y_prob_train, y_prob_test]
                # Note: order mixed if not careful
            ),
        },
    }
