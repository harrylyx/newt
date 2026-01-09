"""
Cross-validation tests comparing newt with toad library.

Uses German Credit dataset to validate:
1. Basic statistics (IV, univariate KS)
2. Binning (chi, quantile, step)
3. WOE encoding
4. End-to-end model metrics (AUC, KS)
"""

import numpy as np  # noqa: F401
import pandas as pd
import toad
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from newt.features.analysis.iv_calculator import calculate_iv
from newt.features.binning.binner import Binner
from newt.metrics.auc import calculate_auc
from newt.metrics.ks import calculate_ks


def load_german_data(path: str) -> pd.DataFrame:
    """Load and preprocess German Credit dataset."""
    columns = [
        "status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status_sex",
        "guarantors",
        "residence_since",
        "property",
        "age",
        "other_installment",
        "housing",
        "existing_credits",
        "job",
        "maintenance_people",
        "telephone",
        "foreign_worker",
        "target",
    ]
    df = pd.read_csv(path, sep=" ", header=None, names=columns)
    # Target: 1=Good, 2=Bad. Map to 0=Good, 1=Bad (default risk modeling)
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df


def get_numeric_features(df: pd.DataFrame) -> list:
    """Get numeric feature columns (excluding target)."""
    return [
        c for c in df.columns if c != "target" and pd.api.types.is_numeric_dtype(df[c])
    ]


def compare_iv(df: pd.DataFrame) -> float:
    """Compare IV calculation between newt and toad."""
    print("\n" + "=" * 60)
    print("IV Comparison")
    print("=" * 60)

    # Toad IV
    toad_iv_df = toad.quality(df, target="target", iv_only=True)
    toad_iv_dict = toad_iv_df.to_dict()

    features = [c for c in df.columns if c != "target"]

    print(f"{'Feature':<25} | {'Toad IV':<10} | {'Newt IV':<10} | {'Diff':<10}")
    print("-" * 65)

    match_count = 0
    total_count = 0

    for feat in features:
        try:
            res = calculate_iv(df, target="target", feature=feat, buckets=10)
            newt_val = res["iv"]
            toad_val = toad_iv_dict.get(feat, np.nan)

            diff = abs(newt_val - toad_val)
            status = "[OK]" if diff < 0.1 else "[FAIL]"
            print(
                f"{feat:<25} | {toad_val:>8.4f}  | {newt_val:>8.4f}  | "
                f"{diff:>8.4f} {status}"
            )

            if diff < 0.1:
                match_count += 1
            total_count += 1
        except Exception as e:
            print(f"{feat:<25} | Error: {e}")

    match_rate = match_count / total_count if total_count > 0 else 0
    print(f"\nMatch Rate (Diff < 0.1): {match_count}/{total_count} = {match_rate:.1%}")
    return match_rate


def compare_univariate_ks(df: pd.DataFrame) -> float:
    """Compare univariate KS calculation between newt and toad."""
    print("\n" + "=" * 60)
    print("Univariate KS Comparison")
    print("=" * 60)

    # Toad quality gives KS as well
    # toad_quality = toad.quality(df, target="target")
    # toad.quality returns DataFrame with columns ['iv', 'gini', 'entropy', 'unique']
    # KS is in toad.metrics.KS for individual feature

    numeric_features = get_numeric_features(df)
    y = df["target"].values

    print(f"{'Feature':<25} | {'Toad KS':<10} | {'Newt KS':<10} | {'Diff':<10}")
    print("-" * 65)

    match_count = 0
    total_count = 0

    for feat in numeric_features:
        try:
            x = df[feat].values

            # Toad KS - uses the feature values directly as scores
            toad_ks = toad.metrics.KS(x, y)

            # Newt KS - same logic
            newt_ks = calculate_ks(y, x)

            diff = abs(newt_ks - toad_ks)
            status = "[OK]" if diff < 0.05 else "[FAIL]"
            print(
                f"{feat:<25} | {toad_ks:>8.4f}  | {newt_ks:>8.4f}  | "
                f"{diff:>8.4f} {status}"
            )

            if diff < 0.05:
                match_count += 1
            total_count += 1
        except Exception as e:
            print(f"{feat:<25} | Error: {e}")

    match_rate = match_count / total_count if total_count > 0 else 0
    print(f"\nMatch Rate (Diff < 0.05): {match_count}/{total_count} = {match_rate:.1%}")
    return match_rate


def compare_chi_binning(df: pd.DataFrame) -> None:
    """Compare Chi-Merge binning between newt and toad."""
    print("\n" + "=" * 60)
    print("Chi-Merge Binning Comparison")
    print("=" * 60)

    numeric_features = get_numeric_features(df)
    n_bins = 5

    for feat in numeric_features[:3]:  # Sample first 3 numeric features
        print(f"\nFeature: {feat}")
        print("-" * 40)

        try:
            # Toad Combiner (ChiMerge)
            c = toad.transform.Combiner()
            c.fit(
                df[[feat, "target"]],
                y="target",
                method="chi",
                min_samples=0.05,
                n_bins=n_bins,
            )
            toad_splits = c.export().get(feat, [])
            print(f"  Toad splits: {toad_splits}")
            print(f"  Toad bins:   {len(toad_splits) + 1}")

            # Newt Binner (ChiMerge)
            binner = Binner()
            binner.fit(
                df[[feat]],
                y=df["target"],
                method="chi",
                n_bins=n_bins,
                min_samples=0.05,
            )
            newt_splits = list(binner.rules_.get(feat, []))
            print(f"  Newt splits: {newt_splits}")
            print(f"  Newt bins:   {len(newt_splits) + 1}")

        except Exception as e:
            print(f"  Error: {e}")


def compare_quantile_binning(df: pd.DataFrame) -> None:
    """Compare equal-frequency (quantile) binning between newt and toad."""
    print("\n" + "=" * 60)
    print("Equal-Frequency (Quantile) Binning Comparison")
    print("=" * 60)

    numeric_features = get_numeric_features(df)
    n_bins = 5

    for feat in numeric_features[:3]:  # Sample first 3 numeric features
        print(f"\nFeature: {feat}")
        print("-" * 40)

        try:
            # Toad Combiner (quantile)
            c = toad.transform.Combiner()
            c.fit(
                df[[feat, "target"]],
                y="target",
                method="quantile",
                n_bins=n_bins,
            )
            toad_splits = c.export().get(feat, [])
            print(f"  Toad splits: {toad_splits}")

            # Newt Binner (quantile)
            binner = Binner()
            binner.fit(
                df[[feat]],
                y=df["target"],
                method="quantile",
                n_bins=n_bins,
            )
            newt_splits = list(binner.rules_.get(feat, []))
            print(f"  Newt splits: {newt_splits}")

        except Exception as e:
            print(f"  Error: {e}")


def compare_step_binning(df: pd.DataFrame) -> None:
    """Compare equal-width (step) binning between newt and toad."""
    print("\n" + "=" * 60)
    print("Equal-Width (Step) Binning Comparison")
    print("=" * 60)

    numeric_features = get_numeric_features(df)
    n_bins = 5

    for feat in numeric_features[:3]:  # Sample first 3 numeric features
        print(f"\nFeature: {feat}")
        print("-" * 40)

        try:
            # Toad Combiner (step)
            c = toad.transform.Combiner()
            c.fit(
                df[[feat, "target"]],
                y="target",
                method="step",
                n_bins=n_bins,
            )
            toad_splits = c.export().get(feat, [])
            print(f"  Toad splits: {toad_splits}")

            # Newt Binner (step)
            binner = Binner()
            binner.fit(
                df[[feat]],
                y=df["target"],
                method="step",
                n_bins=n_bins,
            )
            newt_splits = list(binner.rules_.get(feat, []))
            print(f"  Newt splits: {newt_splits}")

        except Exception as e:
            print(f"  Error: {e}")


def compare_woe(df: pd.DataFrame) -> float:
    """Compare WOE encoding between newt and toad."""
    print("\n" + "=" * 60)
    print("WOE Encoding Comparison")
    print("=" * 60)

    numeric_features = get_numeric_features(df)
    n_bins = 5

    match_count = 0
    total_count = 0

    for feat in numeric_features[:5]:  # Sample first 5 numeric features
        print(f"\nFeature: {feat}")
        print("-" * 40)

        try:
            # Toad: bin then WOE transform
            combiner = toad.transform.Combiner()
            combiner.fit(
                df[[feat, "target"]],
                y="target",
                method="chi",
                n_bins=n_bins,
            )
            binned_toad = combiner.transform(df[[feat]])

            woe_transformer = toad.transform.WOETransformer()
            woe_transformer.fit(binned_toad, df["target"])
            toad_woe = woe_transformer.transform(binned_toad)[feat]

            # Newt: bin and WOE transform
            binner = Binner()
            binner.fit(
                df[[feat]],
                y=df["target"],
                method="chi",
                n_bins=n_bins,
            )
            newt_woe = binner.woe_transform(df[[feat]])[feat]

            # Compare WOE values (sample mean comparison)
            toad_mean = toad_woe.mean()
            newt_mean = newt_woe.mean()
            diff = abs(toad_mean - newt_mean)

            # Correlation between WOE values
            corr = np.corrcoef(toad_woe.values, newt_woe.values)[0, 1]

            status = "[OK]" if corr > 0.9 else "[FAIL]"
            print(f"  Toad WOE mean:  {toad_mean:.4f}")
            print(f"  Newt WOE mean:  {newt_mean:.4f}")
            print(f"  Mean diff:      {diff:.4f}")
            print(f"  Correlation:    {corr:.4f} {status}")

            if corr > 0.9:
                match_count += 1
            total_count += 1

        except Exception as e:
            print(f"  Error: {e}")

    match_rate = match_count / total_count if total_count > 0 else 0
    print(f"\nMatch Rate (Corr > 0.9): {match_count}/{total_count} = {match_rate:.1%}")
    return match_rate


def compare_model_metrics(df: pd.DataFrame) -> dict:
    """Compare end-to-end model AUC and KS between newt and toad pipelines."""
    print("\n" + "=" * 60)
    print("End-to-End Model Metrics Comparison")
    print("=" * 60)

    numeric_features = get_numeric_features(df)

    # Split data
    X = df[numeric_features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # === Toad Pipeline ===
    print("\n--- Toad Pipeline ---")
    try:
        # Combine train data with target for toad
        train_df = X_train.copy()
        train_df["target"] = y_train.values

        # Binning
        combiner = toad.transform.Combiner()
        combiner.fit(train_df, y="target", method="chi", n_bins=5)
        train_binned = combiner.transform(X_train)
        test_binned = combiner.transform(X_test)

        # WOE Transform
        woe_transformer = toad.transform.WOETransformer()
        woe_transformer.fit(train_binned, y_train)
        train_woe_toad = woe_transformer.transform(train_binned)
        test_woe_toad = woe_transformer.transform(test_binned)

        # Fill any NaN with 0
        train_woe_toad = train_woe_toad.fillna(0)
        test_woe_toad = test_woe_toad.fillna(0)

        # Train model
        model_toad = LogisticRegression(max_iter=1000, random_state=42)
        model_toad.fit(train_woe_toad, y_train)

        # Predict
        y_prob_toad = model_toad.predict_proba(test_woe_toad)[:, 1]

        # Metrics
        toad_auc = toad.metrics.AUC(y_prob_toad, y_test)
        toad_ks = toad.metrics.KS(y_prob_toad, y_test)

        print(f"  AUC: {toad_auc:.4f}")
        print(f"  KS:  {toad_ks:.4f}")

        results["toad_auc"] = toad_auc
        results["toad_ks"] = toad_ks

    except Exception as e:
        print(f"  Error: {e}")
        results["toad_auc"] = np.nan
        results["toad_ks"] = np.nan

    # === Newt Pipeline ===
    print("\n--- Newt Pipeline ---")
    try:
        # Combine train data with target for fit
        train_df = X_train.copy()
        train_df["target"] = y_train.values

        # Binning and WOE Transform
        binner = Binner()
        binner.fit(
            X_train,
            y=y_train,
            method="chi",
            n_bins=5,
        )
        train_woe_newt = binner.woe_transform(X_train)
        test_woe_newt = binner.woe_transform(X_test)

        # Fill any NaN with 0
        train_woe_newt = train_woe_newt.fillna(0)
        test_woe_newt = test_woe_newt.fillna(0)

        # Train model
        model_newt = LogisticRegression(max_iter=1000, random_state=42)
        model_newt.fit(train_woe_newt, y_train)

        # Predict
        y_prob_newt = model_newt.predict_proba(test_woe_newt)[:, 1]

        # Metrics
        newt_auc = calculate_auc(y_test, y_prob_newt)
        newt_ks = calculate_ks(y_test, y_prob_newt)

        print(f"  AUC: {newt_auc:.4f}")
        print(f"  KS:  {newt_ks:.4f}")

        results["newt_auc"] = newt_auc
        results["newt_ks"] = newt_ks

    except Exception as e:
        print(f"  Error: {e}")
        results["newt_auc"] = np.nan
        results["newt_ks"] = np.nan

    # === Comparison ===
    print("\n--- Comparison ---")
    auc_diff = abs(results.get("toad_auc", 0) - results.get("newt_auc", 0))
    ks_diff = abs(results.get("toad_ks", 0) - results.get("newt_ks", 0))

    auc_status = "[OK]" if auc_diff < 0.02 else "[FAIL]"
    ks_status = "[OK]" if ks_diff < 0.03 else "[FAIL]"

    print(f"  AUC diff: {auc_diff:.4f} {auc_status} (threshold: 0.02)")
    print(f"  KS diff:  {ks_diff:.4f} {ks_status} (threshold: 0.03)")

    results["auc_diff"] = auc_diff
    results["ks_diff"] = ks_diff

    return results


def run():
    """Run all cross-validation comparisons."""
    print("=" * 60)
    print("NEWT vs TOAD Cross-Validation")
    print("=" * 60)

    path = "d:/Project/newt/examples/data/statlog+german+credit+data/german.data"
    df = load_german_data(path)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    # Run all comparisons
    iv_score = compare_iv(df)
    ks_score = compare_univariate_ks(df)
    compare_chi_binning(df)
    compare_quantile_binning(df)
    compare_step_binning(df)
    woe_score = compare_woe(df)
    model_results = compare_model_metrics(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"IV Match Rate:           {iv_score:.1%}")
    print(f"Univariate KS Match:     {ks_score:.1%}")
    print(f"WOE Correlation Match:   {woe_score:.1%}")
    print(f"Model AUC Diff:          {model_results.get('auc_diff', np.nan):.4f}")
    print(f"Model KS Diff:           {model_results.get('ks_diff', np.nan):.4f}")

    # Overall verdict
    all_pass = (
        iv_score > 0.7
        and ks_score > 0.7
        and woe_score > 0.7
        and model_results.get("auc_diff", 1) < 0.05
        and model_results.get("ks_diff", 1) < 0.05
    )

    if all_pass:
        print("\n[OK] SUCCESS: High agreement with Toad library.")
    else:
        print("\n[WARNING] Some metrics show significant differences.")


if __name__ == "__main__":
    run()
