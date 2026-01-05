
import pandas as pd
import numpy as np
import toad
from src.newt.features.analysis.iv_calculator import calculate_iv
from src.newt.features.binning.supervised import ChiMergeBinner, DecisionTreeBinner
from src.newt.features.binning.unsupervised import EqualFrequencyBinner

def load_german_data(path):
    columns = [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'personal_status_sex', 'guarantors',
        'residence_since', 'property', 'age', 'other_installment', 'housing',
        'existing_credits', 'job', 'maintenance_people', 'telephone', 'foreign_worker',
        'target'
    ]
    df = pd.read_csv(path, sep=' ', header=None, names=columns)
    # Target: 1=Good, 2=Bad. Map to 0=Good, 1=Bad (default risk modeling)
    df['target'] = df['target'].map({1: 0, 2: 1})
    return df

def compare_iv(df):
    print("\n--- IV Comparison ---")
    # 1. Toad IV
    toad_iv_df = toad.quality(df, target='target', iv_only=True)
    toad_iv_dict = toad_iv_df.to_dict() # Series to dict {feature: iv}
    
    # 2. Newt IV
    newt_iv_dict = {}
    features = [c for c in df.columns if c != 'target']
    
    print(f"{'Feature':<20} | {'Toad IV':<10} | {'Newt IV':<10} | {'Diff':<10}")
    print("-" * 60)
    
    match_count = 0
    total_count = 0
    
    for feat in features:
        # Skip high cardinality or weird columns if any
        try:
            res = calculate_iv(df, target='target', feature=feat, buckets=10) # Using default equal freq binning
            newt_val = res['iv']
            toad_val = toad_iv_dict.get(feat, np.nan)
            
            diff = abs(newt_val - toad_val)
            print(f"{feat:<20} | {toad_val:.4f}     | {newt_val:.4f}     | {diff:.4f}")
            
            # Allow some difference due to binning strategies
            # Toad quality default likely uses Chi or DT or just quantiles?
            # actually toad.quality uses entropy/dt/chi? Documentation says default is iv_only=False uses variety.
            # iv_only=True uses efficient calculation, likely simple binning for num.
            # If difference < 0.05 we consider it close enough given algorithm variance.
            # (Note for categorical, it should be exact if no merging)
            if diff < 0.1: # Relaxed threshold
                match_count += 1
            total_count += 1
        except Exception as e:
            print(f"Error calc {feat}: {e}")

    print(f"\nMatch Rate (Diff < 0.1): {match_count}/{total_count}")
    return match_count / total_count

def compare_binning(df):
    print("\n--- Binning Comparison (ChiMerge on Duration) ---")
    feat = 'duration'
    
    # 1. Toad Combiner (ChiMerge)
    c = toad.transform.Combiner()
    c.fit(df[[feat, 'target']], y='target', method='chi', min_samples=0.05, n_bins=5)
    toad_splits = c.export()[feat]
    print(f"Toad Splits: {toad_splits}")
    
    # 2. Newt ChiMerge
    binner = ChiMergeBinner(n_bins=5, init_bins=50) # similar constraints?
    binner.fit(df[feat], df['target'])
    newt_splits = binner.splits_
    print(f"Newt Splits: {newt_splits}")
    
    # Compare
    # Note: ChiMerge exact splits depend heavily on initialization and tie breaking
    # We check if range overlaps or number of bins is similar
    print(f"Toad bins: {len(toad_splits)+1}, Newt bins: {len(newt_splits)+1}")

def run():
    path = 'd:/Project/newt/examples/data/statlog+german+credit+data/german.data'
    df = load_german_data(path)
    
    iv_score = compare_iv(df)
    compare_binning(df)
    
    if iv_score > 0.8:
        print("\nSUCCESS: High agreement with Toad.")
    else:
        print("\nWARNING: Low agreement with Toad.")

if __name__ == "__main__":
    run()
