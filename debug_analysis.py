
import pandas as pd
import numpy as np
import traceback
from src.credit_risk.features.analysis.correlation import calculate_correlation_matrix, get_high_correlation_pairs
from src.credit_risk.features.analysis.iv_calculator import calculate_iv
from src.credit_risk.features.analysis.woe_calculator import calculate_woe_mapping, apply_woe_transform

def debug_analysis():
    print("--- Debugging Feature Analysis ---")
    try:
        np.random.seed(42)
        n = 200
        x1 = np.random.rand(n)
        x2 = x1 * 0.9 + np.random.rand(n) * 0.1
        x3 = np.random.rand(n)
        prob = 1 / (1 + np.exp(-(x1 - 0.5) * 5))
        target = (np.random.rand(n) < prob).astype(int)
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'target': target})
        
        print("Testing Correlation...")
        corr_matrix = calculate_correlation_matrix(df, method='pearson')
        pairs = get_high_correlation_pairs(corr_matrix, threshold=0.8)
        print(f"Correlation pairs: {len(pairs)} found.")
        assert len(pairs) >= 1, "Correlation failed"
        
        print("Testing IV...")
        res = calculate_iv(df, target='target', feature='x1', buckets=5)
        print(f"IV: {res['iv']}")
        assert res['iv'] > 0.02, "IV too low"
        
        print("Testing WoE...")
        df['x1_bin'] = pd.qcut(df['x1'], q=5, duplicates='drop')
        woe_map = calculate_woe_mapping(df, target='target', feature='x1_bin', bins=None)
        print(f"WoE Map keys: {list(woe_map.keys())}")
        
        df_transformed = apply_woe_transform(df, feature='x1_bin', woe_map=woe_map)
        print("Transformation done.")
        print(df_transformed[['x1_bin', 'x1_bin_woe']].head())
        
        assert 'x1_bin_woe' in df_transformed.columns
        assert df_transformed['x1_bin_woe'].dtype == float or np.issubdtype(df_transformed['x1_bin_woe'].dtype, np.number)
        
        
        print("Testing WOEEncoder Class...")
        from src.credit_risk.features.analysis.woe_calculator import WOEEncoder
        encoder = WOEEncoder(buckets=5)
        encoder.fit(df['x1'], df['target'])
        
        print(f"Encoder fit complete. IV: {encoder.iv_}")
        assert encoder.iv_ > 0.02, "Class IV too low"
        assert not encoder.woe_map_ == {}, "Class Map empty"
        assert not encoder.summary_.empty, "Class Summary empty"
        
        transformed_class = encoder.transform(df['x1'])
        print(f"Class transform complete. Shape: {transformed_class.shape}")
        
        assert len(transformed_class) == len(df), "Transform length mismatch"
        assert transformed_class.dtype == float or np.issubdtype(transformed_class.dtype, np.number), "Transform dtype mismatch"
        
        print("ALL CHECKS PASSED")
        
    except Exception:
        with open("error.log", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()

if __name__ == "__main__":
    debug_analysis()
