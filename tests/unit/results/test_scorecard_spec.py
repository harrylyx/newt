import numpy as np
import pandas as pd

from newt.results import BinningRuleSpec


def test_binning_rule_spec_bins_missing_values_separately():
    rule = BinningRuleSpec(feature="score", splits=[0.5], missing_label="Missing")
    values = pd.Series([0.1, 0.9, np.nan], name="score")

    binned = rule.bin_series(values)

    assert binned.iloc[2] == "Missing"
    assert binned.iloc[0] != binned.iloc[1]
