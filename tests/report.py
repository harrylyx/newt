import pickle

import pandas as pd

from newt import Report

df = pd.read_parquet("./examples/data/test_data/all_data.pq")
model = pickle.load(open("./examples/data/test_data/all_model.pkl", "rb"))

report = Report(
    data=df,
    model=model,
    tag="tag",
    score_col="score",
    date_col="listinginfo",
    label_list=["target"],
    score_list=[],
    dim_list=[],
    var_list=[],
    feature_path="",
    report_out_path="./examples/data/test_data/model_report.xlsx",
)

report.generate()
