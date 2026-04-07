import pickle
import time

import pandas as pd

from newt import Report

df = pd.read_parquet("./examples/data/test_data/all_data.pq").sample(
    2000000, random_state=123
)
model = pickle.load(open("./examples/data/test_data/xgb_model.pkl", "rb"))


report = Report(
    data=df,
    model=model,
    tag="tag",
    score_col="xgb_score",
    date_col="listinginfo",
    label_list=["target"],
    score_list=["score"],
    dim_list=["userinfo_9", "education_info2"],
    var_list=[
        "thirdparty_info_period1_6",
        "thirdparty_info_period2_6",
        "thirdparty_info_period4_10",
        "thirdparty_info_period6_2",
        "thirdparty_info_period4_9",
    ],
    feature_path="",
    report_out_path="./examples/data/test_data/model_report.xlsx",
)

s_time = time.time()
report.generate()
print(f"use time: {time.time() - s_time} s")
