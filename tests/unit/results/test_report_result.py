import pandas as pd

from newt.results import ModelReportResult, ReportBlock, ReportSheet


def test_model_report_result_preserves_sheet_order_and_blocks():
    overview = ReportSheet(
        name="总览",
        blocks=[
            ReportBlock(
                title="按月模型效果",
                data=pd.DataFrame({"样本集": ["train"], "AUC": [0.81]}),
            )
        ],
    )
    performance = ReportSheet(name="模型表现")
    result = ModelReportResult(
        sheets={"总览": overview, "模型表现": performance},
        metadata={"score_col": "score_new"},
    )

    assert result.sheet_names == ["总览", "模型表现"]
    assert result.get_sheet("总览").blocks[0].title == "按月模型效果"
    assert result.metadata["score_col"] == "score_new"
