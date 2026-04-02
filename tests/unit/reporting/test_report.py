from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from newt.reporting import Report


def test_report_generate_creates_selected_sheets(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "model_report.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main", "label_alt"],
        score_list=["score_old_a", "score_old_b"],
        dim_list=["channel_dim"],
        var_list=["profile_income", "profile_age"],
        sheet_list=[1, "模型表现"],
        report_out_path=str(output_path),
    )

    generated = report.generate()

    assert generated == str(output_path)
    assert Path(generated).exists()
    assert report.result_.sheet_names == ["总览", "模型表现"]

    workbook = openpyxl.load_workbook(generated)
    assert workbook.sheetnames == ["总览", "模型表现"]
    assert workbook["总览"]["A1"].value == "一、目标与设计方案"
    assert workbook["模型表现"]["A1"].value == "3.1 建模方法选择"


def test_report_overview_metrics_expand_all_labels(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "overview_only.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main", "label_alt"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        var_list=["profile_income"],
        sheet_list=["总览"],
        report_out_path=str(output_path),
    )

    report.generate()

    overview_sheet = report.result_.get_sheet("总览")
    metric_block = overview_sheet.get_block("按月模型效果")
    labels = set(metric_block.data["样本标签"])

    assert labels == {"label_main", "label_alt"}


def test_report_generate_all_sheets_handles_interval_cells(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "all_sheets.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        var_list=["profile_income", "profile_age"],
        report_out_path=str(output_path),
    )

    generated = report.generate()
    workbook = openpyxl.load_workbook(generated)

    assert workbook.sheetnames == ["总览", "模型设计", "变量分析", "模型表现"]
