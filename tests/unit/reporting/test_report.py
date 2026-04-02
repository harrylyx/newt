from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")
import pandas as pd

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
    tag_block = overview_sheet.get_block("按tag模型效果")
    month_block = overview_sheet.get_block("按月模型效果")
    labels = set(tag_block.data["样本标签"]) | set(month_block.data["样本标签"])

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


def test_report_sorts_month_sections_and_splits_model_binning_tables(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "sorted_sections.xlsx"
    shuffled = report_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    report = Report(
        data=shuffled,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        var_list=["profile_income"],
        sheet_list=["总览", "变量分析", "模型表现"],
        report_out_path=str(output_path),
    )

    report.generate()

    overview_sheet = report.result_.get_sheet("总览")
    tag_metrics = overview_sheet.get_block("按tag模型效果").data
    monthly_metrics = overview_sheet.get_block("按月模型效果").data
    assert set(tag_metrics["观察点月"]) == {""}
    assert set(monthly_metrics["样本集"]) == {""}
    train_months = monthly_metrics.loc[
        monthly_metrics["样本标签"] == "label_main",
        "观察点月",
    ].tolist()
    assert train_months == ["202401", "202402", "202403", "202404"]

    variable_sheet = report.result_.get_sheet("变量分析")
    feature_month_blocks = [
        block for block in variable_sheet.blocks if block.title.endswith("按月效果")
    ]
    assert feature_month_blocks
    assert feature_month_blocks[0].data["month"].tolist() == sorted(
        feature_month_blocks[0].data["month"].tolist()
    )

    performance_sheet = report.result_.get_sheet("模型表现")
    block_titles = [block.title for block in performance_sheet.blocks]
    assert "3.2 按tag模型效果" in block_titles
    assert "3.2 按月模型效果" in block_titles
    split_index = block_titles.index("3.3 模型分箱表现")
    bin_blocks = performance_sheet.blocks[split_index + 1 :]

    assert [block.title for block in bin_blocks[:4]] == ["train", "test", "oot", "oos"]
    assert [block.title for block in bin_blocks[4:8]] == [
        "202401",
        "202402",
        "202403",
        "202404",
    ]
    assert all(block.blank_rows_after == 3 for block in bin_blocks)
    assert "样本集" not in bin_blocks[0].data.columns
    assert "观察点月" not in bin_blocks[0].data.columns


def test_report_formats_iv_with_four_decimals_and_places_chart_after_monthly_table(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "variable_analysis.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        var_list=["profile_income"],
        sheet_list=["变量分析"],
        report_out_path=str(output_path),
    )

    generated = report.generate()
    workbook = openpyxl.load_workbook(generated)
    worksheet = workbook["变量分析"]
    analysis_sheet = report.result_.get_sheet("变量分析")
    analysis_block = analysis_sheet.get_block("2. 变量分析")

    analysis_header_row = next(
        row[0].row
        for row in worksheet.iter_rows()
        if any(cell.value == "iv_train" for cell in row)
    )
    iv_train_column = next(
        cell.column
        for cell in worksheet[analysis_header_row]
        if cell.value == "iv_train"
    )
    assert (
        worksheet.cell(analysis_header_row + 1, iv_train_column).number_format
        == "0.0000"
    )
    missing_train_column = next(
        cell.column
        for cell in worksheet[analysis_header_row]
        if cell.value == "缺失率_train"
    )
    gain_per_column = next(
        cell.column
        for cell in worksheet[analysis_header_row]
        if cell.value == "gain_per"
    )
    weight_per_column = next(
        cell.column
        for cell in worksheet[analysis_header_row]
        if cell.value == "weight_per"
    )
    assert (
        worksheet.cell(analysis_header_row + 1, missing_train_column).number_format
        == "0.00%"
    )
    assert (
        worksheet.cell(analysis_header_row + 1, gain_per_column).number_format
        == "0.00%"
    )
    assert (
        worksheet.cell(analysis_header_row + 1, weight_per_column).number_format
        == "0.00%"
    )
    assert analysis_block.data["gain"].tolist() == sorted(
        analysis_block.data["gain"].tolist(),
        reverse=True,
    )
    assert analysis_block.data["序号"].tolist() == list(
        range(1, len(analysis_block.data) + 1)
    )

    bin_header_row = next(
        row[0].row
        for row in worksheet.iter_rows()
        if any(cell.value == "iv" for cell in row)
        and any(cell.value == "woe" for cell in row)
    )
    bin_iv_column = next(
        cell.column for cell in worksheet[bin_header_row] if cell.value == "iv"
    )
    assert worksheet.cell(bin_header_row + 1, bin_iv_column).number_format == "0.0000"
    first_bin_block = next(
        block for block in analysis_sheet.blocks if block.title.endswith("分箱表")
    )
    bad_rates = first_bin_block.data["bad_rate"].dropna().tolist()
    assert bad_rates == sorted(bad_rates, reverse=True)

    variable_sheet = report.result_.get_sheet("变量分析")
    first_bin_title = next(
        block.title for block in variable_sheet.blocks if block.title.endswith("分箱表")
    )
    first_monthly_title = first_bin_title.replace("分箱表", "按月效果")
    monthly_title_row = next(
        row[0].row
        for row in worksheet.iter_rows()
        if any(cell.value == first_monthly_title for cell in row)
    )

    assert worksheet._charts
    assert worksheet._charts[0].anchor._from.row + 1 > monthly_title_row


def test_report_reorients_score_like_columns_and_records_direction(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "direction.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score"],
        sheet_list=["总览"],
        report_out_path=str(output_path),
    )

    report.generate()

    score_directions = report.result_.metadata["score_directions"]
    score_record = next(
        item for item in score_directions if item["分数字段"] == "score"
    )
    assert score_record["原始方向"] == "高分代表低风险"
    assert score_record["报表计算方向"] == "已转为坏向分"
    assert score_record["原始AUC"] < 0.5

    overview_sheet = report.result_.get_sheet("总览")
    direction_block = overview_sheet.get_block("分数字段方向说明")
    assert {"分数字段", "原始方向", "报表计算方向", "判断依据"}.issubset(
        direction_block.data.columns
    )

    paired_block = overview_sheet.get_block("按tag新老模型对比(score)").data
    score_auc = (
        paired_block.loc[paired_block["模型"] == "score", "AUC"].dropna().tolist()
    )
    assert score_auc
    assert min(score_auc) > 0.5


def test_report_pairs_new_old_model_comparisons_by_group(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "paired_compare.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a", "score"],
        sheet_list=["总览"],
        report_out_path=str(output_path),
    )

    report.generate()

    overview_sheet = report.result_.get_sheet("总览")
    tag_compare_a = overview_sheet.get_block("按tag新老模型对比(score_old_a)").data
    tag_compare_score = overview_sheet.get_block("按tag新老模型对比(score)").data
    month_compare_score = overview_sheet.get_block("按月新老模型对比(score)").data

    assert tag_compare_a["模型"].tolist() == ["score_new", "score_old_a"] * 4
    assert tag_compare_score["模型"].tolist() == ["score_new", "score"] * 4
    assert month_compare_score["模型"].tolist() == ["score_new", "score"] * 4


def test_report_model_binning_tables_sort_by_bad_rate_desc(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "model_bins.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["模型表现"],
        report_out_path=str(output_path),
    )

    report.generate()

    performance_sheet = report.result_.get_sheet("模型表现")
    train_block = performance_sheet.get_block("train")
    month_block = performance_sheet.get_block("202401")
    for block in [train_block, month_block]:
        bad_rates = block.data["bad_rate"].dropna().tolist()
        assert bad_rates == sorted(bad_rates, reverse=True)
