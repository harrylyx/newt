from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import newt.reporting.report as report_module
from newt.reporting import Report

openpyxl = pytest.importorskip("openpyxl")


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
    assert report.result_.sheet_names == ["总览", "3.模型表现"]

    workbook = openpyxl.load_workbook(generated)
    assert workbook.sheetnames == ["总览", "3.模型表现"]
    assert workbook["总览"]["A1"].value == "一、目标与设计方案"
    assert workbook["3.模型表现"]["A1"].value == "一、建模方法选择"


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

    assert workbook.sheetnames == [
        "总览",
        "1.模型设计",
        "2.变量分析",
        "3.模型表现",
        "附1 分维度对比",
        "附2 新老模型对比",
        "附3 画像变量",
    ]


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
    assert set(tag_metrics["观察点月"]) == {"20240131-20240430"}
    assert set(monthly_metrics["样本集"]) == {"train,test,oot,oos"}
    train_months = monthly_metrics.loc[
        monthly_metrics["样本标签"] == "label_main",
        "观察点月",
    ].tolist()
    assert train_months == ["202401", "202402", "202403", "202404"]

    variable_sheet = report.result_.get_sheet("2.变量分析")
    feature_month_blocks = [
        block for block in variable_sheet.blocks if block.title.endswith("按月效果")
    ]
    assert feature_month_blocks
    assert feature_month_blocks[0].data["month"].tolist() == sorted(
        feature_month_blocks[0].data["month"].tolist()
    )

    performance_sheet = report.result_.get_sheet("3.模型表现")
    perf_tag_metrics = performance_sheet.get_block("二、按tag模型效果").data
    perf_month_metrics = performance_sheet.get_block("三、按月模型效果").data
    assert set(perf_tag_metrics["观察点月"]) == {"20240131-20240430"}
    assert set(perf_month_metrics["样本集"]) == {"train,test,oot,oos"}
    block_titles = [block.title for block in performance_sheet.blocks]
    assert "二、按tag模型效果" in block_titles
    assert "三、按月模型效果" in block_titles
    split_index = block_titles.index("四、模型分箱表现")
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
    worksheet = workbook["2.变量分析"]
    analysis_sheet = report.result_.get_sheet("2.变量分析")
    analysis_block = analysis_sheet.get_block("二、变量分析")

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
        block for block in analysis_sheet.blocks if block.title.endswith("训练分箱表")
    )
    min_values = first_bin_block.data["min"].dropna().tolist()
    assert min_values == sorted(min_values)

    variable_sheet = report.result_.get_sheet("2.变量分析")
    first_train_bin_title = next(
        block.title
        for block in variable_sheet.blocks
        if block.title.endswith("训练分箱表")
    )
    first_oot_bin_title = first_train_bin_title.replace("训练分箱表", "OOT分箱表")
    first_monthly_title = first_train_bin_title.replace("训练分箱表", "按月效果")
    monthly_title_row = next(
        row[0].row
        for row in worksheet.iter_rows()
        if any(cell.value == first_monthly_title for cell in row)
    )

    chart_blocks = [block for block in variable_sheet.blocks if block.chart is not None]
    chart_sources = [block.chart.source_block_title for block in chart_blocks]
    assert first_train_bin_title in chart_sources
    assert first_oot_bin_title in chart_sources
    assert worksheet._charts
    assert worksheet._charts[0].anchor._from.row + 1 > monthly_title_row


def test_report_records_auc_only_direction_for_score_like_columns(
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
    assert score_record["报表计算方向"] == "AUC和Lift按风险方向，其余指标保持原始分数"
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
    score_lift = (
        paired_block.loc[paired_block["模型"] == "score", "10%lift"].dropna().tolist()
    )
    assert score_lift
    assert min(score_lift) > 0.0


def test_report_keeps_raw_score_values_in_model_binning(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "raw_score_bins.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["模型表现"],
        report_out_path=str(output_path),
    )

    report.generate()

    performance_sheet = report.result_.get_sheet("3.模型表现")
    train_block = performance_sheet.get_block("train").data
    finite_min = (
        train_block["min"].replace([-float("inf"), float("inf")], pd.NA).dropna()
    )
    finite_max = (
        train_block["max"].replace([-float("inf"), float("inf")], pd.NA).dropna()
    )

    assert (finite_min >= 0).all()
    assert (finite_max >= 0).all()


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
    assert set(tag_compare_a["观察点月"]) == {"20240131-20240430"}
    assert set(tag_compare_score["观察点月"]) == {"20240131-20240430"}
    assert set(month_compare_score["样本集"]) == {"train,test,oot,oos"}


def test_report_overview_corr_precision_and_portrait_wide_layout(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "overview_corr_portrait.xlsx"
    shifted = report_frame.copy()
    # Force clearly different bin boundaries between models.
    shifted["score_old_a"] = shifted["score_old_a"] * 100 + 100

    report = Report(
        data=shifted,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        var_list=["profile_income", "profile_age"],
        sheet_list=["总览"],
        report_out_path=str(output_path),
    )

    report.generate()

    overview_sheet = report.result_.get_sheet("总览")
    corr = overview_sheet.get_block("OOT相关性矩阵").data
    numeric_corr = corr.drop(columns=["模型"]).apply(pd.to_numeric, errors="coerce")
    assert np.allclose(
        numeric_corr.to_numpy(dtype=float),
        np.round(numeric_corr.to_numpy(dtype=float), 4),
        equal_nan=True,
    )

    portrait_income = overview_sheet.get_block("1.profile_income").data
    portrait_age = overview_sheet.get_block("2.profile_age").data
    expected_columns = ["模型", *[str(i) for i in range(1, 11)], "Missing"]
    assert list(portrait_income.columns) == expected_columns
    assert list(portrait_age.columns) == expected_columns
    assert set(portrait_income["模型"]) == {"score_new", "score_old_a"}
    assert set(portrait_age["模型"]) == {"score_new", "score_old_a"}


def test_report_model_design_distribution_layout_and_tag_order(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "model_design.xlsx"

    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["模型设计"],
        report_out_path=str(output_path),
    )

    report.generate()

    design_sheet = report.result_.get_sheet("1.模型设计")
    raw_distribution = design_sheet.get_block("原始样本分布表").data
    dev_distribution = design_sheet.get_block("开发样本分布表").data
    model_distribution = design_sheet.get_block("建模样本分布情况表").data

    assert "样本集" not in raw_distribution.columns
    assert "样本集" not in dev_distribution.columns
    assert raw_distribution["月"].tolist() == ["202401", "202402", "202403", "202404"]
    assert dev_distribution["月"].tolist() == ["202401", "202402", "202403", "202404"]
    assert model_distribution["样本集"].tolist() == ["train", "test", "oot", "oos"]


def test_report_model_binning_tables_sort_by_bin_order(
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

    performance_sheet = report.result_.get_sheet("3.模型表现")
    train_block = performance_sheet.get_block("train")
    month_block = performance_sheet.get_block("202401")
    for block in [train_block, month_block]:
        min_values = block.data["min"].dropna().tolist()
    assert min_values == sorted(min_values)


def test_report_parallel_sheets_preserve_order_and_record_runtime_options(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "parallel_report.xlsx"
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
        parallel_sheets=True,
        max_workers=2,
        engine="python",
        memory_mode="compact",
        report_out_path=str(output_path),
    )

    report.generate()

    assert report.result_.sheet_names == [
        "总览",
        "1.模型设计",
        "2.变量分析",
        "3.模型表现",
        "附1 分维度对比",
        "附2 新老模型对比",
        "附3 画像变量",
    ]
    options = report.result_.metadata["report_compute_options"]
    assert options == {
        "engine": "python",
        "max_workers": 2,
        "parallel_sheets": True,
        "memory_mode": "compact",
        "metrics_mode": "exact",
    }
    assert report.result_.metadata["report_compute_top_timings"]


@pytest.mark.parametrize(
    ("dim_list", "score_list", "var_list", "expected_appendix_names"),
    [
        ([], [], [], []),
        (["channel_dim"], [], [], ["附1 分维度对比"]),
        ([], ["score_old_a"], [], ["附1 新老模型对比"]),
        ([], [], ["profile_income"], ["附1 画像变量"]),
        (
            ["channel_dim"],
            ["score_old_a"],
            ["profile_income"],
            ["附1 分维度对比", "附2 新老模型对比", "附3 画像变量"],
        ),
    ],
)
def test_report_appendix_sheet_numbering(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
    dim_list,
    score_list,
    var_list,
    expected_appendix_names,
):
    output_path = tmp_path / "appendix_numbering.xlsx"
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        dim_list=dim_list,
        score_list=score_list,
        var_list=var_list,
        report_out_path=str(output_path),
    )

    report.generate()

    expected_names = [
        "总览",
        "1.模型设计",
        "2.变量分析",
        "3.模型表现",
        *expected_appendix_names,
    ]
    assert report.result_.sheet_names == expected_names


def test_report_sheet_selector_rejects_numbered_sheet_name(
    report_frame,
    fake_lightgbm_model,
):
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["1.模型设计"],
    )

    with pytest.raises(ValueError, match="Unknown sheet name"):
        report.generate()


def test_report_overview_reuses_child_sheet_blocks(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
    monkeypatch,
):
    from newt.reporting import tables

    call_count = {"value": 0}
    original = tables._build_split_metrics_tables

    def _counted(*args, **kwargs):
        call_count["value"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tables, "_build_split_metrics_tables", _counted)

    output_path = tmp_path / "overview_reuse.xlsx"
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
        sheet_list=["总览", "模型表现", "分维度对比", "新老模型对比", "画像变量"],
        report_out_path=str(output_path),
    )

    report.generate()

    overview = report.result_.get_sheet("总览")
    performance = report.result_.get_sheet("3.模型表现")
    dimensional = report.result_.get_sheet("附1 分维度对比")
    comparison = report.result_.get_sheet("附2 新老模型对比")
    portrait = report.result_.get_sheet("附3 画像变量")

    pd.testing.assert_frame_equal(
        overview.get_block("按tag模型效果").data.reset_index(drop=True),
        performance.get_block("二、按tag模型效果").data.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        overview.get_block("按月模型效果").data.reset_index(drop=True),
        performance.get_block("三、按月模型效果").data.reset_index(drop=True),
    )
    dimensional_titles = [
        block.title
        for block in dimensional.blocks
        if block.title.endswith("channel_dim")
    ]
    assert dimensional_titles == ["1.channel_dim"]
    for title in dimensional_titles:
        pd.testing.assert_frame_equal(
            overview.get_block(title).data.reset_index(drop=True),
            dimensional.get_block(title).data.reset_index(drop=True),
        )
    pd.testing.assert_frame_equal(
        overview.get_block("按tag新老模型对比(score_old_a)").data.reset_index(
            drop=True
        ),
        comparison.get_block("按tag新老模型对比(score_old_a)").data.reset_index(
            drop=True
        ),
    )
    pd.testing.assert_frame_equal(
        overview.get_block("按月新老模型对比(score_old_a)").data.reset_index(drop=True),
        comparison.get_block("按月新老模型对比(score_old_a)").data.reset_index(
            drop=True
        ),
    )
    pd.testing.assert_frame_equal(
        overview.get_block("OOT相关性矩阵").data.reset_index(drop=True),
        comparison.get_block("OOT相关性矩阵").data.reset_index(drop=True),
    )
    portrait_titles = [
        block.title for block in portrait.blocks if block.title.startswith("1.")
    ]
    assert portrait_titles == ["1.profile_income"]
    for title in portrait_titles:
        pd.testing.assert_frame_equal(
            overview.get_block(title).data.reset_index(drop=True),
            portrait.get_block(title).data.reset_index(drop=True),
        )
    assert call_count["value"] == 1


def test_report_feature_dictionary_headers_and_title_right_text(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    feature_df = pd.DataFrame(
        [
            {
                "英文名": "feature_a",
                "中文名": "特征A中文",
                "来源": "thirdparty",
                "指标表英文名": "metric_feature_a",
            },
            {
                "英文名": "feature_b",
                "中文名": "特征B中文",
                "来源": "thirdparty",
                "指标表英文名": "metric_feature_b",
            },
            {
                "英文名": "profile_income",
                "中文名": "收入",
                "来源": "profile",
                "指标表英文名": "metric_profile_income",
            },
        ]
    )

    output_path = tmp_path / "feature_dict_report.xlsx"
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        var_list=["profile_income"],
        feature_df=feature_df,
        sheet_list=["变量分析", "画像变量"],
        report_out_path=str(output_path),
    )

    generated = report.generate()

    variable_sheet = report.result_.get_sheet("2.变量分析")
    top_block = variable_sheet.get_block("1.feature_a 特征A中文")
    assert top_block.title == "1.feature_a 特征A中文"
    feature_table = variable_sheet.get_block("二、变量分析").data.set_index("vars")
    assert feature_table.loc["feature_a", "指标表英文名"] == "metric_feature_a"
    assert feature_table.loc["feature_b", "指标表英文名"] == "metric_feature_b"

    portrait_sheet = report.result_.get_sheet("附1 画像变量")
    portrait_block = portrait_sheet.get_block("1.profile_income 收入")
    assert portrait_block.title == "1.profile_income 收入"

    workbook = openpyxl.load_workbook(generated)
    variable_ws = workbook["2.变量分析"]
    variable_row = next(
        row
        for row in variable_ws.iter_rows()
        if any(cell.value == "1.feature_a 特征A中文" for cell in row)
    )
    assert any(cell.value == "1.feature_a 特征A中文" for cell in variable_row)

    portrait_ws = workbook["附1 画像变量"]
    portrait_row = next(
        row
        for row in portrait_ws.iter_rows()
        if any(cell.value == "1.profile_income 收入" for cell in row)
    )
    assert any(cell.value == "1.profile_income 收入" for cell in portrait_row)


def test_report_feature_dictionary_legacy_table_name_alias_fills_metric_table_name(
    tmp_path,
):
    from newt.reporting import tables

    feature_dict_path = tmp_path / "legacy_feature_dict.csv"
    pd.DataFrame(
        [
            {
                "英文名": "feature_a",
                "中文名": "特征A中文",
                "来源": "thirdparty",
                "表名": "legacy_metric_feature_a",
            }
        ]
    ).to_csv(feature_dict_path, index=False)

    parsed = tables._load_feature_dictionary(str(feature_dict_path))
    assert {"英文名", "中文名", "来源", "指标表英文名"}.issubset(parsed.columns)
    assert parsed.loc[0, "指标表英文名"] == "legacy_metric_feature_a"


def test_report_feature_dictionary_prefers_explicit_metric_table_name(
    tmp_path,
):
    from newt.reporting import tables

    feature_dict_path = tmp_path / "mixed_feature_dict.csv"
    pd.DataFrame(
        [
            {
                "英文名": "feature_a",
                "中文名": "特征A中文",
                "来源": "thirdparty",
                "指标表英文名": "explicit_metric_feature_a",
                "表名": "legacy_metric_feature_a",
            }
        ]
    ).to_csv(feature_dict_path, index=False)

    parsed = tables._load_feature_dictionary(str(feature_dict_path))
    assert parsed.loc[0, "指标表英文名"] == "explicit_metric_feature_a"


def test_report_runtime_option_validation_rejects_invalid_engine(
    report_frame,
    fake_lightgbm_model,
):
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        engine="invalid-engine",
    )

    with pytest.raises(ValueError, match="engine must be 'auto', 'rust' or 'python'"):
        report.generate()


def test_report_auto_engine_resolves_to_python_when_native_missing(
    report_frame,
    fake_lightgbm_model,
    monkeypatch,
):
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        engine="auto",
    )
    monkeypatch.setattr(report_module, "load_native_module", lambda: None)
    assert report._resolve_engine() == "python"


def test_report_rust_engine_raises_when_native_missing(
    report_frame,
    fake_lightgbm_model,
    monkeypatch,
):
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        engine="rust",
    )
    monkeypatch.setattr(report_module, "load_native_module", lambda: None)
    with pytest.raises(ImportError, match="Rust engine requested"):
        report.generate()


def test_report_runtime_option_validation_rejects_invalid_feature_df(
    report_frame,
    fake_lightgbm_model,
):
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        feature_df="feature_dict.csv",
    )

    with pytest.raises(
        ValueError,
        match="feature_df must be a pandas DataFrame when provided",
    ):
        report.generate()


def test_report_runtime_option_validation_rejects_invalid_metrics_mode(
    report_frame,
    fake_lightgbm_model,
):
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        metrics_mode="invalid-mode",
    )

    with pytest.raises(ValueError, match="metrics_mode must be 'exact' or 'binned'"):
        report.generate()


def test_report_metrics_mode_binned_is_exposed_and_usable(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "metrics_mode_binned.xlsx"
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
        metrics_mode="binned",
        report_out_path=str(output_path),
    )

    report.generate()

    options = report.result_.metadata["report_compute_options"]
    assert options["metrics_mode"] == "binned"
    performance_sheet = report.result_.get_sheet("3.模型表现")
    tag_metrics = performance_sheet.get_block("二、按tag模型效果").data
    month_metrics = performance_sheet.get_block("三、按月模型效果").data
    assert tag_metrics["AUC"].notna().all()
    assert month_metrics["KS"].notna().all()


def test_report_supports_scorecard_model_with_lr_summary_and_details_sheet(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    output_path = tmp_path / "scorecard_report.xlsx"
    report = Report(
        data=report_frame,
        model=fake_scorecard_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["变量分析", "评分卡计算明细"],
        report_out_path=str(output_path),
    )

    report.generate()

    assert report.result_.sheet_names == ["2.变量分析", "评分卡计算明细"]

    variable_sheet = report.result_.get_sheet("2.变量分析")
    feature_table = variable_sheet.get_block("二、变量分析").data
    assert {
        "coefficient",
        "std_error",
        "z_value",
        "p_value",
        "ci_lower",
        "ci_upper",
        "odds_ratio",
    }.issubset(feature_table.columns)
    assert {
        "gain",
        "gain_per",
        "weight",
        "weight_per",
    }.isdisjoint(set(feature_table.columns))
    assert feature_table["vars"].tolist() == ["feature_a", "feature_b"]
    assert feature_table["p_value"].notna().all()

    summary_block = variable_sheet.get_block("三、模型统计摘要").data
    assert "AIC" in summary_block["统计项"].values
    assert summary_block["数值"].notna().any()

    details_sheet = report.result_.get_sheet("评分卡计算明细")
    base_block = details_sheet.get_block("一、评分卡计算参数").data
    points_block = details_sheet.get_block("二、评分卡分值拆解").data
    assert {"base_score", "pdo", "intercept_points"}.issubset(base_block.columns)
    assert {"变量", "分箱", "分值"}.issubset(points_block.columns)
    assert "Intercept" in points_block["变量"].values


def test_report_scorecard_details_sheet_sorts_bins_and_places_missing_last(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    from newt.modeling.scorecard import Scorecard

    payload = fake_scorecard_model.to_dict()
    rows = payload["features"]["feature_a"]
    missing_rows = [row for row in rows if str(row["bin"]) == "Missing"]
    non_missing_rows = [row for row in rows if str(row["bin"]) != "Missing"]
    payload["features"]["feature_a"] = missing_rows + list(reversed(non_missing_rows))
    restored_scorecard = Scorecard().from_dict(payload)

    output_path = tmp_path / "scorecard_details_sorted.xlsx"
    report = Report(
        data=report_frame,
        model=restored_scorecard,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["评分卡计算明细"],
        report_out_path=str(output_path),
    )
    report.generate()

    points_block = (
        report.result_.get_sheet("评分卡计算明细").get_block("二、评分卡分值拆解").data
    )
    assert list(dict.fromkeys(points_block["变量"]))[:3] == [
        "Intercept",
        "feature_a",
        "feature_b",
    ]

    feature_a_bins = points_block.loc[
        points_block["变量"] == "feature_a", "分箱"
    ].tolist()
    assert feature_a_bins[-1] == "Missing"

    left_bounds = []
    for label in feature_a_bins[:-1]:
        left_text = (
            str(label).replace("[", "").replace("(", "").split(",", maxsplit=1)[0]
        ).strip()
        left_bounds.append(float("-inf") if left_text == "-inf" else float(left_text))
    assert left_bounds == sorted(left_bounds)


def test_report_scorecard_details_sheet_uses_feature_dictionary_chinese_names(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    feature_df = pd.DataFrame(
        [
            {
                "英文名": "feature_a",
                "中文名": "特征A中文",
            }
        ]
    )
    output_path = tmp_path / "scorecard_details_chinese_names.xlsx"
    report = Report(
        data=report_frame,
        model=fake_scorecard_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        feature_df=feature_df,
        sheet_list=["评分卡计算明细"],
        report_out_path=str(output_path),
    )

    report.generate()

    points_block = (
        report.result_.get_sheet("评分卡计算明细").get_block("二、评分卡分值拆解").data
    )
    assert points_block.columns[:3].tolist() == ["变量", "中文名", "分箱"]

    chinese_names = (
        points_block.drop_duplicates("变量").set_index("变量")["中文名"].to_dict()
    )
    assert chinese_names["Intercept"] == "截距"
    assert chinese_names["feature_a"] == "特征A中文"
    assert chinese_names["feature_b"] == ""


def test_report_scorecard_details_sheet_uses_points_decimals_output(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    from newt.modeling.scorecard import Scorecard

    payload = fake_scorecard_model.to_dict()
    payload["points_decimals"] = 1
    payload["intercept_points"] = 123.456
    payload["features"]["feature_a"][0]["points"] = 12.345
    payload["features"]["feature_a"][1]["points"] = -9.876
    rounded_scorecard = Scorecard().from_dict(payload)

    output_path = tmp_path / "scorecard_details_points_decimals.xlsx"
    report = Report(
        data=report_frame,
        model=rounded_scorecard,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["评分卡计算明细"],
        report_out_path=str(output_path),
    )
    report.generate()

    points_block = (
        report.result_.get_sheet("评分卡计算明细").get_block("二、评分卡分值拆解").data
    )
    export_table = rounded_scorecard.export().rename(
        columns={"feature": "变量", "bin": "分箱", "points": "分值"}
    )
    assert np.allclose(
        points_block["分值"].to_numpy(dtype=float),
        export_table["分值"].to_numpy(dtype=float),
    )
    assert np.allclose(
        points_block["分值"].to_numpy(dtype=float),
        np.round(points_block["分值"].to_numpy(dtype=float), 1),
    )


def test_report_hides_gain_weight_columns_for_logistic_models(
    tmp_path,
    report_frame,
):
    class FakeLogisticModel:
        def __init__(self):
            self.feature_names_in_ = np.array(["feature_a", "feature_b"])

    output_path = tmp_path / "logistic_variable_analysis.xlsx"
    report = Report(
        data=report_frame,
        model=FakeLogisticModel(),
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["变量分析"],
        report_out_path=str(output_path),
    )

    report.generate()

    variable_sheet = report.result_.get_sheet("2.变量分析")
    feature_table = variable_sheet.get_block("二、变量分析").data
    assert {
        "gain",
        "gain_per",
        "weight",
        "weight_per",
    }.isdisjoint(set(feature_table.columns))


def test_report_scorecard_variable_analysis_uses_scorecard_bin_rules_for_iv_and_psi(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    output_path = tmp_path / "scorecard_variable_bins.xlsx"
    report = Report(
        data=report_frame,
        model=fake_scorecard_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["变量分析"],
        report_out_path=str(output_path),
    )

    report.generate()

    variable_sheet = report.result_.get_sheet("2.变量分析")
    feature_table = variable_sheet.get_block("二、变量分析").data.set_index("vars")
    feature_a_train_bin = variable_sheet.get_block("1.feature_a 训练分箱表").data
    feature_a_oot_bin = variable_sheet.get_block("1.feature_a OOT分箱表").data
    feature_a_non_missing = feature_a_train_bin.loc[
        feature_a_train_bin["bin"] != "Missing"
    ]

    # fake_scorecard_model uses split=10.0 for feature_a.
    assert len(feature_a_non_missing) == 2
    assert feature_a_non_missing["max"].dropna().min() == pytest.approx(10.0)
    assert feature_table.loc["feature_a", "iv_train"] == pytest.approx(
        float(feature_a_train_bin["iv"].sum())
    )
    assert feature_table.loc["feature_a", "iv_oot"] == pytest.approx(
        float(feature_a_oot_bin["iv"].sum())
    )

    train_feature = report_frame.loc[report_frame["tag"] == "train", "feature_a"]
    oot_feature = report_frame.loc[report_frame["tag"] == "oot", "feature_a"]
    edges = np.asarray([-np.inf, 10.0, np.inf], dtype=float)

    def _bin_counts(values: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        non_missing_bins = len(edges) - 1
        labels = pd.cut(
            pd.Series(numeric),
            bins=edges,
            include_lowest=True,
            labels=False,
        )
        indices = labels.to_numpy(dtype=float)
        missing = np.isnan(indices)
        indices[missing] = non_missing_bins
        indices = indices.astype(np.int64)
        return np.bincount(indices, minlength=non_missing_bins + 1).astype(float)

    expected_counts = _bin_counts(train_feature)
    actual_counts = _bin_counts(oot_feature)
    expected_prop = expected_counts / max(expected_counts.sum(), 1.0)
    actual_prop = actual_counts / max(actual_counts.sum(), 1.0)
    expected_prop = np.clip(expected_prop, 1e-8, None)
    actual_prop = np.clip(actual_prop, 1e-8, None)
    expected_psi = float(
        np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    )
    assert feature_table.loc["feature_a", "psi"] == pytest.approx(expected_psi)


def test_report_scorecard_variable_analysis_uses_high_precision_split_boundaries(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    from newt.modeling.scorecard import Scorecard

    split = 10.123456789
    payload = fake_scorecard_model.to_dict()
    payload["binning_rules"]["feature_a"]["splits"] = [split]
    payload["features"]["feature_a"] = [
        {"feature": "feature_a", "bin": f"(-inf, {split}]", "woe": -0.4, "points": 8.0},
        {"feature": "feature_a", "bin": f"({split}, inf]", "woe": 0.6, "points": -12.0},
        {"feature": "feature_a", "bin": "Missing", "woe": 0.0, "points": 0.0},
    ]
    precise_scorecard = Scorecard().from_dict(payload)

    frame = report_frame.copy()
    near_boundary = [split - 1e-9, split, split + 1e-9]
    frame.loc[:, "feature_a"] = [
        near_boundary[idx % len(near_boundary)] for idx in range(len(frame))
    ]

    output_path = tmp_path / "scorecard_precise_split_report.xlsx"
    report = Report(
        data=frame,
        model=precise_scorecard,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["变量分析"],
        report_out_path=str(output_path),
    )
    report.generate()

    variable_sheet = report.result_.get_sheet("2.变量分析")
    train_bins = variable_sheet.get_block("1.feature_a 训练分箱表").data
    non_missing = train_bins.loc[train_bins["bin"] != "Missing"]
    assert len(non_missing) == 2
    assert non_missing["max"].dropna().min() == pytest.approx(split)

    train_subset = frame.loc[
        (frame["tag"] == "train") & frame["label_main"].isin([0, 1]), "feature_a"
    ]
    expected_lower_total = int(
        (pd.to_numeric(train_subset, errors="coerce") <= split).sum()
    )
    expected_upper_total = int(
        (pd.to_numeric(train_subset, errors="coerce") > split).sum()
    )
    lower_total = int(non_missing.sort_values("max").iloc[0]["total"])
    upper_total = int(non_missing.sort_values("max").iloc[1]["total"])
    assert lower_total == expected_lower_total
    assert upper_total == expected_upper_total


def test_report_scorecard_model_performance_includes_lr_params(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    output_path = tmp_path / "scorecard_model_performance.xlsx"
    report = Report(
        data=report_frame,
        model=fake_scorecard_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["模型表现"],
        report_out_path=str(output_path),
    )

    report.generate()

    performance_sheet = report.result_.get_sheet("3.模型表现")
    param_table = performance_sheet.get_block("一、建模方法选择").data
    names = set(param_table["参数名称"].tolist())
    assert {
        "base_score",
        "pdo",
        "fit_intercept",
        "method",
        "maxiter",
        "alpha",
    }.issubset(names)


def test_report_model_performance_treats_missing_tag_as_none_segment(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    frame = report_frame.copy()
    frame.loc[0, "tag"] = None
    frame.loc[1, "tag"] = ""

    output_path = tmp_path / "missing_tag_none_segment.xlsx"
    report = Report(
        data=frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["模型表现"],
        report_out_path=str(output_path),
    )

    report.generate()

    performance_sheet = report.result_.get_sheet("3.模型表现")
    tag_metrics = performance_sheet.get_block("二、按tag模型效果").data
    assert "None" in tag_metrics["样本集"].astype(str).tolist()


def test_report_scorecard_missing_lr_stats_keeps_columns_and_does_not_fail(
    tmp_path,
    report_frame,
    fake_scorecard_model,
):
    from newt.modeling.scorecard import Scorecard

    payload = fake_scorecard_model.to_dict()
    payload.pop("feature_statistics", None)
    payload.pop("model_statistics", None)
    restored_scorecard = Scorecard().from_dict(payload)

    output_path = tmp_path / "scorecard_report_no_stats.xlsx"
    report = Report(
        data=report_frame,
        model=restored_scorecard,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["变量分析"],
        report_out_path=str(output_path),
    )

    report.generate()

    variable_sheet = report.result_.get_sheet("2.变量分析")
    feature_table = variable_sheet.get_block("二、变量分析").data
    assert "p_value" in feature_table.columns
    assert feature_table["p_value"].isna().all()


def test_report_supports_amount_metrics_in_related_report_tables(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    frame = report_frame.copy()
    frame["prin_bal_amount"] = np.where(frame["label_main"] == 1, 45.0, 10.0)
    frame["loan_amount"] = 100.0

    output_path = tmp_path / "report_amount_metrics.xlsx"
    report = Report(
        data=frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        sheet_list=["模型表现", "分维度对比", "新老模型对比"],
        report_out_path=str(output_path),
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    report.generate()

    expected_amount_columns = {
        "逾期本金",
        "放款金额",
        "金额坏占比",
        "放款金额占比",
        "逾期本金占比",
        "10%金额lift",
        "5%金额lift",
        "2%金额lift",
        "1%金额lift",
    }

    performance_sheet = report.result_.get_sheet("3.模型表现")
    tag_metrics = performance_sheet.get_block("二、按tag模型效果").data
    month_metrics = performance_sheet.get_block("三、按月模型效果").data
    assert expected_amount_columns.issubset(tag_metrics.columns)
    assert expected_amount_columns.issubset(month_metrics.columns)

    dimensional_sheet = report.result_.get_sheet("附1 分维度对比")
    dimensional_data_block = next(
        block
        for block in dimensional_sheet.blocks
        if block.title.startswith("1.") and not block.data.empty
    )
    assert expected_amount_columns.issubset(dimensional_data_block.data.columns)

    comparison_sheet = report.result_.get_sheet("附2 新老模型对比")
    tag_compare = comparison_sheet.get_block("按tag新老模型对比(score_old_a)").data
    assert expected_amount_columns.issubset(tag_compare.columns)


def test_report_supports_explicit_amount_metrics_sheet_selector(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    frame = report_frame.copy()
    frame["prin_bal_amount"] = np.where(frame["label_main"] == 1, 45.0, 10.0)
    frame["loan_amount"] = 100.0

    output_path = tmp_path / "report_amount_sheet.xlsx"
    report = Report(
        data=frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        sheet_list=["金额指标"],
        report_out_path=str(output_path),
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    report.generate()

    assert report.result_.sheet_names == ["附1 金额指标"]
    amount_sheet = report.result_.get_sheet("附1 金额指标")
    excluded_headcount_columns = {
        "总",
        "好",
        "坏",
        "坏占比",
        "AUC",
        "KS",
        "10%lift",
        "5%lift",
        "2%lift",
        "1%lift",
    }
    required_amount_columns = {
        "放款金额",
        "逾期本金",
        "金额坏占比",
        "放款金额占比",
        "逾期本金占比",
        "金额AUC",
        "金额KS",
        "10%金额lift",
        "5%金额lift",
        "2%金额lift",
        "1%金额lift",
    }

    for block_title in ["按tag模型效果", "按月模型效果", "分维度对比"]:
        block_data = amount_sheet.get_block(block_title).data
        assert excluded_headcount_columns.isdisjoint(set(block_data.columns))
        assert required_amount_columns.issubset(set(block_data.columns))

    assert amount_sheet.get_block("按tag模型效果").data["放款金额"].notna().all()
    assert amount_sheet.get_block("按月模型效果").data["金额AUC"].notna().all()


def test_report_does_not_auto_append_amount_sheet_when_sheet_list_excludes_it(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    frame = report_frame.copy()
    frame["prin_bal_amount"] = np.where(frame["label_main"] == 1, 45.0, 10.0)
    frame["loan_amount"] = 100.0

    output_path = tmp_path / "report_no_auto_amount_sheet.xlsx"
    report = Report(
        data=frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        sheet_list=["模型表现", "分维度对比"],
        report_out_path=str(output_path),
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )

    report.generate()

    assert "附3 金额指标" not in report.result_.sheet_names
    assert all(
        "金额指标" not in sheet_name for sheet_name in report.result_.sheet_names
    )


def test_report_model_comparison_uses_positive_score_intersection(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    frame = report_frame.copy()
    frame.loc[frame.groupby("tag").head(1).index, "score_old_a"] = 0.0
    frame.loc[frame.groupby("tag").nth(1).index, "score_old_a"] = np.nan

    output_path = tmp_path / "report_model_compare_intersection.xlsx"
    report = Report(
        data=frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        sheet_list=["新老模型对比"],
        report_out_path=str(output_path),
    )
    report.generate()

    compare_sheet = report.result_.get_sheet("附1 新老模型对比")
    tag_compare = compare_sheet.get_block("按tag新老模型对比(score_old_a)").data
    new_model_rows = tag_compare.loc[tag_compare["模型"] == "score_new"].set_index(
        "样本集"
    )
    old_model_rows = tag_compare.loc[tag_compare["模型"] == "score_old_a"].set_index(
        "样本集"
    )
    assert (new_model_rows["总"] == old_model_rows["总"]).all()

    score_new = pd.to_numeric(frame["score_new"], errors="coerce")
    score_old = pd.to_numeric(frame["score_old_a"], errors="coerce")
    intersection_mask = (
        score_new.notna()
        & score_old.notna()
        & np.isfinite(score_new.to_numpy(dtype=float))
        & np.isfinite(score_old.to_numpy(dtype=float))
        & score_new.gt(0)
        & score_old.gt(0)
    )
    expected_counts = frame.loc[intersection_mask].groupby("tag").size()
    for tag_value, expected_total in expected_counts.items():
        assert int(new_model_rows.loc[tag_value, "总"]) == int(expected_total)
        assert int(old_model_rows.loc[tag_value, "总"]) == int(expected_total)


def test_report_amount_sheet_model_comparison_uses_positive_score_intersection(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    frame = report_frame.copy()
    frame["prin_bal_amount"] = np.where(frame["label_main"] == 1, 45.0, 10.0)
    frame["loan_amount"] = 100.0
    frame.loc[frame.groupby("tag").head(1).index, "score_old_a"] = 0.0
    frame.loc[frame.groupby("tag").nth(1).index, "score_old_a"] = np.nan

    output_path = tmp_path / "report_amount_compare_intersection.xlsx"
    report = Report(
        data=frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        score_list=["score_old_a"],
        dim_list=["channel_dim"],
        sheet_list=["金额指标", "新老模型对比"],
        report_out_path=str(output_path),
        prin_bal_amount_col="prin_bal_amount",
        loan_amount_col="loan_amount",
    )
    report.generate()

    amount_sheet_name = next(
        name for name in report.result_.sheet_names if "金额指标" in name
    )
    amount_sheet = report.result_.get_sheet(amount_sheet_name)
    tag_compare = amount_sheet.get_block("按tag新老模型对比(score_old_a)").data
    new_model_rows = tag_compare.loc[tag_compare["模型"] == "score_new"].set_index(
        "样本集"
    )
    old_model_rows = tag_compare.loc[tag_compare["模型"] == "score_old_a"].set_index(
        "样本集"
    )
    assert np.allclose(
        new_model_rows["放款金额"].to_numpy(dtype=float),
        old_model_rows["放款金额"].to_numpy(dtype=float),
    )

    score_new = pd.to_numeric(frame["score_new"], errors="coerce")
    score_old = pd.to_numeric(frame["score_old_a"], errors="coerce")
    intersection_mask = (
        score_new.notna()
        & score_old.notna()
        & np.isfinite(score_new.to_numpy(dtype=float))
        & np.isfinite(score_old.to_numpy(dtype=float))
        & score_new.gt(0)
        & score_old.gt(0)
    )
    expected_loan = frame.loc[intersection_mask].groupby("tag")["loan_amount"].sum()
    for tag_value, expected_value in expected_loan.items():
        assert float(new_model_rows.loc[tag_value, "放款金额"]) == pytest.approx(
            float(expected_value)
        )
        assert float(old_model_rows.loc[tag_value, "放款金额"]) == pytest.approx(
            float(expected_value)
        )


def test_report_requires_amount_columns_to_be_provided_in_pairs(
    tmp_path,
    report_frame,
    fake_lightgbm_model,
):
    output_path = tmp_path / "invalid_amount_args.xlsx"
    report = Report(
        data=report_frame,
        model=fake_lightgbm_model,
        tag="tag",
        score_col="score_new",
        date_col="obs_date",
        label_list=["label_main"],
        sheet_list=["模型表现"],
        report_out_path=str(output_path),
        prin_bal_amount_col="prin_bal_amount",
    )

    with pytest.raises(
        ValueError,
        match="prin_bal_amount_col and loan_amount_col must be provided together",
    ):
        report.generate()
