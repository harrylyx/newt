"""Public report orchestration API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

from newt.reporting.excel_writer import ExcelReportWriter
from newt.reporting.model_adapter import ModelAdapter
from newt.reporting.score_prep import prepare_report_scores
from newt.reporting.tables import build_report_result, resolve_sheet_names
from newt.results import ModelReportResult


@dataclass
class Report:
    """Generate a multi-sheet Excel model report."""

    data: pd.DataFrame
    model: object
    tag: str
    score_col: str
    date_col: str
    label_list: Sequence[str]
    score_list: Sequence[str] = field(default_factory=list)
    dim_list: Sequence[str] = field(default_factory=list)
    var_list: Sequence[str] = field(default_factory=list)
    sheet_list: Sequence[object] = field(default_factory=list)
    feature_path: Optional[str] = None
    report_out_path: str = "./out/model_report.xlsx"

    result_: Optional[ModelReportResult] = field(default=None, init=False)

    def generate(self) -> str:
        """Generate the report and return the output path."""
        prepared = self._prepare_data()
        prepared, report_score_columns, score_direction_summary = prepare_report_scores(
            data=prepared,
            tag_col=self.tag,
            label_col=self.label_list[0],
            score_names=[self.score_col, *self.score_list],
        )
        selected_sheets = resolve_sheet_names(self.sheet_list)
        adapter = ModelAdapter(self.model)
        result = build_report_result(
            data=prepared,
            model_adapter=adapter,
            tag_col=self.tag,
            month_col="_report_month",
            label_list=self.label_list,
            score_list=self.score_list,
            primary_score_name=self.score_col,
            report_score_columns=report_score_columns,
            score_direction_summary=score_direction_summary,
            dim_list=self.dim_list,
            var_list=self.var_list,
            feature_path=self.feature_path,
            selected_sheets=selected_sheets,
        )
        writer = ExcelReportWriter()
        output_path = writer.write(result, self.report_out_path)
        self.result_ = result
        return output_path

    def _prepare_data(self) -> pd.DataFrame:
        self._validate_columns()
        prepared = self.data.copy()
        prepared[self.tag] = prepared[self.tag].astype("object")
        prepared["_report_month"] = prepared[self.date_col].apply(_normalize_month)
        return prepared

    def _validate_columns(self) -> None:
        required = [self.tag, self.score_col, self.date_col, *self.label_list]
        optional = [*self.score_list, *self.dim_list, *self.var_list]
        missing = [
            column
            for column in [*required, *optional]
            if column and column not in self.data.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {sorted(set(missing))}")


def _normalize_month(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m")

    text = str(value).strip()
    if text.isdigit() and len(text) == 6:
        return text
    if text.isdigit() and len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d").strftime("%Y%m")

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y%m")
    return text
