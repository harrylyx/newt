"""Excel rendering for model reports."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from newt.results import ModelReportResult, ReportBlock


class ExcelReportWriter:
    """Write report sheets to a styled Excel workbook."""

    TITLE_RIGHT_COLUMN = 6

    def __init__(self) -> None:
        self.font_name = (
            "PingFang SC" if platform.system() == "Darwin" else "Microsoft YaHei"
        )

    def write(self, result: ModelReportResult, output_path: str) -> str:
        """Write a report result to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(
            path,
            engine="xlsxwriter",
            engine_kwargs={"options": {"nan_inf_to_errors": True}},
        ) as writer:
            workbook = writer.book
            formats = self._build_formats(workbook)
            for sheet in result.sheets.values():
                worksheet = workbook.add_worksheet(sheet.name)
                writer.sheets[sheet.name] = worksheet
                worksheet.freeze_panes(1, 0)
                row = 0
                table_ranges = {}
                for block in sheet.blocks:
                    row = self._write_block(
                        writer=writer,
                        worksheet=worksheet,
                        sheet_name=sheet.name,
                        block=block,
                        start_row=row,
                        workbook=workbook,
                        formats=formats,
                        table_ranges=table_ranges,
                    )
                worksheet.set_column(0, 30, 16)
        result.output_path = str(path)
        return str(path)

    def _write_block(
        self,
        writer: pd.ExcelWriter,
        worksheet,
        sheet_name: str,
        block: ReportBlock,
        start_row: int,
        workbook,
        formats: Dict[str, object],
        table_ranges: Dict[str, Dict[str, object]],
    ) -> int:
        row = start_row
        if block.title:
            worksheet.write(start_row, 0, block.title, formats["title"])
            if block.title_right:
                worksheet.write(
                    start_row,
                    self.TITLE_RIGHT_COLUMN,
                    block.title_right,
                    formats["title_right"],
                )
            row = start_row + 1
        if block.note:
            worksheet.write(row, 0, block.note, formats["note"])
            row += 1
        if block.data.empty and block.chart is None:
            return row + block.blank_rows_after

        end_row = row
        if not block.data.empty:
            block.data.to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=row,
                startcol=0,
                index=False,
            )
            self._format_table(
                worksheet=worksheet,
                data=block.data,
                start_row=row,
                formats=formats,
            )
            end_row = row + len(block.data)
            if block.title:
                table_ranges[block.title] = {
                    "start_row": row,
                    "end_row": end_row,
                    "data": block.data,
                }

        if block.chart is not None:
            chart_start_row = row if block.data.empty else end_row + 2
            reserved_rows = self._add_chart(
                worksheet=worksheet,
                block=block,
                start_row=row,
                end_row=end_row,
                workbook=workbook,
                table_ranges=table_ranges,
            )
            return chart_start_row + reserved_rows + block.blank_rows_after

        return end_row + block.blank_rows_after + 1

    def _add_chart(
        self,
        worksheet,
        block: ReportBlock,
        start_row: int,
        end_row: int,
        workbook,
        table_ranges: Dict[str, Dict[str, object]],
    ) -> int:
        chart = workbook.add_chart({"type": "column"})
        data_range = self._resolve_chart_range(block, start_row, end_row, table_ranges)
        data = data_range["data"]
        chart_data_start_row = data_range["start_row"]
        chart_data_end_row = data_range["end_row"]
        category_index = data.columns.get_loc(block.chart.category_column)
        for column in block.chart.value_columns:
            chart.add_series(
                {
                    "name": column,
                    "categories": [
                        worksheet.name,
                        chart_data_start_row + 1,
                        category_index,
                        chart_data_end_row,
                        category_index,
                    ],
                    "values": [
                        worksheet.name,
                        chart_data_start_row + 1,
                        data.columns.get_loc(column),
                        chart_data_end_row,
                        data.columns.get_loc(column),
                    ],
                }
            )
        if block.chart.secondary_value_columns:
            line_chart = workbook.add_chart({"type": "line"})
            for column in block.chart.secondary_value_columns:
                line_chart.add_series(
                    {
                        "name": column,
                        "categories": [
                            worksheet.name,
                            chart_data_start_row + 1,
                            category_index,
                            chart_data_end_row,
                            category_index,
                        ],
                        "values": [
                            worksheet.name,
                            chart_data_start_row + 1,
                            data.columns.get_loc(column),
                            chart_data_end_row,
                            data.columns.get_loc(column),
                        ],
                        "y2_axis": True,
                    }
                )
            chart.combine(line_chart)
        chart.set_title({"name": block.chart.title or block.title})
        chart.set_size({"width": 720, "height": 360})
        insert_row = start_row if block.data.empty else end_row + 2
        worksheet.insert_chart(insert_row, 0, chart)
        return block.chart.height_rows

    def _format_table(
        self, worksheet, data: pd.DataFrame, start_row: int, formats: Dict[str, object]
    ) -> None:
        worksheet.write_row(
            start_row,
            0,
            [str(column_name) for column_name in data.columns],
            formats["header"],
        )
        for column_index, column_name in enumerate(data.columns):
            number_format = self._choose_body_format(column_name, formats)
            worksheet.set_column(
                column_index,
                column_index,
                self._column_width(data[column_name], column_name),
                number_format,
            )
            values = [self._excel_cell_value(value) for value in data[column_name]]
            worksheet.write_column(
                start_row + 1,
                column_index,
                values,
                number_format,
            )

    def _excel_cell_value(self, value):
        if pd.isna(value):
            return None
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool, pd.Timestamp)):
            return value
        return str(value)

    def _build_formats(self, workbook):
        title = workbook.add_format(
            {
                "bold": True,
                "font_name": self.font_name,
                "font_size": 14,
            }
        )
        title_right = workbook.add_format(
            {
                "font_name": self.font_name,
                "font_size": 11,
                "align": "right",
                "valign": "vcenter",
                "font_color": "#666666",
            }
        )
        note = workbook.add_format({"font_name": self.font_name, "italic": True})
        header = workbook.add_format(
            {
                "bold": True,
                "font_name": self.font_name,
                "font_color": "#FFFFFF",
                "bg_color": "#5B9BD5",
                "border": 1,
                "align": "center",
            }
        )
        text = workbook.add_format({"font_name": self.font_name, "border": 1})
        percent = workbook.add_format(
            {"font_name": self.font_name, "border": 1, "num_format": "0.00%"}
        )
        decimal = workbook.add_format(
            {"font_name": self.font_name, "border": 1, "num_format": "0.00"}
        )
        iv_decimal = workbook.add_format(
            {"font_name": self.font_name, "border": 1, "num_format": "0.0000"}
        )
        integer = workbook.add_format(
            {"font_name": self.font_name, "border": 1, "num_format": "#,##0"}
        )
        return {
            "title": title,
            "title_right": title_right,
            "note": note,
            "header": header,
            "text": text,
            "percent": percent,
            "decimal": decimal,
            "iv_decimal": iv_decimal,
            "integer": integer,
        }

    def _choose_body_format(self, column_name: str, formats: Dict[str, object]):
        name = str(column_name)
        lower = name.lower()
        if lower == "iv" or lower.startswith("iv_") or lower.endswith("_iv"):
            return formats["iv_decimal"]
        if (
            lower.startswith("缺失率_")
            or lower.endswith("_per")
            or any(token in name for token in ["占比", "比率"])
            or any(token in lower for token in ["prop", "rate", "坏占比"])
        ):
            return formats["percent"]
        if any(
            token in lower for token in ["auc", "ks", "psi", "lift", "gain", "weight"]
        ):
            return formats["decimal"]
        if any(token in name for token in ["总", "好", "坏", "灰", "count"]) or lower in {
            "bads",
            "goods",
            "total",
        }:
            return formats["integer"]
        return formats["text"]

    def _column_width(self, series: pd.Series, name: str) -> int:
        lengths = [len(str(name))]
        lengths.extend(len(str(value)) for value in series.head(50))
        return min(max(lengths) + 2, 28)

    def _resolve_chart_range(
        self,
        block: ReportBlock,
        start_row: int,
        end_row: int,
        table_ranges: Dict[str, Dict[str, object]],
    ) -> Dict[str, object]:
        if block.chart and block.chart.source_block_title:
            return table_ranges[block.chart.source_block_title]
        return {
            "start_row": start_row,
            "end_row": end_row,
            "data": block.data,
        }
