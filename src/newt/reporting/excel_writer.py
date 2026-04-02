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

    def __init__(self) -> None:
        self.font_name = "PingFang SC" if platform.system() == "Darwin" else "Microsoft YaHei"

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
                for block in sheet.blocks:
                    row = self._write_block(
                        writer=writer,
                        worksheet=worksheet,
                        sheet_name=sheet.name,
                        block=block,
                        start_row=row,
                        workbook=workbook,
                        formats=formats,
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
    ) -> int:
        worksheet.write(start_row, 0, block.title, formats["title"])
        row = start_row + 1
        if block.note:
            worksheet.write(row, 0, block.note, formats["note"])
            row += 1
        if block.data.empty:
            return row + max(block.blank_rows_after, 1)

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
        if block.chart is not None and not block.data.empty:
            self._add_chart(
                worksheet=worksheet,
                block=block,
                start_row=row,
                end_row=end_row,
                workbook=workbook,
                formats=formats,
            )
            end_row += 15
        return end_row + block.blank_rows_after + 1

    def _add_chart(
        self,
        worksheet,
        block: ReportBlock,
        start_row: int,
        end_row: int,
        workbook,
        formats: Dict[str, object],
    ) -> None:
        chart = workbook.add_chart({"type": "column"})
        data = block.data
        category_index = data.columns.get_loc(block.chart.category_column)
        for column in block.chart.value_columns:
            chart.add_series(
                {
                    "name": column,
                    "categories": [worksheet.name, start_row + 1, category_index, end_row, category_index],
                    "values": [
                        worksheet.name,
                        start_row + 1,
                        data.columns.get_loc(column),
                        end_row,
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
                        "categories": [worksheet.name, start_row + 1, category_index, end_row, category_index],
                        "values": [
                            worksheet.name,
                            start_row + 1,
                            data.columns.get_loc(column),
                            end_row,
                            data.columns.get_loc(column),
                        ],
                        "y2_axis": True,
                    }
                )
            chart.combine(line_chart)
        chart.set_title({"name": block.chart.title or block.title})
        worksheet.insert_chart(end_row + 2, 0, chart)

    def _format_table(self, worksheet, data: pd.DataFrame, start_row: int, formats: Dict[str, object]) -> None:
        for column_index, column_name in enumerate(data.columns):
            worksheet.write(start_row, column_index, column_name, formats["header"])
            number_format = self._choose_body_format(column_name, formats)
            worksheet.set_column(column_index, column_index, self._column_width(data[column_name], column_name), number_format)
            for row_offset, value in enumerate(data[column_name], start=1):
                if pd.isna(value) or (
                    isinstance(value, (float, np.floating)) and not np.isfinite(value)
                ):
                    worksheet.write_blank(
                        start_row + row_offset,
                        column_index,
                        None,
                        number_format,
                    )
                elif isinstance(value, (str, int, float, bool, pd.Timestamp)):
                    worksheet.write(
                        start_row + row_offset,
                        column_index,
                        value,
                        number_format,
                    )
                else:
                    worksheet.write(
                        start_row + row_offset,
                        column_index,
                        str(value),
                        number_format,
                    )

    def _build_formats(self, workbook):
        title = workbook.add_format(
            {
                "bold": True,
                "font_name": self.font_name,
                "font_size": 14,
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
        integer = workbook.add_format(
            {"font_name": self.font_name, "border": 1, "num_format": "#,##0"}
        )
        return {
            "title": title,
            "note": note,
            "header": header,
            "text": text,
            "percent": percent,
            "decimal": decimal,
            "integer": integer,
        }

    def _choose_body_format(self, column_name: str, formats: Dict[str, object]):
        name = str(column_name)
        lower = name.lower()
        if any(token in lower for token in ["auc", "ks", "psi", "lift", "gain_per", "weight_per"]):
            return formats["decimal"]
        if any(token in name for token in ["占比", "比率"]) or any(
            token in lower for token in ["prop", "rate", "坏占比"]
        ):
            return formats["percent"]
        if any(token in name for token in ["总", "好", "坏", "灰", "count"]) or lower in {
            "bads",
            "goods",
            "total",
            "weight",
            "gain",
        }:
            return formats["integer"]
        return formats["text"]

    def _column_width(self, series: pd.Series, name: str) -> int:
        lengths = [len(str(name))]
        lengths.extend(len(str(value)) for value in series.head(50))
        return min(max(lengths) + 2, 28)
