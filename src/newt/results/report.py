"""Stable result objects for model reporting."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ReportChart:
    """Chart definition attached to a report block."""

    chart_type: str
    category_column: str
    value_columns: List[str]
    secondary_value_columns: List[str] = field(default_factory=list)
    title: str = ""
    source_block_title: str = ""
    height_rows: int = 20


@dataclass
class ReportBlock:
    """A renderable block inside a report sheet."""

    title: str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""
    chart: Optional[ReportChart] = None
    blank_rows_after: int = 1


@dataclass
class ReportSheet:
    """Ordered collection of blocks for a single sheet."""

    name: str
    blocks: List[ReportBlock] = field(default_factory=list)

    def get_block(self, title: str) -> ReportBlock:
        """Return a block by title."""
        for block in self.blocks:
            if block.title == title:
                return block
        raise KeyError(f"Block '{title}' not found in sheet '{self.name}'.")


@dataclass
class ModelReportResult:
    """Stable container for generated report data."""

    sheets: Dict[str, ReportSheet] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_path: str = ""

    @property
    def sheet_names(self) -> List[str]:
        """Return sheet names in output order."""
        return list(self.sheets.keys())

    def get_sheet(self, name: str) -> ReportSheet:
        """Return a sheet by name."""
        if name not in self.sheets:
            raise KeyError(f"Sheet '{name}' not found.")
        return self.sheets[name]
