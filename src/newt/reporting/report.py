"""Public report orchestration API."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from newt.config import LOGGING
from newt.reporting.excel_writer import ExcelReportWriter
from newt.reporting.model_adapter import ModelAdapter
from newt.reporting.score_prep import prepare_report_scores
from newt.reporting.table_context import ReportBuildOptions
from newt.reporting.tables import build_report_result, resolve_sheet_keys
from newt.results import ModelReportResult

LOGGER = logging.getLogger("newt.reporting.report")


@dataclass
class Report:
    """Orchestrator for generating multi-sheet Excel model reports.

    The Report class serves as the primary entry point for creating professional,
    styled Excel workbooks that summarize model performance, variable distributions,
    and dimensional comparisons.

    Attributes:
        data (pd.DataFrame): The input dataset containing scores, labels, and features.
        model (object): A fitted model object (scikit-learn, LightGBM, XGBoost, etc.)
            used to extract feature importance and parameters.
        tag (str): Column name identifying sample segments (e.g., 'train', 'oot').
        score_col (str): Column name for the primary model score to be analyzed.
        date_col (str): Column name for the observation date (used for monthly trends).
        label_list (Sequence[str]): List of target column names (binary 0/1).
        score_list (Sequence[str]): Optional list of secondary/benchmark scores.
        dim_list (Sequence[str]): Optional list of columns for dimensional comparison.
        var_list (Sequence[str]): Optional list of columns for portrait/feature
            analysis.
        sheet_list (Sequence[object]): Optional list of sheets to include
            (names or indices).
        feature_df (pd.DataFrame, optional): Feature dictionary DataFrame used
            for variable metadata mapping.
        report_out_path (str): File path where the Excel workbook will be saved.
        engine (str): Calculation engine to use: 'rust' (default) or 'python'.
        max_workers (int, optional): Maximum parallel workers for computation.
        parallel_sheets (bool): Whether to calculate different sheets in parallel.
        memory_mode (str): Memory usage strategy: 'compact' (default) or 'standard'.
        metrics_mode (str): Calculation mode: 'exact' (default) or
            'binned' (approximate).
        prin_bal_amount_col (str, optional): Column name for principal-balance
            amount used by optional amount-based report metrics.
        loan_amount_col (str, optional): Column name for loan amount used by
            optional amount-based report metrics.

    Examples:
        >>> from newt import Report
        >>> report = Report(
        ...     data=df,
        ...     model=fitted_model,
        ...     tag="segment",
        ...     score_col="new_score",
        ...     date_col="report_date",
        ...     label_list=["target"],
        ...     report_out_path="./final_report.xlsx"
        ... )
        >>> report.generate()
    """

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
    feature_df: Optional[pd.DataFrame] = None
    report_out_path: str = "./out/model_report.xlsx"
    engine: str = "rust"
    max_workers: Optional[int] = None
    parallel_sheets: bool = True
    memory_mode: str = "compact"
    metrics_mode: str = "exact"
    prin_bal_amount_col: Optional[str] = None
    loan_amount_col: Optional[str] = None

    result_: Optional[ModelReportResult] = field(default=None, init=False)

    def generate(self) -> str:
        """Generate the report and return the output path."""
        _configure_report_logger()
        self._validate_runtime_options()
        resolved_workers = self._resolve_max_workers()
        build_options = ReportBuildOptions(
            engine=self.engine,
            max_workers=resolved_workers,
            parallel_sheets=bool(self.parallel_sheets),
            memory_mode=self.memory_mode,
            metrics_mode=self.metrics_mode,
        )
        stage_timings: List[Tuple[str, float]] = []
        total_start = time.perf_counter()
        LOGGER.debug(
            "Report generation started | rows=%d cols=%d primary_score=%s labels=%s "
            "output=%s engine=%s workers=%d parallel_sheets=%s memory_mode=%s "
            "metrics_mode=%s "
            "peak_rss_mb=%s",
            len(self.data),
            len(self.data.columns),
            self.score_col,
            list(self.label_list),
            self.report_out_path,
            build_options.engine,
            build_options.max_workers,
            build_options.parallel_sheets,
            build_options.memory_mode,
            build_options.metrics_mode,
            _format_peak_rss(),
        )

        step_start = time.perf_counter()
        prepared = self._prepare_data()
        _log_stage(
            stage_timings,
            "prepare_data",
            time.perf_counter() - step_start,
            extra=f"rows={len(prepared)} peak_rss_mb={_format_peak_rss()}",
        )

        step_start = time.perf_counter()
        prepared, report_score_columns, score_direction_summary = prepare_report_scores(
            data=prepared,
            tag_col=self.tag,
            label_col=self.label_list[0],
            score_names=[self.score_col, *self.score_list],
        )
        if build_options.memory_mode == "compact":
            _downcast_float_columns(prepared, report_score_columns.values())
        _log_stage(
            stage_timings,
            "prepare_report_scores",
            time.perf_counter() - step_start,
            extra=(
                "report_scores="
                f"{sorted(report_score_columns.keys())} "
                f"peak_rss_mb={_format_peak_rss()}"
            ),
        )

        step_start = time.perf_counter()
        selected_sheets = resolve_sheet_keys(self.sheet_list)
        _log_stage(
            stage_timings,
            "resolve_sheet_keys",
            time.perf_counter() - step_start,
            extra=f"selected_sheet_keys={selected_sheets}",
        )

        step_start = time.perf_counter()
        adapter = ModelAdapter(self.model)
        _log_stage(
            stage_timings,
            "model_adapter_init",
            time.perf_counter() - step_start,
            extra=f"model_family={adapter.model_family}",
        )

        step_start = time.perf_counter()
        result = build_report_result(
            data=prepared,
            model_adapter=adapter,
            tag_col=self.tag,
            month_col="_report_month",
            raw_date_col=self.date_col,
            label_list=self.label_list,
            score_list=self.score_list,
            primary_score_name=self.score_col,
            report_score_columns=report_score_columns,
            score_direction_summary=score_direction_summary,
            dim_list=self.dim_list,
            var_list=self.var_list,
            feature_df=self.feature_df,
            selected_sheets=selected_sheets,
            prin_bal_amount_col=self.prin_bal_amount_col,
            loan_amount_col=self.loan_amount_col,
            options=build_options,
        )
        _log_stage(
            stage_timings,
            "build_report_result",
            time.perf_counter() - step_start,
            extra=(
                f"sheet_count={len(result.sheet_names)} "
                f"peak_rss_mb={_format_peak_rss()}"
            ),
        )

        step_start = time.perf_counter()
        writer = ExcelReportWriter()
        output_path = writer.write(result, self.report_out_path)
        _log_stage(
            stage_timings,
            "write_excel",
            time.perf_counter() - step_start,
            extra=f"output={output_path} peak_rss_mb={_format_peak_rss()}",
        )

        self.result_ = result
        total_elapsed = time.perf_counter() - total_start
        _log_stage(stage_timings, "total", total_elapsed)
        _log_top_slowest_steps(stage_timings)
        LOGGER.debug(
            "Report generation completed | total_elapsed=%.3fs output=%s "
            "peak_rss_mb=%s",
            total_elapsed,
            output_path,
            _format_peak_rss(),
        )
        return output_path

    def _prepare_data(self) -> pd.DataFrame:
        self._validate_columns()
        normalized_tag = _normalize_report_tag_values(self.data[self.tag])
        prepared = self.data.copy(deep=False)
        prepared = prepared.assign(
            **{
                self.tag: normalized_tag,
                "_report_month": _vectorized_normalize_month(self.data[self.date_col]),
            }
        )
        return prepared

    def _validate_columns(self) -> None:
        required = [self.tag, self.score_col, self.date_col, *self.label_list]
        optional = [
            *self.score_list,
            *self.dim_list,
            *self.var_list,
            self.prin_bal_amount_col,
            self.loan_amount_col,
        ]
        if (self.prin_bal_amount_col is None) ^ (self.loan_amount_col is None):
            raise ValueError(
                "prin_bal_amount_col and loan_amount_col must be provided together"
            )
        missing = [
            column
            for column in [*required, *optional]
            if column and column not in self.data.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {sorted(set(missing))}")

    def _validate_runtime_options(self) -> None:
        if self.engine not in {"rust", "python"}:
            raise ValueError("engine must be 'rust' or 'python'")
        if self.memory_mode not in {"compact", "standard"}:
            raise ValueError("memory_mode must be 'compact' or 'standard'")
        if self.metrics_mode not in {"exact", "binned"}:
            raise ValueError("metrics_mode must be 'exact' or 'binned'")
        if self.max_workers is not None and int(self.max_workers) < 1:
            raise ValueError("max_workers must be >= 1")
        if self.feature_df is not None and not isinstance(
            self.feature_df, pd.DataFrame
        ):
            raise ValueError("feature_df must be a pandas DataFrame when provided")

    def _resolve_max_workers(self) -> int:
        if self.max_workers is not None:
            return max(1, int(self.max_workers))
        cpu_total = os.cpu_count() or 1
        return max(1, min(8, cpu_total))


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


def _vectorized_normalize_month(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series([], dtype="object", index=values.index)
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime("%Y%m").fillna("").astype("object")

    # Optimization for 10M+ rows: truncate to date part (e.g., YYYY-MM-DD)
    # to reduce cardinality from millions to thousands before parsing.
    series = values.astype("object")
    text = series.astype(str).str.strip()
    trunc = text.str.slice(stop=10)
    unique_trunc = pd.unique(trunc)
    # Filter out empty/NaN strings
    unique_trunc = [
        v
        for v in unique_trunc
        if pd.notna(v) and str(v).lower() != "nan" and str(v).strip() != ""
    ]

    if not unique_trunc:
        return pd.Series("", index=values.index, dtype="object")

    # Parse only unique date prefixes (a few thousand vs 10 million)
    parsed_unique = pd.to_datetime(unique_trunc, errors="coerce")
    mapping = {}
    for orig, p in zip(unique_trunc, parsed_unique):
        if pd.notna(p):
            mapping[orig] = p.strftime("%Y%m")
        else:
            # If parsing fails, fall back to original text if it looks like YYYYMM
            mapping[orig] = (
                orig[:6] if (len(orig) >= 6 and orig[:6].isdigit()) else orig
            )

    # Map back to the full series (fast O(N) operation)
    result = trunc.map(mapping).fillna("").astype("object")
    return result


def _configure_report_logger() -> None:
    level_name = str(LOGGING.DEFAULT_LOG_LEVEL).upper()
    level_value = getattr(logging, level_name, logging.DEBUG)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level_value,
            format=LOGGING.DEFAULT_LOG_FORMAT,
        )
    LOGGER.setLevel(level_value)


def _downcast_float_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = pd.to_numeric(numeric, downcast="float")


def _normalize_report_tag_values(values: pd.Series) -> pd.Series:
    normalized = values.astype("object").copy()
    text = normalized.astype(str).str.strip()
    missing_mask = normalized.isna() | text.eq("")
    normalized.loc[missing_mask] = "None"
    return normalized


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource
    except Exception:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if usage <= 0:
        return None
    if os.name == "posix" and "darwin" in os.sys.platform:
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _format_peak_rss() -> str:
    peak = _peak_rss_mb()
    if peak is None or pd.isna(peak):
        return "n/a"
    return f"{peak:.1f}"


def _log_stage(
    timings: List[Tuple[str, float]],
    stage_name: str,
    elapsed: float,
    extra: str = "",
) -> None:
    timings.append((stage_name, float(elapsed)))
    suffix = f" | {extra}" if extra else ""
    LOGGER.debug("Step %s finished | elapsed=%.3fs%s", stage_name, elapsed, suffix)


def _log_top_slowest_steps(
    timings: Sequence[Tuple[str, float]], limit: int = 5
) -> None:
    ranked = sorted(timings, key=lambda item: item[1], reverse=True)
    if not ranked:
        return
    top = ", ".join(f"{name}:{elapsed:.3f}s" for name, elapsed in ranked[:limit])
    LOGGER.debug("Top slow stages | %s", top)
