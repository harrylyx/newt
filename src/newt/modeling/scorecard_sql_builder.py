"""Build ANSI SQL scoring expressions from scorecard specifications."""

from typing import List

import numpy as np

from newt.results import FeatureScoreSpec, ScorecardSpec


class ScorecardSQLBuilder:
    """Generate ANSI SQL for scorecard scoring."""

    def __init__(self, spec: ScorecardSpec):
        self.spec = spec

    def build(
        self,
        table_name: str = "input_table",
        score_alias: str = "score",
        include_breakdown: bool = False,
    ) -> str:
        """Build a SQL query that calculates scorecard points."""
        feature_terms: List[str] = []
        select_columns: List[str] = []

        for feature in self.spec.feature_names:
            feature_spec = self.spec.feature_scores.get(feature)
            binning_rule = self.spec.binning_rules.get(feature)
            if feature_spec is None or binning_rule is None:
                continue

            feature_expr = self._build_feature_expression(
                feature=feature,
                feature_spec=feature_spec,
                splits=binning_rule.splits,
                missing_label=binning_rule.missing_label,
            )
            feature_terms.append(feature_expr)

            if include_breakdown:
                select_columns.append(f"{feature_expr} AS {feature}_points")

        score_expression = self._build_score_expression(feature_terms)
        select_columns.append(f"{score_expression} AS {score_alias}")

        lines = ["SELECT"]
        for idx, column in enumerate(select_columns):
            rendered = [f"  {line}" for line in column.splitlines()]
            if idx < len(select_columns) - 1:
                rendered[-1] = f"{rendered[-1]},"
            lines.extend(rendered)
            if idx < len(select_columns) - 1:
                lines.append("")
        lines.append(f"FROM {table_name}")
        return "\n".join(lines)

    def _build_feature_expression(
        self,
        feature: str,
        feature_spec: FeatureScoreSpec,
        splits: List[float],
        missing_label: str,
    ) -> str:
        """Build a CASE expression for one feature."""
        score_map = feature_spec.score_map()
        clauses = [
            f"WHEN {feature} IS NULL THEN "
            f"{self._format_number(score_map.get(missing_label, 0.0))}"
        ]

        if not splits:
            non_missing_points = score_map.get("(-inf, inf]", 0.0)
            clauses.append(
                f"WHEN {feature} IS NOT NULL THEN "
                f"{self._format_number(non_missing_points)}"
            )
            return self._render_case_expression(clauses)

        split_values = [float(value) for value in splits]
        split_texts = [self._format_number(value) for value in split_values]

        first_label = f"(-inf, {split_values[0]}]"
        clauses.append(
            f"WHEN {feature} <= {split_texts[0]} THEN "
            f"{self._format_number(score_map.get(first_label, 0.0))}"
        )

        for idx in range(1, len(split_values)):
            previous = split_texts[idx - 1]
            current = split_texts[idx]
            bin_label = f"({split_values[idx - 1]}, {split_values[idx]}]"
            clauses.append(
                f"WHEN {feature} > {previous} AND {feature} <= {current} THEN "
                f"{self._format_number(score_map.get(bin_label, 0.0))}"
            )

        last_label = f"({split_values[-1]}, inf]"
        clauses.append(
            f"WHEN {feature} > {split_texts[-1]} THEN "
            f"{self._format_number(score_map.get(last_label, 0.0))}"
        )
        return self._render_case_expression(clauses)

    def _render_case_expression(self, clauses: List[str]) -> str:
        """Render a formatted CASE expression block."""
        lines = ["(", "  CASE"]
        lines.extend([f"    {clause}" for clause in clauses])
        lines.append("    ELSE 0.0")
        lines.append("  END")
        lines.append(")")
        return "\n".join(lines)

    def _build_score_expression(self, feature_terms: List[str]) -> str:
        """Build the formatted total score expression."""
        lines = ["(", f"  {self._format_number(self.spec.intercept_points)}"]
        for feature_expr in feature_terms:
            feature_lines = feature_expr.splitlines()
            lines.append("")
            lines.append(f"  + {feature_lines[0]}")
            lines.extend([f"    {line}" for line in feature_lines[1:]])
        lines.append(")")
        return "\n".join(lines)

    def _format_number(self, value: float) -> str:
        """Format finite numeric values for SQL output."""
        numeric = float(value)
        if not np.isfinite(numeric):
            return "0.0"
        text = format(numeric, ".15g")
        if "e" not in text and "." not in text:
            text = f"{text}.0"
        return text
