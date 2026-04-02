"""State container for scorecard pipeline execution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PipelineState:
    """Mutable state shared across pipeline steps."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None
    X_current: pd.DataFrame = field(init=False)
    X_test_current: Optional[pd.DataFrame] = field(init=False)
    steps: List[str] = field(default_factory=list)
    prefilter: Any = None
    binner: Any = None
    woe_encoders: Dict[str, Any] = field(default_factory=dict)
    postfilter: Any = None
    stepwise: Any = None
    model: Any = None
    scorecard: Any = None
    X_binned: Optional[pd.DataFrame] = None
    X_woe: Optional[pd.DataFrame] = None
    X_test_binned: Optional[pd.DataFrame] = None
    X_test_woe: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        self.X_train = self.X_train.copy()
        self.y_train = self.y_train.copy()
        self.X_test = self.X_test.copy() if self.X_test is not None else None
        self.y_test = self.y_test.copy() if self.y_test is not None else None
        self.X_current = self.X_train.copy()
        self.X_test_current = self.X_test.copy() if self.X_test is not None else None

    @property
    def selected_features(self) -> List[str]:
        """Return the current working feature set."""
        return self.X_current.columns.tolist()
