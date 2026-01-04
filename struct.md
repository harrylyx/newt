risk_toolkit/
├── README.md
├── pyproject.toml
├── poetry.lock
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── configs/
│   ├── base.yaml
│   ├── development.yaml
│   └── production.yaml
├── docs/
│   ├── api/
│   ├── tutorials/
│   ├── benchmarks/
│   └── architecture.md
├── examples/
│   ├── 01-basic-usage.ipynb
│   ├── 02-advanced-analysis.ipynb
│   ├── 03-production-pipeline.ipynb
│   └── data/
│       └── sample/
├── notebooks/
│   ├── exploration/
│   └── research/
├── scripts/
│   ├── benchmark.py
│   ├── profile_memory.py
│   └── validate_data.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── backend/
│   │   ├── stats/
│   │   └── metrics/
│   └── integration/
│       ├── test_pipeline.py
│       └── test_performance.py
└── src/
    └── credit_risk/
        ├── __init__.py
        ├── py.typed
        ├── config/
        │   ├── __init__.py
        │   ├── settings.py
        │   └── validation.py
        ├── backend/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── registry.py
        │   ├── pandas_backend.py
        │   ├── polars_backend.py
        │   ├── dask_backend.py
        │   ├── modin_backend.py
        │   └── cudf_backend.py
        ├── data/
        │   ├── __init__.py
        │   ├── types.py
        │   ├── validation.py
        │   ├── splitting.py
        │   └── partitioning.py
        ├── statistics/
        │   ├── __init__.py
        │   ├── univariate.py
        │   ├── multivariate.py
        │   ├── distribution.py
        │   └── stability.py
        ├── features/
        │   ├── __init__.py
        │   ├── selection/
        │   │   ├── __init__.py
        │   │   ├── filter_methods.py
        │   │   ├── wrapper_methods.py
        │   │   └── embedded_methods.py
        │   ├── transformation/
        │   │   ├── __init__.py
        │   │   ├── binning.py
        │   │   ├── encoding.py
        │   │   └── scaling.py
        │   └── analysis/
        │       ├── __init__.py
        │       ├── iv_calculator.py
        │       ├── woe_calculator.py
        │       └── correlation.py
        ├── metrics/
        │   ├── __init__.py
        │   ├── performance/
        │   │   ├── __init__.py
        │   │   ├── classification.py
        │   │   ├── regression.py
        │   │   └── ranking.py
        │   ├── stability/
        │   │   ├── __init__.py
        │   │   ├── psi_calculator.py
        │   │   ├── csi_calculator.py
        │   │   └── drift_detector.py
        │   └── business/
        │       ├── __init__.py
        │       ├── profit_curve.py
        │       └── economic_value.py
        ├── evaluation/
        │   ├── __init__.py
        │   ├── model_evaluator.py
        │   ├── feature_evaluator.py
        │   └── report_generator.py
        ├── monitoring/
        │   ├── __init__.py
        │   ├── drift_monitor.py
        │   ├── performance_monitor.py
        │   └── alert_system.py
        ├── visualization/
        │   ├── __init__.py
        │   ├── plots/
        │   │   ├── __init__.py
        │   │   ├── distribution_plots.py
        │   │   ├── performance_plots.py
        │   │   └── diagnostic_plots.py
        │   └── dashboard/
        │       ├── __init__.py
        │       ├── base_dashboard.py
        │       └── interactive_dashboard.py
        ├── pipeline/
        │   ├── __init__.py
        │   ├── base_pipeline.py
        │   ├── analysis_pipeline.py
        │   └── monitoring_pipeline.py
        ├── utils/
        │   ├── __init__.py
        │   ├── decorators.py
        │   ├── validators.py
        │   ├── serializers.py
        │   └── logging_config.py
        └── exceptions/
            ├── __init__.py
            ├── data_exceptions.py
            ├── computation_exceptions.py
            └── validation_exceptions.py