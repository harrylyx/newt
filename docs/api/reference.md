# API Reference

This page provides automated documentation for the core components of `newt`.

## Binning

::: newt.features.binning.binner
    options:
      members:
        - Binner

::: newt.features.binning.supervised
    options:
      members:
        - ChiMergeBinner
        - DecisionTreeBinner
        - OptBinningBinner

::: newt.features.binning.unsupervised
    options:
      members:
        - EqualWidthBinner
        - EqualFrequencyBinner
        - KMeansBinner

## Feature Selection

::: newt.features.selection.selector
    options:
      members:
        - FeatureSelector

::: newt.features.selection.stepwise
    options:
      members:
        - StepwiseSelector

## Modeling

::: newt.modeling.logistic
    options:
      members:
        - LogisticModel

::: newt.modeling.scorecard
    options:
      members:
        - Scorecard

## Pipeline

::: newt.pipeline.pipeline
    options:
      members:
        - ScorecardPipeline

## Reporting

::: newt.reporting.report
    options:
      members:
        - Report
