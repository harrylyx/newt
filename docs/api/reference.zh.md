# API 参考

本页面提供了 `newt` 核心组件的自动化文档。

## 分箱 (Binning)

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

## 特征筛选 (Feature Selection)

::: newt.features.selection.selector
    options:
      members:
        - FeatureSelector

## WOE 编码

::: newt.features.analysis.woe_calculator
    options:
      members:
        - WOEEncoder

## 模型建模 (Modeling)

::: newt.modeling.logistic
    options:
      members:
        - LogisticModel

::: newt.modeling.scorecard
    options:
      members:
        - Scorecard

## 流水线 (Pipeline)

::: newt.pipeline.pipeline
    options:
      members:
        - ScorecardPipeline
