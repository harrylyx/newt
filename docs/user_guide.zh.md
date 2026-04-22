# Newt 用户指南

本指南涵盖使用 `newt` 进行信用评分卡开发的端到端工作流程，包括分箱、特征选择、特征转换、建模和评分卡生成。

当前项目支持 Python `>=3.8.5,<3.13`（即 Python `3.8.5` 到 `3.12.x`）。

## 安装

### 从 PyPI 安装

```bash
pip install newt
# 或者使用 uv
uv add newt
```

### 从 GitHub Release 安装（推荐，可获得原生引擎）

包含预编译高性能 Rust 引擎的 wheel 文件可从
[GitHub Releases](https://github.com/harrylyx/newt/releases) 下载。选择与您的操作系统和 Python 版本匹配的 wheel 文件：

```bash
pip install newt-<版本号>-<平台>.whl
```

预编译 wheel 支持的平台：

| 操作系统 | 架构 |
|---|---|
| macOS | arm64 |
| Windows | x86_64 |
| Linux | x86_64, arm64 |

从官方 wheel 安装时无需安装 Rust 工具链。

### 高性能原生引擎 (Rust)

Newt 包含一个高性能的 Rust 扩展，用于加速核心计算路径，包括单特征 IV、批量 IV、PSI、ChiMerge、逐步回归特征选择以及优化的批量逻辑回归拟合。从官方 wheel 安装后可直接使用：

```python
from newt.features.analysis import calculate_batch_iv, calculate_iv

# 默认：engine="auto"（优先 Rust，不可用时回退 Python）
single = calculate_iv(df, target="target", feature="age")
batch = calculate_batch_iv(X, y)

# 强制使用 Rust
single_rust = calculate_iv(df, target="target", feature="age", engine="rust")
batch_rust = calculate_batch_iv(X, y, engine="rust")

# 显式使用 Python 回退实现
single_py = calculate_iv(df, target="target", feature="age", engine="python")
batch_py = calculate_batch_iv(X, y, engine="python")
```

如果 Rust 扩展不可用（例如从源码安装且没有 Rust 工具链），`engine="auto"` 会自动回退到 Python；显式请求 `engine="rust"` 会抛出明确的 `ImportError` 并附带安装指引。Python 实现始终可通过 `engine="python"` 使用。

各组件的默认引擎并不完全相同：

- `calculate_iv` / `calculate_batch_iv`：默认 `engine="auto"`。
- `FeatureSelector` / `FeatureAnalyzer`：默认 `engine="auto"`。
- `Report`：默认 `engine="auto"`。
- `StepwiseSelector`：默认 `engine="auto"`。

以上接口里，`engine="rust"` 是严格模式：如果本机没有可用的原生扩展，会直接报错。

### `opt` 方法的可选依赖范围

`newt[optbinning]` 这组可选依赖目前仅适用于 Python `<3.12`（受上游依赖限制）。
这不影响 Newt 核心包的支持范围，核心仍支持 Python `3.8.5` 到 `3.12.x`。

### 仓库内默认环境分工

在本仓库开发时，建议按用途固定两个环境：

- `.venv`（Python `3.8.5`）：开发、测试、lint、打包修改。
- `.venv-benchmark-3.10`（Python `3.10.19`）：运行 `newt-benchmark` 和 `toad` 对齐验证。

## 目录

1. [特征分箱](#1-特征分箱)
2. [特征选择](#2-特征选择)
3. [分箱特征分析](#3-分箱特征分析)
4. [逻辑回归建模](#4-逻辑回归建模)
5. [评分卡生成](#5-评分卡生成)
6. [完整流程](#6-完整流程)
7. [可视化](#7-可视化)
8. [手动调整](#8-手动调整)
9. [模型部署](#9-模型部署)
10. [评估指标](#10-评估指标)
11. [Excel 模型报告](#11-excel-模型报告)

---

## 1. 特征分箱

分箱的核心类是 `newt.Binner`（或 `newt.features.binning.Binner`）。它为各种分箱算法提供统一接口。

### 支持的分箱方法

- `'chi'`: ChiMerge（卡方分箱）- **默认**，有监督
- `'dt'`: 决策树分箱 - 有监督，寻找最优切分点
- `'opt'`: 最优分箱（约束优化）- 有监督，需要先安装
  `pip install "newt[optbinning]"`
- `'kmean'`: K-Means 聚类 - 无监督
- `'quantile'`: 等频分箱 - 无监督
- `'step'`: 等宽分箱 - 无监督

### 基础用法

```python
import pandas as pd
from newt import Binner

# 加载数据
df = pd.read_csv('data.csv')
target = 'target'  # 二分类目标变量（0/1）

# 初始化 Binner
binner = Binner()

# 拟合分箱模型
# 自动选择数值型列并使用 ChiMerge 进行分箱
binner.fit(df, y=target, method='chi', n_bins=5)

# 将数据转换为分箱编号（0, 1, 2...）
df_binned = binner.transform(df, labels=False)

# 将数据转换为区间字符串（如 '(-inf, 0.5]' 等）
df_labels = binner.transform(df, labels=True)

# 导出分箱规则
rules = binner.export()
# 输出示例：{'age': {'splits': [25.0, 40.0, 55.0], 'woe': {...}, 'iv': 0.42}, 'income': {'splits': [50000.0, 100000.0], 'woe': {...}, 'iv': 0.31}}
```

### 访问分箱结果

```python
# 使用 __getitem__ 访问特定特征的分箱结果
result = binner['age']

# 查看统计信息 DataFrame
print(result.stats)

# 绘制分箱结果图
result.plot()

# 获取 WOE 映射
print(result.woe_map())

# 推荐的手工调箱方式（不要直接操作内部 binners_）
print(binner.get_splits('age'))
binner.set_splits('age', [25.0, 40.0, 55.0])

# 遍历所有特征
for feat in binner:
    print(f"特征：{feat}")
    print(binner[feat].stats)
```

### 分箱特定功能

```python
# 仅对指定列进行分箱
binner.fit(df, y=target, method='dt', n_bins=5, cols=['age', 'income'])

# 加载预定义的规则
custom_rules = {'age': [30.0, 50.0, 70.0]}
binner.load(custom_rules)

# 获取某个特征的统计信息
stats = binner.stats('age')
```

### 单调分箱

单调分箱确保坏账率（事件率）在分箱间单调变化。

```python
# 从数据中自动检测单调趋势
binner.fit(df, y=target, method='chi', monotonic=True)

# 强制坏账率递增趋势
binner.fit(df, y=target, method='dt', monotonic='ascending')

# 强制坏账率递减趋势
# 需要先安装 optbinning 可选依赖
binner.fit(df, y=target, method='opt', monotonic='descending')
```

**注意**：单调调整使用 PAVA（Pool Adjacent Violators Algorithm）算法，通过合并分箱直至实现单调性。

### 使用单独的分箱类

```python
from newt.features.binning import (
    ChiMergeBinner,
    DecisionTreeBinner,
    OptBinningBinner,
    EqualWidthBinner,
    EqualFrequencyBinner,
    KMeansBinner
)

# 为单个特征创建分箱器
binner = ChiMergeBinner(n_bins=5)
binner.fit(df['age'], df[target])
bins = binner.transform(df['age'])
```

---

## 2. 特征选择

`newt.features.selection.FeatureSelector` 提供全面的特征分析和选择功能。

### 预过滤（基于 EDA）

```python
from newt.features.selection import FeatureSelector

# 使用所需指标初始化
selector = FeatureSelector(
    metrics=['iv', 'missing_rate', 'ks', 'correlation'],
    iv_bins=10,
    engine='auto'  # 默认：auto（优先 Rust，不可用时回退 Python）
)

# 计算统计量
selector.fit(df, df[target])

# 查看分析报告
print(selector.report())

# 查看相关性矩阵
print(selector.corr_matrix)

# 应用阈值进行过滤
selector.select(
    iv_threshold=0.02,           # 保留 IV >= 0.02 的特征
    missing_threshold=0.9,       # 移除缺失率 > 90% 的特征
    corr_threshold=0.8           # 移除高度相关的特征对
)

# 获取选中的特征
print(f"选中的特征：{selector.selected_features_}")

# 转换数据
X_filtered = selector.transform(df)
```

### 逐步特征筛选

`StepwiseSelector` 集成了基于 Rust 的**高性能并行引擎**。它能够利用 `Rayon` 技术在多个 CPU 核心上并行评估候选特征，相比传统的单线程实现（如 `statsmodels`），可获得 **20x-40x 的性能加速**。

```python
from newt.features.selection import StepwiseSelector

# 使用 auto 引擎初始化（默认：优先 Rust，不可用时回退 Python）
stepwise = StepwiseSelector(
    direction='both',    # 可选 'forward', 'backward', 或 'both'
    criterion='aic',     # 可选 'pvalue', 'aic', 或 'bic'
    p_enter=0.05,
    p_remove=0.10,
    engine='auto',       # 'auto' | 'rust' | 'python'
    verbose=True         # 显示 tqdm 进度条
)

# 拟合并转换数据（通常在转换后的数据上执行）
# 每一轮特征筛选都会显示实时进度条
X_selected = stepwise.fit_transform(X_transformed, y)

print(f"Selected features: {stepwise.selected_features_}")
```

关于性能提升的详细数据，请参阅 [逐步特征筛选性能基准](benchmarks/stepwise_performance.md)。

### 后过滤（基于模型）

```python
from newt.features.selection import PostFilter

# 使用 PSI 和 VIF 进行后过滤
postfilter = PostFilter(
    psi_threshold=0.25,   # 移除 PSI > 0.25 的特征
    vif_threshold=10.0    # 移除 VIF > 10.0 的特征
)

# 在训练数据上拟合，使用测试数据计算 PSI
X_filtered = postfilter.fit_transform(X_train_transformed, X_test_transformed)

print(f"因 PSI 移除：{postfilter.psi_removed_}")
print(f"因 VIF 移除：{postfilter.vif_removed_}")
```

---

## 3. 分箱特征分析

使用 `Binner` 查看分箱统计、生成后续建模所需的特征矩阵，并进入下游建模步骤。

### 查看特征统计

```python
binner = Binner()
binner.fit(df, y=target, method='chi')

result = binner['age']
print(result.stats)
result.plot()
```

### 为建模转换特征

```python
# 直接使用 Binner 转换所有已拟合特征
X_transformed = binner.woe_transform(df)
```

### 查看分箱汇总

```python
for feat in binner:
    print(feat)
    print(binner[feat].stats)
```

---

## 4. 逻辑回归建模

`newt.modeling.LogisticModel` 提供类似 scikit-learn 的接口用于逻辑回归，底层使用 statsmodels。

### 基础建模

```python
from newt.modeling import LogisticModel

# 初始化模型
model = LogisticModel(
    fit_intercept=True,
    method='bfgs',          # 优化方法
    maxiter=100
)

# 在转换后的特征上拟合
model.fit(X_transformed, df[target])

# 打印模型摘要（类似 R 的 summary()）
print(model.summary())

# 获取系数 DataFrame
coefs = model.get_coefficients()
print(coefs)

# 获取显著特征（p-value < 0.05）
sig_features = model.get_significant_features(p_threshold=0.05)
print(sig_features)
```

### 预测

```python
# 预测概率
y_pred_proba = model.predict_proba(X_transformed_test)

# 预测类别标签（默认阈值：0.5）
y_pred = model.predict(X_transformed_test)

# 自定义阈值
y_pred_custom = model.predict(X_transformed_test, threshold=0.3)
```

### 模型导出

```python
# 推荐：直接保存为 JSON
model.dump("logistic_model.json")

# 轻量恢复模型（不包含训练样本）
restored_model = LogisticModel.load("logistic_model.json")
```

`to_dict/from_dict` 仍保留用于兼容，但推荐优先使用 `dump/load`。

### 与 Scikit-learn 模型配合使用

```python
from sklearn.linear_model import LogisticRegression

# Scikit-learn 模型
lr = LogisticRegression()
lr.fit(X_transformed, y)

# 可直接与 Scorecard 配合使用
scorecard.from_model(lr, binner)
```

### 与 Statsmodels 配合使用

```python
import statsmodels.api as sm

# Statsmodels 模型
X_sm = sm.add_constant(X_transformed)
model_sm = sm.Logit(y, X_sm).fit()

# 可直接与 Scorecard 配合使用
scorecard.from_model(model_sm, binner)
```

---

## 5. 评分卡生成

`newt.modeling.Scorecard` 将逻辑回归系数转换为传统信用评分卡。

### 构建评分卡

```python
from newt.modeling import Scorecard

# 初始化评分卡
scorecard = Scorecard(
    base_score=600,     # 基准分数时的基准分
    pdo=50,             # 使赔率加倍的分数点数
    base_odds=1/15,     # 基准赔率（好/坏比率）
    points_decimals=1,  # 可选：限制分值/得分小数位
)

# 从拟合的模型和分箱器构建
scorecard.from_model(model, binner)
# 可选（调试场景）：保留原始 model/binner 的运行期引用
# scorecard.from_model(model, binner, keep_training_artifacts=True)

# 查看评分卡摘要
print(scorecard.summary())

# 导出完整评分卡
df_scorecard = scorecard.export()
print(df_scorecard)
```

### 计算分数

```python
# 为新数据计算分数
# 注意：X 是原始数据（未分箱，未进行预转换）
scores = scorecard.score(X_new)

print(f"分数：{scores}")
```

### 评分卡导出

```python
# 推荐：直接保存为 JSON
scorecard.dump("scorecard.json")

# 从 JSON 恢复
restored_scorecard = Scorecard.load("scorecard.json")
```

`to_dict/from_dict` 仍保留用于兼容，但推荐优先使用 `dump/load`。

`points_decimals=None`（默认）表示不做额外舍入，保持现有行为。

### 评分卡参数

评分卡分数使用以下公式计算：

```
分数 = Offset + Factor * ln(odds)
```

其中：
- `Offset = base_score - (pdo / ln(2)) * ln(base_odds)`
- `Factor = pdo / ln(2)`

---

## 6. 完整流程

`newt.pipeline.ScorecardPipeline` 为完整的评分卡开发工作流程提供流畅的 API。

### 完整流程示例

```python
from newt.pipeline import ScorecardPipeline

# 使用训练数据初始化流程
pipeline = (
    ScorecardPipeline(X_train, y_train, X_test, y_test)
    # 步骤 1：预过滤
    .prefilter(
        iv_threshold=0.02,
        missing_threshold=0.9,
        corr_threshold=0.8
    )
    # 步骤 2：分箱
    .bin(method='opt', n_bins=5)
    # 步骤 3：特征转换
    .woe_transform()
    # 步骤 4：后过滤
    .postfilter(
        psi_threshold=0.25,
        vif_threshold=10.0
    )
    # 步骤 5：逐步选择
    .stepwise(direction='both', criterion='aic')
    # 步骤 6：构建模型
    .build_model()
    # 步骤 7：生成评分卡
    .generate_scorecard(base_score=600, pdo=50, points_decimals=1)
)

# 为新数据打分
scores = pipeline.score(X_new)
```

### 访问中间结果

```python
# 预过滤报告
print(pipeline.prefilter_result.report())

# 分箱规则
print(pipeline.binner.export())

# 模型摘要
print(pipeline.model.summary())

# 评分卡摘要
print(pipeline.scorecard.summary())

# 流程摘要
print(pipeline.summary())
```

### 流程摘要

```python
# 获取完整的流程执行摘要
summary = pipeline.summary()
print(summary)
# 输出：
# Pre-filter: 50 特征 → 20 特征
# Binning: ChiMerge，5 个分箱
# WOE: 转换 20 个特征
# Post-filter: 移除 2 个特征（PSI）
# Stepwise: 选择 15 个特征（AIC）
# Model: 逻辑回归
# Scorecard: 基准分=600, PDO=50
```

---

## 7. 可视化

可视化模块提供各种绘图功能用于模型解释。

### 分箱可视化

```python
from newt.visualization import plot_binning_result

# 绘制单个特征的分箱结果
fig = plot_binning_result(
    binner=binner,
    X=df,
    y=df[target],
    feature='age',
    figsize=(12, 6)
)

import matplotlib.pyplot as plt
plt.show()
```

### IV 排名

```python
from newt.visualization import plot_iv_ranking

# 绘制按 IV 排名的 Top 特征
fig = plot_iv_ranking(
    iv_dict=pipeline.prefilter_result.eda_summary_.set_index('feature')['iv'].to_dict(),
    top_n=20,
    threshold=0.02
)
plt.show()
```

### PSI 对比

```python
from newt.visualization import plot_psi_comparison

# 绘制所有特征的 PSI 值
fig = plot_psi_comparison(
    psi_dict=pipeline.postfilter_result.psi_,
    threshold=0.25
)
plt.show()
```

---

## 8. 手动调整

有时自动分箱并不完美（例如，坏账率非单调、业务逻辑约束）。您可以手动调整切分点。

### 步骤 1：导出规则

```python
rules = binner.export()
# 输出示例：
# {
#    'age': {'splits': [25.0, 40.0, 55.0], 'woe': {...}, 'iv': 0.42},
#    'income': {'splits': [50000.0, 100000.0], 'woe': {...}, 'iv': 0.31}
# }
```

### 步骤 2：修改规则

编辑切分点列表：
- **合并分箱**：删除一个切分点
- **拆分分箱**：添加一个切分点

```python
# 示例：通过删除 '25.0' 合并 'age' 的前两个分箱
rules['age'] = [40.0, 55.0]

# 示例：为 'income' 添加切分点
rules['income'] = [30000.0, 75000.0, 150000.0]
```

### 步骤 3：加载并重新可视化

```python
# 更新现有 Binner
binner.load(rules)

# 或创建新的
binner_adjusted = Binner()
binner_adjusted.load(rules)

# 通过可视化验证
fig = plot_binning_result(binner_adjusted, df, df[target], 'age')
plt.show()
```

---

## 9. 模型部署

对评分卡满意后，您可以保存配置并在生产环境中使用。

### 保存组件

```python
import json

# 保存分箱规则
with open('binning_rules.json', 'w', encoding='utf-8') as f:
    json.dump(binner.export(), f, ensure_ascii=False)

# 保存轻量模型与评分卡快照
model.dump('model_params.json')
scorecard.dump('scorecard.json')
```

### 在生产环境中加载和使用

```python
# 加载分箱规则
with open('binning_rules.json', 'r', encoding='utf-8') as f:
    rules = json.load(f)

# 创建分箱器并加载规则
production_binner = Binner()
production_binner.load(rules)

# 直接加载评分卡
production_scorecard = Scorecard.load('scorecard.json')

# 可选：加载轻量 LogisticModel 用于审计或复现实验
production_model = LogisticModel.load('model_params.json')

# 为新数据打分
df_new = pd.read_csv('new_data.csv')
scores = production_scorecard.score(df_new)
```

`to_dict/from_dict` 仍保留用于兼容，但推荐优先使用 `dump/load` 进行持久化。

---

## 10. 评估指标

Newt 提供全面的信用风险评估指标。

### AUC（ROC 曲线下面积）

```python
from newt.metrics import calculate_auc

auc = calculate_auc(y_true, y_pred_proba)
print(f"AUC: {auc}")
```

### 基尼系数

```python
from newt.metrics import calculate_gini

gini = calculate_gini(y_true, y_pred_proba)
print(f"基尼系数: {gini}")
```

### KS 统计量

```python
from newt.metrics import calculate_ks

ks = calculate_ks(y_true, y_pred_proba)
print(f"KS: {ks}")
```

### 提升度分析

```python
from newt.metrics import calculate_lift, calculate_lift_at_k

# 按十分位数计算提升度表
lift_df = calculate_lift(y_true, y_pred_proba, bins=10)
print(lift_df)

# 计算 Top K 的提升度（例如 Top 10%）
lift_at_10pct = calculate_lift_at_k(y_true, y_pred_proba, k=0.1)
print(f"Lift@10%: {lift_at_10pct}")
```

### PSI（群体稳定性指标）

```python
from newt.metrics import (
    calculate_feature_psi_against_base,
    calculate_grouped_psi,
    calculate_psi,
    calculate_psi_batch,
)

# 计算训练集和测试集之间的 PSI
psi_values = calculate_psi(X_train_transformed, X_test_transformed)
print(f"PSI 值：{psi_values}")

# 计算单个特征的 PSI
psi_single = calculate_psi(
    X_train_transformed['age'],
    X_test_transformed['age'],
    buckets=10
)
print(f"age 的 PSI：{psi_single}")

# 一个基准分布，对多个对比分布批量计算 PSI
psi_batch = calculate_psi_batch(
    expected=X_train_transformed["age"],
    actual_groups=[X_test_transformed["age"], X_oot_transformed["age"]],
    engine="rust",
)
print(f"批量 PSI：{psi_batch}")

# 按月分组 PSI：每个 tag 内用最新月做基准
monthly_psi = calculate_grouped_psi(
    data=score_frame,
    group_cols=["month"],
    score_col="score",
    partition_cols=["tag"],
    reference_mode="latest",
    reference_col="month",
    engine="rust",
)
print(monthly_psi.head())

# 业务函数：指定 base 切片后，批量计算特征 PSI
feature_psi = calculate_feature_psi_against_base(
    data=score_frame,
    feature_cols=["f1", "f2", "f3"],
    base_col="month",
    base_value="202403",
    compare_col="month",
    compare_values=["202401", "202402", "202403"],
    engine="rust",
)
print(feature_psi.head())
```

### VIF（方差膨胀因子）

```python
from newt.metrics import calculate_vif

vif_values = calculate_vif(X_transformed)
print(vif_values)
```

### 指标解读指南

| 指标 | 优秀 | 可接受 | 较差 |
|--------|------|------------|------|
| AUC | > 0.75 | 0.70 - 0.75 | < 0.70 |
| 基尼系数 | > 0.50 | 0.40 - 0.50 | < 0.40 |
| KS | > 40 | 30 - 40 | < 30 |
| PSI | < 0.10 | 0.10 - 0.25 | > 0.25 |
| VIF | < 5 | 5 - 10 | > 10 |
| Lift@10% | > 3.0 | 2.0 - 3.0 | < 2.0 |

---

## 完整示例工作流程

```python
import pandas as pd
from newt import Binner
from newt.features.selection import FeatureSelector, PostFilter
from newt.modeling import LogisticModel, Scorecard
from newt.pipeline import ScorecardPipeline
from newt.visualization import plot_binning_result, plot_iv_ranking
from newt.metrics import calculate_auc, calculate_ks

# 1. 加载数据
df = pd.read_csv('credit_data.csv')
target = 'default'

# 2. 特征选择
selector = FeatureSelector()  # 默认 engine='auto'
selector.fit(df, df[target])
selector.select(iv_threshold=0.02)
X = selector.transform(df)

# 3. 带单调约束的分箱
binner = Binner()
binner.fit(X, df[target], method='opt', n_bins=5, monotonic=True)

# 访问分箱结果
binner['age'].stats
binner['age'].plot()
X_binned = binner.transform(X, labels=False)

# 4. 特征转换
X_transformed = binner.woe_transform(X)

# 5. 后过滤
postfilter = PostFilter()
X_transformed = postfilter.fit_transform(X_transformed, X_transformed_test)

# 6. 模型构建
model = LogisticModel()
model.fit(X_transformed, df[target])
print(model.summary())

# 7. 评分卡生成
scorecard = Scorecard(base_score=600, pdo=50)
scorecard.from_model(model, binner)
print(scorecard.summary())

# 8. 为新数据打分
df_new = pd.read_csv('new_applications.csv')
scores = scorecard.score(df_new)
print(f"分数：{scores}")

# 9. 计算指标
auc = calculate_auc(df[target], model.predict_proba(X_transformed))
ks = calculate_ks(df[target], model.predict_proba(X_transformed))
print(f"AUC: {auc}, KS: {ks}")

# 10. 可视化
fig = plot_iv_ranking(
    iv_dict=selector.eda_summary_.set_index('feature')['iv'].to_dict()
)
```

---

## 11. Excel 模型报告

`newt.Report` 可以基于已有样本、模型对象和分数字段生成多 sheet 的 Excel 模型报告，支持总览、模型设计、变量分析、模型表现，并会根据输入条件生成可选附录页（分维度对比、新老模型对比、金额指标、画像变量）。

```python
from newt import Report

feature_df = pd.read_csv("./feature_dict.csv")

report = Report(
    data=data,
    model=model,
    tag="tag",
    score_col="score_new",
    date_col="obs_date",
    label_list=["target"],
    score_list=["score_old"],
    dim_list=["channel"],
    var_list=["age", "income"],
    prin_bal_amount_col="prin_bal_amount",  # 可选；需与 loan_amount_col 成对传入
    loan_amount_col="loan_amount",          # 可选
    feature_df=feature_df,                  # 可选
    report_out_path="./out/model_report.xlsx",
    engine="auto",           # 默认：优先 Rust，不可用时回退 Python
    max_workers=8,           # 默认: min(8, cpu_count)
    parallel_sheets=True,    # 默认
    memory_mode="compact",   # 默认: "compact" | "standard"
    metrics_mode="exact",    # 默认: "exact" | "binned"
)

report.generate()
```

常用说明：

- `data` 需要已经包含 `tag`、`score_col`、`date_col`、`label_list` 以及你额外传入的 `score_list`、`dim_list`、`var_list` 对应列
- `model` 只用于提取模型参数和变量重要性，不负责重新打分
- `tag` 是样本分组列，通常取 `train` / `test` / `oot` / `oos`
- `score_col` 是新模型的主分数字段，报表里的“主模型”都围绕它展开
- `date_col` 会被转成 `_report_month`，格式统一成 `YYYYMM`，用于按月统计
- `label_list` 是标签列列表，支持多个标签；第一个标签会作为主标签
- `score_list` 是老模型或对照分数字段，用于和主模型做对比
- `dim_list` 是分维度对比字段，会在“分维度对比”附录页按维度拆表展示，并同步到总览页
- `var_list` 是画像变量列表，会在“画像变量”附录页按变量拆表展示，并同步到总览页；它不是变量分析页的开关
- `feature_df` 是可选的特征字典 DataFrame；如果你当前是文件存储，可先 `pd.read_csv(...)` 读成 DataFrame 再传入。建议按下面 4 列准备（表头名请保持一致）：

| 变量英文名 | 变量中文名 | 变量来源 | 变量指标表英文名 |
|---|---|---|---|
| `英文名` | `中文名` | `来源` | `指标表英文名` |
| `thirdparty_info_period1_6` | `近6个月三方查询次数` | `thirdparty` | `thirdparty_info` |

  兼容说明：如果历史字典仍使用 `表名`，报表会自动映射到 `指标表英文名`。
- `sheet_list` 可选传入序号 `1-5` 或名称来控制输出页面：
  `1=overview`、`2=model_design`、`3=variable_analysis`、`4=model_performance`、`5=scorecard_details`
- 名称可选值包括：
  `总览`、`模型设计`、`变量分析`、`模型表现`、`评分卡计算明细`、`分维度对比`、`新老模型对比`、`金额指标`、`画像变量`
- 不传 `sheet_list` 时，会输出当前输入条件下“可用”的全部页面（是否可用取决于 OOT/tag 覆盖、对比分数字段、模型类型、金额列等）
- `engine` 控制报表计算引擎：`auto`（默认）、`rust` 或 `python`
- `max_workers` 控制并行计算线程数；默认是 `min(8, cpu_count)`
- `parallel_sheets` 控制是否并行计算各个 sheet（Excel 写入仍是串行）
- `memory_mode` 控制内存策略：`compact`（默认）或 `standard`。Compact 模式通过使用下采样类型和优化的按月转换逻辑，显著降低处理千万级数据时的内存占用。
- `metrics_mode` 控制指标计算模式：`exact`（默认）或 `binned`（更快、近似）
- `prin_bal_amount_col` 与 `loan_amount_col` 可成对传入，用于输出金额维度指标，并启用可选附录页 `金额指标`
- 如果你只想看某一部分报表，可以只传对应的 sheet 名称或编号
- 跑报表开发和验收时，建议使用 `uv sync --group dev`

外部配置加载（可选）：

```python
from newt import load_conf

load_conf("./newt_conf.json")
```

`load_conf` 会把外部文件中的配置覆盖到运行时（支持 `.json` / `.toml` / `.yaml` / `.yml`）。

---

## 12. 在 Jupyter 中进行单表交互式分析

除了生成完整的 Excel 报告外，您还可以在 Jupyter Notebook 等交互式环境中直接将各个报表模块作为 Pandas DataFrame 生成。API 会自动处理内部转换（如将日期标准化为月份列）。

### 12.1. 按 Tag 与 按月模型效果拆分

计算按 `tag`（例如 train、oot）和按 `month` 分组的模型表现指标。

```python
import pandas as pd
from newt import calculate_split_metrics

# 返回两个 DataFrame：一个按 tag 分组，一个按 month 分组
tag_metrics, month_metrics = calculate_split_metrics(
    data=df,
    tag_col='tag',
    date_col='obs_date',      # 只需要传入日期列
    label_list=['target'],
    score_col='xgb_score',
    model_name='XGBoost Model',
    score_type='probability',  # 'probability'（值越大风险越高）或 'score'（值越大风险越低）
    prin_bal_amount_col='prin_bal_amount',  # 可选；需与 loan_amount_col 成对传入
    loan_amount_col='loan_amount',          # 可选
)

print(tag_metrics)
print(month_metrics)
```

说明：

- 核心输出指标始终是人头口径：
  `总, 好, 坏, 坏占比, KS, AUC, 10%lift, 5%lift, 2%lift, 1%lift`。
- 当 `prin_bal_amount_col` 与 `loan_amount_col` 成对传入时，会追加金额扩展指标：
  `放款金额, 逾期本金, 金额坏占比, 金额AUC, 金额KS, 10%金额lift, 5%金额lift, 2%金额lift, 1%金额lift`。

### 12.2. 分维度对比

比较按自定义维度拆分的指标（通常仅适用于 OOT 样本）。

```python
from newt import calculate_dimensional_comparison

# 建议仅传入 OOT 数据进行此分析
dim_metrics = calculate_dimensional_comparison(
    data=df[df['tag'] == 'oot'],
    dim_list=['channel', 'education'],
    label_list=['target'],
    score_model_columns=[('XGBoost Model', 'xgb_score')],
    score_type='probability',
    prin_bal_amount_col='prin_bal_amount',  # 可选；需成对传入
    loan_amount_col='loan_amount',          # 可选
)

print(dim_metrics)
```

说明：

- 仅当金额列成对传入时，才会追加金额扩展指标：
  `放款金额, 逾期本金, 金额坏占比, 金额AUC, 金额KS, 10%金额lift, 5%金额lift, 2%金额lift, 1%金额lift`。

### 12.3. 新老模型对比

按 `tag` 或 `month` 对比新模型与旧基准模型的相对表现。

```python
from newt import calculate_model_comparison

# 两个模型之间的正面交锋对比
compare_metrics = calculate_model_comparison(
    data=df,
    tag_col='tag',
    date_col='obs_date',
    label_list=['target'],
    model_columns=[
        ('New XGBoost Model', 'xgb_score'),
        ('Old Logistic Model', 'old_score')
    ],
    group_mode='month',  # 也接受 'tag'
    score_type='probability',
    prin_bal_amount_col='prin_bal_amount',  # 可选；需成对传入
    loan_amount_col='loan_amount',          # 可选
)

print(compare_metrics)
```

说明：

- 仅当金额列成对传入时，才会追加金额扩展指标：
  `放款金额, 逾期本金, 金额坏占比, 金额AUC, 金额KS, 10%金额lift, 5%金额lift, 2%金额lift, 1%金额lift`。

### 12.4. 分箱指标

直接计算分箱层面的模型指标，并可选输出金额维度指标。

```python
from newt import calculate_bin_metrics

# 分位数分箱（默认 q=10）
bin_metrics = calculate_bin_metrics(
    data=df,
    label_col='target',
    score_col='xgb_score',
    q=10,
    prin_bal_amount_col='prin_bal_amount',  # 可选；需成对传入
    loan_amount_col='loan_amount',          # 可选
)

# 自定义分箱边界
custom_bin_metrics = calculate_bin_metrics(
    data=df,
    label_col='target',
    score_col='xgb_score',
    bins=[-float('inf'), 0.2, 0.5, 0.8, float('inf')],
)
```

说明：

- `calculate_bin_metrics` 保持旧版金额列行为。
- 当金额列成对传入时，追加金额列为：
  `逾期本金, 放款金额, 金额坏占比, 放款金额占比, 逾期本金占比, 金额lift`。

---

## 其他资源

- **API 文档**：查看各模块的文档字符串获取详细 API 参考
- **示例**：查看 `examples/` 目录获取更全面的示例
- **配置**：查看 `newt.config` 获取默认参数值
- **开发指南**：查看 `AGENTS.md` 获取开发指南
