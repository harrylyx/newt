# Newt 用户指南

本指南涵盖使用 `newt` 进行信用评分卡开发的端到端工作流程，包括分箱、特征选择、WOE/IV 分析、建模和评分卡生成。

## 目录

1. [特征分箱](#1-特征分箱)
2. [特征选择](#2-特征选择)
3. [WOE & IV 分析](#3-woe--iv-分析)
4. [逻辑回归建模](#4-逻辑回归建模)
5. [评分卡生成](#5-评分卡生成)
6. [完整流程](#6-完整流程)
7. [可视化](#7-可视化)
8. [手动调整](#8-手动调整)
9. [模型部署](#9-模型部署)
10. [评估指标](#10-评估指标)

---

## 1. 特征分箱

分箱的核心类是 `newt.Binner`（或 `newt.features.binning.Binner`）。它为各种分箱算法提供统一接口。

### 支持的分箱方法

- `'chi'`: ChiMerge（卡方分箱）- **默认**，有监督
- `'dt'`: 决策树分箱 - 有监督，寻找最优切分点
- `'opt'`: 最优分箱（约束优化）- 有监督，需要 `optbinning`
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
# 输出示例：{'age': [25.0, 40.0, 55.0], 'income': [50000.0, 100000.0]}
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
    iv_bins=10
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

### 逐步选择

```python
from newt.features.selection import StepwiseSelector

# 初始化逐步选择器
stepwise = StepwiseSelector(
    direction='both',    # 'forward'（向前）、'backward'（向后）或 'both'（双向）
    criterion='aic',     # 'pvalue'、'aic' 或 'bic'
    p_enter=0.05,
    p_remove=0.10
)

# 拟合并转换（通常在 WOE 转换后的数据上）
X_selected = stepwise.fit_transform(X_woe, y)

print(f"选中的特征：{stepwise.selected_features_}")
```

### 后过滤（基于模型）

```python
from newt.features.selection import PostFilter

# 使用 PSI 和 VIF 进行后过滤
postfilter = PostFilter(
    psi_threshold=0.25,   # 移除 PSI > 0.25 的特征
    vif_threshold=10.0    # 移除 VIF > 10.0 的特征
)

# 在训练数据上拟合，使用测试数据计算 PSI
X_filtered = postfilter.fit_transform(X_train_woe, X_test_woe)

print(f"因 PSI 移除：{postfilter.psi_removed_}")
print(f"因 VIF 移除：{postfilter.vif_removed_}")
```

---

## 3. WOE & IV 分析

`newt.features.analysis.WOEEncoder` 处理证据权重（WOE）和信息值（IV）的计算。

### 基础 WOE 编码

```python
from newt.features.analysis import WOEEncoder

# 在分箱数据上拟合
encoder = WOEEncoder(epsilon=1e-8)
encoder.fit(df_binned['feature_name'], df[target])

# 获取 IV
print(f"IV: {encoder.iv_}")

# 获取汇总统计（好/坏分布、WOE、IV 贡献）
print(encoder.summary_)

# 转换为 WOE 值
df_woe = encoder.transform(df_binned['feature_name'])

# 获取 WOE 映射字典
print(encoder.woe_map_)
```

### 批量 WOE 转换

```python
# 对所有分箱特征应用 WOE
X_woe = df_binned.copy()
woe_encoders = {}

for col in df_binned.columns:
    encoder = WOEEncoder()
    encoder.fit(df_binned[col].astype(str), df[target])
    woe_encoders[col] = encoder
    X_woe[col] = encoder.transform(df_binned[col].astype(str))
```

### 使用 Binner 的 WOE 存储

```python
# WOE 编码器自动存储在 binner.woe_storage 中
binner = Binner()
binner.fit(df, y=target, method='chi')

# 访问特定特征的 WOE 编码器
encoder = binner.woe_storage.get('age')
print(encoder.woe_map_)

# 访问所有编码器
print(binner.woe_encoders_)
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

# 在 WOE 转换后的特征上拟合
model.fit(X_woe, df[target])

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
y_pred_proba = model.predict_proba(X_woe_test)

# 预测类别标签（默认阈值：0.5）
y_pred = model.predict(X_woe_test)

# 自定义阈值
y_pred_custom = model.predict(X_woe_test, threshold=0.3)
```

### 模型导出

```python
# 将模型参数导出为字典
model_dict = model.to_dict()
print(model_dict)
# {'intercept': -2.5, 'coefficients': {'age': 0.3, 'income': 0.5}, ...}
```

### 与 Scikit-learn 模型配合使用

```python
from sklearn.linear_model import LogisticRegression

# Scikit-learn 模型
lr = LogisticRegression()
lr.fit(X_woe, y)

# 可直接与 Scorecard 配合使用
scorecard.from_model(lr, binner, woe_encoders)
```

### 与 Statsmodels 配合使用

```python
import statsmodels.api as sm

# Statsmodels 模型
X_sm = sm.add_constant(X_woe)
model_sm = sm.Logit(y, X_sm).fit()

# 可直接与 Scorecard 配合使用
scorecard.from_model(model_sm, binner, woe_encoders)
```

---

## 5. 评分卡生成

`newt.modeling.Scorecard` 将基于 WOE 的逻辑回归系数转换为传统信用评分卡。

### 构建评分卡

```python
from newt.modeling import Scorecard

# 初始化评分卡
scorecard = Scorecard(
    base_score=600,     # 基准分数时的基准分
    pdo=50,             # 使赔率加倍的分数点数
    base_odds=1/15      # 基准赔率（好/坏比率）
)

# 从拟合的模型、分箱器和 WOE 编码器构建
scorecard.from_model(model, binner, woe_encoders)

# 查看评分卡摘要
print(scorecard.summary())

# 导出完整评分卡
df_scorecard = scorecard.export()
print(df_scorecard)
```

### 计算分数

```python
# 为新数据计算分数
# 注意：X 是原始数据（未分箱，未进行 WOE 转换）
scores = scorecard.score(X_new)

print(f"分数：{scores}")
```

### 评分卡导出

```python
# 导出为字典
scorecard_dict = scorecard.to_dict()
```

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
    # 步骤 3：WOE 转换
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
    .generate_scorecard(base_score=600, pdo=50)
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
    woe_encoder=woe_encoders.get('age'),
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

### WOE 模式

```python
from newt.visualization import plot_woe_pattern

# 绘制特征的 WOE 模式
fig = plot_woe_pattern(
    woe_encoder=woe_encoders['age'],
    feature='age'
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
#    'age': [25.0, 40.0, 55.0],
#    'income': [50000.0, 100000.0]
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

# 保存模型参数
with open('model_params.json', 'w', encoding='utf-8') as f:
    json.dump(model.to_dict(), f, ensure_ascii=False)

# 保存评分卡
with open('scorecard.json', 'w', encoding='utf-8') as f:
    json.dump(scorecard.to_dict(), f, ensure_ascii=False)
```

### 在生产环境中加载和使用

```python
# 加载分箱规则
with open('binning_rules.json', 'r', encoding='utf-8') as f:
    rules = json.load(f)

# 创建分箱器并加载规则
production_binner = Binner()
production_binner.load(rules)

# 加载模型参数并重建
with open('model_params.json', 'r', encoding='utf-8') as f:
    model_params = json.load(f)

# 加载 WOE 编码器（需要单独保存/加载）
# ... 加载 woe_encoders 字典 ...

# 重建评分卡
production_scorecard = Scorecard(
    base_score=600,
    pdo=50,
    base_odds=1/15
)
production_scorecard.from_model(model_params, production_binner, woe_encoders)

# 为新数据打分
df_new = pd.read_csv('new_data.csv')
scores = production_scorecard.score(df_new)
```

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
from newt.metrics import calculate_psi

# 计算训练集和测试集之间的 PSI
psi_values = calculate_psi(X_train_woe, X_test_woe)
print(f"PSI 值：{psi_values}")

# 计算单个特征的 PSI
psi_single = calculate_psi(
    X_train_woe['age'],
    X_test_woe['age'],
    buckets=10
)
print(f"age 的 PSI：{psi_single}")
```

### VIF（方差膨胀因子）

```python
from newt.metrics import calculate_vif

vif_values = calculate_vif(X_woe)
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
from newt.features.analysis import WOEEncoder
from newt.modeling import LogisticModel, Scorecard
from newt.pipeline import ScorecardPipeline
from newt.visualization import plot_binning_result, plot_iv_ranking
from newt.metrics import calculate_auc, calculate_ks

# 1. 加载数据
df = pd.read_csv('credit_data.csv')
target = 'default'

# 2. 特征选择
selector = FeatureSelector()
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

# 4. WOE 转换
X_woe = X_binned.copy()
woe_encoders = {}
for col in X_binned.columns:
    encoder = WOEEncoder()
    encoder.fit(X_binned[col].astype(str), df[target])
    woe_encoders[col] = encoder
    X_woe[col] = encoder.transform(X_binned[col].astype(str))

# 5. 后过滤
postfilter = PostFilter()
X_woe = postfilter.fit_transform(X_woe, X_woe_test)

# 6. 模型构建
model = LogisticModel()
model.fit(X_woe, df[target])
print(model.summary())

# 7. 评分卡生成
scorecard = Scorecard(base_score=600, pdo=50)
scorecard.from_model(model, binner, woe_encoders)
print(scorecard.summary())

# 8. 为新数据打分
df_new = pd.read_csv('new_applications.csv')
scores = scorecard.score(df_new)
print(f"分数：{scores}")

# 9. 计算指标
auc = calculate_auc(df[target], model.predict_proba(X_woe))
ks = calculate_ks(df[target], model.predict_proba(X_woe))
print(f"AUC: {auc}, KS: {ks}")

# 10. 可视化
fig = plot_iv_ranking(
    iv_dict=selector.eda_summary_.set_index('feature')['iv'].to_dict()
)
```

---

## 其他资源

- **API 文档**：查看各模块的文档字符串获取详细 API 参考
- **示例**：查看 `examples/` 目录获取更全面的示例
- **配置**：查看 `newt.config` 获取默认参数值
- **开发指南**：查看 `AGENTS.md` 获取开发指南
