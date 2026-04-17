# Newt (中文)

Newt 是一个轻量级的 Python 工具包，专为信用风险建模中的特征分析和统计诊断而设计。

## 核心特性

- **6 种分箱算法**：ChiMerge, Decision Tree, K-Means, 等频, 等宽, Optimal。
- **单调性支持**：完善的单调分箱支持（递增、递减、自动检测）。
- **WOE/IV 分析**：稳健的 WOE 编码、批量 WOE 转换和 IV (Information Value) 计算。
- **特征筛选**：流水线式的特征筛选（IV, PSI, VIF, 逐步回归）。
- **评分卡生成**：端到端的评分卡生成和评分。
- **高性能**：由 Rust 驱动的高性能原生引擎，加速核心计算（分箱、特征选择、IV、PSI）。

## 安装

```bash
pip install newt
```

## 快速开始

- [用户指南](user_guide.zh.md)
- [API 参考](api/reference.zh.md)

推荐流程：先拟合 `Binner`，再调用 `binner.woe_transform(X)`，最后用 `Scorecard.from_model(model, binner)` 生成评分卡。
## 性能基准 (Benchmarks)
- [与 Toad 性能对比](benchmarks/metric_vs_toad.md)
- [PSI 计算性能](benchmarks/psi_performance.md)
- [逐步特征筛选性能](benchmarks/stepwise_performance.md)
- [ChiMerge 分箱性能](benchmarks/chimerge_performance.md)
