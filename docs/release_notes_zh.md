# Newt 最近版本更新说明

本文汇总 `v0.1.1` 到 `v0.1.4` 的主要线上变更。

## v0.1.4（2026-04-04）

### CI/CD 与发布流程

- Linux wheel 构建拆分为两个原生任务：
  - `Linux x86_64`（`ubuntu-latest`）
  - `Linux aarch64`（`ubuntu-24.04-arm`）
- `Build Wheels` 与 `Release` 两个 workflow 均移除了 QEMU 仿真路径。
- wheel 版本矩阵保持为 `cp38` 到 `cp312`（对应 Python `3.8.5` 到 `3.12.x`）。

### 本次发布目的

- 之前 Linux ARM 构建依赖跨架构仿真，稳定性和排障效率都受影响。
- 改为原生 ARM runner 后，架构问题更容易隔离，发布链路更清晰。

## v0.1.3（2026-04-04）

### 打包与流程加固

- 增加了中间修复版本，用于解除 Linux `aarch64` 构建阻塞。
- 延续并固定 Python 支持区间 `>=3.8.5,<3.13`，wheel 只构建 `cp38-cp312`。

## v0.1.2（2026-04-04）

### 报告模块修复

- 统一了多个 sheet 的报告输出口径。
- 修复了报告中的分数方向处理与表格布局问题。
- benchmark 和报告文档与实际运行环境完成同步。

## v0.1.1（2026-04-04）

### IV 引擎与指标对齐

- `calculate_iv` 与 batch IV 改为默认使用 Rust 引擎。
- Python 路径保留，仅在显式指定 `engine="python"` 时使用。
- IV 计算与 `toad` 的平滑口径完成对齐。
- Rust 扩展补齐类别特征 IV，默认 Rust 模式可覆盖数值与类别特征。

### Benchmark 与兼容性策略

- benchmark 固定在 `.venv-benchmark-3.10` 环境直接运行，使用 `toad==0.1.5`。
- 移除了 benchmark 中额外 worker 虚拟环境回退逻辑。
- 官方 Python 支持区间明确为 `3.8.5` 到 `3.12.x`。

## v0.1.4 之后的当前基线

- **核心 Python 支持：** `>=3.8.5,<3.13`
- **wheel 目标：** `cp38`、`cp39`、`cp310`、`cp311`、`cp312`
- **默认本地环境：**
  - `.venv`（Python `3.8.5`）用于开发
  - `.venv-benchmark-3.10`（Python `3.10.19`）用于 benchmark
- **IV 默认引擎：** Rust（单特征与批量）
- **benchmark 对齐目标：** 在统一预处理输入上，Newt 与 toad 的 IV 差异接近 0
