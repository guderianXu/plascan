# PlaScan Changelog

本文件按版本倒序记录用户可感知的主要变更。详细验证记录见 `docs/releases/`。

## v1.1.0-alpha.2 - 2026-06-13

### 新增

- 增加 vcpkg manifest、CMake presets 和 `scripts/env/` 环境脚本，用于跨平台准备 vcpkg、Python、LibTorch 和 CPack 打包环境。
- 增加固定相机足迹重叠配对路径，SFM 在大规模已知相机数据上可优先使用相机投影足迹裁剪候选匹配对。

### 优化

- 已知相机 SFM 默认使用参考球面足迹重叠分析裁剪匹配对；不可用时再回退到顺序窗口和相机中心邻域。
- SFM 和 GUI 特征匹配对内点不足的日志做采样汇总，避免大量无效对刷屏。
- 相机三维视图在密集相机场景下按视口和相机数量限制标签数量，降低相机名称堆叠。
- Windows/CPack 打包和运行时安装规则得到补充。

### 修复

- 修复固定相机数据仅按相机中心选邻居导致错误匹配候选过多的问题。
- 修复大相机集合中相机名称显示过密的问题。

### 验证

- `cmake --build build -j$(nproc)` 通过。
- `ctest --test-dir build --output-on-failure -R 'SfmPairPlanner|OverlapAnalyzer|GuiProjectUtils|gui_project_utils'` 通过，8/8。
- `./tests/test_gui_project_utils` 通过，76/76。
- `python -m py_compile scripts/env/*.py` 通过。
- agisoft aerial GCP 48 张子集 SFM-only 验证通过：48/48 注册，7835 个稀疏点，平均重投影误差 0.75 px。

### 已知问题

- agisoft aerial GCP 全量 444 张完整重建仍需要更长时间的端到端验证；本轮只完成了 48 张子集的 SFM 验证。
- 全量 `ctest` 的历史已知问题仍需单独确认：`TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 可能出现 `dom_png not found`。

## v1.1.0-alpha.1 - 2026-06-13

### 摘要

- 改进三维重建工作流、相机转换、TorchScript 模型、GUI 对话框迁移、重叠对获取和摄影测量测试数据适配。

## v1.0.0 - 历史版本

### 摘要

- PlaScan 早期稳定基线版本。
