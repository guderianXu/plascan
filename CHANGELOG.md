# PlaScan Changelog

本文件按版本倒序记录用户可感知的主要变更。详细验证记录见 `docs/releases/`。

## v1.1.1 - 2026-06-16

### 优化

- README 删除过时的“实测性能”表，重写环境构建说明，明确 Windows 原生构建、vcpkg preset、Python/LibTorch 环境脚本和 CPack 打包路径。
- 平台支持矩阵新增 Windows (NVIDIA)，同步说明 Windows/Linux/macOS 的 CUDA、CLI、Qt6 GUI 和打包支持范围。
- CI workflow 明确为 Linux CPU-only 构建，补齐 Qt OpenGLWidgets 与 Python numpy 依赖，并显式关闭 conda/vcpkg 自动发现，降低 GitHub Actions 环境漂移。

### 修复

- 修复 CPU LibTorch 构建下 `SFMService` 无条件引用 CUDA-only `c10::cuda::CUDACachingAllocator` 的兼容风险。
- 修复 GitHub Actions Configure 阶段因缺少 Qt OpenGLWidgets 开发包而失败的配置问题。

### 验证

- 使用 VS BuildTools CMake/Ninja，并设置 vcpkg、LibTorch、CUDA 运行时 `PATH` 后，`plascan_gui` 与 `test_gui_project_utils` 构建通过。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe` 通过，126/126。

### 已知问题

- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`；该测试可能出现 `dom_png not found`，需要后续单独修复。
- 本轮重点为 CI/README/版本同步，没有重新跑 agisoft aerial GCP 444 张完整 MVS/mesh/DEM/DOM 长时流水线。

## v1.1.0-alpha.4 - 2026-06-15

### 优化

- 优化 Metashape 相机转换，`k1/k2/k3/p1/p2` 畸变参数会写入 PlaScan `.tsai`，避免空三使用被简化为零畸变的相机模型。
- 优化 SFM 对项目元数据相机的处理：GUI 项目元数据只作为增量 SfM/BA 初值，只有显式 `.tsai` 列表才进入固定外参直接三角化。
- 优化已知外参 SfM 质量门，若输入存在多视 track 但输出几乎全退化为两视图点云，会标记失败并给出可定位的日志。

### 修复

- 修复 Metashape aerial GCP 数据经旧转换器导出零畸变相机后，444 张影像空三点云和飞行轨迹明显异常的问题。
- 修复正式 SFM 结果质量元数据中 BA 状态被无条件标记为已应用的问题。

### 验证

- `test_camera_format_converter.exe` 通过，6/6。
- `test_sfm_pipeline.exe` 通过，31/31。
- `test_gui_project_utils.exe` 通过，113/113。
- `python -m unittest tests.test_camera_convert_cli` 通过，6/6。
- agisoft aerial GCP 444 张，CUDA + `--feature-max-image-dim 2048`，SFM-only CLI 验证通过：444/444 注册，211037 个稀疏点，平均重投影误差 1.00 px，`finalTwoViewRatio=0.4191`。

### 已知问题

- 本轮完成的是 SFM-only 验证，MVS/mesh/DEM/DOM 下游仍需单独长时验证。
- 全量 `ctest` 的历史已知问题仍需单独确认：`TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 可能出现 `dom_png not found`。

## v1.1.0-alpha.3 - 2026-06-15

### 优化

- 优化 GUI 切换影像时的响应性，影像读取和特征覆盖层加载改为后台结果回填，避免旧任务覆盖当前画布。
- 优化大规模特征点可视化，批量点层使用单个场景图元绘制，默认特征点颜色调整为蓝色。
- 优化空三前置检查，复用已生成的影像配对计划，并用匹配图连通性/覆盖率判断是否可进入 SfM，避免把未计划的全连接影像对误判为缺失。
- 优化已知相机 SFM 候选配对，在大规模已知位姿数据上使用有界邻域和生成的 pair plan，避免退化到全量 N² 匹配。

### 修复

- 修复 Windows 下保存 `.plascan` 项目时，归档只读句柄阻塞 libzip 临时文件替换而导致 `Renaming temporary file failed: Permission denied` 的问题。
- 修复空三/匹配流程中进度、取消和 UI 线程阻塞相关路径，减少开始空三或检查匹配文件时界面卡死。
- 修复旧式两视图预览稀疏云被误当作正式 SfM 结果继续下游流程的问题，并增强稀疏结果质量元数据。

### 验证

- `cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target plascan_gui -j 8` 通过。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_project_data.exe` 通过，8/8。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe` 通过，113/113。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_sfm_pair_planner.exe` 通过，8/8。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_layer_renderer_batched_overlay.exe` 通过，2/2。
- `ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R 'project_data|gui_project_utils|sfm_pair_planner|layer_renderer'` 匹配并通过 `test_layer_renderer_batched_overlay`，1/1。

### 已知问题

- 本轮完成的是 GUI/SfM 流程和保存路径的回归验证，agisoft aerial GCP 444 张全量端到端空三/重建仍需单独长时验证。
- 全量 `ctest` 的历史已知问题仍需单独确认：`TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 可能出现 `dom_png not found`。

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
