# PlaScan Changelog

本文件按版本倒序记录用户可感知的主要变更。详细验证记录见 `docs/releases/`。

## v1.1.2 - 2026-06-17

### 新增

- 新增 LiDAR / 激光点摄影测量数据集整理文档，覆盖 MUN-FRL、UseGeo、H3D、MARS-LVIG、UAVScenes、NTU VIRAL、DublinCity、DFC 2019 / US3D、ETH3D 等候选数据，用于后续激光点参与 BA、相机-LiDAR 外参验证、MVS/DEM/DOM 质量检查。
- 扩展摄影测量测试数据下载器，增加 `workflow_tags` 和 `--workflow-tag` 选择能力，可按 `ba_constraint_candidate`、`lidar_fusion_validation` 等用途筛选 LiDAR 相邻数据集并生成手动下载 manifest。

### 优化

- 优化 MVS 深度图估计的 CPU 前处理链路：缓存源帧共视稀疏点、并行构建可见性缓存、源帧角度打分提前停止，减少大规模航测数据在进入 CUDA PatchMatch 前的等待时间。
- 优化稀疏 hint 构建：投影稀疏样本只做单次可见点遍历，深度分位采样有界，粗层 hint 使用 OpenCV distance transform 限距离传播，精层仅叠加 seed，避免全图传播把错误深度强行铺满。
- 优化 MVS 支撑掩码与 dense refine：稀疏支撑改为软约束，支撑掩码按精层 PatchMatch 工作尺寸生成，大点云 refine 对输入和内联过滤做上限保护，避免 GUI/CLI 在百万级点云上长时间阻塞。
- 优化 CUDA 灰度图处理路径，避免重复 reference resize，并保留 GPU 灰度图缓存命中统计，便于观察 GPU 利用率不连续时是否被上传/缩放拖慢。

### 修复

- 修复 MVS 后处理中过强稀疏支撑裁剪可能导致有效深度被硬清空的问题，改为置信度软缩放并保留深度。
- 修复深度图局部离群噪点缺少后处理保护的问题，增加局部深度离群过滤和阶段耗时统计，便于定位 `source/range/hint/patchmatch/filter` 瓶颈。
- 修复大规模 dense refine 连续执行第二轮 SOR 或内联大点云过滤导致的卡顿风险。

### 验证

- `python -m unittest tests.test_mvs_scheduler_config tests.test_reconstruct_pipeline_cli tests.test_repo_hygiene tests.test_download_photogrammetry_testdata` 通过，65 个测试，3 个跳过。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Jobs 8` 通过，使用 Windows 原生 MSVC/Ninja、CUDA 13.1、libtorch-cu130 和 `build\windows-vcpkg-cuda-release`。
- `build\windows-vcpkg-cuda-release\tests\test_mvs_pipeline.exe --gtest_brief=1` 通过，17/17。
- `build\windows-vcpkg-cuda-release\tests\test_mvs_types.exe --gtest_brief=1` 通过，16/16。
- `build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_brief=1` 通过，133/133。
- GitHub Actions `build-test` 作为远端门禁；`v1.1.2` 发布提交和 tag 推送后需以 Actions 结果为准。

### 已知问题

- 本轮重点为 MVS/CUDA 前处理与数据集准备，未重新运行 agisoft aerial GCP 444 张完整 MVS/mesh/DEM/DOM 长时流水线。
- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`；该测试可能出现 `dom_png not found`，后续应单独修复 Terrain/DEM/DOM 输出登记或测试数据。

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
