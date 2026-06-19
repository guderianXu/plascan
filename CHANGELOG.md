# PlaScan Changelog

本文件按版本倒序记录用户可感知的主要变更。详细验证记录见 `docs/releases/`。

## v1.1.4 - 2026-06-19

### 新增

- `reconstruct_pipeline_cli` 新增 MVS 调试与资源控制参数，支持限制深度估计帧数、MVS 分辨率、迭代次数、置信度阈值、GPU/CPU frame worker 数，以及融合阶段最大图像边长。
- MVS 报告扩展记录深度图产物、源影像、mask/raw depth/raw confidence、深度后处理统计、融合降采样参数和实际设备信息，便于定位 GUI/CLI 的密集重建问题。
- GUI 密集重建目录树消费项目 metadata，深度图、稠密点云等结果按文件名自然排序并随任务增量刷新。
- SfM 报告增加候选 pair plan、实际匹配图、连通分量、pending/failed/skipped pair、注册影像与质量统计，便于排查只注册少量影像的问题。

### 优化

- MVS 融合改为有界流式窗口，融合阶段可把 6000x4000 depth/confidence 下采样到默认最长边 2048，保留相机内参缩放和颜色采样一致性，显著降低大航测数据内存峰值。
- 对已完成置信度/局部离群过滤的深度图增加快速反投影融合路径，避免每个流式窗口再做昂贵的全量多视 BFS 检查，8 帧 aerial GCP 回归中单批融合约 0.7 秒。
- 流式融合加入点数阈值预聚合，点数接近内存上限前先做 voxel 预聚合，避免长时间运行到后半程才被系统杀死。
- MVS 深度图加载、预取、保存队列、取消检查和后处理日志继续收敛，GUI 点击取消后能更早中断排队保存和后续处理。
- 已知外参进入 BA 时改为 soft pose prior，外参不再默认完全固定；支持位置/旋转 sigma、Huber 和 LiDAR 质量权重参与 BA。
- GUI 特征匹配和空三流程复用更一致的 pair planning 逻辑，大项目优先使用序列窗口、相机中心邻域和已知重叠对，避免默认退回 N² 全匹配。

### 修复

- 修复深度图估计完成后目录树不显示深度图、结果顺序不稳定的问题。
- 修复 MVS 大数据融合阶段把所有深度图和颜色图长期常驻内存导致内存持续上涨的问题。
- 修复融合阶段下采样 depth 后颜色仍按原图尺寸采样，可能导致稠密点颜色错位或丢失的问题。
- 修复深度图局部红色孤立噪点缺少统计和过滤记录的问题，输出 confidence/local outlier 移除数量。
- 修复稀疏重建报告对候选匹配、实际匹配和几何验证失败混在一起导致问题难以定位的问题。
- 修复 Linux CI 在 Build 阶段执行 Qt GUI 测试 discovery 时没有 offscreen 环境，导致 `test_bundle_adjust_dialog_lidar` 构建失败的问题。

### 验证

- `python -m pytest tests\test_mvs_scheduler_config.py -q` 通过，69/69。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target reconstruct_pipeline_cli -Jobs 8` 通过。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -RunTests -CTestRegex 'Sfm|Feature|Match|Mvs|Gui|Lidar|Bundle' -Jobs 8` 通过，172/172；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target test_bundle_adjust_dialog_lidar -RunTests -CTestRegex 'BundleAdjustDialogLidar|ProjectManagerBundleAdjustLidar' -Jobs 8` 通过，3/3。
- aerial GCP 8 帧 MVS 回归通过：`status=ok`，输出 `depth_0.png` 到 `depth_7.png`、`dense_cloud.ply` 1,240,093 点、`dense_cloud_refined.ply` 1,193,941 点；MVS 阶段约 429.8 秒，总流程约 998.0 秒。

### 已知问题

- 本版本验证了 aerial GCP 的 8 帧受控 MVS 回归，没有重新跑完 444 张完整 mesh/DEM/DOM 长链；完整长链仍建议作为后续夜间回归。
- MVS 后处理法向量估计在百万级点云上仍是明显耗时点，已不再导致本次回归崩溃，但后续可以继续做分块/可选法线输出优化。
- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`；该测试可能出现 `dom_png not found`。
- GitHub Actions `build-test` 是远端门禁；`v1.1.4` tag 推送后以 Actions 结果为准。

## v1.1.3 - 2026-06-18

### 新增

- 新增 LiDAR / 激光点点到面约束模块，支持读取带法线 PLY、曲率筛选、体素采样、最近平面查询，以及把 BA track 关联到 LiDAR 平面约束。
- 新增 `bundle_adjust_cli`，可对 `.plascan` 项目执行 headless BA，支持 `--laser-cloud`、LiDAR 关联距离/权重参数、`--ab-compare` 基线/激光约束 A-B 对比和 JSON/CSV/tsai 输出。
- 新增 `feature_match_cli` BA sidecar 输出，传统 BF/FLANN 匹配会同步生成 `.match.json`，包含匹配点坐标、特征索引和匹配分数，供多视 track 构建和 BA CLI 使用。
- 新增 MUN-FRL LiDAR BA 数据准备脚本、LiDAR 法线估计脚本、输入校验脚本和 A-B 结果比较脚本，支持从公开多传感器数据构造近期激光约束测试集。
- GUI 光束法平差对话框新增 LiDAR 约束参数，ProjectManager/BundleAdjustService 能把激光点云路径、关联半径、曲率阈值、权重和 Huber 阈值传入 BA 服务。

### 优化

- `prepare_mun_frl_lidar_ba_project.py` 支持 `--tf-static --camera-frame --body-frame`，会把 ROS 风格 `T_parent_child` 静态外参与 odometry 的 `world -> body` 位姿合成为 PlaScan 约定的 `world -> camera` 位姿。
- `compare_lidar_ba_ab_results.py` 增加固定口径评估，报告共同有效 track RMS、相机中心/旋转漂移、LiDAR 约束数量和 LiDAR 点到面 RMS/median，避免只看全局 RMS 误判效果。
- `testData/README.md` 补充 MUN-FRL lighthouse 从影像/匹配/轨迹/LiDAR 到 `.plascan`、法线估计和 BA A-B 对比的完整命令链。

### 修复

- 修复 headless BA CLI 默认导出评估图时可能依赖 GUI/图表环境的问题，默认关闭评估图导出，并提供 `--export-eval-plot` 显式开启。
- 修复 MUN-FRL `cloud_registered` 样例 PLY 虽有 normal 字段但法线全为 0 导致 LiDAR constraint map 无有效平面样本的问题，通过法线估计预处理生成可用点到面约束点云。

### 验证

- `python -m unittest tests.test_prepare_mun_frl_lidar_ba_project tests.test_compare_lidar_ba_ab_results tests.test_estimate_lidar_normals tests.test_bundle_adjust_cli tests.test_feature_match_cli_sidecar` 通过，11 个测试，其中未设置 CLI 环境变量时 2 个跳过。
- 设置 `PLASCAN_FEATURE_MATCH_CLI` 和 `PLASCAN_BUNDLE_ADJUST_CLI` 后，`python -m unittest tests.test_bundle_adjust_cli tests.test_feature_match_cli_sidecar tests.test_prepare_mun_frl_lidar_ba_project tests.test_compare_lidar_ba_ab_results` 通过，10/10。
- `python -m py_compile testData\prepare_mun_frl_lidar_ba_project.py testData\compare_lidar_ba_ab_results.py testData\estimate_lidar_normals.py` 通过。
- `ctest --test-dir build\windows-vcpkg-cuda-release --output-on-failure -R "lidar|Lidar|Laser|BundleAdjustCliTest|FeatureMatchCliSidecarTest|PrepareMunFrlLidarBaProjectTest|CompareLidarBaAbResultsTest|EstimateLidarNormalsTest"` 通过，19/19。
- MUN-FRL lighthouse 20 张图窗口生成 `mun_frl_lidar_ba_tf.plascan`，BA dry-run 输入为 `cameras=20 tracks=3689 sidecar_v2_pairs=230822 multiview_tracks=3689`。
- MUN-FRL A-B 诊断：`imu_link` body frame 的影像基线约 `0.533 px`，但 3-5 m LiDAR 约束会增大 LiDAR 点到面残差；`velodyne` body frame 可降低 LiDAR 残差但影像基线约 `1.126 px`，因此该样例目前作为 smoke/诊断集，不作为最终精度提升证明。

### 已知问题

- MUN-FRL `lio_body`、`imu_link`、`velodyne` 与相机 frame 的语义仍需进一步核实；当前 LiDAR BA 已能参与优化，但公开样例上的精度提升结论需要更可靠的跨传感器坐标链或更合适的数据切片。
- 本版本没有重新跑 agisoft aerial GCP 444 张完整 MVS/mesh/DEM/DOM 长时流水线。
- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`，该测试可能出现 `dom_png not found`。
- GitHub Actions `build-test` 作为远端门禁；`v1.1.3` 发布提交和 tag 推送后需以 Actions 结果为准。

## v1.1.2 - 2026-06-17

> 2026-06-18 补充：新增 MVS 深度图内存自适应保护、快速二进制深度产物保存和 GUI CUDA 帧流水线自适应，代码已进入 `main`；既有 `v1.1.2` tag 未强制重写。

### 新增

- 新增 LiDAR / 激光点摄影测量数据集整理文档，覆盖 MUN-FRL、UseGeo、H3D、MARS-LVIG、UAVScenes、NTU VIRAL、DublinCity、DFC 2019 / US3D、ETH3D 等候选数据，用于后续激光点参与 BA、相机-LiDAR 外参验证、MVS/DEM/DOM 质量检查。
- 扩展摄影测量测试数据下载器，增加 `workflow_tags` 和 `--workflow-tag` 选择能力，可按 `ba_constraint_candidate`、`lidar_fusion_validation` 等用途筛选 LiDAR 相邻数据集并生成手动下载 manifest。

### 优化

- 优化 MVS 深度图估计的 CPU 前处理链路：缓存源帧共视稀疏点、并行构建可见性缓存、源帧角度打分提前停止，减少大规模航测数据在进入 CUDA PatchMatch 前的等待时间。
- 优化稀疏 hint 构建：投影稀疏样本只做单次可见点遍历，深度分位采样有界，粗层 hint 使用 OpenCV distance transform 限距离传播，精层仅叠加 seed，避免全图传播把错误深度强行铺满。
- 优化 MVS 支撑掩码与 dense refine：稀疏支撑改为软约束，支撑掩码按精层 PatchMatch 工作尺寸生成，大点云 refine 对输入和内联过滤做上限保护，避免 GUI/CLI 在百万级点云上长时间阻塞。
- 优化 CUDA 灰度图处理路径，避免重复 reference resize，并保留 GPU 灰度图缓存命中统计，便于观察 GPU 利用率不连续时是否被上传/缩放拖慢。
- 优化 MVS 深度图缓存策略：根据系统物理内存和可用内存自动决定 full-res 深度图是否常驻内存，内存充足时保留空间换时间，内存不足或运行时压力升高时切换为流式保存并释放已缓存深度图。
- 优化深度图预览/中间结果保存队列，限制后台 full-res 深度帧积压数量，避免保存线程落后时继续放大内存峰值。
- 优化 MVS 深度图产物保存：原始 depth/confidence 从 OpenCV `.yml.gz` 改为 PlaScan 二进制 `.bin`，减少每帧 CPU 压缩/文本序列化耗时，并保留旧 `.yml.gz` 读取兼容。
- 优化深度图预览图保存：预览 PNG 默认限制最长边 2048，避免每帧额外写入 6000x4000 全尺寸伪彩图拖慢 GPU 流水线。
- 优化 GUI 手动深度估计默认调度：线程数足够时自动使用 2 个 CUDA frame workers，让 GPU 帧任务和 CPU/IO 后处理更容易形成流水线。

### 修复

- 修复 MVS 后处理中过强稀疏支撑裁剪可能导致有效深度被硬清空的问题，改为置信度软缩放并保留深度。
- 修复深度图局部离群噪点缺少后处理保护的问题，增加局部深度离群过滤和阶段耗时统计，便于定位 `source/range/hint/patchmatch/filter` 瓶颈。
- 修复大规模 dense refine 连续执行第二轮 SOR 或内联大点云过滤导致的卡顿风险。
- 修复大规模 MVS 深度估计长期运行时无条件缓存每帧 `depth + confidence` full-res Mat，可能在软件崩溃前才暴露 OpenCV 内存分配失败的问题；现在会提前降级并给出日志/错误提示。

### 验证

- `python -m unittest tests.test_mvs_scheduler_config tests.test_reconstruct_pipeline_cli tests.test_repo_hygiene tests.test_download_photogrammetry_testdata` 通过，65 个测试，3 个跳过。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Jobs 8` 通过，使用 Windows 原生 MSVC/Ninja、CUDA 13.1、libtorch-cu130 和 `build\windows-vcpkg-cuda-release`。
- `build\windows-vcpkg-cuda-release\tests\test_mvs_pipeline.exe --gtest_brief=1` 通过，17/17。
- `build\windows-vcpkg-cuda-release\tests\test_mvs_types.exe --gtest_brief=1` 通过，16/16。
- `build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_brief=1` 通过，135/135。
- `python -m unittest tests.test_mvs_scheduler_config` 通过，48/48。
- `python -m unittest tests.test_mvs_scheduler_config tests.test_reconstruct_pipeline_cli tests.test_three_d_reconstruction_cli` 通过，61 个测试，3 个跳过。
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
