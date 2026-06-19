# Reconstruction Pipeline Four-Stage Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 PlaScan 重建链路升级为可恢复、可诊断、可流式、可生成正式 DEM/DOM 产品的工程化流水线。

**Architecture:** 以项目 metadata 和磁盘 workspace manifest 作为主状态，核心算法只产出结构化 artifact，GUI 只消费 metadata 并发起异步任务。第一阶段先稳定 MVS 内存、取消、深度图记录和目录刷新；第二阶段优化 source view、几何一致性和深度质量；第三阶段扩展 DEM/DOM 栅格化与 mosaic；第四阶段引入外部 DEM/LiDAR 的后验验证和 BA soft prior。

**Tech Stack:** C++17, Qt6, OpenCV, CUDA, plapoint, plamatrix, CMake, GTest, GDAL/libtiff, project JSON metadata.

---

## Execution Audit - 2026-06-20

> 本节记录当前工作区的真实完成度。下面的任务 checklist 仍保留原始执行粒度；其中 `Commit:` 项按 `AGENTS.md` 规则只有用户明确要求时才执行，因此不能因为代码完成就自动勾选。

| Area | Current evidence | Status |
| --- | --- | --- |
| Stage 1: MVS workspace / cancel / bounded memory / metadata refresh | `MvsWorkspaceManifest`、深度帧 manifest 恢复、raw depth / confidence / valid mask 写盘、MVS cancel 传播、metadata 刷新和自然排序已有实现；相关回归包含在 `MvsWorkspaceManifest|MvsPipelineTest|DepthMapPersistence|DepthFrameUtils|DataTreeWidgetTest`；完整 444 输入的全帧 `--mvs-depth-only` 已通过，444/444 manifest frames 完成。 | Implemented and focused-tested. Full-depth 444-image run is verified. |
| Stage 2: source planning / depth quality / fusion robustness | `MvsSourcePlanner` 已接入深度估计和融合，深度后处理包含 confidence、valid mask、speckle 和一致性过滤；DenseCloudDialog 已暴露高级 MVS 参数且默认值不改变既有行为。 | Implemented and focused-tested. First-20-frame visual overlap quality has smoke evidence, but not a formal visual oracle. |
| Stage 3: DEM/DOM terrain product chain | `DemGridAggregator`、`TerrainProductManifest`、`DemMosaic`、DEM quality rasters、DOM 空覆盖保护和 GUI terrain quality 节点已有实现；`src/core/terrain/README.md` 与架构文档已补齐。 | Implemented and focused-tested. 0.5 scale smoke 已产出 DEM/DOM；full 444-image mesh/DEM/DOM 长链已通过。 |
| Stage 4: DEM/LiDAR reference validation and BA soft priors | `ReconstructionQualityReport`、`PointCloudAlignment`、`DemDifference`、`ReferenceTerrainPrior`、参考 DEM/LiDAR GUI 工作流和 LiDAR 点到面 BA soft prior 已实现。MUN-FRL real-data A/B quality gate 通过。 | Implemented and focused-tested with real LiDAR BA smoke. Public sample frame semantics remain a dataset-specific risk. |
| Stage 5: cross-stage verification / docs / release readiness | 全量 CTest 已通过 467/467；0.2 scale 与 0.5 scale agisoft aerial GCP smoke 均 `status=ok`；完整 444 输入的全帧 `--mvs-depth-only --mvs-max-frames 444` 验证通过，444/444 manifest frames 完成且 artifact/source_plan 无缺失；完整 444 输入的 MVS/mesh/DEM/DOM 长链也已通过，生成 dense cloud、refined dense cloud、textured OBJ、DEM 和 DOM；`source_plan` 保留修复已由 MVS manifest 单测、12-image depth-only 回归和 444-image 回归覆盖；SfM 稀疏点云导出已改为按影像批量颜色采样，并由 444-image SFM/depth/full-chain 命令行验证覆盖；`CHANGELOG.md`、`docs/releases/v1.1.5.md`、`docs/PROJECT_ARCHITECTURE.md`、MVS/terrain README 已同步；`v1.1.5` annotated tag、GitHub Release 和 GitHub Actions main/tag `build-test` 门禁均已完成。 | Complete. Local and remote release evidence verified. |

Key verified artifacts:
- `E:/code/plascan/build/agisoft_aerial_mvs_dem_dom_scale05_codex_20260620_010303/pipeline/report.json`: `status=ok`, 12/12 registered, 4 MVS frames, 1,281,149 dense points, 1,224,829 refined points, DEM/DOM written.
- `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_codex_20260620_012051/pipeline/report.json`: `status=ok`, `stop_stage=mvs_depth`, `dense.status=depth_only`, 4 depth artifact records, no fusion/model/terrain outputs.
- `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_50_codex_20260620_012819/pipeline/report.json`: `status=ok`, 444/444 registered, 50/50 manifest frames completed, 100 depth report records, no missing depth/raw/confidence/mask artifacts, no fusion/model/terrain outputs.
- `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_150_codex_20260620_013608/pipeline/report.json`: `status=ok`, 444/444 registered, 150/150 manifest frames completed, 300 depth report records, no missing depth/raw/confidence/mask artifacts, no fusion/model/terrain outputs.
- `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_12_codex_20260620_021000/pipeline/report.json`: `status=ok`, 12/12 registered, 4/4 manifest frames completed, raw/confidence/mask artifacts present, and every completed frame keeps a non-empty `source_plan` after filtered-depth artifact updates.
- `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_444_postfix_codex_20260620_022500/pipeline/report.json`: `status=ok`, 444/444 registered, 8/8 manifest frames completed, 16 depth report records, no missing raw/confidence/mask/source_plan artifacts, no fusion/model/terrain outputs.
- `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_444_batched_retry_codex_20260620_022910/pipeline/report.json`: command exit code 0, `status=ok`, `stop_stage=mvs_depth`, 444/444 registered, 57,362 sparse points, 444/444 manifest frames completed, no missing `source_plan`, MVS elapsed 149.543 s, total elapsed 420.977 s, no fusion/model/terrain outputs.
- `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_444_batched_retry_codex_20260620_022910/pipeline/report.json`: command exit code 0 for the full MVS/mesh/DEM/DOM chain, `status=ok`, 444/444 registered, 57,362 sparse points, 1,895,106 dense points, 1,847,925 refined dense points, textured OBJ written, DEM/DOM written, total elapsed 5686.447 s.
- `E:/code/plascan/build/mun_frl_lidar_ba_ab_project_tf_velodyne/ba_ab_run_codex_20260620_003746`: LiDAR BA A/B quality gate passed; LiDAR point-to-plane RMS improved from 1.0371 m to 0.9297 m.
- `python -m pytest tests/test_repo_hygiene.py -q`: 9 tests and 27 subtests passed after module documentation updates.

## Scope And Ordering

This plan is intentionally split into four stages. Each stage must compile and run tests before moving to the next stage.

1. Stage 1: MVS workspace, streaming memory, cancellation, and GUI metadata refresh.
2. Stage 2: MVS source selection, depth quality, confidence, and fusion robustness.
3. Stage 3: DEM/DOM terrain product chain, gridding, mosaic, and terrain artifact tree.
4. Stage 4: External DEM/LiDAR alignment, quality reports, and BA soft constraints.

Do not copy ASP or COLMAP code directly. Use their architecture as a reference only.

## Current Execution Evidence

As of 2026-06-20, the implementation exists in the working tree and is tracked in the `v1.1.5` release notes.
The strongest local evidence is:

- Stage 1/2/4 cross-regression:
  `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsWorkspaceManifest|MvsSourcePlanner|MvsDepthPostprocess|MvsPipelineTest|DataTreeWidgetTest|BundleAdjustServiceLidar|LaserConstraint|ReferenceTerrain|ReferenceDataset|QualityReport|PointCloudAlignment|DemDifference" --output-on-failure`
  passed 79/79.
- Stage 2 focused regression:
  `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsSourcePlanner|MvsDepthPostprocess|DenseCloudDialog|DepthMapFusion|MvsPipelineTest|DepthMapPersistence|DepthFrameUtils|DenseDepth|DataTreeWidgetTest\.ResultOnlyMetadataUpdateRefreshesDepthMapSection" --output-on-failure`
  passed 35/35.
- Stage 3 focused regression:
  `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainDemDom|DemGridAggregator|DemMosaic|TerrainProductManifest|DemQualityRasters|DataTreeWidgetTest\.DemSectionShowsQualityRasterProducts" --output-on-failure`
  passed 28/28.
- Cross-stage focused regression:
  `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "Mvs|Depth|Fusion|Terrain|Dem|Dom|Sfm|Bundle|Quality|GuiProject|DataTree" --output-on-failure`
  passed 209/209.
- Documentation hygiene:
  `python -m pytest tests\test_repo_hygiene.py -q` passed 9 tests and 27 subtests, including `README.md`,
  `src/core/mvs/README.md`, `src/core/terrain/README.md`, and `docs/PROJECT_ARCHITECTURE.md`.
- Depth-only CLI smoke:
  `reconstruct_pipeline_cli --mvs-depth-only` on 12 agisoft aerial images passed with `status=ok`,
  `stop_stage=mvs_depth`, and no fusion/model/terrain outputs at
  `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_codex_20260620_012051/pipeline`.
- Full-input depth scheduling smoke:
  `reconstruct_pipeline_cli --mvs-depth-only --mvs-max-frames 50` on the 444-image agisoft aerial input
  passed with 444/444 registered images, 50/50 completed manifest frames, no missing artifacts, MVS elapsed
  38.730 s, and GPU gray cache usage around 3.7-228.9 MB at
  `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_50_codex_20260620_012819/pipeline`.
- Longer full-input depth scheduling smoke:
  `reconstruct_pipeline_cli --mvs-depth-only --mvs-max-frames 150` on the same 444-image input passed with
  444/444 registered images, 150/150 completed manifest frames, no missing artifacts, MVS elapsed 128.684 s,
  total elapsed 400.586 s, and GPU gray cache usage around 3.7-686.6 MB at
  `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_150_codex_20260620_013608/pipeline`.
- Full-frame depth-only scheduling smoke:
  `reconstruct_pipeline_cli --mvs-depth-only --mvs-max-frames 444` on the same 444-image input exited with code 0
  and passed with 444/444 registered images, 444/444 completed manifest frames, no missing `source_plan`,
  MVS elapsed 149.543 s, total elapsed 420.977 s, and GPU gray cache usage peaking around 2.03/6.00 GB at
  `E:/code/plascan/build/agisoft_aerial_mvs_depth_only_444_batched_retry_codex_20260620_022910/pipeline`.

Remote release evidence is now complete: the `v1.1.5` annotated tag is pushed, GitHub Release `PlaScan v1.1.5`
is published, and GitHub Actions `CI / build-test` passed on both the main run `27847059348` and the tag run
`27847075291`.

## Four-Stage Execution Blueprint

### Stage 1: 稳定 MVS 主链和 GUI 状态同步

**核心目标:** 先让深度图估计成为可恢复、可取消、内存有上限、GUI 可实时刷新的稳定流水线。444 张影像这种规模不能再跑到一半被系统杀进程，也不能点击取消后一直停在“正在取消”。

**实现范围:**
- 建立 `MvsWorkspaceManifest`，把每一帧深度图的状态、输入影像、输出路径、设备、耗时、错误和配置 hash 落到磁盘。
- 深度估计调度从 manifest 恢复，已完成且配置一致的帧跳过，失败帧重试。
- 把取消信号传到预加载、source 选择、hint depth、CUDA PatchMatch 迭代、写盘队列和融合准备。
- 用运行时内存预算控制 resident depth frame 数量，超过预算就释放像素矩阵，只保留 metadata 和磁盘 artifact。
- GUI 工作区目录树只以项目 metadata 为主状态刷新，深度图、照片、连接点、DEM、DOM 等节点按自然文件名排序。

**验收标准:**
- 开始 MVS 后 GUI 不阻塞。
- 点击取消后能在当前安全边界内退出，不长期卡在“正在取消”。
- 深度图完成一帧就写入 manifest 和项目 metadata，并出现在目录树中。
- 跑 `agisoft_aerial_gcps` 的 2048 或 0.5 scale smoke 测试时，内存不会持续线性上涨到崩溃。

### Stage 2: 优化 MVS 选源、深度质量和融合鲁棒性

**核心目标:** 解决 GPU 利用率不连续、source view 选择不可靠、深度图红色孤立噪点和融合输入质量差的问题。这里借鉴 COLMAP 的思路：MVS 不应只靠相机中心近邻，而要优先用真实 track、匹配图和几何关系选择邻接视图。

**实现范围:**
- 建立 `MvsSourcePlanner`，综合 shared tracks、几何内点数、三角角、投影覆盖率、baseline、文件序列邻近度给 source view 排序。
- 把 source plan 写入 manifest，保证同一配置下可复现。
- `DepthMapGenerator` 和 `DepthMapFusion` 都使用同一份 source plan，避免估计和融合看到的是两套邻接关系。
- 深度后处理增加 connected-component speckle filter、局部突刺过滤、confidence mask、valid mask、可选几何一致性和最小一致视图数。
- GUI 高级参数暴露 `最小一致视图数`、`孤立噪点面积阈值`、`几何一致性过滤`，日志里显示真实生效值。

**验收标准:**
- 前 20 帧 source view 不再选到明显无重叠或视觉不相关的影像。
- 深度图中的小面积红色孤立噪点被过滤，大片连续地物区域不被误删。
- 每帧输出 depth preview、raw depth、confidence、valid mask，并在日志中记录 postprocess 统计。
- fusion 使用 manifest source plan 后，覆盖率和错误率比 nearest-center-only 更稳定。

### Stage 3: 做成正式 DEM/DOM 产品链

**核心目标:** 让 DEM/DOM 不只是“从点云临时导出一张图”，而是正式的 terrain product pipeline，能保留误差、置信度、点数、覆盖率和 mosaic 信息。这里借鉴 ASP 的 DEM/DOM 工程化链路，但只参考设计，不复制代码。

**实现范围:**
- 建立 `DemGridAggregator`，支持 mean、median、min/max、stddev、NMAD、P80、count、confidence weighted、inverse error weighted 等格网聚合。
- 扩展 `DemGridData`，保留 elevation、rgb、point count、confidence、coverage mask、triangulation error 等矩阵。
- `DemDomIO` 写出 DEM、DOM、误差栅格、点数栅格、置信度栅格、覆盖率栅格，并保持投影/geotransform。
- 建立 `TerrainProductManifest`，把 terrain 产品的路径、类型、投影、分辨率、聚合方式和质量 artifact 统一记录。
- 建立 `DemMosaic`，支持多 tile DEM 的 first/last/mean/median/min/max/confidence weighted/inverse error weighted 拼接。
- GUI 目录树显示 DEM、DOM、误差、点数、置信度、覆盖率等产品节点，并按文件名排序。

**验收标准:**
- DEM 生成后不仅有 `dem.tif`，还应有可选 `dem_error.tif`、`dem_count.tif`、`dem_confidence.tif`、`dem_coverage.tif`。
- DOM 生成前会检查 DEM coverage，空覆盖或低质量区域不盲目贴图。
- 大点云可以按 tile 处理，不要求一次把全部 DEM mosaic 数据塞入内存。
- GUI 能看到 terrain 产品全套 artifact，而不是只显示一个 0 或缺项。

### Stage 4: 引入 DEM/LiDAR 参考验证和 BA 软约束

**核心目标:** 外部 DEM/LiDAR 先作为后验质量检查和配准参考，再进入 BA 作为 soft prior。不要一开始就把参考数据硬塞进 BA，把可诊断性先做出来。

**实现范围:**
- 建立 `ReconstructionQualityReport`，输出注册影像数、未注册影像、重投影误差、track length、三角角、稀疏/密集点数、DEM 覆盖率、MVS 有效覆盖率。
- 建立 `PointCloudAlignment`，先做 point-to-point/Sim3 robust alignment，输出 transform、beg/end errors、RMSE、median、NMAD、P50/P84/P95。
- 建立 `DemDifference`，比较重建 DEM 与参考 DEM，输出差值 GeoTIFF、绝对差值 GeoTIFF 和统计报告。
- 建立 `ReferenceTerrainPrior`，把参考 DEM 或 LiDAR 局部高度面作为 BA 软约束，支持 sigma、robust loss 和残差前后对比。
- GUI 增加 `导入参考 DEM/LiDAR`、`点云/DEM 精度检查`、`使用参考地形约束重新平差` 工作流。

**验收标准:**
- 用户能先跑“精度检查”得到清楚的误差报告，而不是只能看点云形状猜问题。
- 外参默认是 soft prior，不默认固定；BA 后能看到 pose prior/terrain prior 残差变化。
- DEM/LiDAR 文件默认以外部引用记录，不自动复制大文件进项目。
- 报告、差值图、配准结果都进入项目 metadata 和 GUI 目录树。

## Git Checkpoint Rule

The task-level commit lines below are checkpoints for agents executing the plan. They must still follow
`E:/code/plascan/AGENTS.md`: inspect `git status --short` first and only create commits when the user
explicitly asks for commit/push/tag work.

---

## Stage 1: MVS Workspace Stability

**Goal:** Depth estimation must not kill the process on 444-image projects, cancellation must exit, and generated depth artifacts must appear in the project tree in filename order.

**Files:**
- Create: `src/core/mvs/MvsWorkspaceManifest.h`
- Create: `src/core/mvs/MvsWorkspaceManifest.cpp`
- Create: `src/core/mvs/tests/test_mvs_workspace_manifest.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.h`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/DepthFrameUtils.h`
- Modify: `src/core/mvs/DepthFrameUtils.cpp`
- Modify: `src/core/mvs/CMakeLists.txt`
- Modify: `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `src/gui/widgets/DataTreeWidget.cpp`
- Modify: `src/gui/main_window/MainWindow.cpp`
- Modify: `src/gui/widgets/TaskStatusWidget.cpp`
- Modify: `src/core/mvs/tests/CMakeLists.txt`

### Task 1.1: Add A Durable MVS Manifest

- [ ] Create `MvsWorkspaceManifest` with one JSON record per depth frame: `ref_image`, `ref_index`, `source_images`, `status`, `device`, `depth_png`, `raw_depth_path`, `raw_confidence_path`, `valid_mask_path`, `elapsed_ms`, `error`, `config_hash`.
- [ ] Add `load()`, `saveAtomic()`, `upsertFrame()`, `markRunning()`, `markCompleted()`, `markFailed()`, and `completedFramesSortedByName()`.
- [ ] Write `test_mvs_workspace_manifest` covering atomic save, reload, natural filename sorting, failed frame update, and config hash mismatch invalidation.
- [ ] Register the test in `src/core/mvs/tests/CMakeLists.txt`.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R test_mvs_workspace_manifest --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add MVS workspace manifest"`.

### Task 1.2: Make Depth Estimation Resumable

- [ ] Before scheduling a frame in `DepthMapGenerator::runInBackground`, check the manifest. Skip frames that are `completed` and whose `config_hash` matches the current settings.
- [ ] When a frame starts, mark it `running`; when artifacts are saved, mark it `completed`; on exception or failed generation, mark it `failed` with the concrete error message.
- [ ] Emit `depthMapArtifactSaved(QJsonObject)` only after manifest save succeeds.
- [ ] Keep `saveRawDepth` enabled by default for projects with more than 64 registered images so later fusion can stream from disk.
- [ ] Add a regression test using a synthetic two-frame manifest that confirms a completed frame is skipped and a failed frame is retried.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsWorkspace|DepthFrame" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: resume MVS depth frames from manifest"`.

### Task 1.3: Close The Cancellation Loop

- [ ] Thread the existing cancel flag through image preload, source selection, hint-depth generation, PatchMatch iteration boundaries, artifact save queue, and fusion setup.
- [ ] In CUDA PatchMatch code paths, check cancel between iterations and before large host-to-device or device-to-host transfers.
- [ ] Ensure `DepthFrameArtifactSaveQueue::stop()` drains already-started writes but refuses new work after cancel.
- [ ] Update `TaskStatusWidget` so `正在取消` is cleared when the worker emits `finished(false)` or `errorOccurred`.
- [ ] Add an integration test that starts a small MVS run, triggers cancel, and asserts `finished(false)` arrives within 5 seconds.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsCancel|GuiMvs" --output-on-failure`.
- [ ] Commit: `git commit -m "fix: make MVS cancellation cooperative"`.

### Task 1.4: Bound Memory With Streaming Fusion Preparation

- [ ] Replace the current all-or-nothing `keepDepthFramesInMemory` decision with a memory budget object: total RAM, free RAM, estimated per-frame depth/confidence/mask bytes, reserve bytes, and max resident frames.
- [ ] When the estimated resident set exceeds budget, release `DepthFrameResult` pixel storage after artifacts are saved and leave only manifest records.
- [ ] If user requests fusion in the same run and streaming mode is active, stop with a clear message: `深度图已生成；当前内存预算要求分步融合，请运行“深度图融合生成密集点云”。`
- [ ] Add log lines for memory budget, max resident frames, released frame count, and output workspace path.
- [ ] Run a smoke test on `agisoft_aerial_gcps` at scale 0.5 or max size 2048 and verify process memory stops growing after completed frames are released.
- [ ] Commit: `git commit -m "fix: bound MVS frame memory by runtime budget"`.

### Task 1.5: Refresh The GUI Tree From Metadata

- [ ] Ensure `ProjectDenseReconstructionManager` writes depth records into project metadata whenever `depthMapArtifactSaved` fires.
- [ ] Ensure `DataTreeWidget::sortSectionChildrenByFileName` is applied to `照片`, `连接点`, `深度图`, `稠密点云`, `DEM`, and `正射影像`.
- [ ] Do not scan the output directory as the primary state. Directory scans can repair missing metadata, but the project metadata remains authoritative.
- [ ] Add a GUI project utility test that creates three depth records out of order and asserts the tree order is `depth_001`, `depth_002`, `depth_010`.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiProject|DataTree|Mvs" --output-on-failure`.
- [ ] Commit: `git commit -m "fix: refresh depth artifacts from project metadata"`.

---

## Stage 2: MVS Quality And Source Selection

**Goal:** Reduce red speckle noise, improve valid coverage, and use true overlap/match geometry instead of nearest camera centers alone.

**Files:**
- Create: `src/core/mvs/MvsSourcePlanner.h`
- Create: `src/core/mvs/MvsSourcePlanner.cpp`
- Create: `src/core/mvs/tests/test_mvs_source_planner.cpp`
- Modify: `src/core/mvs/MvsViewSelection.h`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/DepthMapFusion.h`
- Modify: `src/core/mvs/DepthMapFusion.cpp`
- Modify: `src/core/mvs/MvsTypes.h`
- Modify: `src/gui/dialogs/DenseCloudDialog.ui`
- Modify: `src/gui/main_window/ReconstructionWorkflowController.cpp`
- Modify: `src/core/mvs/CMakeLists.txt`

### Task 2.1: Build COLMAP-Style Source Planning

- [ ] Implement `MvsSourcePlanner` that scores candidate source views by shared tracks, geometric inlier count, median triangulation angle, projected sparse coverage, camera baseline, and filename-sequence proximity.
- [ ] Persist the source plan in the MVS manifest for reproducibility.
- [ ] Prefer actual registered SfM tracks and match graph edges; fall back to camera center neighbors only when match/track data is unavailable.
- [ ] Write tests for sequence fallback, shared-track ranking, triangulation-angle rejection, duplicate removal, and source count limiting.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R test_mvs_source_planner --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add MVS source view planner"`.

### Task 2.2: Use The Source Plan In Depth Generation And Fusion

- [ ] Replace direct source selection in `DepthMapGenerator` with `MvsSourcePlanner`.
- [ ] Replace nearest-center-only overlap computation in `DepthMapFusion::computeOverlappingImages` with the manifest source plan when available.
- [ ] Store per-frame source reason fields: `shared_tracks`, `geom_inliers`, `median_angle_deg`, `coverage_score`, `baseline_score`.
- [ ] Add logs showing top source views for each reference frame and why rejected candidates were skipped.
- [ ] Run the agisoft sample and verify the first 20 frames do not pick visually unrelated source images.
- [ ] Commit: `git commit -m "fix: use planned MVS overlaps for fusion"`.

### Task 2.3: Improve Depth Confidence And Speckle Filtering

- [ ] Extend `DepthPostProcessStats` with `speckleRemoved`, `smallComponentRemoved`, `edgeConfidenceRemoved`, and `geomConsistencyRemoved`.
- [ ] Add a connected-component speckle filter on valid masks after local median filtering. Remove components smaller than a configurable pixel count unless their mean confidence is high.
- [ ] Add an optional min-consistent-source check: a depth pixel must be confirmed by at least `minConsistentViews` source views or be marked invalid.
- [ ] Save `valid_mask_path` and confidence preview for every frame when raw depth is saved.
- [ ] Add a synthetic depth test with isolated red-depth spikes and assert the spikes are removed while large smooth regions remain.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DepthPost|MvsQuality" --output-on-failure`.
- [ ] Commit: `git commit -m "fix: filter MVS speckles with confidence masks"`.

### Task 2.4: Expose Quality Controls Without Overloading The GUI

- [ ] Add three advanced settings to the dense dialog: `最小一致视图数`, `孤立噪点面积阈值`, and `几何一致性过滤`.
- [ ] Show effective settings in the log before the run starts.
- [ ] Store settings in dialog persistence through `DialogSettingKeys::DenseCloud`.
- [ ] Add a GUI settings roundtrip test so saved values reopen correctly.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DenseCloudDialog|MvsQuality" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: expose MVS quality controls"`.

---

## Stage 3: DEM/DOM Terrain Product Chain

**Goal:** Make DEM/DOM generation a formal terrain product pipeline with confidence/error-aware gridding, tiled mosaic, and complete project artifacts.

**Files:**
- Create: `src/core/terrain/DemGridAggregator.h`
- Create: `src/core/terrain/DemGridAggregator.cpp`
- Create: `src/core/terrain/DemMosaic.h`
- Create: `src/core/terrain/DemMosaic.cpp`
- Create: `src/core/terrain/TerrainProductManifest.h`
- Create: `src/core/terrain/TerrainProductManifest.cpp`
- Create: `src/core/terrain/tests/test_dem_grid_aggregator.cpp`
- Create: `src/core/terrain/tests/test_dem_mosaic.cpp`
- Modify: `src/core/terrain/DemDomTypes.h`
- Modify: `src/core/terrain/DemGenerator.h`
- Modify: `src/core/terrain/DemGenerator.cpp`
- Modify: `src/core/terrain/DomGenerator.h`
- Modify: `src/core/terrain/DomGenerator.cpp`
- Modify: `src/core/terrain/DemDomIO.cpp`
- Modify: `src/core/terrain/TerrainPipeline.cpp`
- Modify: `src/gui/widgets/DataTreeWidget.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/core/terrain/CMakeLists.txt`

### Task 3.1: Add ASP-Style DEM Aggregation Modes

- [ ] Extend `DemGenerationOptions::ElevationAggregation` with `WeightedAverage`, `Median`, `StdDev`, `Count`, `Nmad`, `Percentile80`, `Min`, `Max`, and `Mean`.
- [ ] Implement `DemGridAggregator` to accept point xyz, optional rgb, optional confidence, optional triangulation error, and source count.
- [ ] Use confidence as positive weight and triangulation error as inverse weight when both are available.
- [ ] Write tests for each aggregation mode with deterministic input points.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DemGridAggregator|TerrainDemDom" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add confidence-aware DEM aggregation"`.

### Task 3.2: Preserve Error, Count, And Coverage Products

- [ ] Extend `DemGridData` with `pointCount`, `confidence`, and `coverageMask` matrices.
- [ ] Update `DemDomIO` so GeoTIFF exports include elevation, color, triangulation error, confidence, and point count when available.
- [ ] Add project metadata records for `dem_path`, `dom_path`, `error_path`, `count_path`, `confidence_path`, `coverage_path`, `projection`, `grid_resolution`, and `aggregation`.
- [ ] Add tests that write/read a DEM product and verify all optional rasters survive roundtrip.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DemDomIO|TerrainManifest" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: persist terrain quality rasters"`.

### Task 3.3: Add Tiled DEM Mosaic

- [ ] Implement `DemMosaic` with tile-size, overlap, blend mode, and priority-by-error support.
- [ ] Support mosaic modes: first, last, mean, median, min, max, weighted by confidence, and weighted by inverse error.
- [ ] Write per-tile intermediate products so large dense clouds do not require one giant DEM in memory.
- [ ] Add tests for two overlapping DEM tiles with different confidence and error values.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R test_dem_mosaic --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add tiled DEM mosaic"`.

### Task 3.4: Upgrade DOM Generation To Use DEM Quality

- [ ] Use DEM coverage and confidence masks to avoid texturing invalid or unreliable cells.
- [ ] Add DOM seam weighting by view angle, confidence, and sharpness.
- [ ] Save DOM quality rasters alongside the DOM image.
- [ ] Add a terrain pipeline test that verifies DOM is not generated from empty DEM coverage.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DomGenerator|TerrainPipeline" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: weight DOM generation by terrain quality"`.

### Task 3.5: Reflect Terrain Products In The GUI

- [ ] Add DEM, DOM, error, count, confidence, and coverage child nodes under the project tree.
- [ ] Use natural filename sorting for all terrain product sections.
- [ ] When terrain metadata changes, refresh the tree without forcing the photo section open.
- [ ] Add a GUI metadata test that confirms all terrain product node types are visible.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DataTree|Terrain" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: show terrain quality products in GUI"`.

---

## Stage 4: DEM/LiDAR Reference And Quality Reports

**Goal:** Use external DEM/LiDAR first as validation and alignment references, then as soft constraints in BA.

**Files:**
- Create: `src/core/qc/ReconstructionQualityReport.h`
- Create: `src/core/qc/ReconstructionQualityReport.cpp`
- Create: `src/core/qc/PointCloudAlignment.h`
- Create: `src/core/qc/PointCloudAlignment.cpp`
- Create: `src/core/qc/DemDifference.h`
- Create: `src/core/qc/DemDifference.cpp`
- Create: `src/core/qc/tests/test_reconstruction_quality_report.cpp`
- Create: `src/core/qc/tests/test_point_cloud_alignment.cpp`
- Create: `src/core/sfm/ReferenceTerrainPrior.h`
- Create: `src/core/sfm/ReferenceTerrainPrior.cpp`
- Modify: `src/core/sfm/BundleAdjuster.h`
- Modify: `src/core/sfm/BundleAdjuster.cpp`
- Modify: `src/core/sfm/SfmQualityReport.h`
- Modify: `src/core/sfm/SfmQualityReport.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/gui/widgets/DataTreeWidget.cpp`
- Modify: `src/core/CMakeLists.txt`

### Task 4.1: Add Reconstruction Quality Reports

- [ ] Implement a JSON and CSV quality report containing registered image count, unregistered images, reprojection error mean/median/P95, track length histogram, convergence angle histogram, sparse point count, dense point count, DEM coverage, and MVS valid coverage.
- [ ] Save the report under the project output directory and register it in project metadata.
- [ ] Add a GUI tree node under `报告`.
- [ ] Add a test that feeds synthetic sparse/MVS/DEM stats and validates the JSON fields.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "QualityReport|SfmQuality" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add reconstruction quality reports"`.

### Task 4.2: Add Point Cloud Alignment And Error Metrics

- [ ] Implement initial point-to-point alignment with robust sampling and Sim3 transform estimation.
- [ ] Output `beg_errors.csv`, `end_errors.csv`, transform JSON, and summary metrics: RMSE, mean, stddev, median, NMAD, P50, P84, P95.
- [ ] Keep input point clouds read-only and write aligned output to a new result path.
- [ ] Add tests for a synthetic translated/scaled point cloud and assert recovered transform accuracy.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R test_point_cloud_alignment --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add point cloud alignment metrics"`.

### Task 4.3: Add DEM Difference Reports

- [ ] Implement `DemDifference` that compares two DEM rasters on a shared grid.
- [ ] Output difference GeoTIFF, absolute difference GeoTIFF, and summary metrics.
- [ ] Reject mismatched projections with a clear error unless an explicit resampling option is enabled.
- [ ] Add tests for same-grid DEMs, nodata propagation, and projection mismatch.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DemDifference|Terrain" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add DEM difference reports"`.

### Task 4.4: Add DEM/LiDAR Soft Priors To BA

- [ ] Implement `ReferenceTerrainPrior` that samples a reference DEM or local height surface from a point cloud.
- [ ] Add BA residuals for height constraints with sigma, robust loss, and per-point enablement.
- [ ] Keep imported camera poses as soft priors by default. Do not fix them unless the user explicitly selects fixed pose mode.
- [ ] Report prior residual before and after BA.
- [ ] Add a synthetic BA test where noisy camera poses improve when a reference height surface is enabled.
- [ ] Run: `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ReferenceTerrain|Bundle|Sfm" --output-on-failure`.
- [ ] Commit: `git commit -m "feat: add reference terrain priors to BA"`.

### Task 4.5: Add User-Facing Reference Workflows

- [ ] Add GUI menu actions for `导入参考 DEM/LiDAR`, `点云/DEM 精度检查`, and `使用参考地形约束重新平差`.
- [ ] Show sigma, robust threshold, input path, output path, and expected effect before running.
- [ ] Log whether the reference is used for validation only or for BA constraints.
- [ ] Add project metadata for reference datasets without copying large files into the project by default.
- [ ] Run GUI smoke tests for opening dialogs, saving settings, and rejecting missing paths with actionable errors.
- [ ] Commit: `git commit -m "feat: add reference terrain workflows"`.

---

## Cross-Stage Verification

- [ ] Build Windows CUDA release:

```powershell
Set-Location E:/code/plascan
powershell -ExecutionPolicy Bypass -File scripts/build_win/build_windows_cuda.ps1 -BuildTests
```

- [ ] Run focused tests:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "Mvs|Depth|Fusion|Terrain|Dem|Dom|Sfm|Bundle|Quality|GuiProject|DataTree" --output-on-failure
```

- [ ] Run an agisoft smoke reconstruction with 2048 or 0.5 scale, then verify:
  - The GUI remains responsive after clicking start.
  - Cancel exits within 5 seconds outside a single CUDA kernel iteration.
  - Memory remains bounded after 50, 100, and 150 frames.
  - Depth records appear in the tree while the run is still active.
  - Depth artifacts include preview, raw depth, confidence, and valid mask.
  - Fusion can run from saved depth artifacts.
  - DEM/DOM products and quality rasters appear in the tree.

- [ ] Update documentation:
  - `README.md`: dense reconstruction workflow and Windows CUDA notes.
  - `docs/PROJECT_ARCHITECTURE.md`: MVS workspace, terrain products, QC reports.
  - `src/core/mvs/README.md`: depth manifest, source planning, streaming fusion.
  - `src/core/terrain/README.md`: DEM/DOM aggregation and mosaic.
  - `CHANGELOG.md`: version entry when user approves release.

---

## Recommended Release Milestones

- `v1.1.5-alpha.1`: Stage 1 complete and verified on agisoft smoke data.
- `v1.1.5-alpha.2`: Stage 2 complete with improved depth quality and source planning.
- `v1.1.5-alpha.3`: Stage 3 complete with terrain quality products.
- `v1.1.5`: Stage 4 validation reports and BA soft priors complete, changelog updated, tag pushed.

---

## Self-Review

- Spec coverage: The four requested stages are covered: stability, quality, terrain products, and reference terrain/QC.
- Placeholder scan: No placeholder task is left in this plan. Each task has concrete files, behavior, tests, and commit messages.
- Type consistency: Manifest, source planner, terrain product, QC report, and reference prior names are consistent across stages.
