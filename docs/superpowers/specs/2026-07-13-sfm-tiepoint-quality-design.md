# 无相机空三连接点质量优化设计

## 目标

让“连接点限制”真实约束无相机增量 SfM 使用的多视轨迹，同时优先保留长轨迹和空间分布均匀的观测；收紧两视轨迹质量门控，避免仅凭注册率和重投影误差将弱摄影测量网判定为可直接进入 MVS。

## 已确认根因

`AerialTriangulationWorkflow` 会把连接点限制传给创建连接点和空三服务。创建连接点使用 `MultiViewTrackBuilder` 完成轨迹级筛选，但无相机 `IncrementalSfm` 随后直接从全部 pairwise matches 构建 `CorrespondenceGraph`。当前 `maxKnownPoseTracksPerImage` 和网格限制只在已知位姿三角化分支生效，因此无相机 SfM 绕过了连接点限额。

最新 `small_test` 结果注册 9/9，但 BA 使用 44,333 条轨迹，最终 44,310 个点中有 39,213 个两视点，两视比例为 88.5%，轨迹长度中位数为 2。现有质量门控的默认拒绝阈值为 95%，因此仍返回 `acceptable_for_mvs=true`。

## 方案选择

### 采用：SfM 图构建前进行多视轨迹级筛选

在 `IncrementalSfm::run()` 构建对应关系索引前，使用现有 `MultiViewTrackBuilder` 从全部已验证匹配构建轨迹。按轨迹长度、置信度和空间网格顺序筛选，再让 `CorrespondenceGraph` 只保留属于获选轨迹的原始 pairwise edges。

该方案不会人为生成未经两视几何验证的新匹配，也不会逐个匹配文件独立截断而破坏跨影像轨迹。已知位姿分支继续在几何检查后执行现有轨迹筛选。

### 不采用：直接读取 `latest_tie_points.json`

该文件适合持久化和查看，但 JSON 体积大，且会让 SfM 依赖工作流缓存文件。核心 SfM 应能由内存输入独立工作，因此不把本地 JSON 作为必需输入。

### 不采用：每个影像对独立截断匹配

单对截断无法保证每影像总量限制，也会优先保留大量两视轨迹，破坏长轨迹和摄影测量网强度。

## 模块边界

- `CorrespondenceGraph` 增加稳定的影像对枚举和“按已选轨迹保留原始边”能力。
- 新增 `CorrespondenceTrackThinner`，负责从重建影像关键点与匹配图构建、筛选多视轨迹，并返回筛选统计。
- `IncrementalSfm` 只负责在未知位姿运行入口调用筛选器和记录诊断，不内嵌筛选实现。
- 空三服务将通用轨迹限额写入 `IncrementalSfmOptions`；字段从仅限 known-pose 的命名改为通用命名，不保留兼容字段。

## 参数语义

- `maxTracksPerImage <= 0`：不限制。
- `maxTracksPerImage > 0`：任一影像参与的获选轨迹数不得超过该值。
- `maxTracksPerGridCell > 0`：在影像尺寸可推导时，对每幅影像每个网格执行上限。
- 轨迹排序固定为：轨迹长度降序、置信度降序、稳定输入顺序升序。
- 筛选后只保留获选轨迹中原本存在且已通过几何验证的 pairwise matches。

## 质量门控

引入两级两视轨迹阈值：

- 两视比例大于 0.70：增加 advisory `high_two_view_track_ratio`，状态为 `warn`，但不单独禁止 MVS。
- 两视比例大于 0.85：增加 blocking warning `too_many_two_view_tracks`，`acceptable_for_mvs=false`，状态为 `blocked`。
- 其它既有失败条件继续作为 blocking warnings。
- JSON 同时输出 `advisories` 与 `warnings`，GUI 和 CLI 仍以 `acceptable_for_mvs` 作为是否阻止下游的稳定字段。

## 测试与验收

- 单元测试证明图筛选优先保留三视长轨迹，并严格满足每影像上限。
- 单元测试证明筛选只保留原始边，不生成不存在的影像对匹配。
- 单元测试证明 70% 到 85% 两视比例产生 advisory，超过 85% 阻止 MVS。
- 运行 `test_multiview_track_builder`、`test_sfm_pipeline`、`test_sfm_quality_report` 和空三工作流测试。
- 运行 `small_test`：注册影像必须保持 9/9；每影像输入 SfM 的轨迹数不得超过 4000；质量报告必须如实反映两视比例；记录点数、RMS、运行时间和峰值显存变化。

## 非本轮范围

- 自适应焦距评分重构。
- 转台/环拍软约束。
- CUDA 共享内参和控制点参数块。
- 全量 SIFT 恢复的 ANN 或分块 CUDA 优化。
