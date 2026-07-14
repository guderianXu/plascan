# core/aerial_triangulation 模块

本模块管理 PlaScan 的对齐照片式空中三角测量流程，职责对应 Metashape 的 Align Photos。
它接收影像、相机先验、连接点/匹配缓存和用户空三参数，输出已定向相机、BA 质量指标、
正式稀疏观测成果和工作流报告。

模块边界：

- `AerialTriangulationWorkflow.*`：把 GUI/CLI 的用户级参数解析为服务级配置，记录最终生效设置。
- `AerialTriangulationService.*`：检查或补齐特征/匹配，构建候选匹配图，调用 SfM 和 BA，写出空三成果。
- `SfmPairPlanner.h` / `SfmMatchDiagnostics.h`：管理候选对规划和匹配图连通性诊断。
- `GuidedRematchService.*`：基于已注册相机和极线带做 guided rematching 候选补点。
- `MatchResultCatalog.*`：扫描已有 `.match` 与 sidecar，选择可用于空三的匹配 variant。
- `ReconstructionPrerequisiteReport.*`：为空三启动前的特征/匹配/连通性预检提供结构化结果。

## 连接点前端策略

空三流程的上游连接点补齐职责与“创建连接点”保持一致：当项目中没有可用连接点，
或用户选择“重置当前对齐”时，本模块会先按空三对话框里的连接点参数生成/刷新特征和匹配，
然后再进入 SfM 注册与 BA。

关键约定：

- `keypointLimit` 是连接点生成阶段的真实全局关键点上限。即使空三精度选择“低”，也不会再把
  连接点特征预算降到低精度预设值；低精度只影响 SfM 内部骨架匹配、图像缩放和保守阈值。
- `tiePointKeypointLimitPerMegapixel` 是可选的按百万像素上限。值为 0 时不启用按面积限制；
  启用后，每张影像实际上限为全局上限和按面积上限的较小值。
- SIFT + LightGlue 是空三默认连接点前端。为了接近 Metashape 的连接点密度，SIFT 在空三连接点
  阶段使用更低的检测阈值，避免普通纹理较弱的数据集只提取到几百个点。
- LightGlue 的匹配预算跟随连接点关键点上限，而不是跟随低精度 SfM 骨架预算。这样全量匹配、
  照片序列匹配和 guided rematching 都能看到完整的连接点候选。
- `tiepointLimit` 同时约束创建连接点输出和无相机 SfM 输入。SfM 按多视轨迹执行每影像限额，
  优先保留长轨迹与高置信度轨迹，不再绕过限制直接消费全部两两匹配。
- 无相机自适应焦距粗筛在注册率相同的候选之间比较摄影测量网质量，包括三角交会角、
  两视轨迹比例、观测空间覆盖和 RMS；不会再仅因约 0.01 px 的 RMS 优势选择弱基线模型。

稀疏质量门控把两视轨迹比例分为 advisory 和 blocking 两级。超过 0.70 会记录
`high_two_view_track_ratio`，超过 0.85 会记录 `too_many_two_view_tracks` 并阻止直接进入 MVS。

## 缓存兼容性

`.match` sidecar 和 `no_match_pairs.json` 会记录连接点前端签名，包括前端版本、关键点上限、
每百万像素限制和 SIFT 检测阈值。当前 SIFT + LightGlue 密集前端要求签名匹配；旧缓存或旧
no-match 记录不会被继续复用，避免“旧的少点匹配”阻止新前端重新生成候选对。

CLI 报告中的 `sfm_diagnostics.feature_keypoint_stats` 会输出本次实际加载的特征数量统计，
用于判断问题出在特征提取、匹配几何验证还是后续 SfM 注册。

非职责：

- 不负责 MVS、深度图、网格、DEM 或 DOM 生产。
- 不替代 `src/core/sfm` 内的 SfM 算法内核；本模块只编排调用。
- 不替代 `src/core/matchphototask` 的创建连接点流程；已有连接点会被本模块复用。
