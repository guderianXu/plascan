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

非职责：

- 不负责 MVS、深度图、网格、DEM 或 DOM 生产。
- 不替代 `src/core/sfm` 内的 SfM 算法内核；本模块只编排调用。
- 不替代 `src/core/matchphototask` 的创建连接点流程；已有连接点会被本模块复用。
