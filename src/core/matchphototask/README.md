# MatchPhotoTask

`matchphototask` 是类 Metashape “匹配照片”流程的编排层。
它拥有高层任务边界，但复用底层模块，不把其它模块搬进本目录。

当前框架：

- `algorithm/` 负责类 Metashape 策略到具体算法计划的映射，当前主线固定为 `SIFT + LightGlue`。
- `pair_selection/` 负责影像对类型、影像对选择策略和 `PairSelector`。
- `runtime/` 负责运行期路径解析、LightGlue 模型查找、匹配 sidecar 写入和 GUI 写回所需记录。
- `task/` 负责 `MatchPhotosTask`、选项、上下文和结果报告。
- `stages/` 负责流程阶段，当前已接入 SIFT 特征提取与 SIFT + LightGlue 两两匹配。
- `tie_points/` 负责最终多视图连接点 track 的构建、筛选和统计摘要。
- `tests/` 放置本模块自己的单元测试。

当前行为：

- `MatchPhotosAlgorithmSelector` 将自动/快速/高精度/困难纹理/CPU/CUDA 预设映射为 SIFT + LightGlue。
- SIFT 负责提供尺度和旋转鲁棒性，LightGlue 负责对 `.sift` 特征做学习型匹配。
- 特征阶段会复用已有 `.sift`；缺失或禁用复用时才重新提取，并把缩放后的关键点坐标还原到原始影像坐标。
- `maxKeypoints` 对应关键点限制；调用方设置 `useExplicitKeypointLimit=true` 时，`0` 表示不限制。
- 开启 `enableGuidedMatching` 且设置 `keypointLimitPerMegapixel` 时，每张影像的关键点上限按
  `每百万像素关键点限制 * 影像百万像素数` 计算。
- 匹配阶段查找 `lightglue_sift_cuda.torchscript` / `lightglue_sift_cpu.torchscript`，写出 `.match` 和同名 `.json` sidecar，供 GUI 匹配查看器读取。
- 几何验证阶段使用 `MatchGeometryFilter` 过滤基础矩阵内点。
- 轨迹阶段通过 `TiePointTrackManager` 管理最终多视图连接点，并复用 `MultiViewTrackBuilder` 合并 track。
- `maxTiePointsPerImage` 对应连接点限制；
  `0` 表示关闭连接点数量稀疏。
- `excludeStationaryTiePoints` 会剔除在多张影像中像方坐标几乎固定的 track，用于过滤转台背景、
  传感器污点或镜头伪影类假连接点。
- `PairSelector` 合并来自手动输入、全量匹配、序列窗口、相机重叠、词汇召回和未来引导重匹配的候选影像对。
- `MatchPhotosTask` 执行算法选择、影像对选择、特征提取、两两匹配、几何验证、轨迹构建和引导匹配报告；
  当前引导匹配 v1 通过关键点密度扩展提高初始匹配候选，尚未做姿态恢复后的二次补匹配。

`src/core/overlap` 保持为可复用的候选生成模块，由 `PairSelector` 调用；
它不会被改名，也不会被嵌入到本模块目录下。
