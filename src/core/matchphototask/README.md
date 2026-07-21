# MatchPhotoTask

`matchphototask` 是类 Metashape “匹配照片”流程的编排层。
它拥有高层任务边界，但复用底层模块，不把其它模块搬进本目录。
它也是工作流中创建连接点的唯一所有者；`aerial_triangulation` 只能通过
`MatchPhotosTask` 请求特征、匹配和轨迹，不能直接实例化提取器或匹配器。

当前框架：

- `algorithm/` 负责类 Metashape 策略到具体算法计划的映射，当前主线固定为 `SIFT + LightGlue`。
- `pair_selection/` 负责影像对类型、影像对选择策略和 `PairSelector`。
- `runtime/` 负责运行期路径解析、LightGlue 模型查找、匹配 sidecar 写入、蒙版约束和 GUI 写回所需记录。
- `task/` 负责 `MatchPhotosTask`、选项、上下文和结果报告。
- `stages/` 负责流程阶段，当前已接入 SIFT 特征提取与 SIFT + LightGlue 两两匹配。
- `tie_points/` 负责最终多视图连接点 track 的构建、筛选和统计摘要。
- `tests/` 放置本模块自己的单元测试。

当前行为：

- `MatchPhotosAlgorithmSelector` 将自动/快速/高精度/困难纹理/CPU/CUDA 预设映射为 SIFT + LightGlue。
- SIFT 负责提供尺度和旋转鲁棒性，LightGlue 负责对 `.sift` 特征做学习型匹配。
- 连接点用途的 SIFT 检测阈值为 `0.0005`（快速模式为 `0.003`）；CUDA SIFT 使用等价的库内阈值映射。
  阈值写入前端签名，修改后旧缓存会自动失效。
- 特征阶段会复用已有 `.sift`；缺失或禁用复用时才重新提取，并把缩放后的关键点坐标还原到原始影像坐标。
- `maskApplyMode=keypoints` 时，特征阶段按项目蒙版过滤关键点和描述子，并强制重新提取特征；
  `maskApplyMode=tiepoints` 时，匹配阶段过滤任一端落入蒙版排除区的连接点。蒙版约定为 `0` 有效、非 `0` 排除。
- `maxKeypoints` 对应每张影像的关键点总量限制；调用方设置 `useExplicitKeypointLimit=true` 时，
  `0` 表示不限制。空中三角测量界面的“关键点限制”始终使用这一语义，不会因启用指导匹配而按像素数放大。
- `keypointLimitPerMegapixel` 是底层调用方可显式启用的独立限制。设置后，每张影像的关键点上限按
  `每百万像素关键点限制 * 影像百万像素数` 计算；它不由 `enableGuidedMatching` 自动开启。
- `enableGuidedMatching` 是显式用户开关，质量档位不能把未勾选状态自动改为开启；sidecar 与
  后续 SfM 必须记录和校验同一个实际值。
- 设备为 `Auto` 时优先尝试 CUDA SIFT 和 CUDA LightGlue，CUDA 不可用时允许回退 CPU；
  显式选择 `Cuda` 时不静默回退，模型或设备不可用会返回明确错误。
- 匹配阶段查找 `lightglue_sift_cuda.torchscript` / `lightglue_sift_cpu.torchscript`，写出 `.match` 和同名 `.json` sidecar，供 GUI 匹配查看器读取。
- 几何验证阶段使用 `MatchGeometryFilter` 的 USAC/MAGSAC 基础矩阵内点。为抑制重复结构伪造的弱几何边，
  少于 64 个内点的像对还必须达到 70% 内点率；强支持像对仍按最小内点数通过。原始匹配数、内点数和内点率会记入像对设置。
- `SiftLightGlueRecovery` 保留 LightGlue 主匹配链路和安全显存预算：高精度强重叠像对被
  LightGlue 输入预算截断时，按需用全量 SIFT 描述子增强；几何验证后的匹配图不连通时，
  只重试跨分量失败边，图连通后立即停止。恢复结果同步写回 `.match` 和 sidecar。
- 轨迹阶段通过 `TiePointTrackManager` 管理最终多视图连接点，并复用 `MultiViewTrackBuilder` 合并 track。
- 轨迹阶段成功后会写出 `assets/tie_points/latest_tie_points.json`，记录影像、track、观测点、
  参数摘要和统计信息，供项目下次打开或后续空三流程复用。
- `maxTiePointsPerImage` 对应连接点限制；
  `0` 表示关闭连接点数量稀疏。
- `excludeStationaryTiePoints` 会剔除在多张影像中像方坐标几乎固定的 track，用于过滤转台背景、
  传感器污点或镜头伪影类假连接点。
- `PairSelector` 合并来自手动输入、全量匹配、序列窗口、相机重叠、词汇召回和未来引导重匹配的候选影像对。
- 无相机通用预选由 `VocabularyOverlapRetriever` 调用 `OverlapPairGraphPlanner`，
  在词汇相似度候选上保留互选 TopK、补足单向 TopK、桥接连通分量，并固定补充序列窗口邻接候选，
  避免 BoW 图看似连通但实际匹配/SfM 有效图断裂。
- `MatchPhotosTask` 执行算法选择、影像对选择、特征提取、两两匹配、几何验证、轨迹构建和引导匹配报告；
  当前引导匹配 v1 通过关键点密度扩展提高初始匹配候选，尚未做姿态恢复后的二次补匹配。

`src/core/overlap` 保持为可复用的候选生成模块，由 `PairSelector` 调用；
它不会被改名，也不会被嵌入到本模块目录下。
