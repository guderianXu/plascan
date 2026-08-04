# core/sfm 模块结构

当前 `sfm` 目录按算法、后处理和项目适配三层组织：

- `common/`：SfM 公共类型和内部并查集。
- `geometry/`：投影、三角化质量和 OpenCV 相机转换；投影约定只在这里定义。
- `graph/`、`tracks/`：对应图、观测网络和多视轨迹。通用空间近邻统一使用 PlaPoint。
- `reconstruction/`、`pose/`、`triangulation/`：重建状态、PnP、增量三角化和初始稀疏点过滤。
- `pipeline/`：`IncrementalSfm` 对外保持一个入口，内部委托给初始像对、影像注册、已知位姿和 BA 协调组件。
- `quality/`、`filtering/`：纯 C++ 质量指标和 PlaPoint 稀疏点云后处理。
- `project/`：项目 JSON、控制点/标记适配、BA 输入构建和质量 JSON 序列化；Qt 仅允许出现在这一层。
- `test/`：SfM 模块自有单元测试。跨模块工作流契约仍位于仓库根 `tests/`。

构建目标与依赖方向如下：

- `sfm_core`：核心算法，不链接 Qt；依赖 Camera、Intersection、BundleAdjust、纯 C++ `control_network`、OpenCV 和 PlaPoint/PlaMatrix。
- `sfm_postprocess`：质量指标和稀疏点云后处理，不链接 Qt；依赖 `sfm_core` 和 PlaPoint。
- `sfm_project`：项目文件和 JSON 适配，可链接 Qt；依赖前两层。
- `sfm`：仅聚合以上三个目标的 `INTERFACE` target，不包含转发头、类型别名或兼容实现。

外部调用方必须直接包含真实模块路径，例如 `pipeline/IncrementalSfm.h`、
`triangulation/InitialSparsePointFilter.h`、`quality/SfmQualityMetrics.h` 或
`project/BaInputBuilder.h`。已删除的过滤器、旧命名和兼容别名不得恢复。

`IncrementalSfm` 是正式多视 SfM 主流程；`TriangulationService` 和
`InitialSparsePointFilter` 用于已有相机/匹配的预览或输入清理，不能替代影像注册、全局 BA 和正式质量门控。

## 无相机粗筛与正式精化

无相机文件且没有用户内参时，`AerialTriangulationPipeline` 使用两级搜索：

- 默认焦距先执行一次尝试，再评估
  `0.55、0.70、0.85、1.2、1.6、2.0、2.4、2.8、3.2、4.0、5.2、6.4、8.0、9.0、10.0`
  中尚未执行的尺度；低焦距候选注册完整也不会跳过窄视场候选。
- 每个候选持有独立 `IncrementalSfm`，粗筛执行配置由 `SfmSearchPolicy` 约束。
- 粗筛最多并行运行 4 个候选，并按总线程数为每个 worker 分配内部线程；结果按候选索引确定性归并，
  进度按全部候选的累计完成比例汇总。
- 粗筛阶段固定每个候选焦距，禁止共享焦距 BA 在候选内部漂移；正式重放确定最佳候选后，才按用户设置释放共享焦距。
- `SfmExecutionProfile::CoarseEvaluation` 将 BA 外层迭代限制为 5、全局精化限制为 1 轮，并把局部 BA 间隔放宽到 6 张影像。
- 候选排序首先比较注册影像数；注册率相同时综合三角交会角、多视轨迹比例、观测空间覆盖、重投影 RMS
  和有界闭环序列质量，避免用微小 RMS 优势选择弱基线模型，也避免单个异常相邻距离比支配排序。
- 正式阶段只重放最佳焦距，但不锁死粗筛初始像对；Guided matching 改变匹配图后必须重新自动选种子。
- 粗筛只读已经准备好的特征和匹配缓存，不写稀疏点云、项目记录或匹配质量报告。
- 初始对 E/F/H 估计和增量 PnP 使用由影像 ID 派生的稳定 RANSAC 种子；并行粗筛不会再改变正式 SfM 的随机状态。
- 标准 3D-2D PnP 始终先独立执行；只有标准求解失败时，照片序列插值/外推才作为恢复初值并使用独立的绝对内点门槛。严格相机中心距离门控仍由单独开关控制，恢复路径不会把未经真实 PnP 验证的相机写回项目。
- “照片序列”参考预选只控制匹配候选和首尾闭环候选，不自动启用相机中心距离先验。位姿序列门控属于独立运动模型，转台数据不能由匹配预选隐式开启。

小型固定焦距 BA 保持使用 Legacy CPU/OpenMP；共享焦距由 Ceres 在同一个非线性问题中与相机、三维点联合优化。
只有相机数和观测数达到 Auto 门槛时才选择 Ceres CUDA。当前 native CUDA 明确是点优化后端，
需要相机位姿或焦距优化时不会进入 Auto 候选。正式 BA 日志会记录相机、轨迹、观测、线程、状态、
实际后端、RMS、耗时和选择或回退原因。

## BA 规范与深度约定

- `SfmBundleAdjustCoordinator` 是 SfM 调用 BA 的唯一协调层。全局 BA 使用确定性锚点，并在无绝对控制时通过
  `SimilarityGaugeNormalizer` 恢复初始锚点和基线尺度，消除单目 SfM 的 7 自由度相似变换规范。
- 局部 BA 把窗口外但观测活动轨迹的已注册相机作为固定边界，避免新注册相机优化时拖动整个世界坐标系。
- 有控制点、比例尺或已知位姿约束时，绝对约束优先，不重复施加无尺度规范。
- 相机前后方判断统一调用 `Camera::positiveDepth()` / `isPointInFront()`；投影、三角化、BA 后过滤
  和 flipped-depth 相机不再各自解释原始相机 Z。
- BA 返回 `BASolveStatus` 和 `solutionUsable`。取消、数值失败或不支持配置不会回写相机和三维点。

## 已知相机与项目元数据

- GUI 项目元数据中的相机参数通常来自 EXIF/GPS 或前置估计，只作为增量 SfM 的相机初值和内参输入，后续 PnP/BA 允许调整位姿。
- 只有调用方显式提供完整 `.tsai` 相机文件列表时，才进入固定已知外参的直接三角化路径。
- 固定已知外参路径如果输入存在多视 track 但输出退化为全两视稀疏点云，应视为失败，不能发布为正式空三结果。

## 匹配配对规划与诊断

空三由 `AerialTriangulationWorkflow` 调用时，连接点阶段和 SfM 阶段共享显式的
`assetsDir` 与 `matchDir`。`matchDir` 指向 `assets/image_matches`，其中每幅影像只有一个
`.pimatch` 二进制分片；不存在独立特征目录。Workflow 解析出唯一的
`assets/tie_points/latest_tie_points.json`，`AerialTriangulationPipeline` 只消费该持久化连接点图，
不再回退扫描旧特征文件、成对匹配文件或 JSON sidecar。

## 输入连接点限额与观测唯一性

- 当达到 60 个几何内点的强匹配边已经覆盖并连通全部影像时，空三服务仅把该强连通核心送入 SfM；弱边只有在维持图连通性所必需时才保留，避免重复纹理的低支持边错误合并多视轨迹。
- 无相机环拍数据允许使用至少 12 个 3D-2D 观测注册桥接影像；原始候选少于 20 个时必须达到 0.80 内点率，不能通过绝对内点数宽松规则绕过小样本保护。
- 无相机增量 SfM 在构建 `CorrespondenceGraph` 索引前，使用 `CorrespondenceTrackThinner` 将已验证的两两匹配整理成多视轨迹。
- `maxTracksPerImage` 是每幅影像参与的多视轨迹上限。筛选顺序为轨迹长度、匹配置信度和稳定输入顺序，不按单个影像对独立截断。
- 筛选后只保留获选轨迹中原本存在的几何验证边，不合成未经验证的新 pairwise match。
- 一个二维特征只能归属一个三维点。增量三角化不得在轨迹延续失败后复用已经被其它三维点占用的观测。

质量报告对两视轨迹使用两级门控：比例超过 0.70 时输出 advisory，超过 0.85 时将 `acceptable_for_mvs` 置为 `false`。注册率和低 RMS 不能替代多视轨迹与交会角检查。

`.pimatch` 邻接块以统一的算法 ID、算法版本、配置指纹和模型指纹构成缓存键。邻接块中
`rawMatchCount == 0` 表示该算法配置已经确认该像对无匹配，可作为负缓存跳过重复推理；
版本或指纹不一致的分片会明确判为不可复用，不做隐式格式转换。

`src/core/aerial_triangulation/reconstruction/SfmPairPlanner.h` 描述大规模项目的 SfM 匹配候选规划。
实际连接点候选生成由 `matchphototask/PairSelector` 执行，大项目默认不做无约束 N^2 全匹配，
而是按以下来源生成候选并合并去重：

- `known_camera_overlap`：已知相机足迹重叠候选，优先级最高。
- `known_camera_spatial_neighbors`：已知相机中心邻域候选，用于跨航带/空间近邻补充。
- `sequence_window`：文件序列窗口候选，用于航线内连续影像。
- `manual_restricted`：调用方显式传入的配对列表。

每个候选 pair 会记录 `sourceTypes`、`priorityScore`、序列距离、相机中心距离和各来源得分。
`AerialTriangulationWorkflow` 把显式 `allowedPairKeys` 交给 `MatchPhotosTask`，因此高优先级 pair
会优先进入缓存检查、自动补匹配和后续诊断；SfM 本身不会重新生成匹配。

`MatchResultCatalog` 直接扫描 `.pimatch` 中的匹配数、几何内点、残差和覆盖率，并把像对规划结果
写入 `sfm_diagnostics.pair_plan`。GUI 工作流报告据此展示候选数和来源分布，判断当前数据是足迹重叠、
空间邻域还是顺序窗口在主导匹配，不再维护另一份匹配质量 sidecar 报告。
