# MatchPhotoTask

`matchphototask` 是类 Metashape “匹配照片”流程的编排层。
它拥有高层任务边界，但复用底层模块，不把其它模块搬进本目录。
它也是工作流中创建连接点的唯一所有者；`aerial_triangulation` 只能通过
`MatchPhotosTask` 请求特征、匹配和轨迹，不能直接实例化提取器或匹配器。

当前框架：

- `algorithm/` 负责类 Metashape 策略到注册算法计划的映射，当前支持
  `auto_sift`、`plamatch_hct`、`sift_lightglue` 和 `loma_r`。
- `plamatch_hct` 是默认算法，自动设备顺序为 CUDA、OpenCL、CPU。显式选择的设备不可用时明确失败；
  Metal 当前不受支持。
- `auto_sift` 可显式选择，自动设备顺序为 CUDA、Metal、OpenCL、OpenCV CPU；不保留 `cuda_sift` 算法别名。
- `plamatch_hct` 是内置 CPU/CUDA/OpenCL 算法。任务缓存同时保存独立的 2,048 点 coarse 流、full 特征
  和 CPU 预建 HCTree；GPU 路径以任务级 batch 复用 full/coarse 描述子驻留。默认自动预选直接复现
  “对齐照片”的流程：PlaMatch coarse 候选图经 HCTree、局部一致性和最多 20 轮森林削减后，
  再与 Source/Estimated/Sequential 参考预选取并集；不再由词汇召回结果限制 PlaMatch 候选范围。
- `pair_selection/` 负责影像对类型、影像对选择策略和 `PairSelector`。
- `runtime/` 负责任务级 SIFT 内存缓存、ONNX 资源解析、本机 TensorRT engine 缓存、蒙版约束和 GUI 写回所需记录。
- `task/` 负责 `MatchPhotosTask`、选项、上下文和结果报告。
- `stages/` 负责流程阶段，按算法能力加载灰度或彩色输入并执行特征与两两匹配。
- `tie_points/` 负责最终多视图连接点 track 的构建、筛选和统计摘要。
- `tests/` 放置本模块自己的单元测试。

当前行为：

- `MatchPhotosAlgorithmSelector` 校验工作流选择的注册算法、设备要求与质量预设。
- “精度”与对齐照片使用相同的五档 API 值，只控制进入特征检测/匹配的首层采样：
  `最高=0`（构造 `(2W-1)×(2H-1)` 半像素格点）、`高=1`（原尺寸）、`中=2`、`低=4`、
  `最低=8`（后三档为整数步长抽样）。默认是“高”。该设置不改变关键点/连接点限制、像对预选、
  SfM 或 BA 参数；`maxImageDim` 为正时才作为独立的显式内存上限。
- SIFT 负责提供尺度和旋转鲁棒性，LightGlue 直接消费任务内存中的 SIFT 关键点与描述子。
- CUDA SIFT 检测阈值为 `0.0005`（快速模式为 `0.003`）；OpenCV CPU 使用独立的 contrast threshold
  `0.02`（快速模式 `0.04`），避免把两个实现的不同量纲混为一个参数。
  阈值写入前端签名，修改后旧缓存会自动失效。
- Auto SIFT 对最短边小于 800 像素的影像使用 2 倍首层和分级降低阈值；对超过质量档最长边的大图先做
  全图粗尺度，再在原分辨率重叠瓦片补充细节。候选经 8×8 空间配额、近邻去重和 RootSIFT 归一化后进入匹配。
- 特征阶段不再写入或读取独立特征文件。一次任务内每幅影像最多提取一次，描述子在最后一个相关像对完成后释放；
  多尺度和瓦片提取都会把关键点坐标还原到原始影像坐标。
- 特征阶段使用有界、保序的多 worker CPU 预取完成影像解码、缩放和蒙版准备。CUDA SIFT 使用两个影像
  流水线 worker：第三方 CUDA SIFT 的设备全局计数器仍由设备锁保护，但上一张影像的 CPU 筛选与
  RootSIFT 后处理可和下一张影像的 GPU 提取重叠。结果记录包含准备、流水线 worker、队列等待和提取耗时。
- `maskApplyMode=keypoints` 时，特征阶段只裁掉高置信度排除区内部的关键点，保留分割边界；
  `maskApplyMode=tiepoints` 时，匹配阶段把蒙版灰度作为软权重，仅硬裁确定排除区。蒙版约定为
  `0` 有效、`255` 确定排除，中间值表示排除概率。
- `maxKeypoints` 对应每张影像的关键点总量限制；调用方设置 `useExplicitKeypointLimit=true` 时，
  `0` 表示不限制。空中三角测量界面的“关键点限制”始终使用这一语义，不会因启用指导匹配而按像素数放大。
- `keypointLimitPerMegapixel` 是底层调用方可显式启用的独立限制。设置后，每张影像的关键点上限按
  `每百万像素关键点限制 * 影像百万像素数` 计算；它不由引导匹配策略自动开启。
- `guidedMatchingMode` 提供 `Disabled / Automatic / Forced` 三态。自动模式只补救几何支持度、
  内点率或空间覆盖偏弱的像对；强制模式处理全部具备可靠基础矩阵或可信参考位姿的像对。
  实际模式和参考位姿使用状态写入 `.pimatch` 配置指纹，后续 SfM 只消费同一结果变体。
- 设备为 `Auto` 时 SIFT 按 CUDA、Metal、OpenCL、CPU 选择；LightGlue 固定使用 TensorRT/CUDA；选择 CPU、CUDA
  不可用或 engine 不兼容时会返回明确错误，不做 TorchScript/CPU 回退。
- LightGlue ONNX 可通过 `lightGlueTensorRtEnginePath` 显式传入，也可由
  `PLASCAN_LIGHTGLUE_TENSORRT_ENGINE` 指定。运行时按 ONNX 内容、本机 TensorRT 完整版本和 GPU
  Compute Capability 构建环境指纹缓存；历史本机 engine 仅保留读取兼容。
- 准备 TensorRT engine 时会显示 ONNX 校验、缓存检查、解析、构建、I/O 校验和写入阶段；缓存未命中会
  给出格式升级或环境字段变化等具体原因。TensorRT 内核搜索没有可信百分比，该阶段使用忙碌进度并持续
  显示真实已耗时，同时每 30 秒写入一次控制台日志。
- 匹配阶段加载本机缓存的 TensorRT engine。最终结果由 `ImageMatchRepository` 对称提交为“一幅影像一个
  `.pimatch` 分片”，同一文件保存所有相邻影像，并按算法版本与配置变体隔离本影像观测、模型指纹、
  置信度和残差，避免不同特征编号空间互相覆盖。
- PlaMatch-HCT 在关闭引导匹配时直接采用算法内部的局部一致性结果，与“对齐照片”保持一致：不少于 8 条
  对应即接纳该像对，不再叠加 Fundamental/USAC 质量门。其它算法及启用引导匹配的 PlaMatch-HCT 仍使用
  `MatchGeometryVerifier` 的 USAC/MAGSAC 基础矩阵与连续支持度、内点率和空间覆盖质量门。
- GPU 匹配并发按算法独立解析：CUDA SIFT 使用与关键点数线性相关的双描述子缓冲模型，LightGlue 使用
  自身的二次 attention/相似度模型，LoMa-R 使用独立的 score 张量、描述子与 TensorRT workspace 模型。
  三者不共享经验公式。CUDA SIFT 每个像对 worker 使用独立非阻塞 stream，并在一次描述子上传后完成
  双向匹配；有效 worker 数、单 worker 估算显存和选择原因写入匹配记录。
- 轨迹阶段通过 `TiePointTrackManager` 管理最终多视图连接点，并使用 `ReferenceTrackBuilder` 复现“对齐照片”的
  并查集合并、重复影像观测清理、尺度静止点过滤和逐影像水位式空间选择。该过程不读取算法专属匹配分数。
- 轨迹阶段成功后会写出 `assets/tie_points/latest_tie_points.json`。当前 v3 格式把影像路径集中存于顶层，
  每个观测只保存固定数值行 `[image_id, feature_idx, x, y, scale]`，并继续为每条 track 保存几何验证后的
  `direct_edges`；空三据此区分原始像对匹配与多视轨迹的传递闭包，避免把未经直接验证的观测对用于两视三角化。
  v1/v2 文件仍可读取；v1 会按旧版闭包语义兼容处理，新生成缓存使用 v3 以减少大工程读写和解析开销。
- `maxTiePointsPerImage` 对应参考流程的单幅影像连接点目标值；`0` 表示关闭空间选择。最终轨迹是各影像选择结果
  的并集，并非旧 PlaScan 的每影像/网格硬配额。
- `excludeStationaryTiePoints` 会剔除在多张影像中像方坐标几乎固定的 track，用于过滤转台背景、
  传感器污点或镜头伪影类假连接点；距离阈值是成对关键点平均尺度的 4 倍，并要求尺度比不超过 2。
- Auto 模式启用任一预选时，所有正式算法共用 `PlaMatchHctPairPreselector`。PlaMatch-HCT 直接复用特征阶段产生的独立
  coarse MLDB/HCT 索引；Auto SIFT、SIFT+LightGlue 和 LoMa-R 从各自正式特征中按响应与空间覆盖选取
  最多 2048 点，并将浮点描述子确定性映射为 64-byte 排序签名。两条路径随后执行相同的 HCT 候选搜索、
  局部空间一致性、最小支持阈值和最多 20 轮骨架森林削减，不重新读取影像或提取第二套 coarse 特征。
- 参考预选同样集中在 `PlaMatchHctPairPreselector`，按 Source、Estimated、Sequential 语义与通用候选取并集。
  Source/Estimated 可从完整参考相机读取中心，也可由 CLI 的 `--reference-csv name,x,y,z` 直接提供位置；
  仅启用参考预选时不会为影像额外构造 coarse 特征视图。
  两种预选同时关闭时，Auto 模式跳过 coarse 视图构造和 HCT 预选，直接生成全部 `N(N-1)/2` 个像对；
  显式手工像对模式仍只使用用户提供的像对。`PairSelector` 只负责无需描述子的全量、序列和手动像对生成；
  旧词汇树、旧相机 footprint 和稀疏场景
  候选链已删除。GUI 的独立“重叠度分析”工具仍由 `src/core/overlap/OverlapAnalyzer` 提供，不参与匹配照片预选。
- `MatchPhotosTask` 执行算法选择、影像对选择、特征提取、两两匹配、几何验证、引导重匹配和轨迹构建。
  引导阶段用空间网格枚举初始基础矩阵极线带内的候选边，每条边只计算一次 RootSIFT 距离并同时维护双向
  top-2，随后执行互选和 ratio 门控。新增对应与原对应合并后只运行一次 USAC，并删除新增外点再更新统计。
  参考相机索引、可靠观测邻接和软蒙版在任务级缓存；最终日志分别报告 graph、policy、descriptor、mask、
  filter 和 geometry 累计耗时，便于判断瓶颈来自候选搜索还是几何验证。
- 任务成功构建轨迹后追加稳定的 `matching_funnel` 阶段报告，并在 `MatchPhotosResult::matchingFunnel` 和
  `trackSummary.matching_funnel` 中保留结构化统计。漏斗依次记录全量/入选像对、有原始匹配像对与匹配数、
  guided 前几何通过像对/内点、guided 新增内点、最终几何边和轨迹数，同时输出预选率、匹配产出率、
  几何像对留存率、几何内点率和 guided 增益率，用来区分“候选召回不足”、“特征匹配弱”与“几何退化”。

预选契约集中在 `pair_selection/PlaMatchHctPairPreselector`；`src/core/overlap` 只保留独立重叠度分析能力。
