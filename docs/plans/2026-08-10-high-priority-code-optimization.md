# PlaScan 高优先代码问题优化计划

更新日期：2026-08-10

复核基线：`0fb3b125`

状态：实现与本地门禁已完成（等待提交后的 GitHub required checks）

实施范围：SAFE-01～03、MODEL-01～02、DM-01～04、CONC-01～03、MVS-01～02、BUILD-01～02
与 CI-01 均已落地行为测试和实现；阶段 6 中与本轮 P1 无直接依赖的 GUI 分页及存储哈希去重仍保留为
后续维护项。最终完成状态以本文验证矩阵和 GitHub required checks 全部通过为准。

## 目标与边界

本计划用于修复代码审查中确认的数据安全、密集匹配正确性、并发稳定性、MVS 大项目扩展性和
模型/构建可复现性问题。实施顺序以“用户数据不丢失、计算结果可信、失败可恢复”为首要约束，
性能和结构优化必须建立在行为回归测试之上。

本计划包含 P1 问题，以及直接支撑 P1 修复的测试、缓存版本和构建门禁。以下工作不在本计划中：

- 不重写完整 GUI 或三维渲染架构；相关长期拆分继续由
  `docs/plans/2026-08-09-gui-code-optimization.md` 跟踪。
- 不借修复之机调整无关参数默认值、项目格式或摄影测量质量策略。
- 不机械拆分超大文件；只有被当前阶段触及且能够形成独立测试职责时才拆分。
- 不删除或重生成 `testData/`、`.plascan` 工程和 `resources/models/` 中的现有资源。

基线提交已经加强 LoMa-R 发布包的固定资产白名单、SHA-256/尺寸校验并禁止打包本机 `.engine`，但未修改
下列核心算法、数据清理、任务生命周期或模型导出器。发布包校验作为可复用基础保留，不能替代导出阶段的
provenance 和运行时校验。

## 风险清单

| ID | 优先级 | 问题 | 主要位置 | 目标阶段 |
|---|---|---|---|---|
| SAFE-01 | P1 | 相机转换输出为输入祖先时，覆盖模式可递归删除输入 | `src/core/camera/CameraFormatConverter.cpp` | 1 |
| SAFE-02 | P1 | 项目资源清理信任外部或越界元数据路径 | `src/core/project_workflows/ProjectResourceCleanup.cpp` | 1 |
| SAFE-03 | P1 | 点云细化输入输出同址时先截断源 PLY | `src/core/mvs/DenseCloudRefinementService.cpp` | 1 |
| MODEL-01 | P1 | “替换默认模型”未被消费，固定输出覆盖历史模型 | `src/core/mesh/ModelWorkflowService.cpp`、`src/gui/project/manager/ProjectModelManager.cpp` | 1 |
| DM-01 | P1 | 自研 BM/SGM 的视差参考坐标与验证、三角化相反 | `src/core/dense_match/CostFunctions.cpp` | 2 |
| DM-02 | P1 | 无效视差假设以零代价参与 WTA/SGM | `src/core/dense_match/CostFunctions.cpp`、`CostFunctions.cu` | 2 |
| DM-03 | P1 | 多方向 SGM 原地复用同一递推状态 | `src/core/dense_match/SgmMatcher.cpp` | 2 |
| DM-04 | P1 | 左右一致性检查沿用正向范围并比较了错误符号 | `src/core/dense_match/DenseMatchService.cpp`、`DisparityValidator.cpp` | 2 |
| CONC-01 | P1 | 裸 worker 异常或部分创建失败可触发 `std::terminate` | MVS、匹配和网格 worker | 3 |
| CONC-02 | P1 | 旧快照 prune 可删除刚复制但尚未发布引用的项目共享影像副本 | `ProjectSessionModel`、`ProjectSharedImageStore` | 3 |
| CONC-03 | P1 | 模型任务在项目/Chunk 切换和析构时未及时取消 | `src/gui/project/manager/ProjectModelManager.cpp` | 3 |
| MVS-01 | P1 | 超过 32 个共视视图时系统性丢失真实选源证据 | `src/core/mvs/DepthMapGenerator.cpp` | 4 |
| MVS-02 | P1 | 全量影像预载发生在内存策略之前 | `src/core/mvs/DepthMapGenerator.cpp` | 4 |
| MODEL-02 | P1 | TensorRT 中间产物复用未校验权重和导出配置 | LightGlue、LoMa-R 导出脚本 | 5 |
| BUILD-01 | P2 | Windows 工作流入口不解析 `.exe` | `scripts/workflows`、`scripts/bench` | 5 |
| BUILD-02 | P2 | GUI/Conda 关闭选项仍泄漏对应依赖和路径 | 根 CMake、`PlascanPackages.cmake` | 5 |
| CI-01 | P2 | 缺少 Windows 门禁，关键 Terrain 和 Python 测试未完整进入 CI | `.github/workflows/ci.yml`、`tests/CMakeLists.txt` | 5 |

## 实施原则和依赖

- 阶段 0 是所有修复的前置门禁；每项修复先在本地构造可失败测试，但不得把红色测试提交到主分支。
- 阶段 1、2、3 可在不同提交序列中并行实施，但每个提交必须独立通过受影响目标和测试。
- 阶段 4 复用阶段 3 的安全 worker/取消设施，避免为 MVS 再建立一套并发语义。
- 阶段 5 在核心正确性修复稳定后进入发布门禁；模型 provenance 可提前独立实施。
- 任何会改变视差、选源或模型结果的修复都必须升级对应算法/缓存 schema，禁止复用旧产物冒充新结果。
- 文件和目录写入统一采用同目录临时产物、完整校验、发布和失败回滚；不能先破坏正式产物再计算。
- 路径保护必须同时覆盖绝对/相对别名、大小写差异、符号链接/目录联接和不存在的末级路径。

## 阶段 0：固定基线和回归门禁

### 0.1 建立问题级行为测试

为每个 P1 建立最小确定性测试：

- 路径安全：输入输出等价、输出为输入祖先、`..`、符号链接/目录联接、项目外绝对路径和写入失败。
- 模型历史：连续生成两次时，“不替换”保留两个文件和记录，“替换”只更新显式默认项。
- 密集匹配：非重复纹理的正/负非零视差、左右边界、单点脉冲、左右一致性、方向排列和三角化重投影。
- 并发：worker 注入异常、部分线程创建失败、导入/prune 同步屏障、项目切换和析构期间取消。
- MVS：33/64 个全共视相机，断言每个参考视图都有真实 shared-track 候选且不走无证据兜底。
- 模型导出：更换权重、尺寸、精度或 checkpoint 后不得复用不兼容中间产物。

### 0.2 保存质量和资源基线

- 固定一个双目小场景、一个 64 视图合成场景和一个现有真实 MVS 场景。
- 记录视差有效率、坏点率、三角化重投影误差、每帧选源、深度覆盖率和最终点数。
- 记录峰值 RSS、预载耗时、选源耗时和 worker 数；基准结果写入 `docs/benchmarks/`。
- 记录现有 workspace/manifest schema 和模型 sidecar 字段，明确哪些阶段需要强制失效旧缓存。

阶段完成条件：所有问题都有能在修复前复现、修复后通过的行为测试设计和稳定夹具。

## 阶段 1：数据安全和事务式产物发布

### 1.1 统一路径安全边界

在 `src/common/io` 或等价的无 GUI 公共层提供小型路径工具，负责：

- 解析存在路径和不存在末级路径的规范化身份；
- 判断等价、祖先/后代和是否位于受管根目录；
- 拒绝卷根、项目根等过宽删除目标；
- 在 Windows 与 Linux 上处理大小写、符号链接和目录联接差异；
- 返回包含原始路径、规范化路径和拒绝原因的明确错误。

核心模块只依赖该公共层，GUI 只负责展示风险路径和获取必要确认。

### 1.2 修复四条产物路径

- `CameraFormatConverter`：删除前拒绝输出等于输入、输出为输入祖先或输出为危险根目录；先写 sibling staging
  目录。预检必须收集输入配置、解析源文件和全部影像依赖，输出包含任一输入依赖时拒绝覆盖；同时在写出前
  检查缺失影像和输出文件名冲突。校验相机、影像和清单完整后再发布。
- `ProjectResourceCleanup`：拆成 `buildCleanupPlan()` 和 `executeCleanupPlan()`，区分受管文件、记录专属目录、
  项目外路径、仍被保留记录共享的文件和拒绝路径。外部路径默认只移除引用；递归删除只允许带 run ID 或
  ownership manifest 的专属目录。受管产物先移动到项目内 `.trash/<transaction-id>`，持久化元数据成功后
  再清空，失败时按恢复清单回滚。
- `DenseCloudRefinementService`：规范化比较 input、output 和 report 三条路径；正式 PLY 使用临时文件写入并在
  header、点数和流状态校验成功后替换。多 pass 使用本轮唯一临时目录，避免并发运行共享确定性中间文件名。
- 模型工作流：将 JSON 布尔值转换为显式 `ModelOutputPolicy`。`CreateVersionedResult` 使用稳定 run ID/UUID 的
  独立目录并追加记录；`ReplaceDefault` 只在新模型验证成功后更新默认引用。纹理、MTL 和诊断报告归入同一
  run 目录，旧文件在新文件完整发布前保持可用。

### 1.3 兼容与迁移

- 不强制迁移现有模型记录；旧固定路径仍可打开和纹理化，首次再次生成时补充 run ID/default 标记。
- 对现有外部产品记录只停止自动删除，不擅自搬迁文件。
- 目录发布在 Windows 上使用 staging + backup + rename + rollback，不能假设覆盖已有目录的 rename 原子成功。

阶段完成条件：所有故障注入和路径别名测试通过；任何失败路径都不改变原输入、旧产品或有效元数据。

## 阶段 2：统一自研 BM/SGM 视差语义

### 2.1 固定公共约定

明确规定视差图属于左参考图像素，`d = x_left - x_right`，因此右图采样位置为
`right_x = left_x - disparity`；`minDisparity` 包含、`maxDisparity` 不包含。该约定与当前三角化及
OpenCV SGBM 路径保持一致，并写入 dense-match 公共接口注释和测试夹具。

### 2.2 修复代价体和有效性

- CPU/CUDA 代价函数统一计算 `left[x]` 与 `right[x-d]`，同步有效视差范围。
- 将裸 `CostVolume` 容器升级为携带最小视差、最大视差上界、cost slices 和有效性语义的结构，并提供统一的
  `validDisparityIndexRangeForLeftX()`。
- 无效假设使用有限高代价 sentinel，并维护显式 hypothesis/pixel valid mask；SGM 和 WTA 仍需主动跳过
  无效状态，不得用零、NaN 或 `inf-inf` 参与聚合。
- WTA 只遍历有效假设；没有有效候选、唯一性不足或左右一致性失败时必须清除 valid mask。
- CPU 与 CUDA 对同一小图输出的视差、有效 mask 和代价排序在规定容差内一致。
- 反向视差范围由正向 `[min,max)` 映射为 `[1-max,1-min)`，一致性检查使用
  `abs(d_left_to_right + d_right_to_left)`；不能沿用正向范围或比较同号差值。
- 对齐 CPU/CUDA 的 NCC 平坦窗口和 Ternary Census 有效样本分母语义；五种代价函数都进入 parity 测试。

### 2.3 修复 SGM 聚合

- 每个方向使用独立的路径递推状态，只把完成的方向代价累加到 aggregate volume。
- 首先实现容易对照的参考版本，再使用按行/对角线滚动 buffer 控制内存；不能用共享全图 `L` 换取表面性能。
- 增加方向顺序不变性、1/4/8 方向参考结果以及边界无效假设测试。
- 峰值内存目标为两份完整 cost volume 加扫描线缓冲，不能因 8 个方向增加为 8 份完整体积。

### 2.4 产物隔离

- 升级 dense-match 算法版本和任何包含视差语义的 workspace hash。
- 旧视差允许显式查看，但不得被新三角化流程自动当作已修复语义的缓存复用。
- 用固定场景比较 OpenCV SGBM、自研 BM、自研 SGM；不要求结果完全相同，但几何方向和误差必须一致。
- CUDA parity 门禁覆盖随机纹理、平坦窗口、边界、正负视差和不同核尺寸；validity 必须完全一致，普通代价
  误差不超过 `1e-5`，NCC 不超过 `1e-4`。

阶段完成条件：合成视差精确命中，边界不产生无效最优项，多方向结果不依赖遍历顺序，三角化重投影误差
不超过 `0.25 px`；无噪声平移图内部至少 99% 像素误差不超过 `0.5 px`，并达到阶段 0 的真实数据门限。

## 阶段 3：并发异常、资源发布和会话取消

### 3.1 安全 worker group

提供可复用的 RAII worker group/parallel helper：

- 优先使用 `std::jthread` 或等价的 join-on-destruction 机制；
- 捕获并保存首个 `std::exception_ptr`；
- 首次失败后停止派发并触发协作取消；
- 即使线程创建中途失败也保证所有已启动线程 join；
- join 完成后在拥有任务错误边界的线程重抛；
- OpenMP 循环也在循环体内捕获，异常不得穿过 OpenMP runtime；
- 进度回调保持单调，失败后不得继续写正式产物。

首批迁移 `DepthMapGenerator::preloadImages`、GeometryVerify、DepthCrossViewHoleRepair、PatchMatchCPU 和
DepthMapFusion；随后迁移 SurfaceReconstructor 和其余同类裸 worker。

### 3.2 共享影像引用发布

- 影像复制进项目共享库后立即创建 active reservation/lease，直到包含该 URI 的归档代次成功提交。
- import、引用发布和 prune 使用同一个项目级同步边界；prune 的保留集合基于“已提交引用 + active lease”。
- 真正孤儿先写入 tombstone，只有连续两个已提交代次均未引用且不存在 lease 时才物理删除。
- GC 与项目归档提交解耦；GC 失败只记录可重试警告，不得把已成功的项目保存改判为失败。
- 用测试屏障固定“文件复制完成、元数据尚未发布、旧快照开始 prune”的交错顺序。

### 3.3 模型任务生命周期

- 每个模型任务持有 task ID、项目路径、Chunk ID、session generation、取消令牌和隔离输出目录。
- 在推进 session generation 前，项目关闭、项目/Chunk 切换和 Manager 析构统一请求取消；UI 保持“正在取消”
  直到 worker 确认退出。
- 将现有 `isCancelled` 继续传入点云分支的 `MeshBuildRequest`、重建配置和 `SurfaceReconstructor`，在加载、
  预处理、重建、后处理和保存边界轮询，不能只覆盖深度图与纹理分支。
- 过期任务可以清理 staging 产物，但不得发布文件、写元数据、弹成功提示或覆盖新任务状态。

阶段完成条件：所有注入异常都转化为可定位错误而非进程终止；确定性竞态测试反复运行无丢图；会话切换后
旧任务不再产生或登记正式产物。

## 阶段 4：MVS 选源正确性和有界内存

### 4.1 替换全局 32 视图抽样

- 抽出可单测的 `MvsVisibilityGraphBuilder`，输出每个视图的可见点索引和
  `{viewIndex, sharedTrackCount}` 邻接表。
- 以“每个参考视图都必须参与”为不变量。共视数不超过 32 时保留全 pair；超过 32 时以每个可见视图为中心
  选择固定数量且确定性的 peer，并让起始相位随 point/view 变化，避免固定遗漏同一批索引。
- peer 优先级综合已验证 pair、序列邻近和基线/角度覆盖；已验证 pair 必须直接进入候选图。对 shortlist 再用
  visibility bitset 精确 popcount。
- worker 使用稀疏 pair map，最终合并为邻接表，移除全局及每 worker 的 `N×N` 整数矩阵。
- 没有验证 pair 时可以显式记录 fallback 原因，但不能因索引抽样而把已有证据变成不存在。

### 4.2 在解码前做内存规划

- 先读取影像尺寸/元数据并估算灰度图、准备图、mask、深度帧、保存队列和后端 staging 总预算。
- 小项目总量在预算内时保留 eager 模式；超预算时使用按 worker 数和源视图数确定容量的线程安全
  `MvsImageCache`。缓存提供单飞加载和 `ImageLease`，只常驻当前参考帧及其源视图。
- 相同底层 `cv::Mat` allocation 只计一次；缓存驱逐与正在计算/保存的帧通过 lease 管理。
- 内存不足必须给出所需、可用、当前策略和建议参数，不得依赖 `bad_alloc` 或静默降低质量。

### 4.3 质量和性能门禁

- 33/64 视图测试中每个参考帧都保留真实 shared-track 证据。
- 邻接图对称，1 worker 与 8 worker 结果一致；5000 视图构造测试不得出现 `N² × workerCount` 分配。
- 删除或改写固化 `_pairCommonCounts`、`VisibilityCacheShard` 成员名的源码字符串测试，以邻接图行为测试替代。
- 真实场景的深度覆盖率、融合点数和误差不得低于阶段 0 基线门限。
- 峰值 RSS 随视图数量保持受预算约束；记录 100、500、2000 视图的 pair-cache 估算或实测。
- 受管 MVS 内存峰值不超过计划预算，RSS 允许最多 10% 的分配器/第三方库余量；eager 小项目耗时回退
  不超过 5%，bounded 模式必须保持逐像素输出一致并在 1 秒内响应取消等待。
- 选源策略和缓存布局进入 workspace hash。基线中的 `kMvsDepthAlgorithmRevision` 为 28，实施时必须提升，
  旧批次不得无提示复用。

阶段完成条件：大视图场景不再因固定索引抽样退化，输入缓存峰值可解释且受配置/系统内存约束。

## 阶段 5：模型溯源、构建语义和 CI 门禁

### 5.1 模型中间产物 manifest

- LightGlue ONNX 和 LoMa-R feature/matcher 中间产物各自保存 sidecar manifest。
- 缓存键至少包含权重/checkpoint SHA-256、LoMa-R 源码 revision、模型标识、输入尺寸、动态 shape/profile、
  opset、精度、导出器 schema、Torch/ONNX/ModelOpt/TensorRT 版本和相关构建参数。
- 复用前验证 manifest 与文件哈希；缺失或不匹配时重新导出，matcher-only 模式不兼容时明确拒绝。
- 最终 TensorRT metadata 从实际被消费的 ONNX/engine 输入生成，不能直接抄写本次 CLI 参数。
- 增加唯一的 LoMa-R package composer，一次生成共享 `features_k3840`、`matcher_dynamic` 和三个 K 桶 manifest，
  禁止手工改名或手工修改 JSON。
- `MatchPhotosRuntime` 在构建 engine 前强制校验 manifest 中的 feature/matcher ONNX 哈希；matcher-only 必须读取
  旧 feature provenance，不能把旧 feature 标记成本次 checkpoint。

### 5.2 修复脚本和 CMake 开关

- 提供平台感知的 executable resolver，覆盖构建根、`bin`、`bin/{Release,Debug,RelWithDebInfo}`、`.exe` 和
  用户显式路径；同时接入全流程与 benchmark 脚本，测试不能再固化无后缀路径。
- `PLASCAN_BUILD_GUI=OFF` 时不构建 `plascan_gui`、`gui_project`，也不查找 Widgets、ShaderTools、
  ShaderToolsTools、GuiPrivate；保留 Core/CLI 实际需要的 Qt Core、Gui 和 Concurrent。
- 增加 `PLASCAN_BUILD_GUI_TESTS`，隔离 `test_gui_project_utils`、RHI smoke 等 GUI-only 测试。
- 计算唯一的 `PLASCAN_EFFECTIVE_CONDA_PREFIX`：OFF 优先且完全忽略 cache/环境；ON 时按显式参数、
  `CONDA_PREFIX` 选择。CUDA、include、binutils、RPATH 和配置摘要只读取该值。
- 将根 `cmake_minimum_required` 与 preset 和 `cmake_path` 的实际需求统一到 CMake 3.25。
- 为四个现有但未注册的 Python 模块增加 CTest 条目，并增加全部 `tests/test_*.py` 注册完整性检查。

### 5.3 补齐发布门禁

- 增加 Windows/MSVC CPU 构建和测试 job；CUDA/TensorRT 至少先建立可用 runner 上的编译与 smoke gate。
- 增加 GUI-off/Conda-off 配置 smoke，并用伪造 `CONDA_PREFIX` 证明关闭状态不泄漏环境路径。
- 修复并恢复 `TerrainPipelineGeneratesDemDomFromDirectory`，耗时大数据测试可另设定时 job，但保留必跑小夹具。
- Windows/Linux 发布打包必须依赖对应测试 job 成功；不能只验证 configure/build/package。
- 测试数据下载补充 SHA-256、长度和安全 tar 成员校验，防止损坏缓存和链接逃逸。

阶段完成条件：相同输入和权重得到可复现 provenance；关闭选项真实隔离依赖；Windows/Linux required checks
覆盖可发布配置且没有无解释排除的核心产品测试。

## 阶段 6：相关 P2 收尾

在前五阶段稳定后处理直接相关的用户体验和维护问题：

- 前方交会检查迁入带取消的后台任务，结果使用 `QAbstractTableModel` 分页或虚拟化展示。
- 模型“分割成区块”在后端实现前先禁用或明确标注未支持，不能继续提交无效参数。
- 拆分本计划触及的超大测试文件，将源码字符串契约替换为路径安全、异步生命周期和算法行为测试。
- 合并 `ProjectResourceStore` 连续重复的 SHA-256 校验，并记录导入目标 mtime，减少大文件打开开销。
- 为相机目录与密集点云覆盖发布增加跨进程崩溃恢复标记，关闭“旧产物移入备份、新产物尚未改名”之间
  被强杀时需要人工恢复备份的窗口；当前实现已经保证普通 IO 异常回滚，但不把双重 rename 宣称为断电原子。
- 为模型产物已经发布、元数据尚未最终提交的窗口增加启动时孤儿 run 扫描；继续保留当前所有权标记和
  会话代次保护，禁止无证明递归删除。
- CLI-only Dense 长任务补充统一取消令牌；MVS 已在解码、掩模、预处理与缓存等待边界检查取消，但单次
  OpenCV/GDAL 解码调用本身仍受第三方 API 的不可中断时长限制。

## 验证矩阵

每个实施阶段至少执行下列与范围匹配的检查；涉及 C++、CMake 或测试代码且准备 push 时，还必须运行当前
原生平台可执行的全量测试。

```powershell
cmake --preset windows-vcpkg-release
cmake --build --preset windows-vcpkg-release --parallel
python scripts\env\run_tests.py `
  --test-dir build\windows-vcpkg-release `
  --output-on-failure `
  -R 'DenseMatch|Mvs|Camera|Project|Model|Terrain|Gui'
python scripts\env\run_tests.py `
  --test-dir build\windows-vcpkg-release `
  --output-on-failure
```

模型和 Python 阶段额外执行：

```powershell
python -m py_compile `
  scripts\models\export_lightglue_tensorrt.py `
  scripts\models\export_loma_r_tensorrt.py `
  scripts\workflows\run_full_pipeline.py `
  scripts\bench\run_photogrammetry_benchmarks.py
python -B -m unittest `
  tests.test_full_pipeline_entrypoint `
  tests.test_photogrammetry_benchmark_runner `
  tests.test_ba_cuda_contracts `
  tests.test_compare_point_cloud_to_lidar `
  tests.test_dem_grid_to_height_ply `
  tests.test_verify_marker_pdf
```

构建选项阶段还需在伪造 Conda 环境下执行 headless smoke：

```powershell
$env:CONDA_PREFIX = "$env:TEMP\plascan-poison-conda"
cmake -S . -B build\headless-smoke -G Ninja `
  -DPLASCAN_ENABLE_CONDA=OFF `
  -DPLASCAN_BUILD_GUI=OFF `
  -DPLASCAN_BUILD_GUI_TESTS=OFF `
  -DPLASCAN_ENABLE_CUDA=OFF `
  -DBUILD_TESTS=ON
cmake --build build\headless-smoke --target reconstruct_pipeline_cli
```

Linux/GCC 使用对应 `linux-vcpkg-release` preset 在 Linux 环境或 CI 中执行。CUDA 改动还需使用
`windows-vcpkg-cuda-release` 构建，并在具备设备的环境运行 CPU/CUDA 一致性和模型 smoke test。

### 2026-08-10 本地验证结果

- Windows/MSVC + CUDA/TensorRT 完整构建通过。
- CTest 全量门禁 2411/2411 通过；此前仅剩的共享影像两代 tombstone 测试和 MVS 私有成员源码契约测试
  已修正，并分别定向复测 1/1 通过。
- Windows package smoke 通过，包含 BiRefNet 两项模型资产校验和 87 个运行时 DLL 校验。
- Headless smoke 通过：在 GUI、GUI 测试、Conda、CUDA、TensorRT 全关闭且 cache 注入伪 Conda 路径时，
  `PLASCAN_EFFECTIVE_CONDA_PREFIX` 为空、构建图无 GUI 目标，`reconstruct_pipeline_cli` 成功生成。
- Python `unittest discover` 115/115 通过；本轮 9 个 Python 脚本通过 `py_compile`。
- Dense Match/CLI 65/65、MVS 相关 95/95、项目清理/项目数据/模型生命周期组合 85/85 通过；CUDA
  代价函数与 CPU parity 门禁通过。
- LoMa-R K1024/K2048/K3840 三个 manifest 均为 644 字节，SHA-256 与 `models-v1.1.0` 发布清单一致。
- Linux/GCC 由提交后的 GitHub required checks 验证；其结果将在本计划完成状态中收口。

## 推荐提交顺序

1. 问题级回归夹具与基线记录。
2. 公共路径安全/事务发布工具。
3. 相机转换、项目清理和点云细化接入安全路径。
4. 模型输出历史语义和原子发布。
5. 视差坐标、有效假设和 CPU/CUDA 一致性修复。
6. 独立 SGM 路径聚合和缓存 schema 升级。
7. RAII worker group 及首批 worker 迁移。
8. 共享影像 prune 同步和模型任务 session 取消。
9. MVS 稀疏选源证据统计。
10. MVS 有界输入缓存和内存策略前移。
11. 模型中间产物 manifest 和导出缓存校验。
12. Windows executable、GUI/Conda 开关和 CTest 注册。
13. Windows/Linux CI、Terrain 门禁和测试数据完整性。
14. 相关 GUI 响应和测试文件收尾。

每个提交只承担一种主要风险，不把算法语义修改、持久化格式变更和大规模文件移动混入同一提交。

## 完成定义

- 所有已知 P1 都有行为回归测试，且故障注入不会破坏输入、旧产物或有效项目元数据。
- 自研 BM/SGM 使用统一左参考语义，边界和多方向聚合通过参考实现与 CPU/CUDA 一致性门禁。
- 左右一致性使用正确的反向视差范围和符号；五种 CPU/CUDA 代价函数通过 parity 门禁。
- worker 异常不会终止进程；项目/Chunk 切换后不存在旧任务写回或共享影像丢失。
- 33/64 视图场景不再丢失整组参考帧证据，MVS 峰值内存由显式预算控制。
- 模型引擎 provenance 与实际消费的权重、中间产物和构建参数一致。
- Windows/MSVC 本地全量测试通过，Linux/GCC 和 required CI 通过；CUDA/模型项在相应环境完成 smoke test。
- `CHANGELOG.md`、架构文档、模型文档和版本发布说明在对应实现阶段同步更新。
