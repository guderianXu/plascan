# PlaScan 项目架构文档

行星表面摄影测量处理系统。最后更新: 2026-08-10。

## 顶层目录

```
plascan/
├── src/            # 所有源代码
│   ├── common/     # 通用工具库 (日志, IO, 模型与项目公共能力)
│   ├── core/       # 核心算法库 (相机, 特征, 匹配, 标记控制网, SfM, MVS, LiDAR, 蒙版, 网格, 地形)
│   └── gui/        # Qt6 图形界面
├── cmake/          # 全局 CMake 模块 (依赖查找、源码依赖 superbuild、运行时部署)
├── 3rdparty/       # git submodule：PlaMatrix、PlaPoint、Qt、OpenCV、GDAL 与算法依赖
├── resources/      # 静态资源 (深度学习模型权重, 图标)
├── scripts/        # 按 models/workflows/bench/env/validation 分类的辅助脚本
├── tools/          # 独立工具 (匹配转 CSV)
├── tests/          # 顶层测试
├── data/           # 示例/测试数据
├── docs/           # 设计文档, 规格说明, 计划
│   └── superpowers/
│       ├── specs/  # 功能规格
│       └── plans/  # 实现计划
├── src/cli/        # 按相机、特征、密集重建、重建流程和质量检查分组的命令行入口
├── docker/         # Docker 部署配置
├── CMakeLists.txt  # 根构建文件
└── CLAUDE.md       # AI 助手配置 (代码规范, 项目约定)
```

`scripts/env/run_tests.py` 是跨平台统一测试入口：默认使用全部逻辑线程并行执行 CTest，并允许通过
`CTEST_PARALLEL_LEVEL`、`--jobs` 或原生 CTest `--parallel/-j` 参数覆盖。

## 代码规范

- 单文件 ≤ 400 行，超则拆分
- 嵌套 ≤ 4 层
- Allman 花括号风格 (左花括号独占一行)
- 命名空间 `xjw::<模块名>`

---

## 一、common/ — 通用工具库

```
common/
├── DeterministicOpenCvRansac.h # OpenCV 鲁棒估计的稳定种子和并发临界区
├── log/
│   ├── Logger.h/cpp        # 全局日志单例 (LOG_INFO/LOG_ERROR/LOG_DEBUG 宏)
├── io/
│   ├── PathIO.h/cpp        # UTF-8/本机路径转换、字节读取和原子文件写入
│   ├── ImageIO.h/cpp       # OpenCV 解码及 TIFF/GeoTIFF 的 GDAL 直接读取
│   └── JsonObjectFile.h/cpp # JSON 对象的安全读取与原子写入
├── json/
│   └── JsonObjectMerge.h/cpp # 无业务语义的 JSON 对象深度合并
├── runtime/
│   └── PythonRuntimeLocator.h/cpp # Python 运行时路径解析（环境变量/.venv/兼容文件）
├── model/
│   ├── ModelFileResolver.h/cpp # 区分源码运行、安装包和用户数据目录的模型搜索/安装位置
│   ├── ModelAssetCatalog.h/cpp # GitHub Release 模型资产 URL、大小、SHA-256 和兼容性目录
│   ├── U2NetModelCatalog.h/cpp # U2Net ONNX 文件名、实际路径和安装状态
│   └── BiRefNetModelCatalog.h/cpp # BiRefNet Dynamic ONNX 文件名、实际路径和安装状态
├── project/
│   ├── ProjectIO.h/cpp # 项目目录、临时缓存、资源和产物路径规则
│   ├── ProjectArtifactIO.cpp # 基于规范化影像路径哈希的项目产物寻址
│   ├── ProjectAssetInspection.h/cpp # OBJ/PLY/XYZ 统计及 OBJ 材质/纹理依赖解析
│   ├── ProjectAssetImporter.h/cpp # Metashape/通用点云与模型复制、成果记录构建
│   ├── ProjectMetadata.h/cpp # 项目 JSON、影像 token 与资源路径解析
│   ├── ProjectChunkIndex.h/cpp # Chunk UUID、数字目录映射与只增不复用编号
│   ├── ProjectPackageLayout.h/cpp # 4.0.0 分体工程描述符严格解析
│   ├── PortableProjectFormat.h/cpp # 资源清单和项目 URI
│   ├── PlascanArchive.h/cpp # ZIP 归档封装、安全条目名和流式读写
│   ├── ProjectChunkStore.h/cpp # Chunk 索引、数字目录、格式门禁及持久化
│   ├── ProjectResourceStore.h/cpp # Chunk 资源导入、索引、校验和解析
│   ├── ProjectWorkspaceStore.h/cpp # 工程 URI 与当前 Chunk 路径互转
│   ├── ProjectSession.h/cpp # GUI/CLI 共用的无界面 4.0 工程会话
│   ├── ProjectLock.h/cpp # GUI/CLI 跨进程工程独占写锁
│   ├── ProjectSharedImageStore.h/cpp # 跨 Chunk 内容寻址共享影像库
│   ├── ProjectArchivePath.cpp # 归档条目与目标根目录安全校验
│   ├── ProjectMatchCatalog.h/cpp # image_match_results 逐影像分片索引与像对汇总
│   ├── SparseResultQuality.h/cpp # 稀疏结果类型和质量门控元数据
│   ├── ProjectCommonUtils.h # 项目通用内联工具
│   └── test/ # 项目公共能力的独立模块测试
├── string_utils/
│   ├── StringParsing.h/cpp # 文本中的 double 数值子串尽力提取
│   ├── StringTransform.h/cpp # ASCII 小写、空白裁剪和后缀比较
│   └── test/
│       ├── CMakeLists.txt # 字符串模块测试目标
│       ├── StringParsing_tests.cpp # double 数值子串提取测试
│       └── StringTransform_tests.cpp # ASCII 转换与比较测试
├── result/
│   └── OperationResult.h   # 操作结果包装 (成功/失败 + 错误信息)
└── CMakeLists.txt
```

---

## 二、core/ — 核心算法库

按模块组织，每个模块通过 `plascan_core_add_optional_module()` 注册到 `core/CMakeLists.txt`。

```
core/
├── CMakeLists.txt              # 注册所有子模块
│
├── camera/                     # 相机模型
│   ├── CameraModel.h/cpp       # 面阵/线阵/RPC 共享的只读像点、射线和空间点投影抽象
│   ├── FramePinholeCamera*.h/cpp # CameraModel 面阵实现及 Tsai/Brown-Conrady 状态、投影和文件 IO
│   ├── PlanetaryLineScanCamera*.h/cpp # CameraModel 线阵实现、USGSCSM ISD、逐行时间与月固系时变姿轨
│   ├── RpcCameraModel.h/cpp、RpcCameraCoordinates.cpp # RPC00B、WGS84 坐标转换、正反投影和近似射线
│   ├── RpcCameraImageCorrection.cpp # RPC 归一化影像仿射改正的校验与应用
│   ├── RpcCameraIO.h/cpp       # GDAL RPC metadata domain 与关联 RPC/RPB 旁车导入
│   ├── RpcBiasAdjustment.h/cpp # 基于控制点的 RPC 平移/归一化影像仿射偏差估计
│   ├── RpcStereoIntersection.h/cpp # 双 RPC 像点的 ECEF 迭代前方交会
│   ├── CameraBaseline.h/cpp    # 相机中心基线、指定点三角交会角和平均深度/基线比
│   ├── CameraFormatConverter.h/cpp # Middlebury/EPFL 等外部相机 -> tsai + image_camera.lis
│   ├── ColmapImageUndistorter.h/cpp # 复杂 COLMAP 模型的导入边界预去畸变
│   ├── ProjectCameraIO.h/cpp、ProjectRpcCameraIO.cpp # FramePinhole/RPC 项目元数据适配
│   └── test/                      # 相机测试与诊断程序
│       ├── FramePinholeCamera_tests.cpp
│       ├── RpcCameraModel_tests.cpp
│       ├── CameraBaseline_tests.cpp
│       ├── CameraFormatConverter_tests.cpp
│       ├── test_tsai_loader.cpp
│       └── test.cpp
│
├── camera_reference/           # 独立于解算相机的外部导航参考观测
│   ├── model/                  # 相机参考源、原始/已转换观测、杆臂和稳定 image_uuid 绑定
│   ├── io/                     # camera_reference_set.json 严格 schema 与 QSaveFile 原子读写
│   └── tests/                  # 缺文件、往返、损坏及高版本拒绝测试
│
├── inference/                  # 跨业务模块复用的推理基础设施
│   └── tensorrt/               # TensorRT 能力、ONNX Builder、环境指纹缓存、Session 与张量 ABI
│       ├── TensorRtCapabilities*.h/cpp # 编译态/运行态能力和 CPU stub
│       ├── TensorRtEngineBuilder*.h/cpp # ONNX 首次构建、精度选择和 engine 元数据
│       ├── TensorRtEngineCache.h/cpp # GPU/TensorRT/模型指纹隔离及损坏失效
│       └── TensorRtSession.h/cpp # CUDA 缓冲、张量形状校验和推理会话
│
├── image_matching/             # 唯一局部特征/匹配/持久化模块
│   ├── ImageMatchingAlgorithm.h/cpp # 可扩展算法接口、能力和版本契约
│   ├── ImageMatchingRegistry.h/cpp  # 算法注册入口；auto_sift / sift_lightglue / loma_r
│   ├── FeatureSet.h/cpp        # 任务内关键点与描述子；描述子不持久化
│   ├── ImageMatchTypes.h/cpp   # 观测、邻接变体、置信度、残差和标志位
│   ├── ImageFeaturePointFile.h/cpp # 逐影像 `.pifeature` 特征点几何目录，不含描述子
│   ├── ImageMatchFile.h/cpp    # 逐影像 `.pimatch` v1 唯一二进制读写器
│   ├── ImageMatchIndexFile.h/cpp # 与 payload 签名绑定的轻量 `.pidx` 邻接索引及增量缓存
│   ├── ImageMatchRepository.h/cpp # 对称写入、完整指纹键缓存和按影像查询
│   ├── sift/                   # Auto SIFT 唯一维护边界
│   │   ├── AutoSiftAlgorithm.h/cpp # 注册入口与统一提取/匹配调度
│   │   ├── SiftFeatureExtractor.h/cpp # 大小影像自适应、瓦片、筛选与 RootSIFT
│   │   ├── SiftLowTextureRecovery.h/cpp # 覆盖不足网格的 CLAHE 双通道补点
│   │   ├── SiftComputeBackend.h/cpp # 统一后端选择、可用性与设备端调用接口
│   │   ├── SiftCudaBackend.cpp      # cudaSift 提取与匹配封装
│   │   ├── SiftOpenClBackend.cpp    # OpenCL C 1.2 金字塔、特征与匹配运行时
│   │   ├── SiftOpenClKernels.h      # OpenCL 高斯/DoG/方向/描述子/匹配内核
│   │   ├── SiftMetalBackend.mm      # Apple Metal 命令编码、资源和结果读取
│   │   ├── SiftMetalKernels.h       # Metal 高斯/DoG/方向/描述子/匹配内核
│   │   └── SiftGuidedMatcher.h/cpp  # 基础矩阵约束的补充匹配
│   ├── lightglue/              # TensorRT LightGlue 固定桶推理与后处理
│   ├── sift_lightglue/         # CUDA SIFT + LightGlue 组合与注册实现
│   ├── loma_r/                 # TensorRT DaD/DeDoDe-G 特征与 LoMa-R 匹配
│   ├── geometry/               # USAC/MAGSAC 验证及逐匹配像素残差
│   └── tests/                  # 格式往返、损坏校验、注册和几何测试
│
├── intersection/               # 前方交汇精度检验
│   └── Intersection.h/cpp      # 多射线交汇解算 + 精度评估
│
├── overlap/                    # 重叠度分析
│   ├── OverlapAnalyzer.h/cpp   # 影像对重叠区域计算
│   ├── OverlapPairGraphPlanner.h/cpp # 无相机词汇召回后的连通影像对图规划
│   ├── HierarchicalVocabularyTree.h/cpp # 层次 K-means 视觉词汇树训练与根到叶量化
│   ├── VocabularyOverlapRetriever.h/cpp  # 基于已提取特征描述子的词汇重叠对检索
│   └── GroundBackProjector.h/cpp  # 地面投影
│
├── matchphototask/             # Metashape-like 匹配照片编排层
│   ├── algorithm/
│   │   ├── MatchPhotosAlgorithmPlan.h/cpp # 算法计划：默认 Auto SIFT，可选学习型匹配
│   │   └── MatchPhotosAlgorithmSelector.h/cpp # 类 Metashape 预设到算法计划的映射
│   ├── task/
│   │   ├── MatchPhotosTask.h/cpp    # 统一任务入口，完成算法选择、候选对、匹配和轨迹阶段
│   │   ├── MatchPhotosOptions.h     # 自动/快速/高精度/CPU/CUDA 等任务选项
│   │   ├── MatchPhotosContext.h     # 项目路径、输出目录、影像输入、取消和进度上下文
│   │   └── MatchPhotosResult.h      # 阶段报告、逐影像分片记录、像对诊断和错误信息
│   ├── pair_selection/
│   │   ├── PairTypes.h/cpp          # PairCandidate、PairSource、pair key 规范
│   │   ├── PairSelectionPolicy.h/cpp # 自动/全量/序列/手动等候选策略
│   │   ├── PairSelector.h/cpp       # 合并手动、全量、序列、相机重叠和词汇召回候选
│   │   └── SparseSceneOverlapAnalyzer.h/cpp # 由已有 SfM 稀疏点共视和相机视锥补充候选对
│   ├── runtime/
│   │   ├── MatchPhotosRuntime.h/cpp # ONNX/manifest 解析、本机 engine 准备和配置指纹
│   │   ├── MatchPhotosFeatureCache.h/cpp # 一次任务内的有界 SIFT 特征缓存
│   │   ├── MatchPhotosMaskSupport.h/cpp # 连接点流程蒙版路径解析、关键点/连接点过滤
│   │   └── MatchPhotosParallelism.h/cpp # CUDA 显存预算、LightGlue worker 和几何验证并发解析
│   ├── stages/
│   │   ├── FeatureStage.h/cpp       # SIFT 自动设备提取到任务内存，不生成特征文件
│   │   ├── MatchingStage.h/cpp      # Auto SIFT 或 TensorRT 匹配，生成任务内像对结果
│   │   ├── GeometryVerifyStage.h/cpp # 调用 MatchGeometryVerifier 并填入残差/标志
│   │   ├── TrackBuildStage.h/cpp    # 连接点轨迹阶段边界，委托 tie_points 管理最终多视图 track
│   │   ├── GuidedMatchStage.h/cpp   # 三态 SIFT 双向引导重匹配及任务调度
│   │   ├── GuidedMatchPolicy.h/cpp  # 弱像对、H/F 退化和自适应核线带策略
│   │   └── ReferencePoseEpipolarGeometry.h/cpp # 可信参考相机 E/F 推导与一致性检查
│   ├── tie_points/
│   │   └── TiePointTrackManager.h/cpp # 最终多视图连接点 track 构建、筛选和统计摘要
│   └── tests/                       # matchphototask 模块级测试
│
├── control_points/             # Metashape-like 标记点与测绘控制网络
│   ├── model/                   # MarkerSet、投影状态、控制/检查点和比例尺
│   ├── io/                      # marker_set.json、CSV 和旧 survey_control 单次迁移
│   ├── commands/                # 可撤销 MarkerChangeSet
│   ├── detection/               # AprilTag/非编码检测、合并和 detection_review.json
│   ├── geometry/                # 三角化、预测投影与亚像素几何
│   ├── reference/               # CRS、轴序和坐标转换
│   ├── registration/            # PriorTrack、绝对定向和控制网络解算；control_network target 无 Qt
│   ├── quality/                 # 投影、控制点、检查点和比例尺质量报告
│   ├── print/                   # 共享标靶页面渲染与 PDF 输出
│   └── README.md                # 工作流、支持族、sidecar 和 CLI 说明
│
├── bundle_adjust/              # 光束法平差
│   ├── BundleAdjustTypes.h     # 后端、状态、能力和问题规模
│   ├── BundleAdjustProblem.h   # 观测、轨迹以及 GCP/比例尺/LiDAR 约束
│   ├── BundleAdjustOptions.h   # 数值、标定、后端选择与任务控制配置
│   ├── BundleAdjustResult.h    # 相机/点结果及跨后端诊断统计
│   ├── BundleAdjustSolver.h + BundleAdjust.cpp # BA 求解器门面、自动后端选择和统一质量门控
│   ├── BundleAdjustAdaptiveCameraModel.h/cpp # 基于粗解几何、像面覆盖和约化信息矩阵的逐内参可靠性策略
│   ├── BundleAdjustProjection.h/cpp # 与 FramePinholeCamera 一致的模板投影模型和共享相机快照转换
│   ├── BundleAdjustPlaMatrix.h/cpp # PlaMatrix 联合相机/点/内参 Schur-LM 后端
│   ├── BundleAdjustPlaMatrixProblem.h/cpp # 活动轨迹、标定组、固定块和工作集映射
│   ├── BundleAdjustPlaMatrixModel.cpp # 分组 Brown 内参初始化、边界、阶段与结果发布
│   ├── BundleAdjustPlaMatrixProjection.h/cpp # Brown-Conrady 残差及解析相机/点/内参雅可比
│   ├── BundleAdjustPlaMatrixConstraints.h/cpp # GCP、LiDAR、比例尺、姿态和平面约束
│   ├── BundleAdjustPlaMatrixAssembly*.h/cpp # 完整目标、重投影/测量约束与多块法方程装配
│   ├── BundleAdjustPlaMatrixRuntime.h/cpp # CUDA/OpenCL 设备可用性和显式设备索引校验
│   ├── BundleAdjustValidation.h/cpp # 输入、标定组和 gauge 校验/规范化
│   ├── BundleAdjustQuality.h/cpp # 跨后端正深度、离群点统计和物方约束质量门控
│   ├── README.md               # 调用链、后端能力、状态、规范与质量门控
│   ├── tools/                  # BA 后端基准及真实 sfm_sparse_points/TSAI 重放加载器
│   └── tests/                  # BA 模块级后端、投影模型、自动选择和约束回归测试
│
├── lidar/                      # LiDAR / 激光点约束
│   ├── LaserConstraintTypes.h  # 点到面约束、地图采样和关联统计类型
│   ├── LaserConstraintMap.h/cpp # PLY 点云读取、法线/曲率筛选、最近平面查询
│   ├── LaserConstraintAssociation.h/cpp # BA track 与 LiDAR 平面约束关联
│   ├── PlanetaryLaserShot.h/cpp # 稀疏行星测距数据集、Fixed/Constrained/Free 落点和严格语义校验
│   ├── PlanetaryLaserJson.h/cpp # PlaScan SI JSON v1 严格解析及 ISIS 导入上下文接口
│   ├── PlanetaryLaserIsisJson.cpp # ISIS LidarData 单位换算、球面协方差转 XYZ 和虚拟像点保护
│   ├── PlanetaryLaserBaAdapter.h/cpp # shot/影像唯一映射、frame/杆臂检查及独立 BA 约束装配
│   ├── IsisControlNetworkPvl.h/cpp # ISIS 文本控制网、PointType 和一基像素量测解析
│   ├── PlanetaryLineScanBundleAdjust*.h/cpp # USGSCSM 推扫控制网、逐行投影和 PlaMatrix 稀疏 range 联合 BA
│   └── tests/                  # SI/ISIS 解析、协方差转换、关联歧义、模式边界与安全拒绝测试
│
├── mask/                       # 照片蒙版生成与合成
│   ├── MaskGenerator.h/cpp     # 黑背景/亮度阈值蒙版、蒙版合成和轮廓提取
│   ├── InteractiveMaskAlgorithms.h/cpp # 矩形、连通域魔棒、颜色感知画笔与边缘吸附路径
│   ├── u2net/                  # U2Net ONNX 自动蒙版子模块
│   │   ├── U2NetMaskGenerator.h/cpp # Auto/TensorRT/OpenCV CPU 后端选择、回退策略与状态
│   │   ├── U2NetTensorRtBackend.cpp # ONNX 首次构建/复用本机 engine 并执行 GPU 推理
│   │   ├── U2NetOpenCvCpuBackend.cpp # 永久 CPU-only 的 OpenCV DNN 回退实现
│   │   └── U2NetImageProcessing.h/cpp # 两后端共享的预处理和掩膜后处理
│   └── birefnet/               # BiRefNet Dynamic 推荐自动蒙版子模块
│       ├── BiRefNetMaskGenerator.h/cpp # 固定 1024、CUDA/CPU 自动选择、能力与结果元数据
│       ├── BiRefNetOnnxRuntimeCpuBackend.cpp # 无 CUDA 时可完整执行发布模型的 ONNX Runtime CPU 推理
│       ├── BiRefNetInferenceBackend.h # TensorRT 后端接口和实际设备/精度/engine 元数据
│       ├── BiRefNetTensorRtBackend.cpp # FP16/FP32 本机 engine 首建、缓存复用与 GPU 推理
│       └── BiRefNetImageProcessing.h/cpp # ImageNet+letterbox、raw logits sigmoid 与原尺寸恢复
│
│   AI 蒙版发布边界：安装包只携带可移植 ONNX（BiRefNet 另带 provenance）；TensorRT engine 在目标机
│   首次使用时分别写入用户本地应用数据目录的 `models/{u2net,birefnet_dynamic}/engines/<fingerprint>`，
│   不写回安装树也不进入安装包。U2Net 的 OpenCV 不链接 CUDA/cuDNN；BiRefNet CPU 使用随程序部署的
│   ONNX Runtime，Windows 便携包不分发 cuDNN DLL。
│
├── sfm/                        # Structure-from-Motion
│   ├── common/                 # SfM 公共类型和共享并查集
│   ├── geometry/               # 统一投影、三角化质量和 OpenCV 相机适配
│   ├── graph/
│   │   ├── CorrespondenceGraph.h/cpp      # 对应关系图
│   │   ├── CovisibilityPartitioner.h/cpp  # 确定性加权共视图重叠分块
│   │   └── ObservationNetworkBuilder.h/cpp # PlaPoint KDTree 观测网络构建
│   ├── tracks/
│   │   ├── MultiViewTrackBuilder.h/cpp     # 多视轨迹合并、冲突消解和长轨迹优先筛选
│   │   └── CorrespondenceTrackThinner.h/cpp # 按每影像/网格限额筛选 SfM 输入匹配图
│   ├── pose/PnpSolver.h/cpp    # PnP 位姿解算
│   ├── triangulation/
│   │   ├── Triangulator.h/cpp  # 基础三角化
│   │   └── InitialSparsePointFilter.h/cpp  # 已有相机/轨迹的初始稀疏点过滤
│   ├── reconstruction/SfmReconstruction.h/cpp  # SfM 重建器
│   ├── pipeline/               # IncrementalSfm 编排及初始对/注册/已知位姿/BA 组件
│   │   ├── HierarchicalBaBlockSolver.h/cpp # 块问题装配与跨块固定轨迹约束
│   │   ├── HierarchicalBundleAdjuster.h/cpp # 共视分块并行求解、唯一写回与全局门控
│   │   ├── SfmCalibrationPreviewSampler.h/cpp # 自标定候选的相机/像面半径均衡抽样
│   │   └── SfmBundleAdjustCoordinator.h/cpp # 局部/全局 BA 调度、镜头种子与结果写回
│   ├── quality/                # 无 Qt 的稀疏重建质量指标
│   ├── filtering/              # PlaPoint 稀疏点云工作区和后处理
│   ├── project/                # Qt JSON、控制点/标记和 BA 输入适配
│   ├── ReferenceTerrainPrior.h/cpp # 参考 DEM/LiDAR 局部地形面作为 BA soft prior
│   ├── TriangulationService.h/cpp  # 项目级预览三角化服务
│   └── test/                   # SfM 模块自有 GTest
│
├── mvs/                        # Multi-View Stereo：深度图 manifest、source planning、流式融合
│   ├── MvsTypes.h              # MVS 公共类型
│   ├── DenseCloudRefinementService.h/cpp # 流式 PLY 多轮细化与内存回退，供 CLI/工作流复用
│   ├── StreamingDepthFusionService.h/cpp # 融合窗口、帧缓存、共识配置和分批聚合编排
│   ├── PointCloudArtifactIO.h/cpp # 稠密点云 PLY 目录创建、法向策略和二进制写出
│   ├── MvsWorkspaceManifest.h/cpp # 深度帧状态、产物路径、相机/影像/配置 hash、source plan 与几何来源位序
│   ├── MvsSourcePlanner.h/cpp  # shared tracks / 几何内点 / 覆盖率 / baseline 选源及严格失败像对复核
│   ├── MvsImagePreprocessor.h/cpp # 原图与 valid mask 共用去畸变映射，并生成正深度、零畸变工作相机
│   ├── MvsImageMetadataProbe.h/cpp # 不解码像素的 GDAL 影像头尺寸探测，供全流程内存规划
│   ├── MvsImageCache.h/MvsImageCache.cpp/MvsImageFrame.cpp # provider、single-flight、RAII lease 与分配去重
│   ├── DepthPyramidPolicy.h/cpp # 从最终质量档生成 4D/2D/D 三级 PatchMatch 调度
│   ├── MvsSceneClassifier.h/cpp # 根据相机布局、光轴和稀疏云判定航测/环拍/通用场景
│   ├── DepthPyramidPropagation.h/cpp # 父层深度中心、不确定半径和边缘感知传播
│   ├── DepthPyramidEstimator.h/cpp # Level 3/2/1 编排与逐层摘要
│   ├── DepthCompletenessMetrics.h/cpp # 蒙版内覆盖、小孔/大开口/边界缺失与逐阶段保留率
│   ├── DepthMissingReason.h/cpp # 逐像素缺失原因码、分类汇总和透明诊断预览
│   ├── DepthGapTargetedRecovery.h/cpp # 以邻近实测深度为先验的缺口定向二源 PatchMatch 与合并门控
│   ├── DepthResidualReestimation.h/cpp # 一致性后冻结多视深度层引导的残余空洞局部实测恢复
│   ├── DepthEvidenceConfidence.h/cpp # 独立光度/几何置信度与修正像素因果证据摘要
│   ├── DepthCrossViewHoleRepair.h/cpp # 一致性过滤后以三源逆深度簇保守补回环拍对象缺面
│   ├── DepthProvenance.h/cpp # 最终有效深度的原生/定向/跨视测量/学习候选门控来源编码与统计
│   ├── LearnedDepthCandidateGate.h/cpp # 学习型候选深度的置信度、稀疏几何及多视一致性最终门控
│   ├── DepthFrameQualityGate.h/cpp # 深度帧质量门控及置信度与绝对几何残差联合标定
│   ├── DepthFrameQualificationPolicy.h # GUI/模型/点云共享的主融合帧资格与环拍覆盖数量契约
│   ├── DepthConsistencyCache.h/cpp # 有内存预算的 LRU source 邻域多视一致性缓存
│   ├── DepthMemoryPolicy.h/cpp # 图像/深度/可见图/保存队列/后端 staging 饱和估算与 eager/bounded 决策
│   ├── DepthGeometryConsistency.h/cpp # 断边邻域搜索、相机基线自适应往返验证与一致性投票
│   ├── DepthPoseAlignmentRefiner.h/cpp # 锚定尺度的鲁棒点到平面局部 SE(3) 派生位姿细化
│   ├── DepthPoseRefinementStage.h/cpp # 默认关闭的跨视深度候选采样、安全门与派生相机输出
│   ├── PatchMatchEstimator.cpp  # PatchMatch 公共校验、后端选择和回退
│   ├── PatchMatchHostUtils.h/cpp # 三后端共享的 double 局部相对位姿与无效值感知、尺度稳定深度滤波
│   ├── PatchMatchPhotometricCost.h # 曝光鲁棒强度 NCC、梯度 NCC 与 Census 组合代价
│   ├── PatchMatchCPU.cpp        # 可独立构建的 CPU 组合代价 PatchMatch 实现
│   ├── PatchMatchCUDA.cu/h     # 按设备隔离工作区与图像缓存的 CUDA 组合代价实现
│   ├── PatchMatchNoCUDA.cpp    # 无 CUDA 构建的 GPU 接口存根
│   ├── PatchMatchOpenCL.cpp     # 跨厂商 OpenCL GPU 深度假设搜索、设备枚举与运行时缓存
│   ├── PatchMatchOpenCLKernels.h # OpenCL C 1.2 多源组合光度代价/深度细化 kernel
│   ├── PatchMatchNoOpenCL.cpp   # 无 OpenCL 构建的稳定接口存根
│   ├── DepthComputeScheduler.h/cpp # CPU/CUDA/OpenCL 统一 worker、文件名自然顺序与异构帧调度
│   ├── GpuDeviceLease.h/cpp     # 按 PCI 物理设备标识实施跨 GUI/CLI 进程的 GPU 独占租约
│   ├── DepthMapGenerator.h/cpp # 深度图估计、取消检查、raw depth/confidence/几何支持度/valid mask 写盘
│   ├── MvsVisibilityGraphBuilder.h/cpp # 稀疏共视图、可取消精确 bitset 计数及大视图集有界角度覆盖采样
│   ├── DepthMapFusion.h/cpp    # 深度图融合；流式窗口可用 CUDA/OpenCL 反投影，几何一致性仍在 CPU
│   ├── DepthFrameUtils.h/cpp   # 深度帧存储与按指定输出目录选择批次
│   ├── EpipolarRectifier.h/cpp # 极线校正、工作相机深度范围转换及原相机 Z_cam 回投
│   ├── DisparityTriangulator.h/cpp  # 视差三角化
│   ├── DensePointCloudCUDA.h/cu # CUDA 深度图反投影
│   ├── DensePointCloudOpenCL.h/cpp # OpenCL 深度图反投影与设备查询
│   ├── DensePointCloudNoOpenCL.cpp # 无 OpenCL 构建的稳定接口存根
│   ├── DenseCloudBuilder.h/cpp # CPU/CUDA/OpenCL 密集云构建、实际后端报告与点云过滤
│   ├── SparseCloudPreprocessor.h/cpp  # 稀疏云预处理
│   └── tests/                  # MVS 单元与流水线测试
│       ├── test_mvs_rectifier.cpp
│       ├── test_mvs_depth_pyramid.cpp
│       ├── test_mvs_depth_completeness.cpp
│       ├── test_mvs_depth_pose_alignment.cpp
│       ├── test_mvs_types.cpp
│       └── test_mvs_pipeline.cpp
│
├── project_workflows/          # GUI/CLI 共享的项目级摄影测量工作流配置与资源适配
│   ├── MvsSourcePairQualityLoader.h/cpp # `.pimatch` 几何审计到 MVS source pair 质量的统一桥接
│   ├── PointCloudInputPreparation.h/cpp # 正式稀疏点云加载、过滤与 MVS 输入准备
│   ├── PointCloudWorkflowConfig.h/cpp # 点云/深度质量档位到核心配置的统一转换
│   └── ProjectWorkflowOperations.h/cpp # 稀疏点后处理与地形产品等项目工作流入口
│
├── dense_match/                # CPU/CUDA/OpenCL 立体密集匹配
│   ├── README.md               # 模块文档
│   ├── DenseMatchTypes.h       # 计算后端、代价、算法和子像素枚举
│   ├── DenseMatchConfig.h      # 参数配置结构体
│   ├── DenseMatchBackend.h/cpp # auto/cpu/cuda/opencl 解析、可用性与严格后端选择
│   ├── CostFunctions.h/cpp/cu  # CPU/CUDA 五种代价及 CUDA WTA/置信度/子像素
│   ├── OpenClCostFunctions.cpp # OpenCL 1.2 设备、运行时缓存与数据调度
│   ├── OpenClCostKernels.h     # OpenCL 代价卷、WTA、置信度和抛物线子像素 kernel
│   ├── BlockMatcher.h/cpp      # 设备驻留的 CUDA/OpenCL WTA 块匹配
│   ├── SgmMatcher.h/cpp        # GPU 代价/选择 + CPU 8 方向路径聚合
│   ├── SubpixelRefiner.h/cpp   # 子像素视差精化 (抛物线拟合)
│   ├── DisparityValidator.h/cpp # L-R 一致性/中值滤波/Speckle 过滤
│   ├── DenseMatchService.h/cpp # 服务层: 编排完整匹配流水线
│   ├── opencv/
│   │   └── OpenCVSgbmWrapper.h/cpp  # OpenCV SGBM 封装 (对比算法)
│   └── tests/                  # 后端解析、严格失败与 CPU/GPU 数值一致性测试
│
├── mesh/                       # 网格重建与纹理映射
│   ├── MeshTypes.h             # 网格类型
│   ├── SurfaceReconstructor.h/cpp           # 表面重建主流程
│   ├── SurfaceReconstructorHeightGrid.h/cpp # 高度格网方法
│   ├── SurfaceReconstructorPostprocess.h/cpp # 网格清理、边界环拆分、质量补洞及退化微孔保拓扑收缩
│   ├── SparseSfmPointQualityReader.h/cpp # 流式读取逐点 SfM 质量 sidecar，避免大 JSON DOM 峰值
│   ├── SparseOrbitalScaffoldBuilder.h/cpp # 质量过滤、离群点剔除、体素降采样及径向外法向
│   ├── ScreenedPoissonSurfaceBuilder.h/cpp # 官方 Screened Poisson 的固定版本适配器
│   ├── MeshVoxelTopologyRepair.h/cpp # 保守体素化、闭运算、MC33/三角质量优化及闭合 genus-0 回退
│   ├── OrbitalSparseScaffoldSurfaceBuilder.h/cpp # 环拍稀疏全局载体的 fail-closed 编排
│   ├── MeshIO.cpp              # 网格文件 I/O
│   ├── TextureMapper.h/cpp     # 纹理配置/结果门面及无相机时的顶点色回退
│   ├── CameraTextureMapper.cpp # camera_projected_atlas_v3 兼容路径与 v4 调度
│   ├── CameraTextureMapperV4.h/cpp # 多视图纹理 v4 阶段编排
│   ├── TextureSourcePreprocessor.cpp # 原图/证据相机、清晰度和网格邻接准备
│   ├── TextureOverlapExposure.h/cpp + Solver.cpp # 共同可见 3D 点的 linear-sRGB 鲁棒曝光增益与 fail-closed 诊断
│   ├── TextureVisibilityEvaluator.cpp # 七点证据检查、光度一致性 top-K 评分、ICM 与小孤岛合并
│   ├── TextureChartBuilder.cpp # 按相机标签连通域构建投影 chart
│   ├── TextureAtlasPacker.h/cpp # 自适应 MaxRects/shelf chart 图集调度与缩放搜索
│   ├── TextureAtlasMaxRects.cpp # 有操作预算的确定性无旋转 MaxRects 实现
│   ├── TextureAtlasSampling.h/cpp # 逐纹素深度/掩膜复核、实样本 medoid 与 Natural 鲁棒融合
│   ├── TextureNaturalBlender.h/cpp # linear-sRGB 掩膜金字塔的 Natural 低频融合与主视图细节保留
│   ├── TextureAtlasBaker.cpp   # 图集光栅化、边界填充、锐化及 OBJ/MTL/PNG 输出
│   ├── StudioForegroundMask.h/cpp # 黑色摄影背景检测、主体轮廓提取与可复用前景掩模
│   ├── MeshColorizer.h/cpp     # 网格遮挡检查、鲁棒多视图顶点着色及孤立色斑清理
│   ├── MeshFaceColorOptimizer.h/cpp # 实验性按面主视图投票与共享顶点一致着色
│   ├── MeshQuadricSimplifier.h/cpp # 目标面数驱动、边界/锐边/翻转约束的 QEM 自适应简化
│   ├── MeshFaceOrientation.h/cpp # 共享边面朝向统一及不可定向冲突面的最小清理
│   ├── NativeMeshSimplifier.h/cpp # 原生拓扑安全 QEM 编排、面朝向修复与限位特征保护平滑
│   ├── BoundaryAwareVoxelSimplifier.h/cpp # 多视图剪影保护、内部开放边界可聚类的体素简化后备
│   ├── MeshTopologyQuality.h/cpp # 开放边/非流形/连通分量/三角形长宽比质量门及保拓扑边翻转优化
│   ├── DepthTsdfFinalQualityGate.h/cpp # TSDF 写出前硬质量门；观测曲面仅放宽开放边比例
│   ├── DepthTsdfRecoveryTransaction.h/cpp # TSDF 恢复候选的基线/候选拓扑事务验收与 support 回滚
│   ├── MeshBoundaryAttribution.h/cpp # 最终开放边归因、阶段边界统计和原因着色调试 PLY
│   ├── MeshAcquisitionGapReport.h/cpp # 开放边根因、12×3 表面方位、逐帧来源覆盖与补拍/重算建议 JSON
│   ├── MeshIsotropicRemesher.h/cpp # 内部高长宽比区域的短边合并、长边拆分与拓扑/法向保护
│   ├── IsoSurfaceTopology.h/cpp # 等值面共享边/共享面键、渐近判别和歧义统计
│   ├── ConsistentIsoSurfaceExtractor.h/cpp # 默认共享网格边顶点的一致等值面提取器；避免组件过滤误删三角面
│   ├── Mc33IsoSurfaceExtractor.h/cpp # 可选 MC33 拓扑无歧义等值面适配器
│   ├── DepthMapMeshBuilder.h/cpp # 深度帧 manifest/相机产物加载；缺最终层时按清单安全回退最高可用金字塔层
│   ├── DepthFusionFramePolicy.h/cpp # 环拍视角覆盖度量及防连续视角缺口的帧准入策略
│   ├── DepthMeshCompleteness.h/cpp # 深度观测到最终网格的逐帧召回率与完整性质量门
│   ├── TriangleDistanceIndex.h/cpp # BVH 加速的精确点到三角形距离查询，供网格完整性评估使用
│   ├── DepthRayMetric.h/cpp # camera-Z 深度、欧氏射线距离和世界像素足迹的统一换算
│   ├── DepthTsdfSurfaceBuilder.h/cpp # raw depth/confidence/mask/camera 直接融合 TSDF、提取网格并安全减面
│   ├── DepthTsdfCellSheetRecovery.h/cpp # 按面邻接、跨视图来源与已有表面锚点恢复连续零交叉单元片
│   ├── DepthMeasuredSupportConnectivity.h/cpp # 不改 TSDF 值的实测 support 候选门控、切平面边界保护与归因统计
│   ├── DepthImplicitFieldRegularizer.h/cpp # 等值面提取前的可见性保护、多尺度隐式场正则化与单体素裂缝恢复
│   ├── DepthVisibilityHistogram.h/cpp # 每个 TSDF 样本 9 字节、含精确零中心的有符号距离直方图及鲁棒统计
│   ├── AdaptiveTsdfOctree.h/cpp # 保留零面细节的 2:1 平衡自适应八叉树与面邻接图
│   ├── SparseTgvSolver.h/cpp # 八叉树稀疏图上的 primal-dual 二阶 TGV 隐式场求解器
│   ├── ProcessCpuTimer.h # 跨平台进程 CPU 时间采样，用于模型子阶段并行占用率统计
│   ├── VisibilityOccupancyTsdfCompletion.h/cpp # 深度/轮廓可见性数据项、图割占据载体与拓扑锁定 TSDF 残差融合
│   ├── VisibilityOccupancyCleanup/HandleRepair/WellComposedRepair.* # 占据体分量、柄和良构拓扑修复
│   ├── VisibilityOccupancyDistanceField/BoundaryExtractor/SurfaceBuilder.* # 闭合载体距离场、边界和表面构造
│   ├── VisibilityOccupancyCarrierSubdivision/Fairer/FieldProjector.* # 保拓扑载体细分、平滑和距离场投影
│   ├── RegularGrid3D.h         # CPU/CUDA/OpenCL 共用的规则三维网格布局
│   ├── VisualHullFieldEvaluator.h/cpp # Visual Hull 体素场 CPU 参考语义
│   ├── VisualHullFieldBackend.h/cpp # CPU/CUDA/OpenCL 体素场调度与输入打包
│   ├── VisualHullFieldCUDA.cu  # CUDA 轮廓/深度自由空间体素场评估
│   ├── VisualHullFieldOpenCL.cpp # OpenCL 轮廓/深度自由空间体素场评估
│   ├── VisualHullFieldNoCUDA.cpp/VisualHullFieldNoOpenCL.cpp # 无 GPU 后端构建存根
│   ├── VisualHullReconstructor.h/cpp # Visual Hull 体素场、拓扑闭运算和表面提取
│   ├── ModelOutputPolicy.h/cpp    # 模型/纹理 run 隔离目录、所有权标记与未发布目录安全回收
│   ├── ModelWorkflowService.h/cpp  # 模型工作流服务；保留 PLY 几何并可写 OBJ/MTL/相机纹理图集
├── terrain/                    # 地形产品 (DEM/DOM) 和质量栅格
│   ├── DemDomTypes.h           # DEM/DOM 类型
│   ├── DemGridAggregator.h/cpp # mean/median/NMAD/P80/count/confidence/error weighted 聚合
│   ├── DemMosaic.h/cpp         # CPU/CUDA/OpenCL 同网格多 tile DEM mosaic
│   ├── TerrainProductManifest.h/cpp # DEM/DOM/error/count/confidence/coverage 产品记录
│   ├── DemGenerator.h/cpp      # DEM 生成
│   ├── DemGeneratorFromDepth.cpp  # 从深度图生成 DEM
│   ├── DomGenerator.h/cpp      # DOM 正射影像生成
│   ├── OrthoGenerationOptions.h/cpp # 正射投影、尺寸、区域、融合和覆盖处理的类型化参数
│   ├── TerrainComputeBackend.h/cpp # auto/cpu/cuda/opencl 解析、设备查询与执行报告
│   ├── TerrainGpuBackend.h     # 正射投影和 DEM mosaic 的 GPU 数据契约
│   ├── TerrainCudaBackend.cu   # CUDA 正射投影与 DEM mosaic kernel
│   ├── TerrainOpenClBackend.cpp # OpenCL 正射投影与 DEM mosaic kernel
│   ├── TerrainNoCudaBackend.cpp/TerrainNoOpenClBackend.cpp # 无 GPU 后端构建存根
│   ├── OrthoProjector.h/cpp    # CPU/GPU 调度、DEM 反投影与候选融合
│   ├── OrthoProjectorGpu.h/cpp # 正射影像 GPU 输入打包、执行与结果还原
│   ├── OrthoProjectorGrid.cpp  # DEM 边界裁剪、X/Y 像元、最大尺寸与像素预算规划
│   ├── OrthoProjectorSupport.cpp # 影像/蒙版加载、颜色校正、锐度、重影和小孔洞处理
│   ├── OrthoProjectorInternal.h # 正射投影内部帧与颜色候选结构
│   ├── PointCloudDomGenerator.h/cpp # 彩色点云局部平面/小天体全球 DOM 栅格化
│   ├── SmallBodyGlobalProducts.h/cpp # 体固连全球产品参数、共享栅格与结果 DTO
│   ├── SmallBodyMeshRaycaster.h/cpp # BVH + Möller–Trumbore 体心径向网格求交、颜色/UV 插值
│   ├── SmallBodyGlobalProductGenerator.h/cpp # 0–360°径向 DEM、高程 DEM、DOM 与质量产品
│   ├── GlobalTerrainReportRenderer.h/cpp # 核心侧稳定生成全球产品四联 PNG
│   ├── DemDomIO.h/cpp          # DEM 元数据/栅格、RGB+覆盖 Alpha GeoTIFF 和质量栅格 I/O
│   ├── TerrainPipeline.h/cpp   # 地形流水线 (主入口)
│   ├── projection/
│   │   └── AsteroidProjection.h/cpp  # 小行星投影
│   └── tests/                  # 平面/正射、质量栅格与小天体全球产品测试
│
├── stereo_dem/                 # 带 RPC 的地理立体像对到 DEM/DOM
│   ├── RpcStereoDemGenerator.h/cpp # SIFT 同名点、RPC 前方交会、UTM 点云和 DEM 产品编排
│   ├── RpcDomGenerator.h/cpp   # DEM 网格反算 WGS84、RPC 正射采样和 RGB+Alpha DOM
│   ├── RpcGeospatialSupport.h/cpp # WGS84/UTM 坐标转换与 DEM 行坐标反算
│   └── tests/                  # 仓库 RPC 像对的 DEM/DOM 端到端回归
│
├── qc/                         # 重建质量检查和外部参考验证
│   ├── ReconstructionQualityReport.h/cpp # 注册影像、track、重投影、MVS/DEM 覆盖率、GCP/检查点/比例尺报告
│   ├── SurveyControlImport.h/cpp # GCP/检查点/比例尺 CSV 导入为 survey_control metadata
│   ├── SurveyControlReport.h/cpp # GCP/检查点/比例尺 metadata 统计和残差状态汇总
│   ├── PointCloudAlignment.h/cpp # 点云完整 Sim3 / 最近邻 ICP 配准与 beg/end error CSV
│   ├── ModelMeshRenderer.h/cpp # CPU tile 并行 z-buffer，以 reciprocal-Z 和透视正确属性投影 PLY 网格
│   ├── ModelImageMetrics.h/cpp # 轮廓、覆盖、边缘、SSIM/PSNR 影像空间指标
│   ├── ModelGeometryComparator.h/cpp # 连通分量与参考点云双向最近邻 A+C 几何验收
│   ├── ModelImageQualityEvaluator.h/cpp # GUI/CLI 可复用的统一门控、诊断图和 JSON/CSV 报告
│   ├── ProcessingBaselineManager.h/cpp # 总输入/分阶段快照指纹、参考网格指标和跨版本质量门
│   └── DemDifference.h/cpp     # DEM 差分、绝对差分和统计报告
│
├── aerial_triangulation/       # 对齐照片式空中三角测量，职责对应 Metashape Align Photos
│   ├── model/                  # GUI/CLI 共用 Options、ResolvedConfig、Result DTO
│   ├── workflow/               # 唯一入口与正式 Pipeline
│   ├── preparation/            # MatchPhotosTask 适配、缓存编目和前置检查
│   ├── reconstruction/         # 单次针孔 SfM、RPC 空三、标记点先验、相机内参清洗、候选对与图诊断
│   ├── search/                 # 无相机焦距候选排序和资源策略
│   ├── reporting/              # 稀疏点云、质量元数据和结果记录
│   └── CMakeLists.txt          # 独立 aerial_triangulation target
│
└── image_matching/lightglue/
    └── LightGlueFeatureBudget.h  # LightGlue/SIFT 显存感知关键点预算工具
```

`AerialTriangulationWorkflow` 是“空中三角测量/对齐照片”的唯一用户级入口。重置当前对齐只清除
SfM 位姿和稀疏重建状态；默认仍复用影像身份、算法版本、配置指纹和模型指纹均匹配的 `.pimatch`
变体及连接点。只有用户取消“重用现有匹配”或缓存缺失/不兼容时，workflow 才调用
`MatchPhotosTask` 按工作流设置执行 SIFT + LightGlue 或 LoMa-R，并整理多视连接点。GUI 与
`aerial_triangulation_cli` 不再各自实现连接点补齐逻辑，也不允许 SfM 回退读取旧成对缓存。

正式 Pipeline 在焦距搜索前检查本次全部相机模型。全 RPC00B 批次由
`RpcAerialTriangulationRunner` 保持厂商 RPC 固定，恢复多视连接点轨迹并使用非线性 RPC 前方交会
优化地面点；结果以局部 WGS84 ENU 米制 PLY 写出，同时在 sidecar 保存 ENU 原点、逐点经纬高和
逐相机残差。没有 GCP 时不会伪造 RPC 偏差改正，诊断明确标记为 fixed-sensor point-only adjustment。
RPC 与针孔混合或部分 RPC 缺失的批次直接拒绝，禁止再按影像尺寸生成虚构针孔内参。固定 RPC 提供
绝对传感器几何，因此合法的两景 RPC 空三不适用自由网络针孔 SfM 的两视轨迹占比门槛。
RPC 稀疏结果可登记为正式空三成果，但会标记为不兼容普通针孔 MVS；GUI 的“创建点云/仅生成深度图”
入口会给出明确阻断提示，密集产品应进入 RPC 立体 DEM/DOM 工作流。

“已有 SfM 查漏”把当前相机位姿和 `sfm_sparse_points.json` 作为上一轮解算证据：先按稀疏点观测建立
共视关系，再将稀疏场景的鲁棒包围体采样投影到相机视锥，补回共视点不足但几何上仍有重叠的影像对。
该模式不使用地球参考球面；旧 SfM 数据缺失或不完整时安全退回自动/序列候选。对内向环拍且相机间距
连续的数据，workflow 会自动保留闭环序列位姿恢复，但候选对仍由已有 SfM 查漏而不是退回纯序列匹配。

连接点阶段的 GPU 调度按真实资源而不是 CPU 线程数决定：CUDA SIFT 保持单执行上下文并由
任务级预取队列重叠 CPU 解码；LightGlue 每个 worker 独占 engine context 和 stream，自动并发受可用显存约束，
发生 OOM 时串行补跑未完成影像对。几何验证直接消费任务内像点，并以最多 8 路 CPU 并发运行固定随机种子的
USAC-MAGSAC。PnP、增量三角化和轨迹图仍保留 CPU，
因为当前基准未达到 20% 的端到端 GPU 收益门槛。

无相机 `IncrementalSfm` 会在构建对应关系索引前再次按用户的连接点限制执行轨迹级筛选，确保
SfM、BA 和创建连接点阶段使用相同的每影像限额语义。焦距粗筛覆盖 `0.55` 到 `10.0` 的视场范围，
最多并行 4 个候选并遵守总线程预算；排序首先保证注册覆盖，再联合评价多视比例、交会角、空间覆盖、
重投影误差和有界闭环序列质量。最终质量报告对两视轨迹比例执行 0.70 advisory / 0.85 blocking 门控。

`sfm/ReferenceTerrainPrior.h/cpp` 把参考 DEM 或 LiDAR 局部高度面接入 BA soft prior。参考地形默认作为软约束参与诊断，
不把已知外参硬固定；BA 报告应记录 pose prior / terrain prior 优化前后的残差。

`core/lidar` 中并存两条不能混用的激光路径。`LaserConstraintMap` / `LaserConstraintAssociation` 处理带法向
PLY 扫描表面，通过最近平面建立普通 BA track 的点到面约束；`PlanetaryLaserShot` / `PlanetaryLaserJson` /
`PlanetaryLaserIsisJson` / `PlanetaryLaserBaAdapter` 处理 LOLA、MOLA 一类稀疏测距 shot，不要求法向，也不做
最近邻关联。后者把每个 shot 建成独立辅助落点参数块，并用
`(||P - (C + R * leverArm)|| - range) / sigmaRange` 约束同期相机。Fixed 落点保持常量，Constrained 落点使用
完整 `3 x 3` XYZ 协方差白化先验，Free 落点必须由至少两台非零基线相机的真实 `measured` 像点约束。
普通 track 与激光 shot 的点和统计完全分离。

PlaScan SI JSON v1 显式保存目标、天体固连 frame、激光 frame、TDB ET 时间、单程/往返语义、杆臂和
image measure 类型。ISIS `LidarData` 缺失的目标/frame/传感器模型/range 类型/杆臂由调用方上下文补齐；
其 `aprioriMatrix` 按 `(latitude rad, longitude rad, radius m)` 球面协方差读取，再经球面到 XYZ 雅可比
转换为完整米制协方差。ISIS 由落点反投影得到的 measures 始终标记为 `projected`，不会进入真实像点残差。
当前适配器只接受唯一同期映射的静态 frame camera，严格拒绝 line-scan、未经换算的 round-trip range、
坐标系不一致和歧义关联；真实 `measured` 像点未映射时默认失败，ISIS `serialNumber` 等产品标识由调用方按
求解相机索引提供稳定别名，工程 `image_uuid` 由 GUI/CLI 自动按相机顺序合并。完整 ID 唯一命中优先于
filename/stem 回退。当前一个 shot 只允许一台同期相机；ISIS 多 `simultaneousImages` 共享同一落点的完整行为
尚未实现。像点协方差当前只接受可精确转成核心标量权重的 `sigma^2 I`；各向异性或相关矩阵明确拒绝。
`ephemeris_time_s` 只保留到报告，尚无 SPICE/逐行时变轨迹求值。Auto 模式的 PlaMatrix range 候选失败或被质量
门控拒绝时直接失败，禁止回退到不支持测距约束的 Legacy。行星激光 dry-run 仍执行数据、传感器模型、
坐标系和别名预校验；初始落点与杆臂修正后的发射点重合时也会在求解前拒绝。

LRO NAC / LOLA 推扫数据不经过上述静态适配器，而使用独立
`camera/PlanetaryLineScanCamera` 与 `lidar/PlanetaryLineScanBundleAdjust` P0。相机从 USGSCSM ISD 解析
逐行曝光时间、Hermite 位置、四元数姿态和 LRO NAC 畸变；控制网 PVL 的 ISIS `(1,1)` 像素中心在
投影边界统一转换到 CSM `(0.5,0.5)`。普通 Free 控制点以观测行瞬时射线三角化，range 默认在 shot
TDB ET 求相机中心；`isis_line` 仅用于上游回归时复现虚拟 measure 的行时刻，虚拟像点不会进入影像残差。
P0 每景只优化月固系一个 6DoF 刚性偏差，并明确限制为 `MOON_ME`、单程、单同期影像、零杆臂和 Free
控制点；遇到 Fixed/Constrained 控制点或非零杆臂会拒绝，防止丢失地面先验或仪器外参。名义姿轨
当前直接采用原始 ISD Hermite/SLERP，尚未复刻 USGSCSM 重采样后的 4/8 阶 Lagrange 状态求值。
独立 CLI 因此要求显式接受该插值近似，并拒绝复用已有本工具产物的输出目录，避免误当成官方模型或
把不同运行的无激光/有激光结果混合。

该链路由 `src/core/lidar/tests/test_planetary_laser_json.cpp` 和
`test_planetary_laser_ba_adapter.cpp` 覆盖格式、ISIS 协方差、projected/measured 类型、像点权重与关联安全边界；
`src/core/bundle_adjust/tests/test_bundle_adjust_plamatrix_constraints.cpp` 覆盖 Fixed/Constrained/Free、杆臂和后端能力；
`tests/test_bundle_adjust_service_planetary_laser.cpp` 覆盖 SI/ISIS 服务端别名/工程 UUID 关联、严格相机顺序、结果写出与
line-scan 拒绝；`tests/test_planetary_laser_preview.cpp` 覆盖 GUI 预览和质量摘要。
`tests/test_reference_dataset_planetary_laser.cpp` 验证 ISIS 点签名识别，并防止普通 `points` JSON 被误分类。

`sfm/pipeline/SfmBundleAdjustCoordinator` 是空三调用 `bundle_adjust` 的正式入口。它构造局部或全局
相机/轨迹问题、固定边界相机、转发进度，并只回写 `solutionUsable=true` 的结果。无绝对控制时，
`SimilarityGaugeNormalizer` 在全局 BA 后恢复确定性锚点和初始基线尺度；有控制点或比例尺时由绝对约束接管规范。
注册相机达到大型问题阈值后，`CovisibilityPartitioner` 按已验证匹配数划分唯一核心和重叠边界，
`HierarchicalBundleAdjuster` 在总线程预算内并行执行块内 BA。重叠相机固定在同一进入坐标系，核心相机和
三维点按唯一所有权写回；同时被块外已注册相机观测的跨块轨迹保持三维坐标固定，但仍以
重投影残差约束块内相机，避免独立改点造成合并接缝。`HierarchicalBaBlockSolver` 负责单块问题装配、gauge 固定与质量门控。
无论上次分块是否通过全局门控，都要等模型增长达到半个目标块后才再试，避免在相邻周期重复支付失败成本。
周期完整全局 BA 由该块级稳定化替代，最终只执行一次共享内参与块间接缝精化。
该调度只依赖共视网络和计算规模，不按航测、转台、相机编号或轨迹形状推断场景。
最终共享内参 BA 同样不硬分类“对地/环拍”：`BundleAdjustAdaptiveCameraModel` 在粗解上对
`f/aspect/cx/cy/k1/k2/k3/p1/p2` 分别计算结构消元后的增量信息评分、典型扰动敏感度和几何/像面覆盖证据，
再生成 PlaMatrix 参数掩码。弱平行块保留低阶模型，汇聚、多高度且外围覆盖充分时才逐项扩展；
请求、调度、有效和实际写回状态连同逐参数可靠度、门控证据均进入 SfM 诊断。完整已知位姿路径固定
输入标定；多轮重三角化/BA 对已应用状态做累计，后续 no-op 不会把前轮有效自标定误判成失败。多起点与
迭代轮次可沿用上一轮内参作为数值初值，但所有相对边界和弱先验都锚定到 `IncrementalSfm::run()`
生命周期内按影像 ID 保存的首次有效标定参考，周期性、最终和重试全局 BA 不会逐调用复合漂移；单次迭代的多起点
只在首轮执行，并通过 `SfmCalibrationPreviewSampler` 限制为确定性的相机/像面半径均衡子集；第一轮保留
60 次上限，后续已有可复用镜头种子时恢复配置迭代数。局部/分块固定内参子问题不携带全局位置式参考，
且不允许 Legacy 做不等价模型回退。

`bundle_adjust` 由 PlaMatrix 在同一非线性问题中联合优化相机、三维点、分组完整 Brown 内参和物方约束。
Legacy CPU 保留 point-only 固定相机问题。所有后端统一返回状态、可用性、取消、回退原因和耗时，
正常 Auto 路径只有未通过状态或质量门控时才回退，不再无条件重复完整 Legacy BA。
`plamatrix_cpu`、`plamatrix_cuda` 和 `plamatrix_opencl` 是正式联合 BA 路径：PlaScan
负责完整 Brown-Conrady 投影、解析相机/点/内参雅可比、物方约束、gauge、取消与统一质量复核，PlaMatrix 负责通用 Huber
权重、二分块法方程、Schur 消元、CPU 稠密 Cholesky/块 Jacobi-PCG、CUDA/OpenCL CSR 块 Jacobi-PCG 和 LM 阻尼。
外层 LM 对拒绝步复用法方程，CPU 按问题规模自动选择直接或迭代解法；track 和推扫观测以固定连续分片并行装配，
再按线程序确定性合并。设备路径复用经过完整邻接签名校验的 CSR 拓扑、装配缓冲和求解工作区；CUDA Schur
装配输出通过设备内复制直接交给 PCG。OpenCL 因 NVIDIA 595.84 跨队列 buffer handoff 会触发驱动崩溃，
当前保留数值主机 handoff，但固定拓扑和装配缓冲仍常驻复用。
CPU 稠密路径直接按固定 slot 顺序装配 Schur 下三角，不再经过 CSR；常见 3×3/9×3 块使用固定尺寸内核，
128 阶以上用自动 block size 的 POTRF/TRSM/SYRK 分块 Cholesky。报告分别记录小块求逆、Schur 累加、
CSR 转换、Cholesky、三角求解、残差检查和点块回代，以便把装配与线性求解回归分开定位。
三个后端使用同一问题与测试数据并报告实际设备；Auto 对联合问题按规模选择 PlaMatrix CPU/CUDA/OpenCL，
GPU 失败只回退 PlaMatrix CPU。完整 Brown 共享内参和 GCP/LiDAR/比例尺/姿态/激光测距约束均有跨 PlaMatrix 后端一致性回归。
行星激光 range shot 的生产实现使用 PlaMatrix；后端能力表和输入校验阻止 Legacy CPU 静默忽略
该约束。结果单独返回参与求解的 shot 数、优化前后 range RMS 和逐 shot 落点/残差，不污染普通影像
重投影 RMS、track 过滤计数或有效 track 比例。

无相机文件且没有用户内参时，`AerialTriangulationPipeline` 始终评估广域焦距尺度，并按注册覆盖、
多视网络强度、点数和重投影质量排序。正式阶段重放最高质量的非默认候选，并重新自动选择初始像对。
焦距初始化与“自适应相机模型拟合”解耦：后者只决定正式 BA 是否释放共享焦距，关闭后仍使用搜索得到的
最佳初始焦距作为固定内参。粗筛不写 PLY、项目结果或匹配缓存。
E/F/H 和 PnP 的 OpenCV RANSAC 调用由稳定种子保护，结果不应随粗筛 worker 数变化。照片序列外推/插值仅作为
PnP 初值，未经 3D-2D 几何验证的相机不会计入正式注册覆盖率。重置当前对齐但复用项目内参时，
`CameraIntrinsicPriorSanitizer` 会仅在同尺寸相机存在占主导焦距群时修正超过 2 倍的旧 SfM 焦距离群值；
它不复用旧外参，并将修正数量、焦距中位值和影像列表写入 SfM 诊断。

`aerial_triangulation/reporting/QualityReportWriter.cpp` 在内存中构造稀疏点、逐相机残差、BA 摘要和
SfM 诊断；`AerialTriangulationResultWriter.cpp` 再把 `sfm_sparse.ply` 与 `sfm_sparse_points.json`
原子写入本次显式输出目录。连接点候选图、实际匹配图和 pair 状态仍由 `matchphototask` 的报告负责。

---

## 三、gui/ — Qt6 图形界面

### 目录结构

```
gui/
├── main.cpp                    # 应用入口 (QApplication, 全局字体, 异常处理)
├── CMakeLists.txt              # gui_runtime 复用库与轻量应用入口；测试不再重复编译生产 GUI 源
│
├── main_window/                # 主窗口层
│   ├── MainWindow.h/cpp        # QMainWindow 派生, 顶层 UI 编排
│   ├── ProjectUiHydrator.h/cpp # 分阶段刷新项目界面，并通过代次号丢弃过期请求
│   ├── ProjectLifecyclePresenter.h/cpp # 项目打开/保存进度、脏状态标题及关闭后保存
│   ├── ProjectTaskStatusController.h/cpp # 状态栏任务控件、取消路由及概览快照
│   ├── FeatureVisualizationController.h/cpp # 特征点/残差显示选项、项目级持久化与画布同步
│   ├── MenuWorkflowController.h/cpp       # "工作流程" 菜单业务控制器
│   ├── ReconstructionWorkflowController.h/cpp  # 生成模型/纹理工作流程对话框协调
│   └── WorkspacePanelController.h/cpp     # Dock/工具栏可见性、菜单动作与项目状态统一管理
│
├── menu/
│   ├── MainMenu.h/cpp          # 菜单栏/工具栏动作编排；入口、分区构建器和动作描述表分责
│   └── ToolbarButton.h/cpp     # 统一快捷栏模板：尺寸、绘制状态、普通/下拉按钮工厂
│
├── markers/                    # 标记点 GUI、工程 repository 与后台检测
│   ├── MarkerWorkspaceController.h/cpp # 右键编辑、撤销、检测合并和 sidecar 协调
│   ├── MarkerReferencePanel.h/cpp      # 标记角色、参考坐标和精度编辑
│   ├── MarkerProjectionPanel.h/cpp     # 多影像投影状态查看
│   ├── MarkerFocusMeasurementDialog.h/cpp # 双影像聚焦量测和预测确认
│   ├── DetectMarkersDialog.h/cpp       # 项目级后台检测、整体进度和取消
│   ├── MarkerDetectionReviewDialog.h/cpp # 候选预览、归入已有标记或丢弃
│   └── PrintMarkersDialog.h/cpp        # 标靶物理布局和 PDF 输出
│
├── reference/                  # Metashape-like 参考面板的数据模型与项目协调
│   ├── CameraReferenceTreeModel.h/cpp  # 相机源值、解算估计值、误差及未匹配记录树
│   ├── ProjectCameraReferenceRepository.h/cpp # 独立 sidecar 生命周期与 metadata 摘要
│   └── CameraReferenceController.h/cpp # Metashape TXT 导入、源值导出和安全状态说明
│
├── dialogs/                    # 按业务域组织的参数与结果对话框
│   ├── application/            # 关于、工作流程报告等应用级对话框
│   │   └── reporting/GlobalTerrainReportPage.h/cpp # 当前 Chunk 全球 DEM/DOM 四联图与坐标警告
│   ├── camera/                 # 相机校准前后对比、相机查看/转换、前方交汇、测量控制
│   │   ├── CameraCalibrationData.h/cpp   # 固化空三输入先验/最终内参，按图像中心转换 cx/cy
│   │   └── CameraCalibrationDialog.h/cpp # 初始/调整内参、释放状态、相机分组与照片列表
│   ├── image/                  # 蒙版等单影像处理
│   │   ├── GenerateMaskDialog.h/cpp # 经典/U2Net/BiRefNet 方法、真实设备/尺寸、模型状态与下载入口
│   │   └── MaskEditorSettingsDialog.h/cpp # 交互蒙版的添加/擦除、容差、笔刷、吸附和透明度设置
│   ├── reconstruction/         # 空三、模型、纹理、DEM/正射工作流程
│   │   ├── MapProjectDialog.h/cpp         # 正射对话框生命周期、运行进度与取消
│   │   ├── MapProjectDialogLayout.cpp     # 产品模式、双栏参数、输出和进度分组布局
│   │   ├── MapProjectDialogSettings.cpp   # 稳定 token、设置往返、输入校验与控件联动
│   │   └── MapProjectDialogEstimate.cpp   # DEM 元数据读取、真实像元/范围和内存估算
│   ├── tie_points/             # 连接点创建/清理/查看与重叠分析；清理支持重投影误差、重建不确定度、图像计数、投影精度和交会角的后台候选预览
│   ├── shared/                 # 对话框共享样式与布局辅助
│   └── README.md               # 分类规则与新增对话框约束
│
├── widgets/                    # 自定义 Qt 控件
│   ├── CanvasWidget.h/cpp              # 2D 影像/图层渲染画布；整体视图旋转不修改影像或摄影测量坐标
│   ├── MaskEditor.h/cpp                # 蒙版编辑状态、图形预览、撤销/重做与快捷键交互
│   ├── ImageViewWidget.h/cpp           # 2D 影像缩放/平移控件
│   ├── DualImageViewer.h/cpp           # 双图并列匹配查看器，协调左右影像与稀疏连线
│   ├── MatchLineOverlay.h/cpp          # 匹配线叠加层 (稀疏 → 连线)
│   ├── PhotoStripWidget.h/cpp          # 可视区缩略图、有界预取队列与 LRU 缓存
│   ├── WorkPanelWidget.h/cpp           # 下方工作面板，展示运行中任务、实时用时和进度
│   ├── MatchPointBatchItem.h/cpp       # 匹配查看器单图元批量端点绘制，避免逐点 QGraphicsItem
│   ├── MatchSpatialIndex.h/cpp         # 匹配点二维网格索引与可视区域候选查询
│   ├── MatchGpuRenderer.h/cpp          # 基于 QRhi 的匹配线与端点 GPU 批量渲染器
│   ├── DataTreeWidget.h/cpp            # Metashape 式工作区汇总 → Chunk → 资源分组树
│   ├── WorkspaceSectionIcons.h/cpp     # 工作区、Chunk、影像及成果类型语义图标
│   ├── ReferencePanelWidget.h/cpp      # 相机参考、标记、标尺三段树及源值/估计值/误差模式
│   ├── ReferenceMarkerModels.h/cpp     # 控制/检查点及控制/检查标尺汇总模型
│   ├── ObservationNetworkView.h/cpp    # 观测网络可视化
│   └── WorkspaceCenterWidget.h/cpp     # 工作区布局管理及模型/影像/对比/观测网络模式通知
│
├── project/                    # 项目管理层
│   ├── manager/
│   │   ├── ProjectManager.h/cpp # 项目管理器；含参考激光 JSON 导入、frame/坐标系确认和 BA 启动
│   │   ├── ProjectLifecycleController.h/cpp          # 创建、异步打开/结果加载、保存与关闭
│   │   ├── ProjectMaskWorkflowController.h/cpp       # 蒙版对话框、异步生成/交互保存、取消及结果登记
│   │   ├── ProjectMaskInferenceAdapter.h/cpp         # U2Net/BiRefNet 后端适配和实际模型/设备/engine 元数据
│   │   ├── ProjectSparseReconstructionManager.h/cpp  # 稀疏重建与连接点质量剔除，产出一致的 PLY/sidecar 并通知三维视图刷新
│   │   ├── ProjectPointCloudWorkflowController.h/cpp # 点云工作流协调：深度估计/复用、流式融合与结果登记
│   │   ├── ProjectModelManager.h/cpp                 # 从已有点云/深度图生成模型，不隐式启动稠密流程
│   │   ├── ProjectTerrainProductsManager.h/cpp       # 局部 DEM、原生小天体全球 DEM/DOM、取消与 Chunk 隔离登记
│   │   ├── ProjectTerrainRpcProducts.cpp             # RPC 立体 DEM/正射 DOM 异步执行、取消、质量成果与项目登记
│   │   ├── ProjectCameraSetupManager.h/cpp           # 相机设置管理
│   │   └── ProjectUiCommands.h/cpp                   # UI 命令
│   ├── services/
│   │   ├── BundleAdjustService.h/cpp                 # BA 服务；解析/装配行星 range shot 并写独立摘要
│   │   ├── ProjectCameraImportService.h/cpp          # 相机导入
│   │   ├── MetashapeCameraReferenceImporter.h/cpp    # WGS84 相机参考与 GNSS 杆臂 TXT 严格解析
│   │   ├── ProjectSessionFacade.h/cpp                 # ProjectManager 与 ProjectData 间的会话查询/修改兼容门面
│   │   └── ProjectTiePointResultService.h/cpp        # 单一当前连接点、覆盖清理与真实删除
│   └── support/                 # 支持/辅助类
│       ├── ProjectBundleAdjustExecution.h/cpp       # BA 执行
│       ├── ProjectBundleAdjustWorkflow.h/cpp        # BA 工作流
│       ├── ProjectCameraInitialization.h/cpp        # 相机初始化
│       ├── ProjectDepthBatchLineage.h/cpp           # 路径无关的深度输入签名
│       ├── ProjectModelResultPolicy.h/cpp            # 模型 schema v2、默认版本迁移及完整产物登记策略
│       ├── ProjectModelTaskLifecycle.h/cpp           # 模型任务身份、会话门控、取消与未发布 run 回收
│       ├── ProjectRunArtifactValidator.h/cpp         # 发布前 run 诊断身份、路径归属和实体产物校验
│       ├── ProjectModelWorkflowPolicy.h/cpp         # 模型线程预算及深度批次完整性/代次兼容策略
│       ├── ProjectSessionContext.h                  # 异步写回会话身份（项目、Chunk、generation）
│       ├── ProjectOpenGuard.h/cpp                   # 统一项目已打开前置检查和用户提示
│       ├── ProjectTerrainRequests.h                 # DEM 类型化请求及边界校验
│       ├── ProjectMetadataOperations.h/cpp          # 元数据操作
│       ├── ProjectResultRecords.h/cpp               # 结果记录
│       ├── ProjectSfmWorkflow.h/cpp                 # SfM 工作流
│       ├── ProjectSparseWorkflow.h/cpp              # 稀疏工作流
│       ├── ProjectSurveyControl.h/cpp               # GCP/检查点/比例尺 CSV/Agisoft TXT 导入与 sidecar 摘要
│       └── ProjectWorkflowReports.h/cpp             # 工作流报告
│
├── tasks/                      # 异步任务执行器
│   └── GuiTaskRunner.h         # GUI 后台任务生命周期守护：runGuardedWithOutcome/postGuarded
│
├── views/
│   ├── LayerRenderer.h/cpp             # 图层渲染器
│   ├── LayerOverlayItems.h/cpp          # 批量特征/残差覆盖层及后台蒙版轮廓数据准备
│   ├── LayerFeatureLoader.h/cpp         # 特征文件解析与关键点加载
│   ├── FeatureResidualLoader.h/cpp      # 按当前影像异步筛选真实重投影残差
│   ├── CameraSceneWidget.h/cpp           # 三维场景生命周期、模型/点云与相机的 RHI 渲染及交互；独立红色层异步预览候选剔除连接点
│   ├── CameraSceneImageCache.h/cpp        # 相机缩略图/原图缓存、失败记忆、LRU 预算与失效
│   ├── CameraSceneRhiResources.h          # 三维场景缓冲、管线、相机图集资源状态 DTO
│   ├── CameraSceneWidgetOverlay.cpp      # 三维场景的 QPainter 交互覆盖层与相机标签绘制
│   ├── CameraSceneWidgetLegends.cpp      # 三维场景的图例和后台加载进度绘制
│   ├── CameraSceneViewMath.h/cpp        # 相机平面、视角选择与本地轴数学
│   ├── ObjRenderPreparation.h/cpp       # 静态网格 VBO、三角/线框 IBO 与 UV 接缝流准备
│   ├── SceneGeometryPreparation.h/cpp   # 点云 GPU 数据、空间分块、视锥/LOD 计划与后台框选
│   ├── TiePointVisualization.h/cpp       # 连接点逐点质量解析、指标范围与有界候选索引采样
│   ├── PointCloudEditPreparation.h/cpp  # 点云增量删除、撤销数据与 GPU 数据重建
│   └── PointCloudSnapshotIO.h/cpp       # 点云编辑暂存、格式分派与同目录原子替换
│
├── config/                     # 配置管理
│   ├── AppConfigManager.h/cpp          # 应用配置
│   ├── ImageViewRotationSettings.h/cpp # 项目级、按稳定 image_uuid 索引的查看旋转角度
│   └── settings/
│       ├── DialogSettingStore.h/cpp    # 对话框 JSON 设置的加载、保存与路径解析
│       ├── DialogSettingKeys.h         # 各对话框设置键名
│       ├── GuiSettingsStore.h           # GUI QSettings 工厂
│       ├── WindowStateManager.h/cpp    # 窗口状态持久化
│       ├── FileDialogStateManager.h/cpp # 文件对话框状态
│       └── RecentProjectsManager.h/cpp # 最近项目管理
│
├── panels/
│   └── LogPanel.h/cpp          # 下方控制台；日志级别过滤、清空、保存和有界等宽输出
│
├── platform/
│   ├── ProjectFileIntegration.h/cpp # 启动工程参数解析与 Windows 当前用户文件关联
│   └── TaskbarProgressController.h/cpp # 聚合项目任务进度并映射到 Windows 任务栏图标
│
├── log/                        # GUI 层日志 (使用 #include 快捷方式)
├── cmake/                      # GUI 构建配置
│   ├── GuiSources.cmake        # 源文件清单 (所有 .cpp/.h)
│   ├── GuiCoreLinking.cmake    # 核心库条件链接
│   ├── GuiInstall.cmake        # 安装规则
│   └── cmake/                  # Qt 宏
└── packaging/                  # 打包配置
```

PlaScan 工程采用 Metashape 式 `name.plascan + name.files` 结构。
`ProjectPackageLayout` 只接受类型和版本严格匹配的 4.0.0 描述文件，并拒绝旧单体 ZIP；
`name.files/project.zip` 只保存一个 `doc.json`，其中包含稳定 `project_id`、Chunk 索引
和项目 UI 状态。Chunk 的 `project_config.camera_model_policy` 记录相机模型策略；缺失时默认
`frame_pinhole`，显式线阵值为 `isis_usgscsm_linescan`，未知非空值拒绝解释。每个 Chunk 由稳定 UUID
标识，并映射到 `1/`、`2/`、`3/` 等数字目录；
目录号由持久化的 `next_chunk_directory` 单调分配，删除后不复用。Chunk 的核心元数据、
结果、配置和资源索引位于其 `chunk.zip/doc.json` 的独立字段。原始影像位于工程级
`.files/shared/images/<sha256>/`，其他大型成果直接位于对应数字目录。共享影像库、
`assets/`、`bundle_adjust/`、`reconstruction/` 和 `reports/` 均在对应流程首次写入时
按需创建，空 Chunk 只包含 `chunk.zip`。GUI 和 CLI 的 BA 运行产物统一写入当前 Chunk
的 `bundle_adjust/<run>/`，不再混入 `assets/`；生效相机参数仍写回 Chunk 文档，
综合报告继续位于 `reports/`。
其中 `images[*].camera` 只表示导入/初始化/空三得到的相机模型与当前解算结果；外部
GNSS/IMU/POS 观测独立存入 `assets/camera_references/camera_reference_set.json`，按
`image_uuid` 绑定，并同时保留 raw 与 resolved 状态。未完成 CRS、姿态轴向和杆臂方向
归一化的 raw 数据可以在“参考”面板查看，但不会静默标记为可用 BA 先验。控制点、检查点
和标尺继续存入 `assets/control_points/marker_set.json`。
`ProjectWorkspaceStore` 只解析 `plascan:///chunk/...` 逻辑 URI，不再接受
`plascan:///workspace/...`。旧版根级 `workspace/` 分体工程和旧版
单体工程均明确拒绝加载，并保持旧文件不变。归档条目在组合物理路径前执行跨平台名称、
大小写冲突和目标根目录边界校验。
项目配置按应用设置、工作流配置、项目视图状态和运行时缓存四层管理，避免机器路径混入工程。
Chunk 保存将核心、结果、配置和资源索引合并为一次 `doc.json` 更新，并推进 `revision`；
资源索引按引用集合和文件大小/修改时间增量维护。工程打开期间持有
`.files/.plascan.lock`，避免 GUI/CLI 并发覆盖。`ProjectData` 的关闭/析构路径先同步 drain 最新归档，
归档失败时写完整临时恢复快照，成功后才释放 reservation、运行工作区和项目锁；关闭失败会让
create/open 保留原会话。异步持久化结果同时按提交代次与会话代次仲裁，旧会话 queued callback
不会污染同路径重开后的状态。共享影像只在所有 Chunk 和有效临时恢复快照都解除引用后清理；
临时元数据存在但不可验证时 GC 保守中止。
格式结构、路径安全规则和拒绝策略见
[`docs/project/PLASCAN_PROJECT_FORMAT.md`](project/PLASCAN_PROJECT_FORMAT.md)。

参考数据工作流将 schema 为 `plascan.planetary_laser_dataset` 的 JSON 登记为
`planetary_laser_shots`。GUI 的 `ProjectManager` 在参考约束重新平差前加载并校验 PlaScan SI JSON，
拒绝 line-scan 和 round-trip，要求用户显式确认当前相机求解 frame；非零杆臂还要确认传感器 frame。
确认后的选项交给后台 `BundleAdjustService`，主线程不直接执行求解。ISIS `LidarData` 因缺少必要上下文，
GUI 不猜测补齐；用户需通过 `bundle_adjust_cli --laser-range-isis-*` 导入，或先转换为 PlaScan SI JSON。
服务在 `ba_run_summary.json` 的 `planetary_laser_range_summary` 中保存数据源、关联统计、range RMS 和逐 shot
结果，无有效 shot 时不写回相机。

### 菜单结构

```
PlaScan 菜单栏
├─ 文件
│  ├─ 新建 / 打开 / 保存 / 最近打开
│  ├─ 导入 → 导入点云... / 导入模型...
│  └─ 导出 / 退出
├─ 视图
│  └─ 影像、模型与窗口显示控制
├─ 工作流程
│  ├─ 添加照片 / 添加文件夹
│  ├─ 空中三角测量...
│  ├─ 创建点云... / 生成模型... / 生成纹理...
│  ├─ 创建 DEM / 生成正射影像
│  └─ 设置...
├─ 模型
│  └─ 显示/隐藏项目及模型显示模式
├─ 工具
│  ├─ 连接点 / 标记 / 前方交汇精度检验
│  ├─ 手动点云剔除 / 相机校准 / 相机格式转换
│  └─ 参考 DEM/LiDAR 导入与精度检查
└─ 帮助
```

模型生成仍由一个统一入口负责几何重建；已经存在模型时，可通过“工作流程 → 生成纹理”独立重建
OBJ/MTL/PNG 纹理产物。旧网格重建、模型导出对话框及顶层“重建”菜单均已删除，
避免两套几何入口产生不同设置和结果。

“文件 → 导入”只读取 Metashape 已导出的标准成果，不解析 `.psx`、`.oc3` 等内部格式。点云
支持带顶点色的 OBJ、PLY、XYZ，导入后登记到 `dense_cloud_results`，可直接作为 DEM 或点云
DOM 输入；模型支持 OBJ、PLY，OBJ 的 MTL 与其引用纹理会一起复制并登记到 `model_results`。
扫描和复制在后台线程执行，写回前校验项目与 Chunk 会话代次。

### 数据流 (稀疏重建 → 密集重建 → 模型)

```
影像导入
  │
  ├─ 1. CUDA SIFT + TensorRT LightGlue → 每影像一个 `.pifeature` 目录和 `.pimatch` 分片
  │     ├─ SIFT 描述子只存在于任务内存，`.pifeature` 保存全部提取点几何
  │     └─ `.pimatch` 保存原始匹配引用的像点、匹配、残差和版本指纹
  ├─ 2. 多视连接点轨迹整理             → latest_tie_points.json
  ├─ 3. 空中三角测量 / 增量式 SfM       → 相机位姿 + 稀疏点云
  │     ├─ 构建观测网络
  │     ├─ 初始化相机位姿 (PnP)
  │     ├─ 三角化 → 初始稀疏点云
  │     ├─ 光束法平差 (Bundle Adjustment)
  │     └─ 稀疏点云后处理
  │
  ├─ 4. 密集重建
  │     ├─ 密集匹配 (dense_match)          → 逐像素视差图
  │     ├─ 深度图估计 (PatchMatch)          → 深度图（生成模型缺失时自动执行）
  │     ├─ 深度图融合                       → 密集点云
  │     └─ 密集点云后处理
  │
  └─ 5. 模型生成 / 地形产品
        ├─ 网格重建 (物体 Poisson / 航测 height-grid) → 三角网格
        ├─ 纹理映射                         → 带纹理模型
        ├─ DEM 生成                         → 数字高程模型
        └─ DOM 正射影像生成                 → 正射影像
```

正射影像使用独立的异步数据流：

```text
MenuWorkflowController
  -> MapProjectDialog（常规/RPC 产品模式、DEM/彩色点云、投影参考、输出估算、进度与取消）
  -> ProjectManager -> ProjectTerrainProductsManager
  -> GuiTaskRunner::runGuardedWithOutcome
  -> 产品模式分流
       ├─ 常规 -> core/project_workflows::runOrthoProduct -> TerrainPipeline
       └─ RPC -> stereo_dem::RpcDomGenerator
  -> OrthoGenerationOptions
       ├─ DEM + Images -> OrthoProjector
       └─ PointCloud + PointColors -> PointCloudDomGenerator
  -> DemDomIO（RGB+覆盖 Alpha GeoTIFF 或 RGBA PNG）
  -> project_results.ortho_results[]
```

对话框从项目读取最新相对 DEM、稠密点云、影像相机和蒙版就绪数，显示实际坐标系、X/Y 像元、
裁剪边界、输出宽高和预计内存。全球点云模式还显示自动或手动的天体中心、平均参考半径和
中央经线。运行期间参数被锁定，`orthoPipelineStarted`、
`orthoPipelineProgressChanged` 和 `orthoPipelineFinished` 把后台状态回传同一对话框；
取消通过共享原子标志传入核心投影循环，切换项目后旧任务不会写回当前项目。

生产正射链支持三种有效组合：`dem_grid + dem + images`、`planar + point_cloud + point_colors`
和 `cylindrical + point_cloud + point_colors`。平面点云模式按 XY 落格并以最大 Z 选择可见颜色；
全球模式以点云 XYZ 为体固连轴，按经纬度展开到完整 `2πR × πR` 等距圆柱网格，并将实际中心、
参考半径、中央经线、投影 WKT 和仿射变换写入结果。尺寸可使用独立 X/Y 像元或最大边像素，
并受统一像素预算限制。若有效 DEM 表面没有任何相机影像覆盖，核心直接失败，不登记全黑成果。有效覆盖
会进入 GeoTIFF/PNG Alpha；当前 `ortho_projector_v1` 尚未建立逐相机地形遮挡深度缓冲，
因此陡峭地形仍需质量复核，不能把 Alpha 当作遮挡正确性的证明。

新增计算后端统一接受 Auto、CPU、CUDA 和 OpenCL：只有 Auto 会按 CUDA → OpenCL GPU → CPU
顺序降级；显式 CUDA/OpenCL 在构建未包含后端、设备不可用、索引非法或执行失败时直接报错，
不静默替换为 CPU。MVS 与 terrain 的执行报告保存实际后端、设备和 Auto 回退原因。各路径的设备边界如下：

| 模块 | CUDA/OpenCL 执行范围 | 当前 CPU 边界 |
|---|---|---|
| `dense_match` | Block Match 的代价卷、WTA、置信度和抛物线子像素保持设备驻留；SGM 负责代价卷和最终选择 | SGM/MGM 路径递推，以及 L-R、中值、Speckle 和影像支持验证；OpenCV SGBM 也是 CPU-only |
| MVS 密集云/融合 | `DenseCloudBuilder` 深度反投影；流式窗口融合可显式用 GPU 预计算参考帧世界坐标图 | 重投影一致性与观测融合；融合 `Auto` 保持 CPU 以避免额外全图缓冲，全局多帧 BFS 的显式 GPU 请求会失败 |
| Mesh Visual Hull | 规则网格上的轮廓、连续距离场和深度自由空间体素评估 | 输入准备、拓扑闭运算、MC33/Marching Cubes 表面提取和后处理 |
| Terrain | DEM 正射逐像元投影、候选融合和同网格 DEM mosaic | `ghost_filter`、孔洞连通域与颜色传播；显式 GPU + `ghost_filter` 会失败，Auto 回退 CPU |

Terrain OpenCL 正射投影使用双精度世界坐标，设备须支持 `cl_khr_fp64` 或 `cl_amd_fp64`。
流式 MVS 的 GPU 后端只改变反投影执行位置，不改变 CPU 上的多视几何准入与融合语义。

深度图的磁盘 manifest 哈希覆盖估计参数、影像路径/大小/修改时间、相机内外参、
匹配对质量约束和稀疏点云内容。项目结果中的深度批次还记录当前影像、相机与正式空三结果的
输入签名；签名只覆盖稳定影像身份、相机几何和空三代次，不受外部路径归档为
`plascan:///` URI 等存储迁移影响。旧版路径敏感签名仅在逐帧深度相机与当前工程相机一致时兼容；
重新平差、修改相机或切换空三结果后，旧深度图仍不会直接参与融合或模型生成。
深度结果另带 `algorithm_revision`；影响几何质量的生产算法升级会递增该值，生成模型工作流仅透明
复用当前修订版的完整批次，旧批次保留在磁盘并先触发当前多视深度重算。
可复用的密集点云必须与深度批次的目录、数量、配置哈希和输入签名一致，并通过 PLY 头与
非零顶点数检查。

MVS 深度统一表示与工件 `camera_model` 对应相机坐标系的正向 `Z_cam`，无效值为 0。原始畸变域、
去畸变工作域和极线校正域不能混用：影像及 valid mask 使用同一去畸变映射；校正域深度返回工作域时，
按校正相机反投影并重新计算原工作相机的轴向深度，confidence/support 等标量属性只做最近邻重映射。
极线校正的搜索范围也先转换到校正相机的 `Z_cam`。PatchMatch 的 CPU/CUDA/OpenCL 相对位姿统一在
double 世界坐标中形成局部基线后再降为 float，深度后处理只统计有效邻域并使用对数深度范围权重，
避免无效零值污染以及米/毫米单位改变滤波结果。跨视几何尚未计算时，质量门把该指标标记为不可用，
不得由光度置信度推算伪造的一致性分数。

从 MVS revision 37 起，逐像素 `geometry_source_mask` 的 bit 位序由独立的
`geometry_source_indices` 持久化；它表示一致性与跨视修复实际使用的来源序列，不能用较短的 PatchMatch
`source_indices` 代替。写入、复用、完整性审计和 TSDF 载入都会检查表长、重复/非法来源及表长外置位；
同一批次的一致性来源在逐帧后置质量门运行前冻结，避免较早帧被降为 `Rejected` 后按处理顺序级联移除
较晚帧的来源。若某帧没有任何可用几何来源，则全零来源掩码与空位序表作为显式“无来源证据”状态
一起省略；非零掩码没有精确位序表仍会在写盘前失败，TSDF 也不会回退到 `source_indices` 伪造位序。
模型工作流拒绝 revision 37 以前缺少精确位序契约的环拍深度工件，要求重新估计深度图。

工作区树只显示一个不可打开的聚合“深度图”节点，用于表明当前深度帧数量、质量档位和
过滤模式；不会为每帧深度或预览图创建独立资源。聚合节点支持整组删除，删除时清理最终层、
Level 1/2/3 栅格及对应项目元数据，但不会删除源照片。照片工具栏的“显示深度图”按
`ref_image` 精确匹配深度结果；最终层和各金字塔层分别计算可用状态，某一级别未保存时只
禁用该级别，不会锁住整个深度按钮。GUI 新建配置默认保存 Level 2/3 可视化栅格。检查深度
期间临时隐藏特征点和匹配残差，关闭后恢复用户原有偏好。`valid_coverage` 是规范覆盖率字段，质量报告同时兼容
`depth_quality.valid_coverage`、旧 `valid_ratio` 以及有效像素计数；没有可用测量时显示“—”，
不得解释为 0%。

深度 manifest 同时区分最终有效深度 `valid_mask_path` 与项目/内容允许区域
`support_mask_path`。`depth_completeness` 记录蒙版内覆盖、小内部孔、大内部开口、边界相连缺失，
以及输出过滤、多视一致性和最终后处理前后的有效数。质量门按实际 source view 数和过滤档位选择
一致性策略：单源只剔除明确矛盾，多源在轻度/中度/严格过滤下分别要求至少 1/2/3 个独立源确认；
环拍轻度过滤且至少四源时要求两个源确认。双源相对深度阈值为 10%/6%/3%；三源以上的航测
阈值为 3%/1.5%/0.75%，汇聚环拍阈值为 1.25%/0.8%/0.5%，并执行源到参考的往返投影检查。
蒙版内 0.80 覆盖门槛只作用于 project/content 约束蒙版；`full_image` 航测帧沿用边缘/内部帧的
场景化一致性阈值，不按整幅影像覆盖率误降级。

最终深度帧还保存 `missing_reason_path`（无损 `uint8` 原因码）和
`missing_reason_preview_path`（透明彩色预览）。原因码区分蒙版外、PatchMatch 未求解、低置信度、
局部离群、小连通域、多视几何冲突、跨视证据不足和未分类缺失；被跨视修复或锚定修复重新赋值的
像素会恢复为有效状态。`missing_reason_summary` 保存同一分类的逐帧计数和蒙版内缺失率，流式与
常驻内存的一致性路径使用同一结构。模型属性汇总所用深度批次的缺失率和主要原因。

环拍对象在首轮深度估计后，会先用投影 SfM 锚点审计原生深度的绝对深度残差；证据充分但已经达到
`validation_only` 或 `rejected` 范围的帧禁止执行缺口恢复，避免把错误深度层扩展成高覆盖率诊断图。
缺少足够稀疏锚点时保持非阻塞。通过预检的帧才会对蒙版内仍无解且不超过安全面积上限的缺口执行
一次定向恢复。恢复区域由最近的有效实测深度建立有限距离先验，只使用 source plan 排名前两位的
源视图重新运行 PatchMatch；候选必须同时通过置信度、相对先验深度差和双假设逆深度一致性门限，
随后仍参与常规跨视一致性与后处理。
产物通过 `targeted_gap_recovered_mask_path` 标记最终仍有效的恢复像素，并在
`targeted_gap_recovery_diagnostics` 中记录请求、候选、接受及拒绝数量。该路径不对整片缺口做
几何插值；诊断重放可用 `--disable-targeted-gap-recovery` 生成同输入 A/B 基线。

最终有效深度另保存 `depth_provenance_path`（无损 `uint8` 来源码）和
`depth_provenance_summary`。来源码区分原生 PatchMatch、缺口定向 PatchMatch、跨视图测量恢复与
锚定插值，且所有正深度像素必须恰好属于一种来源。模型工作流选择“禁用插值”时，TSDF 仍接受
前三类有影像测量证据的深度，只拒绝锚定插值来源，并在融合统计中单独记录策略拒绝数；旧版没有
来源图的非当前深度批次不会被当前算法修订透明复用。

MVS 自动场景分类先验证航测几何；其它采集只有在相机中心互异、近似共面、围绕稀疏云中心形成
半径稳定且最大角缺口不超过 120° 的闭合环，并且光轴收敛中位数/P90 同时通过时，才启用环拍专属
恢复和连续几何证据。其余前向、共线、重合、非平面或不完整采集统一进入通用 `custom` 配置并使用
中等过滤，避免把“非航测”等同于“环拍”。该决策写入深度工件，旧 Auto 分类结果不会透明复用。

MVS 源规划优先使用从当前存储匹配结果经 USAC/MAGSAC 验证的像对。空匹配文件或少于最少
内点数的匹配只表示验证证据缺失，不能当作已证明失败；剩余源位由共享轨迹几何补足。16 视图及
以上的密集环拍高质量任务最多使用六源，12 视图等稀疏环拍强制封顶为四个近邻源；通用 GUI
质量预设传入的更大候选数不能覆盖这个场景上限，避免把约 90° 的第三环邻居当成逐像素必需确认。
环拍最大三角化角从实际候选角度分布自适应计算，并受四源 70°、六源 90°
安全上限约束。1024 级高质量影像的深度金字塔保留 `4→2→1` 全分辨率末层；大图继续使用配置的
最终降采样以控制显存和运行时间。默认仍把最终结果放大到 prepared raster 尺寸；
CLI 可显式开启默认关闭的原生最终网格实验，但仅 `custom` 且未极线校正的帧会生效。
深度相机按实际网格尺寸以半像素约定缩放，同时继续保留全分辨率 prepared raster/camera 供纹理和重放。
所有像素域后处理配置仍解释为 prepared full-raster 像素，并按实际 grid/raster 的线性或面积比例量化；
ds4 上不足一个网格像素的 3x3 局部核会变为 identity，而不是扩大为约 12 个原图像素。
融合会先在 full-raster 域应用少视图/流式运行时覆盖，再按每个目标帧各自的实际网格独立缩放重投影阈值与局部梯度半径；
manifest 中的融合参数明确标记为运行时覆盖前的 base 值，避免误解为最终窗口阈值。
逐帧 artifact/manifest 显式保存 `effective_native_final_depth_grid`、raster/grid 尺寸、x/y/linear/area scale
及 configured/effective 参数。环拍、航测、极线校正或未解析场景会失败关闭到全尺寸契约，开关也纳入工作区配置 hash。
一致性开始前会冻结整批来源资格和每帧位序表，随后发生的帧级
拒绝不会按处理顺序删掉晚帧来源。没有任何逐像素来源证据时，工件同时省略全零来源掩码和空位序表；
非零掩码缺少精确位序仍会失败关闭。以上行为写入 source plan、算法修订号和重放报告，旧深度缓存
不会静默复用。
OpenCL 深度核保留参考图局部内存和运行时缓存；离散 OpenCL GPU 保持单设备执行槽，统一内存核显使用
最多两个有界执行槽交错独立帧，以覆盖 Windows 驱动在超大 kernel 链之间的提交空档。最高质量档不削减深度候选/细化
采样，也不启用 `-cl-fast-relaxed-math`。CUDA 与 OpenCL 共用确定性逆深度初始化、采样/传播预算、
棋盘传播候选、随机扰动、5% 默认 hint 搜索半径和光度唯一性语义；设备端浮点执行与图像缓存布局
仍允许产生小幅数值差异，并由跨后端掩膜平面回归测试约束。自动后端会把去重后的 CUDA 与 OpenCL
设备加入同一帧级调度队列，按实测收益分配不同参考帧；没有可用 GPU 时才使用原生 CPU。显式
CUDA/OpenCL 无可用设备时明确失败，同一物理 GPU 始终只有一份跨进程租约。
OpenCL 默认用两个主机准备槽覆盖初始帧和一致性后残余重估的 CPU 投影/蒙版准备。统一内存核显为两槽
各保留一条 command queue 和工作区，离散设备仍共用单执行槽。批次日志以首个队列开始到末个队列结束为墙钟，记录队列占用率、
调用间空档、队列内非 kernel 时间和端到端 kernel duty，避免把任务管理器的低频采样尖峰直接当成
设备空闲。

深度回归工具位于 `scripts/validation`。`compare_plascan_depth_runs.py` 对两个 workspace 的最终帧按
`ref_index + image basename` 配对，报告 pooled coverage、mask IoU、相对深度、几何支持、逆深度离散度
和 acceptance 转移，并提供可选质量门禁；批次分位数使用确定性有界 reservoir，精确保留计数、均值和
最大值。`compare_mvs_depth_to_metashape.py` 将原始深度残差、诊断性全局尺度归一化后的形状残差，以及
匹配相机中心成对距离推导的参考坐标尺度分开报告，避免把不同坐标单位误判为 PatchMatch 深度尺度错误。
2026-08-12 三类真实数据结果见
`docs/benchmarks/2026-08-12-depth-rendering-real-data-validation.md`。
`plascan_eth3d_surface_eval` 独立加载 ETH3D `scan_alignment.mlp`，对 TSDF/Visual Hull 三角网格报告
确定性双向采样距离、连通分量、边界/非流形边和 Euler 数。它不把逐帧深度提升自动视为模型提升；
通用场景的默认关闭深度层实验只在连通的低可靠区域搜索局部 PatchMatch 假设。候选限定为实测原生或
投影深度，并由公共的置信度加权 robust 几何代价、投影误差、三独立源和双基线方向门裁决，因此 CPU、
CUDA、OpenCL 仅负责生成假设，准入语义不在三套后端中重复。Ambiguous 像素最多连续修正 1%，Rejected
像素满足严格代价优势后才允许换层；任何接受像素都保存真实来源位图、逆深度统计与 evidence confidence。
对应跨视阶段快照可携带一次九通道 geometry-rerank 审计矩阵，后续阶段不重复该不变载荷；所有快照仍是
non-authoritative、预算受限且默认关闭。
从 MVS revision 44 起，几何修正试验同时保存原始光度置信度和独立跨视几何置信度。
只有实际被修正的像素具备三个独立来源、两个基线方向、可复算的几何代价优势，且帧级稀疏残差与
离散几何核仍通过时，质量门才能将相对 retention 损失解释为有因果证据的转换；绝对覆盖、连通性、
搜索边界和稀疏误差门始终不会被这个解释绕过。没有实际修正像素的 treatment 也不能间接提权。
报告会明确区分该采样指标与 ETH3D 官方遮挡感知指标。

PlaPoint 的 CPU-owned 高层接口对稀疏/稠密点云预处理独立执行 CUDA → OpenCL → CPU 选择，覆盖
体素降采样、统计/半径离群点过滤、KNN 法向估计和高度格网；显式设备请求不静默替换。OpenCL 的公共
基础设施由 PlaMatrix 提供，要求 OpenCL C 1.2，包括设备枚举与选择、context/command queue、
program cache、device buffer 和执行封装；PlaPoint 只保留点云领域 kernel。第一阶段这些高层算法的
输入输出仍驻留 CPU，主机索引、
排序、属性聚合和协方差/SVD 等阶段尚未全部迁移到设备端；PlaMatrix 尚未宣称支持 OpenCL GEMM 或 SVD。
Poisson 的稀疏 PCG 后端与预处理设备独立，可使用 PlaMatrix 的 CUDA、OpenCL 或 CPU 后端。OpenCL
路径一次上传 CPU-owned CSR 系统，在设备端执行 SpMV、Jacobi 预条件、向量更新和分层归约；显式
OpenCL 严格失败，Auto 则按 CUDA → OpenCL → CPU 回退并保留未收敛迭代结果作为下一后端初值。

`depth_tsdf` 直接消费深度帧，不经过密集点云。深度产物统一存储物理前向的正 camera-Z；
`DepthRayMetric` 按像素反投影换算离轴欧氏射线距离，并以对称半像素边界射线估计横纵世界像素足迹，
为后续基于像素尺度的跨视图容差和可见性前后偏移提供同一度量。环拍工作区把帧分为主融合帧、低权重
`validation_only` 辅助表面帧和真正拒绝帧：辅助帧只在表面支持带内补充有效深度，不估计包围盒，
也不投票自由空间。默认的辅助表面融合还要求体素位于 Primary 实测表面的两体素邻域内；
脱离该邻域的 Auxiliary 证据被计数并拒绝，不能单独扩张融合域或生成新组件。帧均值置信度只连续降权，
默认不再整帧硬剔除；显式启用硬剔除时仍需通过
相机环向覆盖保护，不能形成超过中位视角间隔两倍的连续角度缺口。
环拍默认关闭支持蒙版自由空间雕刻；显式启用时需要至少五个参考相机一致，而且已有表面证据的体素
会否决该自由空间票。航测和无法识别场景的旧工作区继续忽略掩膜外采样。TSDF 通用默认截断带为
7.5 体素。环拍高细节路径会从有效深度及逆深度相对离散度估计 `depth × relative_spread / voxel_size`
的 P90；只有该不确定度超过基础截断带的 1.2 倍时，才按 `0.4 × P90` 增加截断带和表面支持带，
通常限制在 12 体素以内。若环向覆盖分析同时检测到显著相邻视角缺口，则使用更保守的
`1.5 × P90` 自适应系数，并把上限放宽到 16 体素，以容纳缺口两侧相机姿态和深度之间的投影偏差。
没有显著缺口时不会使用该额外宽带，避免 Dino 的邻近表面或 Temple 的细柱被不必要地融合。
报告保存样本数、P90、增加量、环拍缺口策略是否生效及实际带宽；显式
`tsdfUncertaintyAdaptiveTruncation=false` 或 `tsdfOrbitalGapAdaptiveTruncation=false` 可分别关闭
基础不确定度适配或缺口附加适配。跨视修补像素的局部 TSDF 锚定恢复仍保留为实验开关，但同输入
A/B 对开放边帮助不足，因此不进入环拍默认值。环拍高细节路径还会按输入深度图
短边自动限制可靠体素分辨率，并按分辨率平方同步缩放目标
面数；640×480 深度图的 GUI 384/24 万面请求实际配置为 192/6 万面，避免用超过输入采样能力的体素
和面片拟合像素噪声。可通过 `tsdfOrbitalAdaptiveResolution=false` 显式关闭该策略。普通插值仅填边界
不超过 48 条边且物理直径不超过 10 体素的小闭环，并在 JSON 中记录
单/多视支持、拒绝原因、分量面数/包围盒及补洞前后边界数。超高质量档还会剥离两轮至少含两条开放边
且带弱相机支持顶点的终端悬挂三角形，以及只有一条开放边但三个顶点均为弱支持的薄片，并执行两轮
限位边界平滑；候选面和实际移除面数会单独记录。PLY

当环拍小天体的局部深度未通过完整性/拓扑判定、自然背景策略已经生效，并且工作区提供正式 SfM PLY
与逐点质量 sidecar JSON 的完整配对时，模型工作流优先调用
`OrbitalSparseScaffoldSurfaceBuilder` 构造独立的闭合全局载体。PLY 与 sidecar 被视为不可拆分的同源工件：
`SparseSfmPointQualityReader` 流式读取 `point_xyz`、`track_len`、`rms_reproj_px` 和
`triangulation_angle_deg`/`min_tri_angle_deg` 字段，不把数百 MiB 的 JSON 一次性展开为 DOM；读取后
必须验证逐点坐标与 PLY 对齐。手动剔点导致 PLY 点数少于同源 sidecar 时，仅当每个剩余 PLY 点都能按
原顺序和坐标容差证明为 sidecar 的有序子集时，才重新关联对应质量记录；sidecar 少于 PLY、顺序被
打乱或任一点无法匹配仍立即失败，禁止按索引截断或猜测。默认只保留轨迹长度至少 3、重投影 RMS
不大于 `1.5 px`、
最小三角化角不小于
`5°` 的点，再执行有限值、全局径向及统计离群点过滤和 PlaPoint 体素降采样，并以鲁棒中心生成向外径向
法向。稀疏 SfM 点不注入 TSDF，因为它们不具备深度像素的自由空间和实测表面语义。

过滤后的有向点由 `ScreenedPoissonSurfaceBuilder` 交给官方
[PoissonRecon](https://github.com/mkazhdan/PoissonRecon) Screened Poisson 实现；源码通过
`3rdparty/PoissonRecon` submodule 固定在 commit `262b0f539d404057d1f36e1adc07fc9388678899`。默认参数为
depth 9、`pointWeight=4`、
`samplesPerNode=1.5`、scale 1.1、8 次求解迭代、CG 精度 `1e-3`，并启用 manifold 提取。无效样本和零
法向被拒绝，`pointWeight` 必须严格大于 0，不能静默退化为非 screened Poisson。这里的官方适配器与
PlaPoint 通用表面重建中的 Poisson/PCG 后端是两条独立实现，环拍稀疏全局载体使用前者。

Poisson 输出先只保留最大面连通分量，剔除卫星碎片。若该分量还不是单连通、闭合、genus-0 的二流形，
`MeshVoxelTopologyRepair` 必须执行保守三角形体素化、六邻域外部空域洪泛和从小到大的形态学闭运算；
必要时只保留最大实体分量。候选只有在体素实体为单分量且 Euler 数为 1、其边界为单分量闭合二流形且
表面 Euler 数为 2 时才能返回。默认闭运算搜索上限为 8 个体素，但仍从半径 0 开始，并在首个严格通过
的结果停止；COLMAP building 回归数据在半径 4 首次通过，因此提高搜索上限不会强制已通过模型使用
更大的闭运算。已选择该补全分支后，配对不完整、sidecar 缺失或错位、过滤后点数不足、
Screened Poisson 失败、拓扑修复无法证明上述条件，或最终深度完整性/质量门失败，都会立即停止写出，
不会仅凭原始 PLY 或旧载体静默继续。两项稀疏工件均未提供时不会伪装成该补全分支，仍由常规 TSDF
交付门决定是否允许输出。

常规环拍任意 3D 深度主体使用双分辨率隐式表面路径。72 级规则网格执行可见性占据图割：深度前方提供
空域证据，深度邻域提供表面/实体证据，轮廓只作有界先验；占据体按“柄修复、良构修复、内部气泡
清理”循环到固定点。该规则网格只是低分辨率的内外符号/拓扑先验，不再把单元外壳直接作为最终
网格。完成器把其符号约束与原生高分辨率 TSDF 残差合成，然后由 MC33 在 192/384 级场上插值真实
零交叉；被占据先验恢复的样本允许参与符号变化，避免再被“两个端点都必须有原生深度支持”的门限
切掉。这与 Open3D 在目标 TSDF 分辨率上插值零交叉的几何原则一致，同时保留 PlaScan 的多视可见性
补全。72 级单元边界提取仍保留为显式兼容/诊断开关，但不再是环拍默认值。
当模型对话框把插值设为“已禁用”时，模型工作流只禁止深度图插值：锚定插值来源、无效像素近邻恢复、
TGV 未支撑样本恢复、隐式轴向缺口恢复、无约束视觉外壳替换以及按深度/轮廓推断的大孔补面均关闭。
环拍默认的可见性占据先验和深度/轮廓/自由空间约束仍保留，用于在实测深度样本之间构造连续曲面，
而不是要求每个输出三角形都落在原生深度像素上。提取后只允许对有界微小边界环做网格域封口；该步骤
只增加三角面，不创建、覆盖或外推任何深度/置信度/掩膜像素，可通过 `holeFill=false` 显式关闭。禁用插值
的环拍闭合网格还允许一次受拓扑事务保护的保守表面细化：只读取 Primary 帧中具有原生/跨视实测来源的
最近像素，拒绝辅助帧、缺失来源和锚定插值样本，并将单次可接受位移混合上限固定为 0.25。各轮候选还要
相对细化入口网格，在全部 Primary 相机的 prepared 彩色栅格域中满足累计顶点投影位移 P95 不超过 0.75
像素；无彩色栅格时才回退到深度栅格域。超限时依次回退到 0.125/0.0625 的更小混合步长，投影不足则
失败关闭。该步骤只移动既有顶点，不修改任何深度工件；面积、
体积、闭合流形、相机像素位移或深度完整性门不通过时仍整体回滚。报告记录配置预算、采样数、实际接受
P95、最大评估 P95 和被相机域门拒绝的候选数。
禁用插值也继续在高分辨率融合场中定位零交叉；72 级占据单元边界仅能通过显式兼容/诊断开关直接输出，避免低
分辨率量化造成顶点减少和轮廓锯齿。禁用插值不绕过最终深度完整性质量门；报告会记录视觉外壳是否被
抑制、占据先验是否生效以及实际增加的封口面数。2026-08-08 的 Hyb2
CUDA-only 深度回归中，14/14 帧进入融合，直接载体兼容路径为 69,402 顶点/138,800 面，开放边和
非流形边均为 0、Euler 数为 2；最终深度完整性中位/P10/最低为 0.8907/0.8557/0.8326。该兼容结果用于
闭合基线，不再作为默认几何细节路径。对应旧混合设备结果只有 4/14 帧进入融合，并在模型正面形成整侧缺失。

这里的规则网格图割只借鉴 AliceVision 的“相机到表面为空、表面后方为实”的可见性投票原则，并不
等价于 AliceVision `fuseCut`：后者先按像素足迹融合带相机集合的 3D 点，再在 Delaunay 四面体图上做
射线图割和实体角后处理。项目若引入真正的 Delaunay 后端，应作为独立重建器和当前高分辨率 TSDF
基线做同输入 A/B，不能再把规则体素外壳描述为 AliceVision 实现。

该路径不依赖先生成密集点云，也不使用大孔三角扇掩盖失败零面；默认输出必须保持分量数、Euler 数
和闭合二流形拓扑。最后在不改变面连接的前提下，以多视深度、置信度、来源数和逆深度离散度执行
受限顶点细化，面积、体积、法向或拓扑门失败即按轮回退。可分别用
`tsdfVisibilityOccupancyCompletion=false` 和 `tsdfVisibilityOccupancyDepthRefinement=false` 显式关闭；报告记录
占据先验分辨率、完成场来源、MC33 支持策略、强制边界样本、拓扑签名、细化位移与是否回退。2026-08-02
的 GUI 默认等价回归中，Dino、Temple 和 hyb2 均为单连通闭合二流形；Dino/hyb2 Euler 数为 2，Temple
柱间真实开口以 Euler `-4` 保留且外部开放边为 0。Dino 原生深度相对 Metashape 的法线绝对点积中位数
由旧 72 级单元外壳的约 `0.819` 提高到 `0.906`，30 度内法线一致率由 `43.4%` 提高到 `57.0%`；表面积
由 `0.1075` 降到 `0.0880`，更接近参考 `0.0728`。Chamfer 从 `0.00414` 变为 `0.00440`，说明残余表面
偏差主要仍在深度/位姿与融合场，而非最终三角面载体。不能只凭闭合质量门
宣称已经达到 Metashape 的表面质量。

模型阶段定位可通过诊断设置 `retainModelStageSnapshots=true` 显式开启。工作流会在当前唯一 model run 的
`stage_snapshots/` 下以不覆盖方式保存提取面、闭合清理后、简化后、三角质量优化后、builder 降噪后、
实测深度细化后和最终降噪后的 PLY，并在 `model_result.json` 中记录每一阶段的 SHA-256、大小、拓扑、
Euler/genus 和三角质量。该设置默认关闭，正常生成不承担额外 PLY 写盘开销；显式快照写入或校验失败会
终止本次模型发布，避免报告与实际几何不一致。

旧工作区若 manifest 指向的最终深度文件已经缺失，但 `pyramid_levels` 中仍有成功且存在的产物，模型加载
只回退到清单内最高分辨率的可用层，并按栅格比例缩放相机内参；几何计数/来源位掩码用最近邻、连续
逆深度统计用面积采样同步降尺度。没有 manifest 时目录扫描明确忽略 `depth_*_level_*.bin`，避免把中间
产物误识别成无相机的最终帧。

384 级任意 3D 模型且目标不超过 24 万面时，目标面数优先使用项目原生拓扑安全 QEM 的 link condition、
法线偏差和翻转约束减面，并执行有绝对位移上限的特征保护平滑；简化前先统一共享边面朝向，
无法全局定向的极少数冲突面按最小集合移除。候选被显式关闭、未达到目标或未通过拓扑质量门时，
会以不带平滑的原生 QEM 进行一次保守重试。若碎边使 QEM 停在目标两倍以上，
则以半体素起步、12 万面为质量地板执行自适应体素聚类后备。后备结果会清理重复/非流形面、再次执行
受限小孔填充，并比较前后开放边、悬挂边界点和非流形边；任一拓扑指标增长超过 10% 时自动回退。
体素聚类把开放边界顶点与内部顶点、不同开放边界分量分开聚合，避免轮廓质心被内部点向内拉缩。
384 级任意 3D 模型且目标不超过 24 万面时，后备体素结果若仍高于质量地板，会自动再执行一次同样
受拓扑门保护的 QEM 抛光；Dino 六源对照中面数由 391,610 降至 315,835，边界边由 46,656 降至
42,847，轮廓边缘 P90 由 7.91 降至 7.65 像素，而 Temple 的 85,208 面候选未触发该阶段。
QEM 候选的拓扑、边界、锐边和法向翻转检查按只读网格并行执行；同批边收缩只互斥真正共享一环面
邻域的候选，不再误扩展到两环。连续两轮减面不足 0.1% 且仍远离目标时，QEM 会记录停滞并转入已有
体素后备，避免重复全网格空扫描。Dino 2,449,640 面的同配置回归中，简化耗时由 260.8 秒降至
93.4 秒，最终 311,913 面、41,463 条开放边，质量指标未回退。
所有主 QEM、后备 QEM、最终补洞后 QEM 和补洞候选还必须位于 TSDF 包围盒外扩两体素范围内；
数值病态收缩若把顶点推离有效体积会整候选回退，避免小尺度模型产生跨数十模型单位的爆炸面片。
JSON 中记录 `voxel_fallback_*` 统计，便于区分 QEM 正常达标与后备减面。
Ultra 384 且目标面数不超过 12 万的路径在所有简化完成后，还会重新检测最终开放边界环。只有不含
多视图剪影保护顶点、边数不超过 128 且物理直径不超过 32 体素的内部闭环才允许补洞；若新增面超过
10%、非流形/悬挂边界增长、开放边没有下降或瘦长三角形比例超过 5%，整次最终补洞会自动回退。
`final_hole_fill_*` 记录保护顶点/孔数量、修复孔与新增面、前后开放边和三角形质量。没有任何可靠
多视图剪影保护顶点时不执行该阶段，避免把真实大开口误封。
三角形质量优化后还会执行一次残余微孔收口，只处理不含剪影保护顶点、边数不超过 16 且直径不超过
4 体素的小闭环；开放边未下降、非流形/悬挂边增长、面数增长超过 5% 或瘦长面恶化时整次回退。
`residual_micro_hole_fill_*` 单独记录尝试、接受、保护/修复孔数量及前后拓扑质量。
环拍物体对深度帧均值置信度执行 median/MAD 低尾检测并连续降权。硬拒绝为显式选项；若启用，
还受最少保留帧数、最大拒绝比例和环向覆盖共同约束。报告保存主/辅助/拒绝帧数量、原始
`ref_index`、中位视角间隔、最大角度缺口及其比值。
网格后处理结束后，`DepthMeshCompleteness` 从每帧有效且位于支持蒙版内的深度采样反投影到世界坐标，
通过 `TriangleDistanceIndex` 的 BVH 计算到真实三角形表面的精确最近距离，避免减面后大三角形因仅采样
顶点或面心而产生假性漏检，并输出逐帧召回率、P10 和中位数。环拍帧还带普通、最大角缺口边界和
缺口对侧角色；质量门单独检查缺口边界最低召回。基础 TSDF 完整性只作为阶段诊断，环拍视觉外壳
补全、深度约束细化、减面和最终降噪完成后，才对实际写出的最终网格执行硬质量门。错误信息只列出
真正未达到阈值的条件，并附最差帧号、召回率和角色，不把整侧缺失或大面积空洞的结果登记为成功模型。
环拍工作区的完整性距离容差默认取至少一个 TSDF 截断带，吸收同一隐式表面带内的量化误差；召回率
阈值保持不变。GUI 模型任务日志记录输入配置、各阶段首次状态、基础及最终完整性、网格规模和输出路径，
进度在核心阶段和 GUI 汇报层均保持单调，补全候选被拒绝时记录原因及实际采用的回退表面。
公共 `Logger` 的终端输出与 GUI sink 相互独立：默认终端显示带时间戳的 INFO/WARN/ERROR，文件和 GUI
仍接收完整 DEBUG 诊断。设置 `PLASCAN_TERMINAL_LOG_LEVEL=DEBUG` 可在终端展开逐层 PatchMatch、相机、
掩码和融合拒绝统计；默认 MVS 终端只输出阶段配置、每帧一行结果、批次汇总及异常。
Ultra 384 的低面数输出（目标面数不超过 12 万）会在等值面提取前启用受约束局部表面片支持：候选
必须满足独立的观测权重门、至少两个几何来源、逆深度离散度、自由空间比例、核心邻域来源重叠和
法向一致性。该路径用于在强减面/体素后备前补足局部 TSDF 支持；高面数 QEM 默认关闭，避免在不触发
后备重采样的网格上增加裂边或扰动纹理。`tsdfSurfacePatchSupport` 和
`tsdfMinimumSurfacePatchObservationWeight` 可显式覆盖自动策略，JSON 会记录实际阈值和各拒绝原因。
在 Marching Cubes 之前还可通过 `tsdfGlobalImplicitRegularization` 显式启用粗到细隐式场正则化。
该阶段只在拥有共同几何来源、至少两个坐标轴具有双侧支撑的样本上，用轴向二阶预测的稳健中值
抑制曲率噪声，并锁定原始零值和观测符号；未支撑样本的单体素裂缝恢复另有独立开关，默认关闭。
即使启用，候选也必须被双侧已支撑样本夹住，且预测 TSDF 与直接表面证据一致，不跨越大开口。
输出 JSON 单独记录裂缝候选/恢复数、场更新次数、平均/最大改变量和耗时。该能力目前保持显式启用，
待 Dino 与 Temple 的拓扑和投影质量 A/B 都通过后再决定是否纳入自动质量档。
另一条隐式场路径由 `tsdfAdaptiveTgvRegularization` 控制：融合时为每个样本累计固定 9 字节
有符号距离直方图；中央桶精确表示零，正负证据统计不再把零值计入任一侧。零面附近保持一级体素，
远离零面的同号平稳块合并成 2:1 平衡八叉树，再在稀疏
面邻接图上以 primal-dual 方法求解二阶 TGV。直方图桶中心只用于鲁棒限幅，桶内仍保留原始 TSDF
精确位置，避免量化推动零面。未支撑样本只能通过共享几何来源、正负角点和重复单元投票成组恢复，
不能逐体素扩张。全射线后方符号场另有 `tsdfAdaptiveTgvUseGlobalVisibilityField` 实验开关，但 Dino
验证表明它会形成膨胀视觉外壳，因此默认关闭。报告记录直方图/全局可见性样本数、八叉树输入和叶
节点数、合并/平衡拆分数、TGV 迭代/曲率/更新量、恢复样本数及两阶段耗时。求解只激活
`|TSDF| <= 0.85` 的窄带节点，远场保持融合值，从而避免为不影响零面的数千万远场样本构图；
默认一/二阶权重为 0.12/0.08，数据保真为 0.08，避免强 L1 保真把更新量钳成零。整条路径在 Dino/Temple
同输入 A/B 中显著降低面数、开放边和法线粗糙度后，已纳入 384 分辨率且启用原生拓扑安全简化、
目标面数不超过 24 万的
环拍高细节自动策略；用户显式设置 `tsdfAdaptiveTgvRegularization=false` 时仍可关闭。该自动策略同时
启用保守的轮廓带零交叉支持，轮廓候选仍使用 0.45 的绝对 TSDF 门、至少两个几何来源和双单元投票；
0.55、三次边界确认以及全射线后方符号场等失败实验均未进入默认值。
MC33 提取后默认执行“受支持符号变化”过滤：每个保留面所在单元必须同时含有被观测支持的非正值和
正值角点。未观测区域的默认正值不能再与真实负值带形成第二层伪零交叉；被拒绝面数写入
`mc33_rejected_unsupported_cell_face_count`，开关写入
`effective_mc33_require_supported_sign_change`。当用户禁用深度插值但启用可见性约束载体时，会关闭这个
单元级后过滤；否则它会在 hyb2 一类窄带数据上拒绝过半数已经由观测与可见性共同约束的连通表面。
禁用深度插值且有效 TSDF 分辨率不低于 320 时，环拍可见性占据载体默认使用 96 级而非 72 级，减少
粗载体对已测量细轮廓的符号量化。低分辨率、允许插值及通用场景保持原默认值；显式
`tsdfVisibilityOccupancyResolution` 始终优先。模型报告同时记录实际载体分辨率与自动策略来源。
任意 3D 深度模型在 320 及以上分辨率、目标面数不超过 24 万时即启用原生拓扑安全简化/MC33 路径，避免旧式
Marching Cubes 为每个三角形生成三个独立顶点后再依赖坐标焊接恢复连通性；后者会放大边界裂缝、法线
不连续和三角片感。384 及以上仍额外启用下述高细节正则化策略。
同一原生高细节路径仍以 2 像素深度有效边界腐蚀抑制外沿噪声，但会恢复其中第一圈满足几何支持至少 4、
逆深度相对离散度不超过 0.01 且仍位于支持掩膜内的像素。超过 24 万面或显式关闭拓扑安全简化
的路径默认不启用该恢复，配置和恢复
像素数分别写入 `effective_*boundary_recovery*` 与 `boundary_recovered_depth_valid_pixel_count`。
单波段 TIFF 在请求彩色读取时会由公共 ImageIO 明确扩展为三通道 BGR，避免顶点着色器把有效灰度影像
误判为无颜色源。顶点色经过网格 z-buffer、深度、
视角及颜色离群检查；OBJ 纹理使用原始相机影像的逐面投影 UV 图集，不再把顶点色作全局平面烘焙。
`camera_projected_atlas_v4` 对三顶点、三边中点和质心执行支持掩膜与深度一致性检查，每个面保留至多
16 个候选视角、逐纹素融合至多 8 个有效样本，再以 ICM 相邻面能量和小孤岛合并生成连续主视角标签。
相同标签的连通面构成裁剪 chart，经确定性 MaxRects 打包后逐纹素反投影；Natural、加权平均和最佳
视角模式分别执行真实的多视图采样，Natural 可按中值颜色剔除鬼影。真实共享边建立稳健 chart 偏移
图，边界带应用完整校正，chart 内部仅应用 0.35 的有界校正；单通道线性校正上限保持 0.08，避免把
真实亮暗区域全局压平。无可靠视图的面使用安全回退颜色，不能采样照片黑背景。
输出报告包含严格/宽松映射面、chart/视角数、各类拒绝原因、图集占用率、中位纹素密度、接缝色差
和峰值内存估算；`camera_projected_atlas_v3` 仅作为配置级兼容回退保留。
GUI 的生成模型入口默认请求 OBJ，同时始终保留 `model_from_mesh.ply` 作为几何/兼容回退；项目记录保存
`model_obj`、`model_mtl`、`texture_image`/`texture_png` 和最终显示路径。工作区选择模型时优先异步加载
OBJ，在后台解析 MTL 与纹理图，并在 QRhi 网格管线中按 `faceTextureIndices` 展开每个面角的 UV；
`ObjRenderPreparation` 同时在后台生成源顶点对齐的静态 VBO、三角 IBO、去重线框 IBO 和独立 UV
接缝顶点流；Shaded、Solid、Elevation 与 Wireframe 只切换 shader uniform/索引流，不再重新展开或
上传整张网格。`SceneGeometryPreparation` 在同一后台阶段生成点云基础 VBO、独立观测数属性以及
中心、P95 半径、AABB/OBB 等空间摘要，GUI 线程只负责 GPU 上传；场景类型或显示开关变化后，
失活的 VBO、IBO、模型纹理、照片纹理与缩略图图集在下一帧渲染线程中释放，避免历史最大资源常驻
显存。OBJ 读取器
使用连续文件缓冲、`from_chars` 和扁平三角形数组，避免逐行字符串流和嵌套面数组；相机纹理导出按
“相机 + 空间顶点”复用 UV，只在主视图接缝处拆分纹理坐标。MTL、纹理或完整 UV 缺失时回退到顶点
颜色，不在 GUI 主线程执行文件解析、图像读取或百万面三角形展开。照片纹理按原始颜色显示，不重复
叠加网格漫反射；场景半径下限随模型尺度变化，紧凑模型加载后无需反复手动放大。
未计算顶点颜色且没有纹理的模型默认使用 `Solid` 面法线显示，用户能直接看到真实三角面；存在
照片派生顶点颜色或完整纹理时才使用颜色显示与较柔和照明。顶点颜色只改变外观，不改变网格拓扑，
不会再用平滑白色材质把“无颜色”伪装成“已计算颜色”。

点云手动剔除与撤销只保存增量 undo，并在后台将结果写入同目录临时文件；GUI 校验当前加载代际后
再原子替换目标文件，过期任务只清理暂存文件。保存按最终扩展名分派 XYZ、OBJ 和 PLY writer，PLY
使用 BinaryLE，避免大点云编辑时的 ASCII 数字格式化热点。切换模型或销毁场景时会协作取消仍在
复制属性、重建 GPU 数据或等待写盘的旧编辑任务；第三方 writer 已开始后会写完临时文件再丢弃。

XYZ、PLY、OBJ 场景加载和手动框选均采用 single-flight/latest-only：旧 reader 若正处于第三方解析
函数内会自然完成，但不会再与后续多个大任务并发；解析后的准备阶段和框选扫描使用协作取消标志，
连接点观测 sidecar 也只保留最后请求，避免快速切换时并发解析多个大 JSON；框选索引固定为 32 位并
限制初始预留容量。RHI 更新批次只有在 `beginPass` 消费后才提交缩略图缓存
淘汰和上传状态；任一资源创建失败会先释放未提交批次，再恢复 VBO/IBO、纹理和图集的 dirty 状态。
非 8-bit GeoTIFF 显示缓存按规范化源路径加锁，在目标目录暂存并原子替换，完整图与缩略图并发请求
不会再覆盖同一个半成品 `_8.tif`。

最终深度证据除几何支持数外，还保存来源相机 bit mask、逆深度均值/相对离散度和跨视图修补掩膜；
旧清单缺少这些字段时按空证据读取。模型质量报告对每个视图保存双向边缘距离、P90 长尾位置及其
来源数、逆深度离散度和缺失阶段，用于区分深度缺失、TSDF 未成面和模型新增毛边。
深度算法修订版 15 在空间补洞之前对投影源深度做遮挡层聚类：与原生深度一致的主层在逆深度域
受限校正，至少三个来源一致且原生层被更多视图否决时才切换深度，缺失像素只接收至少两个真实
投影来源的主层。选择结果同步更新来源位图、逆深度矩和置信度，并以逐类计数写入清单诊断。
TSDF 对自适应几何冲突采用连续鲁棒权重，低冲突观测保持原权重，高冲突观测平滑衰减而不是在
阈值附近跳变；跨视图修补值仍只进行表面带融合，不能因来源数增加而自动获得完整自由空间权利。
环拍场景的锚定深度修复会先在权威支持蒙版内保留轮廓保护带，再处理由狭窄通道连接到轮廓的内部
缺深区域；项目蒙版排除区和真实开口始终不进入插值域。候选仍必须通过跨视锚点、边界逆深度离散度
和相对面积门，输出以低置信度及 `crossViewRepairedMask` 标记。清单中的
`cross_view_repair_diagnostics` 记录逐原因拒绝、轮廓保护和最终后处理插值统计。

处理基线由 `ProcessingBaselineManager` 统一管理。基线 JSON 保存相机版本、参与融合的深度帧和
影响几何的处理参数快照，并可分别冻结相机、深度、融合和网格阶段的 SHA-256 指纹，阻止输入漂移
后的结果混入同一组 A/B，同时在比较报告中直接列出发生漂移的阶段。参考网格与候选
网格统一统计面数、开放边、连通分量、瘦长面、归一化表面积，以及流形相邻三角面的法线夹角
中位数、P90 和超过 30 度的比例。面数降低但法线折叠、表面积膨胀或开放边恶化时仍判定失败，
不能再以“达到目标面数”替代平滑度和拓扑验收。
启用表面降噪时，TSDF 等值面清理前后各执行一次法线感知降噪；最终阶段额外进行一轮受限切向
整理。开放边及其保护环保持固定，候选只有在顶点仍位于 TSDF 包围盒、拓扑计数不变、瘦长面比例
不恶化且相邻面法线粗糙度实际下降时才会被接受，否则自动回退。报告分别保存最终阶段的移动顶点
数、切向整理顶点数，以及法线角和高宽比三角面比例的前后值，避免把过度平滑误判为质量提升。

---

## 四、cli/ — 命令行工具

用于测试、批处理和脚本编排的算法入口。源码和构建定义按领域分组，产物仍是独立可执行文件，
从而保持现有脚本兼容，并避免不相关模块之间产生强制依赖。

```
cli/
├── CMakeLists.txt            # 公共目标工厂与模块注册
├── README.md                 # 模块职责、目标清单和扩展约定
├── camera/                   # 相机格式转换；tests/ 保存相机 CLI 测试
├── control_points/           # 标靶检测与打印；tests/ 保存标靶 CLI 测试
├── features/                 # 特征提取、匹配与连接点生成；tests/ 就近维护
├── dense/                    # 极线校正、密集匹配、三角化和点云细化；tests/ 就近维护
├── reconstruction/           # 可独立执行的重建阶段与诊断工具；tests/ 就近维护
├── workflows/                # GUI“工作流程”菜单入口；Options/Runner/Progress/ReportContext 分责；tests/ 就近维护
├── quality/                  # 模型影像质量验收；tests/ 就近维护
├── terrain/                  # RPC 立体 DEM/DOM 入口
├── common/                   # 公共路径/token/控制台/JSON/输出策略/摄影测量列表基础设施
│   └── tests/                # 公共解析测试与 CliTestSupport
└── third_party/              # CLI11 单头文件依赖
```

各领域目录维护自己的 `CMakeLists.txt` 和直接依赖；顶层不再集中罗列所有工具依赖。
`plascan_cli_support` 由所有入口共享路径、token、UTF-8 控制台、JSON 和输出目录策略；
`plascan_cli_photogrammetry_common` 再提供摄影测量清单、相机 JSON 和蒙版索引能力。
工程型入口通过 `plascan_common_project` 的 `ProjectSession` 创建或打开根索引指定的默认
Chunk，统一处理项目 URI、共享影像、产物目录、相机与结果写回；也可通过 `--chunk-id`
或 `--chunk-name` 显式选择 Chunk。旧工程不会由 CLI 隐式迁移。
`workflows/` 提供空中三角测量、三维重建 CLI、生成模型、多视图纹理、DEM/正射完整流水线，以及原生
`small_body_terrain_cli` 全球径向 DEM/DOM 入口；GUI 不再提供旧版
“三维重建”一键对话框，改由空三、密集处理和生成模型入口分阶段执行。菜单中的项目输入操作通过
CLI 的输入清单与项目参数表达，不复制项目导入 UI。一键重建的深度融合、密集点云
细化与点云产物写出由 `core/mvs` 服务实现，入口不保留算法副本。
CLI 测试同样由各领域目录注册并放在对应 `tests/` 下，顶层 `tests/` 不再维护 CLI 聚合测试目标。

`reconstruction/cli_bundle_adjust.cpp` 同时暴露两套显式互斥的激光入口：`--laser-cloud` 读取扫描 PLY
并建立点到面约束；`--laser-range-data` 读取行星稀疏测距 JSON。后者要求
`--laser-range-camera-frame`，非零杆臂还要求相机传感器 frame 与数据 `laser_frame` 一致；ISIS JSON
通过 `--laser-range-isis-target`、`--laser-range-isis-body-frame`、`--laser-range-isis-laser-frame`、
`--laser-range-isis-sensor-model`、`--laser-range-isis-range-type` 和 `--laser-range-isis-lever-arm` 补齐上下文。
工程 `image_uuid` 自动按 BA 相机顺序合并；`--laser-range-image-alias CAMERA_INDEX=IMAGE_ID` 可重复提供 ISIS
`serialNumber` 等额外稳定标识。`unknown` 语义、
跳过未映射 shot 和忽略未映射真实 measured 像点均需单独显式开关。CLI 不进行最近时间关联或隐式坐标转换。

**统一约定**：
- `--help` / `-h` — 打印中文参数说明；CLI11 入口还提供 `--version`
- 稳定参数使用 `--kebab-case`，枚举值使用可持久化的英文 ID
- 退出码: 0=成功, 1=参数错误, 2=I/O 错误, 3=算法错误
- 进度/错误信息 → stderr，结果信息 → stdout
- `--verbose`、JSON 配置文件等只在需要它们的命令中提供，不属于全局参数
- `three_d_reconstruction_cli` 支持 `--stop-after-sfm`、`--skip-mvs`、`--skip-mesh` 分阶段运行，用于大数据 benchmark 和问题定位。
- `scripts/bench/run_photogrammetry_benchmarks.py` 扫描 `prepared/plascan/image_camera.lis`，批量调用三维重建 CLI 并汇总 JSON。

GUI/CLI 的工作流映射、按钮语义、稳定枚举值和用户示例集中维护在
`docs/GUI_CLI_GUIDE.md`，避免在架构文档中复制易过期的完整命令帮助。

**标准摄影测量流程与 CLI 覆盖**:

```
阶段 1: 稀疏重建 (GUI 完成)
  ├─ CUDA SIFT + TensorRT LightGlue   → 每影像一个 `.pimatch` 二进制分片
  ├─ 多视连接点轨迹                    → latest_tie_points.json
  └─ 光束法平差 / 增量SfM           → 精化相机 + 稀疏点云
     (GUI/CLI 由 AerialTriangulationWorkflow 编排；bundle_adjust_cli 可在已有项目分片上做 headless BA/A-B)

阶段 2: 密集重建 (CLI 可用)
  ├─ feature_match_cli    双影像匹配  → 两个 `.pimatch` 分片
  ├─ bundle_adjust_cli    光束法平差  → ba_run_summary.json / A-B 对比 JSON / 行星激光 range 摘要
  ├─ marker_detect_cli    标靶检测    → plascan.marker-detections.v1 JSON
  ├─ marker_print_cli     标靶打印    → A4/Letter PDF
  ├─ rectify_cli          极线校正    → 校正影像对 + 单应矩阵 .xml
  ├─ dense_match_cli      密集匹配    → 视差图 .tif
  └─ triangulate_cli      视差三角化  → 密集点云 .ply
```

**CLI 用法**:
```bash
# 两影像 + 两相机 → 密集点云
rectify_cli -L L.tif -R R.tif --camL camL.txt --camR camR.txt -o rect
dense_match_cli -L rect_L.tif -R rect_R.tif -o disp.tif --device cuda
triangulate_cli -d disp.tif --rect-params rect.xml \
    --camL camL.txt --camR camR.txt -o cloud.ply
```

## 五、已知技术债务

| 问题 | 位置 | 建议 |
|------|------|------|
| 三维场景实现仍较大 | `gui/views/CameraSceneWidget.cpp` | 缓存、RHI 资源 DTO、覆盖层、几何准备与点云编辑已拆分；继续将具体图集上传/绘制方法提取为独立渲染器 |
| 空三真实数据回归仍需扩大 | `core/aerial_triangulation` | 持续加入环拍、航带、弱纹理和控制点数据集 |
| 构建依赖 4 个系统符号链接 | `/lib64/libm.so.6`, `libnvrtc-builtins.so.13.0` 等 | 见 `CONTEXT.md` 系统依赖 |

## 六、构建系统

- **根**: `CMakeLists.txt` — 强制使用 vcpkg manifest toolchain；也可切换到 Qt/OpenCV/GDAL/AprilTag 源码依赖 superbuild
- **依赖**: `cmake/PlascanPackages.cmake` (统一 find_package)
- **源码依赖**: `cmake/PlascanSourceDependencies.cmake` 固定 Qt 6.11.2、OpenCV 5.0.0、
  GDAL 3.12.4 和 AprilTag 3.4.5，只初始化 Qt 的 `qtbase` 与 `qtshadertools`，并安装到 source-deps preset
  的共享前缀；PoissonRecon 从固定 submodule 直接提供头文件；`cmake/source-deps/vcpkg.json` 仅提供
  PROJ、带 RTree 的 SQLite、libgeotiff、libtiff、zlib 和图像编解码等底层依赖。GDAL 的 HDF5 与
  NetCDF 等可选驱动默认关闭，并显式启用 GTiff、HFA、JPEG、PNG、VRT、PDS/ISIS 和 JP2OpenJPEG；
  OpenCV 的 AVIF 自动探测也关闭，
  Qt 则使用内置 libpng，避免它先加载同名系统库而破坏 OpenCV 的 vcpkg libpng ABI。安装后的其它
  源码依赖保留其 vcpkg 链接目录 RUNPATH，确保开发构建和测试加载同一 installed tree 的动态库
- **OpenCV 边界**: C++ 与 Python 运行时均要求 OpenCV 5；C++ 直接使用拆分后的 `features`、`geometry`
  和 `stereo` 模块，不保留 OpenCV 4 头文件、模块名或调用签名兼容层
- **ONNX Runtime**: 固定 1.29.0 官方预编译包并校验 SHA-256，下载归档跨 preset 缓存在
  `build/env/downloads/onnxruntime/1.29.0/`，也可显式指定离线归档
- **TensorRT SDK**: Windows 标准配置入口默认探测并复用 TensorRT 10.15.1.29 CUDA 13.1 C++ SDK；用户一次性
  接受 NVIDIA 许可后可自动下载、校验并安装到 `build/env/sdk/tensorrt/10.15.1.29/`。SDK 不可用时保留
  原生 CUDA 后端，神经网络推理自动回退 ONNX Runtime CPU；CUDA 不可用时整体回退 CPU
- **Core**: 每个子模块独立 `CMakeLists.txt`, 通过 `plascan_core_add_optional_module()` 注册
- **NVRTC**: 由显式配置的 CUDA Toolkit/TensorRT SDK 提供，不拼接环境管理器私有路径
- **CUDA**: 默认探测标准 CUDA 编译器并全局 `enable_language(CUDA)`；探测失败自动回退 CPU，也可通过
  `PLASCAN_ENABLE_CUDA=OFF` 显式关闭
- **Linux CUDA/OpenCL 开发构建**: `linux-vcpkg-cuda-opencl-release` 使用独立 vcpkg installed tree，
  同时启用原生 CUDA、PlaMatrix CUDA/OpenCL；本机 CUDA 13.1 基线固定 GCC 13 host compiler 与 `sm_89`，
  并通过项目 `cuda` vcpkg overlay 显式传递 CUDA compiler，避免混用系统旧版 `nvcc`；
  TensorRT 不由 vcpkg 提供，Linux 使用外部 SDK，Windows 可由标准配置脚本安装固定 SDK
- **测试**: `-DBUILD_TESTS=ON` → CTest；按改动范围优先跑相关测试，再决定是否跑全量
## GUI 模块边界（2026-08）

- `src/common/project/ProjectSessionModel.*`、`ProjectDocumentModel.*` 和三类项目配置管理器负责
  QtCore 项目会话、文档分域与持久化；GUI 直接使用 common 中的项目接口，不再保留
  `src/gui/project/data` 旧包含路径。资源清理模块安装的 path-only open preflight 会在
  `ProjectData` 解析归档和校验资源索引前恢复事务区产物，避免缺失资源先阻断恢复入口。
- `src/core/project_workflows` 负责 DEM/正射、稀疏点后处理、点云参数/输入准备、参考数据检查和
  生成资源清理，通过独立 `project_workflows` 目标供 GUI、CLI 和测试复用。资源清理按
  `ProjectResourceCleanup{Plan,Artifacts,Transaction,TransactionManifest,Purge,Recovery}` 拆分计划、产物边界、
  WAL 事务、不可逆清除态和启动恢复职责，worker 仅消费不可变计划，不访问 GUI 或 `ProjectData` QObject。
- `ProjectManager` 以门面形式持有项目生命周期、蒙版、点云、稀疏重建、模型、地形产品和相机设置
  控制器；其对 `ProjectData` 的 UI 设置、元数据/影像查询、相机替换和交会结果兼容调用统一经过
  `ProjectSessionFacade`。GUI 中不存在“稠密重建管理器”；`ProjectPointCloudWorkflowController`
  只协调深度估计与点云融合。
- `MainWindow` 按布局、菜单绑定、项目绑定和 UI 状态拆分实现；项目打开/保存展示由
  `ProjectLifecyclePresenter` 管理，状态栏任务由 `ProjectTaskStatusController` 管理，特征点/残差显示配置由
  `FeatureVisualizationController` 管理。任务栏图标进度由 `ProjectTaskStatusController` 聚合项目打开、保存、
  DEM、正射和后台重建任务，再通过平台适配器显示；非 Windows 平台保持无副作用。
  `DataTreeWidget` 按模型、填充、上下文菜单、资源元数据和相机对齐判定拆分实现。
- 正射流程为 `MenuWorkflowController -> ProjectManager -> ProjectTerrainProductsManager ->`
  `project_workflows::runOrthoProduct`，请求在 GUI 边界转换为 `OrthoGenerationRequest`。
