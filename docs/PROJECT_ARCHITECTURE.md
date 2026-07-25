# PlaScan 项目架构文档

行星表面摄影测量处理系统。最后更新: 2026-07-14。

## 顶层目录

```
plascan/
├── src/            # 所有源代码
│   ├── common/     # 通用工具库 (日志, IO, 模型与项目公共能力)
│   ├── core/       # 核心算法库 (相机, 特征, 匹配, 标记控制网, SfM, MVS, LiDAR, 蒙版, 网格, 地形)
│   └── gui/        # Qt6 图形界面
├── cmake/          # 全局 CMake 模块 (依赖查找, 包管理)
├── 3rdparty/       # 第三方库源码 (LightGlue)
├── resources/      # 静态资源 (深度学习模型权重, 图标)
├── scripts/        # Python 辅助脚本和 benchmark runner
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
│   └── PathIO.h/cpp        # UTF-8/本机路径转换、原子文件写入和 OpenCV 图像读写封装
├── model/
│   ├── FeatureExtractorModelCatalog.h/cpp # 特征提取器模型候选文件名与托管路径识别
│   ├── TorchScriptModelResolver.h/cpp # 模型搜索路径解析（PLASCAN_MODEL_DIR、源码树和安装目录）
│   ├── Sam21ModelCatalog.h/cpp # SAM2.1 checkpoint / TorchScript 文件名和安装状态
│   ├── U2NetModelCatalog.h/cpp # U2Net ONNX 文件名和安装状态
│   └── test/
│       └── FeatureExtractorModelCatalog_tests.cpp # 特征模型候选顺序和路径识别测试
├── project/
│   ├── ProjectIO.h/cpp # 项目目录、临时缓存、资源和产物路径规则
│   ├── ProjectMetadata.h/cpp # 项目 JSON、影像 token 与资源路径解析
│   ├── ProjectCameraIO.h/cpp # Camera JSON/TSAI 读取、转换与序列化
│   ├── ProjectMatchCatalog.h/cpp # 项目匹配文件、sidecar 与负缓存编目
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
│   ├── Camera.h/cpp            # 唯一相机模型：内参、Brown-Conrady 畸变、位姿和正深度归一化
│   ├── CameraFormatConverter.h/cpp # Middlebury/EPFL 等外部相机 -> tsai + image_camera.lis
│   └── test/                      # 相机测试与诊断程序
│       ├── Camera_tests.cpp
│       ├── CameraFormatConverter_tests.cpp
│       ├── test_tsai_loader.cpp
│       └── test.cpp
│
├── feature_extractors/         # 特征点检测 (8 种算法)
│   ├── IExtractor.h             # 提取器虚接口
│   ├── ExtractorFactory.h/cpp   # 工厂: 根据算法名创建提取器
│   ├── FeatureOutput.h          # 通用特征输出结构 (所有提取器共用)
│   ├── FeatureFileIO.h/cpp      # 特征二进制 I/O (8 种文件后缀, magic bytes)
│   ├── FeatureData.h/cpp        # 特征数据容器 (fromFeatureOutput, 推荐匹配器)
│   ├── superpoint/              # SuperPoint (256d, GPU/CPU)
│   │   ├── SuperPoint.h/cpp     # TorchScript 推理
│   │   └── tests/
│   ├── disk/                    # DISK (128d, GPU/CPU)
│   │   ├── DiskExtractor.h/cpp  # TorchScript 推理
│   │   └── tests/
│   ├── aliked/                  # ALIKED (128d, GPU/CPU)
│   │   ├── AlikedExtractor.h/cpp # TorchScript 推理
│   │   └── tests/
│   ├── tradition/               # 传统算法 (SIFT/SURF/ORB/AKAZE, CPU; SIFT 可选 CUDA)
│   │   ├── TraditionalFeatureExtractor.h/cpp
│   │   └── test_*.cpp
│   └── dedode/                  # DeDoDe Python 提取器说明与脚本入口
│
├── feature_match/              # 特征点匹配和端到端匹配
│   ├── IMatcher.h              # 匹配器虚接口
│   ├── MatcherFactory.h/cpp    # 工厂 (独立库 feature_match_factory)
│   ├── match.h/cpp             # 通用匹配结果结构
│   ├── MatchFileIO.h/cpp       # 通用匹配文件 I/O (.match 索引格式 + 坐标格式)
│   ├── MatchExportIO.h/cpp     # CSV/COLMAP 等调试和交换格式导出
│   ├── MatchGeometryFilter.h/cpp # RANSAC/USAC 几何粗差剔除
│   ├── MatchVisualization.h/cpp # 匹配连线可视化导出
│   ├── superglue/
│   │   ├── SuperGlueMatcher.h/cpp      # SuperGlue TorchScript 推理
│   │   ├── export_torchscript.py       # SuperGlue 模型导出
│   │   └── usage_examples.cpp          # SuperGlue 推理示例
│   ├── lightglue/
│   │   └── LightGlueMatcher.h/cpp      # LightGlue 推理
│   ├── loftr/                  # LoFTR C++ TorchScript 端到端匹配器
│   └── tradition/
│       └── TraditionalFeatureMatcher.h/cpp  # BFMatcher/FLANN
│
├── intersection/               # 前方交汇精度检验
│   └── Intersection.h/cpp      # 多射线交汇解算 + 精度评估
│
├── overlap/                    # 重叠度分析
│   ├── OverlapAnalyzer.h/cpp   # 影像对重叠区域计算
│   ├── OverlapPairGraphPlanner.h/cpp # 无相机词汇召回后的连通影像对图规划
│   ├── VocabularyOverlapRetriever.h/cpp  # 基于已提取特征描述子的词汇重叠对检索
│   └── GroundBackProjector.h/cpp  # 地面投影
│
├── matchphototask/             # Metashape-like 匹配照片编排层
│   ├── algorithm/
│   │   ├── MatchPhotosAlgorithmPlan.h/cpp # 算法计划：当前主线 SIFT + LightGlue
│   │   └── MatchPhotosAlgorithmSelector.h/cpp # 类 Metashape 预设到算法计划的映射
│   ├── task/
│   │   ├── MatchPhotosTask.h/cpp    # 统一任务入口，完成算法选择、pair selection、特征和匹配阶段
│   │   ├── MatchPhotosOptions.h     # 自动/快速/高精度/CPU/CUDA 等任务选项
│   │   ├── MatchPhotosContext.h     # 项目路径、输出目录、影像输入、取消和进度上下文
│   │   └── MatchPhotosResult.h      # 阶段报告、特征文件记录、匹配文件记录和错误信息
│   ├── pair_selection/
│   │   ├── PairTypes.h/cpp          # PairCandidate、PairSource、pair key 规范
│   │   ├── PairSelectionPolicy.h/cpp # 自动/全量/序列/手动等候选策略
│   │   └── PairSelector.h/cpp       # 合并手动、全量、序列、相机重叠和词汇召回候选
│   ├── runtime/
│   │   ├── MatchPhotosRuntime.h/cpp # 输出路径、LightGlue 模型查找、匹配 sidecar 写入
│   │   └── MatchPhotosMaskSupport.h/cpp # 连接点流程蒙版路径解析、关键点/连接点过滤
│   ├── stages/
│   │   ├── FeatureStage.h/cpp       # SIFT 特征提取/复用，输出 assets/ip/*.sift
│   │   ├── MatchingStage.h/cpp      # SIFT + LightGlue 两两匹配，输出 assets/matches/*.match + JSON sidecar
│   │   ├── GeometryVerifyStage.h/cpp # 调用 MatchGeometryFilter 做基础矩阵几何验证
│   │   ├── SiftLightGlueRecovery.h/cpp # 高精度预算截断增强与断图跨分量恢复
│   │   ├── TrackBuildStage.h/cpp    # 连接点轨迹阶段边界，委托 tie_points 管理最终多视图 track
│   │   └── GuidedMatchStage.h/cpp   # 引导重匹配阶段占位
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
│   ├── BundleAdjust.h/cpp      # BA 公共接口、自动后端选择、legacy CPU 后端调度、后端状态回传
│   ├── BundleAdjustCeres.h/cpp # Ceres CPU/CUDA 后端，记录 dense Schur CPU/GPU、setup/solve 耗时和回退原因
│   ├── BundleAdjustProjection.h # 与 Camera 一致的模板投影模型，供 Ceres AutoDiff 固定相机残差复用
│   ├── BundleAdjustNativeCuda.h/cpp/.cu # PlaScan 自研 CUDA 后端入口和 GPU 点块求解
│   ├── BundleAdjustNativeCudaWorkset.h/cpp # 将 Camera/BATrack 扁平化为 CUDA 连续工作集
│   ├── BundleAdjustNativeCudaTypes.h / *Kernels.cuh # CUDA 侧数据类型、点块 kernel 和设备函数
│   ├── tools/                  # BA 后端基准工具，例如 ba_backend_benchmark
│   └── tests/                  # BA 模块级后端、投影模型、自动选择和约束回归测试
│
├── lidar/                      # LiDAR / 激光点约束
│   ├── LaserConstraintTypes.h  # 点到面约束、地图采样和关联统计类型
│   ├── LaserConstraintMap.h/cpp # PLY 点云读取、法线/曲率筛选、最近平面查询
│   └── LaserConstraintAssociation.h/cpp # BA track 与 LiDAR 平面约束关联
│
├── mask/                       # 照片蒙版生成与合成
│   ├── MaskGenerator.h/cpp     # 黑背景/亮度阈值蒙版、蒙版合成和轮廓提取
│   ├── Sam21MaskGenerator.h/cpp # SAM2.1 TorchScript encoder/decoder 推理，支持 CPU/CUDA
│   └── u2net/                  # U2Net ONNX 自动蒙版子模块
│       └── U2NetMaskGenerator.h/cpp # OpenCV DNN 推理，CPU 可用、CUDA 取决于 OpenCV DNN 后端
│
├── sfm/                        # Structure-from-Motion
│   ├── common/                 # SfM 公共类型和共享并查集
│   ├── geometry/               # 统一投影、三角化质量和 OpenCV 相机适配
│   ├── graph/
│   │   ├── CorrespondenceGraph.h/cpp      # 对应关系图
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
│   ├── MvsWorkspaceManifest.h/cpp # 深度帧状态、产物路径、相机/影像/配置 hash 和 source plan
│   ├── MvsSourcePlanner.h/cpp  # shared tracks / 几何内点 / 覆盖率 / baseline 选源
│   ├── MvsImagePreprocessor.h/cpp # 原图去畸变并生成正深度、零畸变的 Camera 工作值
│   ├── DepthPyramidPolicy.h/cpp # 从最终质量档生成 4D/2D/D 三级 PatchMatch 调度
│   ├── MvsSceneClassifier.h/cpp # 根据相机布局、光轴和稀疏云厚度判定环拍/航测场景
│   ├── DepthPyramidPropagation.h/cpp # 父层深度中心、不确定半径和边缘感知传播
│   ├── DepthPyramidEstimator.h/cpp # Level 3/2/1 编排与逐层摘要
│   ├── DepthCompletenessMetrics.h/cpp # 蒙版内覆盖、小孔/大开口/边界缺失与逐阶段保留率
│   ├── DepthCrossViewHoleRepair.h/cpp # 一致性过滤后以三源逆深度簇保守补回环拍对象缺面
│   ├── DepthFrameQualityGate.h/cpp # 深度帧 Accepted/ValidationOnly/Rejected 质量门控
│   ├── DepthConsistencyCache.h/cpp # 有内存预算的 LRU source 邻域多视一致性缓存
│   ├── DepthGeometryConsistency.h/cpp # 断边邻域搜索、相机基线自适应往返验证与一致性投票
│   ├── PatchMatchCUDA.cu/h     # PatchMatch CUDA 实现
│   ├── PatchMatchNoCUDA.cpp    # PatchMatch CPU 回退
  │   ├── DepthMapGenerator.h/cpp # 深度图估计、取消检查、raw depth/confidence/几何支持度/valid mask 写盘
│   ├── DepthMapFusion.h/cpp    # 深度图融合 → 密集点云，支持 manifest source plan 和流式融合
│   ├── DepthFrameUtils.h/cpp   # 深度帧存储与按指定输出目录选择批次
│   ├── EpipolarRectifier.h/cpp # 极线校正
│   ├── DisparityFilter.h/cpp   # 视差滤波
│   ├── DisparityTriangulator.h/cpp  # 视差三角化
│   ├── DensePointCloudCUDA.cu  # 密集点云 CUDA
│   ├── DenseCloudBuilder.h/cpp # 密集云构建器与点云过滤
│   ├── SparseCloudPreprocessor.h/cpp  # 稀疏云预处理
│   ├── SparseCloudValidator.h/cpp     # 稀疏云验证
│   ├── StereoDenseCloudPipeline.h/cpp # 立体密集云流水线 (主入口)
│   ├── StereoDenseCloudPipelineOutput.h/cpp  # 流水线输出
│   ├── StereoDenseCloudPipelinePaths.h/cpp   # 流水线路径管理
│   ├── PointCloudTifIO.h/cpp   # 点云 TIFF I/O
│   ├── AspPointCloudMetrics.h/cpp  # ASP 兼容点云指标
│   └── tests/                  # MVS 单元与流水线测试
│       ├── test_mvs_rectifier.cpp
│       ├── test_mvs_depth_pyramid.cpp
│       ├── test_mvs_depth_completeness.cpp
│       ├── test_mvs_types.cpp
│       └── test_mvs_pipeline.cpp
│
├── dense_match/                # 密集匹配模块 (新, 将逐步替换 mvs 中的匹配)
│   ├── README.md               # 模块文档 (详见该文件)
│   ├── DenseMatchTypes.h       # CostFunction/StereoAlgorithm/SubpixelMode 枚举
│   ├── DenseMatchConfig.h      # 参数配置结构体
│   ├── CostFunctions.h/cpp/cu  # 5 种代价函数 (AD/SD/NCC/Census/TernaryCensus)
│   ├── BlockMatcher.h/cpp      # WTA 块匹配 (CPU/CUDA 自动调度)
│   ├── SgmMatcher.h/cpp        # SGM/MGM 半全局匹配 (8方向路径聚合)
│   ├── SubpixelRefiner.h/cpp   # 子像素视差精化 (抛物线拟合)
│   ├── DisparityValidator.h/cpp # L-R 一致性/中值滤波/Speckle 过滤
│   ├── DenseMatchService.h/cpp # 服务层: 编排完整匹配流水线
│   ├── opencv/
│   │   └── OpenCVSgbmWrapper.h/cpp  # OpenCV SGBM 封装 (对比算法)
│   └── tests/ (6 个测试文件, 22 项)
│
│   └── tests/
│
├── mesh/                       # 网格重建与纹理映射
│   ├── MeshTypes.h             # 网格类型
│   ├── SurfaceReconstructor.h/cpp           # 表面重建主流程
│   ├── SurfaceReconstructorHeightGrid.h/cpp # 高度格网方法
│   ├── SurfaceReconstructorPostprocess.h/cpp # 网格清理、边界环拆分、质量补洞及退化微孔保拓扑收缩
│   ├── MeshIO.cpp              # 网格文件 I/O
│   ├── TextureMapper.h/cpp     # 纹理映射
│   ├── CameraTextureMapper.cpp # 按相机投影、深度一致性和视角评分生成逐面 UV 图集
│   ├── MeshColorizer.h/cpp     # 网格遮挡检查、鲁棒多视图顶点着色及孤立色斑清理
│   ├── MeshFaceColorOptimizer.h/cpp # 实验性按面主视图投票与共享顶点一致着色
│   ├── MeshQuadricSimplifier.h/cpp # 目标面数驱动、边界/锐边/翻转约束的 QEM 自适应简化
│   ├── BoundaryAwareVoxelSimplifier.h/cpp # 多视图剪影保护、内部开放边界可聚类的体素简化后备
│   ├── MeshTopologyQuality.h/cpp # 开放边/非流形/连通分量/三角形长宽比质量门及保拓扑边翻转优化
│   ├── MeshIsotropicRemesher.h/cpp # 内部高长宽比区域的短边合并、长边拆分与拓扑/法向保护
│   ├── DepthMapMeshBuilder.h/cpp # 深度帧 manifest/相机产物加载与 legacy 路径适配
│   ├── DepthTsdfSurfaceBuilder.h/cpp # raw depth/confidence/mask/camera 直接融合 TSDF、提取网格并安全减面
│   ├── VisualHullReconstructor.h/cpp # 显式 legacy/诊断 Visual Hull 路径
│   ├── ModelWorkflowService.h/cpp  # 模型工作流服务；保留 PLY 几何并可写 OBJ/MTL/相机纹理图集
├── terrain/                    # 地形产品 (DEM/DOM) 和质量栅格
│   ├── DemDomTypes.h           # DEM/DOM 类型
│   ├── DemGridAggregator.h/cpp # mean/median/NMAD/P80/count/confidence/error weighted 聚合
│   ├── DemMosaic.h/cpp         # 多 tile DEM mosaic 与按质量融合
│   ├── TerrainProductManifest.h/cpp # DEM/DOM/error/count/confidence/coverage 产品记录
│   ├── DemGenerator.h/cpp      # DEM 生成
│   ├── DemGeneratorFromDepth.cpp  # 从深度图生成 DEM
│   ├── DomGenerator.h/cpp      # DOM 正射影像生成
│   ├── DemDomIO.h/cpp          # DEM/DOM 和 dem_error/count/confidence/coverage I/O
│   ├── TerrainPipeline.h/cpp   # 地形流水线 (主入口)
│   ├── projection/
│   │   └── AsteroidProjection.h/cpp  # 小行星投影
│   └── tests/ (5 个测试)
│
├── qc/                         # 重建质量检查和外部参考验证
│   ├── ReconstructionQualityReport.h/cpp # 注册影像、track、重投影、MVS/DEM 覆盖率、GCP/检查点/比例尺报告
│   ├── SurveyControlImport.h/cpp # GCP/检查点/比例尺 CSV 导入为 survey_control metadata
│   ├── SurveyControlReport.h/cpp # GCP/检查点/比例尺 metadata 统计和残差状态汇总
│   ├── PointCloudAlignment.h/cpp # 点云完整 Sim3 / 最近邻 ICP 配准与 beg/end error CSV
│   ├── ModelMeshRenderer.h/cpp # CPU tile 并行 z-buffer，将 PLY 网格投影到 MVS 实际相机
│   ├── ModelImageMetrics.h/cpp # 轮廓、覆盖、边缘、SSIM/PSNR 影像空间指标
│   ├── ModelGeometryComparator.h/cpp # 连通分量与参考点云双向最近邻 A+C 几何验收
│   ├── ModelImageQualityEvaluator.h/cpp # GUI/CLI 可复用的统一门控、诊断图和 JSON/CSV 报告
│   └── DemDifference.h/cpp     # DEM 差分、绝对差分和统计报告
│
├── aerial_triangulation/       # 对齐照片式空中三角测量，职责对应 Metashape Align Photos
│   ├── model/                  # GUI/CLI 共用 Options、ResolvedConfig、Result DTO
│   ├── workflow/               # 唯一入口与正式 Pipeline
│   ├── preparation/            # MatchPhotosTask 适配、缓存编目和前置检查
│   ├── reconstruction/         # 单次 SfM、标记点先验、相机内参清洗、候选对与图诊断
│   ├── search/                 # 无相机焦距候选排序和资源策略
│   ├── reporting/              # 稀疏点云、质量元数据和结果记录
│   └── CMakeLists.txt          # 独立 aerial_triangulation target
│
└── feature_match/lightglue/
    └── LightGlueFeatureBudget.h  # LightGlue/SIFT 显存感知关键点预算工具
```

`AerialTriangulationWorkflow` 是“空中三角测量/对齐照片”的唯一用户级入口。重置当前对齐时，workflow
先用 `MatchPhotosTask` 重新提取 SIFT、执行 LightGlue 匹配并整理多视连接点，再把同一组
`assetsDir/featureDir/matchDir` 和持久化连接点图交给 `AerialTriangulationPipeline`。GUI 与
`aerial_triangulation_cli` 不再各自实现连接点补齐逻辑，也不允许 SfM 回退读取另一套项目缓存。
合法的 V2 零匹配 sidecar 作为已确认负缓存消费，不重复扫描或生成。

无相机 `IncrementalSfm` 会在构建对应关系索引前再次按用户的连接点限制执行轨迹级筛选，确保
SfM、BA 和创建连接点阶段使用相同的每影像限额语义。焦距粗筛首先保证注册覆盖，再按多视比例、
三角交会角、空间覆盖和 RMS 选择正式重放候选；最终质量报告对两视轨迹比例执行 0.70 advisory / 0.85 blocking 门控。

`sfm/ReferenceTerrainPrior.h/cpp` 把参考 DEM 或 LiDAR 局部高度面接入 BA soft prior。参考地形默认作为软约束参与诊断，
不把已知外参硬固定；BA 报告应记录 pose prior / terrain prior 优化前后的残差。

`bundle_adjust` 的 `native_cuda` 后端已接入统一 BA 接口、Auto 选择和质量门控。当前首期实现把有效
Camera/BATrack 观测扁平化为 CUDA 工作集，在固定相机投影下优化三维点块，并把 setup/solve/total、
活动相机/track/观测数、接受步和线性残差写回报告。相机 Schur/PCG 更新尚未作为已完成能力发布，
因此文档和 UI 只把它描述为首期 native CUDA BA 加速路径。

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
├── CMakeLists.txt              # GUI 构建 (链接所有模块)
│
├── main_window/                # 主窗口层
│   ├── MainWindow.h/cpp        # QMainWindow 派生, 顶层 UI 编排
│   ├── ProjectUiHydrator.h/cpp # 分阶段刷新项目界面，并通过代次号丢弃过期请求
│   ├── MenuWorkflowController.h/cpp       # "工作流程" 菜单业务控制器
│   ├── ReconstructionWorkflowController.h/cpp  # "重建" 菜单业务控制器
│   └── WorkspacePanelController.h/cpp     # Dock/工具栏可见性、菜单动作与项目状态统一管理
│
├── menu/
│   ├── MainMenu.h/cpp          # 菜单栏/工具栏动作编排；按工作区模式切换三维与影像专属工具组
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
├── dialogs/                    # 对话框 (31 个)
│   ├── FeatureMatchingDialog.h/cpp          # 特征匹配参数
│   ├── FeatureExtractionDialog.h/cpp/ui     # 多算法特征提取参数
│   ├── DenseMatchDialog.h/cpp/Ui.cpp        # 密集匹配参数 (新增)
│   ├── DepthMapEstimateDialog.h/cpp         # 深度图估计参数
│   ├── DepthFusionDialog.h/cpp              # 深度图融合参数
│   ├── DenseCloudDialog.h/cpp               # 密集点云参数
│   ├── DenseCloudRefineDialog.h/cpp         # 密集点云后处理
│   ├── AerialTriangulationDialog.h/cpp      # 空中三角测量
│   ├── BundleAdjustDialog.h/cpp             # 光束法平差参数
│   ├── TriangulationDialog.h/cpp            # 三角化参数
│   ├── MatchViewerDialog.h/cpp              # 统一连接点查看器 (稀疏+密集 Tab)
│   ├── MatchPairSelectorDialog.h/cpp        # 匹配对选择器
│   ├── ObservationNetworkDialog.h/cpp       # 观测网络参数
│   ├── InitCameraPoseDialog.h/cpp           # 相机位姿初始化
│   ├── SparseCloudPostProcessDialog.h/cpp   # 稀疏点云后处理
│   ├── ForwardIntersectionCheckDialog.h/cpp # 前方交汇检测
│   ├── ForwardIntersectionResultsDialog.h/cpp  # 前方交汇结果
│   ├── CameraModel3DDialog.h/cpp            # Qt RHI/Vulkan 3D 查看；异步加载 PLY 与 OBJ/MTL/面级 UV 纹理
│   ├── CameraConvertDialog.h/cpp            # 外部相机格式转换
│   ├── SurveyControlDialog.h/cpp            # 控制点/检查点/比例尺导入和查看
│   ├── CreateDemDialog.h/cpp                # DEM 生成参数
│   ├── MapProjectDialog.h/cpp               # 地图投影参数
│   ├── GenerateModelDialog.h/cpp            # “工作流程 → 生成模型”唯一入口
│   ├── MeshReconstructionDialog.h/cpp       # 旧网格参数组件（不再直接暴露菜单）
│   ├── TextureMappingDialog.h/cpp           # 旧纹理参数组件（不再直接暴露菜单）
│   ├── ModelExportDialog.h/cpp               # 旧导出参数组件（不再直接暴露菜单）
│   ├── OverlapAnalysisDialog.h/cpp          # 重叠度分析
│   ├── VocabularyOverlapDialog.h/cpp/ui     # 基于特征词汇获取重叠对
│   ├── FeaturePointVisualizationDialog.h/cpp  # 特征点可视化
│   ├── SimplePointCloudDialog.h/cpp         # 简单点云查看
│   ├── StereoProcessingDialog.h/cpp         # 立体重建
│   ├── MVSProgressDialog.h/cpp              # MVS 进度
│   ├── WorkflowReportDialog.h/cpp           # 工作流程报告
│   └── settings/                            # 对话框设置持久化支持
│
├── widgets/                    # 自定义 Qt 控件 (10 个)
│   ├── CanvasWidget.h/cpp              # 2D 影像/图层渲染画布；整体视图旋转不修改影像或摄影测量坐标
│   ├── ImageViewWidget.h/cpp           # 2D 影像缩放/平移控件
│   ├── DualImageViewer.h/cpp           # 双图并列查看器 (左右影像 + 匹配线)
│   ├── MatchLineOverlay.h/cpp          # 匹配线叠加层 (稀疏 → 连线)
│   ├── DisparityHeatmapOverlay.h/cpp   # 视差热力图叠加层 (密集 → 热力图/新增)
│   ├── DataTreeWidget.h/cpp            # 项目数据树 (左侧面板)
│   ├── ReferencePanelWidget.h/cpp      # 参考信息面板
│   ├── ObservationNetworkView.h/cpp    # 观测网络可视化
│   └── WorkspaceCenterWidget.h/cpp     # 工作区布局管理及模型/影像/对比/观测网络模式通知
│
├── project/                    # 项目管理层
│   ├── data/
│   │   ├── ProjectData.h/cpp    # 项目数据入口：core/results 分域、归档与临时缓存持久化
│   │   └── ProjectFilesManager.h/cpp  # project_files.json / project_results.json 内存模型
│   ├── archive/
│   │   └── PlascanArchive.h/cpp # ZIP 归档封装
│   ├── manager/
│   │   ├── ProjectManager.h/cpp # 项目管理器 (核心协调器)
│   │   ├── ProjectReconstructionManager.h/cpp       # 重建任务管理
│   │   ├── ProjectSparseReconstructionManager.h/cpp  # 稀疏重建管理
│   │   ├── ProjectDenseReconstructionManager.h/cpp   # 密集重建管理
│   │   ├── ProjectModelGenerationWorkflow.h/cpp      # 生成模型编排：自动深度估计、融合、网格
│   │   ├── ProjectModelManager.h/cpp                 # 模型管理
│   │   ├── ProjectTerrainProductsManager.h/cpp       # 地形产品管理
│   │   ├── ProjectCameraSetupManager.h/cpp           # 相机设置管理
│   │   ├── ProjectTaskDispatcher.h/cpp               # 任务调度器
│   │   └── ProjectUiCommands.h/cpp                   # UI 命令
│   ├── services/
│   │   ├── BundleAdjustService.h/cpp                 # BA 服务
│   │   ├── ProjectBaInputBuilder.h/cpp               # BA 输入构建
│   │   ├── ProjectCameraImportService.h/cpp          # 相机导入
│   │   ├── ProjectTriangulationService.h/cpp         # 三角化服务
│   │   ├── ProjectResourceCleanupService.h/cpp       # 通用资源清理
│   │   └── ProjectTiePointResultService.h/cpp        # 单一当前连接点、覆盖清理与真实删除
│   └── support/                 # 支持/辅助类
│       ├── ProjectSupportUtils.h/cpp               # 通用工具
│       ├── ProjectBundleAdjustExecution.h/cpp       # BA 执行
│       ├── ProjectBundleAdjustWorkflow.h/cpp        # BA 工作流
│       ├── ProjectCameraInitialization.h/cpp        # 相机初始化
│       ├── ProjectDenseWorkflowConfig.h/cpp         # 密集工作流配置
│       ├── ProjectModelWorkflowPolicy.h/cpp         # 模型源判定、输入签名校验、深度批次复用与质量参数映射
│       ├── ProjectMetadataOperations.h/cpp          # 元数据操作
│       ├── ProjectResultRecords.h/cpp               # 结果记录
│       ├── ProjectSfmWorkflow.h/cpp                 # SfM 工作流
│       ├── ProjectSparseWorkflow.h/cpp              # 稀疏工作流
│       ├── ProjectSurveyControl.h/cpp               # GCP/检查点/比例尺 CSV 导入和项目 metadata 持久化
│       ├── ProjectWorkflowUtils.h/cpp               # 工作流工具
│       └── ProjectWorkflowReports.h/cpp             # 工作流报告
│
├── tasks/                      # 异步任务执行器
│   ├── FeatureExtractionRunner.h/cpp  # 特征提取异步执行
│   └── GuiTaskRunner.h         # GUI 后台任务生命周期守护：runGuarded/postGuarded
│
├── views/
│   ├── LayerRenderer.h/cpp             # 图层渲染器
│   ├── LayerOverlayItems.h/cpp          # 批量特征点、残差向量与匹配覆盖层
│   ├── LayerFeatureLoader.h/cpp         # 特征文件解析与关键点加载
│   └── FeatureResidualLoader.h/cpp      # 按当前影像异步筛选真实重投影残差
│
├── config/                     # 配置管理
│   ├── AppConfigManager.h/cpp          # 应用配置
│   ├── ImageViewRotationSettings.h/cpp # 项目级、按影像路径索引的查看旋转角度
│   ├── ProjectConfigManager.h/cpp      # 项目配置
│   ├── ProjectUiConfigManager.h/cpp    # UI 配置
│   ├── ProjectWorkflowConfigManager.h/cpp  # 工作流配置
│   ├── JsonMergeUtil.h/cpp             # JSON 合并工具
│   └── settings/
│       ├── GlobalSettings.h/cpp        # 全局设置
│       ├── DialogSettingStore.h/cpp    # 对话框设置记忆化存储
│       ├── DialogSettingKeys.h         # 各对话框设置键名
│       ├── WindowStateManager.h/cpp    # 窗口状态持久化
│       ├── FileDialogStateManager.h/cpp # 文件对话框状态
│       ├── RecentProjectsManager.h/cpp # 最近项目管理
│       └── ProjectDialogJsonSettingBase.h/cpp  # JSON 设置基类
│
├── panels/
│   └── LogPanel.h/cpp          # 日志面板 (QPlainTextEdit)
│
├── log/                        # GUI 层日志 (使用 #include 快捷方式)
├── cmake/                      # GUI 构建配置
│   ├── GuiSources.cmake        # 源文件清单 (所有 .cpp/.h)
│   ├── GuiCoreLinking.cmake    # 核心库条件链接
│   ├── GuiInstall.cmake        # 安装规则
│   └── cmake/                  # Qt 宏
├── compat/
│   └── QtTorchMacroGuard.h     # Qt/Torch 宏兼容
└── packaging/                  # 打包配置
```

### 菜单结构

```
项目    视图    工作流程          重建                      工具              帮助
├新建   ├放大  ├添加照片/文件夹   ├稀疏重建                ├重叠度获取       └关于
├打开   ├缩小  ├空中三角测量     │├特征点提取              ├前方交汇精度检验
├保存   ├重置  ├创建密集点云     │├获取重叠对...           │├检测交汇
├最近   ├操控球├生成模型         │├创建连接点              │└查看结果
│打开   ├特征点├创建 DEM         │├构建观测网络...         ├手动点云剔除
├导出   │可视化├生成正射影像     │├初始化相机位姿...       ├连接点查看
├最小化 └窗口                    │├生成初始稀疏点云...     ├相机格式转换...
└退出                            │├光束法平差优化...       └查看工作流程报告
                                 │└稀疏点云后处理...
                                 ├密集重建
                                 │├密集匹配...        ← 新增
                                 │├深度图估计...
                                 │├深度图融合...
                                 │└密集点云后处理...
                                 └（模型生成统一由“工作流程 → 生成模型”进入）
```

模型生成只有一个用户入口。旧网格重建、纹理映射和模型导出对话框仍作为内部兼容组件保留，但不再
出现在“重建”菜单，避免两套入口产生不同设置和结果。

### 数据流 (稀疏重建 → 密集重建 → 模型)

```
影像导入
  │
  ├─ 1. 特征提取 (SuperPoint/DISK/ALIKED/SIFT/...) → .sp/.dsk/.alk/... 文件
  ├─ 2. 特征匹配 (SuperGlue/LightGlue/BF/FLANN) → .match 文件
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

深度图的磁盘 manifest 哈希覆盖估计参数、影像路径/大小/修改时间、相机内外参、
匹配对质量约束和稀疏点云内容。项目结果中的深度批次还记录当前影像、相机与正式空三结果的
输入签名；重新平差、修改相机或切换空三结果后，旧深度图不会直接参与融合或模型生成。
深度结果另带 `algorithm_revision`；影响几何质量的生产算法升级会递增该值，生成模型工作流仅透明
复用当前修订版的完整批次，旧批次保留在磁盘并先触发当前多视深度重算。
可复用的密集点云必须与深度批次的目录、数量、配置哈希和输入签名一致，并通过 PLY 头与
非零顶点数检查。

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
以及输出过滤、多视一致性和最终后处理前后的有效数。质量门按实际 source view 数选择一致性策略；
只有 1–2 个源视图时不再错误套用 16 帧工程规模的严格确认门槛。
蒙版内 0.80 覆盖门槛只作用于 project/content 约束蒙版；`full_image` 航测帧沿用边缘/内部帧的
场景化一致性阈值，不按整幅影像覆盖率误降级。

`depth_tsdf` 直接消费 accepted 深度帧，不经过密集点云；`validation_only` 只参与诊断，不进入默认
融合。强单次观测有独立权重门槛。环拍物体工作区只有在至少五个参考相机都将采样判为支持掩膜外时，
才把该证据作为低权重自由空间；单视图不能雕刻模型，航测和无法识别场景的旧工作区默认仍只忽略
掩膜外采样。TSDF 默认截断带为 7.5 体素；普通插值仅填边界不超过 48 条边且物理直径不超过 10 体素的小闭环，并在 JSON 中记录
单/多视支持、拒绝原因、分量面数/包围盒及补洞前后边界数。超高质量档还会剥离两轮至少含两条开放边
且带弱相机支持顶点的终端悬挂三角形，以及只有一条开放边但三个顶点均为弱支持的薄片，并执行两轮
限位边界平滑；候选面和实际移除面数会单独记录。PLY
目标面数优先使用保开放边、锐边、link condition 和面翻转检查的 QEM；若碎边使 QEM 停在目标两倍以上，
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
JSON 中记录 `voxel_fallback_*` 统计，便于区分 QEM 正常达标与后备减面。
Ultra 384 且目标面数不超过 12 万的路径在所有简化完成后，还会重新检测最终开放边界环。只有不含
多视图剪影保护顶点、边数不超过 128 且物理直径不超过 32 体素的内部闭环才允许补洞；若新增面超过
10%、非流形/悬挂边界增长、开放边没有下降或瘦长三角形比例超过 5%，整次最终补洞会自动回退。
`final_hole_fill_*` 记录保护顶点/孔数量、修复孔与新增面、前后开放边和三角形质量。没有任何可靠
多视图剪影保护顶点时不执行该阶段，避免把真实大开口误封。
三角形质量优化后还会执行一次残余微孔收口，只处理不含剪影保护顶点、边数不超过 16 且直径不超过
4 体素的小闭环；开放边未下降、非流形/悬挂边增长、面数增长超过 5% 或瘦长面恶化时整次回退。
`residual_micro_hole_fill_*` 单独记录尝试、接受、保护/修复孔数量及前后拓扑质量。
Ultra 384 的低面数输出（目标面数不超过 12 万）会在等值面提取前启用受约束局部表面片支持：候选
必须满足独立的观测权重门、至少两个几何来源、逆深度离散度、自由空间比例、核心邻域来源重叠和
法向一致性。该路径用于在强减面/体素后备前补足局部 TSDF 支持；高面数 QEM 默认关闭，避免在不触发
后备重采样的网格上增加裂边或扰动纹理。`tsdfSurfacePatchSupport` 和
`tsdfMinimumSurfacePatchObservationWeight` 可显式覆盖自动策略，JSON 会记录实际阈值和各拒绝原因。
同一低面数路径仍以 2 像素深度有效边界腐蚀抑制外沿噪声，但会恢复其中第一圈满足几何支持至少 4、
逆深度相对离散度不超过 0.01 且仍位于支持掩膜内的像素。高面数路径默认不启用该恢复，配置和恢复
像素数分别写入 `effective_*boundary_recovery*` 与 `boundary_recovered_depth_valid_pixel_count`。
顶点色经过网格 z-buffer、深度、
视角及颜色离群检查；OBJ 纹理使用原始相机影像的逐面投影 UV 图集，不再把顶点色作全局平面烘焙。
逐面相机选择检查三顶点、三边中点和质心的支持掩膜，严格路径同时检查深度一致性；无可靠视图的面
使用安全回退颜色，不能采样照片黑背景。`camera_projected_atlas_v3` 还会统计共享顶点邻域的相机票数，
在候选视图仍通过七点掩膜/深度检查且评分不低于原视图 65% 时，抑制孤立面的相机跳变；报告记录
`texture_coherence_adjusted_face_count`，便于区分纹理接缝修复与几何变化。
GUI 的生成模型入口默认请求 OBJ，同时始终保留 `model_from_mesh.ply` 作为几何/兼容回退；项目记录保存
`model_obj`、`model_mtl`、`texture_image`/`texture_png` 和最终显示路径。工作区选择模型时优先异步加载
OBJ，在后台解析 MTL 与纹理图，并在 Vulkan 网格管线中按 `faceTextureIndices` 展开每个面角的 UV；
`ObjRenderPreparation` 同时在后台生成可直接上传的交错顶点缓冲，GUI 线程只负责 GPU 上传。OBJ 读取器
使用连续文件缓冲、`from_chars` 和扁平三角形数组，避免逐行字符串流和嵌套面数组；相机纹理导出按
“相机 + 空间顶点”复用 UV，只在主视图接缝处拆分纹理坐标。MTL、纹理或完整 UV 缺失时回退到顶点
颜色，不在 GUI 主线程执行文件解析、图像读取或百万面三角形展开。照片纹理按原始颜色显示，不重复
叠加网格漫反射；场景半径下限随模型尺度变化，紧凑模型加载后无需反复手动放大。

最终深度证据除几何支持数外，还保存来源相机 bit mask、逆深度均值/相对离散度和跨视图修补掩膜；
旧清单缺少这些字段时按空证据读取。模型质量报告对每个视图保存双向边缘距离、P90 长尾位置及其
来源数、逆深度离散度和缺失阶段，用于区分深度缺失、TSDF 未成面和模型新增毛边。

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
├── workflows/                # GUI“工作流程”菜单入口；Options/Runner/Progress/Report 分责；tests/ 就近维护
├── quality/                  # 模型影像质量验收；tests/ 就近维护
├── common/                   # 公共路径/token/控制台/JSON/输出策略/摄影测量列表基础设施
│   └── tests/                # 公共解析测试与 CliTestSupport
└── third_party/              # CLI11 单头文件依赖
```

各领域目录维护自己的 `CMakeLists.txt` 和直接依赖；顶层不再集中罗列所有工具依赖。
`plascan_cli_support` 由所有入口共享路径、token、UTF-8 控制台、JSON 和输出目录策略；
`plascan_cli_photogrammetry_common` 再提供摄影测量清单、相机 JSON 和蒙版索引能力。
`workflows/` 对齐 GUI 的空中三角测量、三维重建、生成模型和 DEM/正射完整流水线；菜单中的项目
输入操作通过 CLI 的输入清单与项目参数表达，不复制项目导入 UI。一键重建的深度融合、密集点云
细化与点云产物写出由 `core/mvs` 服务实现，入口不保留算法副本。
CLI 测试同样由各领域目录注册并放在对应 `tests/` 下，顶层 `tests/` 不再维护 CLI 聚合测试目标。

**统一约定**：
- `--help` / `-h` — 打印参数说明
- `--config <file>` — JSON 配置文件（可与命令行参数合并，命令行优先）
- `-V` / `--verbose` — 详细诊断日志
- 退出码: 0=成功, 1=参数错误, 2=I/O 错误, 3=算法错误
- 进度/错误信息 → stderr，结果信息 → stdout
- `three_d_reconstruction_cli` 支持 `--stop-after-sfm`、`--skip-mvs`、`--skip-mesh` 分阶段运行，用于大数据 benchmark 和问题定位。
- `scripts/run_photogrammetry_benchmarks.py` 扫描 `prepared/plascan/image_camera.lis`，批量调用三维重建 CLI 并汇总 JSON。

**标准摄影测量流程与 CLI 覆盖**:

```
阶段 1: 稀疏重建 (GUI 完成)
  ├─ 特征提取 (SuperPoint/DISK/ALIKED/...) → .sp/.dsk/.alk/... 文件
  ├─ 特征匹配 (SuperGlue/LightGlue/BF/FLANN) → .match 文件 + .match.json sidecar
  └─ 光束法平差 / 增量SfM           → 精化相机 + 稀疏点云
     (GUI/CLI 由 AerialTriangulationWorkflow 编排；bundle_adjust_cli 可在已有项目和 match sidecar 上做 headless BA/A-B)

阶段 2: 密集重建 (CLI 可用)
  ├─ feature_extract_cli  特征提取    → .sp/.dsk/.alk 等
  ├─ feature_match_cli    特征匹配    → .match 文件 + .match.json sidecar
  ├─ bundle_adjust_cli    光束法平差  → ba_run_summary.json / A-B 对比 JSON
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
dense_match_cli -L rect_L.tif -R rect_R.tif -o disp.tif --cuda
triangulate_cli -d disp.tif --rect-params rect.xml \
    --camL camL.txt --camR camR.txt -o cloud.ply
```

## 五、已知技术债务

| 问题 | 位置 | 建议 |
|------|------|------|
| 多个 GUI 文件超 1000 行 | `gui/dialogs/`, `gui/main_window/` | 提取 UI 构建逻辑 |
| `mvs/` 和 `dense_match/` 有重复逻辑 | `SubpixelRefiner` 两个版本 | 统一到 `dense_match/` |
| 空三真实数据回归仍需扩大 | `core/aerial_triangulation` | 持续加入环拍、航带、弱纹理和控制点数据集 |
| 构建依赖 4 个系统符号链接 | `/lib64/libm.so.6`, `libnvrtc-builtins.so.13.0` 等 | 见 `CONTEXT.md` 系统依赖 |
| ALIKED 导出脚本依赖 lightglue pip 包 | `scripts/export_disk_aliked.py` | 已内联 pure-PyTorch DCN |
| `export_models.py` DISK/ALIKED 部分废弃 | `scripts/export_models.py` | 移除或更新接口 |

## 六、构建系统

- **根**: `CMakeLists.txt` — `PLASCAN_CONDA_PREFIX` 变量 (可覆盖), CUDA 自动查找
- **依赖**: `cmake/PlascanPackages.cmake` (统一 find_package)
- **Core**: 每个子模块独立 `CMakeLists.txt`, 通过 `plascan_core_add_optional_module()` 注册
- **NVRTC**: RPATH 自动配置 conda/pip CUDA 库路径
- **CUDA**: 全局 `enable_language(CUDA)`, 自动查找 conda nvcc
- **测试**: `-DBUILD_TESTS=ON` → CTest；按改动范围优先跑相关测试，再决定是否跑全量
