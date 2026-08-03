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
│   ├── ModelFileResolver.h/cpp # 通用模型搜索路径解析（PLASCAN_MODEL_DIR、源码树和安装目录）
│   └── U2NetModelCatalog.h/cpp # U2Net ONNX 文件名和安装状态
├── project/
│   ├── ProjectIO.h/cpp # 项目目录、临时缓存、资源和产物路径规则
│   ├── ProjectArtifactIO.cpp # 基于规范化影像路径哈希的项目产物寻址
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
│   ├── Camera.h/cpp            # 唯一相机模型：内参、Brown-Conrady 畸变、位姿和正深度归一化
│   ├── CameraBaseline.h/cpp    # 相机中心基线、指定点三角交会角和平均深度/基线比
│   ├── CameraFormatConverter.h/cpp # Middlebury/EPFL 等外部相机 -> tsai + image_camera.lis
│   ├── ProjectCameraIO.h/cpp   # Camera JSON/TSAI 与项目元数据适配
│   └── test/                      # 相机测试与诊断程序
│       ├── Camera_tests.cpp
│       ├── CameraBaseline_tests.cpp
│       ├── CameraFormatConverter_tests.cpp
│       ├── test_tsai_loader.cpp
│       └── test.cpp
│
├── image_matching/             # 唯一局部特征/匹配/持久化模块
│   ├── ImageMatchingAlgorithm.h/cpp # 可扩展算法接口、能力和版本契约
│   ├── ImageMatchingRegistry.h/cpp  # 算法注册入口；sift_lightglue / loma_r
│   ├── FeatureSet.h/cpp        # 任务内关键点与描述子，不持久化
│   ├── ImageMatchTypes.h/cpp   # 观测、邻接变体、置信度、残差和标志位
│   ├── ImageMatchFile.h/cpp    # 逐影像 `.pimatch` v1 唯一二进制读写器
│   ├── ImageMatchRepository.h/cpp # 对称写入、完整指纹键缓存和按影像查询
│   ├── sift/                   # CUDA SIFT 提取
│   ├── lightglue/              # TensorRT LightGlue 固定桶推理与后处理
│   ├── sift_lightglue/         # CUDA SIFT + LightGlue 组合与注册实现
│   ├── loma_r/                 # TensorRT DaD/DeDoDe-G 特征与 LoMa-R 匹配
│   ├── tensorrt/               # 通用 engine 会话、CUDA 缓冲和张量 ABI 校验
│   ├── geometry/               # USAC/MAGSAC 验证及逐匹配像素残差
│   └── tests/                  # 格式往返、损坏校验、注册和几何测试
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
│   │   ├── MatchPhotosTask.h/cpp    # 统一任务入口，完成算法选择、候选对、匹配和轨迹阶段
│   │   ├── MatchPhotosOptions.h     # 自动/快速/高精度/CPU/CUDA 等任务选项
│   │   ├── MatchPhotosContext.h     # 项目路径、输出目录、影像输入、取消和进度上下文
│   │   └── MatchPhotosResult.h      # 阶段报告、逐影像分片记录、像对诊断和错误信息
│   ├── pair_selection/
│   │   ├── PairTypes.h/cpp          # PairCandidate、PairSource、pair key 规范
│   │   ├── PairSelectionPolicy.h/cpp # 自动/全量/序列/手动等候选策略
│   │   └── PairSelector.h/cpp       # 合并手动、全量、序列、相机重叠和词汇召回候选
│   ├── runtime/
│   │   ├── MatchPhotosRuntime.h/cpp # 输出路径、LightGlue engine 查找和配置指纹
│   │   ├── MatchPhotosFeatureCache.h/cpp # 一次任务内的有界 SIFT 特征缓存
│   │   ├── MatchPhotosMaskSupport.h/cpp # 连接点流程蒙版路径解析、关键点/连接点过滤
│   │   └── MatchPhotosParallelism.h/cpp # CUDA 显存预算、LightGlue worker 和几何验证并发解析
│   ├── stages/
│   │   ├── FeatureStage.h/cpp       # CUDA SIFT 提取到任务内存，不生成特征文件
│   │   ├── MatchingStage.h/cpp      # TensorRT LightGlue 匹配并提交逐影像 `.pimatch`
│   │   ├── GeometryVerifyStage.h/cpp # 调用 MatchGeometryVerifier 并填入残差/标志
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
│   ├── BundleAdjustCeresPlanning.h/cpp # Dense/Sparse/Iterative Schur 规划和 CUDA 显存预算
│   ├── BundleAdjustProjection.h # 与 Camera 一致的模板投影模型，供 Ceres AutoDiff 固定相机残差复用
│   ├── BundleAdjustValidation.h/cpp # 输入、标定组和 gauge 校验/规范化
│   ├── BundleAdjustQuality.h/cpp # 跨后端正深度、离群点统计和物方约束质量门控
│   ├── BundleAdjustNativeCuda.h/cpp/.cu # PlaScan 自研 CUDA 后端入口和 GPU 点块求解
│   ├── BundleAdjustNativeCudaWorkset.h/cpp # 将 Camera/BATrack 扁平化为 CUDA 连续工作集
│   ├── BundleAdjustNativeCudaTypes.h / *Kernels.cuh # CUDA 侧数据类型、点块 kernel 和设备函数
│   ├── README.md               # 调用链、后端能力、状态、规范与质量门控
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
│   ├── DepthPoseAlignmentRefiner.h/cpp # 锚定尺度的鲁棒点到平面局部 SE(3) 派生位姿细化
│   ├── DepthPoseRefinementStage.h/cpp # 默认关闭的跨视深度候选采样、安全门与派生相机输出
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
│       ├── test_mvs_depth_pose_alignment.cpp
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
│   ├── TextureMapper.h/cpp     # 纹理配置/结果门面及无相机时的顶点色回退
│   ├── CameraTextureMapper.cpp # camera_projected_atlas_v3 兼容路径与 v4 调度
│   ├── CameraTextureMapperV4.h/cpp # 多视图纹理 v4 阶段编排
│   ├── TextureSourcePreprocessor.cpp # 原图/证据相机、清晰度、曝光和网格邻接准备
│   ├── TextureVisibilityEvaluator.cpp # 七点证据检查、top-K 评分、ICM 与小孤岛合并
│   ├── TextureChartBuilder.cpp # 按相机标签连通域构建投影 chart
│   ├── TextureAtlasPacker.h/cpp # 确定性 MaxRects chart 图集打包
│   ├── TextureAtlasBaker.cpp   # 多视图融合、鬼影过滤、填充、锐化及 OBJ/MTL/PNG 输出
│   ├── MeshColorizer.h/cpp     # 网格遮挡检查、鲁棒多视图顶点着色及孤立色斑清理
│   ├── MeshFaceColorOptimizer.h/cpp # 实验性按面主视图投票与共享顶点一致着色
│   ├── MeshQuadricSimplifier.h/cpp # 目标面数驱动、边界/锐边/翻转约束的 QEM 自适应简化
│   ├── MeshFaceOrientation.h/cpp # 共享边面朝向统一及不可定向冲突面的最小清理
│   ├── OpenMeshSimplifier.h/cpp # OpenMesh 拓扑约束减面、法向翻转保护与限位平滑
│   ├── BoundaryAwareVoxelSimplifier.h/cpp # 多视图剪影保护、内部开放边界可聚类的体素简化后备
│   ├── MeshTopologyQuality.h/cpp # 开放边/非流形/连通分量/三角形长宽比质量门及保拓扑边翻转优化
│   ├── MeshBoundaryAttribution.h/cpp # 最终开放边归因、阶段边界统计和原因着色调试 PLY
│   ├── MeshIsotropicRemesher.h/cpp # 内部高长宽比区域的短边合并、长边拆分与拓扑/法向保护
│   ├── IsoSurfaceTopology.h/cpp # 等值面共享边/共享面键、渐近判别和歧义统计
│   ├── ConsistentIsoSurfaceExtractor.h/cpp # 项目自有共享顶点一致等值面实验提取器
│   ├── Mc33IsoSurfaceExtractor.h/cpp # 可选 MC33 拓扑无歧义等值面适配器
│   ├── DepthMapMeshBuilder.h/cpp # 深度帧 manifest/相机产物加载；缺最终层时按清单安全回退最高可用金字塔层
│   ├── DepthFusionFramePolicy.h/cpp # 环拍视角覆盖度量及防连续视角缺口的帧准入策略
│   ├── DepthMeshCompleteness.h/cpp # 深度观测到最终网格的逐帧召回率与完整性质量门
│   ├── TriangleDistanceIndex.h/cpp # BVH 加速的精确点到三角形距离查询，供网格完整性评估使用
│   ├── DepthRayMetric.h/cpp # camera-Z 深度、欧氏射线距离和世界像素足迹的统一换算
│   ├── DepthTsdfSurfaceBuilder.h/cpp # raw depth/confidence/mask/camera 直接融合 TSDF、提取网格并安全减面
│   ├── DepthTsdfCellSheetRecovery.h/cpp # 按面邻接、跨视图来源与已有表面锚点恢复连续零交叉单元片
│   ├── DepthImplicitFieldRegularizer.h/cpp # 等值面提取前的可见性保护、多尺度隐式场正则化与单体素裂缝恢复
│   ├── DepthVisibilityHistogram.h/cpp # 每个 TSDF 样本 9 字节、含精确零中心的有符号距离直方图及鲁棒统计
│   ├── AdaptiveTsdfOctree.h/cpp # 保留零面细节的 2:1 平衡自适应八叉树与面邻接图
│   ├── SparseTgvSolver.h/cpp # 八叉树稀疏图上的 primal-dual 二阶 TGV 隐式场求解器
│   ├── VisibilityOccupancyTsdfCompletion.h/cpp # 深度/轮廓可见性数据项、图割占据载体与拓扑锁定 TSDF 残差融合
│   ├── VisibilityOccupancyCleanup/HandleRepair/WellComposedRepair.* # 占据体分量、柄和良构拓扑修复
│   ├── VisibilityOccupancyDistanceField/BoundaryExtractor/SurfaceBuilder.* # 闭合载体距离场、边界和表面构造
│   ├── VisibilityOccupancyCarrierSubdivision/Fairer/FieldProjector.* # 保拓扑载体细分、平滑和距离场投影
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
│   ├── OrthoGenerationOptions.h/cpp # 正射投影、尺寸、区域、融合和覆盖处理的类型化参数
│   ├── OrthoProjector.h/cpp    # DEM 高程点到项目相机的逐像元反投影与候选融合
│   ├── OrthoProjectorGrid.cpp  # DEM 边界裁剪、X/Y 像元、最大尺寸与像素预算规划
│   ├── OrthoProjectorSupport.cpp # 影像/蒙版加载、颜色校正、锐度、重影和小孔洞处理
│   ├── OrthoProjectorInternal.h # 正射投影内部帧与颜色候选结构
│   ├── DemDomIO.h/cpp          # DEM 元数据/栅格、RGB+覆盖 Alpha GeoTIFF 和质量栅格 I/O
│   ├── TerrainPipeline.h/cpp   # 地形流水线 (主入口)
│   ├── projection/
│   │   └── AsteroidProjection.h/cpp  # 小行星投影
│   └── tests/ (6 个测试)
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
│   ├── ProcessingBaselineManager.h/cpp # 总输入/分阶段快照指纹、参考网格指标和跨版本质量门
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
└── image_matching/lightglue/
    └── LightGlueFeatureBudget.h  # LightGlue/SIFT 显存感知关键点预算工具
```

`AerialTriangulationWorkflow` 是“空中三角测量/对齐照片”的唯一用户级入口。重置当前对齐只清除
SfM 位姿和稀疏重建状态；默认仍复用影像身份、算法版本、配置指纹和模型指纹均匹配的 `.pimatch`
变体及连接点。只有用户取消“重用现有匹配”或缓存缺失/不兼容时，workflow 才调用
`MatchPhotosTask` 按工作流设置执行 SIFT + LightGlue 或 LoMa-R，并整理多视连接点。GUI 与
`aerial_triangulation_cli` 不再各自实现连接点补齐逻辑，也不允许 SfM 回退读取旧成对缓存。

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

`sfm/pipeline/SfmBundleAdjustCoordinator` 是空三调用 `bundle_adjust` 的正式入口。它构造局部或全局
相机/轨迹问题、固定边界相机、转发进度，并只回写 `solutionUsable=true` 的结果。无绝对控制时，
`SimilarityGaugeNormalizer` 在全局 BA 后恢复确定性锚点和初始基线尺度；有控制点或比例尺时由绝对约束接管规范。

`bundle_adjust` 的 `native_cuda` 后端已接入统一 BA 接口和质量门控。当前实现把有效 Camera/BATrack
观测扁平化为 CUDA 工作集，在固定相机投影下优化三维点块；能力表明确标记它不更新相机和共享焦距，
因此需要联合相机 BA 时 Auto 不会选择该后端。Ceres CPU/CUDA 在同一非线性问题中联合优化相机、三维点
和分组共享焦距；CPU 按问题规模选择 Dense/Sparse/Iterative Schur，CUDA dense 求解前执行显存预算。
Legacy CPU 保留小型固定焦距问题。所有后端统一返回状态、可用性、取消、回退原因和耗时，
正常 Auto 路径只有未通过状态或质量门控时才回退，不再无条件重复完整 Legacy BA。

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
│   ├── ProjectLifecyclePresenter.h/cpp # 项目打开/保存进度、脏状态标题及关闭后保存
│   ├── ProjectTaskStatusController.h/cpp # 状态栏任务控件、取消路由及概览快照
│   ├── MenuWorkflowController.h/cpp       # "工作流程" 菜单业务控制器
│   ├── ReconstructionWorkflowController.h/cpp  # 生成模型/纹理工作流程对话框协调
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
├── dialogs/                    # 按业务域组织的参数与结果对话框
│   ├── application/            # 关于、工作流程报告等应用级对话框
│   ├── camera/                 # 相机查看/转换、前方交汇、测量控制
│   ├── image/                  # 蒙版等单影像处理
│   ├── reconstruction/         # 空三、模型、纹理、DEM/正射工作流程
│   │   ├── MapProjectDialog.h/cpp         # 正射对话框生命周期、运行进度与取消
│   │   ├── MapProjectDialogLayout.cpp     # 投影、参数、区域、输出和进度分组布局
│   │   ├── MapProjectDialogSettings.cpp   # 稳定 token、设置往返、输入校验与控件联动
│   │   └── MapProjectDialogEstimate.cpp   # DEM 元数据读取、真实像元/范围和内存估算
│   ├── tie_points/             # 连接点创建/清理/查看与重叠分析
│   ├── shared/                 # 对话框共享样式与布局辅助
│   └── README.md               # 分类规则与新增对话框约束
│
├── widgets/                    # 自定义 Qt 控件 (10 个)
│   ├── CanvasWidget.h/cpp              # 2D 影像/图层渲染画布；整体视图旋转不修改影像或摄影测量坐标
│   ├── ImageViewWidget.h/cpp           # 2D 影像缩放/平移控件
│   ├── DualImageViewer.h/cpp           # 双图并列查看器 (左右影像 + 匹配线)
│   ├── MatchLineOverlay.h/cpp          # 匹配线叠加层 (稀疏 → 连线)
│   ├── DisparityHeatmapOverlay.h/cpp   # 视差热力图叠加层 (密集 → 热力图/新增)
│   ├── DataTreeWidget.h/cpp            # Metashape 式工作区汇总 → Chunk → 资源分组树
│   ├── WorkspaceSectionIcons.h/cpp     # 工作区、Chunk、影像及成果类型语义图标
│   ├── ReferencePanelWidget.h/cpp      # 参考信息面板
│   ├── ObservationNetworkView.h/cpp    # 观测网络可视化
│   └── WorkspaceCenterWidget.h/cpp     # 工作区布局管理及模型/影像/对比/观测网络模式通知
│
├── project/                    # 项目管理层
│   ├── data/
│   │   ├── ProjectData.h/cpp    # 项目数据入口：core/results 分域、归档与临时缓存持久化
│   │   └── ProjectFilesManager.h/cpp  # Chunk doc.json 的 core/results 内存模型
│   ├── archive/
│   │   └── *.h # 指向 src/common/project 存储层的 GUI 兼容头
│   ├── manager/
│   │   ├── ProjectManager.h/cpp # 项目管理器 (核心协调器)
│   │   ├── ProjectLifecycleController.h/cpp          # 创建、异步打开/结果加载、保存与关闭
│   │   ├── ProjectMaskWorkflowController.h/cpp       # 蒙版对话框、异步生成、取消及结果登记
│   │   ├── ProjectSparseReconstructionManager.h/cpp  # 稀疏重建管理
│   │   ├── ProjectPointCloudWorkflowController.h/cpp # 点云工作流协调：深度估计/复用、流式融合与结果登记
│   │   ├── ProjectModelManager.h/cpp                 # 从已有点云/深度图生成模型，不隐式启动稠密流程
│   │   ├── ProjectTerrainProductsManager.h/cpp       # 从已有点云生成 DEM，以及正射后台任务与结果登记
│   │   ├── ProjectCameraSetupManager.h/cpp           # 相机设置管理
│   │   └── ProjectUiCommands.h/cpp                   # UI 命令
│   ├── services/
│   │   ├── BundleAdjustService.h/cpp                 # BA 服务
│   │   ├── ProjectBaInputBuilder.h/cpp               # BA 输入构建
│   │   ├── ProjectCameraImportService.h/cpp          # 相机导入
│   │   ├── ProjectTriangulationService.h/cpp         # 三角化服务
│   │   ├── ProjectResourceCleanupService.h           # 旧包含路径兼容层；实现位于 core/project_workflows
│   │   └── ProjectTiePointResultService.h/cpp        # 单一当前连接点、覆盖清理与真实删除
│   └── support/                 # 支持/辅助类
│       ├── ProjectSupportUtils.h/cpp               # 通用工具
│       ├── ProjectBundleAdjustExecution.h/cpp       # BA 执行
│       ├── ProjectBundleAdjustWorkflow.h/cpp        # BA 工作流
│       ├── ProjectCameraInitialization.h/cpp        # 相机初始化
│       ├── ProjectDenseWorkflowConfig.h             # 旧包含路径兼容层；实现已迁移到 core/project_workflows
│       ├── ProjectReferenceDatasets.h               # 旧包含路径兼容层；参考数据业务已迁移到 core/project_workflows
│       ├── ProjectModelWorkflowPolicy.h/cpp         # 模型线程预算、输入签名及深度批次完整性/代次兼容校验
│       ├── ProjectSessionContext.h                  # 异步写回会话身份（项目、Chunk、generation）
│       ├── ProjectTerrainRequests.h                 # DEM 类型化请求及边界校验
│       ├── ProjectMetadataOperations.h/cpp          # 元数据操作
│       ├── ProjectResultRecords.h/cpp               # 结果记录
│       ├── ProjectSfmWorkflow.h/cpp                 # SfM 工作流
│       ├── ProjectSparseWorkflow.h/cpp              # 稀疏工作流
│       ├── ProjectSurveyControl.h/cpp               # GCP/检查点/比例尺 CSV 导入和项目 metadata 持久化
│       ├── ProjectWorkflowUtils.h/cpp               # 工作流工具
│       └── ProjectWorkflowReports.h/cpp             # 工作流报告
│
├── tasks/                      # 异步任务执行器
│   └── GuiTaskRunner.h         # GUI 后台任务生命周期守护：runGuarded/postGuarded
│
├── views/
│   ├── LayerRenderer.h/cpp             # 图层渲染器
│   ├── LayerOverlayItems.h/cpp          # 批量特征点、残差向量与匹配覆盖层
│   ├── LayerFeatureLoader.h/cpp         # 特征文件解析与关键点加载
│   ├── FeatureResidualLoader.h/cpp      # 按当前影像异步筛选真实重投影残差
│   ├── CameraSceneViewMath.h/cpp        # 相机平面、视角选择与本地轴数学
│   └── ObjRenderPreparation.h/cpp       # OBJ 显示数据准备
│
├── config/                     # 配置管理
│   ├── AppConfigManager.h/cpp          # 应用配置
│   ├── ImageViewRotationSettings.h/cpp # 项目级、按稳定 image_uuid 索引的查看旋转角度
│   ├── ProjectConfigManager.h/cpp      # Chunk project_config，仅工作流参数
│   ├── ProjectUiConfigManager.h/cpp    # 根 doc.json 的 ui_state 显示状态
│   ├── ProjectWorkflowConfigManager.h/cpp  # 工作流配置
│   └── settings/
│       ├── DialogSettingStore.h/cpp    # 对话框设置记忆化存储
│       ├── DialogSettingKeys.h         # 各对话框设置键名
│       ├── GuiSettingsStore.h           # GUI QSettings 工厂
│       ├── WindowStateManager.h/cpp    # 窗口状态持久化
│       ├── FileDialogStateManager.h/cpp # 文件对话框状态
│       ├── RecentProjectsManager.h/cpp # 最近项目管理
│       └── ProjectDialogJsonSettingBase.h/cpp  # JSON 设置基类
│
├── panels/
│   └── LogPanel.h/cpp          # 日志面板 (QPlainTextEdit)
│
├── platform/
│   └── ProjectFileIntegration.h/cpp # 启动工程参数解析与 Windows 当前用户文件关联
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
和项目 UI 状态。每个 Chunk 由稳定 UUID 标识，并映射到 `1/`、`2/`、`3/` 等数字目录；
目录号由持久化的 `next_chunk_directory` 单调分配，删除后不复用。Chunk 的核心元数据、
结果、配置和资源索引位于其 `chunk.zip/doc.json` 的独立字段。原始影像位于工程级
`.files/shared/images/<sha256>/`，其他大型成果直接位于对应数字目录。共享影像库、
`assets/`、`bundle_adjust/`、`reconstruction/` 和 `reports/` 均在对应流程首次写入时
按需创建，空 Chunk 只包含 `chunk.zip`。GUI 和 CLI 的 BA 运行产物统一写入当前 Chunk
的 `bundle_adjust/<run>/`，不再混入 `assets/`；生效相机参数仍写回 Chunk 文档，
综合报告继续位于 `reports/`。
`ProjectWorkspaceStore` 继续兼容 `plascan:///workspace/...` 逻辑 URI，但只在当前 Chunk
数字目录内解析，不再使用根级 `workspace/`。旧版根级 `workspace/` 分体工程和旧版
单体工程均明确拒绝加载，并保持旧文件不变。归档条目在组合物理路径前执行跨平台名称、
大小写冲突和目标根目录边界校验。
项目配置按应用设置、工作流配置、项目视图状态和运行时缓存四层管理，避免机器路径混入工程。
Chunk 保存将核心、结果、配置和资源索引合并为一次 `doc.json` 更新，并推进 `revision`；
资源索引按引用集合和文件大小/修改时间增量维护。工程打开期间持有
`.files/.plascan.lock`，避免 GUI/CLI 并发覆盖。共享影像只在所有 Chunk 都解除引用后清理。
格式结构、路径安全规则和拒绝策略见
[`docs/project/PLASCAN_PROJECT_FORMAT.md`](project/PLASCAN_PROJECT_FORMAT.md)。

### 菜单结构

```
项目    视图    工作流程          重建                      工具              帮助
├新建   ├放大  ├添加照片/文件夹   ├稀疏重建                ├重叠度获取       └关于
├打开   ├缩小  ├空中三角测量     │├特征点提取              ├前方交汇精度检验
├保存   ├重置  ├创建密集点云     │├获取重叠对...           │├检测交汇
├最近   ├操控球├生成模型         │├创建连接点              │└查看结果
│打开   ├特征点├创建 DEM         │├构建观测网络...         ├手动点云剔除
├导出   │可视化├生成正射影像     │├初始化相机位姿...       ├连接点查看
├最小化 └窗口  ├生成纹理         │├生成初始稀疏点云...     ├相机格式转换...
└退出                            │├光束法平差优化...       └查看工作流程报告
                                 │└稀疏点云后处理...
                                 ├密集重建
                                 │├密集匹配...        ← 新增
                                 │├深度图估计...
                                 │├深度图融合...
                                 │└密集点云后处理...
                                 └（模型生成统一由“工作流程 → 生成模型”进入）
```

模型生成仍由一个统一入口负责几何重建；已经存在模型时，可通过“工作流程 → 生成纹理”独立重建
OBJ/MTL/PNG 纹理产物。旧网格重建和模型导出对话框仍作为内部兼容组件保留，不再出现在“重建”
菜单，避免两套几何入口产生不同设置和结果。

### 数据流 (稀疏重建 → 密集重建 → 模型)

```
影像导入
  │
  ├─ 1. CUDA SIFT + TensorRT LightGlue → 每影像一个 `.pimatch` 分片
  │     └─ SIFT 描述子只存在于任务内存，分片保存像点、匹配、残差和版本指纹
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
  -> MapProjectDialog（参数、DEM 元数据估算、进度与取消）
  -> ProjectManager -> ProjectTaskDispatcher
  -> ProjectTerrainProductsManager -> GuiTaskRunner::runGuarded
  -> ProjectWorkflowUtils::runOrthoProduct
  -> TerrainPipeline -> OrthoGenerationOptions / OrthoProjector
  -> DemDomIO（RGB+覆盖 Alpha GeoTIFF 或 RGBA PNG）
  -> project_results.ortho_results[]
```

对话框从项目读取最新相对 DEM、影像相机和蒙版就绪数，显示 DEM 坐标系、真实 X/Y 像元、
裁剪边界、输出宽高和预计内存。运行期间参数被锁定，`orthoPipelineStarted`、
`orthoPipelineProgressChanged` 和 `orthoPipelineFinished` 把后台状态回传同一对话框；
取消通过共享原子标志传入核心投影循环，切换项目后旧任务不会写回当前项目。

当前生产正射链仅支持 `dem_grid` 地理/本地 DEM 网格投影、`dem` 表面和 `images` 颜色源。
混合模式支持 `mosaic`、`weighted_average`、`first_valid`；尺寸可使用独立 X/Y 像元或
最大边像素，并可与 DEM 范围相交裁剪。颜色校正、锐度权重、鲁棒重影过滤、小孔洞填充和
项目蒙版均进入类型化核心参数。平面投影、圆柱投影和全局接缝线优化尚未实现，GUI 对应控件
明确禁用。若有效 DEM 表面没有任何相机影像覆盖，核心直接失败，不登记全黑成果。有效覆盖
会进入 GeoTIFF/PNG Alpha；当前 `ortho_projector_v1` 尚未建立逐相机地形遮挡深度缓冲，
因此陡峭地形仍需质量复核，不能把 Alpha 当作遮挡正确性的证明。

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
以及输出过滤、多视一致性和最终后处理前后的有效数。质量门按实际 source view 数和过滤档位选择
一致性策略：单源只剔除明确矛盾，多源在轻度/中度/严格过滤下分别要求至少 1/2/3 个独立源确认；
环拍轻度过滤且至少四源时要求两个源确认。双源相对深度阈值为 10%/6%/3%；三源以上的航测
阈值为 3%/1.5%/0.75%，汇聚环拍阈值为 1.25%/0.8%/0.5%，并执行源到参考的往返投影检查。
蒙版内 0.80 覆盖门槛只作用于 project/content 约束蒙版；`full_image` 航测帧沿用边缘/内部帧的
场景化一致性阈值，不按整幅影像覆盖率误降级。

MVS 源规划优先使用从当前存储匹配结果经 USAC/MAGSAC 验证的像对。空匹配文件或少于最少
内点数的匹配只表示验证证据缺失，不能当作已证明失败；剩余源位由共享轨迹几何补足。16 视图及
以上的密集环拍高质量任务最多使用六源，12 视图等稀疏环拍强制封顶为四个近邻源；通用 GUI
质量预设传入的更大候选数不能覆盖这个场景上限，避免把约 90° 的第三环邻居当成逐像素必需确认。
环拍最大三角化角从实际候选角度分布自适应计算，并受四源 70°、六源 90°
安全上限约束。1024 级高质量影像的深度金字塔保留 `4→2→1` 全分辨率末层；大图继续使用配置的
最终降采样以控制显存和运行时间。以上行为写入 source plan、算法修订号和重放报告，旧深度缓存
不会静默复用。

`depth_tsdf` 直接消费深度帧，不经过密集点云。深度产物统一存储物理前向的正 camera-Z；
`DepthRayMetric` 按像素反投影换算离轴欧氏射线距离，并以对称半像素边界射线估计横纵世界像素足迹，
为后续基于像素尺度的跨视图容差和可见性前后偏移提供同一度量。环拍工作区把帧分为主融合帧、低权重
`validation_only` 辅助表面帧和真正拒绝帧：辅助帧只在表面支持带内补充有效深度，不估计包围盒，
也不投票自由空间。帧均值置信度只连续降权，默认不再整帧硬剔除；显式启用硬剔除时仍需通过
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

环拍任意 3D 模型默认使用双分辨率隐式表面路径。72 级规则网格执行可见性占据图割：深度前方提供
空域证据，深度邻域提供表面/实体证据，轮廓只作有界先验；占据体按“柄修复、良构修复、内部气泡
清理”循环到固定点。该规则网格只是低分辨率的内外符号/拓扑先验，不再把单元外壳直接作为最终
网格。完成器把其符号约束与原生高分辨率 TSDF 残差合成，然后由 MC33 在 192/384 级场上插值真实
零交叉；被占据先验恢复的样本允许参与符号变化，避免再被“两个端点都必须有原生深度支持”的门限
切掉。这与 Open3D 在目标 TSDF 分辨率上插值零交叉的几何原则一致，同时保留 PlaScan 的多视可见性
补全。72 级单元边界提取仍保留为显式兼容/诊断开关，但不再是环拍默认值。

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

旧工作区若 manifest 指向的最终深度文件已经缺失，但 `pyramid_levels` 中仍有成功且存在的产物，模型加载
只回退到清单内最高分辨率的可用层，并按栅格比例缩放相机内参；几何计数/来源位掩码用最近邻、连续
逆深度统计用面积采样同步降尺度。没有 manifest 时目录扫描明确忽略 `depth_*_level_*.bin`，避免把中间
产物误识别成无相机的最终帧。

384 级任意 3D 模型且目标不超过 24 万面时，目标面数优先使用 OpenMesh 的 link condition、
法线偏差和翻转约束减面，并执行有绝对位移上限的平滑；进入 OpenMesh 前先统一共享边面朝向，
无法全局定向的极少数冲突面按最小集合移除。OpenMesh 不可用、被显式关闭或候选未通过拓扑门时，
才回退到保开放边、锐边、link condition 和面翻转检查的项目 QEM。若碎边使 QEM 停在目标两倍以上，
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
同输入 A/B 中显著降低面数、开放边和法线粗糙度后，已纳入 384 分辨率且启用 OpenMesh、
目标面数不超过 24 万的
环拍高细节自动策略；用户显式设置 `tsdfAdaptiveTgvRegularization=false` 时仍可关闭。该自动策略同时
启用保守的轮廓带零交叉支持，轮廓候选仍使用 0.45 的绝对 TSDF 门、至少两个几何来源和双单元投票；
0.55、三次边界确认以及全射线后方符号场等失败实验均未进入默认值。
MC33 提取后默认执行“受支持符号变化”过滤：每个保留面所在单元必须同时含有被观测支持的非正值和
正值角点。未观测区域的默认正值不能再与真实负值带形成第二层伪零交叉；被拒绝面数写入
`mc33_rejected_unsupported_cell_face_count`，开关写入
`effective_mc33_require_supported_sign_change`。
同一 OpenMesh 高细节路径仍以 2 像素深度有效边界腐蚀抑制外沿噪声，但会恢复其中第一圈满足几何支持至少 4、
逆深度相对离散度不超过 0.01 且仍位于支持掩膜内的像素。超过 24 万面或显式关闭 OpenMesh
的路径默认不启用该恢复，配置和恢复
像素数分别写入 `effective_*boundary_recovery*` 与 `boundary_recovered_depth_valid_pixel_count`。
顶点色经过网格 z-buffer、深度、
视角及颜色离群检查；OBJ 纹理使用原始相机影像的逐面投影 UV 图集，不再把顶点色作全局平面烘焙。
`camera_projected_atlas_v4` 对三顶点、三边中点和质心执行支持掩膜与深度一致性检查，每个面只保留
前四个候选视角，再以 ICM 相邻面能量和小孤岛合并生成连续主视角标签。相同标签的连通面构成裁剪
chart，经确定性 MaxRects 打包后逐纹素反投影；Natural、加权平均和最佳视角模式分别执行真实的
多视图采样，Natural 可按中值颜色剔除鬼影。无可靠视图的面使用安全回退颜色，不能采样照片黑背景。
输出报告包含严格/宽松映射面、chart/视角数、各类拒绝原因、图集占用率、中位纹素密度、接缝色差
和峰值内存估算；`camera_projected_atlas_v3` 仅作为配置级兼容回退保留。
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
├── workflows/                # GUI“工作流程”菜单入口；Options/Runner/Progress/Report 分责；tests/ 就近维护
├── quality/                  # 模型影像质量验收；tests/ 就近维护
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
`workflows/` 提供空中三角测量、三维重建 CLI、生成模型和 DEM/正射完整流水线；GUI 不再提供旧版
“三维重建”一键对话框，改由空三、密集处理和生成模型入口分阶段执行。菜单中的项目输入操作通过
CLI 的输入清单与项目参数表达，不复制项目导入 UI。一键重建的深度融合、密集点云
细化与点云产物写出由 `core/mvs` 服务实现，入口不保留算法副本。
CLI 测试同样由各领域目录注册并放在对应 `tests/` 下，顶层 `tests/` 不再维护 CLI 聚合测试目标。

**统一约定**：
- `--help` / `-h` — 打印参数说明
- `--config <file>` — JSON 配置文件（可与命令行参数合并，命令行优先）
- `-V` / `--verbose` — 详细诊断日志
- 退出码: 0=成功, 1=参数错误, 2=I/O 错误, 3=算法错误
- 进度/错误信息 → stderr，结果信息 → stdout
- `three_d_reconstruction_cli` 支持 `--stop-after-sfm`、`--skip-mvs`、`--skip-mesh` 分阶段运行，用于大数据 benchmark 和问题定位。
- `scripts/bench/run_photogrammetry_benchmarks.py` 扫描 `prepared/plascan/image_camera.lis`，批量调用三维重建 CLI 并汇总 JSON。

**标准摄影测量流程与 CLI 覆盖**:

```
阶段 1: 稀疏重建 (GUI 完成)
  ├─ CUDA SIFT + TensorRT LightGlue   → 每影像一个 `.pimatch` 二进制分片
  ├─ 多视连接点轨迹                    → latest_tie_points.json
  └─ 光束法平差 / 增量SfM           → 精化相机 + 稀疏点云
     (GUI/CLI 由 AerialTriangulationWorkflow 编排；bundle_adjust_cli 可在已有项目分片上做 headless BA/A-B)

阶段 2: 密集重建 (CLI 可用)
  ├─ feature_match_cli    双影像匹配  → 两个 `.pimatch` 分片
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

## 六、构建系统

- **根**: `CMakeLists.txt` — `PLASCAN_CONDA_PREFIX` 变量 (可覆盖), CUDA 自动查找
- **依赖**: `cmake/PlascanPackages.cmake` (统一 find_package)
- **Core**: 每个子模块独立 `CMakeLists.txt`, 通过 `plascan_core_add_optional_module()` 注册
- **NVRTC**: RPATH 自动配置 conda/pip CUDA 库路径
- **CUDA**: 全局 `enable_language(CUDA)`, 自动查找 conda nvcc
- **测试**: `-DBUILD_TESTS=ON` → CTest；按改动范围优先跑相关测试，再决定是否跑全量
## GUI 模块边界（2026-08）

- `src/common/project/ProjectSessionModel.*`、`ProjectDocumentModel.*` 和三类项目配置管理器负责
  QtCore 项目会话、文档分域与持久化；`src/gui/project/data`、`src/gui/config` 只保留兼容头。
- `src/core/project_workflows` 负责 DEM/正射、稀疏点后处理、点云参数/输入准备、参考数据检查和
  生成资源清理，通过独立 `project_workflows` 目标供 GUI、CLI 和测试复用。
- `ProjectManager` 以门面形式持有项目生命周期、蒙版、点云、稀疏重建、模型、地形产品和相机设置
  控制器。GUI 中不存在“稠密重建管理器”；`ProjectPointCloudWorkflowController` 只协调深度估计与点云融合。
- `MainWindow` 按布局、菜单绑定、项目绑定和 UI 状态拆分实现；项目打开/保存展示由
  `ProjectLifecyclePresenter` 管理，状态栏任务由 `ProjectTaskStatusController` 管理。
  `DataTreeWidget` 按模型、填充、上下文菜单、资源元数据和相机对齐判定拆分实现。
- 正射流程为 `MenuWorkflowController -> ProjectManager -> ProjectTerrainProductsManager ->`
  `project_workflows::runOrthoProduct`，请求在 GUI 边界转换为 `OrthoGenerationRequest`。
