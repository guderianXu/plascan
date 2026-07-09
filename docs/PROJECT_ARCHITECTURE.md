# PlaScan 项目架构文档

行星表面摄影测量处理系统。最后更新: 2026-07-07。

## 顶层目录

```
plascan/
├── src/            # 所有源代码
│   ├── common/     # 通用工具库 (日志, 数学, 空间索引)
│   ├── core/       # 核心算法库 (相机, 特征, 匹配, SfM, MVS, LiDAR, 蒙版, 网格, 地形, 密集匹配)
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
├── src/cli/        # 命令行工具 (独立于 GUI 的算法入口)
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
├── log/
│   ├── Logger.h/cpp        # 全局日志单例 (LOG_INFO/LOG_ERROR/LOG_DEBUG 宏)
├── math/
│   ├── Vec.h               # 向量运算模板
│   └── Vec3Ops.h           # 3D 向量特化
├── io/
│   └── PathIO.h/cpp        # UTF-8/本机路径转换、原子文件写入和 OpenCV 图像读写封装
├── model/
│   ├── TorchScriptModelResolver.h/cpp # 模型搜索路径解析（PLASCAN_MODEL_DIR、源码树和安装目录）
│   ├── Sam21ModelCatalog.h/cpp # SAM2.1 checkpoint / TorchScript 文件名和安装状态
│   └── U2NetModelCatalog.h/cpp # U2Net ONNX 文件名和安装状态
├── project/
│   └── ProjectCommonUtils.h # 项目通用工具
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
│   ├── Camera.h/cpp            # 通用相机 (Pinhole + 位姿)
│   ├── CameraFormatConverter.h/cpp # Middlebury/EPFL 等外部相机 -> tsai + image_camera.lis
│   ├── PositiveDepthCameraModel.h/cpp  # 正深度约束相机
│   └── Camera_tests.cpp
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
│   │   ├── TrackBuildStage.h/cpp    # 连接点轨迹阶段边界，委托 tie_points 管理最终多视图 track
│   │   └── GuidedMatchStage.h/cpp   # 引导重匹配阶段占位
│   ├── tie_points/
│   │   └── TiePointTrackManager.h/cpp # 最终多视图连接点 track 构建、筛选和统计摘要
│   └── tests/                       # matchphototask 模块级测试
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
│   ├── common/SfmTypes.h       # SfM 公共类型
│   ├── graph/
│   │   ├── CorrespondenceGraph.h/cpp      # 对应关系图
│   │   └── ObservationNetworkBuilder.h/cpp # 观测网络构建
│   ├── pose/PnpSolver.h/cpp    # PnP 位姿解算
│   ├── triangulation/
│   │   ├── Triangulator.h/cpp  # 基础三角化
│   │   └── InitialSparsePointCloudTriangulator.h/cpp  # 初始稀疏点云
│   ├── reconstruction/SfmReconstruction.h/cpp  # SfM 重建器
│   ├── pipeline/IncrementalSfm.h/cpp  # 增量式 SfM 流水线
│   ├── filtering/
│   │   ├── SfmPointCloudFilter.h/cpp       # 点云过滤
│   │   └── SparsePointCloudProcessor.h/cpp # 稀疏点云后处理
│   ├── ReferenceTerrainPrior.h/cpp # 参考 DEM/LiDAR 局部地形面作为 BA soft prior
│   ├── BaInputBuilder.h/cpp    # BA 输入构建器，合并匹配 tracks 与 Survey Control 约束
│   └── TriangulationService.h/cpp  # 三角化服务
│
├── mvs/                        # Multi-View Stereo：深度图 manifest、source planning、流式融合
│   ├── MvsTypes.h              # MVS 公共类型
│   ├── MvsWorkspaceManifest.h/cpp # 深度帧状态、产物路径、配置 hash 和 source plan
│   ├── MvsSourcePlanner.h/cpp  # shared tracks / 几何内点 / 覆盖率 / baseline 选源
│   ├── PatchMatchCUDA.cu/h     # PatchMatch CUDA 实现
│   ├── PatchMatchNoCUDA.cpp    # PatchMatch CPU 回退
│   ├── DepthMapGenerator.h/cpp # 深度图估计、取消检查、raw depth/confidence/valid mask 写盘
│   ├── DepthMapFusion.h/cpp    # 深度图融合 → 密集点云，支持 manifest source plan 和流式融合
│   ├── DepthFrameUtils.h/cpp   # 深度帧工具
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
│   └── EpipolarRectifier_tests.cpp
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
│   ├── SurfaceReconstructorPostprocess.h/cpp # 网格后处理
│   ├── MeshIO.cpp              # 网格文件 I/O
│   ├── TextureMapper.h/cpp     # 纹理映射
│   ├── ModelWorkflowService.h/cpp  # 模型工作流服务
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
│   ├── PointCloudAlignment.h/cpp # 点云 Sim3 / 最近邻平移配准与 beg/end error CSV
│   └── DemDifference.h/cpp     # DEM 差分、绝对差分和统计报告
│
├── aerial_triangulation/       # 对齐照片式空中三角测量，职责对应 Metashape Align Photos
│   ├── AerialTriangulationWorkflow.h/cpp  # 用户级空三参数解析、预选策略映射、服务调用封装
│   ├── AerialTriangulationService.h/cpp   # 特征/匹配检查、相机注册、BA、空三成果和质量记录
│   ├── GuidedRematchService.h/cpp         # 基于已注册相机的 guided rematching 候选补匹配
│   ├── MatchResultCatalog.h/cpp           # 匹配缓存多算法 variant 编目、兼容性状态与最佳结果选择
│   ├── ReconstructionPrerequisiteReport.h/cpp # 空三前置数据完整性、匹配图连通性和补齐建议
│   ├── SfmPairPlanner.h                   # SfM 匹配候选规划：足迹重叠/空间邻域/序列窗口/手工配对
│   └── SfmMatchDiagnostics.h              # 候选图/实际匹配图连通性诊断
│
└── pipeline/                   # 通用流水线桥接 (GUI 可调用)
    ├── FeatureMatchRunner.h/cpp  # 特征匹配异步执行器
    └── LightGlueFeatureBudget.h  # LightGlue/SIFT 显存感知关键点预算工具
```

`sfm/ReferenceTerrainPrior.h/cpp` 把参考 DEM 或 LiDAR 局部高度面接入 BA soft prior。参考地形默认作为软约束参与诊断，
不把已知外参硬固定；BA 报告应记录 pose prior / terrain prior 优化前后的残差。

`bundle_adjust` 的 `native_cuda` 后端已接入统一 BA 接口、Auto 选择和质量门控。当前首期实现把有效
Camera/BATrack 观测扁平化为 CUDA 工作集，在固定相机投影下优化三维点块，并把 setup/solve/total、
活动相机/track/观测数、接受步和线性残差写回报告。相机 Schur/PCG 更新尚未作为已完成能力发布，
因此文档和 UI 只把它描述为首期 native CUDA BA 加速路径。

`aerial_triangulation/AerialTriangulationService.cpp` 在匹配阶段写出 `assets/reports/matching_quality_report.json` 和
`assets/reports/matching_quality_report.csv`。报告记录候选图/实际匹配图连通性、pair 来源统计、优先级、pending/failed/skipped
状态和失败原因，作为后续 guided rematching、track 优化和 Metashape/ASP/Colmap 对比的基线输入。

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
│   ├── MenuWorkflowController.h/cpp       # "工作流程" 菜单业务控制器
│   └── ReconstructionWorkflowController.h/cpp  # "重建" 菜单业务控制器
│
├── menu/
│   └── MainMenu.h/cpp          # 菜单栏/工具栏构建 (所有 QAction 创建)
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
│   ├── CameraModel3DDialog.h/cpp            # 相机模型 3D 查看 (Qt RHI/Vulkan)
│   ├── CameraConvertDialog.h/cpp            # 外部相机格式转换
│   ├── SurveyControlDialog.h/cpp            # 控制点/检查点/比例尺导入和查看
│   ├── CreateDemDialog.h/cpp                # DEM 生成参数
│   ├── MapProjectDialog.h/cpp               # 地图投影参数
│   ├── ModelGenerationDialog.h/cpp          # 模型生成
│   ├── MeshReconstructionDialog.h/cpp       # 网格重建参数
│   ├── TextureMappingDialog.h/cpp           # 纹理映射参数
│   ├── ModelExportDialog.h/cpp              # 模型导出
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
│   ├── CanvasWidget.h/cpp              # 2D 影像/图层渲染画布 (QGraphicsView)
│   ├── ImageViewWidget.h/cpp           # 2D 影像缩放/平移控件
│   ├── DualImageViewer.h/cpp           # 双图并列查看器 (左右影像 + 匹配线)
│   ├── MatchLineOverlay.h/cpp          # 匹配线叠加层 (稀疏 → 连线)
│   ├── DisparityHeatmapOverlay.h/cpp   # 视差热力图叠加层 (密集 → 热力图/新增)
│   ├── DataTreeWidget.h/cpp            # 项目数据树 (左侧面板)
│   ├── ReferencePanelWidget.h/cpp      # 参考信息面板
│   ├── WindowPanel.h/cpp               # 窗口面板组件
│   ├── ObservationNetworkView.h/cpp    # 观测网络可视化
│   └── WorkspaceCenterWidget.h/cpp     # 工作区布局管理
│
├── project/                    # 项目管理层
│   ├── data/
│   │   ├── ProjectData.h/cpp    # 项目数据入口：core/results 分域、归档与临时缓存持久化
│   │   └── ProjectFilesManager.h/cpp  # project_files.json / project_results.json 内存模型
│   ├── io/
│   │   └── ProjectIO.h/cpp      # 项目目录、临时缓存和产物路径规则
│   ├── archive/
│   │   └── PlascanArchive.h/cpp # ZIP 归档封装
│   ├── manager/
│   │   ├── ProjectManager.h/cpp # 项目管理器 (核心协调器)
│   │   ├── ProjectReconstructionManager.h/cpp       # 重建任务管理
│   │   ├── ProjectSparseReconstructionManager.h/cpp  # 稀疏重建管理
│   │   ├── ProjectDenseReconstructionManager.h/cpp   # 密集重建管理
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
│   │   └── ProjectResourceCleanupService.h/cpp       # 资源清理
│   └── support/                 # 支持/辅助类
│       ├── ProjectSupportUtils.h/cpp               # 通用工具
│       ├── ProjectBundleAdjustExecution.h/cpp       # BA 执行
│       ├── ProjectBundleAdjustWorkflow.h/cpp        # BA 工作流
│       ├── ProjectCameraInitialization.h/cpp        # 相机初始化
│       ├── ProjectDenseWorkflowConfig.h/cpp         # 密集工作流配置
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
│   ├── GuiTaskRunner.h         # GUI 后台任务生命周期守护：runGuarded/postGuarded
│   └── ../core/pipeline/FeatureMatchRunner.h/cpp  # 特征匹配异步执行
│
├── views/
│   └── LayerRenderer.h/cpp     # 图层渲染器
│
├── config/                     # 配置管理
│   ├── AppConfigManager.h/cpp          # 应用配置
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
                                 └模型生成
                                  ├网格重建...
                                  ├纹理映射...
                                  └模型导出...
```

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
  │     ├─ 深度图估计 (PatchMatch)          → 深度图
  │     ├─ 深度图融合                       → 密集点云
  │     └─ 密集点云后处理
  │
  └─ 5. 模型生成 / 地形产品
        ├─ 网格重建 (Poisson/MarchingCubes) → 三角网格
        ├─ 纹理映射                         → 带纹理模型
        ├─ DEM 生成                         → 数字高程模型
        └─ DOM 正射影像生成                 → 正射影像
```

---

## 四、cli/ — 命令行工具

独立于 GUI 的算法入口，用于测试、批处理和脚本编排。

```
cli/
├── CMakeLists.txt            # CLI 统一构建
├── cli_common.h              # 公共基础设施 (参数解析, JSON 配置, 退出码)
├── cli_dense_match.cpp       # 密集匹配 CLI
├── cli_camera_convert.cpp    # 外部相机格式转换 CLI
├── cli_reconstruct_pipeline.cpp # GUI 等价一键重建 / 三维重建 CLI
├── cli_feature_extract.cpp   # 特征提取 CLI (8 种算法, 工厂模式)
├── cli_feature_match.cpp     # 特征匹配 CLI (工厂模式, 自动检测算法)
├── cli_bundle_adjust.cpp      # 光束法平差 CLI (支持 LiDAR 约束和 A/B 对比)
└── tests/                    # CLI 端到端测试脚本
```

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
     (GUI 由 AerialTriangulationService 编排；bundle_adjust_cli 可在已有项目和 match sidecar 上做 headless BA/A-B)

阶段 2: 密集重建 (CLI 可用)
  ├─ feature_extract_cli  特征提取    → .sp/.dsk/.alk 等
  ├─ feature_match_cli    特征匹配    → .match 文件 + .match.json sidecar
  ├─ bundle_adjust_cli    光束法平差  → ba_run_summary.json / A-B 对比 JSON
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
| AerialTriangulationService 耦合到 GUI | `core/aerial_triangulation/AerialTriangulationService.cpp` 依赖 `ProjectIO` | 提取 headless AerialTriangulationService, 启用 sfm_cli |
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
