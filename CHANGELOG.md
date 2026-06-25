# PlaScan Changelog

本文件按版本倒序记录用户可感知的主要变更。详细验证记录见 `docs/releases/`。

## Unreleased

### 新增

- 重建质量报告新增 `survey_control` 汇总，支持从项目 metadata 统计控制点、检查点和比例尺数量、启用数量、残差 RMSE、最大残差和质量状态，为后续 GCP/检查点/scale bar 生产闭环打基础。
- 新增 Survey Control CSV 导入核心解析器，支持按 `role/type/kind` 区分控制点、检查点和比例尺，并解析 `id/x/y/z/sigma/enabled/from_id/to_id/measured_m` 等基础字段。
- GUI 项目支持层新增 `ProjectSurveyControl`，可把 Survey Control CSV 导入并持久化到项目 `survey_control` metadata，同时记录 `source_path/imported_at/format`。
- GUI 工具菜单新增“测绘控制...”入口和 `SurveyControlDialog`，可导入 Survey Control CSV，并以控制点、检查点、比例尺三张表查看当前项目控制数据。
- GUI 项目 `report_results` 摘要同步写入控制点、检查点和比例尺数量/RMSE，目录树或报告窗口后续可直接消费这些 metadata。
- 光束法平差新增比例尺/标尺软约束，`BaInputBuilder` 可把 `survey_control.scale_bars` 映射为两条控制点 track 之间的距离约束，并在 BA JSON 中输出 `scale_bar_constraints_summary`。
- Windows CTest 新增 `PlascanTestRuntime.cmake`，为 SuperPoint/DISK/ALIKED 等 Torch 测试自动补齐 LibTorch、CUDA 和 vcpkg 运行时 PATH，避免测试目录缺 DLL 时出现 `0xc0000135`。
- GUI 新增 `GuiTaskRunner`，提供 `runGuarded/postGuarded` 生命周期守护后台任务入口，首批迁移相机 SFM 初始化、空中三角测量预检/SFM 和三维重建空三流程，减少关闭项目、切换工程或窗口销毁后后台回调访问已释放对象的风险。
- `GuiTaskRunner::runGuarded` 在后台任务真正开始前先检查 GUI owner 是否仍然有效，并把完整 DEM 自动流水线入口纳入该 runner，降低关闭项目/窗口后启动长任务的风险。
- DEM 自动流水线在启动 MVS 前记录已有密集点云数量，完成后只扫描本次新增且文件真实存在的 dense cloud 记录，并用共享连接状态统一清理 metadata/MVS 信号，避免误用旧点云或泄漏连接句柄。
- DEM 自动流水线在 MVS 新增 dense cloud 后，会把深度图直接 DEM 和点云回退 DEM 生成放入 `GuiTaskRunner::runGuarded` 后台任务；metadata 回调只做新增结果筛选和轻量请求准备，减少大数据 DEM/DOM IO 与栅格生成卡住 GUI 的风险。
- 手动“从密集点云创建相对 DEM”入口改用 `GuiTaskRunner::runGuarded` 启动 DEM/DOM IO 与栅格生成，避免关闭项目/窗口后 open-coded `QtConcurrent` 仍启动地形产品任务。
- 旧的 stereo/point2dem 相对 DEM 入口也改用 `GuiTaskRunner::runGuarded` 后台执行 DEM/DOM IO 与栅格生成，不再在 Async 接口里直接调用带弹窗的同步 wrapper。
- “生成正射影像”入口改用 `GuiTaskRunner::runGuarded` 后台执行 DOM/ortho IO 与栅格生成，主线程只负责参数检查、结果登记和提示，避免大 DEM/影像生成正射时卡住 GUI。
- 移除 `ProjectTerrainProductsManager` 中已无生产调用的同步弹窗 wrapper，地形产品入口统一使用非 UI 核心函数配合 `GuiTaskRunner` 回到主线程登记结果。
- 光束法平差核心优化进度回调改用 `QPointer<ProjectManager>` 守护，避免后台 BA 迭代中关闭项目或窗口后继续向已销毁的 `ProjectManager` 投递进度事件。
- 深度图估计入口的进度、深度图 artifact 登记和完成回调改用 `QPointer<ProjectDenseReconstructionManager>` 守护，降低 MVS 运行中关闭项目/窗口后的悬挂回调风险。
- 稠密点云生成入口的进度、深度图 artifact、点云保存和完成回调统一复用 `QPointer<ProjectDenseReconstructionManager>`，继续收敛 MVS 长任务关闭/切换工程时的生命周期风险。
- 深度图融合入口改用 `GuiTaskRunner::runGuarded` 启动流式融合 worker，避免关闭项目/窗口后 open-coded `QtConcurrent` 仍启动长时间融合任务。
- 密集点云后处理入口改用 `GuiTaskRunner::runGuarded` 启动大点云加载、离群过滤、体素降采样和法向估计任务，避免 owner 已销毁后继续启动重型点云处理。
- 两视预览三角化和稀疏点云后处理入口改用 `GuiTaskRunner::runGuarded`，移除 `ProjectSparseReconstructionManager` 中 open-coded `QtConcurrent`/`invokeMethod` 回调，降低关闭项目/窗口后的后台回调风险。
- 深度图估计和稠密点云生成的稀疏点云预处理 worker 改用 `QPointer<DepthMapGenerator>`，并把 `setSparseCloud/start` 投递回 generator 所在线程，避免后台预处理完成后访问已释放 generator。
- 网格重建和纹理映射后台任务改用 `QPointer<ProjectModelManager>` 守护进度、完成回调和结果登记，降低模型/纹理长任务运行中关闭窗口后的悬挂回调风险。
- 3D 模型视图的 XYZ/PLY/OBJ 异步加载完成回调改用 `QPointer<CameraSceneWidget>` 守护，降低关闭 3D 视图、切换模型或项目时后台加载回调访问已释放对象的风险。
- 光束法平差后台执行入口改用 `GuiTaskRunner::runGuarded`，与相机初始化、DEM、MVS/模型长任务使用同一套 GUI owner 生命周期检查，避免 open-coded `QtConcurrent` 在 owner 已释放后继续启动工作。
- `LayerRenderer` 的影像读取、OpenCV/GDAL fallback 和 8-bit 缓存逻辑拆到 `LayerImageLoader`，渲染类只保留场景图层、特征点和匹配线绘制职责，降低后续维护成本。
- `LayerRenderer` 的特征点覆盖层和匹配线 item 构造逻辑继续拆到 `LayerOverlayItems`，渲染类只负责把 overlay item 加入/移出 scene，便于后续独立优化特征点绘制和匹配查看器显示。
- `LayerRenderer` 的特征文件查找、`FeatureFileIO` 解码和 score 回填逻辑拆到 `LayerFeatureLoader`，让渲染器不再直接包含 `FeatureOutput`/LibTorch 相关头文件，并把 MSVC C4267 外部模板 warning 限定在 loader 内部。
- `LayerRenderer` 的匹配拼接 debug PNG 输出、scene item 类型诊断和 SHA1 文件命名拆到 `LayerStitchedDebug`，渲染类不再直接负责调试图写盘和场景项日志。
- `ProjectManager.cpp` 移除未使用的 `SuperPoint`/`SuperGlue`/`FeatureFileIO` 旧算法头和 `QtTorchMacroGuard`，避免主项目 manager 编译单元引入 LibTorch 头导致 MSVC C4267 外部模板 warning。
- 新增项目级 `.clang-format`，固定 C++17、4 空格、Allman 花括号、120 列和禁止 tab 等格式基线；当前不对全仓做机械格式化，后续改动按该配置逐步收敛。
- 新增 `docs/superpowers/plans/2026-06-21-survey-control-quality-loop.md`，记录测绘控制质量闭环的第一批实现计划和后续 GCP/CRS/DOM 扩展顺序。

### 优化

- 密集匹配执行逻辑从 `ProjectManager`/`ReconstructionWorkflowController` 拆到 `DenseMatchRunner`，workflow controller 的 `QtConcurrent` worker 不再捕获 `this`，也不再依赖 GUI manager 生命周期。
- `FeatureExtractionRunner` 和 `FeatureMatchRunner` 对 LibTorch/ATen 触发的 MSVC C4267 外部模板 warning 使用编译单元级局部隔离，避免 Windows CUDA GUI 构建日志继续被第三方头文件窄化警告刷屏。
- `SuperPointBatch.cpp` 拆分描述子诊断和密集描述子采样路径中的超长语句，并新增行宽回归测试，继续收敛旧 SuperPoint 编译单元的格式债。
- `SuperPoint.cpp` 拆分 CSV 描述子写出诊断路径中的超长语句，并新增行宽回归测试，继续收敛 SuperPoint 旧接口实现的格式规范。
- `BundleAdjustDialog.cpp` 拆分信号连接、设置恢复和结果表格更新中的超长语句，并新增行宽回归测试，继续收敛 GUI 对话框格式规范。
- `CanvasWidget.cpp` 拆分匹配查看器诊断日志、特征异步加载和方向估计路径中的超长语句，并新增行宽回归测试，继续收敛图像视图代码格式规范。
- `LayerRenderer.cpp` 拆分特征覆盖层、拼接影像和匹配线日志中的超长语句，并新增行宽回归测试，继续收敛图层渲染代码格式规范。
- `VocabularyOverlapDialog.cpp` 拆分相机重叠诊断、描述子转换和参数恢复中的超长语句，并新增行宽回归测试，继续收敛重叠对规划对话框格式规范。
- `ProjectModelManager.cpp` 拆分网格/纹理结果登记和 Poisson/简化参数恢复中的超长语句，并新增行宽回归测试，继续收敛模型生成 manager 格式规范。
- `cli_bundle_adjust.cpp` 拆分 BA A/B 对比指标、输入摘要和质量门禁判断中的超长语句，并新增行宽回归测试，继续收敛 CLI 格式规范。
- 手动特征提取和 DEM 自动流水线中的特征提取步骤改为向 `FeatureExtractionRunner` 传递 `QPointer<ProjectManager>`，runner 在回写 `appendIpfindResult` 前再次检查项目管理器生命周期，避免关闭/切换项目后后台特征任务回调已释放对象。
- 观测网络构建后台任务改用 `QPointer<ProjectManager>` 和项目路径 guard 回写进度、结果和完成信号，避免用户关闭项目或切换工程后旧任务写回当前项目。
- `FeatureMatchRunner` 公共入口改为接收 `QPointer<ProjectManager>`，手动匹配和 DEM 自动流水线匹配步骤不再把裸 `ProjectManager*` 捕进后台 worker；匹配结果写回时会重新检查项目仍然有效且路径未切换。
- 3D 视图 PLY 异步加载的进度回报改为通过 `QMetaObject::invokeMethod(..., Qt::QueuedConnection)` 回到 `CameraSceneWidget` 所在线程，避免后台加载线程直接通过 GUI 对象发射进度信号。
- CanvasWidget 特征点异步加载改为按请求创建 watcher，并用 generation guard 丢弃旧影像/旧后缀的迟到结果，避免切换影像后旧特征点覆盖当前画布。
- CanvasWidget 不再直接包含 `SuperPoint`、`FeatureOutput` 或 `FeatureFileIO`，改为复用 `LayerFeatureLoader::loadFeatureKeypointsFromFile()`；视图层只消费 `cv::KeyPoint`，LibTorch/ATen 头文件和 MSVC C4267 warning 继续隔离在 feature loader/runner 编译单元内。
- `CanvasWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持影像异步加载、特征点缓存、迟到结果丢弃、缩放和平移行为不变。
- `GlobalSettings`、`ProjectDialogJsonSettingBase` 和 `DialogSettingStore` 去除 tab 缩进或旧 `m_` 私有成员，并将私有成员收敛为 `_settings` / `_plascanPath` / `_dialogKey`，作为 settings 小模块逐步迁移 `_lowerCamelCase` 私有成员规范的起点。
- `AppConfigManager` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持窗口状态、最近项目和文件对话框子管理器访问接口不变。
- `ProjectConfigManager`、`ProjectUiConfigManager` 和 `ProjectWorkflowConfigManager` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，继续收敛 GUI 配置管理层命名规范。
- `ProjectSupportUtils.h` 去除残留 tab 缩进，让 GUI 项目支持层头文件继续按 4 空格缩进规范收敛。
- `WindowPanel` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，并整理该小组件内的紧凑控制语句花括号风格。
- `ReferencePanelWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛参考面板组件命名规范。
- `TaskStatusWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持状态栏进度、取消请求和取消中状态行为不变。
- `DisparityHeatmapOverlay` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持密集匹配视差热力图透明 mask 和无效像素显示行为不变。
- `MatchLineOverlay` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持匹配查看器连线、高亮、内点过滤和可见匹配缓存行为不变。
- `ImageViewWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 Qt Designer `ui.m_view` 对象名不变，继续收敛匹配查看器单图视图命名规范。
- `DualImageViewer` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 Qt Designer `ui.m_leftView/ui.m_rightView/ui.m_splitter` 对象名不变，继续收敛匹配查看器双图容器命名规范。
- `WorkspaceCenterWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 Qt Designer 子对象名不变，继续收敛中央工作区视图切换组件命名规范。
- `LogPanel` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 Qt Designer `ui.m_*` 对象名不变，继续收敛日志面板命名规范。
- `DataTreeWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 Qt Designer `ui.m_view` 对象名不变，继续收敛工作区目录树命名规范。
- `LayerRenderer` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持影像图层、特征点覆盖层、匹配线和拼接调试输出行为不变。
- `LayerOverlayItems` 内部特征点覆盖层 item 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持特征点裁剪、排序和绘制行为不变。
- `Logger` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持日志文件轮转、sink 分发和 Qt 字符串适配接口不变。
- `ProjectDashboardWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持概览页摘要、任务、参考数据、质量指标和报告表格行为不变。
- `MainMenu` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持菜单构建、UI 绑定、工具栏动作和最近项目菜单行为不变。
- `MenuWorkflowController` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，并把对话框设置缓存指针从 `public` 收回 `private`，保持特征提取、词汇重叠、空三、三维重建、DEM/DOM 和报告入口行为不变。
- `ReconstructionWorkflowController` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持稀疏重建、密集重建、模型生成对话框入口和设置缓存行为不变。
- `PlascanArchive` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，并整理归档读写封装中的局部控制语句花括号风格，保持 `.plascan` 读写行为不变。
- `ProjectFilesManager` 私有成员和拆分数据模型注释从 `m_` 迁移到 `_lowerCamelCase`，保持 `project_files.json` / `project_results.json` 拆分、惰性结果加载和匹配结果精简记录行为不变。
- `ProjectData` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持 `.plascan` 创建/打开/保存、惰性 results 加载、临时 metadata 和防抖归档同步行为不变。
- `ProjectTerrainProductsManager` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持 DEM/DOM 自动流水线、密集点云 DEM 和正射入口行为不变。
- `ThreeDReconstructionDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续按小模块逐步收敛 GUI 成员命名规范。
- `Camera` 核心类内部私有成员从 `_fu/_R/_C` 等短名迁移到 `_focalX/_cameraToWorldRotation/_cameraCenter` 等 `_lowerCamelCase` 描述性命名，公开 accessor 和 Tsai 文件字段保持兼容。
- `PositiveDepthCameraModel` 像素变换缓存私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持正深度投影、反投影和 rectified pixel transform 行为不变。
- `ForwardIntersectionResultsDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，并顺手整理该小对话框内的紧凑控制语句花括号风格。
- `DenseCloudRefineDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理该小对话框内的紧凑控制语句和连接语句格式。
- `DepthFusionDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理该小对话框内的紧凑控制语句和连接语句格式。
- `DepthMapEstimateDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理该小对话框内的紧凑控制语句和连接语句格式。
- `CameraConvertDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，继续按小模块逐步收敛 GUI 成员命名规范。
- `MeshReconstructionDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理该小对话框内的紧凑控制语句和连接语句格式。
- `ModelExportDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理该小对话框内的紧凑控制语句和连接语句格式。
- `TextureMappingDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理该小对话框内的紧凑控制语句和连接语句格式。
- `MapProjectDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，并整理该小对话框内的紧凑控制语句和循环花括号风格。
- `SurveyControlDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持测绘控制 CSV/表格查看行为不变。
- `CreateDemDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理该小对话框内的紧凑控制语句花括号风格。
- `MVSProgressDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理进度/取消回调中的紧凑控制语句花括号风格。
- `TriangulationDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，并整理三角化参数应用/运行回调中的紧凑控制语句花括号风格。
- `WorkflowReportDialog` / `ReportChartWidget` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续按小模块逐步收敛 GUI 成员命名规范。
- `FeaturePointVisualizationDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛特征点可视化配置模块的命名规范。
- `ObservationNetworkDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛观测网络参数/预览对话框命名规范。
- `ObservationNetworkView` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持观测网络初始布局、力导向迭代、边/标签缓存和节点选择高亮行为不变。
- `OverlapAnalysisDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛影像重叠度分析对话框命名规范。
- `SparseCloudPostProcessDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名和动态创建控件 objectName 不变，继续收敛稀疏点云后处理对话框命名规范。
- `InitCameraPoseDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名和动态匹配/特征类型控件 objectName 不变，继续收敛相机初始化对话框命名规范。
- `ForwardIntersectionCheckDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛前方交汇检测对话框命名规范。
- `DenseMatchDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛密集匹配参数对话框命名规范。
- `dense_match` 核心小类的配置和输入缓存私有成员从 `m_cfg/m_left/m_right` 迁移到 `_config/_left/_right`，保持 BM/SGM/OpenCV SGBM/亚像素细化/验证链路行为不变。
- `SubpixelRefiner` 移除 MSVC 会忽略的 OpenMP `collapse(2)` 子句，保留外层并行，避免 Windows CUDA 构建继续输出 C4849 warning。
- `GroundBackProjector::DemSurface` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持 DEM 点云加载、KD 树高程查询和影像覆盖反投影行为不变。
- `LaserConstraintMap` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持 LiDAR PLY 加载、体素降采样、KD 树索引和最近平面查询行为不变。
- `SparseCloudValidator` 私有成员从 `m_opts` 迁移到 `_options`，保持稀疏点云 PLY/XYZ 质量检查、有限坐标统计和最小点数判定行为不变。
- `DepthMapFusion` 私有成员从 `m_config/m_filteredDepths` 迁移到 `_config/_filteredDepths`，保持多视深度融合、流式首帧融合和一致性过滤深度图输出行为不变。
- `DepthMapGenerator` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持深度图调度、manifest 复用、图像缓存、source view 缓存和融合过滤深度图输出行为不变。
- `MvsWorkspaceManifest` 私有成员从 `m_configHash/m_frames` 迁移到 `_configHash/_frames`，保持深度图 manifest 读写、完成帧自然排序和可复用缓存判定行为不变。
- `TerrainProductManifest` 私有成员从 `m_records` 迁移到 `_records`，保持 DEM/DOM 产品 manifest 读写、质量栅格路径保留和自然排序行为不变。
- `MultiViewTrackBuilder` 及其内部并查集缓存私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持多视 track 合并、冲突拆分和置信度统计行为不变。
- `StereoDenseCloudPipeline` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持 stereo dense pipeline 默认配置、取消入口和 OriginalDepth/RectifiedDisparity 路径选择行为不变。
- `MatchPairSelectorDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛匹配对选择器命名规范。
- `FeatureMatchingDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛特征匹配参数对话框命名规范。
- `FeatureExtractionDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛特征提取参数对话框命名规范。
- `BundleAdjustDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛光束法平差参数对话框命名规范。
- `DenseCloudDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛稠密点云参数对话框命名规范。
- `MatchViewerDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛匹配查看器命名规范。
- `VocabularyOverlapDialog` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保留 `.ui` 生成对象名不变，继续收敛词汇重叠度分析对话框命名规范。
- `SFMService.cpp` 和 `VocabularyOverlapDialog.cpp` 对 LibTorch/ATen 触发的 MSVC C4267 外部模板 warning 使用编译单元级局部隔离，避免 Windows CUDA GUI 构建日志继续被这两个 Torch-heavy 编译单元刷屏。
- DISK/ALIKED/LoFTR Torch 包装器和 `ExtractorFactory` 内部 adapter 私有成员从 `m_` 迁移到 `_lowerCamelCase`，并把 LibTorch C4267 warning guard 局部化到对应 Torch-heavy 编译单元，保持模型加载、CUDA 选择和默认 DISK/ALIKED 入口行为不变。
- `MatcherFactory` 内部 SuperGlue/LightGlue/Python adapter 私有成员从 `m_` 迁移到 `_lowerCamelCase`，并对该 Torch-heavy 工厂编译单元补齐 MSVC C4267 局部 warning guard，保持 matcher 创建、Python 兜底和 CUDA 参数传递行为不变。
- `ProjectManager` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持项目数据访问、任务分发、BA 预览缓存、取消标志和质量报告刷新入口行为不变。
- `ProjectModelManager` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持网格重建、纹理映射、模型结果登记和 `QPointer` 生命周期守护行为不变。
- `ProjectReconstructionManager` 子 manager 指针从 `m_` 迁移到 `_lowerCamelCase`，保持稀疏、密集、模型重建任务分发和进度信号转发行为不变。
- `ProjectDenseReconstructionManager` 及其内部深度帧 LRU cache 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持深度图估计、深度图融合、稠密点云生成/后处理和 MVS 取消入口行为不变。
- `ProjectSparseReconstructionManager` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持两视预览三角化、稀疏点云后处理和 guarded runner 回调行为不变。
- `ProjectCameraSetupManager` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持单张/批量相机导入、EXIF/内参初始化和相机 SFM 初始化入口行为不变。
- `MainWindow` 私有成员从 `m_` 迁移到 `_lowerCamelCase`，保持菜单接线、项目管理、状态栏任务、仪表盘刷新、特征/匹配入口和工作区视图切换行为不变。
- 删除 GUI 工程服务层中已无生产目标引用的 `ProjectBaInputBuilder.cpp` / `ProjectTriangulationService.cpp` 空壳兼容编译单元；兼容接口继续保留在对应头文件中，测试目标直接链接 core 实现。
- GUI 工程 BA 输入和三角化调用方直接依赖 `src/core/sfm/BaInputBuilder.h` / `TriangulationService.h`，删除 `ProjectBaInputBuilder.h` / `ProjectTriangulationService.h` 这层 header-only 兼容 wrapper，减少旧接口暴露面。
- `LayerRenderer.h` 拆分超长的拼接影像接口声明，并新增头文件 120 列风格回归测试，保持影像拼接、特征点和匹配线渲染接口行为不变。
- `FeatureExtractionRunner.cpp` 拆分超长的特征提取配置日志格式串，并新增 120 列风格回归测试，保持 DISK/ALIKED TorchScript 提取入口和配置日志内容不变。
- `LightGlueMatcher.h` 拆分超长的 `_filterScores` 声明，并新增头文件 120 列风格回归测试，保持 LightGlue 匹配器接口行为不变。
- `PointCloudPreprocess.h` 拆分点云预处理默认设备参数的超长声明，并新增头文件 120 列风格回归测试，保持体素降采样和统计去噪接口行为不变。
- `Intersection.cpp` 拆分前方交汇健康检查的超长布尔表达式，并新增源文件 120 列风格回归测试，保持正深度和有限值判定逻辑不变。
- `EpipolarRectifier.cpp` 拆分 MVS 极线校正诊断日志的超长格式串，并新增源文件 120 列风格回归测试，保持日志输出内容和校正逻辑不变。
- `OverlapAnalyzer.cpp` 拆分重叠对阈值计算的超长表达式，并新增源文件 120 列风格回归测试，保持中心距离阈值和重叠得分逻辑不变。
- `VocabularyOverlapRetriever.cpp` 拆分 TF-IDF 词频权重计算中的超长 `document_frequency` 下标表达式，并新增源文件 120 列风格回归测试，保持 dense/inverted pair scoring 行为不变。
- `BaInputBuilder.cpp` 拆分 BA 输入构造中影像路径规范化的超长调用，并新增源文件 120 列风格回归测试，保持控制点 track 和比例尺约束构造行为不变。

### 验证

- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.BaInputBuilderSourceKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `BaInputBuilder.cpp:525` 超长路径规范化调用；拆分调用后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target sfm -Jobs 8`、`-Target test_ba_input_builder`、`-Target test_gui_project_utils` 和 `-Target plascan_gui` 均通过，确认 SfM 核心库、BA 输入构造测试、GUI 工具测试和主 GUI 可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.BaInputBuilderSourceKeepsLinesWithinStyleLimit|BaInputBuilderSurveyControl" --output-on-failure` 通过，3/3，验证 BA 输入构造源文件行宽、Survey Control 控制点 track 和比例尺约束回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.VocabularyOverlapRetrieverSourceKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `VocabularyOverlapRetriever.cpp:796` 超长 TF-IDF 表达式；拆分 `word_index` 临时量后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target overlap -Jobs 8`、`-Target test_vocabulary_overlap_retriever`、`-Target test_gui_project_utils` 和 `-Target plascan_gui` 均通过，确认词汇重叠核心库、单测、GUI 工具测试和主 GUI 可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.VocabularyOverlapRetrieverSourceKeepsLinesWithinStyleLimit|VocabularyOverlapRetriever" --output-on-failure` 通过，5/5，验证词汇重叠源文件行宽、dense/inverted scoring、维度检查和取消回调回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.OverlapAnalyzerSourceKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `OverlapAnalyzer.cpp:495` 超长阈值表达式；拆分表达式后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target overlap -Jobs 8`、`-Target test_overlap_analyzer`、`-Target test_gui_project_utils` 和 `-Target plascan_gui` 均通过，确认重叠分析核心库、单测、GUI 工具测试和主 GUI 可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.OverlapAnalyzerSourceKeepsLinesWithinStyleLimit|OverlapAnalyzer" --output-on-failure` 通过，3/3，验证重叠分析源文件行宽和 ReferenceSphere 重叠分析回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.EpipolarRectifierSourceKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `EpipolarRectifier.cpp:216` 超长诊断日志；拆分相邻字符串字面量后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target mvs -Jobs 8`、`-Target test_mvs_rectifier_unit`、`-Target test_gui_project_utils` 和 `-Target plascan_gui` 均通过，确认 MVS 核心库、极线校正单测、GUI 工具测试和主 GUI 可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.EpipolarRectifierSourceKeepsLinesWithinStyleLimit|EpipolarRectifier|DisparityTriangulator\.TransposedRectifiedDepthTriangulationKeepsLowReprojectionError" --output-on-failure` 通过，8/8，验证极线校正源文件行宽、水平/垂直基线校正和转置校正深度三角化回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.IntersectionSourceKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `Intersection.cpp:236` 超长健康检查表达式；拆分表达式后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target intersection -Jobs 8`、`-Target test_gui_project_utils` 和 `-Target plascan_gui` 均通过，确认前方交汇核心库、GUI 工具测试和主 GUI 可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.IntersectionSourceKeepsLinesWithinStyleLimit|ForwardIntersection|Triangulation|test_initial_sparse_triangulator" --output-on-failure` 通过，26/26，验证前方交汇源文件行宽、前方交汇对话框、空三/三角化 UI、三角化服务和多视 track 三角化回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.PointCloudPreprocessHeaderKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `PointCloudPreprocess.h` 超长声明；拆分两个默认参数声明后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target meshing -Jobs 8`、`-Target test_mesh_reconstructor` 和 `-Target plascan_gui` 均通过，确认点云预处理头文件变更可重新编译网格库、网格测试和主 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.PointCloudPreprocessHeaderKeepsLinesWithinStyleLimit|MeshReconstructor" --output-on-failure` 通过，8/8，验证点云预处理头文件行宽和网格重建流式高度网格/Poisson fallback 回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.LightGlueMatcherHeaderKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `LightGlueMatcher.h` 超长声明；拆分声明后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target lightglue_matcher -Jobs 8`、`-Target test_gui_project_utils` 和 `-Target plascan_gui` 均通过，确认 LightGlue 匹配器、GUI 工具测试和主 GUI 目标可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.LightGlueMatcherHeaderKeepsLinesWithinStyleLimit|FeatureMatchingDialogTest\.(DefaultsToLightGlueAlgorithm|ProjectAvailableSuffixesConstrainLightGlueChoicesAfterApplyingSettings)|FeatureNamingCleanupTest\.MatcherFactorySuppressesLibTorchC4267Warnings|CodeStyleTest\.MatcherFactoryUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，5/5，验证 LightGlue 头文件行宽、默认匹配算法选择、项目可用后缀过滤和 matcher 工厂命名/warning guard 回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureExtractionRunnerTest\.SourceLinesStayWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `FeatureExtractionRunner.cpp` 超长日志行；拆分日志格式串后通过。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureExtractionRunnerTest\.(SourceLinesStayWithinStyleLimit|DiskAndAlikedUseNativeTorchscriptExtractor|FeatureExtractionLogUsesSelectedAlgorithmName)|GuiAsyncLifetimeTest\.FeatureExtractionRunnerUsesGuardedProjectManagerCallbacks|FeatureExtractionDialogTest\.(NativeFeatureRunnerReceivesGrayscaleRange|DiskSelectionShowsResolvedModelPath|DefaultsToDiskAlgorithm)" --output-on-failure` 通过，7/7，验证特征提取 runner 行宽、DISK/ALIKED 原生提取、日志名称、生命周期 guard 和灰度阈值配置链路保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `FeatureExtractionRunner.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.LayerRendererHeaderKeepsLinesWithinStyleLimit" --output-on-failure` 在生产代码调整前按预期失败，确认新增风格回归测试能抓到 `LayerRenderer.h` 超长声明；拆分声明后通过。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.LayerRendererHeaderKeepsLinesWithinStyleLimit|CodeStyleTest\.LayerRendererUsesLowerCamelPrivateMemberNames|CanvasWidgetResponsivenessTest\.LayerRenderer" --output-on-failure` 通过，6/6，验证 `LayerRenderer` 头文件行宽、成员命名和渲染职责拆分断言保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `LayerRenderer` 相关 GUI 编译单元并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.MainWindowUsesLowerCamelPrivateMemberNames" --output-on-failure` 在生产代码迁移前按预期失败，确认新增风格回归测试能抓到 `MainWindow` 旧 `m_` 成员；迁移后通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MainWindow.cpp`、菜单/重建 workflow controller 相关编译单元并链接 GUI。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_gui_project_utils -Jobs 8` 通过，重新编译 GUI 工具/风格测试目标。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MainWindow|GuiAsyncLifetimeTest\.FeatureMatchRunnerUsesGuardedProjectManagerCallbacks|DenseMatchCancelTest\.StatusBarCancelSignalStopsRemainingPairs" --output-on-failure` 通过，14/14，验证 `MainWindow` 命名迁移后菜单接线、仪表盘、状态栏取消、特征匹配 guard 和工作区刷新相关回归保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.DepthMapGeneratorUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `DepthMapGenerator` 私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_mvs_pipeline -Jobs 8` 通过，重新编译依赖 `DepthMapGenerator` 的 MVS 测试目标。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_gui_project_utils -Jobs 8` 通过，重新编译 GUI 工具/风格测试目标。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新链接依赖 `DepthMapGenerator` 的 Qt GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.DepthMapGeneratorUsesLowerCamelPrivateMemberNames|MvsPipelineTest|MvsWorkspaceManifest|MvsSourcePlanner|MvsDepthPostprocess|DepthMapPersistence|DepthFrameUtils" --output-on-failure` 通过，42/42，验证 MVS manifest、source planner、深度后处理、持久化和 pipeline 子集。
- `python -m pytest tests/test_mvs_scheduler_config.py -q` 通过，72/72，验证 MVS 调度源码结构、内存策略、取消检查和 source view 缓存断言已同步到新成员命名。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.(TorchFeatureWrappersUseLowerCamelPrivateMemberNames|ExtractorFactoryUsesLowerCamelPrivateMemberNames)|FeatureNamingCleanupTest.TorchFeatureWrappersSuppressLibTorchC4267Warnings|FeatureExtractionRunnerTest.DiskAndAlikedUseNativeTorchscriptExtractor|FeatureExtractionDialogTest.(DiskSelectionShowsResolvedModelPath|DefaultsToDiskAlgorithm|DiskSelectionHidesSuperPointOnlyAdvancedRows|DiskSelectionShowsAdvancedApplicabilityHint)" --output-on-failure` 先失败后通过，8/8，验证 Torch 特征包装器/工厂命名迁移、C4267 局部 guard 和 DISK/ALIKED GUI 默认入口保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target feature_extractors_traditional -Jobs 8` 通过，重新编译 `ExtractorFactory.cpp`，且不再输出 LibTorch/ATen C4267 warning。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新链接依赖 DISK/ALIKED/LoFTR/Torch 特征包装器的 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MatcherFactoryUsesLowerCamelPrivateMemberNames|FeatureNamingCleanupTest.MatcherFactorySuppressesLibTorchC4267Warnings" --output-on-failure` 先失败后通过，2/2，验证 `MatcherFactory` 内部 adapter 命名迁移和 C4267 局部 guard。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "AlgorithmCompat|FeatureMatch|MatcherFactory|LightGlue|SuperGlue|LoFTR" --output-on-failure` 通过，18/18，验证匹配算法兼容、匹配 runner/sidecar、LightGlue/SuperGlue/LoFTR 相关入口保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target feature_match_factory -Jobs 8` 通过，重新编译 `MatcherFactory.cpp`，且不再输出 LibTorch/ATen C4267 warning。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ProjectManager` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectManagerUsesLowerCamelPrivateMemberNames|GuiAsyncLifetimeTest.(BundleAdjustProgressCallbackUsesQPointerGuard|BundleAdjustUsesGuardedTaskRunner)|BundleAdjustStatusBarTest.UsesAtProgressWidgetWithCancelableCoreOptimization|FeatureNamingCleanupTest.ProjectManagerDoesNotIncludeLegacyTorchAlgorithmHeaders|ProjectReferenceTerrainBaTest.MenuWorkflowStartsBundleAdjustWithReferenceTerrainSettings|ProjectManagerQualityReportTest.PipelineStageBoundariesRefreshReconstructionQualityReport|DenseMatchCancelTest.StatusBarCancelSignalStopsRemainingPairs" --output-on-failure` 通过，8/8，验证 `ProjectManager` 命名迁移后 BA 进度/取消、参考地形 BA、质量报告刷新、密集匹配取消和旧 Torch 头清理断言保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectManager.cpp` 及依赖该头的 GUI 编译单元并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetimeTest.ProjectModelTasksUseQPointerGuards|CodeStyleTest.ProjectModelManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，2/2，验证 `ProjectModelManager` 私有成员迁移和模型长任务 `QPointer` guard 保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectModelManager.cpp`、`ProjectReconstructionManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectReconstructionManagerUsesLowerCamelPrivateMemberNames|GuiAsyncLifetimeTest.ProjectModelTasksUseQPointerGuards|ProjectTriangulationUiTest|CodeStyleTest.ProjectModelManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，5/5，验证 `ProjectReconstructionManager` 命名迁移后重建任务分发相关轻量回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectReconstructionManager.cpp`、`ProjectTaskDispatcher.cpp`、`ProjectManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectDenseReconstructionManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ProjectDenseReconstructionManager` 和内部深度帧 LRU cache 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectDenseReconstructionManagerUsesLowerCamelPrivateMemberNames|DenseDepthCameraLookupTest|GuiAsyncLifetimeTest.(EstimateDepthMapCallbacksUseQPointerGuard|GenerateDenseCloudCallbacksUseQPointerGuard|DenseCloudFusionUsesGuardedTaskRunner|DenseCloudRefineUsesGuardedTaskRunner|DepthMapSparsePreloadWorkersGuardGeneratorLifetime)|ProjectManagerQualityReportTest.PipelineStageBoundariesRefreshReconstructionQualityReport|DenseCloudRefineTest" --output-on-failure` 通过，10/10，验证密集重建 manager 命名迁移后深度缓存相机查找、MVS 生命周期守护、质量报告刷新和点云后处理设备日志保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectDenseReconstructionManager.cpp`、`ProjectReconstructionManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectSparseReconstructionManagerUsesLowerCamelPrivateMemberNames|ProjectTriangulationUiTest|SparseCloudPostProcessDialogTest|CodeStyleTest.SparseCloudPostProcessDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，10/10，验证 `ProjectSparseReconstructionManager` 命名迁移、三角化 UI 和稀疏点云后处理对话框回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectSparseReconstructionManager.cpp`、`ProjectReconstructionManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectCameraSetupManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ProjectCameraSetupManager` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectCameraSetupManagerUsesLowerCamelPrivateMemberNames|GuiAsyncLifetimeTest.CameraSetupUsesGuiTaskRunnerForBackgroundSfm|ProjectCameraImportServiceTest|InitCameraPoseDialogTest" --output-on-failure` 通过，5/5，验证相机 setup manager 命名迁移、相机导入服务、相机 SFM 初始化生命周期守护和初始化对话框匹配配置保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectCameraSetupManager.cpp`、`ProjectManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MainMenuUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `MainMenu` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MainMenu|MainWindowMenu|CodeStyleTest.MainMenuUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，13/13，验证主菜单动作、UI 绑定和菜单接线保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MainMenu.cpp`、`MainWindow.cpp`、`MenuWorkflowController.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MenuWorkflowControllerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `MenuWorkflowController` 私有成员迁移到 `_lowerCamelCase` 且对话框设置缓存不再暴露在 `public` 段。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MenuWorkflowControllerUsesLowerCamelPrivateMemberNames|MenuWorkflowControllerTest|MainWindowMenuWiringTest|AerialTriangulationWorkflowTest|GuiAsyncLifetimeTest.FeatureExtractionRunnerUsesGuardedProjectManagerCallbacks|FeatureNamingCleanupTest.GuiOrchestrationDoesNotExposeLegacyAlgorithmSpecificInterfaces|FeatureExtractionRunnerTest.FeatureExtractionLogUsesSelectedAlgorithmName|VocabularyOverlapDialogTest.(SupportsCameraModelModeAndStatusProgress|DisplaysResultsWithoutOverwritingSummary)|MainMenuTest.WorkflowMenuExposesOnlyOneClickThreeDReconstruction" --output-on-failure` 通过，17/17，验证菜单工作流编排、空三入口、词汇重叠设置写回、特征提取 runner guard 和报告/重建入口保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MenuWorkflowController.cpp`、`MainWindow.cpp` 并链接 GUI。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_gui_project_utils -Jobs 8` 通过，重新编译 GUI 工具测试。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.ReconstructionWorkflowControllerUsesLowerCamelPrivateMemberNames" --output-on-failure` 在生产代码迁移前按预期失败，确认新增风格回归测试能抓到 `ReconstructionWorkflowController` 旧 `m_` 成员。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_gui_project_utils -Jobs 8` 和 `-Target plascan_gui` 均通过，验证重建工作流控制器迁移后 GUI 工具测试与主程序可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.ReconstructionWorkflowControllerUsesLowerCamelPrivateMemberNames|CodeStyleTest\.MenuWorkflowControllerUsesLowerCamelPrivateMemberNames|GuiAsyncLifetimeTest\.ObservationNetworkWorkerUsesGuardedProjectManagerCallbacks|DenseMatchCancelTest\.StatusBarCancelSignalStopsRemainingPairs|DenseMatchRunnerTest\.DenseMatchWorkerDoesNotCaptureWorkflowControllerThis|InitCameraPoseDialogTest\.SfmInitializationUsesSelectedMatchPipeline" --output-on-failure` 通过，6/6，验证重建菜单入口、观测网络异步守护、密集匹配取消和初始化 SfM 参数链路保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.PlascanArchiveUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `PlascanArchive` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "PlascanArchive|ProjectData" --output-on-failure` 通过，9/9，验证 `.plascan` 写入替换、项目 metadata 保存和迁移逻辑保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `PlascanArchive.cpp`、`ProjectData.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectFilesManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ProjectFilesManager` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectFilesManagerUsesLowerCamelPrivateMemberNames|ProjectFilesManagerTest|ProjectDataCameraMetadataTest|ProjectWorkflowReportsTest" --output-on-failure` 通过，6/6，验证项目文件 metadata/results 分流、相机 metadata 清理和重建质量报告登记保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectFilesManager.cpp`、`ProjectData.cpp` 及相关项目支持编译单元并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectDataUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ProjectData` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectDataUsesLowerCamelPrivateMemberNames|ProjectDataCameraMetadataTest|ProjectFilesManagerTest|ProjectWorkflowReportsTest|ProjectReferenceDatasetsTest|ProjectReferenceTerrainBaTest" --output-on-failure` 通过，15/15，验证项目 metadata/results 保存、参考数据登记、参考地形 BA 和重建质量报告登记保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectData.cpp` 及相关项目支持编译单元并链接 GUI。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_gui_project_utils -Jobs 8` 通过，重新编译 GUI 工具测试。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectTerrainProductsManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ProjectTerrainProductsManager` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainPipeline|TerrainProducts|CodeStyleTest.ProjectTerrainProductsManagerUsesLowerCamelPrivateMemberNames|GuiAsyncLifetimeTest.FeatureExtractionRunnerUsesGuardedProjectManagerCallbacks|ProjectManagerQualityReportTest.PipelineStageBoundariesRefreshReconstructionQualityReport" --output-on-failure` 通过，16/16，验证 DEM/DOM 异步入口、质量报告刷新和生命周期守护断言保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectTerrainProductsManager.cpp`、`ProjectManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DenseMatchCoreUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 dense_match 核心私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DenseMatchCoreUsesLowerCamelPrivateMemberNames|BlockMatcher|SgmMatcher|SubpixelRefiner|DisparityValidator|DenseMatchIntegration" --output-on-failure` 通过，18/18，验证 BM/SGM/亚像素细化、视差验证和 dense match 集成链路保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_dense_match_unit -Jobs 8` 通过，重新编译 `dense_match`，且不再输出 `SubpixelRefiner.cpp` 的 OpenMP C4849 warning。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新链接依赖 `dense_match` 的 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.GroundBackProjectorUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `DemSurface` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "OverlapAnalyzer|CodeStyleTest.GroundBackProjectorUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，3/3，验证基准球面重叠分析和 `GroundBackProjector` 风格断言保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `GroundBackProjector.cpp`、`OverlapAnalyzer.cpp` 相关依赖并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "LaserConstraintMap|CodeStyleTest.LaserConstraintMapUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，5/5，验证 LiDAR PLY/高度面/体素降采样和命名迁移断言保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.SparseCloudValidatorUsesLowerCamelPrivateMemberNames|MvsPipeline" --output-on-failure` 通过，21/21，验证 `SparseCloudValidator` 命名迁移后 MVS pipeline 基础链路保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.DepthMapFusionUsesLowerCamelPrivateMemberNames" --output-on-failure` 在生产代码迁移前按预期失败，确认新增风格回归测试能抓到 `m_config/m_filteredDepths` 旧命名。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_mvs_pipeline -Jobs 8`、`-Target test_gui_project_utils`、`-Target plascan_gui` 均通过，验证 `DepthMapFusion` 成员迁移后 MVS 测试目标、GUI 工具测试和 GUI 主程序可重新构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest\.DepthMapFusionUsesLowerCamelPrivateMemberNames|MvsPipelineTest\.DepthMapFusion" --output-on-failure` 通过，6/6，验证多视深度融合、取消、双视快速路径、过滤深度图和 source planning 行为不变。
- `python -m pytest tests/test_mvs_scheduler_config.py -q` 通过，72/72，验证 MVS/GUI 结构检查已同步到 `_lowerCamelCase` 成员命名。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新链接依赖 `mvs` 的 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsWorkspaceManifest|CodeStyleTest.MvsWorkspaceManifestUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，8/8，验证深度图 manifest 读写、排序、缓存复用和命名迁移断言保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainProductManifest|CodeStyleTest.TerrainProductManifestUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，4/4，验证 DEM/DOM 产品 manifest 读写、排序和命名迁移断言保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MultiViewTrackBuilder|CodeStyleTest.MultiViewTrackBuilderUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，5/5，验证多视 track 合并、冲突拆分、置信度统计和命名迁移断言保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.StereoDenseCloudPipelineUsesLowerCamelPrivateMemberNames|TerrainPipelineAsyncTest.StereoPoint2DemRunsOffGuiThread" --output-on-failure` 先失败后通过，2/2，验证 stereo dense pipeline 命名迁移和 terrain 异步 stereo 入口保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_gui_project_utils -Jobs 8` 通过，重新编译 GUI 工具测试。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CanvasFeatureLoadCallbacksUseRequestGeneration|CanvasWidgetDoesNotIncludeTorchExtractorHeaders|LayerRendererDelegatesFeatureFileLoadingToDedicatedLoader" --output-on-failure` 通过，3/3。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `CanvasWidget.cpp`、`LayerFeatureLoader.cpp`、`LayerRenderer.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.LayerOverlayItemsUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `LayerOverlayItems` 内部特征点覆盖层 item 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "LayerOverlayItems|LayerRenderer|CodeStyleTest.LayerOverlayItemsUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，6/6；`powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `LayerOverlayItems.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|FeatureNamingCleanup|CanvasWidgetResponsiveness|LayerRenderer" --output-on-failure` 通过，29/29。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，540/540；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CanvasFeatureLoadCallbacksUseRequestGeneration|StaleFeatureLoadsDoNotPaintOverCurrentImage" --output-on-failure` 先失败后通过，验证 CanvasWidget 新增 generation 成员使用 `_lowerCamelCase` 命名且旧结果丢弃逻辑仍有效。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.CanvasWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `CanvasWidget` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CanvasWidget|CanvasFeature|LayerRenderer" --output-on-failure` 通过，11/11，验证 CanvasWidget 命名迁移后影像异步加载、特征点迟到结果丢弃和 LayerRenderer 委托测试保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `CanvasWidget.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ObservationNetworkViewUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ObservationNetworkView` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ObservationNetwork|CodeStyleTest.ObservationNetworkViewUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，3/3，验证观测网络 worker 生命周期检查、视图命名和对话框命名测试保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ObservationNetworkView.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.SettingsFilesUse" --output-on-failure` 先失败后通过，验证 settings 小模块不再含 tab 且私有成员使用 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `GlobalSettings.cpp`、`ProjectDialogJsonSettingBase.cpp`、`DialogSettingStore.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectConfigManagersUseLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 GUI 配置管理类私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "Config|Settings|DialogSetting|ProjectWorkflow" --output-on-failure` 通过，34/34，验证配置、设置持久化和相关工作流配置回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectConfigManager.cpp`、`ProjectUiConfigManager.cpp`、`ProjectWorkflowConfigManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.AppConfigManagerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `AppConfigManager` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "AppConfig|WindowState|RecentProjects|FileDialog|Config|Settings" --output-on-failure` 通过，35/35，验证应用配置聚合入口和子配置管理器回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `AppConfigManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.GuiSupportFilesUseSpacesInsteadOfTabs" --output-on-failure` 先失败后通过，验证 `ProjectSupportUtils.h` 不再含 tab 缩进。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.WindowPanelUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `WindowPanel` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MainMenuTest|CodeStyleTest.WindowPanelUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，12/12，验证 `WindowPanel` 迁移后主菜单窗口面板入口保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `WindowPanel.cpp` / `MainMenu.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ReferencePanelWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ReferencePanelWidget` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ReferencePanelWidget|CodeStyleTest.ReferencePanelWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，1/1，验证参考面板命名迁移回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ReferencePanelWidget.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.TaskStatusWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `TaskStatusWidget` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TaskStatusWidgetTest|CodeStyleTest.TaskStatusWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，2/2，验证状态栏任务控件命名迁移后进度与取消状态行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `TaskStatusWidget.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DisparityHeatmapOverlayUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `DisparityHeatmapOverlay` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DisparityHeatmapOverlayTest|CodeStyleTest.DisparityHeatmapOverlayUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，2/2，验证视差热力图透明 mask 行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `DisparityHeatmapOverlay.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MatchLineOverlayUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `MatchLineOverlay` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MatchLineOverlay|MatchViewer(SidecarOrder|EmptyMatch|Visualization)|CodeStyleTest.MatchLineOverlayUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，5/5，验证匹配连线 overlay 命名迁移后匹配查看器 sidecar 顺序、空匹配和坐标显示回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MatchLineOverlay.cpp`、`DualImageViewer.cpp`、`MatchViewerDialog.cpp`、`ForwardIntersectionCheckDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ImageViewWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ImageViewWidget` 私有成员迁移到 `_lowerCamelCase` 且保留 `ui.m_view`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ImageViewWidget|MatchViewerVisualization" --output-on-failure` 通过，3/3，验证单图视图命名迁移后匹配查看器图像方向、共享加载器和异步失败上报保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ImageViewWidget.cpp`、`MatchLineOverlay.cpp`、`DualImageViewer.cpp`、`MatchViewerDialog.cpp`、`ForwardIntersectionCheckDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DualImageViewerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `DualImageViewer` 私有成员迁移到 `_lowerCamelCase` 且保留 `ui.m_leftView/ui.m_rightView/ui.m_splitter`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DualImageViewer|MatchLineOverlay|ImageViewWidget|MatchViewer(SidecarOrder|EmptyMatch|Visualization)|CodeStyleTest.DualImageViewerUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，7/7，验证双图容器命名迁移后匹配查看器加载、连线、单图显示和坐标回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `DualImageViewer.cpp`、`MatchViewerDialog.cpp`、`ForwardIntersectionCheckDialog.cpp`、`WorkspaceCenterWidget.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.WorkspaceCenterWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `WorkspaceCenterWidget` 私有成员迁移到 `_lowerCamelCase` 且保留 Qt Designer 子对象名。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `WorkspaceCenterWidget.cpp`、`MainWindow.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.LogPanelUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `LogPanel` 私有成员迁移到 `_lowerCamelCase` 且保留 Qt Designer `ui.m_*` 对象名。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "LogPanel|CodeStyleTest.LogPanelUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，1/1，验证日志面板命名迁移回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `LogPanel.cpp`、`MainWindow.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DataTreeWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `DataTreeWidget` 私有成员迁移到 `_lowerCamelCase` 且保留 Qt Designer `ui.m_view`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DataTreeWidget|CodeStyleTest.DataTreeWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，9/9，验证目录树命名迁移后临时模型、结果刷新、文件名排序、路径解析和激活行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `DataTreeWidget.cpp`、`MainWindow.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.LayerRendererUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `LayerRenderer` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "LayerRenderer|CodeStyleTest.LayerRendererUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，5/5，验证渲染器命名迁移后影像加载、overlay 绘制、特征文件加载和拼接 debug 委托保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `LayerRenderer.cpp`、`CanvasWidget.cpp`、`ImageViewWidget.cpp`、`WorkspaceCenterWidget.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.LoggerUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `Logger` 私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_gui_project_utils -Jobs 8` 通过，重新编译 `Logger.cpp`、`plascan_common_log.lib` 及依赖该日志库的测试目标。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ProjectDashboardWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `ProjectDashboardWidget` 私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ProjectDashboard|CodeStyleTest.ProjectDashboardWidgetUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，11/11，验证概览页命名迁移后 dashboard 汇总、表格和主窗口同步回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectDashboardWidget.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ThreeDReconstructionDialogUsesLowerCamelPrivateMemberNames|ThreeDReconstructionDialogTest" --output-on-failure` 先失败后通过，验证三维重建对话框私有成员迁移到 `_lowerCamelCase` 且 UI 默认值/空三模式行为保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.CameraUsesDescriptiveLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 `Camera` 私有成员命名不再使用 `_fu/_R/_C` 等旧短名。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "Camera(Basic|Intrinsics|Projection|ToPositiveDepthModel|Test|FormatConverter)|PositiveDepthCameraModel|ProjectSupportUtilsTest.CameraJsonRoundTripPreservesUnitsAndDepthDirection|ProjectCameraImportServiceTest.BatchImportRecordsActualTsaiSourceFileInCameraMeta|ProjectDataCameraMetadataTest.SetImageCamerasClearsLegacyTopLevelCameraFile" --output-on-failure` 通过，31/31。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CameraToPositiveDepthModel|PositiveDepthCameraModelTest|CodeStyleTest.PositiveDepthCameraModelUsesLowerCamelPrivateMemberNames" --output-on-failure` 通过，9/9，验证正深度相机模型投影/反投影和像素变换缓存命名迁移保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ForwardIntersectionResultsDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证前方交汇结果对话框私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DenseCloudRefineDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证密集点云后处理对话框私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DepthFusionDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证深度图融合对话框私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DepthMapEstimateDialogUsesLowerCamelPrivateMemberNames|DepthMapEstimateDialogTooltipTest" --output-on-failure` 先失败后通过，验证深度图估计对话框私有成员迁移到 `_lowerCamelCase` 且代价函数 tooltip 行为保持可用。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.CameraConvertDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证相机格式转换对话框私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MeshReconstructionDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证网格重建对话框私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MeshReconstructionDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ModelExportDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证模型导出对话框私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ModelExportDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.TextureMappingDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证纹理映射对话框私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `TextureMappingDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MapProjectDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证地图投影/正射参数对话框私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MapProjectDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.SurveyControlDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证测绘控制对话框私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "SurveyControlDialogTest.PopulatesTablesFromProjectMetadata" --output-on-failure` 通过，验证测绘控制表格仍能从项目 metadata 填充。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `SurveyControlDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.CreateDemDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 DEM 创建对话框私有成员迁移到 `_lowerCamelCase`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CreateDemDialogTest" --output-on-failure` 通过，验证 DEM 创建对话框的一键 DEM 工作流 UI 和手动运行保护仍可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `CreateDemDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MVSProgressDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证 MVS 进度对话框私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MVSProgressDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.TriangulationDialogUsesLowerCamelPrivateMemberNames|ProjectTriangulationUiTest" --output-on-failure` 先失败后通过，验证三角化对话框私有成员迁移到 `_lowerCamelCase` 且三角化 UI metadata/长任务 guard 回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `TriangulationDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.WorkflowReportDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证工作流报告对话框和图表控件私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `WorkflowReportDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.FeaturePointVisualizationDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证特征点可视化配置对话框私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `FeaturePointVisualizationDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ObservationNetworkDialogUsesLowerCamelPrivateMemberNames|GuiAsyncLifetimeTest.ObservationNetworkWorkerUsesGuardedProjectManagerCallbacks" --output-on-failure` 先失败后通过，验证观测网络对话框私有成员迁移到 `_lowerCamelCase` 且观测网络 worker 生命周期 guard 回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ObservationNetworkDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.OverlapAnalysisDialogUsesLowerCamelPrivateMemberNames" --output-on-failure` 先失败后通过，验证影像重叠度分析对话框私有成员迁移到 `_lowerCamelCase`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `OverlapAnalysisDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.SparseCloudPostProcessDialogUsesLowerCamelPrivateMemberNames|SparseCloudPostProcessDialogTest" --output-on-failure` 先失败后通过，验证稀疏点云后处理对话框私有成员迁移到 `_lowerCamelCase` 且项目结果筛选、设置同步、外部 PLY 输入行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `SparseCloudPostProcessDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.InitCameraPoseDialogUsesLowerCamelPrivateMemberNames|InitCameraPoseDialogTest" --output-on-failure` 先失败后通过，验证相机初始化对话框私有成员迁移到 `_lowerCamelCase` 且匹配 pipeline 设置行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `InitCameraPoseDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.ForwardIntersectionCheckDialogUsesLowerCamelPrivateMemberNames|ForwardIntersectionCheckDialogTest" --output-on-failure` 先失败后通过，验证前方交汇检测对话框私有成员迁移到 `_lowerCamelCase` 且自动模式 sidecar token 兼容行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ForwardIntersectionCheckDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DenseMatchDialogUsesLowerCamelPrivateMemberNames|DenseMatchDialogLayoutTest" --output-on-failure` 先失败后通过，验证密集匹配参数对话框私有成员迁移到 `_lowerCamelCase` 且右侧参数滚动区/底部按钮布局保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `DenseMatchDialog.cpp` / `DenseMatchDialogUi.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MatchPairSelectorDialogUsesLowerCamelPrivateMemberNames|MatchPairSelectorOverlapCandidatesTest" --output-on-failure` 先失败后通过，验证匹配对选择器私有成员迁移到 `_lowerCamelCase` 且重叠候选显示行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MatchPairSelectorDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.FeatureMatchingDialogUsesLowerCamelPrivateMemberNames|FeatureMatchingDialogTest" --output-on-failure` 先失败后通过，验证特征匹配参数对话框私有成员迁移到 `_lowerCamelCase` 且默认 LightGlue/特征后缀选择行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `FeatureMatchingDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.FeatureExtractionDialogUsesLowerCamelPrivateMemberNames|FeatureExtractionDialogTest" --output-on-failure` 先失败后通过，验证特征提取参数对话框私有成员迁移到 `_lowerCamelCase` 且 DISK 默认值、CUDA 选择、灰度阈值和模型路径行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `FeatureExtractionDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.BundleAdjustDialogUsesLowerCamelPrivateMemberNames|BundleAdjustDialogTest|BundleAdjustDialogLidarTest" --output-on-failure` 先失败后通过，验证光束法平差参数对话框私有成员迁移到 `_lowerCamelCase` 且按钮布局、LiDAR 参数发射和持久化行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `BundleAdjustDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.DenseCloudDialogUsesLowerCamelPrivateMemberNames|DenseCloudDialogTest" --output-on-failure` 先失败后通过，验证稠密点云参数对话框私有成员迁移到 `_lowerCamelCase` 且高级 MVS 质量参数记录保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `DenseCloudDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.MatchViewerDialogUsesLowerCamelPrivateMemberNames|MatchViewer" --output-on-failure` 先失败后通过，验证匹配查看器私有成员迁移到 `_lowerCamelCase` 且匹配 sidecar 顺序、空匹配文件打开和影像显示方向回归保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `MatchViewerDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CodeStyleTest.VocabularyOverlapDialogUsesLowerCamelPrivateMemberNames|VocabularyOverlapDialogTest" --output-on-failure` 先失败后通过，验证词汇重叠度分析对话框私有成员迁移到 `_lowerCamelCase` 且 UI 控件、异步检索、相机模型模式和结果表格行为保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `VocabularyOverlapDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureNamingCleanupTest.TorchHeavyTranslationUnitsSuppressLibTorchC4267Warnings" --output-on-failure` 先失败后通过，验证 `SFMService.cpp` 和 `VocabularyOverlapDialog.cpp` 均有局部 C4267 warning guard。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `SFMService.cpp`、`VocabularyOverlapDialog.cpp` 并链接 GUI；构建日志未发现 `warning C4267`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureNamingCleanupTest.GuiTestsDoNotCompileObsoleteCompatibilityTranslationUnits" --output-on-failure` 先失败后通过，验证 GUI 工具测试不再编译空壳兼容 `.cpp`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，确认删除空壳兼容 `.cpp` 后 GUI 目标仍可构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureNamingCleanupTest\\.(GuiSfmCallersUseCoreServicesDirectly|GuiTestsDoNotCompileObsoleteCompatibilityTranslationUnits)|TriangulationServiceTest|ProjectTriangulationUiTest" --output-on-failure` 先失败后通过，验证 GUI/测试调用方已直连 core SFM 服务且三角化流程保持可用。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，确认删除 header-only 兼容 wrapper 后 GUI 目标仍可构建。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetimeTest.CameraSceneAsyncLoadCallbacksUseQPointerGuards" --output-on-failure` 先失败后通过，验证 PLY 加载 worker 不再直接从后台线程 emit GUI 进度信号。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `CameraModel3DDialog.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|CameraModel3DDialog|ProjectModel|MeshReconstructor|TextureMapper" --output-on-failure` 通过，29/29。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，538/538；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetimeTest.FeatureMatchRunnerUsesGuardedProjectManagerCallbacks" --output-on-failure` 先失败后通过，验证手动特征匹配和 `FeatureMatchRunner` 写回路径使用 `QPointer<ProjectManager>` guard。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `FeatureMatchRunner.cpp`、`MainWindow.cpp`、`ProjectTerrainProductsManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|FeatureMatchRunner|FeatureMatchCliSidecar|TerrainPipelineAsync|FeatureNamingCleanup" --output-on-failure` 通过，30/30。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，538/538；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetimeTest.ObservationNetworkWorkerUsesGuardedProjectManagerCallbacks" --output-on-failure` 先失败后通过，验证观测网络后台任务不再使用裸 `ProjectManager*` 回写进度/结果。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ReconstructionWorkflowController.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|ObservationNetwork|FeatureNamingCleanup" --output-on-failure` 通过，19/19。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，537/537；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetimeTest.FeatureExtractionRunnerUsesGuardedProjectManagerCallbacks" --output-on-failure` 先失败后通过，验证手动特征提取和 DEM 流水线特征提取均使用 `QPointer<ProjectManager>` 回写 metadata。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `FeatureExtractionRunner.cpp`、`MenuWorkflowController.cpp` 和 `ProjectTerrainProductsManager.cpp` 并链接 GUI。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|FeatureNamingCleanup|TerrainPipelineAsync|FeatureExtraction" --output-on-failure` 通过，37/37。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，536/536；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureNamingCleanupTest.TorchRunnerWarningsStayLocalToRunnerTranslationUnits" --output-on-failure` 先失败后通过，验证 Torch runner 内有局部 C4267 warning guard。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `FeatureExtractionRunner.cpp` 和 `FeatureMatchRunner.cpp`，构建日志未再出现这两个编译单元的 `warning C4267`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureNamingCleanupTest|DenseMatchRunnerTest|DenseMatchCancelTest" --output-on-failure` 通过，5/5。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，535/535；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译并链接 `plascan_gui`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DenseMatchRunnerTest.DenseMatchWorkerDoesNotCaptureWorkflowControllerThis" --output-on-failure` 通过，1/1。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DenseMatch|FeatureNamingCleanupTest.ProjectManagerDoesNotIncludeLegacyTorchAlgorithmHeaders" --output-on-failure` 通过，8/8。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Jobs 8` 通过，重新编译 GUI/project/tasks 相关目标并同步 LibTorch 运行时 DLL。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，534/534；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target test_reconstruction_quality_report -Jobs 8` 通过。
- `E:/code/plascan/build/windows-vcpkg-cuda-release/src/core/qc/test_survey_control_import.exe` 通过，3/3。
- `E:/code/plascan/build/windows-vcpkg-cuda-release/src/core/qc/test_reconstruction_quality_report.exe` 通过，3/3。
- `E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=MainMenuTest.ToolsMenuExposesSurveyControlAction:MainWindowTest.ReferenceDatasetActionsConnectToProjectManager:ProjectSurveyControlTest.ImportsCsvIntoProjectMetadata:SurveyControlDialogTest.PopulatesTablesFromProjectMetadata` 通过，4/4。
- `C:/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "SurveyControl|ReconstructionQualityReport|QualityReport|ReferenceDatasetActionsConnectToProjectManager" --output-on-failure` 通过，17/17。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "BundleAdjust(Lidar|ControlPoint|ScaleBar)ConstraintTest|BaInputBuilderSurveyControl|BundleAdjustServiceLidarTest\\.RunWrites(ScaleBar|ControlPoint)ConstraintSummary|SfmSparseResultMetadataTest\\.BundleAdjustAutoEnablesSurveyControlConstraints|BundleAdjustCliTest" --output-on-failure` 通过，11/11。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "SuperPointTest|DiskExtractorTest|AlikedExtractorTest" --output-on-failure` 通过，13/13。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Jobs 8` 通过，重新编译 `CameraModel3DDialog.cpp` 并链接 `plascan_gui`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainPipelineAsync|GuiAsyncLifetime|TerrainDemDom|TerrainProductManifest" --output-on-failure` 通过，36/36。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainPipelineAsyncTest\\.(StereoPoint2DemRunsOffGuiThread|DenseCloudDemRunsOffGuiThread|AutoDemGenerationRunsOffGuiThreadAfterMvs)|GuiAsyncLifetimeTest\\.TerrainAndDenseBackgroundCallbacksUseQPointerGuards" --output-on-failure` 通过，4/4。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainPipelineAsyncTest\\.(MapProjectRunsOffGuiThread|StereoPoint2DemRunsOffGuiThread|DenseCloudDemRunsOffGuiThread|AutoDemGenerationRunsOffGuiThreadAfterMvs)|GuiAsyncLifetimeTest\\.TerrainAndDenseBackgroundCallbacksUseQPointerGuards" --output-on-failure` 通过，5/5。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainPipelineAsyncTest\\.(TerrainProductsManagerDropsBlockingUiWrappers|MapProjectRunsOffGuiThread|StereoPoint2DemRunsOffGuiThread|DenseCloudDemRunsOffGuiThread|AutoDemGenerationRunsOffGuiThreadAfterMvs)|GuiAsyncLifetimeTest\\.TerrainAndDenseBackgroundCallbacksUseQPointerGuards" --output-on-failure` 通过，6/6。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ProjectTriangulationUiTest|GuiAsyncLifetime|AerialTriangulationWorkflow|SparseCloudPostProcess|TerrainPipelineAsync" --output-on-failure` 通过，33/33。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过，重新编译 `ProjectSparseReconstructionManager.cpp` 并链接 `plascan_gui`。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainPipelineAsync|GuiAsyncLifetime" --output-on-failure` 通过，15/15。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|BundleAdjust" --output-on-failure` 通过，36/36。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|MvsPipelineTest|DepthMapFusion|DenseCloudRefine" --output-on-failure` 通过，36/36。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，525/525；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime" --output-on-failure` 通过，11/11。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "GuiAsyncLifetime|InitCameraPose|SfmServiceKnownPoseMode|SfmServicePairPlanning|AerialTriangulationWorkflow" --output-on-failure` 通过，14/14。
- `python -m pytest tests/test_repo_hygiene.py -q` 通过，11/11，27 个 subtest 通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过。
- `powershell -NoProfile -ExecutionPolicy Bypass -File E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -BuildOnly -Jobs 8` 通过，固定 Windows CUDA 主构建目录无需增量编译。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release --output-on-failure` 通过，533/533；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled benchmark，未运行。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "test_layer_renderer_batched_overlay|CanvasWidgetResponsivenessTest.LayerRendererDelegatesOverlayDrawingToDedicatedItems" --output-on-failure` 通过，2/2。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CanvasWidgetResponsivenessTest.LayerRendererDelegatesFeatureFileLoadingToDedicatedLoader|test_layer_renderer_batched_overlay" --output-on-failure` 通过，2/2。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CanvasWidgetResponsivenessTest.LayerRendererDelegatesStitchedPairDebugOutput|test_layer_renderer_batched_overlay" --output-on-failure` 通过，2/2。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "FeatureNamingCleanupTest.ProjectManagerDoesNotIncludeLegacyTorchAlgorithmHeaders" --output-on-failure` 通过，1/1；`plascan_gui` 重新编译 `ProjectManager.cpp` 时未再输出 LibTorch C4267 warning。

## v1.1.6 - 2026-06-21

### 新增

- 当前发布目标同步到 `v1.1.6`，包括根项目版本、core 项目版本、Release 文档和版本一致性测试。
- SfM 匹配链路新增 guided matching v1 诊断字段，记录候选 pair 来源、优先级、guided eligibility、epipolar band 和跳过/失败原因，便于排查航测数据注册率不足。
- 多视 track 新增置信度和来源统计，BA observation 可使用 track length、feature scale、feature score 和 matcher/source confidence 做质量感知加权。
- MVS manifest 扩展记录 source view 数量、source quality score、平均 depth confidence 和有效像素数，为后续深度图质量报告、目录树增量刷新和 MVS 调参提供稳定 metadata。
- GUI 特征点可视化与特征提取 runner 改为通用命名，移除旧的 `SuperPoint*` / `SuperGlue*` GUI 接口，避免用户误以为当前流程仍绑定某个旧算法。

### 优化

- 匹配/空三报告更清晰地区分候选图、实际匹配图、几何验证、guided rematching 和 BA 输入质量，减少“点很多但空三失败”时的排查成本。
- MVS source planning 现在把 shared tracks、几何内点、基线和重叠质量汇总成可记录的 source quality score，深度图结果不再只靠输出目录扫描判断状态。
- 网格/密集点云相关流程继续向流式、分块和可诊断方向收敛，降低大航测数据在 PLY/mesh 阶段出现 `bad allocation` 时的定位难度。
- Windows CUDA 构建与运行说明继续收敛到固定构建目录、固定 `libtorch-cu130` 和 CUDA 13.1 运行时环境。

### 修复

- 修正 Windows 上 `DenseMatchIntegrationTest.SaveAndReloadDisparity` 使用硬编码 `/tmp/test_disparity.tif`，导致没有 `/tmp` 目录时保存视差失败的问题。
- 修正 MVS depth artifact 后续更新可能覆盖已有非空 source plan 的问题，确保 filtered depth 和 manifest 更新仍保留原始选源依据。
- 修正旧 `SuperPointRunner`、`SuperGlueRunner`、`SuperPointVisualizationDialog` 命名残留导致的接口混乱，相关 GUI 入口改用通用特征提取/可视化命名。
- 修正部分 GUI 匹配查看、影像加载和密集点云/模型 metadata 刷新路径中状态提示不够明确的问题，失败时更容易定位到图像路径、PLY 规模或运行时依赖。

### 验证

- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Jobs 8` 通过；脚本注入 VS Dev、CUDA 13.1、`libtorch-cu130` 和 vcpkg 环境，构建系统报告 `ninja: no work to do`。
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_win\build_windows_cuda.ps1 -BuildOnly -RunTests -CTestRegex "Mvs|Sfm|Feature|Match|Bundle|Lidar|Gui" -Jobs 8` 通过，208/208；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 保持 disabled。
- `python -m pytest tests\test_repo_hygiene.py -q` 通过，9 项测试和 27 个 subtest 全部通过。
- `python -m pytest tests\test_gui_algorithm_alignment.py -q` 通过，7/7。
- `DenseMatchIntegrationTest.SaveAndReloadDisparity` 在 Windows 临时目录修复后通过。
- 覆盖过的关键测试包含 `test_gui_project_utils`、`test_mvs_source_planner`、`test_mvs_workspace_manifest`、`test_sfm_pair_planner`、`test_multiview_track_builder` 和 `test_sfm_pipeline`。
- GitHub Actions 会在 `main` 和 `v1.1.6` 推送后触发，远端 CI 结果以 GitHub Actions 页面为准；Release 正文同步记录本地验证结果和已知风险。

### 已知问题

- `PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 仍是 disabled benchmark，本版本未把它纳入通过项。
- 本次发布前重点复跑了 MVS/SfM/Feature/Match/Bundle/LiDAR/Gui 相关 208 项回归；完整长链大数据验证仍建议在夜间任务中继续跟踪内存峰值、mesh 平滑质量和 DEM/DOM 质量。

## v1.1.5 - 2026-06-19

### 新增

- 当前开发和发版目标统一同步到 `v1.1.5`，包括根项目版本、core 项目版本、Release 文档和版本一致性测试。
- 新增四阶段重建链路优化计划，覆盖 MVS workspace/取消/内存、MVS 选源和深度质量、DEM/DOM 正式产品链、DEM/LiDAR 参考验证与 BA soft prior。
- 新增 GitHub Release 说明要求：推送 tag 后必须创建或更新同名 Release，正文必须写清新增、优化、修复、验证和已知问题，不能只写 `PlaScan vX.Y.Z`。
- 参考 DEM 已接入 BA soft prior 执行链路，GUI 可从参考数据预检进入参考地形平差，并在 BA 报告中记录参考地形约束摘要。
- 参考地形平差 GUI 入口现在可在没有 DEM、但存在 role=ba_prior 的 PLY LiDAR/点云时，显式启动 LiDAR 点到面 BA soft prior。
- 参考数据精度检查现在会落地可追溯 artifact：DEM 差分 GeoTIFF、DEM 绝对差分 GeoTIFF、点云配准前/后误差 CSV 和 Sim3 transform JSON。
- 参考点云精度检查支持非配对点云：当重建点云和 LiDAR/参考点云点数不一致时，自动使用最近邻平移配准并输出同样的误差 CSV、transform JSON 和 RMSE/P95 指标。
- 新增 `src/core/mvs/README.md` 和 `src/core/terrain/README.md`，记录 MVS manifest/source planning/streaming fusion 与 DEM/DOM terrain product chain。
- `reconstruct_pipeline_cli` 新增 `--mvs-depth-only`，可只跑 MVS 深度图估计并写出 depth/raw/confidence/mask/manifest，用于大航测数据的显存、取消和恢复验证，且不会进入融合、网格或 terrain 阶段。
- `README.md` 增加 Windows CUDA 固定构建入口、`libtorch-cu130` 运行时说明和四阶段重建链路状态摘要。

### 优化

- 后续 `v1.1.5` Release 正文优先从 `docs/releases/v1.1.5.md` 或同步内容生成，方便用户在 GitHub Release 页面直接看到版本更新内容。
- `docs/PROJECT_ARCHITECTURE.md` 已同步 MVS workspace manifest、MVS source planner、terrain product manifest、DEM mosaic、QC report 和 ReferenceTerrainPrior 模块边界。
- 四阶段计划的推荐里程碑统一改为 `v1.1.5-alpha.1`、`v1.1.5-alpha.2`、`v1.1.5-alpha.3` 和 `v1.1.5`，避免继续引用旧的 `v1.1.2` 目标版本。
- BA 参考地形约束使用 sigma、最大关联距离和 Huber delta 作为可诊断软约束，不默认把参考高程当作硬约束。
- 参考 DEM/LiDAR 质量报告不再只有摘要指标，报告 metadata 会记录配套 artifact 的绝对路径，方便 GUI 和后续报告窗口直接打开。
- 点云质量报告会记录 `cloud_alignment_method`，区分 `paired_similarity` 与 `nearest_neighbor_translation`，避免用户误以为所有点云都必须逐点配对。
- 稠密点云 GUI 现在暴露最小一致视图数、几何一致性过滤、最大重投影误差、孤立噪点面积阈值和融合最长边，并保持既有默认值不变。
- 深度图融合在新任务开始时会清空上一轮输出点和过滤深度缓存；如果用户在融合开始前或早期取消，不会把旧点云/旧过滤深度暴露给上层当作有效结果。

### 修复

- 修正仓库版本一致性检查仍固定到 `v1.1.4` 的问题，版本 hygiene 测试改为检查 `v1.1.5`。
- 修正发版流程示例仍使用旧版本号的问题，示例 tag 更新为 `v1.1.5`。
- 修正 SuperPoint 单测只用相对路径查找模型和测试影像，导致仓库根目录或 CTest 工作目录下找不到已存在模型的问题。
- 修正深度图融合收到取消时可能保留调用方复用输出数组里的旧点的问题，避免 GUI/CLI 取消后误读 stale fused points。
- 修正同一 MVS 帧写入过滤后深度产物时，空 `source_plan` 会覆盖初始深度产物已记录 source planning 依据的问题；manifest 更新现在会保留已有非空 `source_plan`。
- 修正 SfM 稀疏点云导出颜色采样在完整航测数据上按点反复读大图、或长期缓存过多彩色图像的问题；现在先按影像分组收集采样请求，每张影像只加载一次并批量填充点云颜色。

### 验证

- `scripts\build_win\build_windows_cuda.ps1 -Target test_gui_project_utils -RunTests -CTestRegex "ReferenceDataset|ReferenceDatasets|ReferenceQualityCheck|ReferenceTerrain|MainWindowTest\.ReferenceDatasetActionsConnectToProjectManager" -Jobs 8` 通过，12/12。
- `scripts\build_win\build_windows_cuda.ps1 -Target test_gui_project_utils -RunTests -CTestRegex "QualityReportComputesSameGridDemDifferenceMetrics|QualityReportComputesPairedPointCloudAlignmentMetrics" -Jobs 8` 通过，2/2，验证 DEM 差分栅格和点云配准 artifact 写盘。
- `scripts\build_win\build_windows_cuda.ps1 -Target test_gui_project_utils -RunTests -CTestRegex "AlignsUnpairedCloudsWithNearestNeighborTranslation|QualityReportAlignsUnpairedReferenceCloudByNearestNeighbor|PointCloudAlignment" -Jobs 8` 通过，4/4，验证 paired 与非配对最近邻点云报告链路。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "ReferenceDataset|ReferenceDatasets|ReferenceQualityCheck|ReferenceTerrain|ProjectReferenceTerrainBa|PointCloudAlignment|DemDifference|QualityReport" --output-on-failure` 通过，28/28。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -R "ProjectReferenceTerrainBa|ProjectReferenceDatasetsTest\.TerrainPriorPreflightReportsBundleAdjustReadiness|BundleAdjustServiceLidar|BundleAdjustLidar" --output-on-failure` 通过，10/10，验证 GUI 入口、服务层和核心 LiDAR 点到面约束链路。
- `scripts\build_win\build_windows_cuda.ps1 -Target test_mvs_pipeline -RunTests -CTestRegex "DepthMapFusionCancelBeforeWorkClearsStaleOutput" -Jobs 8` 先失败后通过，验证取消前已有旧 fused points 时会被清空。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsWorkspaceManifest|MvsSourcePlanner|MvsDepthPostprocess|MvsPipelineTest|DataTreeWidgetTest|BundleAdjustServiceLidar|LaserConstraint|ReferenceTerrain|ReferenceDataset|QualityReport|PointCloudAlignment|DemDifference" --output-on-failure` 通过，79/79。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsSourcePlanner|MvsDepthPostprocess|DenseCloudDialog|DepthMapFusion|MvsPipelineTest|DepthMapPersistence|DepthFrameUtils|DenseDepth|DataTreeWidgetTest\.ResultOnlyMetadataUpdateRefreshesDepthMapSection" --output-on-failure` 通过，35/35，验证 Stage 2 source planning、深度质量、融合取消和 GUI 高级 MVS 参数。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainDemDom|DemGridAggregator|DemMosaic|TerrainProductManifest|DemQualityRasters|DataTreeWidgetTest\.DemSectionShowsQualityRasterProducts" --output-on-failure` 通过，28/28，验证 Stage 3 DEM/DOM 产品链、质量栅格、mosaic、terrain manifest 和 GUI DEM 质量节点。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "Mvs|Depth|Fusion|Terrain|Dem|Dom|Sfm|Bundle|Quality|GuiProject|DataTree" --output-on-failure` 通过，209/209，验证四阶段相关 MVS、SFM/BA、terrain、QC 和 GUI metadata/data tree 回归可一起通过。
- `ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsWorkspaceManifest|MvsSourcePlanner|DepthMapPersistence|DepthFrameUtils|MvsPipelineTest" --output-on-failure` 通过，34/34；其中 `MvsWorkspaceManifest.CompletedFrameUpdatePreservesExistingSourcePlan` 先失败后通过，验证过滤后 depth 更新不再抹掉 `source_plan`。
- MUN-FRL 20 帧 real-data `bundle_adjust_cli --ab-compare --laser-cloud ... --fail-on-quality-gate` 通过，输出 `build/mun_frl_lidar_ba_ab_project_tf_velodyne/ba_ab_run_codex_20260620_003746`；LiDAR 点到面 RMS 从 1.0371 m 降到 0.9297 m，重投影 RMS 从 1.1258 px 降到 1.1154 px，quality gate 通过。
- agisoft aerial GCP 12 张航空影像受控 MVS/DEM/DOM smoke 通过，输出 `build/agisoft_aerial_mvs_dem_dom_codex_20260620_004632/pipeline`；12/12 注册，MVS 限 4 帧，生成 4 组 depth/raw/confidence/mask，融合 626,240 点、精化 606,131 点，并生成 `terrain/products/dem.tif` 与 `terrain/products/dom.png`。
- agisoft aerial GCP 12 张航空影像 0.5 scale 受控 MVS/DEM/DOM smoke 通过，输出 `build/agisoft_aerial_mvs_dem_dom_scale05_codex_20260620_010303/pipeline`；12/12 注册，SfM 1,259 点，平均重投影误差 0.8472 px，MVS 限 4 帧，生成 4 组 depth/raw/confidence/mask，融合 1,281,149 点、精化 1,224,829 点，并生成 8x8 `terrain/products/dem.tif` 与 `terrain/products/dom.png`，总耗时 322.390 s。
- agisoft aerial GCP 12 张航空影像 `--mvs-depth-only` smoke 通过，输出 `build/agisoft_aerial_mvs_depth_only_codex_20260620_012051/pipeline`；报告 `status=ok`、`stop_stage=mvs_depth`、`dense.status=depth_only`，生成 4 条 depth artifact 记录，没有写出融合点云、网格或 terrain 产品。
- agisoft aerial GCP 12 张航空影像 `--mvs-depth-only` source-plan 回归通过，输出 `build/agisoft_aerial_mvs_depth_only_12_codex_20260620_021000/pipeline`；12/12 注册，4/4 个 MVS manifest frame 完成，raw depth、confidence、valid mask 和 `source_plan` 均完整保留，报告 `status=ok`、`stop_stage=mvs_depth`。
- agisoft aerial GCP 完整 444 张输入的 post-fix `--mvs-depth-only --mvs-max-frames 8` 验证通过，输出 `build/agisoft_aerial_mvs_depth_only_444_postfix_codex_20260620_022500/pipeline`；444/444 注册，SfM 57,383 点，平均重投影误差约 0.87 px，8/8 个 MVS manifest frame 完成，raw depth、confidence、valid mask 和 `source_plan` 均完整保留，MVS 阶段 5.340 s，总流程 280.006 s。
- agisoft aerial GCP 完整 444 张输入的 `--mvs-depth-only --mvs-max-frames 50` 验证通过，输出 `build/agisoft_aerial_mvs_depth_only_50_codex_20260620_012819/pipeline`；444/444 注册，SfM 57,379 点，平均重投影误差 0.8661 px，50/50 个 MVS manifest frame 完成且无缺失 artifact，报告 `stop_stage=mvs_depth`，未进入融合、网格或 terrain 阶段。
- agisoft aerial GCP 完整 444 张输入的 `--mvs-depth-only --mvs-max-frames 150` 验证通过，输出 `build/agisoft_aerial_mvs_depth_only_150_codex_20260620_013608/pipeline`；444/444 注册，SfM 57,403 点，平均重投影误差 0.8661 px，150/150 个 MVS manifest frame 完成且无缺失 artifact，MVS 阶段 128.684 s，总流程 400.586 s，GPU gray cache usage 约 3.7-686.6 MB。
- agisoft aerial GCP 完整 444 张输入的全帧 `--mvs-depth-only --mvs-max-frames 444` 验证通过，输出 `build/agisoft_aerial_mvs_depth_only_444_batched_retry_codex_20260620_022910/pipeline`；命令退出码 0，报告 `status=ok`、`stop_stage=mvs_depth`，444/444 注册，SfM 57,362 点，平均重投影误差 0.8664 px，444/444 个 MVS manifest frame 完成且 `source_plan` 无缺失；MVS 目录写出 444 组 depth preview/raw depth/raw confidence/valid mask，MVS 阶段 149.543 s，总流程 420.977 s，GPU gray cache usage 最高约 2.03/6.00 GB，未进入融合、网格或 terrain 阶段。
- agisoft aerial GCP 完整 444 张输入的完整 MVS/mesh/DEM/DOM 长链验证通过，复用同一输出目录 `build/agisoft_aerial_mvs_depth_only_444_batched_retry_codex_20260620_022910/pipeline`；命令退出码 0，444/444 注册，SfM 57,362 点，MVS 444/444 帧完成，稠密点云 1,895,106 点、精化点云 1,847,925 点，生成 `model/products/textured_model.obj`、`terrain/products/dem.tif` 和 `terrain/products/dom.png`；总耗时 5686.447 s，其中 MVS 3416.110 s、mesh 2043.480 s、terrain 45.984 s。
- `scripts\build_win\build_windows_cuda.ps1 -Target test_gui_project_utils -RunTests -CTestRegex "ReferenceTerrain|ProjectReferenceTerrainBa|ProjectReferenceDatasets" -Jobs 8` 通过，10/10。
- `scripts\build_win\build_windows_cuda.ps1 -Target test_gui_project_utils -RunTests -CTestRegex "MvsWorkspace|MvsSourcePlanner|TerrainProductManifest|DemGridAggregator|DemMosaic|DemDifference|PointCloudAlignment|ReconstructionQualityReport|ReferenceTerrain|ProjectReferenceTerrainBa" -Jobs 8` 通过，33/33。
- `scripts\build_win\build_windows_cuda.ps1 -Target reconstruct_pipeline_cli -BuildOnly -Jobs 8` 通过。
- `scripts\build_win\build_windows_cuda.ps1 -Target plascan_gui -BuildOnly -Jobs 8` 通过。
- `python -m pytest tests\test_mvs_scheduler_config.py -q` 通过，72/72；验证 DenseCloudDialog 高级 MVS 参数会进入 `DepthGenConfig`，并覆盖 `--mvs-depth-only` 报告、跳过阶段语义和 SfM 稀疏导出批量颜色采样。
- `scripts\build_win\build_windows_cuda.ps1 -Target test_gui_project_utils -RunTests -CTestRegex "DenseCloudDialog|DenseCloudRefine" -Jobs 8` 通过，3/3。
- `python -m pytest tests\test_repo_hygiene.py::RepoHygieneTest::test_reconstruction_stage_docs_cover_new_pipeline_modules -q` 先失败后通过，验证 README、MVS/terrain README 与架构文档覆盖四阶段新增模块。
- `python -m pytest tests\test_repo_hygiene.py -q` 通过，9 项测试和 27 个 subtest 全部通过。
- 带 vcpkg、LibTorch 和 CUDA 运行时 PATH 的全量 CTest 通过，467/467；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 保持 disabled。
- `v1.1.5` annotated tag 已推送到 GitHub，解引用到 release commit `88d16f3ad7f4a844e0e225e8c6d11e8ab675e753`；GitHub Release 已创建为 `PlaScan v1.1.5`。
- GitHub Actions `CI / build-test` 已通过：release commit 的 main run `27847833504` 成功，tag run `27847913501` 成功。

### 已知问题

- agisoft aerial GCP 444 张完整 MVS/mesh/DEM/DOM 长链已通过，但 mesh/terrain 交接阶段观察到一次约 26.5 GB 的私有内存峰值；后续仍应把该峰值纳入内存自适应/分块网格化优化。
- agisoft aerial GCP 0.2 scale smoke 的法向量估计阶段出现过 `BLAS : Bad memory unallocation!` 非致命警告，进程最终 `status=ok` 且 DEM/DOM 写盘成功；0.5 scale smoke 本次未复现该警告，后续可单独跟踪 BLAS/法线估计释放路径。

## v1.1.4 - 2026-06-19

### 新增

- `reconstruct_pipeline_cli` 新增 MVS 调试与资源控制参数，支持限制深度估计帧数、MVS 分辨率、迭代次数、置信度阈值、GPU/CPU frame worker 数，以及融合阶段最大图像边长。
- MVS 报告扩展记录深度图产物、源影像、mask/raw depth/raw confidence、深度后处理统计、融合降采样参数和实际设备信息，便于定位 GUI/CLI 的密集重建问题。
- GUI 密集重建目录树消费项目 metadata，深度图、稠密点云等结果按文件名自然排序并随任务增量刷新。
- SfM 报告增加候选 pair plan、实际匹配图、连通分量、pending/failed/skipped pair、注册影像与质量统计，便于排查只注册少量影像的问题。

### 优化

- MVS 融合改为有界流式窗口，融合阶段可把 6000x4000 depth/confidence 下采样到默认最长边 2048，保留相机内参缩放和颜色采样一致性，显著降低大航测数据内存峰值。
- 对已完成置信度/局部离群过滤的深度图增加快速反投影融合路径，避免每个流式窗口再做昂贵的全量多视 BFS 检查，8 帧 aerial GCP 回归中单批融合约 0.7 秒。
- 流式融合加入点数阈值预聚合，点数接近内存上限前先做 voxel 预聚合，避免长时间运行到后半程才被系统杀死。
- MVS 深度图加载、预取、保存队列、取消检查和后处理日志继续收敛，GUI 点击取消后能更早中断排队保存和后续处理。
- 已知外参进入 BA 时改为 soft pose prior，外参不再默认完全固定；支持位置/旋转 sigma、Huber 和 LiDAR 质量权重参与 BA。
- GUI 特征匹配和空三流程复用更一致的 pair planning 逻辑，大项目优先使用序列窗口、相机中心邻域和已知重叠对，避免默认退回 N² 全匹配。

### 修复

- 修复深度图估计完成后目录树不显示深度图、结果顺序不稳定的问题。
- 修复 MVS 大数据融合阶段把所有深度图和颜色图长期常驻内存导致内存持续上涨的问题。
- 修复融合阶段下采样 depth 后颜色仍按原图尺寸采样，可能导致稠密点颜色错位或丢失的问题。
- 修复深度图局部红色孤立噪点缺少统计和过滤记录的问题，输出 confidence/local outlier 移除数量。
- 修复稀疏重建报告对候选匹配、实际匹配和几何验证失败混在一起导致问题难以定位的问题。
- 修复 Linux CI 在 Build 阶段执行 Qt GUI 测试 discovery 时没有 offscreen 环境，导致 `test_bundle_adjust_dialog_lidar` 构建失败的问题。

### 验证

- `python -m pytest tests\test_mvs_scheduler_config.py -q` 通过，69/69。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target reconstruct_pipeline_cli -Jobs 8` 通过。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -Jobs 8` 通过。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target plascan_gui -RunTests -CTestRegex 'Sfm|Feature|Match|Mvs|Gui|Lidar|Bundle' -Jobs 8` 通过，172/172；`PatchMatchCudaBenchmarkTest.CompareParallelAndLegacySweepAfterWarmup` 为 disabled。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Target test_bundle_adjust_dialog_lidar -RunTests -CTestRegex 'BundleAdjustDialogLidar|ProjectManagerBundleAdjustLidar' -Jobs 8` 通过，3/3。
- aerial GCP 8 帧 MVS 回归通过：`status=ok`，输出 `depth_0.png` 到 `depth_7.png`、`dense_cloud.ply` 1,240,093 点、`dense_cloud_refined.ply` 1,193,941 点；MVS 阶段约 429.8 秒，总流程约 998.0 秒。

### 已知问题

- 本版本验证了 aerial GCP 的 8 帧受控 MVS 回归，没有重新跑完 444 张完整 mesh/DEM/DOM 长链；完整长链仍建议作为后续夜间回归。
- MVS 后处理法向量估计在百万级点云上仍是明显耗时点，已不再导致本次回归崩溃，但后续可以继续做分块/可选法线输出优化。
- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`；该测试可能出现 `dom_png not found`。
- GitHub Actions `build-test` 是远端门禁；`v1.1.4` tag 推送后以 Actions 结果为准。

## v1.1.3 - 2026-06-18

### 新增

- 新增 LiDAR / 激光点点到面约束模块，支持读取带法线 PLY、曲率筛选、体素采样、最近平面查询，以及把 BA track 关联到 LiDAR 平面约束。
- 新增 `bundle_adjust_cli`，可对 `.plascan` 项目执行 headless BA，支持 `--laser-cloud`、LiDAR 关联距离/权重参数、`--ab-compare` 基线/激光约束 A-B 对比和 JSON/CSV/tsai 输出。
- 新增 `feature_match_cli` BA sidecar 输出，传统 BF/FLANN 匹配会同步生成 `.match.json`，包含匹配点坐标、特征索引和匹配分数，供多视 track 构建和 BA CLI 使用。
- 新增 MUN-FRL LiDAR BA 数据准备脚本、LiDAR 法线估计脚本、输入校验脚本和 A-B 结果比较脚本，支持从公开多传感器数据构造近期激光约束测试集。
- GUI 光束法平差对话框新增 LiDAR 约束参数，ProjectManager/BundleAdjustService 能把激光点云路径、关联半径、曲率阈值、权重和 Huber 阈值传入 BA 服务。

### 优化

- `prepare_mun_frl_lidar_ba_project.py` 支持 `--tf-static --camera-frame --body-frame`，会把 ROS 风格 `T_parent_child` 静态外参与 odometry 的 `world -> body` 位姿合成为 PlaScan 约定的 `world -> camera` 位姿。
- `compare_lidar_ba_ab_results.py` 增加固定口径评估，报告共同有效 track RMS、相机中心/旋转漂移、LiDAR 约束数量和 LiDAR 点到面 RMS/median，避免只看全局 RMS 误判效果。
- `testData/README.md` 补充 MUN-FRL lighthouse 从影像/匹配/轨迹/LiDAR 到 `.plascan`、法线估计和 BA A-B 对比的完整命令链。

### 修复

- 修复 headless BA CLI 默认导出评估图时可能依赖 GUI/图表环境的问题，默认关闭评估图导出，并提供 `--export-eval-plot` 显式开启。
- 修复 MUN-FRL `cloud_registered` 样例 PLY 虽有 normal 字段但法线全为 0 导致 LiDAR constraint map 无有效平面样本的问题，通过法线估计预处理生成可用点到面约束点云。

### 验证

- `python -m unittest tests.test_prepare_mun_frl_lidar_ba_project tests.test_compare_lidar_ba_ab_results tests.test_estimate_lidar_normals tests.test_bundle_adjust_cli tests.test_feature_match_cli_sidecar` 通过，11 个测试，其中未设置 CLI 环境变量时 2 个跳过。
- 设置 `PLASCAN_FEATURE_MATCH_CLI` 和 `PLASCAN_BUNDLE_ADJUST_CLI` 后，`python -m unittest tests.test_bundle_adjust_cli tests.test_feature_match_cli_sidecar tests.test_prepare_mun_frl_lidar_ba_project tests.test_compare_lidar_ba_ab_results` 通过，10/10。
- `python -m py_compile testData\prepare_mun_frl_lidar_ba_project.py testData\compare_lidar_ba_ab_results.py testData\estimate_lidar_normals.py` 通过。
- `ctest --test-dir build\windows-vcpkg-cuda-release --output-on-failure -R "lidar|Lidar|Laser|BundleAdjustCliTest|FeatureMatchCliSidecarTest|PrepareMunFrlLidarBaProjectTest|CompareLidarBaAbResultsTest|EstimateLidarNormalsTest"` 通过，19/19。
- MUN-FRL lighthouse 20 张图窗口生成 `mun_frl_lidar_ba_tf.plascan`，BA dry-run 输入为 `cameras=20 tracks=3689 sidecar_v2_pairs=230822 multiview_tracks=3689`。
- MUN-FRL A-B 诊断：`imu_link` body frame 的影像基线约 `0.533 px`，但 3-5 m LiDAR 约束会增大 LiDAR 点到面残差；`velodyne` body frame 可降低 LiDAR 残差但影像基线约 `1.126 px`，因此该样例目前作为 smoke/诊断集，不作为最终精度提升证明。

### 已知问题

- MUN-FRL `lio_body`、`imu_link`、`velodyne` 与相机 frame 的语义仍需进一步核实；当前 LiDAR BA 已能参与优化，但公开样例上的精度提升结论需要更可靠的跨传感器坐标链或更合适的数据切片。
- 本版本没有重新跑 agisoft aerial GCP 444 张完整 MVS/mesh/DEM/DOM 长时流水线。
- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`，该测试可能出现 `dom_png not found`。
- GitHub Actions `build-test` 作为远端门禁；`v1.1.3` 发布提交和 tag 推送后需以 Actions 结果为准。

## v1.1.2 - 2026-06-17

> 2026-06-18 补充：新增 MVS 深度图内存自适应保护、快速二进制深度产物保存和 GUI CUDA 帧流水线自适应，代码已进入 `main`；既有 `v1.1.2` tag 未强制重写。

### 新增

- 新增 LiDAR / 激光点摄影测量数据集整理文档，覆盖 MUN-FRL、UseGeo、H3D、MARS-LVIG、UAVScenes、NTU VIRAL、DublinCity、DFC 2019 / US3D、ETH3D 等候选数据，用于后续激光点参与 BA、相机-LiDAR 外参验证、MVS/DEM/DOM 质量检查。
- 扩展摄影测量测试数据下载器，增加 `workflow_tags` 和 `--workflow-tag` 选择能力，可按 `ba_constraint_candidate`、`lidar_fusion_validation` 等用途筛选 LiDAR 相邻数据集并生成手动下载 manifest。

### 优化

- 优化 MVS 深度图估计的 CPU 前处理链路：缓存源帧共视稀疏点、并行构建可见性缓存、源帧角度打分提前停止，减少大规模航测数据在进入 CUDA PatchMatch 前的等待时间。
- 优化稀疏 hint 构建：投影稀疏样本只做单次可见点遍历，深度分位采样有界，粗层 hint 使用 OpenCV distance transform 限距离传播，精层仅叠加 seed，避免全图传播把错误深度强行铺满。
- 优化 MVS 支撑掩码与 dense refine：稀疏支撑改为软约束，支撑掩码按精层 PatchMatch 工作尺寸生成，大点云 refine 对输入和内联过滤做上限保护，避免 GUI/CLI 在百万级点云上长时间阻塞。
- 优化 CUDA 灰度图处理路径，避免重复 reference resize，并保留 GPU 灰度图缓存命中统计，便于观察 GPU 利用率不连续时是否被上传/缩放拖慢。
- 优化 MVS 深度图缓存策略：根据系统物理内存和可用内存自动决定 full-res 深度图是否常驻内存，内存充足时保留空间换时间，内存不足或运行时压力升高时切换为流式保存并释放已缓存深度图。
- 优化深度图预览/中间结果保存队列，限制后台 full-res 深度帧积压数量，避免保存线程落后时继续放大内存峰值。
- 优化 MVS 深度图产物保存：原始 depth/confidence 从 OpenCV `.yml.gz` 改为 PlaScan 二进制 `.bin`，减少每帧 CPU 压缩/文本序列化耗时；当前版本只接受 `.bin` 深度帧作为正式缓存格式。
- 优化深度图预览图保存：预览 PNG 默认限制最长边 2048，避免每帧额外写入 6000x4000 全尺寸伪彩图拖慢 GPU 流水线。
- 优化 GUI 手动深度估计默认调度：线程数足够时自动使用 2 个 CUDA frame workers，让 GPU 帧任务和 CPU/IO 后处理更容易形成流水线。

### 修复

- 修复 MVS 后处理中过强稀疏支撑裁剪可能导致有效深度被硬清空的问题，改为置信度软缩放并保留深度。
- 修复深度图局部离群噪点缺少后处理保护的问题，增加局部深度离群过滤和阶段耗时统计，便于定位 `source/range/hint/patchmatch/filter` 瓶颈。
- 修复大规模 dense refine 连续执行第二轮 SOR 或内联大点云过滤导致的卡顿风险。
- 修复大规模 MVS 深度估计长期运行时无条件缓存每帧 `depth + confidence` full-res Mat，可能在软件崩溃前才暴露 OpenCV 内存分配失败的问题；现在会提前降级并给出日志/错误提示。

### 验证

- `python -m unittest tests.test_mvs_scheduler_config tests.test_reconstruct_pipeline_cli tests.test_repo_hygiene tests.test_download_photogrammetry_testdata` 通过，65 个测试，3 个跳过。
- `scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Jobs 8` 通过，使用 Windows 原生 MSVC/Ninja、CUDA 13.1、libtorch-cu130 和 `build\windows-vcpkg-cuda-release`。
- `build\windows-vcpkg-cuda-release\tests\test_mvs_pipeline.exe --gtest_brief=1` 通过，17/17。
- `build\windows-vcpkg-cuda-release\tests\test_mvs_types.exe --gtest_brief=1` 通过，16/16。
- `build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_brief=1` 通过，135/135。
- `python -m unittest tests.test_mvs_scheduler_config` 通过，48/48。
- `python -m unittest tests.test_mvs_scheduler_config tests.test_reconstruct_pipeline_cli tests.test_three_d_reconstruction_cli` 通过，61 个测试，3 个跳过。
- GitHub Actions `build-test` 作为远端门禁；`v1.1.2` 发布提交和 tag 推送后需以 Actions 结果为准。

### 已知问题

- 本轮重点为 MVS/CUDA 前处理与数据集准备，未重新运行 agisoft aerial GCP 444 张完整 MVS/mesh/DEM/DOM 长时流水线。
- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`；该测试可能出现 `dom_png not found`，后续应单独修复 Terrain/DEM/DOM 输出登记或测试数据。

## v1.1.1 - 2026-06-16

### 优化

- README 删除过时的“实测性能”表，重写环境构建说明，明确 Windows 原生构建、vcpkg preset、Python/LibTorch 环境脚本和 CPack 打包路径。
- 平台支持矩阵新增 Windows (NVIDIA)，同步说明 Windows/Linux/macOS 的 CUDA、CLI、Qt6 GUI 和打包支持范围。
- CI workflow 明确为 Linux CPU-only 构建，补齐 Qt OpenGLWidgets 与 Python numpy 依赖，并显式关闭 conda/vcpkg 自动发现，降低 GitHub Actions 环境漂移。

### 修复

- 修复 CPU LibTorch 构建下 `SFMService` 无条件引用 CUDA-only `c10::cuda::CUDACachingAllocator` 的兼容风险。
- 修复 GitHub Actions Configure 阶段因缺少 Qt OpenGLWidgets 开发包而失败的配置问题。

### 验证

- 使用 VS BuildTools CMake/Ninja，并设置 vcpkg、LibTorch、CUDA 运行时 `PATH` 后，`plascan_gui` 与 `test_gui_project_utils` 构建通过。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe` 通过，126/126。

### 已知问题

- CI 仍暂时排除历史已知失败 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory`；该测试可能出现 `dom_png not found`，需要后续单独修复。
- 本轮重点为 CI/README/版本同步，没有重新跑 agisoft aerial GCP 444 张完整 MVS/mesh/DEM/DOM 长时流水线。

## v1.1.0-alpha.4 - 2026-06-15

### 优化

- 优化 Metashape 相机转换，`k1/k2/k3/p1/p2` 畸变参数会写入 PlaScan `.tsai`，避免空三使用被简化为零畸变的相机模型。
- 优化 SFM 对项目元数据相机的处理：GUI 项目元数据只作为增量 SfM/BA 初值，只有显式 `.tsai` 列表才进入固定外参直接三角化。
- 优化已知外参 SfM 质量门，若输入存在多视 track 但输出几乎全退化为两视图点云，会标记失败并给出可定位的日志。

### 修复

- 修复 Metashape aerial GCP 数据经旧转换器导出零畸变相机后，444 张影像空三点云和飞行轨迹明显异常的问题。
- 修复正式 SFM 结果质量元数据中 BA 状态被无条件标记为已应用的问题。

### 验证

- `test_camera_format_converter.exe` 通过，6/6。
- `test_sfm_pipeline.exe` 通过，31/31。
- `test_gui_project_utils.exe` 通过，113/113。
- `python -m unittest tests.test_camera_convert_cli` 通过，6/6。
- agisoft aerial GCP 444 张，CUDA + `--feature-max-image-dim 2048`，SFM-only CLI 验证通过：444/444 注册，211037 个稀疏点，平均重投影误差 1.00 px，`finalTwoViewRatio=0.4191`。

### 已知问题

- 本轮完成的是 SFM-only 验证，MVS/mesh/DEM/DOM 下游仍需单独长时验证。
- 全量 `ctest` 的历史已知问题仍需单独确认：`TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 可能出现 `dom_png not found`。

## v1.1.0-alpha.3 - 2026-06-15

### 优化

- 优化 GUI 切换影像时的响应性，影像读取和特征覆盖层加载改为后台结果回填，避免旧任务覆盖当前画布。
- 优化大规模特征点可视化，批量点层使用单个场景图元绘制，默认特征点颜色调整为蓝色。
- 优化空三前置检查，复用已生成的影像配对计划，并用匹配图连通性/覆盖率判断是否可进入 SfM，避免把未计划的全连接影像对误判为缺失。
- 优化已知相机 SFM 候选配对，在大规模已知位姿数据上使用有界邻域和生成的 pair plan，避免退化到全量 N² 匹配。

### 修复

- 修复 Windows 下保存 `.plascan` 项目时，归档只读句柄阻塞 libzip 临时文件替换而导致 `Renaming temporary file failed: Permission denied` 的问题。
- 修复空三/匹配流程中进度、取消和 UI 线程阻塞相关路径，减少开始空三或检查匹配文件时界面卡死。
- 修复旧式两视图预览稀疏云被误当作正式 SfM 结果继续下游流程的问题，并增强稀疏结果质量元数据。

### 验证

- `cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target plascan_gui -j 8` 通过。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_project_data.exe` 通过，8/8。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe` 通过，113/113。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_sfm_pair_planner.exe` 通过，8/8。
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_layer_renderer_batched_overlay.exe` 通过，2/2。
- `ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R 'project_data|gui_project_utils|sfm_pair_planner|layer_renderer'` 匹配并通过 `test_layer_renderer_batched_overlay`，1/1。

### 已知问题

- 本轮完成的是 GUI/SfM 流程和保存路径的回归验证，agisoft aerial GCP 444 张全量端到端空三/重建仍需单独长时验证。
- 全量 `ctest` 的历史已知问题仍需单独确认：`TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 可能出现 `dom_png not found`。

## v1.1.0-alpha.2 - 2026-06-13

### 新增

- 增加 vcpkg manifest、CMake presets 和 `scripts/env/` 环境脚本，用于跨平台准备 vcpkg、Python、LibTorch 和 CPack 打包环境。
- 增加固定相机足迹重叠配对路径，SFM 在大规模已知相机数据上可优先使用相机投影足迹裁剪候选匹配对。

### 优化

- 已知相机 SFM 默认使用参考球面足迹重叠分析裁剪匹配对；不可用时再回退到顺序窗口和相机中心邻域。
- SFM 和 GUI 特征匹配对内点不足的日志做采样汇总，避免大量无效对刷屏。
- 相机三维视图在密集相机场景下按视口和相机数量限制标签数量，降低相机名称堆叠。
- Windows/CPack 打包和运行时安装规则得到补充。

### 修复

- 修复固定相机数据仅按相机中心选邻居导致错误匹配候选过多的问题。
- 修复大相机集合中相机名称显示过密的问题。

### 验证

- `cmake --build build -j$(nproc)` 通过。
- `ctest --test-dir build --output-on-failure -R 'SfmPairPlanner|OverlapAnalyzer|GuiProjectUtils|gui_project_utils'` 通过，8/8。
- `./tests/test_gui_project_utils` 通过，76/76。
- `python -m py_compile scripts/env/*.py` 通过。
- agisoft aerial GCP 48 张子集 SFM-only 验证通过：48/48 注册，7835 个稀疏点，平均重投影误差 0.75 px。

### 已知问题

- agisoft aerial GCP 全量 444 张完整重建仍需要更长时间的端到端验证；本轮只完成了 48 张子集的 SFM 验证。
- 全量 `ctest` 的历史已知问题仍需单独确认：`TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 可能出现 `dom_png not found`。

## v1.1.0-alpha.1 - 2026-06-13

### 摘要

- 改进三维重建工作流、相机转换、TorchScript 模型、GUI 对话框迁移、重叠对获取和摄影测量测试数据适配。

## v1.0.0 - 历史版本

### 摘要

- PlaScan 早期稳定基线版本。
