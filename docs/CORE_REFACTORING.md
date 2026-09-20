# Core 渐进重构

这轮已完成原计划的六类职责提取：影像/源计划/单帧估计、工件发布与恢复、
同步 MVS 服务、模型工作流分解、TSDF 阶段化及统一执行控制。
保持 GUI/CLI 入口、项目格式、质量阈值和 recovered 生产算法不变；
不整体搬迁顶层目录，也不把仍使用 QtCore 数据的算法声称为 Qt-free。

## 依赖与使用入口

| 职责 | 接口 / target | 边界 |
| --- | --- | --- |
| 深度帧、金字塔数据 | `DepthFrameResult.h`、`DepthPyramidTypes.h` / `mvs_contracts` | QtCore/OpenCV/相机数据；无生成器、后台调度 |
| 融合前同步处理 | `depth_processing/DepthPostprocessor.h` / `mvs_depth_processing` | 置信度、离群、小连通域及质量统计；不写工件 |
| 工作区存储与重放 | `DepthMatStorage.h`、`DepthArtifactIO.h`、manifest/replay、点云 IO / `mvs_storage` | 数据契约和通用 IO；不依赖计算后端或同步流程 |
| 深度后端、融合帧读取 | 原后端接口 / `mvs_backend` | 算法及融合适配；向下依赖存储，不依赖流程服务或 QtConcurrent |
| 同步深度流程 | `MvsPipelineService.h` / `mvs_pipeline` | 调用后端和私有阶段；由调用者选择线程 |
| Qt 异步深度任务 | `gui/project/tasks/DepthMapTask.h` / `gui_project` | GUI 专用信号、重复启动拒绝、取消、持有并 join future |
| 网格重建算法 | `DepthTsdfSurfaceBuilder.h` 等 / `meshing_algorithms` | 算法和 recovered 资源；不依赖模型工作流 |
| 模型业务编排 | `ModelWorkflowService.h` / `model_workflow` | 输入、参数、质量策略、取色/纹理及成果发布 |
| 通用控制 | `task_runtime/WorkflowExecution.h` / `task_runtime` | 纯 C++ 取消标志、归一化进度和成功/失败/取消结果 |

下层 target 不反向链接 `mvs`、`meshing` 或 QtConcurrent。
网格公开头文件使用的 PlaPoint 类型通过 PUBLIC 依赖明确提供，
不再依靠 GUI 或大测试目标偶然传递包含路径。

### MVS 阶段

`src/core/mvs/pipeline/` 内按实际职责拆出：

- 影像准备、掩膜、prepared raster 发布及资源策略。
- 稀疏可见性、源计划、深度范围、hint/seed、支撑域先验。
- 单帧估计、后端桥接、融合帧组装、批次/流式一致性、候选恢复。
- 保存队列、manifest/证据发布，以及 recovered 场景生产编排；实际矩阵/预览 IO 由独立存储库执行。

`mvs_storage` 在 2026-09-17 进一步拆出，持有 manifest/replay、二进制矩阵、预览及点云 IO/验证实现。
它不包含计算服务、后端或 pipeline 私有头文件，不向上链接 `mvs_backend`/`mvs_pipeline`。
矩阵 IO 保留既有函数名、命名空间和 40 字节文件头，`DepthFrameUtils.h` 通过包含最小存储头文件保持
现有消费者编译兼容，没有增加转发函数。持久化记录、缓存指纹/revision 和产物发布顺序不变。
保存队列和工作区发布事务仍由同步流程拥有；本轮没有把流程状态搬进存储库。

私有头文件只用于阶段之间共享实现类型和声明，不是新增公共算法 API。
旧生成器及其静态转发入口已删除，同步服务中的重复后处理转发也已删除。
新代码处理深度数据应包含数据或后处理头文件；执行流程应直接链接 mvs_pipeline。

`execute()` 同步执行，回调在执行线程调用；同一个服务拒绝重叠执行。
输入、配置、events 必须在执行前设置，执行期间不得修改或销毁服务。
取消可从另一线程请求。同步执行不隐式清除已置位的共享取消标志；
新任务通过显式 `resetCancellation()` 重置，Qt 适配器在启动 worker 前重置。
适配器在任务运行期间拒绝修改输入/配置/输出目录，析构先请求取消再等待 worker。

### 模型工作流

`src/core/mesh/workflow/` 分离了生成契约、参数解析、输入准入、轨道默认策略、
表面质量/降噪、取色/纹理阶段、成果发布，以及点云/深度/纹理入口。

深度产品入口仍只接受显式 `recovered_ooc`，需要 `recovered_model_input`，
不接受 recovered UV/OBJ 导出，也不回退旧算法。
原入口在恒真 recovered 分支后的旧 TSDF/Visual Hull/稀疏支架代码原本不可达；
现已删除 `LegacyDepthModelStages.cpp` 及其私有声明和链接入口。
源码契约测试改为约束当前 recovered 生产路径，不再检查已删除的不可达分支。
公开旧参数/质量策略函数仍保留，测试和显式算法验证入口继续可用。

### TSDF 阶段

`src/core/mesh/tsdf/` 分离读取、输入规划、布局、观测、恢复、
统计序列化、体素积分、支持域恢复和曲面后处理。
主流程仍依次执行积分 → 支持域恢复 → 等值面 → 清理 → 简化 → 最终质量检查。
短路返回保留原失败/取消顺序，不会继续写出失败阶段的结果。

私有阶段状态以引用传递既有数组及统计，不复制 TSDF/weight 等大体素缓冲；
OpenMP 积分计数保留在积分阶段局部，原 reduction 与运算顺序不变。
网格修复、补洞和 QEM 回退仍保留原拓扑/边界/包围盒门禁。
流式一致性事务、TSDF 支持域算法、统计字段表等高内聚文件仍可能超过
400 行；没有为满足行数而拆散事务或更改算法。

### 统一控制与未完成迁移

四种模型请求只保留 `execution`：共享取消标志或外部任务取消检查、UTF-8 stage 和
0..1 ratio。GUI/CLI 调用方已迁移，旧公开 `isCancelled`/`progress` 字段及双回调桥接已删除。
私有 `bindWorkflowCallbacks` 将唯一控制转换为现有算法需要的回调；子请求传递相同的控制，
不会重复报告进度。非有限/越界 ratio 归一化。
TSDF options 也已只保留 `execution`；不再提供旧取消/进度回调与共享控制并存的双接口。
结果适配通过明确布尔取消字段分类，不依赖中文错误字符串猜测取消。
这提供协作取消，不宣称暂停、checkpoint 恢复或任意算法瞬时中断能力。
调用方回调应快速返回，不抛异常、不在执行回调内销毁服务。

## 保持的行为契约

- Accepted 帧才能参与融合；ValidationOnly 可作为一致性来源；失败帧不能由准入枚举单独提升。
- streaming 释放大像素数据但保留支撑、相机、证据身份和诊断；终态释放也清理支撑。
  共享读取者的数据不被强制失效，两种释放均可重复调用。
- 稀疏支撑只降低域外置信度，不删除深度；低置信保留仍要求真实几何证据。
- 后处理维持置信度 → 局部离群 → 小连通域顺序、像素域缩放和移除比例回退。
- 几何来源位序、工件身份、manifest 审计、准入和 recovered 保存/通知顺序不变。

边界基线显式迁移 target 名称及提取后的 QtCore 声明。
扫描器把实现私有头文件也按 header 计数，因此基线条目增加不等于新增 GUI 依赖。
Qt Widgets / GUI include 和反向流程依赖禁令未放宽；
WorkflowExecution 保持纯 C++，阶段仍有 QtCore 数据和内部 IO。

## 验证

2026-09-17 存储提取已在 Linux/GCC 的现有 `build/linux-source-release`（CUDA/OpenCL 已启用）验证：

- `cmake --build build/linux-source-release --target mvs_pipeline gui_project mvs_depth_reprocess_cli
  reconstruct_pipeline_cli test_mvs_storage test_mvs_workspace_manifest test_mvs_pipeline_service
  test_mvs_pipeline test_mvs_dense_cloud_refinement test_source_contracts -j 8` 通过；
  `test_workflow_cli` 与 `test_gui_project_utils` 另行构建通过。
- 独立存储 6/6、工作区 43/43、点云细化/IO 10/10、同步服务 4/4、MVS 169/169、源码契约 75/75、
  CLI 52/52、GUI 深度数据/存储/生命周期定向 29/29 通过。
- `python scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
  -R '^(MvsStorageContract\.|CoreBoundaryContractTest$)'` 7/7 通过，无跳过。
- Python 边界单测 18/18、`py_compile tests/test_core_boundaries.py`、边界扫描（711 条冻结项、0 禁止依赖）、
  新文件 clang-format 检查及 `git diff --check` 通过；两个 CLI `--help` 通过。
- 矩阵/预览/路径函数体与提取前实现比较，仅格式不同；独立测试实际链接不含 `mvs_backend`、
  `mvs_pipeline` 或 Qt Concurrent/Widgets/Gui。存储保留 QtCore 数据表示，并非 Qt-free。

本轮没有重新运行完整真实场景重建，也没有可用的 Windows/MSVC 验证环境；历史 MSVC 结果仅对应
下述早期阶段，不能用于声称本轮改动已通过 Windows 验证。

新增独立行为测试只链接对应同步库，不依赖 Qt 生成器：

- WorkflowExecutionContract：进度归一化、取消合并、结果/错误定位。
- MvsPipelineServiceContract：同线程执行、完成一次、共享预取消、重复执行拒绝、显式重置。
- ModelWorkflowExecutionContract / TsdfWorkflowExecutionContract：
  各入口取消、外部任务取消、嵌套进度只报告一次、输入失败与成功结果契约。
- 原数据/后处理、深度工件重放、MVS、模型/TSDF、GUI 和仿真测试继续保留。
- MvsStorageContract 独立链接 `mvs_storage`，验证确定性文件头与旧 padding、非连续证据矩阵、
  截断读取和写盘失败、预览无效像素、manifest 往返/有序重放/缺失蒙版、点云写入与完整性验证。

`tests/CoreImplementationBundles.h` 只为原有源码结构契约指定拆分后的显式文件集；
不是行为测试替身。边界测试另检查实际 CMake 依赖、纯 C++ 控制、
TSDF 后处理顺序和非拥有数组引用，以及产品不调用旧验证分支。

原生 Windows 验证入口：

```powershell
cmake --build --preset windows-source-release --parallel 8
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_core_boundaries.py'
.venv/Scripts/python.exe scripts/validation/check_core_boundaries.py
git diff --check
```

Linux/GCC、macOS/Apple Clang 必须在各自环境验证，不能从 MSVC 结果推定。
本轮原生 MSVC Release 完整构建及全量 CTest 已验证：3159 项登记，3141 项通过、
17 项跳过、1 项禁用基准，0 失败。跳过包括 11 项未配置 MC33 的关联测试、
3 项外部 Metashape/OBJ/真实网格夹具、1 项可选 TLS 网络验证及 2 项 offscreen 菜单交互。
6 个相关 Python 模块合计 64 个单测通过；边界扫描 728 条审查冻结项、0 禁止依赖。
CI YAML 解析及 12 个 build-test shell step 的 bash -n 通过；远端 CI 尚未执行。
以上数量记录阶段提取后的验证；兼容清理后的最终验证与剩余项见
[`COMPATIBILITY_CLEANUP.md`](COMPATIBILITY_CLEANUP.md)。本次按用户要求推送后不跟踪 CI。
粗/中/精三维物体与地形仿真生成/评分测试保留；源码重构与完整真实重建精度验证
是不同门禁，不能把评分脚本单测说成全部重建链路已完成端到端实测。

## CI Python 运行时修复

Linux build-test 在统一 .venv 安装 NumPy/Pillow/SciPy/rasterio/trimesh，再从固定 OpenCV 5.0.0
submodule 构建绑定并安装到同一环境，不混用系统 OpenCV 4 或不同 wheel。
安装 python3-dev，显式提供 Python/NumPy/site 路径；在主构建前检查 prefix、
5.0.0 版本、PNG/TIFF 内存编解码。headless/GUI 配置指定相同 Python3_EXECUTABLE。
Windows 独立 run_tests.py 在没有显式 PROJ_DATA/PROJ_LIB 时，从实际构建 CMakeCache 的
vcpkg 安装目录及 triplet 绑定存在的 proj.db，保持与构建 PROJ DLL 配套；不硬编码开发机路径。
远端 CI 状态只能依据发布这些改动后的 GitHub Actions 运行结果。


## 2026-09-17 无调用代码清理

生产 `mvs_pipeline` 删除无调用的旧单帧深度、源计划缓存、驻留/流式一致性、残差恢复、
学习候选和位姿候选阶段，以及其专用一致性投票、保存队列、稀疏范围与融合帧组装私有包装。
发布方直接使用 `DepthFrameResult` 的源计划和诊断，不再读取不会填充的旧缓存。
CUDA-only recovered 调度删除不可达 OpenCL/异构分支；设备不可用时明确失败，取消和工件发布接口保留。

删除没有生产/测试消费者的 `OrbitalSparseScaffoldSurfaceBuilder`；下层独立算法保留。
DEM mosaic/产品清单、RPC 控制点偏差估计、处理基线管理拆为三个 `EXCLUDE_FROM_ALL` 静态库：
`terrain_utilities`、`camera_rpc_adjustment`、`qc_baseline`。它们不再进入生产库依赖，测试显式链接。
头文件和函数接口不变，外部使用者需要链接新目标。TSDF 帧加载仍用于纹理流程，因此未整体删除 TSDF。

源码契约删除针对旧生产阶段的存在性检查，保留独立稀疏投影/后端及 recovered 契约。
Python 边界测试检查可选库不进入产品依赖、旧私有入口与 OpenCL 调度不返回。

本轮 Linux/GCC 验证使用现有 `build/linux-source-release`（CUDA/OpenCL 已启用）：

- `cmake --build build/linux-source-release --target mvs_pipeline model_workflow plascan_gui
  mvs_depth_reprocess_cli reconstruct_pipeline_cli test_mvs_pipeline test_mvs_pipeline_service
  test_source_contracts test_gui_project_utils test_workflow_cli test_mesh_reconstructor
  test_dem_mosaic test_terrain_product_manifest test_processing_baseline_manager test_camera_unit
  test_sparse_orbital_scaffold_builder test_screened_poisson_surface_builder
  test_mesh_voxel_topology_repair --parallel 4`：分批执行并全部通过。
- 直接运行 `test_mvs_pipeline` 169/169、`test_source_contracts` 74/74、`test_workflow_cli` 52/52 通过。
  CLI 旧源码契约不再要求不存在的私有阶段；配置字段和独立算法分别检查，不能据此声称旧实验
  配置已接入 recovered 的固定选源/投票流程。
- `python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
  -R '^(DemMosaic\.|TerrainProductManifest\.|ProcessingBaselineManagerTest\.|RpcCameraModel|CoreBoundaryContractTest$)'`
  21/21 通过；RPC 全部相关套件与同步服务另行筛选运行，共 18/18（同步服务 6 项）通过。
- GUI offscreen 定向筛选私有成员、地形清单、工作区、深度发布、金字塔、异步生命周期和密集配置，
  17/17 通过；网格入口/帧读取等 `DepthMapMeshBuilderTest.*` 15/15 通过。
- 稀疏载体下层算法、Screened Poisson 和体素拓扑修复筛选 15 项：14 通过，
  `MeshVoxelTopologyRepairTest.FallsBackWhenSmoothingErasesSubVoxelBody` 因未配置 MC33 跳过。
  `RepairsOptInRealMesh` 未提供真实网格夹具，明确排除；未运行真实场景重建。
- Python 边界单测 20/20、`python3 -m py_compile tests/test_core_boundaries.py`、
  `python3 scripts/validation/check_core_boundaries.py`（709 条冻结项、0 禁止依赖）、
  修改代码 clang-format 检查和 `git diff --check` 通过。
- 生成的 Ninja 链接依赖确认 GUI 与两个主要 CLI 不包含三个新可选库。

未执行 Windows/MSVC 验证、commit、push 或版本操作；本轮临时编译/测试日志已清理。

## 2026-09-17 Qt 呈现与项目恢复绑定拆分

将标靶排版/PDF 写入和全球地形报告绘制移到 `src/adapters/qt`，分别由
`marker_print_qt`、`terrain_report_qt` 提供，GUI、CLI 和相关测试显式链接。
原头文件 include 路径保留，但外部消费者需要链接新目标；core 生产目标禁止反向链接适配库。
`terrain` 不再直接链接 QtGui。`control_points` 的检测接口仍使用 QImage，因此仍依赖 QtGui。

小天体成果生成通过 `SmallBodyPreviewWriter` 接收呈现能力。请求预览却未提供写入器时，
在创建输出目录前明确失败；关闭预览不需要写入器。回调在原成果事务内写入临时路径，
失败或取消不会发布任何成果。报告绘制和打印实现的 C++ token 与迁移前一致，
字体及 QGuiApplication 判断仍留在 Qt 适配库，本轮未改变绘制行为。

项目资源清理保留同步恢复操作，将打开前预检、QObject 连接和重复安装属性迁入
`project_recovery_qt` 的 `ProjectResourceRecoveryBinding::install()`。
原 `ProjectResourceCleanupService::installAutomaticRecovery()` 已移除，ProjectManager 与恢复测试
改用适配层入口；连接仍随 ProjectData 生命周期管理。

本轮 Linux/GCC 验证使用现有 `build/linux-source-release`：

- `cmake --build build/linux-source-release --target plascan_gui marker_print_cli marker_detect_cli
  small_body_terrain_cli test_small_body_global_products test_marker_pdf_writer test_apriltag_detector
  test_project_resource_cleanup test_source_contracts test_gui_project_utils --parallel 4` 通过。
- `python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
  -R '^(SmallBody|MarkerPdfWriterTest\.|AprilTagDetectorTest\.|ProjectResourceCleanupTest\.|ProjectResourceRecoveryBindingTest\.|CoreBoundaryContractTest$)'`
  43/43 通过，无跳过；包含预览缺失、失败回滚、取消回滚和重复安装连接数量验证。
- `test_gui_project_utils` offscreen 定向运行资源清理、地形源码约束、标靶菜单、
  异步生命周期、打开响应和任务生命周期测试，14/14 通过；`test_source_contracts` 74/74 通过。
- Python 边界单测 21/21、仓库规范单测 23/23、`py_compile tests/test_core_boundaries.py`、
  边界扫描（696 条冻结项、0 禁止依赖）、修改代码 clang-format 检查和 `git diff --check` 通过。
- 同一合成彩色八面体的迁移前后 CLI 产物比较：报告 PNG 与六个 GeoTIFF 逐字节一致；
  JSON 去除生成时间并规范化输出根目录后相同。

未运行全量测试、真实场景完整重建或 Windows/MSVC 验证。本轮临时产物已清理，
未执行 commit、push 或版本操作。

## 2026-09-18 标靶图像接口与 CPU headless 配置

`MarkerDetector`、AprilTag 和非编码检测改为接收二维 `CV_8UC1` 灰度 `cv::Mat`，
角点由 `QPolygonF` 改为 `QVector<QPointF>`，`control_points` 删除 QtGui 链接。
检测同步借用输入存储，支持 ROI/行填充，非法类型或蒙版尺寸抛出明确错误。
新增 `MarkerImageInput.cpp` 共享输入校验；AprilTag 的行跨度在转为第三方 int32_t 前检查范围。
QtCore 的点、容器、字符串和项目 JSON 类型仍保留。

新增 `marker_detection_qt` 和 `xjw::app::markers::detectMarkers()`。
QImage 应用调用方迁移到此入口并显式链接适配库；直接调用 core 的外部消费者需要自行提供灰度矩阵。
适配器保留 Qt Grayscale8 转换和 `qGray(mask.pixel())` 蒙版语义，不用另一种色彩权重替代。
常见 Grayscale8 图像/蒙版借用 Qt 存储，避免额外整图复制；其它蒙版转换为独立灰度矩阵。
图像和转换存储在同步调用结束前保持有效，检测不保留输入引用。

新增 `PLASCAN_BUILD_QT_PRESENTATION` 开关，关闭时不注册 PDF/地形报告库、打印 CLI 和对应呈现测试；
桌面 GUI 要求开启此选项。Linux、Windows 和 macOS 均登记 CPU `source-headless-release` preset，
关闭 GUI、GUI 测试、呈现、CUDA、OpenCL 和 TensorRT，复用既有源码依赖安装。
无呈现构建的小天体 CLI 默认输出 GeoTIFF/JSON，显式 `--preview` 明确失败且不创建输出目录。
呈现开启时默认 PNG 行为保留；`--no-preview` 仅初始化 QCoreApplication，不加载图形平台插件。
产品和预览事务失败/取消测试在 headless 中保留，实际打印/PNG 测试在呈现开启时执行。

本轮 Linux/GCC 验证：

- 通过 `python3 scripts/env/configure_with_env.py --preset linux-source-headless-release
  --no-tensorrt-auto-install` 配置 headless；源码依赖已存在，本次没有重建依赖。
- 使用 `cmake --build build/<preset> --target ... --parallel 4`（headless 为 2）分批构建通过。
  桌面包括 `plascan_gui`、两个标靶 CLI、小天体 CLI、control_points 全部模块测试、
  `test_source_contracts`、`test_gui_project_utils`、`test_reference_models`、
  `test_marker_task_runner` 和 `test_marker_detection_job_builder`；
  headless 包括 `marker_detect_cli`、`small_body_terrain_cli`、`test_marker_detector_input`、
  `test_marker_image_adapter`、`test_non_coded_target_detector`、`test_marker_detect_cli`、
  `test_small_body_global_products`。
- `python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
  -R '^(Marker|AprilTagDetectorTest\.|NonCodedTargetDetectorTest\.|Detection|ControlNetwork|CoordinateReference|SmallBody|CoreBoundaryContractTest$)'`
  84/84 通过，无跳过；涵盖 GUI 面板、审查、画布、任务进度/取消、Qt 图像转换和 core 输入契约。
- `python3 scripts/env/run_tests.py --test-dir build/linux-source-headless-release --output-on-failure
  -R '^(MarkerDetectorInputTest\.|MarkerImageAdapterTest\.|NonCodedTargetDetectorTest\.|MarkerDetectCliTest\.|SmallBody|CoreBoundaryContractTest$)'`
  25/25 通过，无跳过；`test_source_contracts` 74/74 通过。
- Python 边界单测 23/23、仓库规范单测 23/23、`py_compile tests/test_core_boundaries.py`、
  边界扫描（689 条冻结项、0 禁止依赖）、修改代码 clang-format 检查和 `git diff --check` 通过。
- headless Ninja 目标图不包含桌面程序、Qt Widgets/GuiPrivate、PDF/报告库或打印 CLI；
  `readelf -d` 确认纯检测测试与小天体 headless CLI 的 Qt 依赖只有 QtCore。
- 合成圆形标靶的迁移前、桌面与 headless 检测 JSON 相同。
  在清空 DISPLAY/WAYLAND_DISPLAY、指定不存在的 Qt 平台插件后，桌面 `--no-preview` 和
  headless CLI 均成功输出；六个 GeoTIFF 逐字节一致，报告 JSON 去除生成时间并规范化输出根目录后相同。

headless 整套默认目标和全量测试未运行；只验证上述受影响目标。Windows/MSVC 和 macOS preset 尚未实测，
未运行完整真实场景重建。SfM 着色、空三图像读取、网格纹理和应用层标靶输入仍需要公开 QtGui，
因此 headless 不是无 Qt 的 engine。本轮临时产物已清理，正式 headless preset 构建树保留供复用；
未执行 commit、push 或版本操作。

## 2026-09-18 SfM 稀疏点云着色移除 QtGui

TriangulationService 改用公共 ImageIO 的八位三通道 BGR cv::Mat 读取/缓存影像，按 RGB 写入 PLY；
sfm_project 删除 PUBLIC QtGui 链接。SfM 源码不再使用 QImage/QRgb，项目适配仍保留 QtCore。
几何过滤、原始坐标的 qRound/边界夹取和多视整数平均保持不变。
读取显式设置 IMREAD_IGNORE_ORIENTATION，避免 EXIF 旋转改变匹配坐标对应的像素。

成功和失败读取均按相机缓存，只尝试一次。不可读视图跳过颜色采样，并在日志及结果 JSON 的
color_read_failures 数组中记录 image_path/error；没有任何有效颜色的点继续导出中性灰几何，
计入 uncolored_point_count。上述诊断字段仅在出现对应问题时写入，正常成功 JSON 不增加字段。
TIFF 沿用公共 GDAL/UInt16 转换约定；修复 ImageIO 对 COLOR 与 IGNORE_ORIENTATION 组合标志的判断，
保证单波段 TIFF 请求彩色时仍扩展为三通道 BGR。

新增 sfm/test/test_triangulation_service.cpp 独立测试，显式链接匹配、图像 IO 和 SfM 项目能力，
不链接 QtGui；公共 ImageIO 增加组合标志及 EXIF 方向测试。

本轮 Linux/GCC 验证使用现有桌面与 CPU headless 构建树：

- `cmake --build build/linux-source-release --target sfm_project plascan_gui test_triangulation_service
  test_common_image_io test_ba_input_builder test_ba_track_builder test_initial_sparse_triangulator
  test_sfm_quality_report test_source_contracts test_gui_project_utils --parallel 4` 通过。
- `cmake --build build/linux-source-headless-release --target sfm_project test_triangulation_service
  test_common_image_io test_ba_input_builder test_ba_track_builder test_initial_sparse_triangulator
  test_sfm_quality_report --parallel 2` 通过。
- `python3 scripts/env/run_tests.py --test-dir build/<preset> --output-on-failure
  -R '^(TriangulationColorTest\.|ImageIOTest\.|BaInputBuilder|BaTrackBuilderTest\.|ProjectMatchInputReaderTest\.|InitialSparsePointCloudTriangulatorTest\.|SfmQualityReportTest\.|CoreBoundaryContractTest$)'`
  在上述两棵构建树分别运行，均 28/28 通过，无跳过。
- GUI offscreen 筛选 `TriangulationServiceTest.*:ProjectTriangulationUiTest.*:CodeStyleTest.*Triangulation*`，
  实际运行 3/3 通过；test_source_contracts 74/74 通过。
- 临时编译迁移前 TriangulationService 实现，与当前实现分别运行彩色 PNG 半像素/多视平均及边界夹取夹具，
  两组 PLY 逐字节一致；此证据仅对应这些无损夹具，不代表所有图像解码器/格式都逐字节相同。
- readelf 确认桌面与 headless 的独立三角化测试只有 QtCore Qt 依赖，无 QtGui；
  Python 边界单测 24/24、仓库规范单测 23/23、py_compile tests/test_core_boundaries.py、
  边界扫描（688 条冻结项、0 禁止依赖）、修改行 clang-format 检查和 git diff --check 通过。

未运行全量测试、真实场景完整重建或 Windows/MSVC/macOS 验证。
剩余直接 QtGui 职责为网格纹理、空三图像读取和应用层标靶转换；headless 仍需要 QtGui。
本轮临时产物已清理，未执行 commit、push 或版本操作。

## 2026-09-18 空三移除 QtGui 图像读取

空三生产库移除私有 QtGui 链接，显式链接公共图像 IO 和日志能力；生产源码不再使用
QImageReader、QImage 或 QColor。QtCore 路径、容器、QSize、JSON 等值类型继续保留。

- 新增 common/io/ImageSizeReader.cpp，在 ImageIO.h 暴露 readImageSize()。通过 GDAL
  栅格头获取原始宽高，BMP 用固定 26 字节头回退，不解码整幅影像、不应用 EXIF 旋转。
  支持范围以启用的 GDAL 栅格驱动及 BMP 为准；失败返回无效 QSize，可选错误参数包含路径与原因。
- SfmAttemptRunner 输入尺寸、Pipeline 批次 EXIF 焦距先验及 QualityReportWriter 覆盖率尺寸
  统一调用该接口。质量报告仍在不可读时回退到关键点范围。EXIF 解析继续使用公共 IO。
- 针孔结果导出与 RPC 点云颜色改用公共 ImageIO 的 CV_8UC3 BGR 图像，转成 RGB 输出，
  显式忽略 EXIF 旋转。保留首个有效轨迹观测、qRound、边界裁剪和默认灰色几何；
  针孔路径按影像分组读取，RPC 缓存成功与失败，读取错误记录影像路径与原因。
  16 位 TIFF 沿用公共 GDAL 的 1/256 转换约定，不声称所有格式与 Qt 解码器逐字节相同。
- RPC 着色回归暴露并修复 imageCoordinate() 的已有悬空引用：
  QMap::value() 返回临时 std::vector，其 at() 元素不能跨语句引用；改用 constFind() 引用原容器。
  此函数也用于 RPC 几何解算，因此同时重跑 RPC 几何与颜色校验。
- 新增公共尺寸和独立空三颜色测试；原有空三测试夹具改用公共 ImageIO，显式声明所需链接，
  新增及本次调整的图像测试临时产物位于 build/tmp/aerial-image-io-tests/，由 QTemporaryDir 清理。

本轮 Linux/GCC 验证：

```bash
cmake --build build/linux-source-release --target plascan_gui \
  test_aerial_image_io test_image_size_reader test_common_image_io test_sfm_attempt_runner \
  test_aerial_triangulation_pipeline test_aerial_triangulation_result_writer \
  test_rpc_aerial_triangulation_runner test_sfm_pair_planner test_sfm_search_policy \
  test_reconstruction_prerequisites test_match_result_catalog test_aerial_triangulation_workflow \
  test_tie_point_preparation test_camera_intrinsic_prior_sanitizer test_adaptive_focal_search \
  test_triangulation_service test_source_contracts test_gui_project_utils --parallel 4

cmake --build build/linux-source-headless-release --target aerial_triangulation \
  test_aerial_image_io test_image_size_reader test_common_image_io test_sfm_attempt_runner \
  test_aerial_triangulation_pipeline test_aerial_triangulation_result_writer \
  test_rpc_aerial_triangulation_runner test_sfm_pair_planner test_sfm_search_policy \
  test_reconstruction_prerequisites test_match_result_catalog test_aerial_triangulation_workflow \
  test_tie_point_preparation test_camera_intrinsic_prior_sanitizer test_adaptive_focal_search \
  test_triangulation_service --parallel 4

python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure -R '^(AdaptiveFocalSearchTest\.|AerialTriangulation|CameraIntrinsicPriorSanitizerTest\.|MatchResultCatalogTest\.|ReconstructionPrerequisiteReportTest\.|RpcAerialTriangulationRunnerTest\.|SfmAttemptRunnerTest\.|SfmGuidedMatchPlannerTest\.|SfmMatchDiagnosticsTest\.|SfmPairPlannerTest\.|SfmSearchPolicyTest\.|TiePointPreparationTest\.|RasterInputs/AerialImageColorTest\.|ImageIOTest\.|ImageSizeReaderTest\.|TriangulationColorTest\.|CoreBoundaryContractTest$)'
```

上述桌面构建与 156/156 定向 CTest 通过；同一测试筛选在 CPU headless 构建树为 135/135 通过，
两者均无跳过。桌面结果额外包含空三 GUI 相关发现测试。颜色测试覆盖 RGB 顺序、半像素取整、
实际图像边界、16 位单波段 TIFF、缺失及损坏影像保留几何；尺寸测试覆盖 PNG/JPEG/BMP/TIFF、
Unicode 路径、错误路径、BMP 顶部向下存储和 100000×80000 稀疏 TIFF 的头读取。
最初发现当前 GDAL 未启用 BMP 驱动，补充 BMP 头部回退后重新通过。
RPC 使用仓库影像及人工构造观测回归，按 sidecar 观测索引逐点核对 PLY 颜色，
不代表真实连接点的完整场景重建。

- QT_QPA_PLATFORM=offscreen 的 test_gui_project_utils 筛选
  AerialTriangulation*:TriangulationServiceTest.*:ProjectTriangulationUiTest.*:CodeStyleTest.*Triangulation*，
  实际 23/23 通过；三角化灰色输出夹具的缺失影像警告符合预期。test_source_contracts 74/74 通过。
- 桌面和 headless 的 test_aerial_image_io、test_rpc_aerial_triangulation_runner 经 readelf -d 检查，
  Qt NEEDED 项均只有 QtCore，无 QtGui。
- Python 边界单测 25/25、仓库规范单测 23/23、py_compile tests/test_core_boundaries.py、
  边界扫描（687 条冻结项、0 禁止依赖）、新增 C++ clang-format 检查及 git diff --check 通过。

未运行整个项目全量测试、真实连接点完整重建、Windows/MSVC 或 macOS 验证。
剩余直接 QtGui 职责集中在网格纹理和应用层标靶转换；headless 仍需要 QtGui。
本轮临时产物已清理，未执行 commit、push 或版本操作。

## 2026-09-18：空三匹配诊断与焦距搜索容器标准化

`reconstruction/SfmMatchDiagnostics.h` 的匹配图统计、负缓存分类和已知位姿引导匹配规划改用
`std::vector`、`std::set`、`std::map`、`std::pair` 与 `std::string`。`search/AdaptiveFocalSearch`
的候选列表也改用 `std::vector`。这些接口只表达纯算法数据，不需要 Qt 容器、QString 或 Qt JSON；
流水线在写入现有 QJson 诊断和工程日志时才做边界转换。排序、候选优先级、连通分量统计和
最大候选数语义保持不变。

边界基线从 685 项降为 669 项，移除了上述两个公开头文件的 Qt include/symbol 债务。
Linux/GCC 定向验证：

```bash
cmake --build build/linux-source-headless-release --target test_sfm_pair_planner \
  test_adaptive_focal_search test_aerial_triangulation_pipeline --parallel 4
python3 scripts/env/run_tests.py --test-dir build/linux-source-headless-release --output-on-failure \
  -R 'SfmPairPlanner|AdaptiveFocalSearch|AerialTriangulationPipeline'
cmake --build build/linux-source-release --target test_sfm_pair_planner \
  test_adaptive_focal_search test_aerial_triangulation_pipeline test_gui_project_utils --parallel 4
python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure \
  -R 'SfmPairPlanner|AdaptiveFocalSearch|AerialTriangulationPipeline|GuiProjectUtils'
python3 scripts/validation/check_core_boundaries.py
```

上述 headless 和桌面定向测试各 34 项通过。未执行 Windows/MSVC、macOS 或全量测试，
也未执行真实大规模连接点重建；工程 DTO、项目 JSON、匹配准备和质量报告仍是 QtCore 适配边界。

## 2026-09-18：PlaScan 项目文件系统布局模块

新增 `common/plafs/` 和 `plascan_common_plafs`，将文件系统职责分为四个小对象：

- `PlaFile` 只封装物理文件路径、存在性和原子字节读写；
- `PlaDir` 只封装目录存在性、创建、空目录检查和安全的非递归成员检查，不提供递归删除；
- `PlaChunkLayout` 只计算当前 Chunk 的 `assets`、匹配、控制点、蒙版、相机参考、恢复缓存、导入/打包资源和结果路径；
- `PlaProjectLayout` 只计算 `.plascan`、`.files`、共享目录和数字 Chunk 的项目级路径。

模块使用 `std::filesystem::path`，不依赖 Qt、JSON、影像库或项目会话。项目 XML、归档内容和
`.pimatch`/连接点/稀疏成果格式仍由各自模块负责。现有 Qt `ProjectPackageLayout` 和 `ProjectIO`
已委托标准布局对象计算物理路径，`ProjectWorkspaceStore`、`ProjectAssetImporter` 和项目资源删除逻辑
也通过 `ProjectPathBridge` 复用同一套 Chunk 目录规则，保持对 GUI 和旧调用方的接口兼容；运行根注册、
默认 Chunk 选择、复制/归档事务和递归删除仍留在 Qt 项目适配层。

验证：headless 定向 CTest 33 项、桌面定向 CTest 40 项通过，覆盖 `PlaFsTest`、
`ProjectPackageLayoutTest`、`ProjectIOTest`、资产导入和 Chunk 会话；两套配置的 `ProjectDataTest` 又各自
完整通过 45 项。headless `test_plafs` 的 `readelf -d` 不包含 Qt NEEDED 项。
本轮没有迁移运行时 Chunk 注册表或递归资源清理，避免把路径布局和会话生命周期混在一起。

## 2026-09-18：空三标准数值入口与独立文件模块

新增 `common/file/`，拆为 `plascan_common_file` 和 `plascan_common_json_io`。
前者仅使用 C++20 和原子提交所需的最小平台 API，后者增加 header-only nlohmann/json；
两套 vcpkg manifest 都声明直接依赖。文件模块不包含项目 schema 或点云业务。
它提供 Unicode 路径、目录创建、二进制读取和原子流式/字节串写入。提交前同步临时文件，
保留旧文件权限和现有文件符号链接，写入/同步/替换失败保持旧目标并清理自己的临时文件。
不承诺目录项断电持久性或多个成果文件的事务性。

新增 `aerial_triangulation_engine`，生产针孔与 RPC runner 已使用这一标准类型入口：

- `PinholeInput` 传入已解析相机、关键点图、人工 prior track 和比例尺，返回原 IncrementalSfmResult。
- `RpcInput` 传入标准相机 map、连接点图和残差门限；RPC 轨迹、交会细化、坐标和流程分别组织，
  返回点、逐相机残差、ENU 原点和标准错误字符串。
- `TiePointGraph` 使用标准路径/map/vector。`TiePointGraphReader` 用 nlohmann/json 读取 v1/v2/v3，
  保留 image_id 重映射、64 位原始特征索引、压缩观测与直接边/v1 闭包语义。
  工程归档 token 的解析通过标准函数回调从 Qt 门面传入。
- Pipeline 计时改为 std::chrono::steady_clock。针孔/RPC 共用标准流 little-endian PLY 编码，
  PLY 和原有 schema 的 JSON sidecar 都通过新文件模块发布。

旧 GUI/CLI 工作流接口保留，项目相机和标记数据在数值入口之前适配。
外部相机仍保留内存结果的来源路径，SfM/BA 默认参数和成果格式未变。
工程 DTO、匹配准备、候选规划、质量报告构建与 sidecar 的 QJson 序列化仍属于 Qt 工作流门面。
数值入口自身无 Qt 类型/符号，但 camera/common IO 与 SfM/common log 的传递依赖仍含 Qt Core。
这一阶段不称为整个空三 target Qt-free；后续需拆共享依赖与标准质量报告，再迁移门面目录。

实际验证使用 Linux/GCC 的两个已有构建树，没有执行其他平台或全项目测试：

```bash
cmake --build build/linux-source-release --target test_standard_file_io test_standard_aerial_engine \
  test_sfm_attempt_runner test_rpc_aerial_triangulation_runner test_aerial_triangulation_result_writer \
  test_aerial_triangulation_pipeline test_gui_project_utils aerial_triangulation_cli \
  three_d_reconstruction_cli --parallel 4
python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure \
  -R 'StandardFileIO|StandardAerialEngine|SfmAttemptRunner|RpcAerialTriangulation|AerialTriangulation(ResultWriter|Pipeline|Workflow|Dialog|ModuleLayout)'
build/linux-source-release/tests/test_source_contracts
python3 scripts/env/run_tests.py --test-dir build/linux-source-headless-release --output-on-failure \
  -R 'StandardFileIO|StandardAerialEngine|SfmAttemptRunner|RpcAerialTriangulation|AerialTriangulation|SfmPairPlanner|SfmSearchPolicy|ReconstructionPrerequisite|MatchResultCatalog|TiePointPreparation|CameraIntrinsicPriorSanitizer|AdaptiveFocal'
python3 scripts/validation/check_core_boundaries.py
python3 -m unittest tests.test_core_boundaries tests.test_repo_hygiene
```

桌面定向 CTest 91 项、CPU headless 定向 CTest 125 项、source contracts 74 项、Python 边界/卫生 48 项通过。
当时的边界基线移除连接点图公开 Qt 容器债务和多余 QMap include，共 685 项，无禁止依赖；后续容器标准化已将当前基线降至 669 项。
`readelf` 验证标准文件模块测试程序未链接 Qt；`nm` 验证新引擎自身未引用 Qt 符号。
引擎测试程序仍链接 Qt Core，符合上述 camera/common IO 与日志传递依赖的现状。
桌面与 CPU headless 的空三/三维重建 CLI 均构建并运行 --help 通过。
本次脚本、快照和测试临时目录已清理，没有执行 commit、push、tag 或 Release。
