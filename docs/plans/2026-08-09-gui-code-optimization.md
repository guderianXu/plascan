# PlaScan GUI 编码优化实施计划

更新日期：2026-08-09

## 目标与边界

本计划用于继续收敛 PlaScan GUI 的并发安全、主线程响应、状态复用和模块职责。优化必须保持现有项目格式、
工作流参数、中文 UI 行为和核心计算结果不变，并同时兼容 Windows/MSVC 与 Linux/GCC。

当前三维视图的逐帧几何绘制已经通过 QRhi/Vulkan 执行。本计划不重新设计渲染后端，重点处理加载阶段的
CPU 重复工作、GUI 主线程阻塞、任务生命周期和过大的协调类。

实施约束：

- 不在 GUI 主线程执行不可控时长的文件读取、图像解码、几何分析或工作流计算。
- 不通过静默 CPU 回退掩盖 GPU、文件或模型加载失败。
- 不做机械式拆文件；每次拆分必须形成可独立测试的真实职责。
- 项目 JSON 继续作为持久化边界，GUI 内部逐步改用 typed snapshot/model。
- 每个阶段独立构建、测试和提交；出现行为回归时可以按阶段回滚。
- 当前工作区中的相机参考和 MVS 改动先完成自身验证，再进入与其重叠的重构阶段。

## 当前基线

GUI 已有可复用基础：

- `ProjectUiHydrator` 能合并快速连续的元数据刷新，并通过 generation 丢弃过期阶段。
- `GuiTaskRunner` 已提供受对象生命周期保护的后台任务结果回调。
- `TaskbarProgressController` 已隔离 Windows 原生任务栏实现。
- `CameraSceneWidget` 使用 QRhi 静态缓冲、dirty 标志和 shader 切换显示模式，不存在逐帧 CPU 光栅化回退。
- 相机参考 WIP 已开始按 controller、repository、model 分层，应继续保持这个方向。

仍需解决的主要问题：

1. DEM 可重复启动并写入同一默认输出目录，任务进度信号没有 task ID。
2. 重叠度分析、蒙版轮廓和部分热力图处理仍会阻塞 GUI 主线程。
3. 图片预览与正式工作流共享 Qt 全局线程池，旧图片任务缺少协作取消。
4. 状态栏、Dashboard 和任务栏分别维护同一任务状态。
5. 项目元数据变化仍触发多个控件全量解析和重建。
6. `CameraSceneWidget`、`ProjectManager`、`MenuWorkflowController` 和 `MainMenu` 职责过多。
7. GUI 测试集中在超大测试文件中，并存在较多源码字符串契约测试。

## 阶段 0：正确性门禁与确定性清理

### 0.1 DEM 单任务和输出占用保护

涉及模块：

- `src/gui/project/manager/ProjectTerrainProductsManager.*`
- `src/gui/main_window/MenuWorkflowController.cpp`
- `src/core/terrain/TerrainPipeline.*`

实施：

- 为 DEM 请求分配稳定 task ID，并让进度、完成和取消信号携带 task ID。
- Manager 保存当前 DEM task context；同一项目、Chunk 或规范化输出目录只允许一个写任务。
- 第二个冲突请求必须明确拒绝或要求用户取消现有任务，不能并发写同一 `dem.tif`。
- 产品先写入任务级临时目录，全部成功后再发布到正式目录。
- 项目关闭、Chunk 切换和 Manager 析构时请求协作取消，过期任务不得写回元数据。

验收：

- 连续启动两个相同输出请求时，第二个请求被确定拒绝，正式文件未被截断或覆盖。
- 两个不同输出目录的任务若允许并行，其进度只进入各自对话框。
- 项目切换后旧任务不弹窗、不写元数据、不留下“进行中”任务栏状态。

### 0.2 当前相机参考 WIP 的合入门禁

涉及模块：

- `src/gui/reference/CameraReferenceController.cpp`
- `src/gui/main_window/MainWindowProjectBindings.cpp`
- `src/gui/main_window/MenuWorkflowController.cpp`

实施：

- CSV 导出在 `QSaveFile::commit()` 前显式 flush，并检查 `QTextStream::status()`。
- 标记参考导入返回 changed/success 结果；用户取消时不重载 repository，也不清空 undo stack。
- 相机标定窗口采用单实例或 session-bound 实例；项目关闭/切换后窗口失效。
- 所有修改操作在执行前核对打开窗口时的项目路径和 session generation。
- 已有参考集合的替换确认覆盖 matched、unmatched、source 和 lever-arm 任一非空状态。

验收：

- 导出的短 CSV 和大 CSV 内容完整，写入失败不会显示成功。
- 取消导入后 marker undo stack 保持不变。
- 切换项目后旧相机标定窗口不能修改新项目。

### 0.3 删除确定无生产引用的代码

首批清理：

- 删除未接入生产状态的 `CameraImageSelectionState` 及其孤立测试。
- 删除 `MatchPairSelectorDialog` 中仅转发到 snapshot 静态实现的无调用包装函数。
- 删除无调用的 `ImageViewWidget::calculateZoomFactor()`、
  `ProjectUiCommands::openProjectByDialog()` 和 `MarkerWorkspaceController::undoStack()` 接口。
- 后续通过编译器、链接器、`rg` 和行为测试逐项确认，不使用基于命名猜测的批量删除。

状态（2026-08-09）：以上首批确定性清理已实施；跨模块路径工具合并留在后续独立提交，避免把行为修改
混入纯删除提交。

验收：

- 生产代码中不存在目标符号引用。
- 相关 GUI/CameraScene 测试通过。
- 不为保留脆弱源码字符串测试而继续保留无调用生产接口。

## 阶段 1：统一后台任务和图片加载

### 1.1 ProjectTaskContext

在现有 `GuiTaskRunner` 上增加组合式 `ProjectTaskContext`/`runProjectTask()`，统一保存：

- typed task ID；
- 项目路径、Chunk ID 和 session generation；
- cancellation source/token；
- 后台线程池选择；
- 单调进度发布；
- success、failure、cancelled 三种完成状态；
- 完成后的唯一清理路径。

不建立继承式 Manager 基类。`ProjectManager` 保持 façade，具体工作流持有 task context。

首批迁移：

1. `ProjectTerrainProductsManager`
2. `TiePointWorkflowController`
3. `ProjectPointCloudWorkflowController`
4. `ProjectManager` 中的 Bundle Adjust
5. `MenuWorkflowController` 中的 Aerial Triangulation

### 1.2 GuiImageLoadService

新增共享图片加载服务：

- 使用独立、有界的 preview I/O 线程池，不占用正式重建任务线程池。
- 按规范化路径、项目路径、目标尺寸和显示模式组成缓存 key。
- 同 key 请求去重；控件使用 latest-only request generation。
- 支持协作取消，控件析构不在 GUI 线程 `waitForFinished()`。
- 缩略图直接使用目标尺寸解码，不先创建全分辨率 `QImage`。

首批接入 `CanvasWidget`、`ImageViewWidget`、`PhotoStripWidget` 和 CameraScene 相机平面图片。

### 1.3 主线程热点迁移

- `OverlapAnalysisDialog`：影像准备、DEM 加载和 overlap analyze 全部进入后台任务；结果表改用 model。
- 蒙版轮廓：后台读取、提取和简化轮廓，GUI 线程只安装结果。
- 深度叠加：single-flight latest-only，按图像行或批次检查取消；避免过期任务继续全图扫描。
- 视差热力图：后台生成原始热力图并缓存与 viewport 匹配的显示结果。
- 工作流报告：后台读取/解析大 JSON，GUI 分页或按需构建明细表。

验收：

- GUI 控件析构路径不再调用无界 `waitForFinished()`。
- 快速切换 100 次影像时，活跃 preview job 数不超过线程池上限，只有最后请求更新 UI。
- 固定大图/大蒙版用例中，事件循环心跳 P95 延迟低于 50 ms；具体机器基线记录在 benchmark 文档。
- 取消或切换项目后无过期 overlay、弹窗和项目写回。

## 阶段 2：统一任务状态和项目展示快照

### 2.1 ProjectTaskModel

建立 typed `ProjectTaskModel` 作为唯一任务状态源。任务 runner 只更新 model，以下组件只订阅：

- 状态栏 `TaskStatusWidget`
- `ProjectDashboardWidget`
- `TaskbarProgressController`

任务记录包含 ID、显示名、阶段、value/maximum、可取消状态、项目 session 和最终结果。删除从状态栏控件
反向生成 Dashboard JSON 的路径。

验收：新增一种工作流任务时，只注册一次 descriptor，不再分别修改三套进度代码。

### 2.2 ProjectMetadataView 与 ProjectChangeSet

建立不可变 `ProjectMetadataView`：

- 在 JSON 持久化边界一次性归一化 `project_files` 和结果数组。
- 建立 image UUID/path、result path、reference ID 等索引。
- 为 Dashboard、DataTree、PhotoStrip、WorkspaceCenter、ReferencePanel 提供 typed selector。
- 元数据事件携带 revision 和 change flags，例如 `ImagesChanged`、`ResultsChanged`、
  `ReferencesChanged`、`UiChanged`。

保留现有 `projectMetadataChanged(QJsonObject)` 作为迁移期兼容信号，消费者迁移完成后再删除。

### 2.3 增量 Qt Model/View

- PhotoStrip 从 `QListWidget` 迁移到 `QAbstractListModel`，使用稳定 image UUID 做插入、删除和状态变化。
- DataTree 使用定制 model 或增量更新现有 model，不再每次清空并重建整棵树。
- Dashboard 将摘要计算结果与表格 presenter 分离，仅更新变化的 section。
- ReferencePanel 刷新时保留选择、展开和滚动位置。

验收：

- 只改变任务报告时不重建照片列表和相机姿态。
- 只改变一个影像状态时 model 只发出对应行的 `dataChanged`。
- 10,000 张照片用例保持可交互，批次加载期间单次 GUI 事件处理不超过既定预算。

## 阶段 3：拆分三维视图并优化加载产物

保留 `CameraSceneWidget` 作为 QRhi 生命周期、输入事件和高层绘制顺序外壳，依次抽出：

1. `SceneLoadCoordinator`：加载队列、session、generation 和解析结果。
2. `CameraImageCache`：相机平面图片请求、缓存和 atlas 输入。
3. `RhiSceneRenderer`：buffer、pipeline、texture、uniform 和 draw call。
4. `SceneOverlayRenderer`：gizmo、图例、进度和相机标签。
5. `PointCloudEditController`：框选、删除、undo 和持久化请求。

加载阶段同步优化：

- 合并点云 AABB、标量范围和 VBO 打包遍历。
- 线框索引在首次切换到 wireframe 时生成。
- 纹理顶点按 `(position, normal, uv)` 去重并使用索引缓冲。
- overlay 的 geometry/raise 只在 resize/show/layer change 时执行，逐帧只 repaint 脏区域。

验收：

- `CameraSceneWidget.cpp` 只保留协调职责，目标控制在约 800 行以内；子模块原则上不超过 400 行。
- 全部模型显示模式、相机图片锁定、手动删除和 undo 行为保持一致。
- 同一模型重构前后截图、包围盒、顶点/索引数量和 GPU 错误路径测试通过。

## 阶段 4：收敛项目、工作流和菜单边界

### 4.1 ProjectManager 与工作流

- `ProjectManager` 只保留项目 façade、会话入口和公共信号。
- Bundle Adjust 移入 `ProjectBundleAdjustController`。
- 空三前置分析移入无 QWidget 依赖的 `SparsePrerequisiteAnalyzer`。
- 空三执行和写回移入 `AerialTriangulationWorkflowRunner`。
- `setupProjectManager()` 按 photo、tie-points、markers、data-tree、reference 等域拆分绑定函数。

### 4.2 MainMenu 单一描述

建立 typed `ActionId + ActionDescriptor + MenuSpec`：

- Designer 主窗口和无 Designer 测试窗口走同一个 find-or-create builder。
- action 文案、objectName、快捷键、checkable、默认值和菜单位置只有一个事实来源。
- 图标迁移到 Qt resource 或独立 `ToolbarIconFactory`。
- About/Python 环境等业务响应移出菜单构建器。

验收：

- 删除第二套 fallback 菜单定义。
- 任一 action 的生产与测试构建结果一致。
- `ProjectManager`、`MenuWorkflowController` 和 `MainMenu` 不再包含完整计算工作流。

## 阶段 5：测试结构和持续性能门禁

- 将 `test_gui_project_utils.cpp` 按 metadata、widgets、workflow、task、platform 等域拆分。
- 源码字符串测试仅保留架构/依赖边界门禁；交互和异步语义改为行为测试。
- 增加 destruction-during-load、rapid-switch、project-switch、double-start 和 stale-result 测试。
- 为 GUI 事件循环延迟、图片队列深度、metadata refresh 次数和场景准备耗时增加可重复 benchmark。
- Windows 本地完成 MSVC 受影响目标、相关测试和全量可执行测试；Linux/GCC 由对应环境或 CI 验证。

## 推荐提交顺序

1. GUI 死代码清理。
2. DEM 单任务和 task ID。
3. 相机参考 WIP 正确性门禁。
4. `ProjectTaskContext` 与 `ProjectTaskModel`。
5. `GuiImageLoadService` 和主线程热点迁移。
6. `ProjectMetadataView` 与增量 models。
7. `CameraSceneWidget` 分层。
8. ProjectManager/AT/MainMenu 分层。
9. GUI 测试拆分与性能门禁。

每个提交都必须能独立编译和测试。不得在一个提交中同时进行数据格式修改、行为修改和大规模文件移动。

## 完成定义

- 不存在已知的 GUI 主线程无界等待或大文件同步计算路径。
- 所有后台任务具有 task ID、session、取消和唯一完成语义。
- 状态栏、Dashboard 和任务栏读取同一个任务状态源。
- 元数据消费者按 change set 增量更新，不再无条件全量重建。
- 三维视图、项目 façade、菜单和工作流控制器职责可独立测试。
- GUI 相关定向测试、当前平台全量测试和 required CI 均通过。
