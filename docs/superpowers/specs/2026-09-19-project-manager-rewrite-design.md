# ProjectManager 一次性重写设计规格

日期：2026-09-19
状态：已实施；Linux/GCC 定向验证完成，保留工作区既有非本任务失败
范围：GUI project orchestration；不改变 `.plascan` 数据格式或摄影测量算法

## 1. 目标与成功标准

`ProjectManager` 当前约 3,186 行、持有十余类跨域状态，并在构造函数中直接创建和连接多个控制器。一次性重写的目标是把项目会话、长任务编排、资源生命周期和 UI 消息适配拆成有明确所有权的服务，同时在同一切换中迁移全部调用者和源码契约测试。

完成后必须满足：

1. `ProjectManager` 只保留 Qt façade/信号桥接和少量命令路由，不持有 `QFuture`、session generation、BA 预览缓存、AT cancel flag 或资源清理事务状态。
2. 任何后台结果写回项目前，都通过统一的 `ProjectSessionContext` 校验项目路径、Chunk 和 generation；项目切换、关闭和对象析构都会取消并隔离旧任务。
3. 资源清理的 prepare/execute/finalize、WAL 恢复和路径安全继续由 core 服务负责；GUI 只负责任务生命周期、互斥和消息适配。
4. 所有原有菜单、主窗口、对话框、Browser bridge、任务状态栏和测试改用新服务接口；不保留旧 God Object 的内部实现或双轨逻辑。
5. 现有用户可见行为（中文文案、进度信号、取消语义、BA 预览确认、`.plascan` 写回顺序）保持不变。
6. 新服务拥有独立行为测试，受影响 GUI/core target 在 Linux/GCC 上构建通过；无法运行的 Windows/MSVC 或真实大场景验证必须明确记录。

## 2. 边界与非目标

### 2.1 本次包含

- 重写 `src/gui/project/manager/ProjectManager.*` 的依赖和实现边界。
- 将现有 `ProjectSessionFacade` 升级为真正的 GUI `ProjectSession`，并把 session generation、数据查询、相机/交会结果写回集中到该服务。
- 将生命周期控制器、蒙版、稀疏、点云、模型、地形和相机控制器统一纳入 `ProjectTaskOrchestrator` 的所有权与信号编排。
- 将影像/资产/参考数据导入、生成成果删除、pack 和 portable export 的 GUI 编排移入 `ProjectResourceService` 与 `ProjectResourceCleanupCoordinator`。
- 将 `QMessageBox`、`QInputDialog` 和文件选择结果适配到 `ProjectUiMessageAdapter`；后台服务不直接依赖 `QWidget`。
- 同步迁移所有生产调用者、CMake 源文件列表、测试和架构文档。

### 2.2 本次不包含

- 不修改 `src/common/project/ProjectSession.*` 的无界面 4.0 会话模型；GUI `ProjectSession` 是独立的应用层服务。
- 不修改项目 JSON schema、路径格式、资源清理事务格式、相机模型算法、MVS/SfM/DEM/网格算法或模型资源。
- 不引入继承式 `Manager` 基类、全局事件总线或隐式 CPU/GPU 回退。
- 不重做 `MainWindow` 的布局、渲染、菜单视觉样式或任务栏 UI；只改其依赖注入和信号订阅边界。

## 3. 现状基线

当前 `ProjectManager` 仍直接承担以下职责：

| 现有职责 | 主要位置 | 目标归属 |
|---|---:|---|
| 项目创建/打开/保存/关闭、Chunk 操作 | `ProjectManager.cpp:707-768, 1885-2010` | `ProjectLifecycleService` + `ProjectSession` |
| 影像扫描/导入、点云/模型/参考数据导入、pack | `785-1205, 1865-1875` | `ProjectResourceService` |
| Survey/参考地形对话框与参数确认 | `1215-1676` | `ProjectResourceService` + `ProjectUiMessageAdapter` |
| 生成数据删除、异步 cleanup future、生命周期锁 | `1678-1863, 2013-2028` | `ProjectResourceCleanupCoordinator` |
| UI 设置、元数据/影像查询、相机/交会结果读写 | `1880-2119, 2550-2825, 3086-3098` | `ProjectSession` + `ProjectResultService` |
| BA 执行、取消、预览缓存、确认/丢弃 | `2126-2550, 2958-3078` | `ProjectBundleAdjustController`，由 `ProjectTaskOrchestrator` 编排 |
| 模型/点云/DEM/正射/稀疏任务路由与取消 | `1089-1139, 2826-2985, 3151-3187` | `ProjectTaskOrchestrator` |
| 约 30 组跨控制器 signal 连接 | 构造函数 `462-680` | `ProjectTaskOrchestrator` + `ProjectEventBridge` |

现有蒙版、点云、模型、地形、稀疏重建和相机控制器保留稳定算法实现，但构造依赖改为窄服务，且不再持有 `ProjectManager* _owner`。迁移完成且零引用后，旧 `ProjectSessionFacade`、`ProjectLifecycleController` 和 `ProjectUiCommands` 被删除。

## 4. 目标架构

```text
MainWindow / Menu / Dialogs
          │  typed commands + read models
          ▼
ProjectServiceContainer (组合根)
          ├── ProjectSession
          ├── ProjectLifecycleService
          ├── ProjectResourceService
          ├── ProjectResourceCleanupCoordinator
          ├── ProjectTaskOrchestrator
          │     ├── ProjectBundleAdjustController
          │     ├── ProjectMaskWorkflowController
          │     ├── ProjectSparseReconstructionManager
          │     ├── ProjectPointCloudWorkflowController
          │     ├── ProjectModelManager
          │     ├── ProjectTerrainProductsManager
          │     └── ProjectCameraSetupManager
          └── ProjectUiMessageAdapter (Qt implementation injected)

ProjectManager (薄 QObject 兼容 façade)
          └── owns ProjectServiceContainer + forwards legacy signals

ProjectTaskOrchestrator ──> ProjectSession (snapshot + session gate)
ProjectResourceCleanupCoordinator ──> core::ProjectResourceCleanupService
ProjectUiMessageAdapter ──> QWidget/Qt dialogs only
ProjectData <────────────── ProjectSession (唯一 GUI 数据入口)
```

`ProjectManager` 可以继续作为单一 QObject 信号出口，以避免主窗口同时连接多个服务；它只转发服务信号和调用服务方法，不包含业务分支、后台 lambda 或数据解析。

## 5. 新服务契约

### 5.1 `ProjectSession`

文件：

- `src/gui/project/services/ProjectSession.h`
- `src/gui/project/services/ProjectSession.cpp`

核心接口：

```cpp
class ProjectSession final : public QObject
{
    Q_OBJECT
public:
    explicit ProjectSession(ProjectData* projectData, QObject* parent = nullptr);

    ProjectData* data() const;
    bool hasProject() const;
    QString projectPath() const;
    QString activeChunkId() const;
    ProjectSessionContext context() const;
    bool isCurrent(const ProjectSessionContext& context) const;
    void advanceGeneration();

    bool isDirty() const;
    QJsonObject metadata() const;
    QJsonObject coreMetadata() const;
    QStringList imagesByCategory(const QString& category) const;
    QStringList allImages() const;
    QString matchFile(const QString& firstImage, const QString& secondImage) const;
    QJsonObject loadUiSettings() const;
    void saveUiSettings(const QJsonObject& settings);
    void markWorkspaceDirty();
    void discardTemporaryMetadata();

    bool setCameraInstances(const QMap<QString, QJsonObject>& cameras,
                            int* updatedCount = nullptr,
                            QString* errorMessage = nullptr);
    bool replaceCameraInstances(const QStringList& targetImagePaths,
                                const QMap<QString, QJsonObject>& cameras,
                                int* updatedCount = nullptr,
                                int* clearedCount = nullptr,
                                QString* errorMessage = nullptr);
    bool clearCameraInstances(const QStringList& imagePaths,
                              int* updatedCount = nullptr,
                              QString* errorMessage = nullptr);
    bool appendIntersectionResult(const QJsonObject& result,
                                  QString* errorMessage = nullptr);
    QJsonArray intersectionResults() const;

signals:
    void sessionChanged(const ProjectSessionContext& context);
};
```

`ProjectSession` 连接 `ProjectData::projectOpened`, `projectClosed` 和 `activeChunkChanged`，推进 generation 并发出 `sessionChanged`。取消动作由 `ProjectTaskOrchestrator` 响应该信号执行；`ProjectSession` 不直接拥有任何 worker。

### 5.2 `ProjectTaskContext` 与 `ProjectTaskOrchestrator`

文件：

- `src/gui/project/tasks/ProjectTaskContext.h`
- `src/gui/project/tasks/ProjectTaskOrchestrator.h/.cpp`
- `src/gui/project/tasks/ProjectBundleAdjustController.h/.cpp`

任务上下文至少包含：

```cpp
struct ProjectTaskContext
{
    QString taskId;
    ProjectSessionContext session;
    std::shared_ptr<std::atomic<bool>> cancelFlag;
};
```

`ProjectTaskOrchestrator` 负责：

- 创建并持有现有 workflow controller；
- 统一把 `ProjectSessionContext` 传给后台任务，并在回调入口调用 `session->isCurrent()`；
- 暴露 start/cancel API（BA、AT、稀疏、模型、点云、DEM、正射、蒙版）；
- 统一转发现有进度/完成/结果信号，保持信号签名和 task id 不变；
- 维护 BA preview（pending cameras/before metadata/result）和 AT cancel flag；
- 收到 `ProjectSession::sessionChanged` 时取消所有可取消任务、丢弃旧 preview，并阻止旧结果写回；
- 析构时按“请求取消 → 等待可加入 future → 释放 ProjectData 引用”的顺序关闭。

各现有控制器的构造函数改为接收 `ProjectSession*`、必要的窄回调和 `ProjectUiMessageAdapter*`；禁止新增 `ProjectManager*` 成员。

### 5.3 `ProjectResourceService`

文件：

- `src/gui/project/services/ProjectResourceService.h/.cpp`

负责同步命令和导入流程：影像/文件夹导入、点云/模型导入、参考数据登记、Survey 控制点入口、移除引用和 pack。耗时扫描/复制继续使用 `GuiTaskRunner`，每个回调携带 session context。

它只返回结构化结果或发出领域信号，不直接访问 `ProjectManager` 私有字段；所有提示通过 `ProjectUiMessageAdapter`。

### 5.4 `ProjectResourceCleanupCoordinator`

文件：

- `src/gui/project/services/ProjectResourceCleanupCoordinator.h/.cpp`

核心接口：

```cpp
class ProjectResourceCleanupCoordinator final : public QObject
{
    Q_OBJECT
public:
    ProjectResourceCleanupCoordinator(ProjectSession* session,
                                      ProjectUiMessageAdapter* messages,
                                      QObject* parent = nullptr);
    ~ProjectResourceCleanupCoordinator() override;

    bool isRunning() const;
    bool rejectLifecycleChange(const QString& operation);
    void deleteGeneratedData(const QString& section, const QStringList& resourcePaths,
                             QWidget* requestWidget = nullptr);
    void waitForFinished();

signals:
    void progressChanged(const QString& taskId, int value, int maximum);
    void finished(const QString& taskId);
};
```

实现必须继续调用 core 的 `prepareGeneratedDataCleanup`、`executePreparedCleanup` 和 `finalizePreparedCleanup`，并保留当前 request widget 禁用、session 检查、portable export 互斥和析构等待行为。`ProjectManager` 不再持有 cleanup future 或 finalize lambda。

### 5.5 `ProjectUiMessageAdapter`

文件：

- `src/gui/project/services/ProjectUiMessageAdapter.h/.cpp`

定义不暴露 Qt 对话框类的调用面（信息、警告、错误、确认、文本/数值选择、文件路径选择），提供 `QtProjectUiMessageAdapter` 默认实现和 `FakeProjectUiMessageAdapter` 测试实现。接口只使用 `QString`、`QStringList`、`QWidget*` 和以下自定义值：`enum class UiAnswer { Yes, No, Cancel };`、`struct UiDialogResult { bool accepted = false; QString value; QStringList values; double number = 0.0; };`；只有 Qt 实现包含 `QMessageBox`、`QInputDialog` 和 `QFileDialog`。

所有取消结果必须显式返回 `accepted=false`，不得把取消当作空字符串成功。

## 6. 生命周期与数据流

### 6.1 打开/关闭/Chunk 切换

1. `ProjectLifecycleService` 读取/应用 `ProjectData` snapshot。
2. `ProjectSession` 监听 `ProjectData` 生命周期信号，推进 generation 并发出 `sessionChanged`。
3. `ProjectTaskOrchestrator`、`ProjectResourceService` 和 cleanup coordinator 收到变化后取消/隔离旧任务。
4. `ProjectManager` 的 event bridge 转发旧有 `projectOpened/projectClosed/projectSessionChanged` 等信号，主窗口行为不变。

### 6.2 长任务

1. 命令入口从 `ProjectTaskOrchestrator` 取得当前 `ProjectTaskContext`。
2. worker 只读取不可变 snapshot；进度通过 queued callback 回主线程。
3. 回调先检查对象存活、cancel flag 和 `ProjectSession::isCurrent(context.session)`。
4. 只有通过检查的回调才能写 ProjectData、登记成果或弹出完成消息。
5. 完成/失败/取消都走同一清理路径，释放 task state 并发出既有 finished 信号。

### 6.3 资源清理

1. coordinator 在主线程确认用户意图并调用 core prepare。
2. 成功 prepare 后锁定会话变更、禁用请求控件，后台执行不可变 plan。
3. 主线程完成回调先 finalize persistence，再释放锁/控件，最后按 session 是否仍匹配决定是否呈现结果。
4. 析构先 `waitForFinished()`，再释放 `ProjectSession`/`ProjectData` 相关对象。

## 7. 错误、取消和兼容语义

- 失败信息必须包含操作、路径或任务 ID；不得静默降级。
- 用户取消对话框不弹错误、不修改元数据、不启动 worker。
- 项目切换导致的旧任务结果被丢弃时记录 debug/info 日志，但不弹出过期错误框。
- BA preview 只有在当前 session 且用户确认后写回；取消、失败、切换或析构都会清空 preview。
- 资源清理事务失败时保留 core 返回的 failed/preserved 路径信息，并继续由 recovery binding 在下次打开恢复。
- 公开 façade 的信号签名在本次切换中保持不变；命令方法改为薄转发，内部旧的 `_owner` 回调接口全部删除。

## 8. 测试策略

### 8.1 新增行为测试

- `tests/test_project_session.cpp`：空数据、generation 推进、路径/Chunk 匹配、旧 session 拒绝写回、相机/交会结果委托。
- `tests/test_project_ui_message_adapter.cpp`：确认/取消、错误/信息记录、文件选择取消不产生副作用。
- `tests/test_project_task_orchestrator.cpp`：任务创建与取消、项目切换取消、析构等待、旧回调不写回、BA preview 生命周期。
- `tests/test_project_resource_cleanup_coordinator.cpp`：并发删除互斥、request widget 恢复、session 切换、prepare/execute/finalize 顺序和析构等待。

每个测试遵循 RED → 确认失败 → 最小实现 → GREEN → 回归；不能用纯源码字符串断言替代异步行为测试。

### 8.2 迁移既有测试

- 将 `tests/test_project_session_facade.cpp` 改为 `ProjectSession` 契约并删除旧 facade 测试目标。
- 更新 `tests/test_gui_project_utils.cpp` 中依赖 `ProjectManager` 方法块的契约，保留必要的边界检查，删除对旧内部成员/future/`QMessageBox` 的实现细节断言。
- 更新 `tests/test_source_contracts.cpp` 与 GUI 生命周期测试，断言新服务的依赖方向、无 `ProjectManager* _owner` 和薄 façade 行为。
- 保持资源清理 core 测试、项目数据测试、任务状态栏测试和现有工作流行为测试覆盖不下降。

### 8.3 构建与运行门禁

定向构建/测试：

```bash
cmake --build build/linux-source-release --target gui_project test_project_session \
  test_project_ui_message_adapter test_project_task_orchestrator \
  test_project_resource_cleanup_coordinator test_gui_project_utils -j2
ctest --test-dir build/linux-source-release --output-on-failure \
  -R 'ProjectSession|ProjectTask|ProjectResource|GuiAsync|ProjectManager|Taskbar|SourceContract'
```

完整门禁（当前原生平台）：

```bash
python3 scripts/env/configure_with_env.py --source-deps --build --test
python scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
git diff --check
```

同时运行 `clang-format`（仅本次修改文件/区段）和 Python `py_compile`（如修改脚本）。Windows/MSVC 验证由 CI 或专用环境单独报告，不能用 Linux 结果代替。

## 9. 一次性切换顺序

虽然最终是一次性替换，但每个阶段结束都必须可编译、可测试，避免在同一工作树留下不可诊断的半迁移状态：

1. **冻结契约**：记录全部 ProjectManager API、信号连接和外部调用点；新增目标服务的 RED 测试与架构边界测试。
2. **建立服务骨架**：实现 `ProjectSession`、消息适配和 task context；迁移 session generation、查询/写回和消息调用。
3. **迁移生命周期与资源**：重写 lifecycle/resource/cleanup，删除 PM 对话框、导入和 cleanup future 实现。
4. **迁移任务与 BA**：建立 orchestrator 和 BA controller；将所有工作流控制器改为 session/task 依赖，删除 PM 的异步 lambda 和 preview 字段。
5. **切换调用者**：更新 MainWindow、菜单、对话框、Browser bridge、任务状态控制器和测试构造方式；删除 `ProjectSessionFacade` 与旧 owner API。
6. **收口与验证**：缩减 `ProjectManager` 到 façade，更新 CMake/架构文档，运行定向与全量门禁，审计未提交差异和临时目录。

## 10. 风险与回滚

主要风险是一次性 API 迁移遗漏、Qt queued callback 的对象生命周期变化、BA preview/cleanup 的时序回归和大测试目标编译时间增加。每个阶段保留独立 commit 边界（仅在用户授权 commit 时执行），并以新服务行为测试和现有 GUI 回归作为回滚判断；不得通过恢复旧 God Object 双轨实现来“修复”失败。

## 11. 完成定义

- `ProjectManager.h` 不再声明业务数据、future、generation、preview 或 controller 成员；`ProjectManager.cpp` 只包含 façade 路由和 signal bridge。
- 新服务的头文件不依赖 `ProjectManager.h`，现有 workflow controller 不含 `ProjectManager* _owner`。
- 新增服务测试和既有相关测试全部通过；全量 CTest 无未解释跳过。
- `docs/PROJECT_ARCHITECTURE.md`、`docs/README`/相关 GUI 文档准确描述新边界。
- 只报告实际执行的平台验证；不执行 commit/push/tag/Release，除非用户另行授权。
