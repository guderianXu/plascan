# ProjectManager 一次性重写实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `ProjectManager` 从承载项目状态、异步任务、资源清理和 UI 交互的 God Object 重写为稳定的薄 façade，并把职责迁移到会话、任务编排、资源服务、清理协调器和 UI 消息适配器，同时保持现有项目文件格式、算法流程、信号语义和用户可见行为不变。

**Architecture:** 以 `ProjectServiceContainer` 作为 GUI 项目的组合根。`ProjectSession` 持有当前 `ProjectData` 和 session generation；`ProjectTaskOrchestrator` 持有已有工作流 controller，并通过 `ProjectTaskContext` 传递会话、取消和错误端口；`ProjectResourceService` 负责导入、导出和资源相关 UI 流程；`ProjectResourceCleanupCoordinator` 独占清理 future、锁和 core cleanup transaction；`ProjectBundleAdjustController` 独占 BA 状态机和预览结果；`ProjectUiMessageAdapter` 隔离 QMessageBox/QInputDialog/QFileDialog。`ProjectManager` 只保留 façade 方法、Qt 信号和组合根引用，不再持有任务状态、future、generation map、BA preview map 或直接弹窗。

**Tech Stack:** C++20, Qt6 Widgets/Core, CMake, GTest, QtConcurrent（仅保留在任务/服务实现层），现有 `ProjectData`、`ProjectResourceCleanup` 和 GUI task 基础设施。

**Spec:** [2026-09-19-project-manager-rewrite-design.md](../specs/2026-09-19-project-manager-rewrite-design.md)

**执行状态（2026-09-20）：** 重写和调用方切换已完成；严格边界扫描为零，受影响目标与定向测试通过。
完整 GUI 测试仍有 15 项与本重写无关的深度批次/模型策略失败，因此未把共享脏工作区扩展为跨域修复。
实际批次、审查与验证证据记录在 `.superpowers/sdd/2026-09-19-project-manager-rewrite/progress.md`。

## Global Constraints

- 本计划是一次性切换，但每个任务都必须留下可构建、可定向测试的中间状态；不得把所有改动堆成一次无法定位的巨型提交。
- 不改变 `.plascan` schema、序列化格式、核心算法参数、任务输出目录布局或用户可见中文文案，除非测试证明旧行为本身是 bug 且另有明确授权。
- 保留当前工作区中与本任务无关的 dirty 修改；开始每个执行批次前重新检查 `git status --short` 和运行中的构建/测试进程，不使用 reset、checkout 或批量格式化。
- 新增私有成员使用 `_lowerCamelCase`；C++ 使用 Allman 花括号和仓库 `.clang-format`。只格式化本任务修改的文件。
- 所有异步回调必须携带启动时的 `ProjectSessionContext`，回调落地前通过 `ProjectSession::isCurrent(context)` 校验；取消、失败、切换项目和对象析构都必须使旧回调无效。
- UI 层不允许在主线程同步执行耗时 core 操作；服务只通过 adapter 发出消息，测试使用 fake adapter，不依赖真实桌面。
- 本地构建进程空闲前不启动互相冲突的完整构建。正式验证使用仓库规定的 `scripts/env/configure_with_env.py` 和 `scripts/env/run_tests.py`。

## Review Focus

审阅和实现时优先检查以下五条不变量：

1. 切换项目或 Chunk 后，旧任务的 progress、error、result 信号不能修改新会话。
2. `ProjectManager`、service container 或窗口析构时，不能遗留可访问已释放对象的 future continuation、signal connection 或锁。
3. 用户在导入、删除、BA 预览对话框中选择 Cancel/No 时，不能发生半写入、错误的 dirty 标记或隐藏的资源删除。
4. portable export、generated-data cleanup 和 close/open 生命周期操作之间必须有明确的互斥顺序，且错误时锁一定释放。
5. 每个旧 PM 信号只能有一个来源；迁移后不得出现重复 progress、重复 warning 或重复 `projectChanged`。

## File Map and Ownership

| Boundary | New/primary files | Removed or migrated responsibility |
| --- | --- | --- |
| Session | `src/gui/project/services/ProjectSession.h/.cpp`, `tests/test_project_session.cpp` | `ProjectSessionFacade` 中的状态、generation、查询和 camera write-back |
| UI adapter | `src/gui/project/services/ProjectUiMessageAdapter.h/.cpp`, `tests/test_project_ui_message_adapter.cpp` | PM 中的 QMessageBox/QInputDialog/QFileDialog 直接调用 |
| Cleanup | `src/gui/project/services/ProjectResourceCleanupCoordinator.h/.cpp`, `tests/test_project_resource_cleanup_coordinator.cpp` | PM 中的 cleanup future、锁、prepare/execute/finalize 和取消 |
| Resources/lifecycle | `src/gui/project/services/ProjectResourceService.h/.cpp`, `ProjectLifecycleService.h/.cpp` | PM 中的导入、导出、Survey、open/save/close 编排 |
| Tasks | `src/gui/project/tasks/ProjectTaskContext.h`, `ProjectTaskOrchestrator.h/.cpp`, `ProjectBundleAdjustController.h/.cpp`, task tests | PM 中的 controller ownership、generation gate、AT/BA 状态和任务取消 |
| Composition | `src/gui/project/services/ProjectServiceContainer.h/.cpp` | PM 构造函数中逐个 `new` controller 的逻辑 |
| Façade | `src/gui/project/manager/ProjectManager.h/.cpp` | 只保留兼容 façade、signals 和 service delegation |
| Contracts/docs | `tests/test_source_contracts.cpp`, `tests/CMakeLists.txt`, `src/gui/cmake/GuiSources.cmake`, `docs/PROJECT_ARCHITECTURE.md` | 旧 PM 内部实现契约和过时架构描述 |

## Implementation Tasks

### Task 1 — 固化边界契约与迁移基线

**Files:**

- Create `tests/test_project_manager_architecture.cpp`.
- Modify `tests/CMakeLists.txt` to register `test_project_manager_architecture`.
- Modify `tests/test_source_contracts.cpp` only where the old PM-specific assertions must be replaced by named boundary assertions.
- Create `scripts/validation/check_project_manager_boundary.py` for a deterministic source audit; add its invocation to the test target or documented validation command.

**Interfaces consumed/produced:**

- Consumes the current public declarations in `ProjectManager.h` and current call sites discovered with `rg`.
- Produces a checked-in inventory of forbidden PM dependencies and the required service headers. The test must fail before the new headers exist (red), then become green as each boundary lands.

**Steps:**

- [x] **Step 1: Write the failing architecture test.** Add this test body and a local `readTextFile()` helper to `test_project_manager_architecture.cpp`:

```cpp
TEST(ProjectManagerArchitectureTest, FinalFacadeHasNoWorkflowImplementationDependencies)
{
    const QString header = readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString source = readTextFile(QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));
    EXPECT_TRUE(source.contains(QStringLiteral("ProjectServiceContainer")));
    EXPECT_FALSE(header.contains(QStringLiteral("QFuture")));
    EXPECT_FALSE(header.contains(QStringLiteral("_pendingBaCameraMeta")));
    EXPECT_FALSE(source.contains(QStringLiteral("QMessageBox")));
    EXPECT_FALSE(source.contains(QStringLiteral("QtConcurrent")));
    EXPECT_FALSE(source.contains(QStringLiteral("new ProjectSparseReconstructionManager")));
}
```

- [x] **Step 2: Run the red test.** Run `cmake --build build/linux-source-release --target test_project_manager_architecture -j2` and record the expected failure because the new container/header does not exist yet; do not weaken the assertions.
- [x] **Step 3: Freeze the inventory.** Enumerate every PM public method, signal, direct member and external call site into the test’s expected categories: session, lifecycle, resource, task, cleanup, UI.
- [x] **Step 4: Add the boundary script.** `check_project_manager_boundary.py` must parse the same files, reject forbidden includes/members and report each remaining `ProjectManager*` owner dependency with file and line number. It must accept `--strict` and return exit code 1 on any violation.
- [x] **Step 5: Add the transitional rule.** During Tasks 2–7 the script may accept only a line containing `PROJECT_MANAGER_MIGRATION_SHIM` in an explicitly listed file; Task 8 removes the allow-list and runs the strict mode.

**Verification:**

```bash
python3 scripts/validation/check_project_manager_boundary.py --source-root .
cmake --build build/linux-source-release --target test_project_manager_architecture -j2
ctest --test-dir build/linux-source-release -R test_project_manager_architecture --output-on-failure
```

### Task 2 — 实现 `ProjectSession`，替换旧 `ProjectSessionFacade`

**Files:**

- Create `src/gui/project/services/ProjectSession.h` and `.cpp`.
- Create `tests/test_project_session.cpp`.
- Modify `src/gui/cmake/GuiSources.cmake` and `tests/CMakeLists.txt`.
- Modify current facade tests to exercise `ProjectSession`; keep a short compatibility test only until Task 9.
- Do not delete `ProjectSessionFacade.*` until all consumers are migrated in Task 9.

**Interfaces consumed/produced:**

`ProjectSession` must expose the following contract from the approved specification; it deliberately reuses the existing `ProjectSessionContext` and JSON value types rather than inventing GUI model DTOs:

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
    xjw::gui::project::ProjectSessionContext context() const;
    bool isCurrent(const xjw::gui::project::ProjectSessionContext& context) const;
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
    void sessionChanged(const xjw::gui::project::ProjectSessionContext& context);
};
```

The implementation must use the existing `src/gui/project/support/ProjectSessionContext.h`. `advanceGeneration()` is called by lifecycle/chunk signal handling before `sessionChanged`; no worker or `QFuture` is owned by this class. The old `ProjectSessionFacade` methods map one-for-one to the JSON/query/write-back methods above.

**Steps:**

- [x] **Step 1: Write the failing session tests.** Port the existing null-case tests and add this generation/write-back case to `test_project_session.cpp`:

```cpp
TEST_F(ProjectSessionTest, AdvanceGenerationRejectsOldContextBeforeWriteBack)
{
    const auto oldContext = session.context();
    session.advanceGeneration();
    EXPECT_FALSE(session.isCurrent(oldContext));
}
```

- [x] **Step 2: Run the red test.** Run the session test build baseline; the repository's existing core `test_project_session` target collision was recorded and the GUI target was used for the actual red/green cycle.
- [x] **Step 3: Implement the minimum service.** Copy the one-to-one query/write-back behavior from `ProjectSessionFacade`, store `ProjectData*` and `ProjectSessionContext`, connect lifecycle/chunk signals, and implement `advanceGeneration()` plus `isCurrent()` without adding task state.
- [x] **Step 4: Run green tests.** Run the GUI session/facade target and filtered CTest; migrate the session fixture and retain the target-collision evidence.
- [x] **Step 5: Prove coexistence without cross-task coupling.** The session fixture constructs `ProjectSession` beside the compatibility facade, and the facade forwards to the single `ProjectSession` implementation until Task 9.

**Verification:**

```bash
cmake --build build/linux-source-release --target test_project_session -j2
ctest --test-dir build/linux-source-release -R 'test_project_session|test_project_session_facade' --output-on-failure
```

### Task 3 — 隔离 UI 消息和输入

**Files:**

- Create `src/gui/project/services/ProjectUiMessageAdapter.h` and `.cpp`.
- Create `tests/test_project_ui_message_adapter.cpp` with a fake adapter.
- Modify `src/gui/cmake/GuiSources.cmake`, `tests/CMakeLists.txt`, and `ProjectUiCommands.*` to accept the adapter by reference.
- Keep file-dialog policy (last directory, filters, default suffix) in `ProjectUiCommands`; keep widget calls inside the Qt adapter only.

**Interfaces consumed/produced:**

```cpp
enum class UiAnswer { Yes, No, Cancel };

struct UiDialogResult
{
    bool accepted = false;
    QString text;
    QStringList texts;
    double number = 0.0;
};

class ProjectUiMessageAdapter
{
public:
    virtual ~ProjectUiMessageAdapter() = default;
    virtual void information(QWidget*, const QString& title, const QString& text) = 0;
    virtual void warning(QWidget*, const QString& title, const QString& text) = 0;
    virtual void critical(QWidget*, const QString& title, const QString& text) = 0;
    virtual UiAnswer question(QWidget*, const QString& title, const QString& text,
                              UiAnswer defaultAnswer) = 0;
    virtual UiDialogResult getText(QWidget*, const QString& title, const QString& label,
                                   const QString& initial) = 0;
    virtual UiDialogResult getDouble(QWidget*, const QString& title, const QString& label,
                                     double initial, double minimum, double maximum, int decimals) = 0;
    virtual UiDialogResult getItem(QWidget*, const QString& title, const QString& label,
                                   const QStringList& items, int current) = 0;
    virtual UiDialogResult selectOpenFile(QWidget*, const QString& title, const QString& directory,
                                          const QString& filter) = 0;
    virtual UiDialogResult selectOpenFiles(QWidget*, const QString& title, const QString& directory,
                                           const QString& filter) = 0;
    virtual UiDialogResult selectDirectory(QWidget*, const QString& title, const QString& directory) = 0;
    virtual UiDialogResult selectSaveFile(QWidget*, const QString& title, const QString& directory,
                                          const QString& filter) = 0;
};

class QtProjectUiMessageAdapter final : public ProjectUiMessageAdapter
{
public:
    explicit QtProjectUiMessageAdapter(QWidget* parentWidget);
    // Implementations delegate to Qt widgets; no service calls back into PM.
};
```

**Steps:**

- [x] **Step 1: Write the fake-adapter test.** The fake records calls and returns a queued answer:

```cpp
TEST(ProjectUiMessageAdapterTest, CancelledFileSelectionIsNotAccepted)
{
    FakeProjectUiMessageAdapter adapter;
    adapter.nextFileResult = UiDialogResult{};
    const UiDialogResult result = adapter.selectOpenFile(nullptr, QStringLiteral("打开"), {}, {});
    EXPECT_FALSE(result.accepted);
    EXPECT_TRUE(result.value.isEmpty());
    EXPECT_EQ(adapter.calls.size(), 1U);
}
```

- [x] **Step 2: Run the red test.** The target was registered and the initial missing-interface baseline was recorded by the implementer.
- [x] **Step 3: Implement Qt and fake adapters.** Qt return codes map to `UiAnswer`; every rejected input/file dialog returns `accepted=false`; dialog headers/calls are isolated to the adapter implementation.
- [x] **Step 4: Migrate callers.** `ProjectUiCommands` uses the adapter while retaining last-directory, filter, suffix, and dialog-mode policy.
- [x] **Step 5: Run green tests and the include audit.** Focused target/CTest and diff checks passed; the scoped reviewer independently confirmed the include audit.

**Verification:**

```bash
cmake --build build/linux-source-release --target test_project_ui_message_adapter -j2
ctest --test-dir build/linux-source-release -R test_project_ui_message_adapter --output-on-failure
```

### Task 4 — 抽取 `ProjectResourceCleanupCoordinator`

**Files:**

- Create `src/gui/project/services/ProjectResourceCleanupCoordinator.h` and `.cpp`.
- Create `tests/test_project_resource_cleanup_coordinator.cpp`.
- Reuse `src/core/project_workflows/ProjectResourceCleanup.*`, `ProjectResourceCleanupPlan.*`, and `ProjectResourceCleanupTransaction.*` without changing their core API.
- Modify `ProjectManager.*` only to route cleanup calls through the coordinator; remove the corresponding PM private fields after the route is green.
- Update `src/gui/cmake/GuiSources.cmake` and `tests/CMakeLists.txt`.

**Interfaces consumed/produced:**

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
    void deleteGeneratedData(const QString& section,
                             const QStringList& resourcePaths,
                             QWidget* requestWidget = nullptr);
    void waitForFinished();

signals:
    void progressChanged(const QString& taskId, int value, int maximum);
    void finished(const QString& taskId);
};
```

The coordinator owns the `QFuture`, watcher, mutex/lock and `ProjectSessionContext`. It must perform `prepare -> execute -> finalize` exactly once, release the lock on success, failure, cancellation and destruction, and reject a second operation while busy. `deleteGeneratedData()` preserves the existing PM request-widget disable/restore behavior; portable export uses `rejectLifecycleChange()` rather than a second cleanup implementation.

**Steps:**

- [x] **Step 1: Write the coordinator state-machine test.** Use a fake cleanup executor with gates for prepare/execute/finalize:

```cpp
TEST_F(ProjectResourceCleanupCoordinatorTest, RejectsSecondRequestAndRestoresBusyState)
{
    fakeExecutor.pauseExecute = true;
    coordinator.deleteGeneratedData(QStringLiteral("models"), {QStringLiteral("mesh.obj")});
    EXPECT_TRUE(coordinator.isRunning());
    EXPECT_TRUE(coordinator.rejectLifecycleChange(QStringLiteral("关闭项目")));
    coordinator.deleteGeneratedData(QStringLiteral("models"), {QStringLiteral("other.obj")});
    fakeExecutor.releaseExecute();
    QTRY_VERIFY_WITH_TIMEOUT(!coordinator.isRunning(), 1000);
    EXPECT_TRUE(requestWidget->isEnabled());
}
```

- [x] **Step 2: Run the red test.** The missing coordinator/fake baseline was recorded before implementation.
- [x] **Step 3: Extract the existing state.** PM cleanup future, continuation, request-widget handling and prepare/execute/finalize sequencing now live in the coordinator; core signatures are unchanged.
- [x] **Step 4: Add stale-session and failure cases.** Captured `ProjectSessionContext` gates UI results while finalization and busy-state restoration run on all paths, including destructor wait.
- [x] **Step 5: Run green tests.** Focused coordinator, integration, architecture and diff checks passed; PM cleanup fields and direct cleanup ownership were removed.

**Verification:**

```bash
cmake --build build/linux-source-release --target test_project_resource_cleanup_coordinator -j2
ctest --test-dir build/linux-source-release -R test_project_resource_cleanup_coordinator --output-on-failure
```

### Task 5 — 抽取生命周期和资源服务

**Files:**

- Create `src/gui/project/services/ProjectLifecycleService.h/.cpp`.
- Create `src/gui/project/services/ProjectResourceService.h/.cpp`.
- Create `tests/test_project_lifecycle_service.cpp` and `tests/test_project_resource_service.cpp`.
- Modify `ProjectLifecycleController.*`, `ProjectUiCommands.*`, and the import-related portions of `ProjectManager.*` as migration sources; delete old controller files only in Task 9.
- Update `src/gui/cmake/GuiSources.cmake` and `tests/CMakeLists.txt`.

**Interfaces consumed/produced:**

`ProjectLifecycleService` owns create/open/save/close/export lifecycle sequencing and emits:

```cpp
signals:
    void projectOpened(ProjectData*);
    void projectClosed();
    void projectModified();
    void lifecycleError(const QString& operation, const QString& message);
```

`ProjectResourceService` owns photo/folder/asset/reference/Survey import and resource export. Its public methods must receive `ProjectSession*`, `ProjectUiMessageAdapter*` and `ProjectResourceCleanupCoordinator*` through construction, and return `OperationResult` (success, cancelled, error message) rather than mutating PM flags.

**Steps:**

- [x] **Step 1: Define the result type and write red tests.** Add a small value type in `ProjectResourceService.h`:

```cpp
enum class OperationStatus { Success, Cancelled, Failed };

struct OperationResult
{
    OperationStatus status = OperationStatus::Failed;
    QString message;
    bool succeeded() const { return status == OperationStatus::Success; }
};
```

Then assert that a fake adapter returning `accepted=false` yields `OperationStatus::Cancelled` and leaves the fake `ProjectData` mutation count at zero.
- [x] **Step 2: Run the red tests.** Build `test_project_lifecycle_service test_project_resource_service`; the targets must fail before the services and result type are present.
- [x] **Step 3: Extract lifecycle branches.** Move create/open/save/close/export sequencing into `ProjectLifecycleService`; keep the old `ProjectSessionContext` valid until a `ProjectData` transition succeeds, and only emit `projectOpened` after the new `ProjectData` is valid.
- [x] **Step 4: Extract resource branches.** Move image/folder/asset/reference/Survey/remove/pack branches into `ProjectResourceService`, preserving path normalization and core calls; return `OperationResult` for every user-cancellable command and keep expensive validation/import/staging work off the GUI thread.
- [x] **Step 5: Gate portable export and verify green.** Use the shared `ProjectSession` busy gate plus `cleanup->rejectLifecycleChange()` before export/close, use the adapter for all prompts, make sidecar/staging writes transactional, and test success, invalid path, cancellation, duplicate import and close while busy.

**Verification:**

```bash
cmake --build build/linux-source-release --target test_project_lifecycle_service test_project_resource_service -j2
ctest --test-dir build/linux-source-release -R 'test_project_(lifecycle|resource)_service' --output-on-failure
```

### Task 6 — 建立任务上下文和编排器组合根

**Files:**

- Create `src/gui/project/tasks/ProjectTaskContext.h`.
- Create `src/gui/project/tasks/ProjectTaskOrchestrator.h/.cpp`.
- Create `src/gui/project/services/ProjectServiceContainer.h/.cpp`.
- Create `tests/test_project_task_orchestrator.cpp`.
- Modify all existing workflow controller headers/cpps to receive the context/ports instead of constructing or querying PM directly:
  `ProjectSparseReconstructionManager.*`, `ProjectPointCloudWorkflowController.*`, `ProjectModelManager.*`, `ProjectTerrainProductsManager.*`, `ProjectCameraSetupManager.*`, `ProjectMaskWorkflowController.*`.
- Update `src/gui/cmake/GuiSources.cmake` and `tests/CMakeLists.txt`.

**Interfaces consumed/produced:**

```cpp
struct ProjectTaskContext
{
    QString taskId;
    xjw::gui::project::ProjectSessionContext session;
    std::shared_ptr<std::atomic<bool>> cancelFlag;
};

class ProjectTaskOrchestrator final : public QObject
{
    Q_OBJECT
public:
    ProjectTaskOrchestrator(ProjectSession* session,
                            ProjectUiMessageAdapter* messages,
                            QObject* parent = nullptr);
    ProjectTaskContext context(const QString& taskId) const;
    bool hasActiveTask() const noexcept;
    void cancelActiveTask();
    void invalidateForSessionChange();
    // Each existing workflow entry point keeps its current QJsonObject or
    // ProjectTerrainRequests value type; the orchestrator adds the context gate.
signals:
    void progressChanged(const QString& stage, int percent);
    void taskStarted(const QString& taskId);
    void taskFinished(const QString& taskId, bool success);
    void taskError(const QString& taskId, const QString& message);
};
```

`ProjectServiceContainer` is the only composition root. It owns one session, one UI adapter, one lifecycle service, one resource service, one cleanup coordinator, one task orchestrator and the BA controller. It must not own a second copy of `ProjectData`.

**Steps:**

- [x] **Step 1: Write the stale-result test.** A fake task captures the context at start and reports completion after `session->advanceGeneration()`:

```cpp
TEST_F(ProjectTaskOrchestratorTest, CompletionFromOldContextIsDropped)
{
    const ProjectTaskContext oldContext = orchestrator.context(QStringLiteral("sfm"));
    session.advanceGeneration();
    fakeTask.complete(oldContext);
    EXPECT_EQ(fakeProjectData.writeCount(), 0);
    EXPECT_EQ(spy.count(), 0);
}
```

- [x] **Step 2: Run the red test.** Build `test_project_task_orchestrator`; it must fail until the context gate exists.
- [x] **Step 3: Introduce context and migrate one controller.** Replace PM owner access in `ProjectSparseReconstructionManager` with `ProjectSessionContext`, shared cancel flag and narrow callbacks; keep the triangulation algorithm and signal signatures unchanged.
- [ ] **Step 4: Migrate remaining controllers one at a time.** Apply the same constructor dependency rule to point cloud, model, terrain, camera and mask controllers; preserve queued connections and QObject parents.
- [x] **Step 5: Move construction/wiring into the orchestrator/container.** Move the migrated sparse workflow into the container, forward its legacy signals once, and make `sessionChanged` cancel/invalidate its active context. Remaining controllers stay explicitly scoped for Tasks 7–8.
- [x] **Step 6: Run green tests before BA extraction.** Run the target and workflow filter below; do not start Task 7 while this test or an existing workflow test is failing.

**Verification:**

```bash
cmake --build build/linux-source-release --target test_project_task_orchestrator -j2
ctest --test-dir build/linux-source-release -R 'test_project_task_orchestrator|Sparse|PointCloud|Terrain|Camera|Mask' --output-on-failure
```

### Task 7 — 抽取 `ProjectBundleAdjustController`

**Files:**

- Create `src/gui/project/tasks/ProjectBundleAdjustController.h/.cpp`.
- Create `tests/test_project_bundle_adjust_controller.cpp`.
- Move the BA state, cancel flag, preview maps/results, async continuation and accept/discard logic from `ProjectManager.cpp` into the new controller.
- Modify `ProjectTaskOrchestrator.*`, `ProjectServiceContainer.*`, `ProjectManager.*`, `src/gui/cmake/GuiSources.cmake`, and `tests/CMakeLists.txt`.

**Interfaces consumed/produced:**

```cpp
class ProjectBundleAdjustController final : public QObject
{
    Q_OBJECT
public:
    ProjectBundleAdjustController(ProjectSession* session,
                                  ProjectUiMessageAdapter* messages,
                                  QObject* parent = nullptr);
    void startAsync(const QStringList& images,
                    const QString& outputDir,
                    int threads,
                    bool dryRun,
                    const QJsonObject& extraSettings);
    bool acceptPreview(QString* errorMessage);
    void discardPreview();
    bool cancel();
    bool isRunning() const noexcept;
    bool hasPendingPreview() const noexcept;
signals:
    void progressChanged(const QString& stage, int percent);
    void previewReady(const QJsonObject& preview);
    void finished(bool success, const QString& message);
    void error(const QString& message);
};
```

The controller stores no raw `ProjectManager*`; preview data is tagged with `ProjectSessionContext` and is discarded automatically on `sessionChanged`.

**Steps:**

- [x] **Step 1: Write the preview lifecycle test.** Use the existing BA service fake and JSON payloads:

```cpp
TEST_F(ProjectBundleAdjustControllerTest, SessionChangeDiscardsPendingPreview)
{
    controller.startAsync({QStringLiteral("a.jpg")}, tempDir.path(), 1, true, {});
    fakeBa.completeWithPreview(QJsonObject{{QStringLiteral("output_dir"), tempDir.path()}});
    QTRY_VERIFY(controller.hasPendingPreview());
    session.advanceGeneration();
    EXPECT_FALSE(controller.hasPendingPreview());
    EXPECT_FALSE(controller.acceptPreview(nullptr));
}
```

- [x] **Step 2: Run the red test.** Build `test_project_bundle_adjust_controller`; it must fail until the extracted controller exists.
- [x] **Step 3: Extract without semantic edits.** Move PM’s `startBundleAdjustAsync`, cancel path, preview maps/result JSON, presentation and accept/discard write-back into the controller; retain the current `QStringList`, `QJsonObject`, thread count and dry-run parameters.
- [x] **Step 4: Add token/cancel guards.** Tag each worker callback with `ProjectTaskContext`; clear preview on cancel, failure, session change and destruction before emitting the existing preview signal.
- [x] **Step 5: Verify signal order and remove duplicate state.** Use `QSignalSpy` to assert one progress stream and one finished signal; only after green tests remove all BA fields/helpers from PM.

**Verification:**

```bash
cmake --build build/linux-source-release --target test_project_bundle_adjust_controller -j2
ctest --test-dir build/linux-source-release -R 'test_project_bundle_adjust_controller|BundleAdjust|AerialTriangulation' --output-on-failure
```

### Task 8 — 完成所有 controller 的 session/任务端口迁移

**Files:**

- Modify the exact controller/task inventory discovered after Task 7:
  - `src/gui/project/manager/ProjectSparseReconstructionManager.{h,cpp}`
  - `src/gui/project/manager/ProjectMaskWorkflowController.{h,cpp}`
  - `src/gui/project/manager/ProjectPointCloudWorkflowController.{h,cpp}`
  - `src/gui/project/manager/ProjectModelManager.{h,cpp}`
  - `src/gui/project/manager/ProjectTerrainProductsManager.{h,cpp}` and `ProjectTerrainRpcProducts.cpp`
  - `src/gui/project/manager/ProjectCameraSetupManager.{h,cpp}`
  - `src/gui/project/tasks/ProjectTaskOrchestrator.{h,cpp}`
- Audit, but do not modify merely for uniformity, `ProjectMaskInferenceAdapter.{h,cpp}` and
  `ProjectTaskContext.h`; they already have the intended dependency direction.
- Modify the supporting boundaries required by those controllers:
  `src/common/project/ProjectSessionModel.{h,cpp}` (only for truthful, atomic result/save primitives consumed by the
  guarded GUI-session ports),
  `src/gui/project/services/ProjectSession.{h,cpp}`,
  `src/gui/project/services/ProjectTiePointResultService.{h,cpp}` (only to stage/validate the tie-point replacement,
  commit it with the BA metadata transaction, and run post-commit cleanup),
  `src/gui/project/services/ProjectServiceContainer.{h,cpp}`,
  `src/gui/project/tasks/ProjectBundleAdjustController.{h,cpp}` (only to make delayed preview acceptance one
  context-aware camera/BA/tie-point transaction with truthful retry/discard behavior),
  `src/gui/project/support/ProjectBundleAdjustWorkflow.{h,cpp}` and
  `src/gui/project/support/ProjectMetadataOperations.{h,cpp}` only if required to replace the old split BA/tie-point
  commit helper without duplicating record construction or changing its existing public utility behavior,
  `src/gui/project/support/ProjectModelTaskLifecycle.{h,cpp}`,
  `src/gui/project/support/ProjectCameraInitialization.{h,cpp}`, and
  `src/gui/project/support/ProjectSfmWorkflow.{h,cpp}`.
- Modify `src/gui/project/manager/ProjectManager.{h,cpp}` only for the minimal composition/signal bridge needed to
  transfer controller ownership and workflow state to the orchestrator.  The full façade rewrite and external UI caller
  migration remain Task 9.
- Modify `scripts/validation/check_project_manager_boundary.py`, `tests/CMakeLists.txt`, `tests/test_project_data.cpp`,
  `tests/test_project_task_orchestrator.cpp`, `tests/test_project_model_task_lifecycle.cpp`,
  `tests/test_project_manager_architecture.cpp`, `tests/test_gui_project_utils.cpp`, and
  `tests/test_source_contracts.cpp`.  Reuse existing targets unless a genuinely new behavioral fixture is required.
- If implementation discovers another production dependency, add that exact path to this plan section before editing
  it; the audit output is not a license for an unreviewed dependency.

**Interfaces consumed/produced:**

- No new public API. The produced invariant is that task/resource service implementation files depend on `ProjectSession`, `ProjectTaskContext`, core interfaces and the UI adapter only; they do not depend on `ProjectManager.h`.
- Any unavoidable legacy call must be isolated in one named adapter function and covered by a removal assertion; no scattered compatibility calls are allowed.
- `ProjectSession` provides generation-guarded, GUI-thread write ports so validation and mutation are one operation;
  controllers may not retain `ProjectData*` or perform a check-then-write sequence.
- Delayed BA acceptance uses a narrow `ProjectData` metadata-stage token behind `ProjectSession`: camera updates and the
  BA record are staged silently, then either rolled back without observable mutation or made visible once together with
  the tie-point replacement. Discard/cancel must never leave the pre-writer camera/BA half committed.
- The orchestrator keeps keyed lanes instead of forcing every workflow through one global context. Sparse/BA remain
  mutually exclusive, the automatic point-cloud-to-model chain shares one context, and the existing independent DEM
  and ortho lanes remain independent.

**Steps:**

- [x] Run the boundary script to produce the exact remaining dependency list. The post-Task-7 global baseline is 97
  findings: 39 in the transitional PM façade/composition root, 16 in Task 8 controllers, and 42 in Task 9 UI callers.
- [x] Add a strict Task-8 scope to the boundary script. It must scan the exact controller/task inventory, including split
  implementation files, and reject `ProjectManager`, retained `ProjectData*`, direct generic dialogs, and unguarded
  async write-back. Do not add allow-list shims to make the scope green.
- [x] Add generation-guarded session writes and finish the sparse migration; use this as the reference pattern.
- [ ] Migrate mask, point-cloud/model, terrain, and camera as serial bounded chunks; immediately run each chunk's
  focused tests and preserve public signal names.
- [ ] Replace PM callbacks used only for errors/progress with the context ports and ensure errors include operation and path.
- [ ] Remove transitional macros/shims as soon as the last consumer is migrated.
- [ ] Require the Task-8 scoped strict audit to pass with zero controller/task findings. Also run the global strict audit
  and record that every remaining finding belongs to the enumerated Task-9 façade/composition/UI caller inventory;
  global zero remains the Task-9 gate and the audit itself must not be weakened.

**Verification:**

```bash
python3 scripts/validation/check_project_manager_boundary.py --source-root . --scope task-implementations --strict
python3 scripts/validation/check_project_manager_boundary.py --source-root . --strict
cmake --build build/linux-source-release --target test_gui_project_utils -j2
ctest --test-dir build/linux-source-release -R 'Gui|Project|Workflow|SourceContract' --output-on-failure
```

### Task 9 — 将 `ProjectManager` 收缩为 façade 并迁移外部调用方

**Files:**

- Rewrite `src/gui/project/manager/ProjectManager.h/.cpp` around `std::unique_ptr<ProjectServiceContainer>` and the existing QObject signal surface.
- Modify `src/gui/mainwindow/MainWindow.*`, `src/gui/mainwindow/*`, `src/gui/project/dialogs/*`, `src/gui/project/*`, and any caller reported by `rg -n 'ProjectManager|_projectManager->' src tests`.
- Modify `src/gui/cmake/GuiSources.cmake` and `tests/CMakeLists.txt`.
- Delete `ProjectSessionFacade.*` and `ProjectLifecycleController.*` only after all references are gone; update includes and install/source lists.

**Interfaces consumed/produced:**

`ProjectManager` keeps only façade accessors and UI-facing signals. Public methods that remain must be one-line delegations to a service and must not expose service internals. New preferred accessors are:

```cpp
ProjectSession& session() const;
ProjectTaskOrchestrator& tasks() const;
ProjectResourceService& resources() const;
ProjectResourceCleanupCoordinator& cleanup() const;
```

Task and bundle-adjust commands go through `tasks()` so the orchestrator-owned admission policy cannot be bypassed.
Do not add a raw mutable `ProjectBundleAdjustController` façade accessor; observation-only access, if a concrete caller
proves it is needed, must be const and must not duplicate command entry points.

Methods that are not needed by external UI callers are deleted, not kept as no-op compatibility methods. For callers that need a transition, use a named façade delegation and mark it for removal in the same task; do not preserve a second implementation.

**Steps:**

- [ ] **Step 1: Write the façade delegation test.** Construct PM with a fake `ProjectData` and assert that the accessor points at the container-owned session and that a lifecycle signal is forwarded once:

```cpp
TEST_F(ProjectManagerFacadeTest, ExposesContainerServicesAndForwardsOpenOnce)
{
    ProjectManager manager(&projectData, nullptr);
    ASSERT_NE(&manager.session(), nullptr);
    QSignalSpy openedSpy(&manager, &ProjectManager::projectOpened);
    QMetaObject::invokeMethod(&projectData,
                              "projectOpened",
                              Qt::DirectConnection,
                              Q_ARG(QString, projectPath));
    EXPECT_EQ(openedSpy.count(), 1);
}
```

- [ ] **Step 2: Run the red/source test.** Build `test_gui_project_utils` and run the strict boundary script; it must fail while PM still owns individual controllers and forbidden state.
- [ ] **Step 3: Replace constructor ownership.** Make PM create only `ProjectServiceContainer`, connect the container’s typed signals once, and expose the five service accessors defined above.
- [ ] **Step 4: Migrate external callers.** Use `rg -n 'ProjectManager|_projectManager->' src tests` to update MainWindow, menus, dialogs, Browser bridge and task status code; each call moves to the matching service accessor or façade signal.
- [ ] **Step 5: Delete duplicate internals.** Remove futures, locks, generation/preview/cancel fields and direct dialog includes; delete `ProjectSessionFacade.*` and `ProjectLifecycleController.*` only after a zero-reference `rg` check.
- [ ] **Step 6: Run green target/tests.** Require one signal emission per source event and a passing strict boundary audit before proceeding to documentation/full regression.

**Verification:**

```bash
python3 scripts/validation/check_project_manager_boundary.py --source-root . --strict
cmake --build build/linux-source-release --target plascan_gui test_gui_project_utils -j2
ctest --test-dir build/linux-source-release -R 'Gui|Project|SourceContract' --output-on-failure
```

### Task 10 — 文档、全量回归和收尾审计

**Files:**

- Modify `docs/PROJECT_ARCHITECTURE.md` to describe the new service graph, ownership and cancellation rules.
- Modify `docs/superpowers/specs/2026-09-19-project-manager-rewrite-design.md` status from approved design to implemented/verified only after evidence exists.
- Update `task_plan.md`, `findings.md`, and `progress.md` with actual commands/results; do not erase historical entries.
- Format only modified C++ files and review `git diff --check`.

**Steps:**

- [ ] Run the strict boundary audit and verify no forbidden PM dependency remains.
- [ ] Wait for/confirm existing unrelated build processes are finished before starting the canonical build.
- [ ] Configure/build/test the native Linux Release preset with the repository’s standard command.
- [ ] Run the relevant GUI/project test filter and then the full local test suite; record skipped tests and external dependency failures explicitly.
- [ ] Manually inspect lifecycle, cancel, stale-session and dialog-cancel paths against the five Review Focus invariants.
- [ ] Confirm no temporary artifacts outside `build/tmp/<task-name>/`; clean only artifacts created by this task.
- [ ] Re-check `git status --short` and ensure no commit, push, tag or release was performed without explicit authorization.

**Verification:**

```bash
python3 scripts/validation/check_project_manager_boundary.py --source-root . --strict
python3 scripts/env/configure_with_env.py --source-deps --build --test
python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure -R 'Gui|Project|Workflow|SourceContract'
python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
git diff --check
```

## Definition of Done

- `ProjectManager.cpp` no longer contains task implementation, cleanup transaction orchestration, BA preview state, generation/cancellation state or direct Qt dialogs; it is a thin façade whose constructor creates only `ProjectServiceContainer`.
- All service/task implementations compile without including `ProjectManager.h`, except the explicitly documented façade integration boundary.
- Session-token checks cover every asynchronous continuation and stale results cannot mutate a replacement project.
- Cleanup/export mutual exclusion and cancellation are tested for success, failure, cancellation and destruction.
- Existing GUI/project/workflow tests pass, new service tests pass, the strict boundary audit passes, and the canonical local build/test command has a recorded result.
- `docs/PROJECT_ARCHITECTURE.md` and task tracking files describe the actual final topology; no obsolete `ProjectSessionFacade`/`ProjectLifecycleController` references remain.

## Spec Coverage Audit

| Spec requirement | Covered by |
| --- | --- |
| Thin PM façade with no future/generation/BA/AT/cleanup state | Tasks 1, 7, 9 |
| Unified `ProjectSessionContext` gate on lifecycle and async write-back | Tasks 2, 5, 6, 7, 8 |
| Core cleanup prepare/execute/finalize and WAL recovery preserved | Task 4 and Task 10 |
| Resource import/export/delete/pack moved to services | Task 5 and Task 9 |
| No direct Qt dialog dependency in business services | Task 3 and Task 8 |
| All production callers and source contracts migrated | Tasks 8, 9 and 10 |
| Independent behavior tests plus Linux build/full CTest evidence | Every task’s verification block and Task 10 |

## Execution Choice

The plan is intentionally written for either `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Because Tasks 2–9 share headers, signal contracts and the same dirty worktree, native sequential execution is the safer default; subagent execution is reasonable only if each agent is assigned exclusive file ownership and the integration steps remain in order.
