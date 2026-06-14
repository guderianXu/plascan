# Sparse Reconstruction Menu Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `重建 -> 稀疏重建` into a production sparse workflow plus `高级工具`, and add a sparse-only `空中三角测量...` entry that stops after official SfM/BA sparse output.

**Architecture:** Keep `MainMenu` responsible only for QAction/menu structure, keep `MainWindow` responsible for signal wiring, and add the sparse-only workflow coordination to `MenuWorkflowController` by reusing `SFMService`. Use existing sparse quality helpers for production gating so preview clouds cannot enter post-processing or downstream production flows.

**Tech Stack:** C++17, Qt6 Widgets, CMake AUTOUIC/AUTOMOC, GTest-style GUI utility tests, existing `SFMService`, existing project metadata helpers.

---

## File Structure

- Modify `src/gui/menu/MainMenu.h`
  - Add an accessor and member for `空中三角测量...`.
  - Update comments so `triangulateAction()` is explicitly the two-view preview tool.
- Modify `src/gui/menu/MainMenu.cpp`
  - Build `稀疏重建主流程` as the first group.
  - Add `高级工具` submenu under `稀疏重建`.
  - Add `空中三角测量...` action in the main sparse flow.
  - Move `查看匹配`, observation network, camera initialization, two-view preview, and standalone BA into `高级工具`.
- Modify `src/gui/main_window/MainWindow.ui`
  - Mirror the menu/action structure for the UI-backed runtime path.
  - Add `actionAerialTriangulation`.
- Modify `src/gui/main_window/MainWindow.cpp`
  - Wire `actionAerialTriangulation` to a sparse-only workflow slot.
  - Keep `actionTriangulate` wired to `openTriangulationDialog()`.
- Modify `src/gui/main_window/MenuWorkflowController.h`
  - Add `openAerialTriangulationDialog()`.
  - Add `startAerialTriangulationWorkflow(const QJsonObject &settings)`.
  - Add helper declarations for prerequisite summary and formal result appending.
- Modify `src/gui/main_window/MenuWorkflowController.cpp`
  - Reuse `ThreeDReconstructionDialog` in sparse-only mode or set sparse-only labels after adding dialog support.
  - Check missing upstream data before running.
  - Show an auto-fill/manual-choice prompt when upstream data is missing.
  - Run `SFMService` and append only official sparse result metadata.
  - Do not call dense, refine, mesh, DEM, or DOM stages.
- Modify `src/gui/dialogs/ThreeDReconstructionDialog.h`
  - Add a `Mode` enum and `setMode(Mode mode)` so the same core-parameter dialog can present either full 3D reconstruction or sparse-only aerial triangulation.
- Modify `src/gui/dialogs/ThreeDReconstructionDialog.cpp`
  - Update title, description, start button text, and `export_obj` visibility for sparse-only mode.
- Modify `src/gui/dialogs/ThreeDReconstructionDialog.ui`
  - Name labels/buttons clearly enough for code to update them.
- Modify `src/gui/config/settings/DialogSettingKeys.h`
  - Reuse existing `AerialTriangulation` key for sparse-only settings persistence.
- Modify `src/gui/dialogs/SparseCloudPostProcessDialog.cpp`
  - Ensure only production sparse results are listed.
  - Skip preview clouds even when passed in available AT results.
- Modify `tests/test_gui_project_utils.cpp`
  - Add source-level and object-level tests for menu grouping, action labels, sparse-only dialog mode, aerial triangulation wiring, sparse-only workflow stopping point, and post-process input filtering.

---

### Task 1: Add Menu Tests For Main Flow And Advanced Tools

**Files:**
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add a failing menu structure test**

Add this test near the existing `MainMenuTest` cases:

```cpp
TEST(MainMenuTest, SparseReconstructionSeparatesMainFlowAndAdvancedTools)
{
    QMainWindow window;
    MainMenu menu(&window);

    ASSERT_NE(menu.detectFeaturesAction(), nullptr);
    ASSERT_NE(menu.vocabularyOverlapAction(), nullptr);
    ASSERT_NE(menu.matchFeaturesAction(), nullptr);
    ASSERT_NE(menu.aerialTriangulationAction(), nullptr);
    ASSERT_NE(menu.sparseCloudPostProcessAction(), nullptr);
    ASSERT_NE(menu.viewMatchesAction(), nullptr);
    ASSERT_NE(menu.buildObsNetworkAction(), nullptr);
    ASSERT_NE(menu.initCameraPoseAction(), nullptr);
    ASSERT_NE(menu.triangulateAction(), nullptr);
    ASSERT_NE(menu.reconBundleAdjustAction(), nullptr);

    EXPECT_EQ(menu.aerialTriangulationAction()->text(), QStringLiteral("空中三角测量..."));
    EXPECT_EQ(menu.triangulateAction()->text(), QStringLiteral("生成两视预览云..."));
    EXPECT_FALSE(menu.triangulateAction()->text().contains(QStringLiteral("空中三角")));

    const QString source = readProjectSourceFile(QStringLiteral("src/gui/menu/MainMenu.cpp"));
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(source.contains(QStringLiteral("addMenu(tr(\"高级工具\"))")));
    EXPECT_LT(source.indexOf(QStringLiteral("addAction(tr(\"空中三角测量...\"))")),
              source.indexOf(QStringLiteral("addMenu(tr(\"高级工具\"))")));
    EXPECT_GT(source.indexOf(QStringLiteral("addAction(tr(\"生成两视预览云...\"))")),
              source.indexOf(QStringLiteral("addMenu(tr(\"高级工具\"))")));
}
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set PATH=%CD%\build\windows-existingdeps-release\bin;E:\code\plascan\build\windows-vcpkg-release\vcpkg_installed\x64-windows\bin;E:\code\plascan\build\env\libtorch-cu130\libtorch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH% && set QT_QPA_PLATFORM=offscreen && .\build\windows-existingdeps-release\tests\test_gui_project_utils.exe --gtest_filter=MainMenuTest.SparseReconstructionSeparatesMainFlowAndAdvancedTools"
```

Expected: fails to compile because `MainMenu::aerialTriangulationAction()` does not exist, or fails because the action/menu text is not present.

- [ ] **Step 3: Add the action accessor and member**

In `src/gui/menu/MainMenu.h`, add this public accessor with the other sparse reconstruction actions:

```cpp
QAction *aerialTriangulationAction() const;
```

Update the action member block:

```cpp
QAction *m_aerialTriangulationAct{};   ///< 正式空中三角测量（SfM + BA）
QAction *m_triangulateAct{};           ///< 两视预览云三角化
```

- [ ] **Step 4: Rebuild the fallback menu structure**

In `src/gui/menu/MainMenu.cpp`, replace the sparse reconstruction block with:

```cpp
auto *sparseReconMenu = reconMenu->addMenu(tr("稀疏重建"));
m_detectFeaturesAct = sparseReconMenu->addAction(tr("特征点提取"));
m_vocabularyOverlapAct = sparseReconMenu->addAction(tr("重叠对规划..."));
m_matchFeaturesAct  = sparseReconMenu->addAction(tr("连接点匹配"));
m_aerialTriangulationAct = sparseReconMenu->addAction(tr("空中三角测量..."));
m_sparseCloudPostProcessAct = sparseReconMenu->addAction(tr("稀疏点云后处理..."));

sparseReconMenu->addSeparator();
auto *advancedSparseMenu = sparseReconMenu->addMenu(tr("高级工具"));
m_viewMatchesAct = advancedSparseMenu->addAction(tr("查看匹配"));
m_buildObsNetworkAct = advancedSparseMenu->addAction(tr("构建观测网络..."));
m_initCameraPoseAct = advancedSparseMenu->addAction(tr("初始化相机位姿..."));
m_triangulateAct = advancedSparseMenu->addAction(tr("生成两视预览云..."));
m_reconBundleAdjustAct = advancedSparseMenu->addAction(tr("单独光束法平差..."));
```

Remove the later `m_viewMatchesAct = toolsMenu->addAction(tr("连接点查看"));` line from the top-level Tools menu. Keep the rest of the Tools menu unchanged.

Add the accessor implementation:

```cpp
QAction *MainMenu::aerialTriangulationAction() const { return m_aerialTriangulationAct; }
```

- [ ] **Step 5: Update UI-backed menu structure**

In `src/gui/main_window/MainWindow.ui`, add `actionAerialTriangulation` and a `menuSparseAdvancedTools` submenu under `menuSparseReconstruction`.

The sparse menu action order should be:

```xml
<addaction name="actionDetectFeatures"/>
<addaction name="actionVocabularyOverlap"/>
<addaction name="actionMatchFeatures"/>
<addaction name="actionAerialTriangulation"/>
<addaction name="actionSparseCloudPostProcess"/>
<addaction name="separator"/>
<addaction name="menuSparseAdvancedTools"/>
```

The advanced submenu should contain:

```xml
<property name="title">
 <string>高级工具</string>
</property>
<addaction name="actionViewMatches"/>
<addaction name="actionBuildObsNetwork"/>
<addaction name="actionInitCameraPose"/>
<addaction name="actionTriangulate"/>
<addaction name="actionReconBundleAdjust"/>
```

Add the action text:

```xml
<action name="actionAerialTriangulation">
 <property name="text">
  <string>空中三角测量...</string>
 </property>
</action>
```

Update existing action texts:

```xml
<string>重叠对规划...</string>
<string>连接点匹配</string>
<string>查看匹配</string>
<string>构建观测网络...</string>
<string>单独光束法平差...</string>
```

- [ ] **Step 6: Run the focused test and verify it passes**

Run the same command from Step 2.

Expected: `MainMenuTest.SparseReconstructionSeparatesMainFlowAndAdvancedTools` passes.

- [ ] **Step 7: Commit**

```powershell
git add src/gui/menu/MainMenu.h src/gui/menu/MainMenu.cpp src/gui/main_window/MainWindow.ui tests/test_gui_project_utils.cpp
git commit -m "feat: separate sparse reconstruction menu flow"
```

---

### Task 2: Add Sparse-Only Dialog Mode

**Files:**
- Modify: `src/gui/dialogs/ThreeDReconstructionDialog.h`
- Modify: `src/gui/dialogs/ThreeDReconstructionDialog.cpp`
- Modify: `src/gui/dialogs/ThreeDReconstructionDialog.ui`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add a failing dialog mode test**

Add this test near `ThreeDReconstructionDialogTest`:

```cpp
TEST(ThreeDReconstructionDialogTest, AerialTriangulationModeUsesSparseOnlyLabels)
{
    ThreeDReconstructionDialog dialog;
    dialog.setMode(ThreeDReconstructionDialog::Mode::AerialTriangulation);
    dialog.setImageCount(12);
    dialog.setDefaultOutputDir(QStringLiteral("E:/tmp/at"));

    EXPECT_EQ(dialog.windowTitle(), QStringLiteral("空中三角测量"));

    const QJsonObject settings = dialog.collectSettings();
    EXPECT_EQ(settings.value(QStringLiteral("workflow_kind")).toString(),
              QStringLiteral("aerial_triangulation"));
    EXPECT_FALSE(settings.value(QStringLiteral("export_obj")).toBool(true));

    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/ThreeDReconstructionDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(source.contains(QStringLiteral("一键生成正式 SfM/BA 稀疏云")));
    EXPECT_TRUE(source.contains(QStringLiteral("m_exportObjCheck->setVisible(false)")));
}
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set PATH=%CD%\build\windows-existingdeps-release\bin;E:\code\plascan\build\windows-vcpkg-release\vcpkg_installed\x64-windows\bin;E:\code\plascan\build\env\libtorch-cu130\libtorch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH% && set QT_QPA_PLATFORM=offscreen && .\build\windows-existingdeps-release\tests\test_gui_project_utils.exe --gtest_filter=ThreeDReconstructionDialogTest.AerialTriangulationModeUsesSparseOnlyLabels"
```

Expected: compile failure because `setMode` and `Mode` do not exist.

- [ ] **Step 3: Add the dialog mode API**

In `src/gui/dialogs/ThreeDReconstructionDialog.h`, add:

```cpp
enum class Mode
{
    ThreeDReconstruction,
    AerialTriangulation
};

void setMode(Mode mode);
```

Add members:

```cpp
Mode m_mode = Mode::ThreeDReconstruction;
QLabel *m_titleLabel = nullptr;
QPushButton *m_browseBtn = nullptr;
```

- [ ] **Step 4: Name the UI controls for mode-specific labels**

In `src/gui/dialogs/ThreeDReconstructionDialog.ui`, ensure the existing title label is named `m_titleLabel` instead of `titleLabel`, and ensure the browse button is named `m_browseBtn`.

- [ ] **Step 5: Implement sparse-only mode**

In `ThreeDReconstructionDialog::setupUi()`, store the new pointers:

```cpp
m_titleLabel = form.m_titleLabel;
m_browseBtn = form.m_browseBtn;
```

Replace the browse connect with:

```cpp
connect(m_browseBtn, &QPushButton::clicked, this, &ThreeDReconstructionDialog::browseOutputDir);
```

Add this method:

```cpp
void ThreeDReconstructionDialog::setMode(Mode mode)
{
    m_mode = mode;
    if (mode == Mode::AerialTriangulation)
    {
        setWindowTitle(QStringLiteral("空中三角测量"));
        if (m_titleLabel)
        {
            m_titleLabel->setText(QStringLiteral("<b>一键生成正式 SfM/BA 稀疏云</b>"));
        }
        if (m_exportObjCheck)
        {
            m_exportObjCheck->setChecked(false);
            m_exportObjCheck->setVisible(false);
        }
        if (m_startBtn)
        {
            m_startBtn->setText(QStringLiteral("开始空三"));
        }
        return;
    }

    setWindowTitle(QStringLiteral("三维重建"));
    if (m_titleLabel)
    {
        m_titleLabel->setText(QStringLiteral("<b>一键生成三维模型</b>"));
    }
    if (m_exportObjCheck)
    {
        m_exportObjCheck->setVisible(true);
    }
    if (m_startBtn)
    {
        m_startBtn->setText(QStringLiteral("开始"));
    }
}
```

In `collectSettings()`, add:

```cpp
settings[QStringLiteral("workflow_kind")] =
    m_mode == Mode::AerialTriangulation
        ? QStringLiteral("aerial_triangulation")
        : QStringLiteral("three_d_reconstruction");
if (m_mode == Mode::AerialTriangulation)
{
    settings[QStringLiteral("export_obj")] = false;
}
```

- [ ] **Step 6: Run the focused test and verify it passes**

Run the command from Step 2.

Expected: the dialog mode test passes.

- [ ] **Step 7: Commit**

```powershell
git add src/gui/dialogs/ThreeDReconstructionDialog.h src/gui/dialogs/ThreeDReconstructionDialog.cpp src/gui/dialogs/ThreeDReconstructionDialog.ui tests/test_gui_project_utils.cpp
git commit -m "feat: add sparse-only aerial triangulation dialog mode"
```

---

### Task 3: Wire Sparse-Only Aerial Triangulation Workflow

**Files:**
- Modify: `src/gui/main_window/MenuWorkflowController.h`
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/gui/main_window/MainWindow.cpp`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add failing source tests for wiring and stopping point**

Add these tests near the existing workflow controller tests:

```cpp
TEST(AerialTriangulationWorkflowTest, MainWindowWiresSparseOnlyAerialTriangulationAction)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(source.contains(QStringLiteral("aerialTriangulationAction()")));
    EXPECT_TRUE(source.contains(QStringLiteral("openAerialTriangulationDialog")));
}

TEST(AerialTriangulationWorkflowTest, SparseOnlyWorkflowStopsBeforeDenseStages)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("openAerialTriangulationDialog")));
    EXPECT_TRUE(header.contains(QStringLiteral("startAerialTriangulationWorkflow")));
    EXPECT_TRUE(source.contains(QStringLiteral("DialogSettingKeys::AerialTriangulation")));
    EXPECT_TRUE(source.contains(QStringLiteral("setMode(ThreeDReconstructionDialog::Mode::AerialTriangulation)")));
    EXPECT_TRUE(source.contains(QStringLiteral("source\", QStringLiteral(\"aerial_triangulation\")")));

    const int sparseStart = source.indexOf(QStringLiteral("void MenuWorkflowController::startAerialTriangulationWorkflow"));
    ASSERT_GE(sparseStart, 0);
    const int nextFunction = source.indexOf(QStringLiteral("void MenuWorkflowController::startThreeDReconstructionWorkflow"), sparseStart);
    ASSERT_GT(nextFunction, sparseStart);
    const QString sparseBlock = source.mid(sparseStart, nextFunction - sparseStart);
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startThreeDReconstructionDenseStage")));
    EXPECT_FALSE(sparseBlock.contains(QStringLiteral("startThreeDReconstructionMeshStage")));
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set PATH=%CD%\build\windows-existingdeps-release\bin;E:\code\plascan\build\windows-vcpkg-release\vcpkg_installed\x64-windows\bin;E:\code\plascan\build\env\libtorch-cu130\libtorch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH% && set QT_QPA_PLATFORM=offscreen && .\build\windows-existingdeps-release\tests\test_gui_project_utils.exe --gtest_filter=AerialTriangulationWorkflowTest.*"
```

Expected: tests fail because the action and workflow methods do not exist.

- [ ] **Step 3: Add controller declarations**

In `MenuWorkflowController.h`, add public slot:

```cpp
void openAerialTriangulationDialog();
```

Add private method:

```cpp
void startAerialTriangulationWorkflow(const QJsonObject &settings);
```

- [ ] **Step 4: Wire MainWindow action**

In `MainWindow::setupMenuConnections()`, inside the `m_menuWorkflowController` block, add:

```cpp
if (m_mainMenu->aerialTriangulationAction())
{
    connect(m_mainMenu->aerialTriangulationAction(), &QAction::triggered,
            m_menuWorkflowController, &MenuWorkflowController::openAerialTriangulationDialog);
}
```

- [ ] **Step 5: Implement the sparse-only dialog opener**

In `MenuWorkflowController.cpp`, add:

```cpp
void MenuWorkflowController::openAerialTriangulationDialog()
{
    if (!m_mainWindow)
    {
        return;
    }

    auto *dlg = new ThreeDReconstructionDialog(m_mainWindow);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setMode(ThreeDReconstructionDialog::Mode::AerialTriangulation);

    const QStringList images = getProjectImages();
    dlg->setImageCount(images.size());

    DialogSettingStore aerialStore(DialogSettingKeys::AerialTriangulation, this);
    if (m_projectManager)
    {
        const QString projectPath = m_projectManager->currentProjectPath();
        const QString assetsDir = ProjectIO::projectAssetsDir(projectPath);
        if (!assetsDir.isEmpty())
        {
            dlg->setDefaultOutputDir(QDir(assetsDir).filePath(QStringLiteral("aerial_triangulation")));
        }
        aerialStore.setProjectPath(projectPath);
        dlg->applySettings(aerialStore.load());
    }

    connect(dlg, &ThreeDReconstructionDialog::settingsChanged, this,
            [store = &aerialStore](const QJsonObject &settings)
    {
        store->save(settings);
    });
    connect(dlg, &ThreeDReconstructionDialog::runRequested, this,
            [this, store = &aerialStore](const QJsonObject &settings)
    {
        store->save(settings);
        startAerialTriangulationWorkflow(settings);
    });

    dlg->exec();
}
```

If the stack-allocated `DialogSettingStore` lifetime is unsafe for signal lambdas after `exec()`, replace it with a controller member `m_aerialTriangulationSetting` like `m_threeDSetting`.

- [ ] **Step 6: Implement the sparse-only workflow by reusing SFMService**

Add `startAerialTriangulationWorkflow()` by copying the SFM stage from `startThreeDReconstructionWorkflow()` and changing only the output path, progress text, source field, success dialog, and removing dense-stage dispatch:

```cpp
void MenuWorkflowController::startAerialTriangulationWorkflow(const QJsonObject &settings)
{
    if (!m_projectManager)
    {
        QMessageBox::warning(m_mainWindow, QStringLiteral("空中三角测量"), QStringLiteral("请先打开项目"));
        return;
    }

    const QStringList images = getProjectImages();
    if (images.size() < 2)
    {
        QMessageBox::warning(m_mainWindow,
                             QStringLiteral("空中三角测量"),
                             QStringLiteral("至少需要 2 张影像才能进行空中三角测量。"));
        return;
    }

    QString outputRoot = settings.value(QStringLiteral("output_dir")).toString().trimmed();
    if (outputRoot.isEmpty())
    {
        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        outputRoot = QDir(assetsDir).filePath(QStringLiteral("aerial_triangulation"));
    }
    outputRoot = QDir::cleanPath(outputRoot);
    QDir().mkpath(outputRoot);

    auto *pm = m_projectManager;
    xjw::gui::SFMServiceOptions opts;
    opts.images = images;
    opts.plascanPath = pm->currentProjectPath();
    opts.projectMeta = pm->coreProjectMeta();
    opts.outputDir = QDir(outputRoot).filePath(QStringLiteral("sfm_sparse"));
    const int workflowThreads = std::max(1, settings.value(QStringLiteral("threads")).toInt(8));
    opts.threads = workflowThreads;
    opts.device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    opts.featureGrayscaleMin = normalizedFeatureGrayscaleMin(settings);
    opts.featureGrayscaleMax = 1.0f;
    opts.cudaParallelPairs = opts.device == QStringLiteral("cpu")
        ? 1
        : std::clamp(std::max(1, workflowThreads / 4), 1, 2);

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    opts.quality = quality == QStringLiteral("fast") ? 1 : 3;
    opts.progressFn = [pm](const QString &stage, int percent)
    {
        QMetaObject::invokeMethod(pm, [pm, stage, percent]()
        {
            emit pm->atProgressChanged(QStringLiteral("空中三角测量: %1").arg(stage), percent);
        }, Qt::QueuedConnection);
    };
    opts.pairMatchedFn = [pm](const QString &img0,
                              const QString &img1,
                              const QString &matchPath,
                              int numMatches)
    {
        QMetaObject::invokeMethod(pm, [pm, img0, img1, matchPath, numMatches]()
        {
            emit pm->matchPairReady(img0, img1, matchPath, numMatches);
        }, Qt::QueuedConnection);
    };

    auto cancelFlag = std::make_shared<std::atomic<bool>>(false);
    pm->setAtCancelFlag(cancelFlag);
    opts.cancelFlag = cancelFlag;
    emit pm->atProgressChanged(QStringLiteral("空中三角测量: 启动..."), 0);

    const QStringList sfmImages = images;
    const QString sfmOutputDir = opts.outputDir;
    (void)QtConcurrent::run([pm, opts, sfmImages, sfmOutputDir]() mutable
    {
        xjw::gui::SFMServiceResult result = xjw::gui::SFMService::run(opts);
        QMetaObject::invokeMethod(pm, [pm, result = std::move(result), sfmImages, sfmOutputDir]() mutable
        {
            for (const auto &sp : result.newSpFiles)
            {
                pm->appendIpfindResult(sp.imagePath, sp.spPath, QJsonObject());
            }
            for (const auto &match : result.newMatchFiles)
            {
                pm->appendIpmatchResult(QStringList{match.matchPath}, match.settings);
            }
            if (!result.pendingCamUpdates.isEmpty())
            {
                int updated = 0;
                QString err;
                if (!pm->setImageCameras(result.pendingCamUpdates, &updated, &err))
                {
                    LOG_WARN(QStringLiteral("空中三角测量: SFM 相机写回失败: %1").arg(err));
                }
            }

            if (result.success && !result.sparseCloudPath.isEmpty())
            {
                QStringList registeredImages;
                const QStringList registeredCameraKeys = result.pendingCamUpdates.keys();
                for (const QString &imagePath : sfmImages)
                {
                    const QString normalized = xjw::gui::project::normalizePath(imagePath);
                    if (registeredCameraKeys.contains(normalized))
                    {
                        registeredImages.append(normalized);
                    }
                }
                QJsonObject resultRecordExtra = result.resultRecordExtra;
                resultRecordExtra[QStringLiteral("source")] = QStringLiteral("aerial_triangulation");
                pm->appendAtResult(result.sparseCloudPath,
                                   result.numPoints3D,
                                   registeredImages,
                                   sfmOutputDir,
                                   resultRecordExtra);
            }

            emit pm->atProgressFinished(result.success);
            if (!result.success)
            {
                QMessageBox::warning(nullptr,
                                     QStringLiteral("空中三角测量"),
                                     result.errorMessage.isEmpty()
                                         ? QStringLiteral("空中三角测量失败。")
                                         : result.errorMessage);
                return;
            }

            if (!xjw::gui::project::isProductionSparseResult(result.resultRecordExtra))
            {
                const QString reason = xjw::gui::project::sparseResultBlockingReason(result.resultRecordExtra);
                QMessageBox::warning(nullptr,
                                     QStringLiteral("空中三角测量"),
                                     reason.isEmpty()
                                         ? QStringLiteral("当前空三稀疏点云质量不足，不能作为生产输入。")
                                         : reason);
                return;
            }

            QMessageBox::information(nullptr,
                                     QStringLiteral("空中三角测量"),
                                     QStringLiteral("正式 SfM/BA 稀疏云已生成。\n注册影像: %1\n稀疏点: %2\n输出: %3")
                                         .arg(result.numRegisteredImages)
                                         .arg(result.numPoints3D)
                                         .arg(result.sparseCloudPath));
        }, Qt::QueuedConnection);
    });
}
```

- [ ] **Step 7: Run focused tests and build**

Run the focused workflow tests from Step 2, then:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build .\build\windows-existingdeps-release --config Release --target plascan_gui"
```

Expected: focused tests pass and `plascan_gui` builds.

- [ ] **Step 8: Commit**

```powershell
git add src/gui/main_window/MenuWorkflowController.h src/gui/main_window/MenuWorkflowController.cpp src/gui/main_window/MainWindow.cpp tests/test_gui_project_utils.cpp
git commit -m "feat: add sparse-only aerial triangulation workflow"
```

---

### Task 4: Add Upstream Missing-Data Choice

**Files:**
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.h`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add failing source test for the missing-data prompt**

Add:

```cpp
TEST(AerialTriangulationWorkflowTest, MissingUpstreamDataOffersAutoFillOrManualReturn)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.cpp"));
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MenuWorkflowController.h"));
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(header.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("SparsePrerequisiteSummary")));
    EXPECT_TRUE(source.contains(QStringLiteral("自动补齐缺失步骤")));
    EXPECT_TRUE(source.contains(QStringLiteral("返回手动处理")));
    EXPECT_TRUE(source.contains(QStringLiteral("autoGenerateMissingMatches = autoFillMissing")));
    EXPECT_TRUE(source.contains(QStringLiteral("缺少连接点")));
}
```

- [ ] **Step 2: Run focused test and verify it fails**

Run:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set PATH=%CD%\build\windows-existingdeps-release\bin;E:\code\plascan\build\windows-vcpkg-release\vcpkg_installed\x64-windows\bin;E:\code\plascan\build\env\libtorch-cu130\libtorch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH% && set QT_QPA_PLATFORM=offscreen && .\build\windows-existingdeps-release\tests\test_gui_project_utils.exe --gtest_filter=AerialTriangulationWorkflowTest.MissingUpstreamDataOffersAutoFillOrManualReturn"
```

Expected: fails because the helper and prompt are not present.

- [ ] **Step 3: Add prerequisite summary type**

In `MenuWorkflowController.h`, add private struct:

```cpp
struct SparsePrerequisiteSummary
{
    int imageCount = 0;
    bool hasFeatures = false;
    bool hasMatches = false;
    QStringList missingMessages;
};

SparsePrerequisiteSummary summarizeSparsePrerequisites(const QStringList &images,
                                                       const QJsonObject &meta,
                                                       const QString &projectPath) const;
bool confirmAutoFillMissingSparseInputs(const SparsePrerequisiteSummary &summary) const;
```

- [ ] **Step 4: Implement prerequisite summary**

In `MenuWorkflowController.cpp`, implement:

```cpp
MenuWorkflowController::SparsePrerequisiteSummary
MenuWorkflowController::summarizeSparsePrerequisites(const QStringList &images,
                                                     const QJsonObject &meta,
                                                     const QString &projectPath) const
{
    SparsePrerequisiteSummary summary;
    summary.imageCount = images.size();
    const QStringList suffixes = xjw::gui::project::projectFeatureSuffixes(projectPath, meta);
    summary.hasFeatures = !suffixes.isEmpty();
    const QJsonArray matches = meta.value(QStringLiteral("ipmatch_results")).toArray();
    summary.hasMatches = !matches.isEmpty();

    if (!summary.hasFeatures)
    {
        summary.missingMessages.append(QStringLiteral("缺少特征：当前项目没有可用于空三的特征文件。"));
    }
    if (!summary.hasMatches)
    {
        summary.missingMessages.append(QStringLiteral("缺少连接点：当前项目没有可用于空三的匹配结果。"));
    }
    return summary;
}
```

Add required include if missing:

```cpp
#include "project/ProjectSupportUtils.h"
```

- [ ] **Step 5: Implement auto-fill/manual prompt**

Add:

```cpp
bool MenuWorkflowController::confirmAutoFillMissingSparseInputs(const SparsePrerequisiteSummary &summary) const
{
    if (summary.missingMessages.isEmpty())
    {
        return true;
    }

    const QString message = QStringLiteral("空中三角测量缺少上游数据：\n\n%1\n\n是否自动补齐缺失步骤？")
        .arg(summary.missingMessages.join(QStringLiteral("\n")));
    QMessageBox box(m_mainWindow);
    box.setIcon(QMessageBox::Question);
    box.setWindowTitle(QStringLiteral("空中三角测量"));
    box.setText(message);
    QPushButton *autoFill = box.addButton(QStringLiteral("自动补齐缺失步骤"), QMessageBox::AcceptRole);
    box.addButton(QStringLiteral("返回手动处理"), QMessageBox::RejectRole);
    box.exec();
    return box.clickedButton() == autoFill;
}
```

- [ ] **Step 6: Use the prompt in sparse-only workflow**

At the start of `startAerialTriangulationWorkflow()`, after `images` validation, add:

```cpp
const SparsePrerequisiteSummary prereq = summarizeSparsePrerequisites(
    images,
    m_projectManager->currentMeta(),
    m_projectManager->currentProjectPath());
const bool autoFillMissing = confirmAutoFillMissingSparseInputs(prereq);
if (!autoFillMissing && !prereq.missingMessages.isEmpty())
{
    return;
}
```

Set service options:

```cpp
opts.autoGenerateMissingMatches = autoFillMissing;
```

If `SFMService` always auto-generates missing features, document that first implementation controls missing matches and uses the dialog choice to decide whether the run proceeds.

- [ ] **Step 7: Run focused tests**

Run the focused test from Step 2.

Expected: passes.

- [ ] **Step 8: Commit**

```powershell
git add src/gui/main_window/MenuWorkflowController.h src/gui/main_window/MenuWorkflowController.cpp tests/test_gui_project_utils.cpp
git commit -m "feat: prompt before auto-filling sparse inputs"
```

---

### Task 5: Enforce Production Inputs In Sparse Post-Processing UI

**Files:**
- Modify: `src/gui/dialogs/SparseCloudPostProcessDialog.cpp`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add failing source test**

Add:

```cpp
TEST(SparseCloudPostProcessDialogTest, ListsOnlyProductionSparseInputs)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/SparseCloudPostProcessDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());
    EXPECT_TRUE(source.contains(QStringLiteral("isProductionSparseResult(record)")));
    EXPECT_TRUE(source.contains(QStringLiteral("continue;")));
    EXPECT_FALSE(source.contains(QStringLiteral("isPairwisePreviewSparseResult(record) &&")));
}
```

- [ ] **Step 2: Run focused test and verify it fails if the production check is missing**

Run:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set PATH=%CD%\build\windows-existingdeps-release\bin;E:\code\plascan\build\windows-vcpkg-release\vcpkg_installed\x64-windows\bin;E:\code\plascan\build\env\libtorch-cu130\libtorch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH% && set QT_QPA_PLATFORM=offscreen && .\build\windows-existingdeps-release\tests\test_gui_project_utils.exe --gtest_filter=SparseCloudPostProcessDialogTest.ListsOnlyProductionSparseInputs"
```

Expected: fails if `setAvailableSparseClouds()` does not filter with `isProductionSparseResult`.

- [ ] **Step 3: Add production filter**

In `SparseCloudPostProcessDialog.cpp`, include:

```cpp
#include "project/SparseResultQuality.h"
```

In `setAvailableSparseClouds(const QJsonArray &results)`, before adding each record to the combo box, add:

```cpp
const QJsonObject record = value.toObject();
if (!xjw::gui::project::isProductionSparseResult(record))
{
    continue;
}
```

Use `record` for the rest of the loop instead of calling `value.toObject()` again.

- [ ] **Step 4: Run focused test**

Run the test from Step 2.

Expected: passes.

- [ ] **Step 5: Commit**

```powershell
git add src/gui/dialogs/SparseCloudPostProcessDialog.cpp tests/test_gui_project_utils.cpp
git commit -m "fix: restrict sparse post-processing to production clouds"
```

---

### Task 6: Full Verification And Push

**Files:**
- No new source files beyond prior tasks.

- [ ] **Step 1: Build GUI target**

Run:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && "C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build .\build\windows-existingdeps-release --config Release --target plascan_gui"
```

Expected: build succeeds and links `bin\plascan_gui.bin.exe`.

- [ ] **Step 2: Run GUI project utility tests**

Run:

```powershell
cmd /c ""C:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set PATH=%CD%\build\windows-existingdeps-release\bin;E:\code\plascan\build\windows-vcpkg-release\vcpkg_installed\x64-windows\bin;E:\code\plascan\build\env\libtorch-cu130\libtorch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH% && set QT_QPA_PLATFORM=offscreen && .\build\windows-existingdeps-release\tests\test_gui_project_utils.exe"
```

Expected: all tests in `test_gui_project_utils.exe` pass.

- [ ] **Step 3: Check status**

Run:

```powershell
git status --short
```

Expected: only intended source/test changes are tracked or staged; `.superpowers/`, `logs/`, and dirty submodule internals are not staged.

- [ ] **Step 4: Push implementation branch**

Run:

```powershell
git push origin codex/sparse-reconstruction-menu-flow
```

Expected: branch is pushed to GitHub.

---

## Self-Review

- Spec coverage:
  - Menu main flow and `高级工具`: Task 1.
  - Sparse-only `空中三角测量...`: Tasks 2 and 3.
  - Missing upstream choice with simplified parameters: Tasks 2, 3, and 4.
  - Production-only post-processing: Task 5.
  - Result record and downstream quality gates: reused from existing `SparseResultQuality` and verified in Task 3/6.
- Placeholder scan:
  - No TBD/TODO placeholders are present.
  - Each implementation step includes file names, code snippets, commands, and expected results.
- Type consistency:
  - `aerialTriangulationAction()` is added to `MainMenu`.
  - `openAerialTriangulationDialog()` and `startAerialTriangulationWorkflow()` are added to `MenuWorkflowController`.
  - `ThreeDReconstructionDialog::Mode::AerialTriangulation` is used consistently.
