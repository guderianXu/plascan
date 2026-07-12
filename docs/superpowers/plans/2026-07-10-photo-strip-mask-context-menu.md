# Photo Strip Mask Context Menu Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ctrl/Shift multi-selection and a photo-strip context menu that sends the selected photos to the existing asynchronous mask-generation workflow.

**Architecture:** `PhotoStripWidget` owns selection and context-menu behavior and emits selected image paths without depending on project services. `MainWindow` forwards the request to a new `ProjectManager` entry point that validates the selected paths and opens the existing `GenerateMaskDialog`; the existing no-argument Tools menu entry remains unchanged.

**Tech Stack:** C++17, Qt 6 Widgets/Test, CMake, GTest, existing PlaScan mask workflow.

---

## File Map

- Modify `src/gui/widgets/PhotoStripWidget.h`: declare the batch mask request signal and context-menu helpers.
- Modify `src/gui/widgets/PhotoStripWidget.cpp`: enable extended selection, preserve multi-selection, and build the context menu.
- Modify `src/gui/main_window/MainWindow.cpp`: forward photo-strip requests to `ProjectManager`.
- Modify `src/gui/project/manager/ProjectManager.h`: expose the selected-images mask-dialog entry point.
- Modify `src/gui/project/manager/ProjectManager.cpp`: validate selected photos and reuse the existing dialog/task implementation.
- Modify `src/gui/project/io/ProjectIO.h`: expose project-relative resource path resolution.
- Modify `src/gui/project/io/ProjectIO.cpp`: resolve metadata paths against the `.plascan` directory.
- Modify `tests/test_gui_project_utils.cpp`: add behavioral widget tests and wiring contract tests.

### Task 1: Photo Strip Selection Tests

**Files:**
- Modify: `tests/test_gui_project_utils.cpp:14339`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add the failing extended-selection preservation test**

Add this test after the existing `ClickSelectsPhotoAndActivationOpensPhoto` test:

```cpp
TEST(PhotoStripWidgetTest, ExtendedSelectionSurvivesCurrentPhotoSynchronization)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));

    QJsonArray images;
    images.append(QJsonObject{{QStringLiteral("path"), firstPath}});
    images.append(QJsonObject{{QStringLiteral("path"), secondPath}});

    PhotoStripWidget strip;
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), images}});
    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    ASSERT_EQ(list->count(), 2);
    EXPECT_EQ(list->selectionMode(), QAbstractItemView::ExtendedSelection);

    list->item(0)->setSelected(true);
    list->item(1)->setSelected(true);
    list->setCurrentItem(list->item(1), QItemSelectionModel::NoUpdate);
    strip.setCurrentPhoto(firstPath);

    EXPECT_TRUE(list->item(0)->isSelected());
    EXPECT_TRUE(list->item(1)->isSelected());
    EXPECT_EQ(list->currentItem(), list->item(0));
}
```

- [ ] **Step 2: Run the test and verify RED**

Run from an MSVC developer environment:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 16
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=PhotoStripWidgetTest.ExtendedSelectionSurvivesCurrentPhotoSynchronization
```

Expected: FAIL because the list is still `SingleSelection` and `setCurrentPhoto()` uses `ClearAndSelect` unconditionally.

- [ ] **Step 3: Add the failing context-menu batch tests**

```cpp
TEST(PhotoStripWidgetTest, ContextMenuRequestsMasksForSelectedPhotos)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));

    PhotoStripWidget strip;
    strip.resize(600, 240);
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), firstPath}},
        QJsonObject{{QStringLiteral("path"), secondPath}}
    }}});
    strip.show();
    QCoreApplication::processEvents();

    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    list->item(0)->setSelected(true);
    list->item(1)->setSelected(true);
    QSignalSpy requestSpy(&strip, &PhotoStripWidget::generateMaskRequested);

    const QPoint itemPosition = list->visualItemRect(list->item(0)).center();
    ASSERT_TRUE(QMetaObject::invokeMethod(&strip,
                                          "showPhotoContextMenu",
                                          Qt::DirectConnection,
                                          Q_ARG(QPoint, itemPosition)));
    QCoreApplication::processEvents();
    auto *menu = qobject_cast<QMenu *>(QApplication::activePopupWidget());
    ASSERT_NE(menu, nullptr);
    ASSERT_EQ(menu->actions().size(), 1);
    EXPECT_EQ(menu->actions().first()->text(), QStringLiteral("生成蒙版..."));
    menu->actions().first()->trigger();
    ASSERT_EQ(requestSpy.count(), 1);
    EXPECT_EQ(requestSpy.takeFirst().at(0).toStringList(), QStringList({firstPath, secondPath}));
}

TEST(PhotoStripWidgetTest, ContextMenuSelectsAnUnselectedClickedPhoto)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString firstPath = QDir(tempDir.path()).filePath(QStringLiteral("first.png"));
    const QString secondPath = QDir(tempDir.path()).filePath(QStringLiteral("second.png"));

    PhotoStripWidget strip;
    strip.resize(600, 240);
    strip.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{
        QJsonObject{{QStringLiteral("path"), firstPath}},
        QJsonObject{{QStringLiteral("path"), secondPath}}
    }}});
    strip.show();
    QCoreApplication::processEvents();

    auto *list = strip.findChild<QListWidget *>(QStringLiteral("photoStripList"));
    ASSERT_NE(list, nullptr);
    list->item(0)->setSelected(true);
    QSignalSpy requestSpy(&strip, &PhotoStripWidget::generateMaskRequested);

    const QPoint itemPosition = list->visualItemRect(list->item(1)).center();
    ASSERT_TRUE(QMetaObject::invokeMethod(&strip,
                                          "showPhotoContextMenu",
                                          Qt::DirectConnection,
                                          Q_ARG(QPoint, itemPosition)));
    QCoreApplication::processEvents();
    auto *menu = qobject_cast<QMenu *>(QApplication::activePopupWidget());
    ASSERT_NE(menu, nullptr);
    menu->actions().first()->trigger();

    EXPECT_FALSE(list->item(0)->isSelected());
    EXPECT_TRUE(list->item(1)->isSelected());
    ASSERT_EQ(requestSpy.count(), 1);
    EXPECT_EQ(requestSpy.takeFirst().at(0).toStringList(), QStringList({secondPath}));
}
```

- [ ] **Step 4: Run the context-menu test and verify RED**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 16
```

Expected: compilation FAIL because `PhotoStripWidget::generateMaskRequested` and `showPhotoContextMenu` do not exist.

### Task 2: Photo Strip Context Menu Implementation

**Files:**
- Modify: `src/gui/widgets/PhotoStripWidget.h:27`
- Modify: `src/gui/widgets/PhotoStripWidget.cpp:7-135`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Declare the signal and helpers**

Add to `PhotoStripWidget.h`:

```cpp
signals:
    void photoSelected(const QString &imagePath);
    void photoActivated(const QString &imagePath);
    void generateMaskRequested(const QStringList &imagePaths);

private slots:
    void showPhotoContextMenu(const QPoint &position);

private:
    QStringList selectedPhotoPaths() const;
```

- [ ] **Step 2: Implement extended selection and context-menu behavior**

Add `QAction`, `QMenu`, and `QStringList` includes. In the constructor use:

```cpp
_list->setSelectionMode(QAbstractItemView::ExtendedSelection);
_list->setContextMenuPolicy(Qt::CustomContextMenu);
connect(_list, &QListWidget::customContextMenuRequested,
        this, &PhotoStripWidget::showPhotoContextMenu);
```

Update `setCurrentPhoto()` so an already-selected target keeps the rest of the selection:

```cpp
const QItemSelectionModel::SelectionFlags command = item->isSelected()
    ? QItemSelectionModel::NoUpdate
    : QItemSelectionModel::ClearAndSelect;
_list->setCurrentItem(item, command);
```

Implement the helpers:

```cpp
QStringList PhotoStripWidget::selectedPhotoPaths() const
{
    QStringList paths;
    QSet<QString> seen;
    if (!_list)
    {
        return paths;
    }

    for (int row = 0; row < _list->count(); ++row)
    {
        QListWidgetItem *item = _list->item(row);
        if (!item || !item->isSelected())
        {
            continue;
        }
        const QString path = item->data(PathRole).toString().trimmed();
        const QString key = normalizedPath(path);
        if (!path.isEmpty() && !seen.contains(key))
        {
            seen.insert(key);
            paths.push_back(path);
        }
    }
    return paths;
}

void PhotoStripWidget::showPhotoContextMenu(const QPoint &position)
{
    if (!_list)
    {
        return;
    }
    QListWidgetItem *item = _list->itemAt(position);
    if (!item)
    {
        return;
    }

    const QItemSelectionModel::SelectionFlags command = item->isSelected()
        ? QItemSelectionModel::NoUpdate
        : QItemSelectionModel::ClearAndSelect;
    _list->setCurrentItem(item, command);
    emit photoSelected(item->data(PathRole).toString());

    const QStringList imagePaths = selectedPhotoPaths();
    if (imagePaths.isEmpty())
    {
        return;
    }

    auto *menu = new QMenu(this);
    menu->setAttribute(Qt::WA_DeleteOnClose);
    QAction *generateAction = menu->addAction(tr("生成蒙版..."));
    connect(generateAction, &QAction::triggered, this, [this, imagePaths]()
    {
        emit generateMaskRequested(imagePaths);
    });
    menu->popup(_list->viewport()->mapToGlobal(position));
}
```

- [ ] **Step 3: Run widget tests and verify GREEN**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 16
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=PhotoStripWidgetTest.*
```

Expected: all `PhotoStripWidgetTest.*` tests PASS.

### Task 3: Forward Selected Photos Into Existing Mask Workflow

**Files:**
- Modify: `tests/test_gui_project_utils.cpp:7781`
- Modify: `src/gui/main_window/MainWindow.cpp:666`
- Modify: `src/gui/project/manager/ProjectManager.h:171`
- Modify: `src/gui/project/manager/ProjectManager.cpp:1090`

- [ ] **Step 1: Add the failing wiring contract test**

```cpp
TEST(MainWindowTest, PhotoStripMaskRequestUsesSelectedImages)
{
    const QString mainSource = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    const QString managerHeader = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.h"));
    const QString managerSource = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectManager.cpp"));

    EXPECT_TRUE(mainSource.contains(QStringLiteral("&PhotoStripWidget::generateMaskRequested")));
    EXPECT_TRUE(mainSource.contains(QStringLiteral("openGenerateMaskDialogForImages(imagePaths)")));
    EXPECT_TRUE(managerHeader.contains(
        QStringLiteral("void openGenerateMaskDialogForImages(const QStringList &selectedImages);")));
    EXPECT_TRUE(managerSource.contains(
        QStringLiteral("GenerateMaskDialog dialog(selectedImages, currentImage, _parent)")));
}
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=MainWindowTest.PhotoStripMaskRequestUsesSelectedImages
```

Expected: FAIL because no batch request connection or manager entry point exists.

- [ ] **Step 3: Add the ProjectManager selected-image entry point**

Declare this public slot in `ProjectManager.h`:

```cpp
void openGenerateMaskDialogForImages(const QStringList &selectedImages);
```

Keep the existing no-argument method as the Tools-menu wrapper:

```cpp
void ProjectManager::openGenerateMaskDialog()
{
    const QStringList allImages = _projectData ? _projectData->getAllImages() : QStringList();
    openGenerateMaskDialogForImages(allImages);
}
```

Rename the current implementation to this signature:

```cpp
void ProjectManager::openGenerateMaskDialogForImages(const QStringList &requestedImages)
{
```

Keep the existing project-open and running-task guards. Immediately after the existing empty-project-photo check, insert path validation that preserves project paths and request order:

```cpp
    QHash<QString, QString> projectImages;
    for (const QString &imagePath : allImages)
    {
        projectImages.insert(normalizePath(imagePath), imagePath);
    }

    QStringList selectedImages;
    QSet<QString> seen;
    for (const QString &requestedPath : requestedImages)
    {
        const QString key = normalizePath(requestedPath);
        if (!key.isEmpty() && projectImages.contains(key) && !seen.contains(key))
        {
            seen.insert(key);
            selectedImages.push_back(projectImages.value(key));
        }
    }
    if (selectedImages.isEmpty())
    {
        showWarning(QStringLiteral("没有选中可生成蒙版的照片。"), QStringLiteral("生成蒙版"));
        return;
    }
```

Replace the existing current-image and dialog construction lines with:

```cpp
    const QString currentImage = projectImages.value(normalizePath(_activeImagePath));
    GenerateMaskDialog dialog(selectedImages, currentImage, _parent);
```

No code after dialog construction changes: the existing settings collection, background worker, progress updates, metadata persistence, and `masksGenerated` signal continue to execute once.

- [ ] **Step 4: Connect the photo strip in MainWindow**

Add alongside the existing photo signals:

```cpp
connect(_photoStrip,
        &PhotoStripWidget::generateMaskRequested,
        this,
        [this](const QStringList &imagePaths)
        {
            if (_projectManager)
            {
                _projectManager->openGenerateMaskDialogForImages(imagePaths);
            }
        });
```

- [ ] **Step 5: Build and verify GREEN**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils plascan_gui --parallel 16
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=MainWindowTest.PhotoStripMaskRequestUsesSelectedImages:PhotoStripWidgetTest.*:GenerateMaskDialogTest.*:GenerateMaskWorkflowTest.*
```

Expected: all selected tests PASS and the GUI target links successfully.

### Task 4: Regression Verification

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run the complete GUI project utility test binary**

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe
```

Expected: all tests PASS. If an unrelated pre-existing failure occurs, record its exact test name and output without modifying unrelated code.

- [ ] **Step 2: Check formatting and scoped diff**

```powershell
git diff --check -- src/gui/widgets/PhotoStripWidget.h src/gui/widgets/PhotoStripWidget.cpp src/gui/main_window/MainWindow.cpp src/gui/project/manager/ProjectManager.h src/gui/project/manager/ProjectManager.cpp tests/test_gui_project_utils.cpp
git status --short
```

Expected: no whitespace errors; existing unrelated model-workflow changes remain present and untouched.

- [ ] **Step 3: Do not commit without explicit authorization**

Report the changed files and verification output. The repository instructions require explicit user authorization before staging, committing, or pushing.
