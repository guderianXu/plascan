# Metashape Style GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Metashape-style PlaScan GUI with persistent photo and property panels, selection-driven metadata display, photo-to-camera highlighting, and clearer 3D camera/point-cloud/origin visualization.

**Architecture:** Keep the existing `MainWindow.ui` mostly stable and compose the new panels in C++ so VSCode/Qt Designer remains usable. Centralize selection in `MainWindow`: project tree and photo strip selections update the property widget, central workspace, and `CameraSceneWidget` camera highlight. Keep heavy image and point-cloud inspection out of the GUI thread by using light metadata reads and asynchronous thumbnail loading.

**Tech Stack:** C++17, Qt6 Widgets, Qt6 Concurrent, Qt RHI/Vulkan 3D view, CMake AUTOUIC/AUTOMOC, GTest source-structure tests.

---

## File Structure

- Create `src/gui/widgets/SelectionPropertiesWidget.h`
  - Declares a read-only property table for photos, point clouds, models, DEM, DOM, and generic project resources.
  - Exposes slots that accept the current project metadata and selected resource path.

- Create `src/gui/widgets/SelectionPropertiesWidget.cpp`
  - Implements metadata lookup, light file-system/image inspection, and table rendering.
  - Uses `QImageReader` for image size/format and `QFileInfo` for size/time/path.
  - Does not scan large point-cloud/model files in the first pass; it clearly displays the reason when detailed attributes are not loaded.

- Create `src/gui/widgets/PhotoStripWidget.h`
  - Declares the bottom photo panel.
  - Exposes `loadFromJson()`, `setCurrentPhoto()`, and `photoActivated()`.

- Create `src/gui/widgets/PhotoStripWidget.cpp`
  - Builds an icon-mode `QListWidget`.
  - Loads thumbnails asynchronously with `QtConcurrent::run`.
  - Keeps thumbnail cache in memory only.

- Modify `src/gui/cmake/GuiSources.cmake`
  - Add both new widgets to `GUI_SOURCES` and `GUI_HEADERS`.

- Modify `src/gui/menu/MainMenu.h`
  - Add getters and private members for Workspace, Properties, Photos, and World Origin actions.

- Modify `src/gui/menu/MainMenu.cpp`
  - Create or bind the new actions in both UI-backed and fallback menu paths.
  - Add them to `WindowPanel`.

- Modify `src/gui/main_window/MainWindow.h`
  - Add members for left property panel, bottom photos dock, and new selection helpers.
  - Use `_` prefixed private member names.

- Modify `src/gui/main_window/MainWindow.cpp`
  - Compose the left vertical splitter and bottom Photos dock in C++.
  - Wire project metadata, tree selection, photo-strip selection, menu toggles, and state persistence.

- Modify `src/gui/widgets/WorkspaceCenterWidget.h`
  - Add `highlightCameraForImage()` and `clearHighlightedCamera()` forwarding methods.

- Modify `src/gui/widgets/WorkspaceCenterWidget.cpp`
  - Store `CameraPose::imagePath` from project metadata and forward photo selection to the 3D widget.

- Modify `src/gui/dialogs/CameraModel3DDialog.h`
  - Extend `CameraPose` with `imagePath`.
  - Add highlighted camera and world-origin visibility APIs.

- Modify `src/gui/dialogs/CameraModel3DDialog.cpp`
  - Draw unselected cameras in blue, selected cameras in red/pink, and gate the origin cross behind a menu-controlled flag.
  - Slightly reduce visual weight of the manipulation sphere and point cloud points.

- Modify `tests/test_gui_project_utils.cpp`
  - Add source-structure tests for the new widgets, menu actions, main-window wiring, and 3D highlight/origin APIs.
  - Add one direct `MainMenu` test for the new checkable actions.

## Safety Constraints

- Do not modify `3rdparty/plapoint` or `3rdparty/plamatrix`.
- Do not stage or revert existing dirty files unrelated to this GUI task:
  - `scripts/run_lightglue.py`
  - `src/core/mvs/*`
  - `src/core/pipeline/*`
  - `src/cli/*`
  - `tests/test_match_result_catalog.cpp`
  - LightGlue model files under `resources/models/`
- Avoid editing `src/gui/main_window/MainWindow.ui` unless a C++ menu fallback cannot express the required behavior.
- Preserve Linux paths and Linux build behavior; do not replace cross-platform path logic with Windows-only paths.
- Commit only when the user explicitly asks for a commit. Before any commit, run `git status --short` and stage only GUI task files.

## Tasks

### Task 1: Add Failing Tests For Metashape-Style GUI Contracts

**Files:**
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add source-structure tests near the existing `MainMenuTest` / `CameraSceneWidgetTest` block**

Add this block after `TEST(CameraSceneWidgetTest, CameraVisibilityToggleIsExposedAndGuardsCameraOverlayOnly)`:

```cpp
TEST(MetashapeStyleGuiTest, DeclaresDedicatedPhotoAndPropertyWidgets)
{
    const QString cmake = readProjectSourceFile(QStringLiteral("src/gui/cmake/GuiSources.cmake"));
    const QString propertiesHeader = readProjectSourceFile(
        QStringLiteral("src/gui/widgets/SelectionPropertiesWidget.h"));
    const QString photoHeader = readProjectSourceFile(
        QStringLiteral("src/gui/widgets/PhotoStripWidget.h"));
    ASSERT_FALSE(cmake.isEmpty());

    EXPECT_TRUE(cmake.contains(QStringLiteral("widgets/SelectionPropertiesWidget.cpp")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("widgets/SelectionPropertiesWidget.h")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("widgets/PhotoStripWidget.cpp")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("widgets/PhotoStripWidget.h")));

    EXPECT_TRUE(propertiesHeader.contains(QStringLiteral("class SelectionPropertiesWidget")));
    EXPECT_TRUE(propertiesHeader.contains(
        QStringLiteral("void showPhotoProperties(const QJsonObject &meta, const QString &imagePath);")));
    EXPECT_TRUE(propertiesHeader.contains(
        QStringLiteral("void showResourceProperties(const QJsonObject &meta, const QString &section, const QString &resourcePath);")));
    EXPECT_TRUE(propertiesHeader.contains(QStringLiteral("QTableWidget *_table")));
    EXPECT_FALSE(propertiesHeader.contains(QStringLiteral("m_table")));

    EXPECT_TRUE(photoHeader.contains(QStringLiteral("class PhotoStripWidget")));
    EXPECT_TRUE(photoHeader.contains(QStringLiteral("void loadFromJson(const QJsonObject &meta);")));
    EXPECT_TRUE(photoHeader.contains(QStringLiteral("void setCurrentPhoto(const QString &imagePath);")));
    EXPECT_TRUE(photoHeader.contains(QStringLiteral("void photoActivated(const QString &imagePath);")));
    EXPECT_TRUE(photoHeader.contains(QStringLiteral("QListWidget *_list")));
    EXPECT_FALSE(photoHeader.contains(QStringLiteral("m_list")));
}

TEST(MetashapeStyleGuiTest, MainMenuExposesMetashapeWindowAndOriginActions)
{
    QMainWindow window;
    MainMenu menu(&window);

    ASSERT_NE(menu.toggleWorkspaceAction(), nullptr);
    ASSERT_NE(menu.togglePropertiesAction(), nullptr);
    ASSERT_NE(menu.togglePhotosAction(), nullptr);
    ASSERT_NE(menu.toggleWorldOriginAction(), nullptr);

    EXPECT_EQ(menu.toggleWorkspaceAction()->text(), QStringLiteral("工作区"));
    EXPECT_EQ(menu.togglePropertiesAction()->text(), QStringLiteral("属性"));
    EXPECT_EQ(menu.togglePhotosAction()->text(), QStringLiteral("照片"));
    EXPECT_EQ(menu.toggleWorldOriginAction()->text(), QStringLiteral("显示世界原点"));

    EXPECT_TRUE(menu.toggleWorkspaceAction()->isCheckable());
    EXPECT_TRUE(menu.togglePropertiesAction()->isCheckable());
    EXPECT_TRUE(menu.togglePhotosAction()->isCheckable());
    EXPECT_TRUE(menu.toggleWorldOriginAction()->isCheckable());

    EXPECT_TRUE(menu.toggleWorkspaceAction()->isChecked());
    EXPECT_TRUE(menu.togglePropertiesAction()->isChecked());
    EXPECT_TRUE(menu.togglePhotosAction()->isChecked());
    EXPECT_TRUE(menu.toggleWorldOriginAction()->isChecked());

    QMenu *viewMenu = findTopLevelMenuByTitle(window.menuBar(), QStringLiteral("视图"));
    ASSERT_NE(viewMenu, nullptr);
    EXPECT_TRUE(viewMenu->actions().contains(menu.toggleWorldOriginAction()));
}

TEST(MetashapeStyleGuiTest, MainWindowWiresPhotoPropertiesAndCameraHighlight)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/main_window/MainWindow.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("SelectionPropertiesWidget *_selectionProperties")));
    EXPECT_TRUE(header.contains(QStringLiteral("PhotoStripWidget *_photoStrip")));
    EXPECT_TRUE(header.contains(QStringLiteral("QDockWidget *_photosDock")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setupSelectionPanels()")));
    EXPECT_TRUE(header.contains(QStringLiteral("void selectPhoto(const QString &imagePath, bool openImage)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void selectResource(const QString &section, const QString &resourcePath)")));

    EXPECT_TRUE(source.contains(QStringLiteral("new SelectionPropertiesWidget")));
    EXPECT_TRUE(source.contains(QStringLiteral("new PhotoStripWidget")));
    EXPECT_TRUE(source.contains(QStringLiteral("photoActivated")));
    EXPECT_TRUE(source.contains(QStringLiteral("showPhotoProperties")));
    EXPECT_TRUE(source.contains(QStringLiteral("showResourceProperties")));
    EXPECT_TRUE(source.contains(QStringLiteral("highlightCameraForImage")));
    EXPECT_TRUE(source.contains(QStringLiteral("photos_visible")));
    EXPECT_TRUE(source.contains(QStringLiteral("properties_visible")));
    EXPECT_TRUE(source.contains(QStringLiteral("world_origin_visible")));
}

TEST(MetashapeStyleGuiTest, CameraSceneSupportsHighlightedCameraAndOriginToggle)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const QString workspaceHeader = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.h"));
    const QString workspaceSource = readProjectSourceFile(QStringLiteral("src/gui/widgets/WorkspaceCenterWidget.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(workspaceHeader.isEmpty());
    ASSERT_FALSE(workspaceSource.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("QString imagePath")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setHighlightedCameraPath(const QString &imagePath)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setHighlightedCameraName(const QString &imageName)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void clearHighlightedCamera()")));
    EXPECT_TRUE(header.contains(QStringLiteral("void setShowWorldOrigin(bool show)")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool isWorldOriginVisible() const")));
    EXPECT_TRUE(header.contains(QStringLiteral("QString _highlightedCameraPath")));
    EXPECT_TRUE(header.contains(QStringLiteral("QString _highlightedCameraName")));
    EXPECT_TRUE(header.contains(QStringLiteral("bool _showWorldOrigin = true")));

    EXPECT_TRUE(source.contains(QStringLiteral("CameraSceneWidget::setHighlightedCameraPath")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraSceneWidget::setShowWorldOrigin")));
    EXPECT_TRUE(source.contains(QStringLiteral("QColor(42, 122, 200")));
    EXPECT_TRUE(source.contains(QStringLiteral("QColor(245, 90, 105")));
    EXPECT_TRUE(source.contains(QStringLiteral("if (_showWorldOrigin)")));

    EXPECT_TRUE(workspaceHeader.contains(QStringLiteral("void highlightCameraForImage(const QString &imagePath);")));
    EXPECT_TRUE(workspaceHeader.contains(QStringLiteral("void clearHighlightedCamera();")));
    EXPECT_TRUE(workspaceSource.contains(QStringLiteral("pose.imagePath")));
    EXPECT_TRUE(workspaceSource.contains(QStringLiteral("setHighlightedCameraPath(imagePath)")));
}
```

- [ ] **Step 2: Run the focused test target and verify failure**

Run:

```powershell
cmake --build build --target test_gui_project_utils --parallel
```

Expected result before implementation:

```text
error: 'class MainMenu' has no member named 'toggleWorkspaceAction'
error: 'class MainMenu' has no member named 'togglePropertiesAction'
error: 'class MainMenu' has no member named 'togglePhotosAction'
error: 'class MainMenu' has no member named 'toggleWorldOriginAction'
```

If the build directory is not configured, run this first:

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
```

Expected configure result:

```text
-- Configuring done
-- Generating done
-- Build files have been written to: E:/code/plascan/build
```

### Task 2: Implement `SelectionPropertiesWidget`

**Files:**
- Create: `src/gui/widgets/SelectionPropertiesWidget.h`
- Create: `src/gui/widgets/SelectionPropertiesWidget.cpp`

- [ ] **Step 1: Create the header**

Use this interface:

```cpp
#pragma once

#include <QJsonObject>
#include <QVector>
#include <QWidget>

class QLabel;
class QTableWidget;

class SelectionPropertiesWidget : public QWidget
{
    Q_OBJECT

public:
    explicit SelectionPropertiesWidget(QWidget *parent = nullptr);

public slots:
    void clearSelection();
    void showPhotoProperties(const QJsonObject &meta, const QString &imagePath);
    void showResourceProperties(const QJsonObject &meta,
                                const QString &section,
                                const QString &resourcePath);

private:
    struct PropertyRow
    {
        QString name;
        QString value;
    };

    void setRows(const QString &title, const QVector<PropertyRow> &rows);
    void appendFileRows(QVector<PropertyRow> *rows, const QString &path) const;
    QJsonObject findImageEntry(const QJsonObject &meta, const QString &imagePath) const;
    QString imageAlignedText(const QJsonObject &entry) const;
    QString cameraCenterText(const QJsonObject &entry) const;
    QString intrinsicsText(const QJsonObject &entry) const;
    static QString fileSizeText(qint64 bytes);

    QLabel *_title = nullptr;
    QTableWidget *_table = nullptr;
};
```

- [ ] **Step 2: Create the source**

Implement the source with light metadata and file inspection:

```cpp
#include "SelectionPropertiesWidget.h"

#include <QFileInfo>
#include <QHeaderView>
#include <QImageReader>
#include <QDateTime>
#include <QJsonArray>
#include <QLabel>
#include <QTableWidget>
#include <QVBoxLayout>

SelectionPropertiesWidget::SelectionPropertiesWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(4);

    _title = new QLabel(tr("未选择"), this);
    _title->setTextInteractionFlags(Qt::TextSelectableByMouse);

    _table = new QTableWidget(this);
    _table->setColumnCount(2);
    _table->setHorizontalHeaderLabels({tr("属性"), tr("值")});
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->verticalHeader()->setVisible(false);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setSelectionMode(QAbstractItemView::NoSelection);
    _table->setAlternatingRowColors(true);

    layout->addWidget(_title);
    layout->addWidget(_table, 1);
}

void SelectionPropertiesWidget::clearSelection()
{
    setRows(tr("未选择"), {});
}

void SelectionPropertiesWidget::showPhotoProperties(const QJsonObject &meta, const QString &imagePath)
{
    QVector<PropertyRow> rows;
    const QFileInfo info(imagePath);
    const QJsonObject entry = findImageEntry(meta, imagePath);

    rows.push_back({tr("名称"), info.fileName()});
    rows.push_back({tr("路径"), imagePath});
    rows.push_back({tr("定向状态"), imageAlignedText(entry)});

    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    const QSize size = reader.size();
    if (size.isValid())
    {
        rows.push_back({tr("尺寸"), QStringLiteral("%1 x %2").arg(size.width()).arg(size.height())});
    }
    rows.push_back({tr("格式"), QString::fromLatin1(reader.format()).toUpper()});

    appendFileRows(&rows, imagePath);

    const QString center = cameraCenterText(entry);
    if (!center.isEmpty())
    {
        rows.push_back({tr("相机中心"), center});
    }
    const QString intrinsics = intrinsicsText(entry);
    if (!intrinsics.isEmpty())
    {
        rows.push_back({tr("内方位"), intrinsics});
    }

    setRows(tr("照片属性"), rows);
}

void SelectionPropertiesWidget::showResourceProperties(const QJsonObject &meta,
                                                       const QString &section,
                                                       const QString &resourcePath)
{
    Q_UNUSED(meta)

    QVector<PropertyRow> rows;
    const QFileInfo info(resourcePath);
    rows.push_back({tr("类型"), section});
    rows.push_back({tr("名称"), info.fileName().isEmpty() ? section : info.fileName()});
    rows.push_back({tr("路径"), resourcePath});
    rows.push_back({tr("扩展名"), info.suffix().isEmpty() ? tr("无") : info.suffix().toLower()});
    appendFileRows(&rows, resourcePath);

    if (section.contains(tr("点云")) || section.contains(tr("连接点")))
    {
        rows.push_back({tr("详细属性"), tr("未扫描详细属性，避免在主界面阻塞大点云加载")});
    }

    setRows(tr("资源属性"), rows);
}

void SelectionPropertiesWidget::setRows(const QString &title, const QVector<PropertyRow> &rows)
{
    if (_title)
    {
        _title->setText(title);
    }
    if (!_table)
    {
        return;
    }

    _table->setRowCount(rows.size());
    for (int row = 0; row < rows.size(); ++row)
    {
        auto *nameItem = new QTableWidgetItem(rows[row].name);
        auto *valueItem = new QTableWidgetItem(rows[row].value);
        nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
        valueItem->setFlags(valueItem->flags() & ~Qt::ItemIsEditable);
        _table->setItem(row, 0, nameItem);
        _table->setItem(row, 1, valueItem);
    }
    _table->resizeRowsToContents();
}

void SelectionPropertiesWidget::appendFileRows(QVector<PropertyRow> *rows, const QString &path) const
{
    if (!rows)
    {
        return;
    }
    const QFileInfo info(path);
    rows->push_back({tr("存在"), info.exists() ? tr("是") : tr("否")});
    if (info.exists())
    {
        rows->push_back({tr("文件大小"), fileSizeText(info.size())});
        rows->push_back({tr("修改时间"), info.lastModified().toString(Qt::ISODate)});
    }
}

QJsonObject SelectionPropertiesWidget::findImageEntry(const QJsonObject &meta, const QString &imagePath) const
{
    const QString targetPath = QFileInfo(imagePath).canonicalFilePath();
    const QString targetName = QFileInfo(imagePath).fileName();
    const QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : images)
    {
        const QJsonObject entry = value.toObject();
        const QString path = entry.value(QStringLiteral("path")).toString();
        const QString name = entry.value(QStringLiteral("name")).toString();
        const QString canonical = QFileInfo(path).canonicalFilePath();
        if ((!targetPath.isEmpty() && canonical == targetPath) || path == imagePath || name == targetName)
        {
            return entry;
        }
    }
    return {};
}

QString SelectionPropertiesWidget::imageAlignedText(const QJsonObject &entry) const
{
    if (entry.isEmpty())
    {
        return tr("未知");
    }
    const bool hasCenter = entry.contains(QStringLiteral("center")) || entry.contains(QStringLiteral("camera_center"));
    const bool aligned = entry.value(QStringLiteral("aligned")).toBool(hasCenter);
    return aligned ? tr("已定向") : tr("未定向");
}

QString SelectionPropertiesWidget::cameraCenterText(const QJsonObject &entry) const
{
    QJsonArray center = entry.value(QStringLiteral("center")).toArray();
    if (center.isEmpty())
    {
        center = entry.value(QStringLiteral("camera_center")).toArray();
    }
    if (center.size() < 3)
    {
        return {};
    }
    return QStringLiteral("%1, %2, %3")
        .arg(center.at(0).toDouble(), 0, 'f', 3)
        .arg(center.at(1).toDouble(), 0, 'f', 3)
        .arg(center.at(2).toDouble(), 0, 'f', 3);
}

QString SelectionPropertiesWidget::intrinsicsText(const QJsonObject &entry) const
{
    const QJsonObject intrinsics = entry.value(QStringLiteral("intrinsics")).toObject();
    if (intrinsics.isEmpty())
    {
        return {};
    }
    return QStringLiteral("fx=%1, fy=%2, cx=%3, cy=%4")
        .arg(intrinsics.value(QStringLiteral("fx")).toDouble(), 0, 'f', 2)
        .arg(intrinsics.value(QStringLiteral("fy")).toDouble(), 0, 'f', 2)
        .arg(intrinsics.value(QStringLiteral("cx")).toDouble(), 0, 'f', 2)
        .arg(intrinsics.value(QStringLiteral("cy")).toDouble(), 0, 'f', 2);
}

QString SelectionPropertiesWidget::fileSizeText(qint64 bytes)
{
    if (bytes < 1024)
    {
        return QStringLiteral("%1 B").arg(bytes);
    }
    if (bytes < 1024 * 1024)
    {
        return QStringLiteral("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    }
    return QStringLiteral("%1 MB").arg(bytes / 1024.0 / 1024.0, 0, 'f', 1);
}
```

- [ ] **Step 3: Run the widget contract test and verify the remaining failures are from missing photo/menu/3D pieces**

Run:

```powershell
cmake --build build --target test_gui_project_utils --parallel
```

Expected result after this task:

```text
error: 'class MainMenu' has no member named 'toggleWorkspaceAction'
```

The `SelectionPropertiesWidget` file-existence checks should no longer be part of the failure list.

### Task 3: Implement `PhotoStripWidget`

**Files:**
- Create: `src/gui/widgets/PhotoStripWidget.h`
- Create: `src/gui/widgets/PhotoStripWidget.cpp`

- [ ] **Step 1: Create the header**

Use this interface:

```cpp
#pragma once

#include <QHash>
#include <QIcon>
#include <QJsonObject>
#include <QWidget>

class QListWidget;
class QListWidgetItem;

class PhotoStripWidget : public QWidget
{
    Q_OBJECT

public:
    explicit PhotoStripWidget(QWidget *parent = nullptr);

public slots:
    void loadFromJson(const QJsonObject &meta);
    void setCurrentPhoto(const QString &imagePath);
    void clearPhotos();

signals:
    void photoActivated(const QString &imagePath);

private:
    struct ThumbnailResult
    {
        QString path;
        QIcon icon;
        bool loaded = false;
    };

    QListWidgetItem *createItem(const QJsonObject &entry);
    void startThumbnailLoad(const QString &imagePath);
    void applyThumbnail(const ThumbnailResult &result);
    static ThumbnailResult loadThumbnail(const QString &imagePath);
    static QString normalizedPath(const QString &imagePath);

    QListWidget *_list = nullptr;
    QHash<QString, QListWidgetItem *> _itemsByPath;
    QHash<QString, QIcon> _thumbnailCache;
};
```

- [ ] **Step 2: Create the source**

Implement icon-mode layout and asynchronous thumbnail loading:

```cpp
#include "PhotoStripWidget.h"

#include <QFileIconProvider>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QImageReader>
#include <QJsonArray>
#include <QListView>
#include <QListWidget>
#include <QPixmap>
#include <QVBoxLayout>
#include <QtConcurrent>

namespace {
constexpr int PathRole = Qt::UserRole + 1;
constexpr int ThumbWidth = 132;
constexpr int ThumbHeight = 88;
}

PhotoStripWidget::PhotoStripWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);

    _list = new QListWidget(this);
    _list->setViewMode(QListView::IconMode);
    _list->setMovement(QListView::Static);
    _list->setResizeMode(QListView::Adjust);
    _list->setSelectionMode(QAbstractItemView::SingleSelection);
    _list->setIconSize(QSize(ThumbWidth, ThumbHeight));
    _list->setGridSize(QSize(220, 140));
    _list->setSpacing(8);
    _list->setUniformItemSizes(true);
    _list->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    _list->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    layout->addWidget(_list);

    connect(_list, &QListWidget::itemClicked, this, [this](QListWidgetItem *item)
    {
        if (!item)
        {
            return;
        }
        emit photoActivated(item->data(PathRole).toString());
    });
    connect(_list, &QListWidget::itemActivated, this, [this](QListWidgetItem *item)
    {
        if (!item)
        {
            return;
        }
        emit photoActivated(item->data(PathRole).toString());
    });
}

void PhotoStripWidget::loadFromJson(const QJsonObject &meta)
{
    clearPhotos();
    const QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &value : images)
    {
        const QJsonObject entry = value.toObject();
        QListWidgetItem *item = createItem(entry);
        if (!item)
        {
            continue;
        }
        _list->addItem(item);
        const QString path = item->data(PathRole).toString();
        _itemsByPath.insert(normalizedPath(path), item);
        startThumbnailLoad(path);
    }
}

void PhotoStripWidget::setCurrentPhoto(const QString &imagePath)
{
    if (!_list)
    {
        return;
    }
    QListWidgetItem *item = _itemsByPath.value(normalizedPath(imagePath), nullptr);
    if (!item)
    {
        return;
    }
    _list->setCurrentItem(item);
    _list->scrollToItem(item, QAbstractItemView::PositionAtCenter);
}

void PhotoStripWidget::clearPhotos()
{
    _itemsByPath.clear();
    if (_list)
    {
        _list->clear();
    }
}

QListWidgetItem *PhotoStripWidget::createItem(const QJsonObject &entry)
{
    const QString path = entry.value(QStringLiteral("path")).toString();
    if (path.isEmpty())
    {
        return nullptr;
    }
    const QString name = entry.value(QStringLiteral("name")).toString(QFileInfo(path).fileName());
    QFileIconProvider provider;
    auto *item = new QListWidgetItem(provider.icon(QFileIconProvider::File), name);
    item->setData(PathRole, path);

    const bool aligned = entry.value(QStringLiteral("aligned")).toBool(
        entry.contains(QStringLiteral("center")) || entry.contains(QStringLiteral("camera_center")));
    item->setToolTip(aligned ? tr("%1\n已定向").arg(path) : tr("%1\n未定向").arg(path));
    return item;
}

void PhotoStripWidget::startThumbnailLoad(const QString &imagePath)
{
    const QString key = normalizedPath(imagePath);
    if (_thumbnailCache.contains(key))
    {
        applyThumbnail({imagePath, _thumbnailCache.value(key), true});
        return;
    }

    auto *watcher = new QFutureWatcher<ThumbnailResult>(this);
    connect(watcher, &QFutureWatcher<ThumbnailResult>::finished, this, [this, watcher]()
    {
        const ThumbnailResult result = watcher->result();
        watcher->deleteLater();
        applyThumbnail(result);
    });
    watcher->setFuture(QtConcurrent::run(&PhotoStripWidget::loadThumbnail, imagePath));
}

void PhotoStripWidget::applyThumbnail(const ThumbnailResult &result)
{
    if (!result.loaded)
    {
        return;
    }
    const QString key = normalizedPath(result.path);
    _thumbnailCache.insert(key, result.icon);
    QListWidgetItem *item = _itemsByPath.value(key, nullptr);
    if (item)
    {
        item->setIcon(result.icon);
    }
}

PhotoStripWidget::ThumbnailResult PhotoStripWidget::loadThumbnail(const QString &imagePath)
{
    QImageReader reader(imagePath);
    reader.setAutoTransform(true);
    reader.setScaledSize(QSize(ThumbWidth, ThumbHeight));
    const QImage image = reader.read();
    if (image.isNull())
    {
        return {imagePath, QIcon(), false};
    }
    return {imagePath, QIcon(QPixmap::fromImage(image)), true};
}

QString PhotoStripWidget::normalizedPath(const QString &imagePath)
{
    const QString canonical = QFileInfo(imagePath).canonicalFilePath();
    return canonical.isEmpty() ? QFileInfo(imagePath).absoluteFilePath() : canonical;
}
```

- [ ] **Step 3: Run the focused test target**

Run:

```powershell
cmake --build build --target test_gui_project_utils --parallel
```

Expected result after this task:

```text
error: 'class MainMenu' has no member named 'toggleWorkspaceAction'
```

The `PhotoStripWidget` file-existence checks should no longer be part of the failure list.

### Task 4: Register New Widgets In CMake

**Files:**
- Modify: `src/gui/cmake/GuiSources.cmake`

- [ ] **Step 1: Add new source files to `GUI_SOURCES`**

Insert the two `.cpp` files near the other `widgets/*` entries:

```cmake
  widgets/PhotoStripWidget.cpp
  widgets/SelectionPropertiesWidget.cpp
```

- [ ] **Step 2: Add new headers to `GUI_HEADERS`**

Insert the two `.h` files near the other `widgets/*` entries:

```cmake
  widgets/PhotoStripWidget.h
  widgets/SelectionPropertiesWidget.h
```

- [ ] **Step 3: Run configure and focused build**

Run:

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --target test_gui_project_utils --parallel
```

Expected result after this task:

```text
error: 'class MainMenu' has no member named 'toggleWorkspaceAction'
```

No missing-source-file errors should remain for `PhotoStripWidget` or `SelectionPropertiesWidget`.

### Task 5: Add Menu Actions For Panels And World Origin

**Files:**
- Modify: `src/gui/menu/MainMenu.h`
- Modify: `src/gui/menu/MainMenu.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add getters to `MainMenu.h`**

Place these next to existing view/window getters:

```cpp
QAction *toggleWorkspaceAction() const;
QAction *togglePropertiesAction() const;
QAction *togglePhotosAction() const;
QAction *toggleWorldOriginAction() const;
```

- [ ] **Step 2: Add private members to `MainMenu.h`**

Place these beside `_toggleLogAct`, `_toggleGizmoAct`, and `_toggleCamerasAct`:

```cpp
QAction *_toggleWorkspaceAct{};
QAction *_togglePropertiesAct{};
QAction *_togglePhotosAct{};
QAction *_toggleWorldOriginAct{};
```

- [ ] **Step 3: Add a small action factory in `MainMenu.cpp`**

Add this helper in the anonymous namespace where existing lookup helpers live:

```cpp
QAction *ensureCheckableAction(QObject *parent,
                               QMenu *menu,
                               const QString &objectName,
                               const QString &text,
                               bool checked,
                               QAction *before = nullptr)
{
    QAction *action = parent ? parent->findChild<QAction *>(objectName) : nullptr;
    if (!action)
    {
        action = new QAction(text, parent);
        action->setObjectName(objectName);
        action->setCheckable(true);
        action->setChecked(checked);
        if (menu)
        {
            if (before)
            {
                menu->insertAction(before, action);
            }
            else
            {
                menu->addAction(action);
            }
        }
    }
    else
    {
        action->setText(text);
        action->setCheckable(true);
        action->setChecked(checked);
        if (menu && !menu->actions().contains(action))
        {
            if (before)
            {
                menu->insertAction(before, action);
            }
            else
            {
                menu->addAction(action);
            }
        }
    }
    return action;
}
```

- [ ] **Step 4: Bind/create actions in the UI-backed `MainMenu` constructor path**

After `_toggleLogAct`, `_toggleGizmoAct`, and `_toggleCamerasAct` are resolved, add:

```cpp
QObject *actionParent = _mainWindow;
_toggleWorkspaceAct = ensureCheckableAction(actionParent,
                                            windowMenu,
                                            QStringLiteral("actionToggleWorkspace"),
                                            tr("工作区"),
                                            true);
_togglePropertiesAct = ensureCheckableAction(actionParent,
                                             windowMenu,
                                             QStringLiteral("actionToggleProperties"),
                                             tr("属性"),
                                             true);
_togglePhotosAct = ensureCheckableAction(actionParent,
                                         windowMenu,
                                         QStringLiteral("actionTogglePhotos"),
                                         tr("照片"),
                                         true);
_toggleWorldOriginAct = ensureCheckableAction(actionParent,
                                              viewMenu,
                                              QStringLiteral("actionToggleWorldOrigin"),
                                              tr("显示世界原点"),
                                              true,
                                              _toggleCamerasAct);
```

Then build `WindowPanel` with the full window list:

```cpp
QList<QAction *> windowActs = {
    _toggleWorkspaceAct,
    _togglePropertiesAct,
    _togglePhotosAct,
    _toggleLogAct,
    _featureInfoAct
};
auto *wp = new WindowPanel(windowMenu);
wp->setActions(windowActs);
auto *wa = new QWidgetAction(windowMenu);
wa->setDefaultWidget(wp);
windowMenu->addAction(wa);
```

- [ ] **Step 5: Bind/create actions in the fallback `MainMenu` constructor path**

When the fallback menus are created programmatically, use the same action names and texts:

```cpp
_toggleGizmoAct = ensureCheckableAction(this,
                                        viewMenu,
                                        QStringLiteral("actionToggleGizmo"),
                                        tr("显示操控球"),
                                        true);
_toggleCamerasAct = ensureCheckableAction(this,
                                          viewMenu,
                                          QStringLiteral("actionToggleCameras"),
                                          tr("显示相机"),
                                          true);
_toggleWorldOriginAct = ensureCheckableAction(this,
                                              viewMenu,
                                              QStringLiteral("actionToggleWorldOrigin"),
                                              tr("显示世界原点"),
                                              true);

_toggleWorkspaceAct = ensureCheckableAction(this,
                                            windowMenu,
                                            QStringLiteral("actionToggleWorkspace"),
                                            tr("工作区"),
                                            true);
_togglePropertiesAct = ensureCheckableAction(this,
                                             windowMenu,
                                             QStringLiteral("actionToggleProperties"),
                                             tr("属性"),
                                             true);
_togglePhotosAct = ensureCheckableAction(this,
                                         windowMenu,
                                         QStringLiteral("actionTogglePhotos"),
                                         tr("照片"),
                                         true);
_toggleLogAct = ensureCheckableAction(this,
                                      windowMenu,
                                      QStringLiteral("actionToggleLog"),
                                      tr("日志"),
                                      true);
```

- [ ] **Step 6: Add getter definitions**

Add near the other getter definitions:

```cpp
QAction *MainMenu::toggleWorkspaceAction() const { return _toggleWorkspaceAct; }
QAction *MainMenu::togglePropertiesAction() const { return _togglePropertiesAct; }
QAction *MainMenu::togglePhotosAction() const { return _togglePhotosAct; }
QAction *MainMenu::toggleWorldOriginAction() const { return _toggleWorldOriginAct; }
```

- [ ] **Step 7: Run focused menu tests**

Run:

```powershell
cmake --build build --target test_gui_project_utils --parallel
ctest --test-dir build --output-on-failure -R "test_gui_project_utils"
```

Expected result after this task:

```text
MetashapeStyleGuiTest.MainMenuExposesMetashapeWindowAndOriginActions passes
MetashapeStyleGuiTest.MainWindowWiresPhotoPropertiesAndCameraHighlight fails
MetashapeStyleGuiTest.CameraSceneSupportsHighlightedCameraAndOriginToggle fails
```

### Task 6: Add Property Panel And Photos Dock To `MainWindow`

**Files:**
- Modify: `src/gui/main_window/MainWindow.h`
- Modify: `src/gui/main_window/MainWindow.cpp`

- [ ] **Step 1: Add forward declarations in `MainWindow.h`**

Add these declarations near existing GUI class declarations:

```cpp
class PhotoStripWidget;
class SelectionPropertiesWidget;
```

- [ ] **Step 2: Add private helpers and members in `MainWindow.h`**

Use `_` private member names:

```cpp
void setupSelectionPanels();
void connectDockAction(QAction *action, QWidget *panel, const QString &settingKey);
void selectPhoto(const QString &imagePath, bool openImage);
void selectResource(const QString &section, const QString &resourcePath);
QJsonObject currentProjectMeta() const;

QSplitter *_leftPanelSplitter{};
SelectionPropertiesWidget *_selectionProperties{};
QDockWidget *_photosDock{};
PhotoStripWidget *_photoStrip{};
```

- [ ] **Step 3: Include the new widgets in `MainWindow.cpp`**

Add:

```cpp
#include "PhotoStripWidget.h"
#include "SelectionPropertiesWidget.h"
#include <QSignalBlocker>
```

- [ ] **Step 4: Call `setupSelectionPanels()` after `setupUi()`**

In the constructor, use this order:

```cpp
setupUi();
setupSelectionPanels();
_mainMenu = new MainMenu(this);
```

- [ ] **Step 5: Implement `setupSelectionPanels()`**

Compose the left lower property panel without editing `MainWindow.ui`:

```cpp
void MainWindow::setupSelectionPanels()
{
    if (!_mainSplitter || !_leftTabs)
    {
        return;
    }

    _selectionProperties = new SelectionPropertiesWidget(this);
    _selectionProperties->setObjectName(QStringLiteral("selectionProperties"));

    _leftPanelSplitter = new QSplitter(Qt::Vertical, _mainSplitter);
    _leftPanelSplitter->setObjectName(QStringLiteral("leftPanelSplitter"));

    const int leftIndex = _mainSplitter->indexOf(_leftTabs);
    if (leftIndex < 0)
    {
        return;
    }
    QWidget *oldLeft = _mainSplitter->replaceWidget(leftIndex, _leftPanelSplitter);
    if (oldLeft)
    {
        oldLeft->setParent(_leftPanelSplitter);
        _leftPanelSplitter->addWidget(oldLeft);
    }
    _leftPanelSplitter->addWidget(_selectionProperties);
    _leftPanelSplitter->setStretchFactor(0, 3);
    _leftPanelSplitter->setStretchFactor(1, 1);

    _photosDock = new QDockWidget(tr("照片"), this);
    _photosDock->setObjectName(QStringLiteral("photosDock"));
    _photosDock->setAllowedAreas(Qt::BottomDockWidgetArea);
    _photoStrip = new PhotoStripWidget(_photosDock);
    _photosDock->setWidget(_photoStrip);
    addDockWidget(Qt::BottomDockWidgetArea, _photosDock);
}
```

- [ ] **Step 6: Implement `connectDockAction()`**

Use one helper for menu-to-panel visibility and persistence:

```cpp
void MainWindow::connectDockAction(QAction *action, QWidget *panel, const QString &settingKey)
{
    if (!action || !panel)
    {
        return;
    }

    connect(action, &QAction::toggled, panel, &QWidget::setVisible);
    connect(action, &QAction::toggled, this, [this, settingKey](bool on)
    {
        saveUiSetting(QJsonObject{{settingKey, on}});
    });

    if (auto *dock = qobject_cast<QDockWidget *>(panel))
    {
        connect(dock, &QDockWidget::visibilityChanged, action, [action](bool visible)
        {
            QSignalBlocker blocker(action);
            action->setChecked(visible);
        });
    }
}
```

- [ ] **Step 7: Wire new menu actions in `setupMenuConnections()`**

Add:

```cpp
connectDockAction(_mainMenu->toggleWorkspaceAction(), _leftPanelSplitter, QStringLiteral("workspace_visible"));
connectDockAction(_mainMenu->togglePropertiesAction(), _selectionProperties, QStringLiteral("properties_visible"));
connectDockAction(_mainMenu->togglePhotosAction(), _photosDock, QStringLiteral("photos_visible"));

if (_mainMenu->toggleWorldOriginAction() && _workspaceCenter && _workspaceCenter->modelView())
{
    connect(_mainMenu->toggleWorldOriginAction(), &QAction::toggled,
            _workspaceCenter->modelView(), &CameraSceneWidget::setShowWorldOrigin);
    connect(_mainMenu->toggleWorldOriginAction(), &QAction::toggled, this, [this](bool on)
    {
        saveUiSetting(QJsonObject{{QStringLiteral("world_origin_visible"), on}});
    });
}
```

- [ ] **Step 8: Implement `currentProjectMeta()`**

Add:

```cpp
QJsonObject MainWindow::currentProjectMeta() const
{
    return _projectManager ? _projectManager->currentMeta() : QJsonObject{};
}
```

- [ ] **Step 9: Implement `selectPhoto()`**

Add:

```cpp
void MainWindow::selectPhoto(const QString &imagePath, bool openImage)
{
    if (imagePath.isEmpty())
    {
        return;
    }

    _lastSelectedImage = imagePath;

    if (_selectionProperties)
    {
        _selectionProperties->showPhotoProperties(currentProjectMeta(), imagePath);
    }
    if (_photoStrip)
    {
        _photoStrip->setCurrentPhoto(imagePath);
    }
    if (_workspaceCenter)
    {
        _workspaceCenter->highlightCameraForImage(imagePath);
        if (openImage)
        {
            _workspaceCenter->showImageView(imagePath);
        }
    }

    saveUiSetting(QJsonObject{{QStringLiteral("active_image_path"), imagePath}});
}
```

- [ ] **Step 10: Implement `selectResource()`**

Add:

```cpp
void MainWindow::selectResource(const QString &section, const QString &resourcePath)
{
    if (_selectionProperties)
    {
        _selectionProperties->showResourceProperties(currentProjectMeta(), section, resourcePath);
    }
    if (_workspaceCenter)
    {
        _workspaceCenter->clearHighlightedCamera();
    }
}
```

- [ ] **Step 11: Update metadata connections in `setupProjectManager()`**

Extend the existing `projectMetadataChanged` connections:

```cpp
connect(_projectManager, &ProjectManager::projectMetadataChanged, this, [this](const QJsonObject &meta)
{
    if (_photoStrip)
    {
        _photoStrip->loadFromJson(meta);
    }
});
```

Connect the photo strip:

```cpp
if (_photoStrip)
{
    connect(_photoStrip, &PhotoStripWidget::photoActivated, this, [this](const QString &path)
    {
        selectPhoto(path, true);
    });
}
```

Replace duplicated image-selection lambdas with:

```cpp
connect(_dataTree, &DataTreeWidget::imageActivated, this, [this](const QString &path)
{
    selectPhoto(path, true);
});

connect(_referencePanel, &ReferencePanelWidget::imageActivated, this, [this](const QString &path)
{
    selectPhoto(path, true);
});
```

At the beginning of the existing `resourceActivated` lambda, add:

```cpp
selectResource(section, path);
```

- [ ] **Step 12: Load photo strip when a project opens**

In `onProjectOpened()`, after `_workspaceCenter->setProjectMeta(...)`, add:

```cpp
if (_photoStrip && _projectManager)
{
    _photoStrip->loadFromJson(_projectManager->currentMeta());
}
```

- [ ] **Step 13: Persist new UI states in `applyUiSettings()`**

Add:

```cpp
const auto applyVisibility = [](QWidget *widget, QAction *action, const QJsonObject &ui, const QString &key)
{
    if (!ui.contains(key))
    {
        return;
    }
    const bool on = ui.value(key).toBool(true);
    if (widget)
    {
        widget->setVisible(on);
    }
    if (action)
    {
        QSignalBlocker blocker(action);
        action->setChecked(on);
    }
};

applyVisibility(_leftPanelSplitter,
                _mainMenu ? _mainMenu->toggleWorkspaceAction() : nullptr,
                ui,
                QStringLiteral("workspace_visible"));
applyVisibility(_selectionProperties,
                _mainMenu ? _mainMenu->togglePropertiesAction() : nullptr,
                ui,
                QStringLiteral("properties_visible"));
applyVisibility(_photosDock,
                _mainMenu ? _mainMenu->togglePhotosAction() : nullptr,
                ui,
                QStringLiteral("photos_visible"));

if (ui.contains(QStringLiteral("world_origin_visible")) && _mainMenu && _mainMenu->toggleWorldOriginAction())
{
    const bool on = ui.value(QStringLiteral("world_origin_visible")).toBool(true);
    QSignalBlocker blocker(_mainMenu->toggleWorldOriginAction());
    _mainMenu->toggleWorldOriginAction()->setChecked(on);
    if (_workspaceCenter && _workspaceCenter->modelView())
    {
        _workspaceCenter->modelView()->setShowWorldOrigin(on);
    }
}
```

- [ ] **Step 14: Run focused tests and build**

Run:

```powershell
cmake --build build --target test_gui_project_utils --parallel
ctest --test-dir build --output-on-failure -R "test_gui_project_utils"
```

Expected result after this task:

```text
MetashapeStyleGuiTest.MainWindowWiresPhotoPropertiesAndCameraHighlight passes
MetashapeStyleGuiTest.CameraSceneSupportsHighlightedCameraAndOriginToggle fails
```

### Task 7: Add Camera Highlight And World-Origin Toggle To 3D View

**Files:**
- Modify: `src/gui/dialogs/CameraModel3DDialog.h`
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`
- Modify: `src/gui/widgets/WorkspaceCenterWidget.h`
- Modify: `src/gui/widgets/WorkspaceCenterWidget.cpp`

- [ ] **Step 1: Extend `CameraPose` in `CameraModel3DDialog.h`**

Add `imagePath`:

```cpp
struct CameraPose
{
    QString name;
    QString imagePath;
    QVector3D center;
    QMatrix3x3 rotation;
};
```

- [ ] **Step 2: Add public 3D view APIs**

Add:

```cpp
void setHighlightedCameraPath(const QString &imagePath);
void setHighlightedCameraName(const QString &imageName);
void clearHighlightedCamera();
void setShowWorldOrigin(bool show);
bool isWorldOriginVisible() const;
```

- [ ] **Step 3: Add private state and helper declarations**

Add:

```cpp
bool isCameraHighlighted(const CameraPose &pose) const;
QString normalizedCameraPath(const QString &imagePath) const;

QString _highlightedCameraPath;
QString _highlightedCameraName;
bool _showWorldOrigin = true;
```

- [ ] **Step 4: Implement highlight and origin setters**

Add to `CameraModel3DDialog.cpp`:

```cpp
void CameraSceneWidget::setHighlightedCameraPath(const QString &imagePath)
{
    _highlightedCameraPath = normalizedCameraPath(imagePath);
    _highlightedCameraName.clear();
    update();
}

void CameraSceneWidget::setHighlightedCameraName(const QString &imageName)
{
    _highlightedCameraPath.clear();
    _highlightedCameraName = imageName;
    update();
}

void CameraSceneWidget::clearHighlightedCamera()
{
    _highlightedCameraPath.clear();
    _highlightedCameraName.clear();
    update();
}

void CameraSceneWidget::setShowWorldOrigin(bool show)
{
    if (_showWorldOrigin == show)
    {
        return;
    }
    _showWorldOrigin = show;
    update();
}

bool CameraSceneWidget::isWorldOriginVisible() const
{
    return _showWorldOrigin;
}
```

- [ ] **Step 5: Implement camera matching helpers**

Add:

```cpp
QString CameraSceneWidget::normalizedCameraPath(const QString &imagePath) const
{
    const QString canonical = QFileInfo(imagePath).canonicalFilePath();
    return canonical.isEmpty() ? QFileInfo(imagePath).absoluteFilePath() : canonical;
}

bool CameraSceneWidget::isCameraHighlighted(const CameraPose &pose) const
{
    if (!_highlightedCameraPath.isEmpty()
        && normalizedCameraPath(pose.imagePath) == _highlightedCameraPath)
    {
        return true;
    }
    if (!_highlightedCameraName.isEmpty()
        && (pose.name == _highlightedCameraName || QFileInfo(pose.imagePath).fileName() == _highlightedCameraName))
    {
        return true;
    }
    return false;
}
```

- [ ] **Step 6: Update camera overlay colors**

In the camera drawing loop, replace the fixed orange/brown colors with:

```cpp
const bool highlighted = isCameraHighlighted(pose);
const QColor frustumColor = highlighted ? QColor(245, 90, 105, 215)
                                        : QColor(42, 122, 200, 165);
const QColor centerColor = highlighted ? QColor(255, 70, 95, 235)
                                       : QColor(32, 100, 180, 210);
const int lineWidth = highlighted ? 2 : 1;
painter.setPen(QPen(frustumColor, lineWidth));
```

Keep labels for highlighted cameras fully visible:

```cpp
if (highlighted || shouldDrawCameraLabel(cameraIndex, labelBudget))
{
    painter.setPen(highlighted ? QColor(80, 0, 0, 240) : QColor(35, 35, 35, 185));
    painter.drawText(labelPoint, pose.name);
}
```

- [ ] **Step 7: Gate the world origin cross**

Wrap the current origin cross block with:

```cpp
if (_showWorldOrigin)
{
    const QVector3D origin(0.0f, 0.0f, 0.0f);
    QPointF originPoint;
    if (projectScenePoint(origin, &originPoint))
    {
        painter.setPen(QPen(QColor(120, 120, 120, 150), 1));
        painter.drawLine(QPointF(originPoint.x() - 14.0, originPoint.y()),
                         QPointF(originPoint.x() + 14.0, originPoint.y()));
        painter.drawLine(QPointF(originPoint.x(), originPoint.y() - 14.0),
                         QPointF(originPoint.x(), originPoint.y() + 14.0));
        painter.drawText(originPoint + QPointF(6.0, -6.0), QStringLiteral("XYZ(0,0,0)"));
    }
}
```

- [ ] **Step 8: Reduce point cloud and manipulation sphere visual weight**

Use smaller point sizes and softer sphere colors in the existing rendering blocks:

```cpp
glPointSize(1.6f);
```

For the manipulation sphere overlay, use:

```cpp
painter.setPen(QPen(QColor(150, 150, 150, 90), 1));
painter.setBrush(QColor(220, 220, 220, 38));
```

- [ ] **Step 9: Add workspace forwarding methods**

In `WorkspaceCenterWidget.h`, add:

```cpp
void highlightCameraForImage(const QString &imagePath);
void clearHighlightedCamera();
```

In `WorkspaceCenterWidget.cpp`, add:

```cpp
void WorkspaceCenterWidget::highlightCameraForImage(const QString &imagePath)
{
    if (_modelView)
    {
        _modelView->setHighlightedCameraPath(imagePath);
    }
}

void WorkspaceCenterWidget::clearHighlightedCamera()
{
    if (_modelView)
    {
        _modelView->clearHighlightedCamera();
    }
}
```

- [ ] **Step 10: Populate `CameraPose::imagePath` from project metadata**

In `WorkspaceCenterWidget::setProjectMeta()`, set:

```cpp
pose.imagePath = entry.value(QStringLiteral("path")).toString();
if (pose.imagePath.isEmpty())
{
    pose.imagePath = entry.value(QStringLiteral("image_path")).toString();
}
```

Keep the existing `pose.name` assignment so name-based fallback still works.

- [ ] **Step 11: Run focused tests**

Run:

```powershell
cmake --build build --target test_gui_project_utils --parallel
ctest --test-dir build --output-on-failure -R "test_gui_project_utils"
```

Expected result after this task:

```text
MetashapeStyleGuiTest.CameraSceneSupportsHighlightedCameraAndOriginToggle passes
```

### Task 8: Final Build, Focused Verification, And Manual GUI Check

**Files:**
- No new production files.
- Verify: GUI executable and focused tests.

- [ ] **Step 1: Re-check workspace before verification**

Run:

```powershell
git status --short
git submodule status --recursive
Get-Process | Where-Object { $_.ProcessName -match 'plascan|feature_extract_cli|feature_match_cli|dense_match_cli|triangulate_cli|cmake|ctest|ninja' } | Select-Object Id,ProcessName,Path
```

Expected:

```text
No 3rdparty submodule dirty state introduced by this task
No long-running build/test process blocks the next command
```

- [ ] **Step 2: Build the GUI and focused test binary**

Run:

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --target plascan_gui --parallel
cmake --build build --target test_gui_project_utils --parallel
```

Expected:

```text
Build succeeds for plascan_gui
Build succeeds for test_gui_project_utils
```

- [ ] **Step 3: Run the focused GUI utility test**

Run:

```powershell
ctest --test-dir build --output-on-failure -R "test_gui_project_utils"
```

Expected:

```text
100% tests passed
```

If unrelated pre-existing tests fail inside the same binary, report the exact failing test names and keep the Metashape-style GUI tests separated in the final verification summary.

- [ ] **Step 4: Run a manual GUI smoke check**

Run:

```powershell
.\build\bin\plascan_gui.bin.exe
```

Manual checks:

```text
视图 -> 窗口 contains 工作区, 属性, 照片, 日志.
照片 panel is visible by default at the bottom.
属性 panel is visible under the left workspace area.
Clicking a photo in the project tree opens the image, updates 属性, highlights the same photo in 照片, and highlights its camera in 3D if pose exists.
Clicking a photo in 照片 opens the image and updates 属性.
Clicking 连接点 or 点云 opens the 3D point-cloud view and updates 属性 without changing the photo selection.
视图 -> 显示世界原点 toggles the small XYZ(0,0,0) origin cross.
视图 -> 显示相机 toggles camera overlay without hiding point cloud/model data.
Unselected cameras are blue and selected camera is red/pink.
The manipulation sphere is lighter than before and does not dominate point-cloud inspection.
```

- [ ] **Step 5: Final status report**

Run:

```powershell
git status --short
```

Report:

```text
Changed GUI files for this task
Existing unrelated dirty files still present and not reverted
Build/test command results
Any manual GUI observations or skipped manual checks
```

Do not commit unless the user asks for a commit after reviewing the verification summary.

## Self-Review Checklist

- Spec coverage:
  - Photo selection property display: Task 2 and Task 6.
  - Point cloud/resource property display: Task 2 and Task 6.
  - Bottom Photos widget default open: Task 3 and Task 6.
  - View -> Window toggles: Task 5 and Task 6.
  - Photo-to-camera highlight: Task 6 and Task 7.
  - Camera visualization colors: Task 7.
  - Point cloud and manipulation sphere visual weight: Task 7.
  - World origin cross explanation and toggle: Task 5, Task 6, and Task 7.
  - State persistence: Task 6.
  - Focused tests and verification: Task 1 and Task 8.

- Type consistency:
  - `PhotoStripWidget::photoActivated(const QString &imagePath)` is consumed by `MainWindow::selectPhoto(const QString &imagePath, bool openImage)`.
  - `WorkspaceCenterWidget::highlightCameraForImage(const QString &imagePath)` forwards to `CameraSceneWidget::setHighlightedCameraPath(const QString &imagePath)`.
  - Private members use `_` prefix in all new C++ declarations.

- Expected risk:
  - The source-structure tests are intentionally conservative because full GUI automation is fragile for Qt dock/splitter layout.
  - Large point-cloud attributes are not scanned synchronously; the user gets explicit “未扫描详细属性” text instead of a frozen GUI.
