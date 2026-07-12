# Workspace Section Icons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 PlaScan 工作区一级分类节点增加 B + B1 风格的语义图标，并保持现有节点显示和交互行为不变。

**Architecture:** 新增独立的 `WorkspaceSectionIcons` 图标工厂，用枚举表达工作区分类并用 `QPainter` 生成多尺寸图标。`DataTreeWidget` 在创建一级节点时显式传入分类枚举，测试从真实树模型读取并验证每个可见分类图标。

**Tech Stack:** C++17、Qt6 Widgets/Gui、CMake、GTest、Ninja/MSVC

---

## 文件结构

- Create: `src/gui/widgets/WorkspaceSectionIcons.h` — 分类枚举和图标工厂公开接口。
- Create: `src/gui/widgets/WorkspaceSectionIcons.cpp` — B 配色及 B1 图形的多尺寸 Qt 绘制实现。
- Modify: `src/gui/widgets/DataTreeWidget.h` — `createSection` 接受类型安全的分类参数。
- Modify: `src/gui/widgets/DataTreeWidget.cpp` — 设置树图标尺寸并为每个一级节点绑定对应图标。
- Modify: `src/gui/cmake/GuiSources.cmake` — 将图标工厂加入 GUI 构建。
- Modify: `tests/CMakeLists.txt` — 将图标工厂加入 `test_gui_project_utils`。
- Create: `tests/test_workspace_section_icons.cpp` — 独立验证图标工厂和真实工作区分类映射。

当前工作区已有其他未提交改动。执行时只补丁式修改上述文件，不覆盖相邻改动；除非用户明确要求，不执行 Git commit。

### Task 1: 用真实工作区模型锁定图标行为

**Files:**
- Create: `tests/test_workspace_section_icons.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 增加图标像素签名辅助函数和失败测试**

在 Qt include 区加入：

```cpp
#include <QCryptographicHash>
```

在测试文件匿名命名空间中加入：

```cpp
QByteArray iconPixelSignature(const QIcon &icon, const QSize &size)
{
    const QImage image = icon.pixmap(size).toImage().convertToFormat(QImage::Format_ARGB32);
    if (image.isNull())
    {
        return {};
    }
    const QByteArray pixels(reinterpret_cast<const char *>(image.constBits()),
                            static_cast<qsizetype>(image.sizeInBytes()));
    return QCryptographicHash::hash(pixels, QCryptographicHash::Sha256);
}
```

新增测试 `DataTreeWidgetTest.WorkspaceSectionsUseDistinctSemanticIcons`。测试构造包含照片、观测网络、连接点、深度图、稠密点云、3D 模型、DEM、正射影像、参考数据和报告的元数据，其中路径字段必须满足当前 `populateFromMeta()` 的显示条件：

```cpp
TEST(DataTreeWidgetTest, WorkspaceSectionsUseDistinctSemanticIcons)
{
    DataTreeWidget tree;
    QJsonObject meta;
    meta[QStringLiteral("images")] = QJsonArray{QStringLiteral("/tmp/image.tif")};
    meta[QStringLiteral("observation_network_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("algorithm"), QStringLiteral("overlap")}}
    };
    meta[QStringLiteral("aerial_triangulation_results")] = QJsonArray{
        QJsonObject{
            {QStringLiteral("sparse_point_count"), 2314},
            {QStringLiteral("files"), QJsonObject{
                {QStringLiteral("sparse_cloud_xyz"), QStringLiteral("/tmp/sparse.xyz")}
            }}
        }
    };
    meta[QStringLiteral("depth_map_results")] = QJsonArray{
        QJsonObject{
            {QStringLiteral("result_type"), QStringLiteral("mvs_depth")},
            {QStringLiteral("depth_png"), QStringLiteral("/tmp/depth.png")}
        }
    };
    meta[QStringLiteral("dense_cloud_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("dense_cloud_xyz"), QStringLiteral("/tmp/dense.ply")}}
    };
    meta[QStringLiteral("model_results")] = QJsonArray{
        QJsonObject{
            {QStringLiteral("model_ply"), QStringLiteral("/tmp/model.ply")},
            {QStringLiteral("vertex_count"), 8},
            {QStringLiteral("face_count"), 12}
        }
    };
    meta[QStringLiteral("dem_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("dem_tif"), QStringLiteral("/tmp/dem.tif")}}
    };
    meta[QStringLiteral("ortho_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("output_path"), QStringLiteral("/tmp/ortho.tif")}}
    };
    meta[QStringLiteral("reference_datasets")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/reference.tif")}}
    };
    meta[QStringLiteral("report_results")] = QJsonArray{
        QJsonObject{{QStringLiteral("path"), QStringLiteral("/tmp/report.json")}}
    };
    tree.loadFromJson(meta);

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->rowCount(), 10);

    QSet<QByteArray> signatures;
    for (int row = 0; row < model->rowCount(); ++row)
    {
        QStandardItem *section = model->item(row, 0);
        ASSERT_NE(section, nullptr);
        ASSERT_FALSE(section->icon().isNull()) << section->text().toStdString();
        const QByteArray signature = iconPixelSignature(section->icon(), QSize(18, 18));
        ASSERT_FALSE(signature.isEmpty()) << section->text().toStdString();
        signatures.insert(signature);
    }
    EXPECT_EQ(signatures.size(), model->rowCount());
}
```

- [ ] **Step 2: 编译并运行单测，确认 RED**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_workspace_section_icons --parallel 8
build\windows-vcpkg-cuda-release\tests\test_workspace_section_icons.exe
```

Expected: 测试在 `EXPECT_EQ(signatures.size(), model->rowCount())` 失败，因为当前所有一级分类都使用 `QStyle::SP_DirIcon`。

### Task 2: 实现类型安全的多尺寸图标工厂

**Files:**
- Create: `src/gui/widgets/WorkspaceSectionIcons.h`
- Create: `src/gui/widgets/WorkspaceSectionIcons.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 定义图标工厂接口**

`WorkspaceSectionIcons.h`：

```cpp
#pragma once

#include <QIcon>

namespace xjw::gui::widgets
{

enum class WorkspaceSection
{
    Photos,
    ObservationNetwork,
    TiePoints,
    DepthMaps,
    DenseCloud,
    Model3D,
    Dem,
    Orthomosaic,
    ReferenceData,
    Reports,
    Unknown
};

QIcon workspaceSectionIcon(WorkspaceSection section);

} // namespace xjw::gui::widgets
```

- [ ] **Step 2: 实现 B 配色和 B1 图形**

`WorkspaceSectionIcons.cpp` 使用透明 `QImage`、抗锯齿 `QPainter` 和 16/18/20/24/32/36/48/64 八种尺寸。坐标以 20×20 设计网格缩放，分类绘制规则固定如下：

| 枚举 | 颜色 | Qt 绘制图元 |
| --- | --- | --- |
| `Photos` | `#D5A52E` | 带标签页的实心文件夹路径 |
| `ObservationNetwork` | `#4B86C5` | 三个圆点和三条连接线 |
| `TiePoints` | `#6482A4` | 四个 5 px 圆点 |
| `DepthMaps` | `#58A6A0` | 两个错位的圆角矩形 |
| `DenseCloud` | `#6482A4` | 七个不同位置的小圆点 |
| `Model3D` | `#737D8B` | 六边多面体，顶面 `#9AA3AF`、右面 `#606A78` |
| `Dem` | `#7A9563` | 浅绿底框和三条贝塞尔等高线 |
| `Orthomosaic` | `#4D8BAC` | 四个拼接块和一条深色地形成果轮廓 |
| `ReferenceData` | `#807A9A` | 圆形靶标和中心十字 |
| `Reports` | `#9A7A55` | 折角文档和三条横线 |
| `Unknown` | `#87919D` | 圆角资源框和中心圆点 |

工厂按以下结构缓存图标，避免每次刷新树时重复绘制：

```cpp
QIcon workspaceSectionIcon(WorkspaceSection section)
{
    static QHash<int, QIcon> cache;
    const int key = static_cast<int>(section);
    const auto found = cache.constFind(key);
    if (found != cache.constEnd())
    {
        return found.value();
    }

    QIcon icon;
    for (const int size : {16, 18, 20, 24, 32, 36, 48, 64})
    {
        icon.addPixmap(drawSectionPixmap(section, size));
    }
    cache.insert(key, icon);
    return icon;
}
```

`drawSectionPixmap()` 必须设置透明背景、`QPainter::Antialiasing`，以 `size / 20.0` 缩放画布并在 `switch (section)` 中按表格绘制全部十一种分支；所有分支都必须产生非透明像素。

- [ ] **Step 3: 将新模块加入两个构建目标**

在 `src/gui/cmake/GuiSources.cmake` 的 widget 源文件和头文件区域分别加入：

```cmake
widgets/WorkspaceSectionIcons.cpp
widgets/WorkspaceSectionIcons.h
```

在 `tests/CMakeLists.txt` 的 `test_gui_project_utils` 源文件列表中加入：

```cmake
${CMAKE_SOURCE_DIR}/src/gui/widgets/WorkspaceSectionIcons.cpp
```

### Task 3: 将语义图标绑定到工作区一级节点

**Files:**
- Modify: `src/gui/widgets/DataTreeWidget.h`
- Modify: `src/gui/widgets/DataTreeWidget.cpp`

- [ ] **Step 1: 修改节点创建接口**

在 `DataTreeWidget.h` 前置声明枚举，并修改两个私有函数：

```cpp
namespace xjw::gui::widgets
{
enum class WorkspaceSection;
}

QStandardItem *createSection(const QString &label,
                             xjw::gui::widgets::WorkspaceSection section);
QStandardItem *createSection(const QString &title,
                             int count,
                             xjw::gui::widgets::WorkspaceSection section);
```

- [ ] **Step 2: 设置树图标尺寸并替换系统文件夹图标**

在 `DataTreeWidget.cpp` 引入 `WorkspaceSectionIcons.h`，构造函数设置：

```cpp
_view->setIconSize(QSize(18, 18));
```

实现节点创建：

```cpp
QStandardItem *DataTreeWidget::createSection(
    const QString &label,
    xjw::gui::widgets::WorkspaceSection section)
{
    auto *nameItem = new QStandardItem(label);
    nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
    nameItem->setIcon(xjw::gui::widgets::workspaceSectionIcon(section));

    auto *pathItem = new QStandardItem(QString());
    auto *storageItem = new QStandardItem(QString());
    pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
    storageItem->setFlags(storageItem->flags() & ~Qt::ItemIsEditable);

    _model->appendRow({nameItem, pathItem, storageItem});
    return nameItem;
}
```

计数重载把 `section` 透传给标签重载。

- [ ] **Step 3: 为所有一级分类显式绑定枚举**

在 `populateFromMeta()` 中按以下映射调用 `createSection`：

```cpp
照片 -> WorkspaceSection::Photos
观测网络 -> WorkspaceSection::ObservationNetwork
连接点 -> WorkspaceSection::TiePoints
深度图 -> WorkspaceSection::DepthMaps
稠密点云 -> WorkspaceSection::DenseCloud
3D模型 -> WorkspaceSection::Model3D
DEM -> WorkspaceSection::Dem
正射影像 -> WorkspaceSection::Orthomosaic
参考数据 -> WorkspaceSection::ReferenceData
报告 -> WorkspaceSection::Reports
```

不得修改节点的显示条件、计数、标题或元数据键。

- [ ] **Step 4: 编译并运行单测，确认 GREEN**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_workspace_section_icons --parallel 8
build\windows-vcpkg-cuda-release\tests\test_workspace_section_icons.exe
```

Expected: 所有 `DataTreeWidgetTest.*` 通过，新测试得到十个非空且像素签名不同的分类图标。

### Task 4: 构建和视觉验证

**Files:**
- Verify only; no additional production files expected.

- [ ] **Step 1: 检查补丁质量**

Run:

```powershell
git diff --check -- src/gui/widgets/WorkspaceSectionIcons.h src/gui/widgets/WorkspaceSectionIcons.cpp src/gui/widgets/DataTreeWidget.h src/gui/widgets/DataTreeWidget.cpp src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt tests/test_gui_project_utils.cpp
```

Expected: 无空白错误；Windows 行尾转换警告可单独说明。

- [ ] **Step 2: 编译 GUI 可执行文件**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target plascan.exe --parallel 8
```

Expected: 退出码 0。如 `plascan.exe` 正在运行导致 `LNK1104`，先报告被占用的进程，不擅自结束用户进程。

- [ ] **Step 3: 运行目标测试**

Run:

```powershell
ctest --test-dir build/windows-vcpkg-cuda-release -C Release --output-on-failure -R "WorkspaceSectionIconsTest"
```

Expected: 工作区树相关测试全部通过，0 failures。

- [ ] **Step 4: 启动 GUI 做人工检查**

启动 `build/windows-vcpkg-cuda-release/bin/plascan.exe`，打开包含多种成果的项目，确认：

- 一级分类图标为 18 px 左右且垂直居中。
- 照片、连接点、深度图、稠密点云、3D 模型、DEM、正射影像、参考数据、报告能够快速区分。
- 3D 模型为分面多面体，DEM 为等高线，正射影像为拼接块。
- 节点选中、高亮、展开、右键菜单及双击激活行为未改变。

不在本计划中执行 commit 或 push；等待用户明确指令。
