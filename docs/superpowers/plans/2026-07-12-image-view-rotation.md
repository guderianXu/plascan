# Image View Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为主工作区单影像查看器增加可持久化的向左/向右 90 度旋转，并保证所有场景覆盖层同步旋转。

**Architecture:** `ImageViewRotationSettings` 作为纯配置助手处理路径键和 JSON；`CanvasWidget` 维护单影像就绪状态、角度和视图矩阵；`MainMenu` 提供菜单及工具栏动作；`MainWindow` 在项目 UI 设置与画布信号之间协调恢复和保存。

**Tech Stack:** C++17、Qt6 Widgets/Core、QGraphicsView、QJsonObject、CMake、GTest、CTest

---

仓库规则要求只有用户明确提出时才提交，因此本计划不包含 `git commit`。实施时保留当前工作区已有改动，只补丁式修改本功能涉及的文件。

## 文件结构

- Create: `src/gui/config/ImageViewRotationSettings.h/.cpp` — 角度规范化、路径键和 JSON 映射。
- Modify: `src/gui/widgets/CanvasWidget.h/.cpp` — 单影像旋转状态和视图变换。
- Modify: `src/gui/menu/MainMenu.h/.cpp` — 左转/右转动作、图标和工具栏按钮。
- Modify: `src/gui/main_window/MainWindow.h/.cpp` — 动作连接、启用状态、恢复与保存。
- Modify: `src/gui/cmake/GuiSources.cmake` — 注册配置助手。
- Modify: `tests/CMakeLists.txt` — 将配置助手加入 `test_gui_project_utils`。
- Modify: `tests/test_gui_project_utils.cpp` — 配置、画布、菜单和集成回归测试。
- Modify: `docs/PROJECT_ARCHITECTURE.md` — 记录影像查看旋转设置边界。

### Task 1: 影像旋转配置助手

**Files:**
- Create: `src/gui/config/ImageViewRotationSettings.h`
- Create: `src/gui/config/ImageViewRotationSettings.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/CMakeLists.txt`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 编写失败测试**

增加 `ImageViewRotationSettingsTest`，覆盖：

```cpp
EXPECT_EQ(normalizeImageViewRotationDegrees(-90), 270);
EXPECT_EQ(normalizeImageViewRotationDegrees(450), 90);
EXPECT_EQ(normalizeImageViewRotationDegrees(45), 0);

QJsonObject rotations;
rotations = withImageViewRotation(rotations, QStringLiteral("G:/中文/IMG.TIF"), 90);
EXPECT_EQ(imageViewRotationForPath(rotations, QStringLiteral("g:/中文/img.tif")), 90);
rotations = withImageViewRotation(rotations, QStringLiteral("G:/中文/IMG.TIF"), 0);
EXPECT_TRUE(rotations.isEmpty());
```

- [ ] **Step 2: 运行 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
```

Expected: 因 `ImageViewRotationSettings.h` 或目标函数不存在而失败。

- [ ] **Step 3: 实现最小配置 API**

在 `xjw::gui::config` 中声明并实现：

```cpp
int normalizeImageViewRotationDegrees(int degrees);
QString imageViewRotationPathKey(const QString &imagePath);
int imageViewRotationForPath(const QJsonObject &rotations, const QString &imagePath);
QJsonObject withImageViewRotation(const QJsonObject &rotations,
                                  const QString &imagePath,
                                  int degrees);
```

非 90 度倍数返回 0；Windows 路径键转小写；0 度删除条目。

- [ ] **Step 4: 注册 CMake 并运行 GREEN**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
$env:QT_QPA_PLATFORM='offscreen'
build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=ImageViewRotationSettingsTest.*
```

Expected: 配置助手测试全部通过。

### Task 2: CanvasWidget 视图旋转

**Files:**
- Modify: `src/gui/widgets/CanvasWidget.h`
- Modify: `src/gui/widgets/CanvasWidget.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 编写画布失败测试**

测试公开接口和变换：

```cpp
CanvasWidget canvas;
canvas.setViewRotationDegrees(0);
canvas.rotateRight();
EXPECT_EQ(canvas.viewRotationDegrees(), 90);
canvas.rotateLeft();
EXPECT_EQ(canvas.viewRotationDegrees(), 0);
canvas.setViewRotationDegrees(-90);
EXPECT_EQ(canvas.viewRotationDegrees(), 270);
```

使用临时 PNG 调用 `showImage()`，处理事件直到 `hasDisplayImage()` 为 true；验证四次右转回到 0，`resetView()` 后角度不变，`showMatchedPair()` 或空路径使旋转动作不改变角度。

- [ ] **Step 2: 运行 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
```

Expected: 因旋转接口不存在而编译失败。

- [ ] **Step 3: 实现状态与信号**

在 `CanvasWidget` 增加：

```cpp
void rotateLeft();
void rotateRight();
void setViewRotationDegrees(int degrees);
int viewRotationDegrees() const;
bool hasDisplayImage() const;

void viewRotationChanged(const QString &imagePath, int degrees);
void displayImageReadyChanged(bool ready);
```

私有状态为 `_viewRotationDegrees` 和 `_singleImageReady`。用户旋转先检查就绪状态，记录视口中心，调用 `QGraphicsView::rotate(delta)`，恢复中心并发出旋转信号。

- [ ] **Step 4: 处理异步加载和重置**

`showImage()` 开始时设置未就绪并将角度重置为 0；加载成功、底图加入场景后设置就绪并发出 `activeImageChanged`，使主窗口同步应用持久化角度，随后 `fitInView()` 保持旋转。空路径、失败和 `showMatchedPair()` 保持未就绪。

将 `resetView()` 改为：

```cpp
resetTransform();
rotate(_viewRotationDegrees);
_zoomFactor = 1.0;
fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
```

- [ ] **Step 5: 运行 GREEN**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=CanvasImageRotationTest.*
```

Expected: 画布旋转测试全部通过。

### Task 3: 菜单和工具栏动作

**Files:**
- Modify: `src/gui/menu/MainMenu.h`
- Modify: `src/gui/menu/MainMenu.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 编写动作失败测试**

实例化 `MainMenu` 后验证：

```cpp
ASSERT_NE(menu.rotateImageLeftAction(), nullptr);
ASSERT_NE(menu.rotateImageRightAction(), nullptr);
EXPECT_EQ(menu.rotateImageLeftAction()->toolTip(), QStringLiteral("向左旋转"));
EXPECT_EQ(menu.rotateImageRightAction()->toolTip(), QStringLiteral("向右旋转"));
EXPECT_FALSE(menu.rotateImageLeftAction()->icon().isNull());
EXPECT_FALSE(menu.rotateImageRightAction()->icon().isNull());
```

并验证 `toolButtonRotateImageLeft`、`toolButtonRotateImageRight` 位于主工具栏，稳定按钮尺寸不小于 44x44，图标尺寸不小于 32x32。

- [ ] **Step 2: 运行 RED**

Expected: 访问器或按钮不存在。

- [ ] **Step 3: 创建动作与图标**

在“视图”菜单中、放大动作之前插入 `actionRotateImageLeft` 和 `actionRotateImageRight`。使用 `QPainter` 生成影像轮廓加弯曲箭头图标，左右方向可直接辨识，禁用态由 Qt 自动生成。

增加访问器和成员：

```cpp
QAction *rotateImageLeftAction() const;
QAction *rotateImageRightAction() const;
QAction *_rotateImageLeftAct{};
QAction *_rotateImageRightAct{};
```

- [ ] **Step 4: 安装工具栏按钮并运行 GREEN**

创建两个 icon-only `QToolButton`，对象名分别为 `toolButtonRotateImageLeft` 和 `toolButtonRotateImageRight`，图标 36x36、按钮 48x48，插入 `actionManualPointCloudPrune` 之前。

```powershell
$env:QT_QPA_PLATFORM='offscreen'
build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=MainMenuImageRotationTest.*
```

Expected: 动作、图标、菜单顺序和工具栏按钮测试通过。

### Task 4: MainWindow 持久化与动作状态

**Files:**
- Modify: `src/gui/main_window/MainWindow.h`
- Modify: `src/gui/main_window/MainWindow.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 编写集成失败测试**

用源码契约和信号测试验证：

- 左右动作分别连接 `CanvasWidget::rotateLeft/rotateRight`。
- `displayImageReadyChanged` 同时控制两个动作的 enabled 状态。
- `applyUiSettings()` 读取 `image_view_rotations`。
- `viewRotationChanged` 更新旋转映射并调用 `saveUiSetting()`。
- 项目关闭后清空映射并禁用动作。

- [ ] **Step 2: 运行 RED**

Expected: 缺少连接和 `image_view_rotations` 处理。

- [ ] **Step 3: 实现主窗口协调**

在 `MainWindow` 增加 `_imageViewRotations`。初始化时禁用旋转动作并连接动作、就绪信号和旋转信号。

`activeImageChanged` 中按路径查询 `_imageViewRotations`，调用 `setViewRotationDegrees()` 后继续保存 `active_image_path`。`viewRotationChanged` 中更新映射并保存：

```cpp
_imageViewRotations = xjw::gui::config::withImageViewRotation(
    _imageViewRotations, imagePath, degrees);
saveUiSetting(QJsonObject{
    {QStringLiteral("image_view_rotations"), _imageViewRotations}
});
```

`applyUiSettings()` 读取映射；`onProjectClosed()` 清空并禁用动作。

- [ ] **Step 4: 运行 GREEN**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=MainWindowImageRotationTest.*:ImageViewRotationSettingsTest.*:CanvasImageRotationTest.*:MainMenuImageRotationTest.*
```

Expected: 本功能全部测试通过。

### Task 5: 文档与最终验证

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Verify: all files above

- [ ] **Step 1: 更新架构文档**

在 GUI config 和 widgets 说明中加入：

```text
ImageViewRotationSettings：维护项目级、按影像路径索引的查看旋转角度；CanvasWidget 仅执行视图变换，不修改影像或摄影测量坐标。
```

- [ ] **Step 2: 运行专项 CTest**

```powershell
ctest --test-dir build/windows-vcpkg-cuda-release -C Release --output-on-failure -R "ImageViewRotation|CanvasImageRotation|MainMenuImageRotation|MainWindowImageRotation"
```

Expected: 相关测试 100% 通过。

- [ ] **Step 3: 构建 Release GUI**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target plascan_gui --parallel 8
```

Expected: `build/windows-vcpkg-cuda-release/bin/plascan.exe` 链接成功。

- [ ] **Step 4: 检查补丁**

```powershell
git diff --check -- src/gui/config/ImageViewRotationSettings.h src/gui/config/ImageViewRotationSettings.cpp src/gui/widgets/CanvasWidget.h src/gui/widgets/CanvasWidget.cpp src/gui/menu/MainMenu.h src/gui/menu/MainMenu.cpp src/gui/main_window/MainWindow.h src/gui/main_window/MainWindow.cpp src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt tests/test_gui_project_utils.cpp docs/PROJECT_ARCHITECTURE.md
```

Expected: 无空白错误；允许既有 CRLF 转换提示。

- [ ] **Step 5: 人工检查**

打开带兴趣点和蒙版的照片，分别左右旋转，确认底图和覆盖层保持对齐；切换两张角度不同的照片后返回，确认分别恢复；保存并重开项目，确认旋转状态保留；切到模型、双图对比或空项目时确认按钮禁用。
