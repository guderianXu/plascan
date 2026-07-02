# PlaScan Metashape 风格 GUI 优化设计

日期：2026-07-02

## 目标

将 PlaScan 主界面调整为更接近 Metashape 的摄影测量工作台体验：

- 选择照片时，左下属性面板显示照片、相机和定向状态。
- 选择点云、连接点、模型、DEM、DOM 时，属性面板显示对应成果属性。
- 主界面底部新增“照片”面板，默认打开，可通过“视图 -> 窗口”打开或关闭。
- 照片面板与 3D 视图联动：选中照片时，对应相机在 3D 视图中高亮。
- 优化 3D 视图中的相机、点云、操控球和世界原点十字显示。

## 当前结构

相关现有组件：

- `MainWindow`：主窗口、菜单连接、ProjectManager 绑定、底部日志 Dock。
- `MainMenu`：菜单动作封装，已有“视图 -> 窗口”面板入口。
- `DataTreeWidget`：工作区资源树，已有 `imageActivated` 和 `resourceActivated` 信号。
- `WorkspaceCenterWidget`：中央影像、三维模型、双图对比、观测网络视图的切换容器。
- `CameraSceneWidget`：OpenGL 3D 视图，已有相机姿态、点云、网格、操控球、坐标轴、原点十字的基础绘制。
- `CanvasWidget`：2D 影像显示和特征点叠加。

当前缺口：

- 属性显示仍主要依赖右键弹窗，缺少持续可见的属性面板。
- 照片缩略图不是主界面常驻 Dock。
- 资源选择、照片选择、属性面板和 3D 相机高亮之间缺少统一联动。
- 3D 相机和点云视觉风格与 Metashape 差距明显。
- 世界原点十字已在 `CameraSceneWidget` 中有基础绘制，但还没有独立菜单开关和清晰语义。

## 界面布局

采用 Metashape 风格布局：

- 左侧上方：工作区树，保留项目资源结构。
- 左侧下方：属性 Dock，显示当前选择对象属性。
- 中央：`WorkspaceCenterWidget`，继续承载影像、模型、点云、观测网络。
- 底部：照片 Dock，默认显示项目照片缩略图。
- 底部日志 Dock 保留，可与照片 Dock 并存或 tabify。

“视图 -> 窗口”中应包含：

- 工作区
- 属性
- 照片
- 日志

默认状态：

- 工作区显示。
- 属性显示。
- 照片显示。
- 日志保持现有默认行为。
- 相机、操控球、世界原点十字默认显示。

## 新增组件

### `PhotoStripWidget`

底部照片面板，职责：

- 从项目 metadata 加载照片列表。
- 显示缩略图、文件名、定向状态。
- 支持单选，后续可扩展多选。
- 发出 `photoActivated(path)` 信号。
- 根据外部选择同步高亮当前照片。

首版属性：

- 使用异步缩略图加载，避免打开大项目时卡 UI。
- 缩略图缓存只保存在内存，不写项目文件。
- 已定向照片显示轻量标记，未定向照片使用灰色状态。

后续扩展：

- 多选。
- 按是否定向过滤。
- 禁用/启用照片。
- 按名称、拍摄时间、匹配数量排序。

### `SelectionPropertiesWidget`

左下属性面板，职责：

- 接收选择上下文：照片、点云、连接点、模型、DEM、DOM、匹配、观测网络。
- 显示只读属性表。
- 不直接做磁盘删除、工程修改或业务处理。

属性来源优先级：

1. 项目 metadata 中已记录的结构化字段。
2. 文件头或轻量 IO 读取的基本属性。
3. 文件系统属性，例如路径、大小、修改时间。

首版显示内容：

- 照片：名称、路径、尺寸、格式、文件大小、是否已定向、相机中心、姿态、焦距、主点。
- 连接点/稀疏点云：路径、点数、颜色通道、包围盒、来源结果、连接点倍率。
- 稠密点云：路径、点数、颜色、法线、confidence/classification 是否存在、生成算法、来源深度图。
- 模型：路径、顶点数、面数、纹理路径、算法。
- DEM：路径、尺寸、分辨率、投影、NoData、coverage/confidence 质量栅格。
- DOM：路径、尺寸、分辨率、投影、来源 DEM/模型。

大文件属性读取必须异步或轻量化。首版不强制完整扫描 LAZ、PLY 大文件；如果只能得到路径和文件大小，应显示“未扫描详细属性”。

## 选择联动

新增统一选择上下文：

```text
DataTreeWidget / PhotoStripWidget
        -> MainWindow 统一选择处理
        -> SelectionPropertiesWidget 更新属性
        -> WorkspaceCenterWidget 打开对应视图
        -> CameraSceneWidget 高亮相机
```

行为规则：

- 点击照片：
  - 中央打开照片。
  - 属性面板显示照片属性。
  - 照片 Dock 高亮该照片。
  - 如果该照片已有相机参数，3D 视图中对应相机高亮。

- 点击连接点/稀疏点云：
  - 中央打开 3D 视图并加载点云。
  - 属性面板显示点云属性。
  - 不改变照片 Dock 选择。

- 点击稠密点云：
  - 中央打开 3D 视图并加载点云。
  - 属性面板显示稠密点云属性。

- 点击模型：
  - 中央打开 3D 视图并加载模型。
  - 属性面板显示模型属性。

- 点击 DEM/DOM/深度图：
  - 中央打开 2D 影像视图。
  - 属性面板显示栅格成果属性。

## 3D 可视化

### 相机

当前相机覆盖层继续由 `CameraSceneWidget` 绘制，但增加选择状态：

- 未选中相机：蓝色视锥体，透明度较高，弱化标签。
- 当前选中相机：红/粉色视锥体，线宽更高，标签清晰。
- 无相机参数照片：不绘制相机，属性面板显示“未定向”。

新增接口：

```cpp
void setHighlightedCameraPath(const QString &imagePath);
void setHighlightedCameraName(const QString &imageName);
```

内部用标准化路径或文件名匹配 `CameraPose`。

### 点云

点云显示目标：

- 默认点尺寸更细，减少“糊成片”的感觉。
- 保留原始 RGB 颜色；无颜色时使用中性灰绿。
- 继续使用深度预写入策略减少后层点穿透。
- 增加后续可配置项：点大小、是否使用真实颜色、是否启用深度轮廓。

首版只暴露内部常量和菜单入口预留，不做复杂设置面板。

### 操控球

保留操控球，但降低视觉权重：

- 透明度降低。
- 线条更细。
- 默认开启，可通过“视图 -> 显示操控球”关闭。

### 世界原点十字

Metashape 点云下方的十字主要用于坐标参考，表示场景原点或参考中心，不是点云数据本身。

PlaScan 增加明确语义：

- 名称：显示世界原点。
- 默认开启。
- 通过“视图 -> 显示世界原点”开关。
- 在原点位置绘制小十字和短标签；当原点离当前场景太远或投影不可见时不绘制。

## 状态持久化

使用现有 UI 设置体系保存：

- 属性 Dock 可见性。
- 照片 Dock 可见性。
- 日志 Dock 可见性。
- 左侧 splitter / dock 大小。
- 3D 显示开关：相机、操控球、世界原点。

不在首版保存：

- 照片 Dock 滚动位置。
- 缩略图缓存。
- 点云显示详细参数。

## 错误处理

- 照片路径缺失：属性面板显示路径缺失，中央视图不切换到空白。
- 缩略图加载失败：显示文件图标和文件名。
- 大点云属性扫描失败：显示基本文件属性和错误摘要，不阻塞主界面。
- 相机高亮找不到对应姿态：属性面板正常显示照片，3D 视图清除高亮。

## 测试计划

新增或扩展 GUI 测试：

- `MainMenuTest`：窗口菜单包含工作区、属性、照片、日志；世界原点开关存在。
- `MainWindowTest`：照片 Dock 和属性 Dock 默认可见。
- `PhotoStripWidgetTest`：能从 metadata 加载照片、显示定向状态、发出选择信号。
- `SelectionPropertiesWidgetTest`：照片、点云、模型、DEM/DOM 属性字段可显示。
- `GuiSelectionLinkageTest`：点击照片会更新属性、打开影像、同步照片面板选择、向 3D 视图发送相机高亮。
- `CameraSceneWidgetTest` 或源代码结构测试：提供相机高亮和世界原点开关接口。

验证命令：

```bash
plascan-dev -Quiet
ninja -C E:\code\plascan\build\windows-vcpkg-cuda-release test_gui_project_utils
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -R "MainMenuTest|MainWindowTest|PhotoStrip|SelectionProperties|GuiSelection|CameraScene|ProjectDashboardWidgetTest|DataTreeWidgetTest" --output-on-failure
```

## 分阶段实施

### 阶段 1：面板和菜单

- 新增 `PhotoStripWidget`。
- 新增 `SelectionPropertiesWidget`。
- 在 `MainWindow.ui` 或 `MainWindow::setupUi()` 中接入属性 Dock 和照片 Dock。
- `MainMenu` 增加窗口菜单动作。
- 接入可见性持久化。

验收：

- 打开项目后底部照片面板默认显示。
- 视图菜单能开关照片、属性、日志。
- 属性面板能显示当前选择的基本信息。

### 阶段 2：选择联动

- `DataTreeWidget` 选择资源时发出统一选择信息。
- `PhotoStripWidget` 选择照片时同步中央视图和属性面板。
- `WorkspaceCenterWidget` 或 `MainWindow` 把照片选择传给 `CameraSceneWidget`。
- `CameraSceneWidget` 支持相机高亮。

验收：

- 点击照片后，中央打开照片，照片 Dock 和属性面板同步。
- 3D 模型视图中对应相机变色。
- 点击点云/模型/DEM/DOM 后属性面板显示对应属性。

### 阶段 3：3D 视觉优化

- 调整相机视锥体颜色、透明度、标签密度。
- 调整点云默认点大小和无颜色点云配色。
- 降低操控球遮挡。
- 增加世界原点开关。

验收：

- 相机高亮接近 Metashape 的蓝/红视觉语义。
- 点云显示更细，不明显糊成块。
- 世界原点十字可开关。
- 操控球对点云遮挡降低。

## 非目标

本设计不包含：

- 重做全部主窗口架构。
- 替换 OpenGL 渲染后端。
- 实现完整 LAZ/COPC 属性扫描。
- 实现点云分类、编辑、裁剪的新业务流程。
- 实现 Metashape 全部窗口和工具栏行为。
