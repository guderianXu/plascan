# PlaScan Vulkan 渲染迁移设计

日期：2026-07-05

## 目标

将 PlaScan 当前 3D 场景的 OpenGL 渲染迁移为 Vulkan 渲染。首期范围聚焦现有 `CameraSceneWidget`，保持用户可见的 3D 视图能力不退化：

- 显示相机姿态、相机视锥体、相机高亮和名称标签。
- 显示稀疏/密集点云、PLY/OBJ 网格、无面片模型点云和包围盒线框。
- 保留旋转、缩放、平移、操控球、手动框选裁剪和异步加载进度覆盖层。
- 移除 3D 视图对 `QOpenGLWidget`、`QOpenGLFunctions_4_3_Core`、`QOpenGLBuffer`、`QOpenGLShaderProgram`、`QOpenGLVertexArrayObject` 的依赖。

## 当前结构

现有 OpenGL 渲染集中在：

- `src/gui/dialogs/CameraModel3DDialog.h`
- `src/gui/dialogs/CameraModel3DDialog.cpp`
- `src/gui/widgets/WorkspaceCenterWidget.ui`
- `src/gui/dialogs/CameraModel3DDialog.ui`
- `src/gui/CMakeLists.txt`
- `cmake/PlascanPackages.cmake`
- `tests/test_gui_project_utils.cpp`

`CameraSceneWidget` 当前继承 `QOpenGLWidget`。它在同一个类中同时负责：

- 读取和缓存相机、点云、网格数据。
- 将 CPU 数据整理为 GPU 顶点数组。
- 创建 OpenGL shader、VAO、VBO。
- 在 `paintGL()` 中绘制点云、网格、模型点和包围盒。
- 使用 `QPainter` 绘制相机覆盖层、操控球、坐标轴和加载进度。
- 处理鼠标、键盘和手动裁剪。

本地构建环境显示 Qt 为 6.11.1，`QRhiWidget` 和 `qsb.exe` 可用，适合使用 Qt RHI 的 Vulkan 后端。没有采用 `QVulkanWindow`，因为当前 3D 视图嵌在 Qt Widgets 布局和 `.ui` 文件中，`QRhiWidget` 能更稳定地保持 QWidget 集成方式。

## 选定方案

采用 `QRhiWidget` 路线：

- `CameraSceneWidget` 改为继承 `QRhiWidget`。
- 构造函数中调用 `setApi(QRhiWidget::Api::Vulkan)`。
- 渲染实现使用 Qt RHI 资源和命令：
  - `QRhiBuffer` 保存顶点数据和 uniform 数据。
  - `QRhiGraphicsPipeline` 保存点云、网格、线框绘制管线。
  - `QRhiShaderResourceBindings` 绑定 MVP、法线矩阵、点大小和光照参数。
  - `QRhiCommandBuffer` 在 `render()` 中提交绘制命令。
- Shader 从内联 GLSL 改为源文件和 `.qsb` 资源。
- Vulkan 初始化失败时不静默回退 OpenGL；3D 视图显示中文错误提示并写入日志。

该方案的边界是“Vulkan 作为渲染后端”，但不直接手写裸 Vulkan swapchain、render pass 和 descriptor 管理。Qt RHI 负责跨平台表面和生命周期管理，实际 API 固定为 Vulkan。

## 组件设计

### `CameraSceneWidget`

保持现有公开 API 不变：

- `setCameraPoses(const QVector<CameraPose> &poses)`
- `setPointCloud(const RenderCloud &cloud)`
- `setMesh(const RenderCloud &mesh)`
- `loadPointCloudFromXyz(const QString &xyzPath)`
- `loadModelFromPly(const QString &plyPath)`
- `loadModelFromObj(const QString &objPath)`
- `setShowGizmo(bool show)`
- `setShowCameras(bool show)`
- `setHighlightedCameraPath(const QString &imagePath)`
- `setHighlightedCameraName(const QString &imageName)`
- `clearHighlightedCamera()`
- 手动裁剪相关方法和信号

生命周期替换为：

- `initialize(QRhiCommandBuffer *cb)`：创建 RHI 管线、uniform buffer、静态资源占位，并记录 Vulkan/RHI 初始化状态。
- `render(QRhiCommandBuffer *cb)`：按需上传资源，开始 render pass，绘制 3D 内容，然后触发覆盖层绘制。
- `releaseResources()`：释放所有 QRhi 资源并重置 dirty 状态。
- `resizeEvent(QResizeEvent *event)`：调用基类并标记依赖视口的资源或 uniform 需要更新。

`projectToScreen()`、相机覆盖层、操控球、坐标轴、手动裁剪和异步加载逻辑继续使用 Qt 矩阵和 `QPainter`，避免把与 GPU API 无关的交互逻辑一起重写。

### 渲染资源

在 `CameraSceneWidget` 内部新增轻量资源结构：

- `RhiBufferSet`：封装一个顶点 buffer、顶点数、stride、dirty 标记。
- `RhiPipelineSet`：封装 graphics pipeline、shader resource bindings、uniform buffer。
- `SceneUniforms`：保存 MVP、model-view、normal matrix、point size、light direction 等 shader 参数。

首期保留一个类内实现，避免大范围文件拆分。若 `CameraModel3DDialog.cpp` 因迁移继续膨胀，再在后续任务中拆出 `CameraSceneRhiResources.*`。

### Shader

新增 shader 源文件：

- `src/gui/shaders/camera_scene_color.vert`
- `src/gui/shaders/camera_scene_color.frag`
- `src/gui/shaders/camera_scene_mesh.vert`
- `src/gui/shaders/camera_scene_mesh.frag`

构建时用 Qt ShaderTools 的 `qsb` 生成：

- `src/gui/shaders/camera_scene_color.vert.qsb`
- `src/gui/shaders/camera_scene_color.frag.qsb`
- `src/gui/shaders/camera_scene_mesh.vert.qsb`
- `src/gui/shaders/camera_scene_mesh.frag.qsb`

`.qsb` 文件纳入 Qt resource，运行时通过 `QShader::fromSerialized()` 加载。Shader 输入语义保持：

- 点云/线框：`location 0 = vec3 position`，`location 1 = vec3 color`。
- 网格/法向点云：`location 0 = vec3 position`，`location 1 = vec3 normal`，`location 2 = vec3 color`。

## 数据流

1. 业务层调用 `CameraSceneWidget` 的现有接口设置相机、点云或模型。
2. Widget 继续在 CPU 侧维护 `_poses`、`_cloud`、缓存中心、半径和 AABB。
3. 数据变化时设置 `_gpuDirty = true`。
4. 下一帧 `render()` 中执行 `uploadGpuData()`：
   - 点云整理为 interleaved `xyz rgb`。
   - 网格整理为 interleaved `xyz normal rgb`。
   - 含法向无面片点云走 mesh pipeline 的 point 绘制。
   - 包围盒整理为 interleaved `xyz rgb`。
5. `render()` 更新 uniform，按点云、网格、模型点、线框顺序绘制。
6. 覆盖层继续通过 `QPainter` 绘制在 widget 上。

## 错误处理

- Shader 资源缺失：显示“Vulkan 渲染着色器资源缺失”，日志记录资源路径。
- RHI 初始化失败：显示“Vulkan 渲染初始化失败，请检查显卡驱动和 Qt Vulkan 支持”，日志记录 API 和 Qt 版本。
- Buffer 创建失败：显示“Vulkan 顶点缓冲创建失败”，日志记录数据类型和字节数。
- `.qsb` 编译失败时 CMake 配置或构建失败，不能运行到 GUI 后才失败。

不做 OpenGL 回退。这样用户能明确知道当前环境是否真正运行 Vulkan 渲染。

## 构建变更

- `cmake/PlascanPackages.cmake` 增加 `ShaderTools` 组件。
- `src/gui/CMakeLists.txt` 移除 `Qt6::OpenGL`、`Qt6::OpenGLWidgets` 链接，增加 `Qt6::ShaderTools` 或对应 shader 编译工具依赖。
- `src/gui/CMakeLists.txt` 增加 shader 编译规则，生成 `.qsb` 并纳入资源。
- `resources/resources.qrc` 或 GUI 专用 qrc 增加 shader 资源。
- 若 Linux/Docker 依赖缺少 Vulkan runtime 或 Qt ShaderTools，更新 `docker/Dockerfile.ubuntu2404` 和构建说明。

## 测试策略

结构测试：

- `tests/test_gui_project_utils.cpp` 不再断言 `QOpenGLFunctions_4_3_Core *_gl = nullptr;`。
- 新增断言 `CameraSceneWidget` 继承 `QRhiWidget`。
- 新增断言构造函数设置 `QRhiWidget::Api::Vulkan`。
- 新增断言不再包含 `QOpenGLWidget`、`QOpenGLShaderProgram`、`QOpenGLBuffer`、`QOpenGLVertexArrayObject`。
- 新增断言 shader `.qsb` 资源注册在 qrc 中。

构建验证：

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target plascan_gui -j 8
```

聚焦测试：

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CameraScene|CameraModel3D|GuiProject|MainWindow|WorkspaceCenter" --output-on-failure
```

必要时再跑更宽 GUI 回归：

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "Gui|gui_project" --output-on-failure
```

手动烟测：

- 启动 `plascan_gui`。
- 打开含相机和点云/模型的项目。
- 切到 3D 视图。
- 确认点云、模型、相机、包围盒、操控球和坐标轴可见。
- 拖拽旋转、滚轮缩放、中键平移。
- 打开 PLY/OBJ 和 XYZ 点云。
- 在 Vulkan 不可用环境确认显示中文错误而不是空白崩溃。

## 非目标

- 不重写 `CanvasWidget` 的 2D 影像绘制。
- 不把所有 GUI 绘制迁移到 Vulkan。
- 不新增 OpenGL/Vulkan 后端运行时切换。
- 不引入裸 Vulkan swapchain 管理。
- 不优化大点云分块、LOD 或 GPU picking。

## 风险

- `QRhiWidget`/QRhi 是 Qt 的 RHI 接口，源码兼容性可能随 Qt 版本变化。当前项目本地 Qt 为 6.11.1，首期按该版本实现。
- Qt Widgets 上的 `QPainter` 覆盖层与 RHI 渲染混合需要实测，若出现闪烁或覆盖层时序问题，需要改为在 `paintEvent()` 或 RHI pass 后单独处理。
- 现有测试大量使用源码字符串断言，迁移时需要同步更新，否则会出现非行为性失败。
- 本次不保留 OpenGL 回退，部分没有 Vulkan 驱动的机器会显示错误提示而不是 3D 内容。

## 批准状态

用户已选择方案 A：`QRhiWidget` + Vulkan API，不创建 worktree，在当前工作区继续实施。
