# Vulkan Rendering Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PlaScan's current `CameraSceneWidget` legacy rendering path with a Vulkan-backed `QRhiWidget` path.

**Architecture:** Keep the existing `CameraSceneWidget` public API and interaction logic, but replace the legacy QWidget renderer lifecycle with `QRhiWidget::initialize()`, `render()`, and `releaseResources()`. Use Qt RHI with `QRhiWidget::Api::Vulkan`, `QRhiBuffer`, `QRhiGraphicsPipeline`, `QRhiShaderResourceBindings`, and Qt ShaderTools `.qsb` shader resources; fail loudly when Qt is not built with Vulkan.

**Tech Stack:** C++17, Qt 6.11 Widgets, Qt RHI, Qt ShaderTools/qsb, CMake, GTest source-structure tests.

---

## File Structure

- Modify `cmake/PlascanPackages.cmake`: require Qt `ShaderTools` and `ShaderToolsTools`; add a configure-time check that QtGui's public features include Vulkan.
- Modify `vcpkg.json`: declare Vulkan dependency for environments that rebuild Qt with Vulkan support.
- Modify `src/gui/CMakeLists.txt`: compile shader sources with `qt_add_shaders()`, remove legacy renderer link targets, and expose QtGui private RHI include dirs when needed.
- Create `src/gui/shaders/camera_scene_color.vert`: vertex shader for point cloud and line rendering.
- Create `src/gui/shaders/camera_scene_color.frag`: fragment shader for vertex-color rendering.
- Create `src/gui/shaders/camera_scene_mesh.vert`: vertex shader for mesh and normal-point rendering.
- Create `src/gui/shaders/camera_scene_mesh.frag`: fragment shader for Phong mesh rendering.
- Modify `src/gui/dialogs/CameraModel3DDialog.h`: replace legacy renderer includes and members with QRhi declarations and resource holders.
- Modify `src/gui/dialogs/CameraModel3DDialog.cpp`: replace legacy renderer lifecycle, resource upload, and draw calls with QRhi equivalents.
- Modify `tests/test_gui_project_utils.cpp`: add failing source-structure tests for Vulkan/QRhi requirements and remove stale legacy renderer expectations.
- Modify `docs/PROJECT_ARCHITECTURE.md`: change the 3D rendering note from the legacy backend to Vulkan/QRhi.
- Modify `docker/Dockerfile.ubuntu2404` and `scripts/build_win/README.md`: document Vulkan runtime/SDK requirement for rebuilds.

## Task 1: Add Vulkan/QRhi Source-Structure Tests

**Files:**
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Write the failing test**

Add these tests near the existing `CameraSceneWidgetTest` and `CodeStyleTest` cases:

```cpp
TEST(CameraSceneWidgetTest, UsesQrhiWidgetWithVulkanBackend)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    EXPECT_TRUE(header.contains(QStringLiteral("#include <QRhiWidget>")));
    EXPECT_TRUE(header.contains(QStringLiteral("class CameraSceneWidget : public QRhiWidget")));
    EXPECT_TRUE(source.contains(QStringLiteral("setApi(QRhiWidget::Api::Vulkan)")));
    EXPECT_TRUE(header.contains(QStringLiteral("void initialize(QRhiCommandBuffer *cb) override;")));
    EXPECT_TRUE(header.contains(QStringLiteral("void render(QRhiCommandBuffer *cb) override;")));
    EXPECT_TRUE(header.contains(QStringLiteral("void releaseResources() override;")));
}

TEST(CameraSceneWidgetTest, RemovesLegacyRenderingDependencies)
{
    const QString header = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.h"));
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    const QString guiCmake = readProjectSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));
    const QString packages = readProjectSourceFile(QStringLiteral("cmake/PlascanPackages.cmake"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());
    ASSERT_FALSE(guiCmake.isEmpty());
    ASSERT_FALSE(packages.isEmpty());

    const QStringList forbidden = {
        legacyWidgetName(),
        legacyFunctionObjectName(),
        legacyBufferName(),
        legacyShaderProgramName(),
        legacyVertexArrayName(),
        legacyInitializeEntryName(),
        legacyResizeEntryName(),
        legacyPaintEntryName(),
    };
    for (const QString &token : forbidden)
    {
        EXPECT_FALSE(header.contains(token)) << qPrintable(token);
        EXPECT_FALSE(source.contains(token)) << qPrintable(token);
    }

    EXPECT_FALSE(guiCmake.contains(legacyQtTargetName()));
    EXPECT_FALSE(guiCmake.contains(legacyQtWidgetTargetName()));
    EXPECT_FALSE(packages.contains(legacyQtTargetList()));
}

TEST(CameraSceneWidgetTest, RegistersQrhiShaderResources)
{
    const QString guiCmake = readProjectSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));
    ASSERT_FALSE(guiCmake.isEmpty());

    EXPECT_TRUE(guiCmake.contains(QStringLiteral("qt_add_shaders(plascan_gui")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_color.vert")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_color.frag")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_mesh.vert")));
    EXPECT_TRUE(guiCmake.contains(QStringLiteral("shaders/camera_scene_mesh.frag")));
}
```

Remove the stale assertion:

```cpp
EXPECT_TRUE(header.contains(legacyFunctionObjectMemberName()));
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CameraSceneWidgetTest|CodeStyleTest.CameraModel3DDialogUsesLowerCamelPrivateMemberNames" --output-on-failure
```

Expected: FAIL because `CameraSceneWidget` still uses the legacy widget path, legacy renderer symbols, and no shader resources.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_gui_project_utils.cpp
git commit -m "test: require vulkan qrhi camera scene backend"
```

## Task 2: Add Build Dependencies And Shader Resources

**Files:**
- Modify: `cmake/PlascanPackages.cmake`
- Modify: `src/gui/CMakeLists.txt`
- Modify: `vcpkg.json`
- Create: `src/gui/shaders/camera_scene_color.vert`
- Create: `src/gui/shaders/camera_scene_color.frag`
- Create: `src/gui/shaders/camera_scene_mesh.vert`
- Create: `src/gui/shaders/camera_scene_mesh.frag`

- [ ] **Step 1: Update Qt package discovery**

In `cmake/PlascanPackages.cmake`, replace the Qt comment and package line with:

```cmake
# 合并所有模块所需组件（Core/Gui/Widgets/Concurrent/ShaderTools）
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Concurrent ShaderTools ShaderToolsTools)
message(STATUS "plascan: found Qt6 ${Qt6_VERSION}")

get_target_property(_PLASCAN_QT_GUI_PUBLIC_FEATURES Qt6::Gui QT_ENABLED_PUBLIC_FEATURES)
if(NOT "vulkan" IN_LIST _PLASCAN_QT_GUI_PUBLIC_FEATURES)
  message(FATAL_ERROR
    "PlaScan Vulkan rendering requires QtGui built with Vulkan support. "
    "Install Vulkan SDK/loader/headers and rebuild the vcpkg Qt package.")
endif()
```

- [ ] **Step 2: Update vcpkg manifest**

Add Vulkan to `vcpkg.json` dependencies:

```json
"vulkan",
```

Place it next to the existing `"qtbase"` dependency so a fresh manifest install has Vulkan headers/loader before Qt is configured.

- [ ] **Step 3: Add shader source files**

Create `src/gui/shaders/camera_scene_color.vert`:

```glsl
#version 440

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

layout(std140, binding = 0) uniform SceneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    mat4 uNormalMat;
    vec4 uLightDirPointSize;
} ubuf;

layout(location = 0) out vec3 vColor;

void main()
{
    gl_Position = ubuf.uMVP * vec4(aPos, 1.0);
    gl_PointSize = ubuf.uLightDirPointSize.w;
    vColor = aColor;
}
```

Create `src/gui/shaders/camera_scene_color.frag`:

```glsl
#version 440

layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 fragColor;

void main()
{
    fragColor = vec4(vColor, 1.0);
}
```

Create `src/gui/shaders/camera_scene_mesh.vert`:

```glsl
#version 440

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

layout(std140, binding = 0) uniform SceneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    mat4 uNormalMat;
    vec4 uLightDirPointSize;
} ubuf;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;

void main()
{
    gl_Position = ubuf.uMVP * vec4(aPos, 1.0);
    gl_PointSize = ubuf.uLightDirPointSize.w;
    vNormal = mat3(ubuf.uNormalMat) * aNormal;
    vColor = aColor;
}
```

Create `src/gui/shaders/camera_scene_mesh.frag`:

```glsl
#version 440

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;

layout(std140, binding = 0) uniform SceneUniforms
{
    mat4 uMVP;
    mat4 uModelView;
    mat4 uNormalMat;
    vec4 uLightDirPointSize;
} ubuf;

layout(location = 0) out vec4 fragColor;

vec3 srgbToLinear(vec3 c)
{
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec3 linearToSrgb(vec3 c)
{
    return pow(clamp(c, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
}

void main()
{
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(ubuf.uLightDirPointSize.xyz);
    float diff = max(dot(n, lightDir), 0.0);
    vec3 baseLinear = srgbToLinear(vColor);
    vec3 litLinear = baseLinear * (0.55 + 0.75 * diff);
    fragColor = vec4(linearToSrgb(litLinear), 1.0);
}
```

- [ ] **Step 4: Register shaders in CMake**

In `src/gui/CMakeLists.txt`, before `target_link_libraries(plascan_gui PRIVATE ...)`, add:

```cmake
qt_add_shaders(plascan_gui camera_scene_shaders
  PREFIX
    "/shaders"
  FILES
    shaders/camera_scene_color.vert
    shaders/camera_scene_color.frag
    shaders/camera_scene_mesh.vert
    shaders/camera_scene_mesh.frag
)
```

In the link list, remove the legacy renderer Qt targets.

Add:

```cmake
Qt6::ShaderTools
```

If `QRhi` headers are not visible through `Qt6::Gui`, add:

```cmake
target_include_directories(plascan_gui PRIVATE ${Qt6Gui_PRIVATE_INCLUDE_DIRS})
```

- [ ] **Step 5: Run RED-to-GREEN build configuration check**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target qsb_plascan_gui_camera_scene_shaders -j 8
```

Expected in the current local environment: configure/build may fail with the new fatal Vulkan feature check because the installed QtGui target currently has `vulkan` disabled. This is an expected environment failure until Qt is rebuilt with Vulkan support. If the environment has Vulkan-enabled Qt, expected result is shader `.qsb` generation success.

- [ ] **Step 6: Commit**

```powershell
git add cmake/PlascanPackages.cmake src/gui/CMakeLists.txt vcpkg.json src/gui/shaders
git commit -m "build: add vulkan qrhi shader pipeline inputs"
```

## Task 3: Replace Legacy Widget Declarations With QRhi Declarations

**Files:**
- Modify: `src/gui/dialogs/CameraModel3DDialog.h`
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`

- [ ] **Step 1: Replace includes and base class**

In `CameraModel3DDialog.h`, replace the legacy widget, buffer, shader program, and vertex-array includes.

with:

```cpp
#include <QRhiWidget>
#include <QScopedPointer>
```

Forward declare:

```cpp
class QRhiBuffer;
class QRhiCommandBuffer;
class QRhiGraphicsPipeline;
class QRhiRenderTarget;
class QRhiShaderResourceBindings;
class QShader;
```

Change the `CameraSceneWidget` base class from the legacy Qt rendering widget

to:

```cpp
class CameraSceneWidget : public QRhiWidget
```

- [ ] **Step 2: Replace lifecycle declarations**

Replace the legacy initialize, resize, and paint lifecycle declarations

with:

```cpp
void initialize(QRhiCommandBuffer *cb) override;
void render(QRhiCommandBuffer *cb) override;
void releaseResources() override;
void resizeEvent(QResizeEvent *event) override;
```

- [ ] **Step 3: Add QRhi resource members**

Replace legacy renderer members with:

```cpp
struct RhiBufferSet
{
    QScopedPointer<QRhiBuffer> vertexBuffer;
    QByteArray vertexData;
    int vertexCount = 0;
    int strideBytes = 0;
    bool dirty = true;
};

struct RhiPipelineSet
{
    QScopedPointer<QRhiBuffer> uniformBuffer;
    QScopedPointer<QRhiShaderResourceBindings> bindings;
    QScopedPointer<QRhiGraphicsPipeline> pipeline;
    QString vertexShaderPath;
    QString fragmentShaderPath;
};

struct SceneUniforms
{
    QMatrix4x4 mvp;
    QMatrix4x4 modelView;
    QMatrix4x4 normalMatrix;
    QVector4D lightDirPointSize;
};

QString _renderError;
bool _rhiReady = false;
bool _pipelinesDirty = true;
RhiBufferSet _pointBuffer;
RhiBufferSet _meshBuffer;
RhiBufferSet _modelPointBuffer;
RhiBufferSet _lineBuffer;
RhiPipelineSet _colorPointPipeline;
RhiPipelineSet _colorLinePipeline;
RhiPipelineSet _meshTrianglePipeline;
RhiPipelineSet _meshPointPipeline;
```

- [ ] **Step 4: Update base calls in event handlers**

Replace legacy widget base event calls with `QRhiWidget::mousePressEvent(event)` and similar calls.

- [ ] **Step 5: Run test**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CameraSceneWidgetTest" --output-on-failure
```

Expected: Some source-structure checks pass; build-related checks may still fail until implementation removes source legacy renderer tokens.

- [ ] **Step 6: Commit**

```powershell
git add src/gui/dialogs/CameraModel3DDialog.h src/gui/dialogs/CameraModel3DDialog.cpp
git commit -m "refactor: declare camera scene qrhi widget backend"
```

## Task 4: Implement QRhi Initialization, Shader Loading, And Buffer Upload

**Files:**
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`

- [ ] **Step 1: Add QRhi includes**

Add:

```cpp
#include <QFile>
#include <QRhiWidget>
#include <rhi/qrhi.h>
#include <rhi/qshader.h>
```

- [ ] **Step 2: Add shader loader helper**

Add an unnamed-namespace helper:

```cpp
QShader loadSceneShader(const QString &resourcePath, QString *errorMessage)
{
    QFile file(resourcePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Vulkan 渲染着色器资源缺失：%1").arg(resourcePath);
        }
        return {};
    }
    QShader shader = QShader::fromSerialized(file.readAll());
    if (!shader.isValid() && errorMessage)
    {
        *errorMessage = QStringLiteral("Vulkan 渲染着色器加载失败：%1").arg(resourcePath);
    }
    return shader;
}
```

- [ ] **Step 3: Implement constructor Vulkan API selection**

Change constructor initializer to:

```cpp
CameraSceneWidget::CameraSceneWidget(QWidget *parent)
    : QRhiWidget(parent)
{
    setApi(QRhiWidget::Api::Vulkan);
    setSampleCount(4);
    ...
}
```

Remove `QSurfaceFormat` setup.

- [ ] **Step 4: Implement `initialize()`**

Implement:

```cpp
void CameraSceneWidget::initialize(QRhiCommandBuffer *cb)
{
    Q_UNUSED(cb);
    _renderError.clear();
    _rhiReady = rhi() && api() == QRhiWidget::Api::Vulkan;
    if (!_rhiReady)
    {
        _renderError = QStringLiteral("Vulkan 渲染初始化失败，请检查显卡驱动和 Qt Vulkan 支持。");
        LOG_ERROR("%s", qPrintable(_renderError));
        return;
    }

    _colorPointPipeline.vertexShaderPath = QStringLiteral(":/shaders/camera_scene_color.vert.qsb");
    _colorPointPipeline.fragmentShaderPath = QStringLiteral(":/shaders/camera_scene_color.frag.qsb");
    _colorLinePipeline = _colorPointPipeline;
    _meshTrianglePipeline.vertexShaderPath = QStringLiteral(":/shaders/camera_scene_mesh.vert.qsb");
    _meshTrianglePipeline.fragmentShaderPath = QStringLiteral(":/shaders/camera_scene_mesh.frag.qsb");
    _meshPointPipeline = _meshTrianglePipeline;

    _gpuDirty = true;
    _pipelinesDirty = true;
}
```

- [ ] **Step 5: Implement `releaseResources()`**

Release all `QScopedPointer` members:

```cpp
void CameraSceneWidget::releaseResources()
{
    _pointBuffer.vertexBuffer.reset();
    _meshBuffer.vertexBuffer.reset();
    _modelPointBuffer.vertexBuffer.reset();
    _lineBuffer.vertexBuffer.reset();
    _colorPointPipeline.uniformBuffer.reset();
    _colorPointPipeline.bindings.reset();
    _colorPointPipeline.pipeline.reset();
    _colorLinePipeline.uniformBuffer.reset();
    _colorLinePipeline.bindings.reset();
    _colorLinePipeline.pipeline.reset();
    _meshTrianglePipeline.uniformBuffer.reset();
    _meshTrianglePipeline.bindings.reset();
    _meshTrianglePipeline.pipeline.reset();
    _meshPointPipeline.uniformBuffer.reset();
    _meshPointPipeline.bindings.reset();
    _meshPointPipeline.pipeline.reset();
    _rhiReady = false;
    _gpuDirty = true;
    _pipelinesDirty = true;
}
```

- [ ] **Step 6: Implement CPU data upload preparation**

Change `uploadGpuData()` so it no longer calls GL APIs. It should fill:

```cpp
_pointBuffer.vertexData = QByteArray(reinterpret_cast<const char *>(data.constData()),
                                     data.size() * int(sizeof(float)));
_pointBuffer.vertexCount = int(_cloud.size());
_pointBuffer.strideBytes = 6 * int(sizeof(float));
_pointBuffer.dirty = true;
```

Repeat for mesh/model/line buffers with existing CPU data generation logic.

- [ ] **Step 7: Commit**

```powershell
git add src/gui/dialogs/CameraModel3DDialog.cpp
git commit -m "feat: prepare camera scene qrhi resources"
```

## Task 5: Implement QRhi Pipelines And Draw Calls

**Files:**
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`

- [ ] **Step 1: Add buffer upload helper**

Add:

```cpp
bool CameraSceneWidget::ensureRhiBuffer(RhiBufferSet *buffer, QRhiResourceUpdateBatch *updates)
{
    if (!buffer || buffer->vertexData.isEmpty() || buffer->vertexCount <= 0)
    {
        return true;
    }
    if (!buffer->vertexBuffer || buffer->vertexBuffer->size() != quint32(buffer->vertexData.size()))
    {
        buffer->vertexBuffer.reset(rhi()->newBuffer(QRhiBuffer::Static,
                                                    QRhiBuffer::VertexBuffer,
                                                    quint32(buffer->vertexData.size())));
        if (!buffer->vertexBuffer->create())
        {
            _renderError = QStringLiteral("Vulkan 顶点缓冲创建失败。");
            return false;
        }
        buffer->dirty = true;
    }
    if (buffer->dirty)
    {
        updates->uploadStaticBuffer(buffer->vertexBuffer.data(),
                                    buffer->vertexData.constData());
        buffer->dirty = false;
    }
    return true;
}
```

- [ ] **Step 2: Add pipeline creation helper**

Add:

```cpp
bool CameraSceneWidget::ensurePipeline(RhiPipelineSet *pipeline,
                                       QRhiGraphicsPipeline::Topology topology,
                                       const QRhiVertexInputLayout &inputLayout)
{
    if (!pipeline || pipeline->pipeline)
    {
        return true;
    }
    QString error;
    const QShader vertexShader = loadSceneShader(pipeline->vertexShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragmentShader = loadSceneShader(pipeline->fragmentShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    pipeline->uniformBuffer.reset(rhi()->newBuffer(QRhiBuffer::Dynamic,
                                                   QRhiBuffer::UniformBuffer,
                                                   208));
    if (!pipeline->uniformBuffer->create())
    {
        _renderError = QStringLiteral("Vulkan uniform 缓冲创建失败。");
        return false;
    }

    pipeline->bindings.reset(rhi()->newShaderResourceBindings());
    pipeline->bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            pipeline->uniformBuffer.data())
    });
    pipeline->bindings->create();

    pipeline->pipeline.reset(rhi()->newGraphicsPipeline());
    pipeline->pipeline->setTopology(topology);
    pipeline->pipeline->setShaderStages({
        { QRhiShaderStage::Vertex, vertexShader },
        { QRhiShaderStage::Fragment, fragmentShader }
    });
    pipeline->pipeline->setVertexInputLayout(inputLayout);
    pipeline->pipeline->setShaderResourceBindings(pipeline->bindings.data());
    pipeline->pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    pipeline->pipeline->setSampleCount(sampleCount());
    pipeline->pipeline->setDepthTest(true);
    pipeline->pipeline->setDepthWrite(true);
    pipeline->pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    QRhiGraphicsPipeline::TargetBlend blend;
    blend.enable = true;
    blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
    blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    pipeline->pipeline->setTargetBlends({ blend });
    if (!pipeline->pipeline->create())
    {
        _renderError = QStringLiteral("Vulkan 图形管线创建失败。");
        return false;
    }
    return true;
}
```

- [ ] **Step 3: Implement `render()`**

Implement `render()` so it:

1. fills a white background and error text with `QPainter` if `_rhiReady` is false;
2. calls `uploadGpuData()` when `_gpuDirty` is true;
3. builds MVP/model-view matrices using the existing frame math;
4. creates resource updates and uploads dirty buffers;
5. begins pass with white clear color and default depth clear;
6. sets viewport to `QRhiViewport(0, 0, pixelSize.width(), pixelSize.height())`;
7. draws `_pointBuffer`, `_meshBuffer`, `_modelPointBuffer`, `_lineBuffer`;
8. ends pass;
9. calls `drawOverlay()` after pass.

Use command-buffer calls:

```cpp
cb->setGraphicsPipeline(pipeline.pipeline.data());
cb->setShaderResources(pipeline.bindings.data());
const QRhiCommandBuffer::VertexInput vertexInput(buffer.vertexBuffer.data(), 0);
cb->setVertexInput(0, 1, &vertexInput);
cb->draw(buffer.vertexCount);
```

- [ ] **Step 4: Run source tests**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CameraSceneWidgetTest|CodeStyleTest.CameraModel3DDialogUsesLowerCamelPrivateMemberNames" --output-on-failure
```

Expected: Source-structure tests pass if the test binary is current; if CMake fatal-check prevents running in current environment, report that local Qt must be rebuilt with Vulkan support.

- [ ] **Step 5: Commit**

```powershell
git add src/gui/dialogs/CameraModel3DDialog.cpp src/gui/dialogs/CameraModel3DDialog.h
git commit -m "feat: render camera scene through vulkan qrhi"
```

## Task 6: Update Documentation And Run Verification

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `docker/Dockerfile.ubuntu2404`
- Modify: `scripts/build_win/README.md`

- [ ] **Step 1: Update architecture docs**

Replace the legacy rendering note in `docs/PROJECT_ARCHITECTURE.md`:

```text
CanvasWidget.h/cpp              # 3D 渲染画布 (legacy backend)
```

with:

```text
CameraModel3DDialog.h/cpp       # 3D 场景视图 (Qt RHI/Vulkan)
```

- [ ] **Step 2: Update build docs**

In `docker/Dockerfile.ubuntu2404`, update the header comment and apt packages to include Vulkan support:

```dockerfile
# PlaScan 构建环境 — Ubuntu 24.04 + CUDA 12.5 + LibTorch + Qt6 + Vulkan
...
vulkan-tools libvulkan-dev glslang-tools
```

In `scripts/build_win/README.md`, add a short requirement:

```markdown
## Vulkan 渲染依赖

GUI 三维视图使用 Qt RHI 的 Vulkan 后端。Windows vcpkg 构建需要安装 Vulkan SDK，或确保 manifest 环境能提供 Vulkan loader 和 headers，并在 Qt6 `qtbase` 构建时启用 Vulkan public feature。若 CMake 报告 QtGui 未启用 Vulkan，需要清理并重建 vcpkg 的 Qt 包。
```

- [ ] **Step 3: Run configure/build verification**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target plascan_gui -j 8
```

Expected if environment is ready: build exits 0. Expected in the current checked environment before Qt rebuild: CMake/build fails with the explicit message requiring QtGui Vulkan support.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "CameraScene|CameraModel3D|GuiProject|MainWindow|WorkspaceCenter" --output-on-failure
```

Expected if environment is ready: relevant tests pass. If configure/build was blocked by missing Qt Vulkan support, report the block and do not claim tests passed.

- [ ] **Step 5: Commit and push**

```powershell
git status --short
git add docs/PROJECT_ARCHITECTURE.md docker/Dockerfile.ubuntu2404 scripts/build_win/README.md
git commit -m "docs: document vulkan rendering requirements"
git push origin main
```

## Self-Review

- Spec coverage: The plan covers QRhiWidget/Vulkan API selection, shader resources, legacy backend removal, build dependency checks, error handling, source tests, docs, and verification.
- Placeholder scan: No `TBD`, `TODO`, or open-ended placeholders are intentionally left.
- Type consistency: Resource names use `RhiBufferSet`, `RhiPipelineSet`, and `SceneUniforms` consistently across tasks.
- Known blocker: The current local QtGui target marks Vulkan as disabled. The plan makes this a configure-time failure and documents that Qt/vcpkg must be rebuilt with Vulkan support before full build verification can pass.
