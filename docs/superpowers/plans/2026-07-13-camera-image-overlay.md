# Camera Image Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 3D model view camera thumbnails occlude by real view depth and make “显示图像” behave like Metashape's image overlay while preserving free model rotation and true foreground/background composition.

**Architecture:** Put deterministic camera-selection, view-depth sorting, projection, and world-plane math in a small header/source pair that can be unit tested without constructing a QRhi widget. Keep labels and ordinary camera thumbnails in the existing `QPainter` overlay, but render the active photograph as a world-space camera plane through a dedicated QRhi texture pipeline before or after the model draw calls.

**Tech Stack:** C++17, Qt 6 Widgets/Gui/Concurrent, QRhi/Vulkan, Qt Shader Tools (`qt_add_shaders`), GTest, CMake.

## Global Constraints

- Preserve all existing user changes; especially do not overwrite the dirty `MainWindow.cpp`, `MainMenu.*`, `tests/CMakeLists.txt`, or `tests/test_gui_project_utils.cpp` content.
- Do not stop the running `plascan.exe` without the user's permission.
- Do not modify SfM, MVS, camera calibration, model generation, submodules, model resources, or project data formats.
- Keep image decoding asynchronous and keep GUI-thread work bounded to cache/resource updates.
- Use C++17, four-space indentation, Allman braces, and `_lowerCamel` private member names.
- Do not create a Git commit unless the user explicitly requests one.
- Invalid intrinsics fall back to the existing 45-degree projection; missing images leave the model usable and produce a path-specific warning.

---

### Task 1: Deterministic camera-view math

**Files:**
- Create: `src/gui/dialogs/CameraSceneViewMath.h`
- Create: `src/gui/dialogs/CameraSceneViewMath.cpp`
- Create: `tests/test_camera_scene_view_math.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Qt `QMatrix3x3`, `QMatrix4x4`, `QVector`, `QVector3D`.
- Produces: `CameraViewCandidate`, `cameraForwardDirection(...)`, `currentWorldViewDirection(...)`, `selectCameraForView(...)`, `farToNearCameraIndices(...)`, and `calibratedProjection(...)` in namespace `xjw::gui::camera_scene`.

- [ ] **Step 1: Add failing tests for near/far ordering and direction selection**

```cpp
#include <gtest/gtest.h>
#include "CameraSceneViewMath.h"

using namespace xjw::gui::camera_scene;

TEST(CameraSceneViewMathTest, SortsFarCameraBeforeNearCameraRegardlessOfInputOrder)
{
    QMatrix4x4 worldToView;
    worldToView.setToIdentity();
    const QVector<QVector3D> centers{QVector3D(0, 0, -2), QVector3D(0, 0, -9)};
    EXPECT_EQ(farToNearCameraIndices(centers, worldToView), QVector<int>({1, 0}));
}

TEST(CameraSceneViewMathTest, ChoosesAvailableCameraWhoseForwardDirectionMatchesView)
{
    const QVector<CameraViewCandidate> candidates{
        {0, QVector3D(0, 0, -1), QVector3D(0, 0, 3), true},
        {1, QVector3D(1, 0, 0), QVector3D(3, 0, 0), true},
        {2, QVector3D(0, 0, -1), QVector3D(0, 0, 2), false},
    };
    EXPECT_EQ(selectCameraForView(candidates, QVector3D(0, 0, -1), QVector3D()), 0);
}
```

- [ ] **Step 2: Register the test target and verify RED**

Append a standalone target without changing existing target entries:

```cmake
add_executable(test_camera_scene_view_math
    test_camera_scene_view_math.cpp
    ${CMAKE_SOURCE_DIR}/src/gui/dialogs/CameraSceneViewMath.cpp
)
target_link_libraries(test_camera_scene_view_math PRIVATE Qt6::Core Qt6::Gui GTest::gtest_main)
target_include_directories(test_camera_scene_view_math PRIVATE ${CMAKE_SOURCE_DIR}/src/gui/dialogs)
gtest_discover_tests(test_camera_scene_view_math)
```

Run:

```powershell
cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON
cmake --build build/windows-vcpkg-cuda-release --target test_camera_scene_view_math -j 8
```

Expected: compilation fails because `CameraSceneViewMath.h` and its functions do not exist.

- [ ] **Step 3: Implement the minimal stable selection and depth sorting API**

```cpp
struct CameraViewCandidate
{
    int index = -1;
    QVector3D forward;
    QVector3D center;
    bool imageAvailable = false;
};

QVector<int> farToNearCameraIndices(const QVector<QVector3D> &centers,
                                    const QMatrix4x4 &worldToView);
int selectCameraForView(const QVector<CameraViewCandidate> &candidates,
                        const QVector3D &worldViewDirection,
                        const QVector3D &sceneCenter);
QVector3D cameraForwardDirection(const QMatrix3x3 &cameraToWorld,
                                 bool depthAxisFlipped);
QVector3D currentWorldViewDirection(const QQuaternion &viewRotation);
```

Sort with `std::stable_sort` on positive view distance `-(worldToView * QVector4D(center, 1)).z()`, descending. Select the maximum normalized forward/view dot product; break ties by how directly the camera points toward `sceneCenter`, then by lower original index.

- [ ] **Step 4: Verify GREEN for sorting and selection**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_camera_scene_view_math -j 8
build/windows-vcpkg-cuda-release/tests/test_camera_scene_view_math.exe `
  --gtest_filter="CameraSceneViewMathTest.SortsFarCameraBeforeNearCameraRegardlessOfInputOrder:CameraSceneViewMathTest.ChoosesAvailableCameraWhoseForwardDirectionMatchesView"
```

Expected: 2 tests pass.

- [ ] **Step 5: Add failing calibrated-projection tests**

```cpp
TEST(CameraSceneViewMathTest, CalibratedProjectionMapsOpticalAxisToPrincipalPoint)
{
    const QMatrix4x4 projection = calibratedProjection(
        1000.0f, 1000.0f, 600.0f, 400.0f, 1600, 1000, 0.1f, 100.0f, 1, 1);
    const QVector3D ndc = (projection * QVector4D(0, 0, -2, 1)).toVector3DAffine();
    EXPECT_NEAR((ndc.x() * 0.5f + 0.5f) * 1600.0f, 600.0f, 1e-3f);
    EXPECT_NEAR((1.0f - (ndc.y() * 0.5f + 0.5f)) * 1000.0f, 400.0f, 1e-3f);
}

TEST(CameraSceneViewMathTest, InvalidIntrinsicsUseFiniteFortyFiveDegreeFallback)
{
    const QMatrix4x4 projection = calibratedProjection(
        0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0.1f, 100.0f, 1, 1);
    const float *values = projection.constData();
    for (int i = 0; i < 16; ++i) EXPECT_TRUE(std::isfinite(values[i]));
}
```

- [ ] **Step 6: Run projection tests to verify RED**

Expected: compilation fails because `calibratedProjection(...)` is missing.

- [ ] **Step 7: Implement off-center projection with fallback**

Add:

```cpp
QMatrix4x4 calibratedProjection(float focalX,
                                float focalY,
                                float principalX,
                                float principalY,
                                int imageWidth,
                                int imageHeight,
                                float nearPlane,
                                float farPlane,
                                int uAxisSign,
                                int vAxisSign);
```

For valid positive dimensions/focal lengths, construct an off-center frustum from the four image edges and principal point. For invalid values, call `QMatrix4x4::perspective(45.0f, width / height, nearPlane, farPlane)` with a finite aspect fallback of `1.0f`.

- [ ] **Step 8: Run all math tests and inspect the diff**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_camera_scene_view_math -j 8
build/windows-vcpkg-cuda-release/tests/test_camera_scene_view_math.exe
git diff --check -- src/gui/dialogs/CameraSceneViewMath.* tests/test_camera_scene_view_math.cpp `
  src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt
```

Expected: all math tests pass and `git diff --check` reports no errors. Do not commit.

---

### Task 2: Image-overlay state and free-orbit camera selection

**Files:**
- Modify: `src/gui/dialogs/CameraModel3DDialog.h`
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`
- Modify: `tests/test_camera_scene_view_math.cpp`

**Interfaces:**
- Consumes: Task 1 math helpers.
- Produces: enriched `CameraPose`, `activeCameraImagePoseIndex()`, `updateActiveCameraForView()`, and shared free-orbit scene matrices used by model and image-plane rendering.

- [ ] **Step 1: Add failing state-policy tests to the pure helper fixture**

Add a small `CameraImageSelectionState` to the math module and test that `lock(index)` preserves the index across direction changes and `unlock()` re-enables selection. Add a render contract test ensuring image mode does not replace the free-orbit matrix.

```cpp
TEST(CameraSceneViewMathTest, LockKeepsActiveCameraUntilUnlocked)
{
    CameraImageSelectionState state;
    state.setActiveIndex(3);
    state.setLocked(true);
    EXPECT_EQ(state.resolveAutomaticIndex(7), 3);
    state.setLocked(false);
    EXPECT_EQ(state.resolveAutomaticIndex(7), 7);
}
```

- [ ] **Step 2: Verify RED, then implement the minimal state helper**

Run the single test and expect a missing-type failure. Implement only active index and locked selection policy, then rerun and expect PASS.

- [ ] **Step 3: Extend `CameraPose` with calibrated display data**

```cpp
float focalX = 0.0f;
float focalY = 0.0f;
float principalX = 0.0f;
float principalY = 0.0f;
int imageWidth = 0;
int imageHeight = 0;
int uAxisSign = 1;
int vAxisSign = 1;
bool depthAxisFlipped = false;
```

Populate these fields in `readCamerasFromMeta()` from `camera.intrinsics()` and `camera.depthAxisFlipped()`. Read image dimensions from the camera object's existing `image_width` and `image_height` keys first; when absent, retain zero until `loadCameraPlaneImage()` returns a decoded `QImage` size.

- [ ] **Step 4: Replace highlighted-first active-image selection with view matching**

Implement `updateActiveCameraForView()` by building Task 1 candidates from `_poses`. A valid candidate must have a non-empty image path. When `_cameraImageLocked` is true, retain the locked path/name. Otherwise select from the free-view direction; explicit photo highlighting is only an initial preference before the first direction match, not a permanent override.

- [ ] **Step 5: Preserve free view throughout image mode**

On the false→true transition of `setShowCameraImage`, select an active camera without changing `_viewRot`, `_zoomScale`, or `_sceneOffsetPx`. On true→false, clear only transient active-camera state. `setCameraImageLocked()` stores or clears the stable path/name and never disables orbit interaction.

- [ ] **Step 6: Centralize 3D matrices**

Create `sceneMatrices()` returning model-view and projection matrices used by both `render()` and `projectToScreen()`. It always preserves the current orbit math. The selected photo is converted to a world-space plane from its camera center, right/up axes, and image aspect ratio, then rendered with the same MVP as the model.

- [ ] **Step 7: Verify state and existing source contracts**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_camera_scene_view_math test_gui_project_utils -j 8
build/windows-vcpkg-cuda-release/tests/test_camera_scene_view_math.exe
build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter="CameraSceneWidgetTest.*"
```

Expected: math and existing camera-scene tests pass. If the running GUI locks `plascan_gui`, report it rather than terminating the process. Do not commit.

---

### Task 3: QRhi image compositing pipeline

**Files:**
- Create: `src/gui/shaders/camera_scene_image.vert`
- Create: `src/gui/shaders/camera_scene_image.frag`
- Modify: `src/gui/CMakeLists.txt`
- Modify: `src/gui/dialogs/CameraModel3DDialog.h`
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`
- Modify: `tests/test_gui_project_utils.cpp`

**Interfaces:**
- Consumes: Task 2 active camera and view/projection state plus the existing asynchronous `CameraPlaneImageResult`.
- Produces: QRhi texture/sampler/bindings/pipeline resources and `drawActiveCameraImage(QRhiCommandBuffer *)`.

- [ ] **Step 1: Add failing shader and draw-order contract tests**

```cpp
TEST(CameraSceneWidgetTest, RegistersQrhiCameraImageShaders)
{
    const QString cmake = readProjectSourceFile(QStringLiteral("src/gui/CMakeLists.txt"));
    EXPECT_TRUE(cmake.contains(QStringLiteral("camera_scene_image.vert")));
    EXPECT_TRUE(cmake.contains(QStringLiteral("camera_scene_image.frag")));
}

TEST(CameraSceneWidgetTest, DrawsBackgroundImageBeforeModelAndForegroundImageAfterModel)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    EXPECT_TRUE(source.contains(QStringLiteral("drawActiveCameraImage")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraImageDisplayLayer::Background")));
    EXPECT_TRUE(source.contains(QStringLiteral("CameraImageDisplayLayer::Foreground")));
}
```

- [ ] **Step 2: Run the two tests to verify RED**

Expected: both tests fail because the shader files/QRhi draw function are absent.

- [ ] **Step 3: Add the texture shaders and register them**

Vertex shader:

```glsl
#version 440
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;
layout(location = 0) out vec2 uv;
void main() { uv = texCoord; gl_Position = vec4(position, 0.0, 1.0); }
```

Fragment shader:

```glsl
#version 440
layout(binding = 0) uniform sampler2D imageTexture;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;
void main() { fragColor = texture(imageTexture, uv); }
```

Register both files in the existing `qt_add_shaders(plascan_gui camera_scene_shaders ...)` block.

- [ ] **Step 4: Add QRhi resources with explicit ownership**

Add scoped texture, sampler, vertex buffer, bindings, and image pipeline members. Release/reset them in `releaseResources()`. Recreate the texture only when the active decoded image size or cache key changes; upload with the frame's existing `QRhiResourceUpdateBatch`.

- [ ] **Step 5: Draw the photo in the correct pass order**

Within the existing single render pass:

```cpp
if (_cameraImageDisplayLayer == CameraImageDisplayLayer::Background)
{
    drawActiveCameraImage(cb);
}
drawSceneGeometry(cb, uniforms);
if (_cameraImageDisplayLayer == CameraImageDisplayLayer::Foreground)
{
    drawActiveCameraImage(cb);
}
```

The image pipeline uses no depth test/write. Texture coordinates preserve the image aspect ratio and principal-point mapping; uncovered viewport area retains the normal white clear color. Remove the old `drawSelectedCameraImage(QPainter, ...)` calls so the selected large image is not drawn twice.

- [ ] **Step 6: Make asynchronous failure visible but non-blocking**

Extend `CameraPlaneImageResult` with `QString errorMessage`. On decode failure set a message containing `imagePath`; on the GUI thread log it once per cache key, keep the model rendered, and do not draw an opaque placeholder.

- [ ] **Step 7: Build shaders and run contract tests**

Run:

```powershell
cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON
cmake --build build/windows-vcpkg-cuda-release --target plascan_gui test_gui_project_utils -j 8
build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe `
  --gtest_filter="CameraSceneWidgetTest.RegistersQrhiCameraImageShaders:CameraSceneWidgetTest.DrawsBackgroundImageBeforeModelAndForegroundImageAfterModel"
```

Expected: shader compilation succeeds and both tests pass. Do not commit.

---

### Task 4: Depth-correct ordinary thumbnails and regression verification

**Files:**
- Modify: `src/gui/dialogs/CameraModel3DDialog.cpp`
- Modify: `tests/test_gui_project_utils.cpp`
- Modify: `docs/superpowers/specs/2026-07-13-camera-image-overlay-design.md` only if implementation reveals a documented behavior mismatch

**Interfaces:**
- Consumes: Task 1 `farToNearCameraIndices(...)` and Task 2 centralized matrices.
- Produces: stable two-phase overlay painting: planes far-to-near, then labels/highlights.

- [ ] **Step 1: Add a failing source contract for depth-sorted drawing**

```cpp
TEST(CameraSceneWidgetTest, SortsCameraThumbnailPlanesByViewDepth)
{
    const QString source = readProjectSourceFile(QStringLiteral("src/gui/dialogs/CameraModel3DDialog.cpp"));
    EXPECT_TRUE(source.contains(QStringLiteral("farToNearCameraIndices")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawCameraPlane")));
    EXPECT_TRUE(source.contains(QStringLiteral("drawCameraLabel")));
}
```

- [ ] **Step 2: Verify RED**

Run the single test. Expected: failure because camera plane and label drawing are still one array-order loop.

- [ ] **Step 3: Split overlay painting into stable phases**

Compute camera centers and the current world-to-view/model-view matrix, call `farToNearCameraIndices(...)`, and draw only projected thumbnail polygons in that order. In a second loop draw labels, stems, and highlighted outlines. Keep label-budget logic and async thumbnail requests unchanged.

- [ ] **Step 4: Verify targeted tests**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_camera_scene_view_math test_gui_project_utils -j 8
build/windows-vcpkg-cuda-release/tests/test_camera_scene_view_math.exe
build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe `
  --gtest_filter="CameraSceneWidgetTest.*:MainMenuModelDisplayTest.*"
```

Expected: all targeted tests pass.

- [ ] **Step 5: Run proportional GUI regression coverage**

```powershell
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R "Gui|CameraScene|Workspace"
```

Expected: relevant tests pass. Record exact failures if pre-existing user changes make broader GUI tests fail.

- [ ] **Step 6: Perform manual visual verification when the GUI executable is available**

Use the user's existing project and reproduce the supplied camera alignment:

1. Enable cameras and thumbnails.
2. Align two photo planes on one screen ray and verify the nearer plane covers the farther plane.
3. Enable “显示图像” and rotate between several image directions; verify the active photo changes.
4. Verify foreground photo-over-model and background model-over-photo.
5. Lock/unlock the image and disable image display; verify free-view restoration.

If rebuilding requires replacing the running `plascan.exe`, ask the user to close it; do not terminate it automatically.

- [ ] **Step 7: Final workspace audit**

```powershell
git diff --check
git status --short
git diff -- src/gui/dialogs/CameraSceneViewMath.h src/gui/dialogs/CameraSceneViewMath.cpp `
  src/gui/dialogs/CameraModel3DDialog.h src/gui/dialogs/CameraModel3DDialog.cpp `
  src/gui/shaders/camera_scene_image.vert src/gui/shaders/camera_scene_image.frag `
  src/gui/CMakeLists.txt src/gui/cmake/GuiSources.cmake tests/test_camera_scene_view_math.cpp `
  tests/CMakeLists.txt tests/test_gui_project_utils.cpp
```

Expected: no whitespace errors; every diff hunk is either this task's change or a preserved pre-existing user change. Do not stage or commit.
