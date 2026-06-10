# Dialog UI Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for independent dialog groups. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `src/gui/dialogs` hand-written Qt layouts to Qt Designer `.ui` files while preserving current runtime behavior.

**Architecture:** Each dialog gets a matching `.ui` file for static layout and widget construction. Existing C++ keeps data binding, settings persistence, dynamic population, validation, signals, and custom rendering. Complex dialogs may keep dynamic content in C++ but should move their stable outer shell and containers to `.ui`.

**Tech Stack:** Qt6 Widgets, CMake AUTOUIC, C++17, xmllint/uic validation, existing GTest/ctest GUI tests.

---

## Version And Safety Rules

- Work on branch `gui-dialog-ui-migration-20260610`.
- Do not commit unless the user explicitly requests it.
- Do not revert or overwrite existing dirty files. Current known dirty GUI files include `src/gui/cmake/GuiSources.cmake`, `src/gui/dialogs/CameraModel3DDialog.*`, `src/gui/dialogs/SuperPointDialog.*`, `src/gui/main_window/MainWindow.*`, `src/gui/tasks/SuperPointRunner.cpp`, and `src/gui/widgets/WorkspaceCenterWidget.cpp`.
- Add `.ui` files under `src/gui/dialogs/`.
- Keep business logic in C++. `.ui` files define layout, object names, static labels, basic ranges, and static default widget properties.
- Prefer the existing member names as object names, for example `m_algorithmCombo`, `m_runBtn`, `m_imageList`.
- For combo boxes requiring `itemData`, set item data in C++ after `ui.setupUi(this)` if Qt Designer XML cannot represent the exact current values.

## Migration Pattern

- [ ] Add `dialogs/<DialogName>.ui`.
- [ ] Include the generated header in the dialog implementation: `#include "ui_<DialogName>.h"`.
- [ ] Replace manual layout creation with:

```cpp
Ui::<DialogName> form;
form.setupUi(this);
m_existingMember = form.m_existingMember;
```

- [ ] Keep signal connections in C++.
- [ ] Keep data population, validation, default model lookup, project image loading, and file dialog behavior in C++.
- [ ] Delete only obsolete manual widget construction from the target dialog after the `.ui` path is wired.
- [ ] Validate the `.ui` file with `xmllint --noout`.
- [ ] Validate uic generation with `uic <file>.ui -o /tmp/<file>_ui.h`.
- [ ] Build `plascan_gui` and run focused GUI tests.

## Dialog Groups

### Task 1: Simple Form Dialogs

**Files:**
- `src/gui/dialogs/TextureMappingDialog.*`
- `src/gui/dialogs/ModelExportDialog.*`
- `src/gui/dialogs/ModelGenerationDialog.*`
- `src/gui/dialogs/MeshReconstructionDialog.*`
- `src/gui/dialogs/DenseCloudRefineDialog.*`
- `src/gui/dialogs/DepthFusionDialog.*`
- `src/gui/dialogs/TriangulationDialog.*`
- `src/gui/dialogs/StereoProcessingDialog.*`

- [ ] Create `.ui` files with direct form/group layouts.
- [ ] Map existing member widgets from generated UI to existing member pointers.
- [ ] Keep run/accept/reject connections in C++.
- [ ] Run `xmllint` and `uic` on each new `.ui`.

### Task 2: Medium Parameter Dialogs

**Files:**
- `src/gui/dialogs/DenseMatchDialog.*`
- `src/gui/dialogs/SuperPointDialog.*`
- `src/gui/dialogs/FeatureMatchingDialog.*`
- `src/gui/dialogs/BundleAdjustDialog.*`
- `src/gui/dialogs/SparseCloudPostProcessDialog.*`
- `src/gui/dialogs/InitCameraPoseDialog.*`
- `src/gui/dialogs/CreateDemDialog.*`

- [ ] Reuse `DenseMatchDialog.ui` as the visual baseline.
- [ ] Keep dynamic algorithm pages, collapsible behavior, `settingsChanged`, and model path logic in C++ where needed.
- [ ] Preserve current defaults exactly.
- [ ] Run focused compile after each two dialogs to catch generated header issues early.

### Task 3: Complex And Custom-View Dialogs

**Files:**
- `src/gui/dialogs/CameraModel3DDialog.*`
- `src/gui/dialogs/ForwardIntersectionCheckDialog.*`
- `src/gui/dialogs/ForwardIntersectionResultsDialog.*`
- `src/gui/dialogs/MatchPairSelectorDialog.*`
- `src/gui/dialogs/MatchViewerDialog.*`
- `src/gui/dialogs/SuperPointVisualizationDialog.*`
- `src/gui/dialogs/ObservationNetworkDialog.*`
- `src/gui/dialogs/OverlapAnalysisDialog.*`
- `src/gui/dialogs/WorkflowReportDialog.*`
- `src/gui/dialogs/AerialTriangulationDialog.*`
- `src/gui/dialogs/DenseCloudDialog.*`
- `src/gui/dialogs/MVSProgressDialog.*`
- `src/gui/dialogs/MapProjectDialog.*`

- [ ] Move stable outer layout, button bars, tables, tabs, and container widgets to `.ui`.
- [ ] Keep custom scene widgets, image viewers, charts, report cards, and runtime-generated tables/cards in C++.
- [ ] For `CameraModel3DDialog.*`, preserve current dirty changes and inspect before editing.
- [ ] Run compile after each complex dialog.

## Verification

- [ ] `find src/gui/dialogs -name '*.ui' -print0 | xargs -0 -n1 xmllint --noout`
- [ ] `for f in src/gui/dialogs/*.ui; do /home/xjw/anaconda3/bin/uic "$f" -o "/tmp/$(basename "$f" .ui)_ui.h"; done`
- [ ] `cd build && cmake .. -DBUILD_TESTS=ON`
- [ ] `cd build && cmake --build . -j$(nproc) --target plascan_gui`
- [ ] `cd build && ctest --output-on-failure -R 'Gui|gui_project'`
- [ ] If broader tests are run and `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` fails with `dom_png not found`, report it as the known historical failure rather than claiming all tests pass.
