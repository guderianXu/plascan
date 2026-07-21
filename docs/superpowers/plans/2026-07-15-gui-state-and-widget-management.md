# GUI State And Widget Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate transient Dock-state persistence, stale project UI hydration, and context-insensitive commands while simplifying the main-window ownership and empty-state layout.

**Architecture:** `WorkspacePanelController` persists explicit show/close state rather than transient tab visibility. A small `ProjectUiHydrator` owns generation-based staged UI updates. `MainMenu` receives the active workspace and image readiness so menu actions and toolbar controls use one availability policy.

**Tech Stack:** C++17, Qt 6 Widgets, CMake/Ninja, GoogleTest.

---

### Task 1: Explicit panel visibility

**Files:**
- Modify: `src/gui/main_window/WorkspacePanelController.h`
- Modify: `src/gui/main_window/WorkspacePanelController.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] Add a GoogleTest that tabifies two `QDockWidget` instances and asserts selecting the other tab does not emit a false persisted visibility value.
- [ ] Run `test_gui_project_utils --gtest_filter=WorkspacePanelControllerTest.*` and confirm the new test fails because `visibilityChanged(false)` is persisted.
- [ ] Track `!widget->isHidden()` per registry entry and emit only when that explicit state changes.
- [ ] Re-run the filtered tests and confirm they pass.

### Task 2: Generation-safe project UI hydration

**Files:**
- Create: `src/gui/main_window/ProjectUiHydrator.h`
- Create: `src/gui/main_window/ProjectUiHydrator.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `src/gui/main_window/MainWindow.h`
- Modify: `src/gui/main_window/MainWindow.cpp`
- Modify: `tests/CMakeLists.txt`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] Add a test with two staged metadata requests and assert the second request cancels remaining stages from the first request.
- [ ] Run the new test and confirm it fails because `ProjectUiHydrator` does not exist.
- [ ] Implement a QObject-owned staged callback runner with a monotonically increasing generation and `cancel()`.
- [ ] Configure four MainWindow stages for dashboard/reference, data tree, workspace center, and photo strip.
- [ ] Delete the duplicate nested `scheduleProjectUiHydration()` chain and route project open and metadata changes through the hydrator.
- [ ] Re-run hydrator and MainWindow tests.

### Task 3: Context-aware image commands

**Files:**
- Modify: `src/gui/menu/MainMenu.h`
- Modify: `src/gui/menu/MainMenu.cpp`
- Modify: `src/gui/main_window/MainWindow.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] Add tests that image commands are disabled in model mode, remain disabled without a loaded image, and become enabled in image mode with a ready image.
- [ ] Run the tests and confirm the current menu keeps image actions enabled in model mode.
- [ ] Extend the contextual-toolbar API with image readiness and depth-overlay availability.
- [ ] Update MainWindow on view-mode, image-ready, and depth-availability signals.
- [ ] Re-run MainMenu and MainWindow tests.

### Task 4: Main-window and menu ownership

**Files:**
- Modify: `src/gui/main_window/MainWindow.h`
- Modify: `src/gui/main_window/MainWindow.ui`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] Add source-contract tests requiring MainWindow-owned widgets/managers to remain private and the Designer View menu to contain no runtime command ordering.
- [ ] Run the tests and confirm the current public block and duplicate `.ui` actions fail.
- [ ] Keep only `canvas()` as a public accessor and move all mutable members back under `private:`.
- [ ] Remove View-menu action ordering from the `.ui`; retain action declarations for binding and let `MainMenu` be the only ordering source.
- [ ] Re-run source-contract tests.

### Task 5: Empty-state layout

**Files:**
- Modify: `src/gui/widgets/ProjectDashboardWidget.cpp`
- Modify: `src/gui/main_window/MainWindow.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] Add tests asserting empty task/reference/quality/report tables are hidden and populated tables become visible.
- [ ] Run the tests and confirm empty tables remain visible.
- [ ] Apply row-count-based table visibility after each refresh.
- [ ] Reduce the default photo Dock allocation from 210 to 120 pixels while preserving its visible default.
- [ ] Re-run dashboard and layout tests.

### Task 6: Verification

**Files:**
- Modify if needed: `docs/PROJECT_ARCHITECTURE.md`

- [ ] Build `test_gui_project_utils` and `plascan_gui` with the Windows vcpkg CUDA release preset directory.
- [ ] Run the focused controller, menu, MainWindow, dashboard, and hydrator tests.
- [ ] Launch `build/windows-vcpkg-cuda-release/bin/plascan.exe` and visually inspect Model mode, View menus, Dock actions, and empty-state layout.
- [ ] Run `git diff --check` on the touched files and confirm there are no stale `WindowPanel` or duplicate hydration references.
