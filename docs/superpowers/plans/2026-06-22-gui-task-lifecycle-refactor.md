# GUI Task Lifecycle And Manager Split Plan

> For future agents: keep this as follow-up work after the immediate crash and blocking fixes are verified. Do not combine the whole refactor with unrelated algorithm changes.

## Goal

Make long-running GUI workflows safer and easier to maintain by centralizing background task lifecycle handling, then splitting the largest managers/dialogs by responsibility.

## Current Risk

- `ProjectTerrainProductsManager` previously inferred automatic DEM completion from `projectMetadataChanged` and `dense_cloud_results.last()`. The first mitigation now records the dense result count before MVS and consumes only the newly appended result.
- `ProjectTerrainProductsManager::startDemFromDenseCloudAsync()` previously ran DEM generation on the GUI thread. The first mitigation now runs the terrain work in a background task and returns to the GUI thread only for metadata, refresh, and dialogs.
- Several GUI managers had background lambdas and queued callbacks capturing raw `this`. The first mitigation adds `QPointer` guards to the terrain, dense reconstruction, and BA paths touched by this review.

## Follow-Up Tasks

### Task 1: Introduce A Small GUI Task Runner

- Add a reusable helper under `src/gui/project/tasks` or `src/gui/tasks`.
- Standardize:
  - owner `QPointer`;
  - cancel flag ownership;
  - progress emission;
  - success/failure completion;
  - queued GUI callback guard;
  - project path/session validation.
- Migrate one workflow first, preferably DEM/DOM generation, before touching MVS.

### Task 2: Split `ProjectDenseReconstructionManager`

- Extract depth-map estimation orchestration.
- Extract depth-map fusion orchestration.
- Extract dense-cloud post-processing orchestration.
- Keep `ProjectDenseReconstructionManager` as a thin facade that wires signals and delegates to focused runners.

### Task 3: Split `ProjectTerrainProductsManager`

- Extract DEM generation worker.
- Extract DOM/orthophoto generation worker.
- Extract automatic DEM pipeline state machine.
- Replace metadata-change polling with direct completion payloads where possible.

### Task 4: Split `CameraModel3DDialog`

- Extract PLY loading and preview sampling.
- Extract camera overlay rendering.
- Extract point cloud/model rendering state.
- Keep the dialog responsible for UI composition and settings only.

### Task 5: Split `LayerRenderer`

- Extract image loading and display-cache logic.
- Extract feature point overlay rendering.
- Extract match line overlay rendering.
- Keep `LayerRenderer` as a coordinator around `QGraphicsScene`.

## Verification

- Add source-structure tests for every migrated workflow to prevent raw `this` captures from returning.
- Build `plascan_gui`.
- Run the focused GUI utility tests around MVS, DEM metadata, image loading, and task status.
- Manually verify cancel and close-project behavior on at least one long-running DEM/MVS workflow.
