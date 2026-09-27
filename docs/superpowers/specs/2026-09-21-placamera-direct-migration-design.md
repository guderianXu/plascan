# PlaCamera Direct Migration Design

**Date:** 2026-09-21
**Status:** Awaiting review
**Scope:** Replace PlaScan's legacy camera runtime with PlaCamera as the only camera model.

## Goal

Make `placamera` the authoritative camera model for PlaScan production code. Frame-pinhole, RPC00B, and planetary line-scan workflows shall consume PlaCamera definitions and instances directly. The migration must not introduce a legacy facade, backend switch, runtime fallback, or bidirectional legacy-model compatibility layer.

The existing `.plascan` project JSON field layout is preserved where practical, but it is decoded directly into PlaCamera state. Preserving an on-disk field name is not permission to retain the old C++ camera types.

## Architecture

PlaCamera owns model-independent camera identity, definition/instance ownership, projection contracts, optimization layouts, state validation, and the three built-in model families. PlaCoordinate owns shared frame/time types. PlaScan owns file-format readers, project orchestration, operation policies, reference observations, solver-specific packets, and GUI/CLI presentation.

The data flow becomes:

```text
Tsai / GDAL RPC / line-scan ISD / project JSON
                         |
                         v
PlaScan direct loaders and project store
                         |
                         v
placamera::CameraInstanceSet
                         |
                         v
SfM / BA / MVS / RPC terrain / line-scan BA / GUI
```

The current `src/core/placamera_adapter` directory is not a permanent compatibility layer. Its model conversion code will be removed or rewritten as direct PlaCamera loaders and project persistence code. No production function will accept a legacy camera class after its migration task completes.

## Ownership boundaries

### PlaCamera core

- `CameraDefinition`, `RasterModel`, and `CameraInstanceSet` identities.
- Frame-pinhole, RPC00B, and planetary line-scan model mathematics.
- Projection, imaging locus, inverse evaluation, stereo intersection, and model-local optimization.
- Canonical definition/instance state validation and serialization.
- `FrameId`, `TimeScale`, and `TimeReference` through PlaCoordinate.

### PlaScan camera/application layer

- Tsai, GDAL RPC, and planetary line-scan ISD readers.
- Existing project JSON field layout and atomic file transactions.
- `CameraOperationPlan` and workflow capability requirements.
- Camera reference observations, GNSS/IMU/POS resolution, uncertainty propagation, and solver priors.
- Camera baseline reports, intrinsic-prior heuristics, format conversion, and image undistortion.
- Qt, OpenCV, GPU, CLI, and GUI-specific camera packets.

These application-layer components must construct or consume PlaCamera objects directly. They must not construct legacy `FramePinholeInstance`, `RpcInstance`, `LineScanInstance`, or `FramePinholeNumericState` objects.

## Direct migration stages

### Stage 1: Canonical PlaCamera project runtime

Replace the old project runtime's model construction with a direct PlaCamera store. Decode `camera_definitions` and `camera_instances` into `placamera::CameraInstanceSet`, preserve stable image/instance/definition IDs, and write updates atomically. The store accepts a mixed-frame project graph; each homogeneous operation validates `requireCommonGroundFrame()` or applies an explicit coordinate-frame policy. Keep the on-disk project fields stable unless PlaCamera state requires an explicit schema version change.

The old `camera_project::CameraProjectRuntime` and `camera_models::CameraModelRegistry` must not remain on the production load path after this stage.

### Stage 2: Frame-pinhole production consumers

Migrate the frame-pinhole path first because it is shared by SfM, aerial triangulation, intersection, MVS preparation, quality checks, and GUI session services. Change public interfaces at module boundaries to consume PlaCamera frame models or explicitly named numerical packets derived from them. A numerical packet used by GPU/OpenCL/CUDA code is a solver representation, not a legacy compatibility type.

The migration includes project loading, external Tsai import, SfM/BA inputs, sparse intersection, MVS camera preparation, and frame-pinhole GUI services. Existing projection parity tests remain as regression tests only; no shadow runtime is added.

### Stage 3: RPC production consumers

Make `RpcRasterIO` construct `placamera::RpcModel` directly. Migrate RPC aerial triangulation, stereo intersection, DEM/DOM, and related mesh paths to use the PlaCamera RPC contract. Retain explicit ellipsoid and geodetic/Cartesian frame checks.

### Stage 4: Planetary line-scan production consumers

Make the ISD reader construct `placamera::LineScanModel` directly. Migrate planetary line-scan bundle adjustment, laser-range constraints, project persistence, and CLI output. Preserve timing, trajectory, detector geometry, and nonzero trajectory bias without passing through a legacy line-scan type.

### Stage 5: Legacy removal and optional CSM integration

After all production references are removed, delete legacy model classes, legacy model factories, legacy project-runtime model construction, and unused CMake targets. Add an optional `placamera_csm` target only if external CSM plugin/ISD interoperability is required; it must wrap PlaCamera directly and must not become a second internal camera model.

## Error and identity rules

- A missing or ambiguous `ImageId`, `CameraInstanceId`, `CameraDefinitionId`, or `FrameId` is a hard error before entering a solver.
- Model-family changes, frame changes, image-size changes, and definition/instance identity changes are rejected rather than silently repaired.
- Non-WGS84 RPC ellipsoids and line-scan trajectory bias remain valid PlaCamera state; if the current project schema cannot encode them, the project store returns a structured error and does not mutate the file.
- No production path silently falls back to a legacy camera implementation.
- External file parsing failures identify the source path, image identity, model family, and rejected field.

## Testing strategy

Each migration stage follows test-first development:

1. Add a test that calls the new PlaCamera-facing interface and fails because the old type is still required.
2. Change the smallest production boundary to accept PlaCamera state.
3. Run the focused unit and integration tests.
4. Run the affected end-to-end workflow with a real project or tracked dataset.
5. Remove the old include/target from that boundary and use compilation failures to enumerate remaining consumers.

Required gates before deleting the legacy camera tree:

- PlaCamera standalone Release, strict-warning, and sanitizer tests.
- PlaScan camera, project, coordinate, SfM, RPC, MVS, and line-scan tests.
- At least one real frame-pinhole project, one tracked RPC stereo project, and one real planetary line-scan dataset.
- Linux/GCC and Windows/MSVC builds for the migrated production targets.
- A repository-wide search showing no production include or symbol reference to the legacy camera model classes.

## Non-goals

- No CSM API compatibility is implied by this migration.
- No new Eigen, GeographicLib, CSPICE, or mandatory PROJ dependency is introduced into PlaCamera core.
- No unrelated refactor of solver, GUI, task runtime, or project resource code is allowed unless required to remove a legacy camera dependency.
- No automatic conversion of old runtime objects is retained after a module has migrated.

## Acceptance criteria

The migration is complete only when:

1. PlaCamera is the only camera model type visible at PlaScan production boundaries.
2. Project runtime, external loaders, SfM/BA, MVS, RPC terrain, line-scan BA, CLI, and GUI consume PlaCamera directly.
3. The old camera model, registry, and project-runtime construction paths are deleted or reduced to non-production historical migration tooling.
4. Existing project files can be read and atomically written through direct PlaCamera state, unless a documented schema migration is required.
5. All required platform and real-data gates pass.
