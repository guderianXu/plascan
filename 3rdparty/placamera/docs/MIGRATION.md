# PlaCamera migration status

PlaCamera 0.2 now owns PlaScan's reusable camera models and camera-file IO.
`src/core/camera` contains only an `INTERFACE` compatibility target; it has no
C++ implementation or wrapper API. Remaining migration work is limited to
solver-specific numerical state outside that module.

## Unreleased API consolidation

Recoverable APIs now return `Result<T>` or `Result<void>`. Callers inspect
`ok()`, `errorCode()`, `message()`, or `error()` and obtain values with
`value()` or `takeValue()`. This replaces public combinations of `bool`,
`std::optional`, output values, and `std::string*` error parameters. Numerical
evaluation keeps `EvaluationResult<T>` and exposes the same error inspection
surface.

Frame import now returns reusable `CentralCameraGeometry`. Bind it to an image
with `bindCentralCamera(geometry, binding)`, which returns the immutable
`CameraModelPtr<CentralCameraModel>` used by `CameraInstanceSet`. Direct
`FramePinholeModel::create` remains available for local value-oriented code.
Registry creation, state restoration, numeric-state promotion, format import,
and optimization updates now return `CameraModelPtr<T>` or `RasterModelPtr`
directly; remove manual `unique_ptr` to `shared_ptr` conversions.

`ImportedCamera` has one `compatibility` object. Replace
`additionalCalibrationTerms`, `hasUnmappedRadialDistortion`, and
`hasUnmappedCalibrationTerms` checks with
`compatibility.isExactlyRepresentable()` and inspect `sourceOnlyTerms` or
`unsupportedReason` when reporting a rejection. `ImportedProjectCamera` uses
`sourceColmapCamera` and `sourceImageId`; the latter is a source-file integer,
not a PlaCamera `ImageId`.

Project import records are now flat: iterate `result.cameras` and read
`camera.imageName`, `camera.calibration`, `camera.center`, `camera.acquisition`,
and `camera.compatibility` directly. Replace the unreleased
`result.frames[i].geometry` nesting with `result.cameras[i]`. The old
`Dataset*` type names remain aliases for low-level parser source compatibility.

Use `makeCentralCameraGeometry` for imported Perspective, Fisheye,
EquidistantFisheye, EquisolidFisheye, Spherical, and Cylindrical sensors.
`makeDatasetFramePinhole` and `bindFramePinhole` remain forwarding compatibility
entry points. Metashape import preserves the projection type, complete frame
calibration, exact image-centre/principal-offset decomposition, and
rolling-shutter mode. Its `metashapeCalibration` member is authoritative;
`intrinsicMatrix` and `distortion` are an inspection view and binding rejects a
conflict instead of choosing one silently.

See [USAGE.md](USAGE.md) and the `examples` directory for complete migrated
call patterns.

## Readiness matrix

| Area | Status | Evidence or boundary |
|---|---|---|
| Shared coordinate identity | Ready | `FrameId`, `TimeScale`, and `TimeReference` are PlaCoordinate public types. |
| Central frame model | Ready | Six projection families, complete Metashape calibration, rolling shutter, acquisition/sensor topology, projection, locus, signed optical-axis depth, solver state, selectable optimization updates, and triangulation. Typed and numeric states share the same projection/ray kernels. Inverse projection rejects pixels outside each model's invertible domain. |
| RPC00B model | Ready | Geodetic/Cartesian projection, inverse-at-height, locus, explicitly tagged normalized-image or physical ground-coordinate correction, domain-specific optimization layout, robust bias fit, and stereo intersection. Instance schema 1 migrates to the normalized-image schema 2 form; the two domains never convert implicitly. |
| Planetary line-scan model | Ready | Piecewise timing, simple/frame-composed trajectory, full Metashape `f/cx/cy/b1/b2/k1..k4/p1..p4` calibration, detector-affine composition, explicit half-pixel convention, projection, locus, and the compatible seven-parameter global bias. Direct-sample trajectories additionally expose solver-owned per-knot position/rotation blocks, all 13 calibration blocks, fixed flags, anisotropic priors, second-order smoothness, scene time offset with prior, detector masks, and one-time immutable writeback. Projection from that numeric state performs no model reconstruction or successful-call allocation. |
| Collections and dispatch | Ready | Definition/instance identities, capability gates, instance-set selection, and thread-safe factories. |
| Persistence | Ready | Deterministic definition/instance JSON, strict validation, frame definition schemas 1–3 migration to schema 4, line-scan definition schemas 1–2 migration to schema 3, and frame/RPC/line-scan instance schema 1 migration to schema 2. Frame schema 4 retains exact principal decomposition; line-scan schema 3 retains complete calibration. RPC schema 2 carries a correction-domain discriminator; line-scan instance schema 2 carries knot constraints and the time-offset prior. |
| PlaScan project storage | Ready | `plascan_placamera_runtime` loads canonical project records directly into `CameraInstanceSet` and writes state back atomically; record updates validate with the same PlaCamera loader instead of the old model registry. |
| External camera formats and references | Ready | PlaCamera reads and writes Tsai files, imports USGSCSM ISD files and RPC/RPB rasters, and imports Middlebury, EPFL, COLMAP text, expanded Metashape chunk XML, and raw Metashape GNSS/YPR TXT. Its optional `reference` component owns typed reference resolution and comparison. PlaScan's GUI project service consumes `importCameraProject()` and `makeCentralCameraGeometry()` directly, matches registered images, and binds project image UUIDs. PlaCamera rejects calibration that cannot map losslessly. The former intermediate Tsai dataset converter has been removed. Qt sidecar storage remains in PlaScan. |
| USGS CSM API/plugin | Not implemented | No csmapi target or official CSM conformance tests; do not claim API compatibility. |
| Production call sites | In progress | Project services, RPC aerial triangulation, RPC DEM/DOM, planetary line-scan BA, reference pose comparison and reference resection, and TSDF processing use PlaCamera. `DepthFrameArtifact`, `DepthTsdfFrame`, `VisualHullView`, and MVS `CameraView` own typed PlaCamera frame instances. Depth-manifest discovery and MVS replay construct those instances directly; TSDF loading, visual-hull reconstruction, model-image quality evaluation, MVS image preparation, and recovered scene encoding reuse them without promoting a legacy camera state. Mesh coloring and texture mapping receive PlaCamera frame numeric states and project through its typed API. SfM PnP accepts PlaCamera calibration values directly; BA track and marker identity checks use PlaCamera instances and canonical image IDs. MVS fusion input, depth-grid projection, CPU/CUDA/OpenCL dense-cloud unprojection, persisted depth-camera loading, recovered producer output, depth-frame result, image-preprocessor API, prepared-raster artifact, depth/prepared-camera manifest/snapshot publication, and the currently unused depth-pose refinement candidate stage use typed PlaCamera instances. The depth-pose stage rejects missing/mismatched raster geometry and cross-frame correspondences. Fusion reads the producer's prepared PNG directly and validates PlaCamera identity and raster dimensions. SfM reconstruction and its internal BA, outer frame-camera GUI, and parts of CLI still use the old numeric state at their separate boundaries. The old RPC and line-scan model targets and factories are removed. |

`CameraBaseline` accepts only PlaCamera frame models. MVS `CameraView`, source-view scoring, workspace replay, sparse-support projection, and stored-depth fusion now consume those models directly. Baseline geometry rejects mixed ground frames. The standalone PatchMatch CPU/CUDA/OpenCL backends and depth-pyramid request now consume PlaCamera frame models directly, with explicit raster and frame checks.
MVS production sources and tests no longer reference PlaScan's old frame-pinhole numeric state; rectification, workspace replay, projection and depth tests construct PlaCamera models directly. `CameraBaseline` and its tests now live in PlaCamera, and the MVS contract target links neither PlaScan's `camera` target nor its old numeric-state target.
The GUI MVS camera loader now retains canonical PlaCamera instances through view preparation and stored-depth fusion; it no longer converts them to the PlaScan numeric state and back. View preparation rejects image-header dimensions that disagree with the bound PlaCamera image size.
The one-click reconstruction CLI now also passes the canonical PlaCamera frame instances directly into MVS. It rejects unreadable rasters and image-header dimensions that disagree with the bound model instead of overwriting the camera grid through the old numeric state.
The SfM OpenCV camera adapter now accepts PlaCamera frame definitions and instances directly. Calibrated PnP passes its validated PlaCamera definition to that adapter; the SfM reconstruction camera store and BA solver state still need migration.
External pose-prior alignment now consumes only canonical PlaCamera image IDs and ground-frame IDs in camera order. The SfM and GUI BA callers validate their current solver cameras before passing these targets; this removes the old numeric-state type from the prior-alignment contract without introducing a camera-state adapter.
The project BA marker-control Sim(3) step transforms only canonical PlaCamera instances and tracks; a failed pose construction leaves the camera instances unchanged.
The project match reader and `BaInputBuilder` now retain only canonical PlaCamera frame instances and indexed observations. Automatic BA track initialization and sparse-preview filtering use PlaCamera geometry directly. The standalone BA service accepts those instances, converts directly into PlaBundle solver cameras, and promotes refined PlaCamera native numeric states back into typed models. Mixed ground frames and non-pixel-center frame calibration fail before solving. SfM's internal reconstruction and BA path still use the old PlaScan numeric state.
The GUI and standalone BA CLI no longer declare a direct dependency on the legacy `plabundle_adapter` target; only SfM's internal solver path still uses that adapter. It must be removed with the SfM state cutover, not retained as a compatibility facade.
The pinhole aerial-triangulation input now holds PlaCamera frame instances through `SfmAttemptRunner` and `PinholeEngine`; its previous hand-built legacy numeric state has been removed. The old numeric state is constructed only when feeding the still-unmigrated `IncrementalSfm` solver. PlaCamera frame instances can rebind their raster image size while preserving camera identity, pose, and capture time.
The standalone cross-view depth projection, measured-evidence projection, geometric consistency evaluation, and hole-growth normal calculation also use PlaCamera frame models; projection rejects mismatched ground frames and raster sizes.

## Verified compatibility gates

- Canonical project load, update, reload, and atomic rejection of stale identity.
- Project line-scan frame-composed trajectory, piecewise timing, capture time,
  detector optics, per-knot constraints, time-offset prior, and nonzero bias
  round-trip regressions.
- Per-knot line-scan numerical updates, prior/smoothness residuals, detector
  masks, immutable promotion, and a 1000-projection zero-allocation gate.
- Frozen differential fixtures for line-scan time offset, complete pushbroom
  calibration plus detector composition, exact principal decomposition, both
  RPC correction domains, and anisotropic Metashape YPR covariance.
- Tsai path read/write, ISD path import, RPC raster import, public-header
  boundaries, and optional component CMake exports.
- Both tracked RPC stereo rasters are imported through GDAL and compared at
  seven geodetic points per raster: direct geodetic parity uses `1e-10` pixel
  tolerance and the Cartesian common-model path uses `1e-7` pixel tolerance.
- Standalone build, strict-warning build, sanitizers, install, and external
  component consumers are release gates.

## Current storage constraints

- PlaScan's canonical RPC record describes WGS84 geodetic coordinates; the
  project loader currently supplies WGS84 ellipsoid parameters to PlaCamera.
  A different planetary reference ellipsoid requires an explicit project
  schema change before it can be persisted without loss.
- Project writeback rejects an image/instance/definition identity mismatch.
- Depth-frame manifests now require explicit `instance_id`, `image_id`, and
  `world_frame` for each camera. Discovery creates a typed PlaCamera model
  directly; TSDF loading rejects a missing model or image size that differs
  from the depth grid. Anonymous or mismatched artifacts must be regenerated.
- TSDF fusion rejects a set of typed frame cameras if their ground-frame IDs
  differ; no implicit cross-frame coordinate conversion is applied.
- Visual-hull field evaluation also rejects mixed ground frames. GPU packing
  accepts pixel-center calibration only; unsupported pixel conventions fail
  explicitly (or use the existing Auto CPU fallback).
- Texture mapping rejects depth/color cameras or source views with different
  ground-frame IDs. Mesh coloring skips views whose camera frame conflicts with
  the color camera or other active views; it never silently transforms points
  between ground frames.
- Unknown line-scan trajectory representations fail loading rather than being
  interpreted as direct pose samples.

## Recommended cutover order

1. Move remaining numerical consumers off the legacy frame and line-scan
   model types, without a dual-runtime compatibility layer.
2. Persist results through direct canonical project writeback and extend
   regression coverage for each migrated workflow.
3. Run Linux/GCC and Windows/MSVC production datasets, then remove each legacy
   model after its final consumer is gone.

The field-level `metalign::CameraModel` mapping and the small set of application
metadata that remains outside PlaCamera are documented in
[METALIGN_CAMERA_MAPPING.md](METALIGN_CAMERA_MAPPING.md).
