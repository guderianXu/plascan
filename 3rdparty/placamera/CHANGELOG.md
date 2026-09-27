# Changelog

## Unreleased

### Added

- File and directory camera-project import for Middlebury, EPFL/Strecha,
  COLMAP text, expanded Metashape XML, and raw Metashape reference text.
- Exact COLMAP to Brown-Conrady mapping for pinhole, radial, OPENCV, and
  FULL_OPENCV records whose rational denominator is zero.
- Exact conversion from imported dataset geometry to typed central-camera
  calibration and pose, with explicit rejection of unsupported terms.
- Generic `Result<T>` and `Result<void>` statuses with structured error codes,
  source names, and optional line numbers for recoverable operations.
- `bindCentralCamera`, projection-neutral central-camera aliases, immutable
  camera pointer aliases, and complete project,
  Tsai, RPC raster, and planetary ISD examples.
- Lossless Metashape frame calibration with `f/cx/cy/b1/b2/k1..k4/p1..p4`.
- Full Metashape-compatible line-scan calibration with all 13 independently
  selectable solver parameters, iterative inverse projection, detector-affine
  composition, and explicit zero-based/pixel-centre conversion.
- Exact `image_center` plus `cx_offset/cy_offset` retention for frame and
  line-scan calibration, including import, scaling, optimization, and state
  round trips.
- Perspective, Fisheye, Equidistant Fisheye, Equisolid Fisheye, Spherical, and
  Cylindrical central projections with matched inverse-domain validation.
- Rolling-shutter motion, acquisition topology, sensor mount state, selectable
  optimization parameters, and versioned state persistence.
- Solver-owned line-scan trajectory state with fixed/free per-knot position and
  rotation blocks, anisotropic priors, second-order smoothness, scene time
  offset prior, detector masks, and allocation-free projection.
- Domain-tagged RPC correction for normalized image coordinates or physical
  longitude/latitude/height deltas.
- Canonical pose covariance validation and rigid/Sim(3) propagation, plus a
  Metashape YPR-degree covariance adapter.

### Changed

- Parsing, file IO, format conversion, reference resolution, and model binding
  now use the same result-inspection API instead of mixed booleans, optionals,
  output parameters, and error strings.
- Registry construction, state restoration, numeric-state promotion, imports,
  and optimization updates now return the canonical immutable shared model
  aliases without caller-side smart-pointer conversion.
- Project imports expose flat `cameras[]` records with one
  `CameraImportCompatibility` object for exact representability. COLMAP source
  camera and numeric source image ID remain distinct from PlaCamera identity.
- `makeCentralCameraGeometry` is the preferred import conversion API;
  perspective-oriented spellings remain forwarding compatibility entry points.
- Frame definition state advances to schema 4 and instance state to schema 2;
  older schemas migrate with explicit defaults. Line-scan definition state
  advances to schema 3 for complete calibration.
- RPC and line-scan instance state advance to schema 2. Their schema-1 records
  migrate to normalized-image RPC correction and fixed line-scan knots without
  a time-offset prior, respectively.

## 0.2.0 - 2026-09-21

### Added

- Generic camera-definition, image-bound instance-set, capability, and
  model-neutral optimization contracts.
- Solver-owned frame numeric state, typed promotion, frame triangulation, RPC
  robust bias adjustment, and RPC stereo intersection.
- Optional `placamera::state` component with deterministic canonical JSON,
  strict validation, two-stage built-in factories, and line-scan schema 1 to 2
  migration.
- Bidirectional PlaScan adapters, canonical project load/writeback, atomic
  identity validation, and multi-sample projection parity.
- Real-data regression against both tracked RPC stereo GeoTIFFs.
- Path-based Tsai and USGSCSM ISD file APIs, plus the optional
  `placamera::gdal` RPC/RPB raster component and its installed CMake export.

### Changed

- The installed core and state targets use separate component exports, keeping
  nlohmann/json out of core-only package discovery.
- PlaScan file callers now use PlaCamera directly; its former Tsai, ISD, and
  RPC raster wrappers and their parallel tests have been removed.

### Known limitations

- Some solver internals still use numerical working state, while reusable
  camera models and file formats are owned by PlaCamera.
- csmapi, PROJ, and CSPICE adapters are not implemented, so this release does
  not claim USGS CSM API/plugin compatibility.
- Legacy writeback rejects custom RPC ellipsoids and nonzero line-scan bias
  because the old PlaScan persistence model cannot represent them losslessly.

## 0.1.0 - 2026-09-20

### Added

- Standalone C++20 `placamera::placamera` CMake target and install package.
- Typed frames, camera/image identifiers, pose and time primitives.
- Shared `FrameId`, `TimeScale`, and `TimeReference` aliases backed by
  `placoordinate::types`, eliminating duplicate coordinate identity types.
- Uniform `RasterModel` evaluation contract and capability discovery.
- Frame-pinhole model with Brown-Conrady distortion and imaging loci.
- RPC00B model with geodetic/Cartesian conversion, configurable ellipsoid,
  inverse-at-height, imaging locus, and affine image correction.
- Planetary line-scan model with simple or frame-composed trajectories,
  piecewise timing, detector optics, projection, imaging locus, and bias state.
- Versioned opaque `ModelState` and thread-safe `ModelRegistry`, with codec and
  plugin dependencies kept outside the core ABI.
- Unit, public-header, dependency-boundary, and external-consumer tests.

### Known limitations

- Built-in model-state codecs and CSM, PROJ, SPICE, GDAL, and OpenCV adapters are not
  part of this release.
- PlaScan has a parity adapter, but production code still uses its existing
  camera module.
