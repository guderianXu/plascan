# PlaCamera

PlaCamera is a dependency-light C++20 camera-model library for planetary and
photogrammetric applications. It provides a typed coordinate-frame boundary,
immutable calibration definitions, image-bound model instances, capability
discovery, and a uniform raster-model projection interface.

The project is designed to grow toward the integration role served by USGS
CSM without claiming CSM API or plugin compatibility. A real CSM bridge will
be an optional component that links csmapi; the core library stays independent.

## Current scope

Version 0.2 contains:

- the `CameraDefinition`, `RasterModel`, and identity-checked
  `CameraInstanceSet` contracts;
- explicit image coordinates (`sample`, `line`) and typed ground frames;
- one structured error contract for parsing, file IO, conversion, binding, and
  numerical evaluation, with source and optional line diagnostics;
- projection-neutral central-camera definitions and instances, with compatibility aliases for the original
  frame-pinhole names;
- Perspective, front-hemisphere Fisheye, full-sphere Equidistant/Equisolid Fisheye, Spherical, and Cylindrical
  projections with matched image-to-ray domain checks;
- lossless Metashape frame and line-scan calibration
  (`f/cx/cy/b1/b2/k1..k4/p1..p4`), exact image-centre/principal-offset
  retention, and legacy OpenCV Brown-Conrady radial/tangential distortion;
- rolling-shutter motion, keyframe/capture/master/layer acquisition topology, and sensor-to-master mount state;
- definition-only pixel undistortion for calibration-driven solvers, without requiring an image instance or pose;
- ground-to-image, image-to-locus, and image-to-ground-at-depth operations;
- frame-pinhole signed optical-axis depth and diagnostic signed projection; behind-camera projection leaves
  `positiveDepth` unset and never changes the strict `groundToImage` contract;
- physical camera-center baseline, triangulation angle, front-of-camera checks, and depth-to-baseline ratio;
- signed image axes and negative-depth-axis normalization;
- RPC00B projection/inversion, explicitly tagged normalized-image and physical
  ground-coordinate affine corrections, Cartesian imaging loci, configurable
  reference ellipsoids, and explicit geodetic helpers;
- planetary line-scan timing, piecewise line rates, pose interpolation,
  frame-composed trajectories, complete calibration plus detector-affine
  composition, trajectory bias, and solver-owned per-knot state with
  fixed/free controls, priors, smoothness, scene time offset,
  complete-calibration masks, and detector masks;
- model-neutral optimization layouts and immutable updates for frame, RPC, and
  line-scan instances;
- solver-owned frame numeric state with image-grid scaling and positive-depth
  normalization, frame pair triangulation, robust RPC
  control-point bias adjustment, and RPC stereo intersection;
- thread-safe two-stage definition/instance factories over versioned opaque
  model-state payloads;
- optional deterministic canonical JSON codecs and built-in factories in
  `placamera::state`, with strict field validation and schema migration;
- stream- and path-based ASP/Tsai camera read/write, Middlebury/EPFL/Metashape camera geometry parsing,
  exact promotion of imported geometry to typed central cameras, file-level camera-project import, raw Metashape
  GNSS/YPR reference import, COLMAP text camera/image parsing and normalized-ray projection for supported pinhole
  and fisheye models;
- optional `placamera::reference` typed observation, resolver and reference-geometry component, with canonical
  six-dimensional pose covariance, rigid/Sim(3) propagation, and a Metashape YPR-degree adapter;
- RPC-domain key/value decoding without a core GDAL dependency, optional RPC/RPB raster import in
  `placamera::gdal`, and optional USGSCSM LRO line-scan JSON and file import in `placamera::state`;
- CMake install/export support and an external consumer example.

CSM/PROJ/SPICE adapters remain planned optional components. PlaScan has
canonical project load/writeback, synthetic multi-sample parity, and
tracked-RPC-raster parity. Several production workflows now use PlaCamera
directly, while frame-camera SfM and some application-level camera callers are still being
migrated; see [docs/MIGRATION.md](docs/MIGRATION.md).

## Dependencies

The runtime library requires a C++20 standard library and the dependency-free
`placoordinate::types` target. PlaCamera aliases `FrameId`, `TimeScale`, and
`TimeReference` to PlaCoordinate's public types, so camera, control-point, SfM,
DEM, DOM, and LiDAR code share one coordinate identity. CMake 3.21 or newer is
required to build and install it. The optional `placamera::state` component
uses nlohmann/json privately and does not expose its DOM types. Unit tests use
GTest only when `PLACAMERA_BUILD_TESTS=ON`.
The optional `placamera::reference` component depends on `placoordinate::transform`.
The optional `placamera::gdal` component depends on GDAL and keeps its types out of public headers.

Optional future components and the reason they remain outside the core are
documented in [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## External camera formats

`<placamera/tsai.h>` reads and writes Tsai cameras through standard C++ streams or
`std::filesystem::path`.
Callers supply a definition ID and world frame; PlaCamera does not infer them
from the file path. `<placamera/colmap.h>` parses one `cameras.txt` row and one
`images.txt` header, converts COLMAP's `(0.5, 0.5)` first pixel center to the
raster index convention `(0, 0)`, and projects normalized rays through the
supported COLMAP distortion models. `colmapBrownConradyDistortion` returns an
exact mapping for pinhole, radial, OPENCV, and zero-denominator FULL_OPENCV
records; it returns `UnsupportedModel` for fisheye, thin-prism, or nonzero
rational denominator models instead of approximating them. The caller still locates
images and consumes the following `POINTS2D` row.

`<placamera/dataset_formats.h>` parses Middlebury par, EPFL/Strecha camera,
and Metashape chunk calibration and poses without opening files or locating
images. `CameraImportCompatibility` retains source-only terms and one
explicit unsupported reason. `makeCentralCameraGeometry` validates that the
imported matrix, projection family, and distortion are exactly representable
and returns typed calibration and pose; unsupported source-only lens terms fail
explicitly. `writeTsaiPixelCamera`
emits the versioned, pixel-unit Tsai interchange layout. Applications own paths,
warnings, output transactions, and image matching.

`<placamera/project_import.h>` reads project files and directories into flat
`cameras` records plus unresolved reference observations. COLMAP cameras retain the
exact source lens model; Metashape TXT YPR and GNSS offsets remain raw until
the caller supplies coordinate and orientation conventions. See
[docs/REFERENCE.md](docs/REFERENCE.md).

`<placamera/rpc_metadata.h>` validates RPC00B fields from a key/value map.
The optional `<placamera/rpc_raster.h>` API opens RPC/RPB rasters and constructs typed RPC
instances. `<placamera/isd.h>` decodes a USGSCSM LRO line-scan JSON document or opens an ISD
file and returns a typed model; it is implemented by
the optional `placamera::state` component, which keeps JSON out of the core
link target and public header.

Complete construction, project import, Tsai, RPC correction, line-scan solver,
reference covariance, RPC raster, and planetary ISD
examples are documented in [docs/USAGE.md](docs/USAGE.md).
The exact mapping and remaining lossless gaps relative to the frozen
`metalign::CameraModel` reference are listed in
[docs/METALIGN_CAMERA_MAPPING.md](docs/METALIGN_CAMERA_MAPPING.md).

## Build and test

```bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLACAMERA_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/placoordinate/install
cmake --build build/release
ctest --test-dir build/release --output-on-failure
```

Install and consume the exported target:

```bash
cmake --install build/release --prefix build/install
cmake -S examples/consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/consumer
```

Downstream CMake projects use:

```cmake
find_package(placamera CONFIG REQUIRED) # discovers placoordinate transitively
target_link_libraries(my_target PRIVATE placamera::placamera)
```

Consumers of canonical JSON state request the component explicitly:

```cmake
find_package(placamera CONFIG REQUIRED COMPONENTS state)
target_link_libraries(my_state_target PRIVATE placamera::state)
```

External reference resolution is also opt-in:

```cmake
find_package(placamera CONFIG REQUIRED COMPONENTS reference)
target_link_libraries(my_target PRIVATE placamera::reference)
```

RPC raster import is opt-in:

```cmake
find_package(placamera CONFIG REQUIRED COMPONENTS gdal)
target_link_libraries(my_target PRIVATE placamera::gdal)
```

Core, state, reference, and GDAL targets are installed in separate exports, so a core-only
consumer is not required to find their optional dependencies.

## Minimal example

```cpp
#include <placamera/placamera.h>

#include <optional>
#include <utility>

placamera::FrameCalibration calibration;
calibration.f = 1000.0;
calibration.cx = 500.0;
calibration.cy = 400.0;
calibration.k1 = 0.01;
calibration.principalPointDecomposition =
    placamera::PrincipalPointDecomposition{500.0, 400.0, 0.0, 0.0};

const auto definition = placamera::CentralCameraDefinition::create(
    placamera::CameraDefinitionId("calibration"), calibration,
    placamera::PixelConvention::PixelCenter, placamera::FrameId("body-fixed"),
    false, 1.0, 1, 1, placamera::FrameProjectionModel::Perspective);

placamera::CentralCameraGeometry geometry{
    definition,
    placamera::Pose::create(placamera::FrameId("body-fixed"), {0.0, 0.0, 0.0},
                            {{1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0}})};

placamera::CameraAcquisitionState acquisition;
acquisition.role = placamera::CameraRole::Keyframe;

const auto model = placamera::bindCentralCamera(
    std::move(geometry),
    {placamera::CameraInstanceId("image-model"),
     placamera::ImageId("image"),
     placamera::ImageSize{1000, 800},
     std::nullopt,
     acquisition});
if (!model)
{
    return 1;
}

const auto projection = model.value()->groundToImage(
    {placamera::FrameId("body-fixed"), {1.0, 2.0, 10.0}});
if (!projection)
{
    return 2;
}
```

`CentralCameraDefinition`, `CentralCameraGeometry`, `CentralCameraModel`, and
`bindCentralCamera` are the preferred names for all six central projection
families. The original `FramePinhole*` names and `bindFramePinhole` remain
source-compatible aliases or forwarding entry points.

| Projection | Forward domain | Calibration optimized by PlaCamera layout |
|---|---|---|
| `Perspective` | Nonzero focal-plane depth; normal projection requires positive depth | pose + full frame calibration |
| `Fisheye` | Front hemisphere | pose + full frame calibration |
| `EquidistantFisheye` | Full sphere, excluding the non-invertible antipode boundary | pose + full frame calibration |
| `EquisolidFisheye` | Full sphere, excluding the non-invertible radius-two boundary | pose + full frame calibration |
| `Spherical` | Longitude/latitude panorama | pose |
| `Cylindrical` | Longitude and unbounded cylindrical height | pose |

Rolling-shutter parameters are part of `CameraAcquisitionState`. `Regularized`
exposes image-plane translation, while `Full` exposes three-axis translation
and Rodrigues rotation. See [docs/USAGE.md](docs/USAGE.md) for construction,
import, persistence, optimization, RPC, and line-scan examples.

Direct definition and value construction rejects invalid calibration, image
size, pose, and frame bindings with `placamera::CameraValidationError`. Shared
frame and time values are validated by
`placoordinate::CoordinateValidationError`. Recoverable non-numerical APIs use
`Result<T>` or `Result<void>`; numerical model evaluation uses
`EvaluationResult<T>`. Both expose `ok()`, `errorCode()`, `message()`, and a
structured `CameraError` through `error()`.

`RasterModel` always consumes Cartesian `GroundCoordinate` values in the
model's declared frame. RPC users that start from longitude, latitude, and
ellipsoidal height can call `RpcModel::groundToImageGeodetic`; this keeps the
common imaging-locus contract physically meaningful and avoids hidden CRS
conversion.

## License

MIT. See [LICENSE](LICENSE).
