# PlaCamera usage guide

PlaCamera separates reusable camera geometry from application image identity:

- a definition owns immutable calibration;
- a geometry value combines a definition and pose;
- `bindCentralCamera` adds instance ID, image ID, raster size, optional time, and acquisition topology;
- `CameraModelPtr<T>` is the shared, immutable pointer type used by collections
  and import APIs.

The projection-neutral `CentralCamera*` names cover perspective, fisheye,
spherical, and cylindrical sensors. Existing `FramePinhole*` names remain
available for source compatibility.

## Error handling

Recoverable non-numerical operations return `Result<T>` or `Result<void>`.
This includes parsing, file IO, conversion, model binding, collection updates,
registry construction, persistence, and immutable model updates. Numerical
camera operations such as projection, inversion, and intersection return
`EvaluationResult<T>`. Both support the same basic inspection pattern:

```cpp
const auto result = placamera::loadTsaiFramePinhole(
    path, placamera::CameraDefinitionId("camera-0"), placamera::FrameId("world"));
if (!result)
{
    const placamera::CameraError& error = result.error();
    std::cerr << error.source;
    if (error.line)
    {
        std::cerr << ':' << *error.line;
    }
    std::cerr << ": " << error.message << '\n';
    return 1;
}
```

`CameraError` contains a stable `CameraErrorCode`, a message, an optional source
path or source name, and an optional one-based line number. Use `value()` for a
borrowed result value and `takeValue()` when transferring ownership.

Definition and value constructors still throw `CameraValidationError` for
programming errors such as invalid intrinsics or empty identifiers. Expected
runtime failures do not require an exception/output-parameter combination.

## Build a central camera

```cpp
#include <placamera/placamera.h>

#include <iostream>
#include <optional>
#include <utility>

placamera::FrameCalibration calibration;
calibration.f = 1000.0;
calibration.cx = 500.0;
calibration.cy = 400.0;
calibration.b1 = 2.0;
calibration.b2 = -0.25;
calibration.k1 = 0.01;
calibration.p1 = 0.0002;
calibration.principalPointDecomposition =
    placamera::PrincipalPointDecomposition{500.0, 400.0, 0.0, 0.0};

placamera::SensorMountState mount;
mount.fixedTranslation = true;
mount.fixedRotation = true;

const auto definition = placamera::CentralCameraDefinition::create(
    placamera::CameraDefinitionId("calibration"),
    calibration,
    placamera::PixelConvention::PixelCenter,
    placamera::FrameId("body-fixed"),
    false,
    1.0,
    1,
    1,
    placamera::FrameProjectionModel::Perspective,
    mount);

placamera::CentralCameraGeometry geometry{
    definition,
    placamera::Pose::create(
        placamera::FrameId("body-fixed"),
        {0.0, 0.0, 0.0},
        {{1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0}})};

placamera::CameraAcquisitionState acquisition;
acquisition.role = placamera::CameraRole::Keyframe;
acquisition.captureGroupId = placamera::CaptureGroupId("capture-0001");
acquisition.layerIndex = 0;

auto model = placamera::bindCentralCamera(
    std::move(geometry),
    {placamera::CameraInstanceId("image-model"),
     placamera::ImageId("image"),
     placamera::ImageSize{1000, 800},
     std::nullopt,
     acquisition});
if (!model)
{
    std::cerr << model.message() << '\n';
    return 1;
}

const auto projection = model.value()->groundToImage(
    {placamera::FrameId("body-fixed"), {1.0, 2.0, 10.0}});
if (!projection)
{
    std::cerr << projection.message() << '\n';
    return 1;
}
```

The returned `CameraModelPtr<CentralCameraModel>` can be copied cheaply and
stored in `CameraInstanceSet`. Its pointed-to model cannot be mutated.
Registry construction, state restoration, numeric-state promotion, RPC/ISD
import, and `withOptimizationUpdate` use the same immutable shared ownership;
callers do not need to wrap returned values in another smart pointer.

## Calibration and projection families

`FrameCalibration` is the lossless Metashape-compatible layout:

- sample uses `(f + b1) * x + b2 * y + cx`;
- line uses `f * y + cy`;
- radial distortion uses `k1` through `k4`;
- `p1` and `p2` are the Metashape tangential terms;
- `p3` and `p4` scale the complete tangential field by
  `1 + p3 * r^2 + p4 * r^4`.

The older `FrameIntrinsics` plus `BrownConradyDistortion` constructor keeps the
OpenCV five-coefficient convention. Use `FrameCalibration` for imported
Metashape values so the coefficient convention is explicit.

When a source stores `image_center` and `cx_offset/cy_offset` separately, put
those four values in `principalPointDecomposition` and keep `cx/cy` as the
active absolute principal point. Inverse projection uses the retained split
only when `image_center + offset` still equals the active value. Scaling and
offset optimization update the two terms separately, avoiding a lossy
`cx - offset` reconstruction.

| `FrameProjectionModel` | Mapping | Inverse domain |
|---|---|---|
| `Perspective` | `x/z, y/z` | nonzero focal-plane depth |
| `Fisheye` | equidistant angle, front hemisphere | radius `< pi/2` |
| `EquidistantFisheye` | `rho = theta`, full sphere | radius `< pi` |
| `EquisolidFisheye` | `rho = 2 sin(theta/2)`, full sphere | radius `< 2` |
| `Spherical` | longitude/latitude | longitude `[-pi, pi]`, latitude `[-pi/2, pi/2]` |
| `Cylindrical` | longitude and `y/horizontal` | longitude `[-pi, pi]` |

Brown distortion applies to Perspective and the three fisheye families.
Spherical and Cylindrical definitions retain their pixel calibration for
projection, while their default optimization layout refines pose only.

## Rolling shutter and capture topology

Projection and topology are independent. The definition carries a
`SensorMountState`; each image model carries a `CameraAcquisitionState`:

```cpp
placamera::CameraAcquisitionState acquisition;
acquisition.role = placamera::CameraRole::Keyframe;
acquisition.captureGroupId = placamera::CaptureGroupId("rig-shot-42");
acquisition.masterCameraId = placamera::CameraInstanceId("master-camera-42");
acquisition.layerIndex = 1;
acquisition.rollingShutterMode = placamera::RollingShutterMode::Full;
acquisition.rollingShutter.translation = {0.02, -0.01, 0.005};
acquisition.rollingShutter.rotationVector = {0.0004, -0.0002, 0.0001};
acquisition.rollingShutterInitialized = true;
```

The normalized scan coordinate is `2 * line / image_height - 1`.
Translation and Rodrigues rotation are scaled by that value. `Disabled` adds
no rolling parameters; `Regularized` exposes x/y translation; `Full` exposes
three-axis translation and rotation. `rollingShutterInitialized` records the
initialization lifecycle and does not freeze the parameters.

`masterCameraId` must differ from the current instance ID. Sensor mount and
acquisition validation fail closed for non-finite transforms or inconsistent
master identities.

## Import a camera project

`importCameraProject` recognizes Middlebury, EPFL/Strecha, COLMAP text,
Metashape XML, and Metashape reference text. Import does not invent application
identities or image sizes missing from the source.

```cpp
#include <placamera/formats.h>

const auto imported = placamera::importCameraProject(project_path);
if (!imported)
{
    std::cerr << imported.error().source << ": " << imported.message() << '\n';
    return 1;
}

for (std::size_t index = 0; index < imported->cameras.size(); ++index)
{
    const auto& camera = imported->cameras[index];
    if (!camera.compatibility.isExactlyRepresentable())
    {
        std::cerr << camera.imageName << ": "
                  << camera.compatibility.unsupportedReason.value_or(
                         "unsupported calibration terms")
                  << '\n';
        continue;
    }

    auto geometry = placamera::makeCentralCameraGeometry(
        camera,
        placamera::CameraDefinitionId("definition-" + std::to_string(index)),
        placamera::FrameId("project-world"));
    if (!geometry)
    {
        std::cerr << geometry.message() << '\n';
        return 1;
    }

    placamera::ImageSize image_size{fallback_width, fallback_height};
    if (camera.sourceColmapCamera)
    {
        image_size = {camera.sourceColmapCamera->width, camera.sourceColmapCamera->height};
    }

    auto model = placamera::bindCentralCamera(
        geometry.takeValue(),
        {placamera::CameraInstanceId("instance-" + std::to_string(index)),
         placamera::ImageId(camera.imageName),
         image_size,
         std::nullopt,
         camera.acquisition});
    if (!model)
    {
        std::cerr << model.message() << '\n';
        return 1;
    }
}
```

The import DTO has one source of truth for each concern:

- `camera.calibration` contains the projection family and normalized matrix
  and distortion inspection view;
- `camera.calibration.metashapeCalibration`, when present, is the authoritative
  lossless calibration. Binding verifies that the inspection view agrees with
  it before constructing a model;
- `camera.cameraToWorldRotation` and `camera.center` contain the pose;
- `camera.acquisition` contains rolling-shutter and per-image topology;
- `camera.compatibility` explains source terms that cannot be represented by
  the target central-camera model;
- `sourceColmapCamera` retains the exact COLMAP lens model and raster size;
- `sourceImageId` is the optional source-file ID and is not a PlaCamera
  `ImageId`;
- raw Metashape GNSS/YPR observations remain in `references` until the caller
  chooses coordinate and orientation conventions.

See `examples/project_import.cpp` for a complete executable that binds imported
frames and inserts them into a `CameraInstanceSet`.

## Select optimization parameters

The no-argument layout keeps the original 15-value perspective contract for
legacy perspective cameras. Use `FrameOptimizationSelection` when a solver
needs explicit Metashape calibration or rolling-shutter parameters:

```cpp
auto selection = placamera::FrameOptimizationSelection::none();
selection.set(placamera::FrameOptimizationParameter::PoseRotation)
    .set(placamera::FrameOptimizationParameter::PoseTranslation)
    .set(placamera::FrameOptimizationParameter::F)
    .set(placamera::FrameOptimizationParameter::B1)
    .set(placamera::FrameOptimizationParameter::B2)
    .set(placamera::FrameOptimizationParameter::K4)
    .set(placamera::FrameOptimizationParameter::P3)
    .set(placamera::FrameOptimizationParameter::P4)
    .set(placamera::FrameOptimizationParameter::RollingRotation)
    .set(placamera::FrameOptimizationParameter::RollingTranslation);

const auto layout = model.value()->optimizationLayout(selection);
placamera::OptimizationUpdate update{
    placamera::CameraInstanceId("refined-image-model"),
    placamera::CameraDefinitionId("refined-calibration"),
    std::vector<double>(layout.parameterCount(), 0.0)};
const auto refined = model.value()->withOptimizationUpdate(update, selection);
if (!refined)
{
    std::cerr << refined.message() << '\n';
    return 1;
}
```

The layout removes parameters that the selected projection or rolling-shutter
mode cannot use. A nonzero calibration delta requires a new definition ID;
instance-only pose and rolling updates can omit it.

## Optimize a line-scan trajectory

`LineScanModel::optimizationLayout()` keeps the original seven-value global
translation, rotation, and time-bias contract. A solver that needs individual
trajectory knots creates `LineScanNumericState` once and projects directly from
its owned arrays:

```cpp
#include <placamera/linescan_numeric_state.h>

placamera::LineScanOptimizationSelection selection;
selection.knotPositions = true;
selection.knotRotations = true;
selection.timeOffset = true;
selection.positionSecondDifferenceWeight = 4.0;
selection.rotationSecondDifferenceWeight = 9.0;
selection.detector.focalLength = true;
selection.calibration.f = true;
selection.calibration.cx = true;
selection.calibration.cy = true;
selection.calibration.k1 = true;
selection.calibration.k4 = true;
selection.calibration.p1 = true;
selection.calibration.p4 = true;

auto state_result = placamera::LineScanNumericState::fromModel(line_model, selection);
if (!state_result)
{
    std::cerr << state_result.message() << '\n';
    return 1;
}
auto state = state_result.takeValue();

std::vector<double> delta(state.optimizationLayout().parameterCount(), 0.0);
if (const auto applied = state.applyOptimizationDelta(delta); !applied)
{
    std::cerr << applied.message() << '\n';
    return 2;
}

std::vector<double> regularization(state.regularizationResidualCount());
if (const auto written = state.writeRegularizationResiduals(regularization); !written)
{
    std::cerr << written.message() << '\n';
    return 3;
}

const auto refined = state.toModel(
    placamera::CameraInstanceId("refined-line-instance"),
    state.definitionDirty()
        ? std::optional(placamera::CameraDefinitionId("refined-line-definition"))
        : std::nullopt);
```

Each direct `TrajectorySample` carries `TrajectoryKnotConstraints`. Fixed
position or rotation components are absent from the layout. A positive sigma
adds a normalized prior residual for that axis. Smoothness residuals are second
differences multiplied by the square root of the configured weight and include
fixed boundary knots. `LineScanTimeOffsetPrior` supplies the scene-level time
residual. Detector parameters are opt-in through
`LineScanDetectorOptimizationMask`. A `LineScanOptics::completeCalibration`
uses the same `f/cx/cy/b1/b2/k1..k4/p1..p4` equations as frame calibration;
each term is independently selectable through
`LineScanCalibrationOptimizationMask`. If a retained principal decomposition
exists, `cx/cy` deltas change its offsets and recompute the active absolute
principal. Any calibration or detector change requires a new definition ID
when promoting the state.

Frame-composed trajectories remain evaluable as immutable models, but they are
rejected by `LineScanNumericState::fromModel` because they do not expose direct
per-knot pose variables. Successful `projectAtLine`, `groundToImage`, and
`imageToImagingLocus` calls on the numeric state neither rebuild a model nor
resize storage. See `examples/linescan_solver.cpp` for a complete program.

## Persist and restore camera state

The optional `placamera::state` component serializes definitions and instances
separately and restores them through the registry:

```cpp
#include <placamera/state_codec.h>

placamera::ModelRegistry registry;
if (const auto registered = placamera::registerBuiltinJsonModelFactories(registry);
    !registered)
{
    std::cerr << registered.message() << '\n';
    return 1;
}

const auto definition_json =
    placamera::encodeCameraDefinitionJson(model.value()->pinholeDefinition());
const auto instance_json = placamera::encodeCameraInstanceJson(*model.value());
if (!definition_json || !instance_json)
{
    return 2;
}

const auto definition_state =
    placamera::decodeCameraDefinitionJson(definition_json.value());
const auto instance_state =
    placamera::decodeCameraInstanceJson(instance_json.value());
if (!definition_state || !instance_state)
{
    return 3;
}

const auto restored_definition = registry.createDefinition(definition_state.value());
if (!restored_definition)
{
    return 4;
}
const auto restored =
    registry.createInstance(instance_state.value(), restored_definition.value());
if (!restored)
{
    return 5;
}
```

Frame definition schema 4 stores the projection family, sensor mount, and
optional exact principal-point decomposition;
instance schema 2 stores acquisition topology and rolling motion. The decoder
migrates definition schemas 1 through 3 and instance schema 1 with explicit
defaults. RPC instance schema 2 stores its correction domain and only the
matching payload; schema 1 migrates to normalized-image correction. Line-scan
definition schema 3 stores optional complete calibration, while instance schema
2 stores every direct knot's fixed flags and optional axis sigmas plus the
optional scene time-offset prior. Older line-scan definitions migrate without
complete calibration; instance schema 1 migrates to fixed knots with no prior.
Unknown fields, incompatible model types, and invalid values fail instead of
being ignored.

## Tsai files

Tsai files contain calibration and pose but no PlaCamera image identity:

```cpp
auto geometry = placamera::loadTsaiFramePinhole(
    camera_path,
    placamera::CameraDefinitionId("tsai-definition"),
    placamera::FrameId("world"));
if (!geometry)
{
    std::cerr << geometry.message() << '\n';
    return 1;
}

auto model = placamera::bindCentralCamera(
    geometry.takeValue(),
    {placamera::CameraInstanceId("tsai-instance"),
     placamera::ImageId("image.tif"),
     placamera::ImageSize{width, height}});
```

`saveTsaiFramePinhole` writes geometry back with a `Result<void>` status. See
`examples/tsai_projector.cpp` for loading, binding, and projection.

## RPC correction domains

The normalized-image affine uses normalized uncorrected sample and line as its
independent variables. The physical ground affine uses longitude and latitude
differences from the RPC normalization centre in degrees and height difference
in metres. Select the domain explicitly:

```cpp
placamera::RpcGroundCorrection ground;
ground.sampleOffsetPixels = 0.25;
ground.lineOffsetPixels = -0.5;
ground.sampleLongitudePixelsPerDegree = 0.01;
ground.lineLatitudePixelsPerDegree = -0.02;

const auto model = placamera::RpcModel::createWithCorrection(
    placamera::CameraInstanceId("rpc-instance"),
    placamera::ImageId("rpc-image"),
    definition,
    placamera::ImageSize{2048, 2048},
    placamera::RpcCorrection::groundCoordinates(ground));

if (model.correctionDomain() != placamera::RpcCorrectionDomain::GroundCoordinates ||
    !model.groundCorrection())
{
    return 1;
}
```

`normalizedImageCorrection()` and `groundCorrection()` return null for the
other domain. The legacy `imageCorrection()` accessor throws on a ground-domain
model so a caller cannot silently reinterpret the eight physical coefficients
as the six normalized-image coefficients. Optimization layout and state JSON
also follow the selected domain. See `examples/rpc_correction.cpp` for a
complete executable.

## RPC raster import

The optional `placamera::gdal` component reads RPC00B metadata and raster size
and returns an identified immutable model:

```cpp
const auto model = placamera::importRpcRasterModel(
    raster_path,
    placamera::CameraDefinitionId("rpc-definition"),
    placamera::CameraInstanceId("rpc-instance"),
    placamera::ImageId("rpc-image"),
    placamera::FrameId("EPSG:4978"));
if (!model)
{
    std::cerr << model.error().source << ": " << model.message() << '\n';
    return 1;
}
```

Link `placamera::gdal`. See `examples/rpc_raster_import.cpp` for the complete
command-line program.

## Planetary line-scan ISD import

The optional `placamera::state` component imports a USGSCSM planetary
line-scanner ISD as an immutable model plus source metadata:

```cpp
const auto imported = placamera::importPlanetaryLineScanIsd(
    isd_path,
    placamera::CameraDefinitionId("isd-definition"),
    placamera::CameraInstanceId("isd-instance"),
    placamera::ImageId("isd-image"));
if (!imported)
{
    std::cerr << imported.error().source << ": " << imported.message() << '\n';
    return 1;
}

const placamera::CameraModelPtr<placamera::LineScanModel>& model = imported->instance;
```

An ISD may add `complete_calibration` (the alias
`metashape_calibration` is also accepted) with `f`, optional remaining
Metashape coefficients, `sample_pitch_mm`, and the three-part exact principal
description `image_center`, `cx_offset`, and `cy_offset`. All three principal
fields must appear together. With detector geometry, calibrated offsets from
`cx/cy` are converted through the sample pitch before the detector affine.
`LineScanPixelConvention::PixelCenter` applies the explicit half-pixel shift at
the public boundary; stored complete-calibration coordinates remain zero based.

Link `placamera::state`. See `examples/isd_import.cpp` for the complete
command-line program.

## Build the examples

```bash
cmake -S . -B build/examples \
  -DPLACAMERA_BUILD_EXAMPLES=ON \
  -DPLACAMERA_BUILD_STATE=ON \
  -DPLACAMERA_BUILD_REFERENCE=ON \
  -DPLACAMERA_BUILD_GDAL=ON \
  -DCMAKE_PREFIX_PATH=/path/to/dependencies
cmake --build build/examples
```
