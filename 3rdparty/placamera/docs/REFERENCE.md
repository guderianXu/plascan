# Camera project import and external references

`placamera::importCameraProject(path, format)` reads Middlebury `*_par.txt`, EPFL
`.camera`, COLMAP text (`cameras.txt` and `images.txt`), an expanded Metashape
`doc.xml`, or a Metashape camera-reference TXT. Auto detection accepts a file
or directory. `importMetashapeDocument(xml)` accepts a document extracted from
`chunk.zip` by an application; PlaCamera itself does not link a ZIP library.

The result has two separate collections:

- `cameras[].calibration` contains the normalized pixel matrix,
  distortion, and central projection family;
- `cameras[]` also carries the camera-to-world pose, rolling-shutter
  acquisition state, and exact-representability diagnostic. A COLMAP frame
  retains its exact source lens model, including fisheye distortion, so
  callers can prepare an undistorted image before exporting a limited format.
- `references` contains raw WGS84 position, roll/pitch/yaw, time and optional
  uncertainty from Metashape TXT. The optional GNSS lever arm is raw too.
  These fields do not become a camera pose until an application supplies the
  orientation convention, lever-arm direction, coordinate context and stable
  image identity.

`makeCentralCameraGeometry(camera, definitionId, worldFrame)` converts a frame to
typed PlaCamera calibration and pose only when its intrinsic matrix,
projection family, and lens terms are represented exactly. Pixel skew is
preserved as `b2`; unknown or source-only distortion terms fail explicitly.
`makeDatasetFramePinhole` remains as a compatibility spelling.

The optional `placamera::reference` CMake component provides typed external
observations, coordinate resolution, pose priors, geometry validation and
comparison. It depends on `placoordinate::transform`; the base PlaCamera target
does not. `ResolvedCameraPosePrior` requires explicit transform provenance.
`ReferenceCameraGeometry` holds a validated PlaCamera frame model, while
`ReferenceCameraPosition` is position-only and cannot be used as a projector.

## Pose covariance contract

`PoseCovariance` always uses the tangent order
`[position X, position Y, position Z, rotation X, rotation Y, rotation Z]`.
Position entries are metres squared, rotation entries are radians squared, and
cross terms use the corresponding product units. Constructors accept a
six-value diagonal, upper-triangular row-major symmetric packing, or a full
row-major 6 by 6 matrix. The full form must be symmetric; all forms are checked
for finite values and the requested positive-semidefinite or positive-definite
policy. Position-only and rotation-only observations remain explicit instead
of receiving invented zero-variance components.

`propagatePoseCovarianceRigid` applies `diag(R, R)`. The Sim(3) variant applies
`diag(scale * R, R)`, so scale affects position and position/rotation cross
terms while rotation stays dimensionless.

Metashape camera-reference YPR input uses degrees and degree-squared covariance.
`adaptMetashapeCameraReference` evaluates `Rz(-yaw) * Rx(pitch) * Ry(roll)` and
maps its covariance through the analytic left-tangent Jacobian into radians
squared:

```cpp
placamera::reference::MetashapeCameraReference source;
source.positionMeters = placamera::Vector3{{1.0, 2.0, 3.0}};
source.yawPitchRollDegrees = placamera::Vector3{{10.0, 2.0, -3.0}};
source.yawPitchRollCovarianceDegreesSquared =
    std::array<double, 9>{{1.0, 0.1, 0.0,
                           0.1, 4.0, 0.2,
                           0.0, 0.2, 9.0}};

const auto observation = placamera::reference::adaptMetashapeCameraReference(
    placamera::ImageId("image"),
    placamera::reference::ReferenceSourceId("metashape"),
    placamera::FrameId("reference"),
    source);
```

See `examples/reference_covariance.cpp` for covariance propagation and error
handling in a complete program.

PlaCamera does not own project `image_uuid` matching or Qt sidecar JSON.
PlaScan keeps reference storage in `src/common/project/camera_reference`, the
reference UI in `src/gui/reference`, and direct camera-project matching and
binding in `src/gui/project/services/ProjectCameraProjectImport.cpp`.
PlaScan asks PlaCamera for the typed geometry, reports conversion failures, and
binds successful records to project image UUIDs. It no longer writes an
intermediate Tsai dataset.
