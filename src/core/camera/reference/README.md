# camera_reference

`camera_reference_core` owns the typed boundary between project camera
instances and external reference observations. It depends only on
`camera_core`; it deliberately does not depend on a concrete projection model.

`camera_reference_geometry` is a separate concrete adapter target for the
frame-pinhole matching/aerial boundary. `geometry/ReferenceCameraGeometry`
contains two deliberately separate value types:

- `ReferenceCameraGeometry` is a projection-ready, numerically validated
  `FramePinholeNumericState` carrying `CameraInstanceId`, `ImageId`, and
  `CoordinateFrameId`.
- `ReferenceCameraPosition` is a position-only prior carrying `ImageId` and
  its frame. It cannot be passed to an epipolar projector.

The maps are keyed by `ImageId`. Image paths may be used by GUI/CLI adapters to
locate an input row, but they are not camera identity and are not stored in the
solver-facing types. `validateReferenceCameraInputs()` checks ordered image
identity alignment, map membership, duplicate IDs, and key/value agreement;
`commonReferenceWorldFrame()` rejects mixed frames before pair planning.

External GNSS/IMU/POS observations remain pose-only until the resolver creates a
`ResolvedCameraPosePrior`. A caller must join that prior with a canonical
projection model explicitly before constructing `ReferenceCameraGeometry`.
RPC and line-scan callers must provide their own model-specific adapter rather
than adding those models as dependencies of `camera_reference_core`.
