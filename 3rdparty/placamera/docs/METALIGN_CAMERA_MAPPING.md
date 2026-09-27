# Mapping from metalign camera state

This document records the field mapping used by PlaCamera's frozen differential
fixture. The reference is the separately maintained metalign camera source at
commit `e5c7b819f49114df3b93010cb17998593a6b53aa`. It describes compatibility
with that source contract; it is not a claim of compatibility with an official
Metashape binary or API.

## Direct mappings

| metalign state | PlaCamera state | Conversion |
|---|---|---|
| `CameraProjectionType::Frame` | `FrameProjectionModel::Perspective` | Direct discriminator mapping. |
| `Fisheye`, `EquidistantFisheye`, `EquisolidFisheye`, `Spherical`, `Cylindrical` | Matching `FrameProjectionModel` value | Direct discriminator mapping. |
| `f`, `cx`, `cy`, `b1`, `b2`, `k1..k4`, `p1..p4` | `FrameCalibration` | Same active pixel calibration and Metashape tangential convention. |
| `image_center`, `cx_offset`, `cy_offset` | `PrincipalPointDecomposition` | Retained independently beside active absolute `cx/cy`; inverse projection uses the split only while their exact sum matches the active principal. |
| `CameraRole`, capture group, master camera, layer, rolling motion | `CameraAcquisitionState` | Stable integer source IDs must first be converted to application-owned typed IDs. |
| sensor master, mount translation/rotation, fixed flags | `SensorMountState` | Direct value mapping after assigning a typed master sensor ID. |
| RPC normalization and four coefficient arrays | `RpcParameters` | Direct RPC00B value mapping. |
| `RpcAffineCorrection` | `RpcGroundCorrection` | Constant plus longitude, latitude, and height slopes; angular slopes are pixels/degree and height slopes are pixels/metre. |
| RPC camera-scoped ownership | separate `RpcModel` instance | Shared definitions remain possible, while the instance carries its tagged correction. |
| pushbroom trajectory sample time and centre | `TrajectorySample::time` and `center` | Direct after selecting the model's time scale and ground frame. |
| pushbroom world-to-camera rotation | `TrajectorySample::cameraToWorldRotation` | Matrix transpose. |
| pushbroom `fixed` | both knot fixed flags | `true` fixes position and rotation; `false` releases both. PlaCamera can represent the finer independent case too. |
| scalar position/rotation accuracy | three-axis knot sigma | Repeat position accuracy on XYZ; convert rotation accuracy from degrees to radians and repeat on XYZ. |
| current scene time offset | `LineScanTrajectoryBias::timeOffsetSeconds` | Direct. |
| zero-centred time-offset accuracy | `LineScanTimeOffsetPrior` | Mean `0`, sigma equal to the positive accuracy; zero accuracy means no free time block/prior. |
| pushbroom `f`, `cx`, `cy`, `b1`, `b2`, `k1..k4`, `p1..p4` | `LineScanOptics::completeCalibration` | Same normalized-plane Brown mapping as frame calibration; pixel coordinates are stored zero based. |
| pushbroom detector affine | `LineScanDetectorGeometry` | Calibrated offsets from `cx/cy` are converted to millimetres with `samplePitchMillimeters`, then passed through the detector affine and sample/line summing. |
| camera reference position and covariance | `MetashapeCameraReference` position fields | Metres and metres squared. |
| camera reference YPR and covariance | `MetashapeCameraReference` YPR fields | Input degrees/degree squared; adapter evaluates `Rz(-yaw) * Rx(pitch) * Ry(roll)` and produces canonical left-tangent radians/radians squared. |

For a registered central camera, metalign stores the projection pose as
world-to-camera `R, t` and may retain an independent centre `C`. PlaCamera stores
camera-to-world rotation and centre, so use `R.transpose()` and preserve `C`
when present. Otherwise compute `C = -R.transpose() * t` once at the adapter
boundary.

The metalign pushbroom `row` and `time_seconds` pairs map losslessly when their
row-to-time relation is represented as PlaCamera `LineTiming` segments. Solver
updates use `LineScanNumericState`; fixed knots are absent from its layout, and
priors and second-difference smoothness are emitted by
`writeRegularizationResiduals`.

## Fields outside a lossless `CameraModel` mapping

The following state still requires application metadata:

- `initial_focal_length` and `resection_threshold_pixels` are solver policy
  values. They are not projection state and have no PlaCamera model field.
- `RpcCameraModel::source` is provenance for a file or metadata record. Keep it
  in the importing application's metadata; the RPC definition stores the
  numerical model.
- `CameraCalibrationScope`, `SensorState::enabled`, and
  `SensorState::camera_count` describe project ownership and participation.
  Shared PlaCamera definitions express the geometry ownership relationship but
  do not serialize those exact project flags and counters.
`RegisteredCamera::focal_prior`, `supporting_points`, `reprojection_error`,
`component_id`, and `aligned` are solve or reconstruction state. They belong in
the caller's reconstruction record rather than a reusable camera model.

The fixture at `test/fixtures/metalign/camera_contract_v1.json` verifies the
shared numerical contract for line-scan time offset, full pushbroom calibration
and detector composition, exact principal decomposition, ground-domain RPC
correction, explicitly separate normalized-image RPC correction, and
anisotropic YPR covariance.
