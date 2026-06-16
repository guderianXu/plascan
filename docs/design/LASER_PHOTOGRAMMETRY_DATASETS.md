# Laser and Photogrammetry Datasets for Future BA Support

This note records candidate datasets for adding laser points / LiDAR point clouds into PlaScan workflows.
The immediate engineering purpose is to prepare future tests for:

- laser point constraints in Bundle Adjustment;
- camera / LiDAR extrinsic sanity checks;
- sparse cloud registration against external control points;
- dense point cloud fusion and filtering quality checks;
- MVS depth and point cloud validation against laser-scanned ground truth;
- DEM / DOM validation against LiDAR-derived terrain or surface products.

The dataset notes below are conservative by design. A "usable" entry means the dataset appears technically useful
for PlaScan experiments after a local license and access check. It does not mean the raw data, or derived fixtures,
may be redistributed from the PlaScan repository.

## Best First Targets

| Priority | Dataset | Sensors and geometry | Best PlaScan role | License / access notes | First PlaScan use |
| --- | --- | --- | --- | --- | --- |
| 1 | [MUN-FRL](https://mun-frl-vil-dataset.readthedocs.io/en/latest/) | Bell 412 helicopter and DJI M600 aerial VIL data; synchronized RGB/global-shutter cameras, downward LiDAR, IMU, GNSS/RTK/PPK, and published calibration. The March 2026 notes add Bell 412 intrinsic/extrinsic calibration data. | Best first target for BA laser constraints and camera / LiDAR extrinsic import because it has synchronized imagery, LiDAR, navigation ground truth, and explicit calibration values. | Dataset page states CC BY 4.0 for raw data and calibration files; publication license differs. Still record exact source URLs, access date, and checksums before deriving fixtures. | Create a tiny calibration + projected LiDAR control-point fixture. |
| 2 | [UseGeo](https://usegeo.fbk.eu/) / [GitHub](https://github.com/3DOM-FBK/UseGeo) | UAV nadir imagery, camera poses, interior parameters, ground-truth depth maps, LiDAR point clouds, and photogrammetric MVS point clouds. | Best first MVS / depth / terrain target because the project explicitly targets SfM, MVS, depth estimation, LiDAR classification, and hybrid LiDAR-photogrammetry adjustment. | Repository states CC BY-NC-SA 4.0 and research-purpose availability. Treat as non-commercial and avoid redistributing derived image/depth patches until reviewed. | Validate depth-map import, LiDAR-to-image projection, and DEM/DOM comparison reports. |
| 3 | [Hessigheim 3D / H3D](https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx) | UAV-based simultaneous LiDAR and imagery from a RIEGL Ricopter; high-density LiDAR around 800 pts/m2, oblique Sony imagery, textured meshes, hybrid adjustment, multiple epochs. | Strong photogrammetry-quality reference for dense cloud fusion, point-to-cloud QA, mesh QA, and later hybrid adjustment tests. | Download requires registration. Current site notes test-set ground truth availability, but terms must be checked before fixture extraction or redistribution. | Sample a tiny tile for point-cloud distance metrics and mesh/point label handling. |
| 4 | [MARS-LVIG](https://mars.hku.hk/dataset.html) | DJI M300 RTK aerial sequences with Livox Avia LiDAR, RGB camera, raw GNSS receiver, DJI RTK pose ground truth, and DJI L1 mapping ground truth. | Good second-wave BA / extrinsic / trajectory target for downward-looking aerial LIVO and long outdoor sequences. | Site states CC BY-NC-SA 4.0 for academic use only; downloads are large Google Drive rosbags. Not suitable for bundled fixtures without separate review. | Import camera-LiDAR-GNSS metadata and run local-only trajectory/extrinsic regression. |
| 5 | [UAVScenes](https://github.com/sijieaaa/UAVScenes) | UAVScenes extends MARS-LVIG with annotated camera images, annotated LiDAR point clouds, 6-DoF poses, camera-LiDAR calibrations, and reconstructed 3D maps. | Useful watchlist dataset for frame-wise LiDAR/image alignment, depth samples, and semantic point-cloud diagnostics once PlaScan has richer metadata import. | Repository states CC BY-NC-SA 4.0 and academic-use intent. It is newer and has open issues, so verify splits, checksums, and calibration files manually. | Use only as local research data until access and fixture policy is approved. |
| 6 | [NTU VIRAL](https://ntu-aris.github.io/ntu_viral_dataset/) | DJI Matrice UAV with two 3D LiDARs, time-synchronized cameras, IMUs, UWB, calibration resources, and ground truth included in bags / CSV. | Useful for camera / LiDAR extrinsic verification, pose-prior experiments, and residual diagnostics. Less direct for DEM/DOM products. | Dataset page states CC BY-NC-SA 4.0 for non-commercial academic use; some data repository pages may require browser/CAPTCHA access. | Validate transform-chain parsing, timestamp offsets, and multi-sensor residual reports. |
| 7 | [DublinCity aerial laser and photogrammetry survey](https://archive.nyu.edu/handle/2451/38684) | City-scale helicopter ALS, full-waveform LiDAR, ortho-rectified RGB/CIR rasters, and oblique imagery over Dublin. | Large urban reference for point-cloud IO, registration, DEM/DOM QA, and orthophoto/oblique-image experiments. Not a small BA/extrinsic starter. | NYU archive records state CC BY 4.0 for the collection and individual flight records; raw files are huge and checksummed. Do not mirror raw downloads. | Later large-cloud IO, tiled DEM/DOM comparison, and city-scale control-point extraction. |
| 8 | [ISPRS Vaihingen 3D Semantic Labeling](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/3d-semantic-labeling.aspx) and [DFC 2019 / US3D](https://www.grss-ieee.org/community/technical-committees/2019-ieee-grss-data-fusion-contest/) | Airborne LiDAR plus aerial/satellite-derived raster products. DFC 2019 includes multi-view satellite images, airborne LiDAR, DSM / disparity / height products, RPC metadata adjusted with LiDAR. | Useful later for DSM/DEM/semantic-height validation, not for UAV BA or camera-LiDAR extrinsic tests. | ISPRS benchmark terms must be checked. DFC 2019 data require registration/terms and forbid dissemination of data packages by registered users. | Later raster height and DSM/DOM QA only. |
| 9 | [ETH3D](https://www.eth3d.net/) and [Tanks and Temples](https://www.tanksandtemples.org/) | Non-aerial image-based reconstruction benchmarks with high-precision laser-scanned ground truth. ETH3D provides laser scan PLY / rendered depth products; Tanks and Temples provides video/image sets and training GT geometry. | Excellent generic MVS, dense cloud, and mesh guardrails, but not representative UAV/airborne BA inputs. | ETH3D page states CC BY-NC-SA 4.0. Tanks and Temples pages contain mixed CC BY wording and non-commercial / no-third-party-distribution terms; treat as restricted. | Keep out of first fixture; use as local-only MVS benchmark references. |
| 10 | [UAVStereo](https://github.com/rebecca0011/UAVStereo) | UAV stereo pairs and dense stereo/depth-oriented assets described with LiDAR-supported meshes. | Useful as a depth-map regression watchlist item, but less direct for BA and LiDAR control-point design. | Verify repository and dataset license before any use. | Optional stereo/MVS tuning after stronger dataset provenance handling exists. |

## Dataset-to-Feature Matrix

| Dataset | BA laser point constraints | Camera / LiDAR extrinsic validation | MVS depth validation | DEM / DOM validation | Notes |
| --- | --- | --- | --- | --- | --- |
| MUN-FRL | Primary | Primary | Secondary | Secondary | Best first calibration + BA fixture; image/LiDAR/navigation streams are synchronized. |
| UseGeo | Primary / secondary | Secondary | Primary | Primary | Explicitly provides camera poses, depth maps, LiDAR clouds, and MVS clouds; license is non-commercial share-alike. |
| H3D | Secondary | Secondary | Primary | Primary | Hybrid adjustment and paired UAV LiDAR/imagery make it strong for final product QA, but registration/download steps are heavier. |
| MARS-LVIG | Primary | Primary | Secondary | Secondary | Strong aerial LIVO candidate with mapping ground truth; large rosbags and NC license make it local-only for now. |
| UAVScenes | Secondary | Primary | Secondary | Secondary | Adds frame-wise labels, poses, calibrations, and 3D maps on top of MARS-LVIG; useful after import schema matures. |
| NTU VIRAL | Primary | Primary | Secondary | No | UAV robotics focus; excellent for transform and residual diagnostics, weak for terrain products. |
| DublinCity | No | No / weak | Secondary | Primary | Great for geospatial products and large-cloud IO; does not provide a compact synchronized BA setup. |
| ISPRS Vaihingen 3D | No | No | No / weak | Secondary | ALS point labels and urban reconstruction reference; useful for height/semantic QA only. |
| DFC 2019 / US3D | No | No | Secondary | Primary | Satellite/RPC benchmark with LiDAR-derived DSM/disparity/height products; access terms restrict redistribution. |
| ETH3D | No | No | Primary | No | Mature laser GT for MVS correctness, but not aerial. |
| Tanks and Temples | No | No | Primary | No | Mature reconstruction benchmark; license/access terms should be treated as restricted. |
| UAVStereo | No | No / weak | Secondary | No | Depth/stereo watchlist item pending license verification. |

## Source Verification Log

| Dataset | Main pages checked | Key facts to preserve in fixture metadata |
| --- | --- | --- |
| MUN-FRL | `https://mun-frl-vil-dataset.readthedocs.io/en/latest/`, `.../Calibration.html` | Access date, dataset sequence name, Google Drive file URL, original bag/calibration checksum, CC BY 4.0 statement URL, camera intrinsics, `camDown_T_lidar`, `body_T_cam*`, GNSS/PPK source. |
| UseGeo | `https://usegeo.fbk.eu/`, `https://usegeo.fbk.eu/data`, `https://github.com/3DOM-FBK/UseGeo` | Access date, strip id, image/depth/pose files, LiDAR point cloud URL, CC BY-NC-SA 4.0 statement URL, camera pose convention, downsample scale. |
| H3D | `https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx`, `/details.aspx`, `/participate.aspx`, `/subscribe.aspx` | Registration requirement, epoch, tile id, LAS/ASCII source, hybrid adjustment / georeferencing note, download email or source record, exact terms accepted. |
| MARS-LVIG | `https://mars.hku.hk/dataset.html` | Sequence id, rosbag URL, DJI L1 mapping GT source, RTK pose source, CC BY-NC-SA 4.0 academic-use statement, sensor setup and altitude. |
| UAVScenes | `https://github.com/sijieaaa/UAVScenes`, paper/project page links | Data interval, scene/run id, `calibration_results.py`, `sampleinfos_interpolated.json`, 6-DoF pose file, CC BY-NC-SA 4.0 statement. |
| NTU VIRAL | `https://ntu-aris.github.io/ntu_viral_dataset/`, sensors/calibration pages | Sequence id, bag/CSV GT source, prism/IMU offset note, calibration bag, non-commercial CC BY-NC-SA 4.0 statement. |
| DublinCity | `https://archive.nyu.edu/handle/2451/38684`, individual flight records such as `2451/38659` | DOI/handle, flight id, archive manifest checksum, CC BY 4.0 rights field, tile/flight filenames, CRS and raster/point-cloud product type. |
| DFC 2019 / US3D | GRSS contest page and DFC GitHub baseline repository | Registration/terms requirement, data owners, no-dissemination rule, tile id, RPC/DSM/disparity/AGL product type. |

## First Small Fixture Plan

Use MUN-FRL as the first BA/extrinsic fixture because it has explicit calibration, aerial imagery, LiDAR, and navigation
ground truth. Use UseGeo as the first depth/DEM companion only after the provenance format is working.

Do not commit raw rosbags, original full-resolution images, original LAS/LAZ files, original depth rasters, or complete
calibration download archives. Commit only tiny derived files that are sufficient to exercise import, math, and report
formatting.

Suggested layout:

```text
testData/laser_photogrammetry/
  mun_frl_lighthouse_tiny/
    README.md
    source_manifest.json
    cameras.json
    lidar_frames.json
    laser_control_points.ply
    laser_observations.csv
    depth_samples.csv
    expected_residual_summary.json
    scripts/
      README.md
```

Derived files to keep small:

- `source_manifest.json`: source dataset name, source page URL, download URL, access date, license name and license URL,
  citation, original file name, original SHA-256 / MD5 when provided or computed locally, extraction command, derived
  file checksums, coordinate frame description, and a `redistribution_review` field.
- `cameras.json`: 2-4 image records with image id, timestamp, width/height, camera model, intrinsics, distortion, pose
  in a local ENU or dataset-local frame, and pose covariance if available. Use generated blank/thumbnail images only if
  the fixture needs image dimensions; do not commit real full-resolution frames unless license review approves it.
- `lidar_frames.json`: 1-2 LiDAR frame records with frame id, timestamp, `T_world_lidar`, `T_body_lidar`, frame covariance,
  point count before/after sampling, source cloud checksum, and fields present in the source cloud.
- `laser_control_points.ply`: 20-100 downsampled or manually selected LiDAR points in local coordinates, with optional
  intensity/color/class/source index and per-point sigma. The point count should be low enough that it cannot substitute
  for the original dataset.
- `laser_observations.csv`: projected or manually checked correspondences from control points to camera pixels, with
  measurement sigma, confidence, occlusion flag, and derivation method.
- `depth_samples.csv`: optional per-image sparse depth samples from LiDAR projection, e.g. no more than 500 rows total
  with image id, pixel, depth in meters, LiDAR frame id, and validity flag.
- `expected_residual_summary.json`: expected import counts and coarse residual ranges, not brittle exact optimizer output.
- `scripts/README.md`: document the local extraction command and required tools, but keep download/extraction scripts
  inert unless the user explicitly supplies raw data paths. Prefer scripts that read raw paths from CLI arguments and
  never fetch data automatically.

Fixture policy:

- Every derived file must have a checksum in `source_manifest.json`.
- Every source dataset must have a source URL, license URL or terms URL, access date, and citation.
- Every fixture must state whether it is derived from CC BY, non-commercial, registration-only, or terms-restricted data.
- Do not include original large data, credentials, registration emails, Google Drive cookies, or private access links.
- If license review is not complete, store the fixture recipe only and keep the derived data out of the repository.

## Future Data Structure Field Checklist

### `LaserControlPoint`

- Identity: `control_point_id`, `source_dataset`, `source_cloud_id`, `source_point_index`, `group_id`.
- Position: `xyz_world`, `world_frame_id`, `crs_epsg`, `local_origin`, `epoch_time`, `units`.
- Geometry attributes: `normal`, `surface_type`, `return_number`, `number_of_returns`, `scan_angle`, `range`.
- Appearance attributes: `intensity`, `reflectance`, `color_rgb`, `semantic_class`, `classification_confidence`.
- Uncertainty: `covariance_3x3`, `sigma_xyz`, `information_weight`, `quality_score`, `active`.
- Provenance: `creation_method`, `sampling_voxel_size`, `source_url`, `license_id`, `source_checksum`,
  `derived_checksum`, `notes`.

### `LaserObservation`

- Identity: `observation_id`, `control_point_id`, `image_id`, `camera_id`, `lidar_frame_id`, `track_id`.
- Measurement: `pixel_xy`, `depth_m`, `range_m`, `bearing`, `point_to_plane_normal`, `measurement_type`.
- Timing and transforms: `timestamp`, `time_offset_s`, `T_camera_lidar_id`, `T_world_lidar_id`, `transform_chain_version`.
- Uncertainty: `covariance_2x2`, `sigma_px`, `sigma_depth_m`, `sigma_range_m`, `weight`, `robust_loss`.
- Visibility and quality: `visibility_state`, `occlusion_flag`, `incidence_angle_deg`, `projection_method`,
  `correspondence_method`, `confidence`, `outlier_state`.
- Diagnostics hooks: `pre_residual`, `post_residual`, `normalized_residual`, `jacobian_block_id`, `enabled`.

### `LiDARFrame`

- Identity: `lidar_frame_id`, `sensor_id`, `sequence_id`, `source_cloud_id`, `scan_index`.
- Timing: `timestamp_start`, `timestamp_end`, `time_reference`, `motion_compensated`, `time_sync_quality`.
- Poses and calibration: `T_world_lidar`, `T_body_lidar`, `T_lidar_imu`, `pose_covariance`, `calibration_version`.
- Sensor model: `lidar_model`, `beam_count`, `horizontal_fov`, `vertical_fov`, `range_min_max`, `scan_rate_hz`.
- Cloud metadata: `point_count_raw`, `point_count_used`, `fields`, `bbox`, `voxel_size`, `format`, `path`, `checksum`.
- Georeferencing: `world_frame_id`, `crs_epsg`, `local_origin`, `height_datum`, `units`.
- Provenance and policy: `source_url`, `license_id`, `access_date`, `redistribution_review`, `notes`.

### Residual Diagnostics

- Scope keys: `residual_id`, `iteration`, `dataset_id`, `camera_id`, `lidar_frame_id`, `control_point_id`,
  `observation_id`, `residual_type`.
- Raw residuals: `reprojection_error_px`, `laser_point_error_m`, `depth_error_m`, `point_to_plane_error_m`,
  `pre_optimization`, `post_optimization`.
- Normalized metrics: `normalized_residual`, `chi_square`, `rmse`, `median`, `p95`, `max`, `inlier_threshold`.
- Weights and robust loss: `measurement_weight`, `robust_weight`, `covariance_used`, `information_matrix_id`.
- State deltas: `camera_pose_delta`, `lidar_extrinsic_delta`, `control_point_delta`, `scale_delta`, `datum_shift`.
- Classification: `inlier_count`, `outlier_count`, `disabled_count`, `outlier_reason`, `quality_flag`.
- Aggregates: per-camera, per-LiDAR-frame, per-control-point, per-spatial-tile, and per-dataset summaries.

## First Implementation Path

1. Add `LaserControlPoint`, `LaserObservation`, and `LiDARFrame` data structures without treating LiDAR controls as
   ordinary SfM tracks.
2. Add a fixture provenance reader that validates `source_manifest.json`, derived file checksums, coordinate frames,
   and license/access notes before loading test data.
3. Add an import path for tiny PLY/LAS/LAZ-derived control point samples and sparse depth samples.
4. Extend Bundle Adjustment with a separately weighted residual family for camera-to-laser point constraints. Keep
   feature-track residuals and laser residuals independently reported.
5. Add diagnostics with per-camera reprojection residual, per-LiDAR-frame laser residual, before/after pose deltas,
   outlier reasons, and summary percentiles.
6. Build the first integration test from a small MUN-FRL-derived fixture or, if redistribution review is incomplete,
   from a synthetic fixture plus a local-only MUN-FRL reproduction recipe.

## Selection Notes

- MUN-FRL is the best first BA dataset because it is aerial, synchronized, explicitly calibrated, and currently has
  clear CC BY 4.0 wording for dataset files.
- UseGeo should be the first depth/DEM companion because it provides UAV images, camera poses, depth maps, LiDAR
  point clouds, and MVS point clouds in a photogrammetry-oriented package.
- H3D is the best photogrammetry-quality dense reference because its UAV LiDAR and imagery are intentionally paired
  and hybrid-adjusted, but registration makes it less convenient for CI.
- MARS-LVIG and UAVScenes are valuable for modern UAV LiDAR-camera alignment and trajectory diagnostics. Their
  non-commercial share-alike terms make them local research targets, not default fixtures.
- NTU VIRAL is valuable for sensor-fusion residual design, but its robotics focus means it is better for pose/extrinsic
  tests than for final DEM/DOM products.
- DublinCity and DFC 2019 are strong later-stage geospatial validation datasets once PlaScan has robust CRS, tiling,
  and large-cloud IO handling.
- ETH3D and Tanks and Temples are not UAV datasets, but they remain useful guardrails for MVS and fusion correctness.

## Download and Licensing Rules

- Do not commit raw dataset downloads to the repository.
- Keep only tiny derived fixtures in `testData/` after license and redistribution review.
- Record source URL, terms/license URL, access date, citation, raw checksum, derived checksum, extraction command, and
  coordinate frame for every fixture.
- Do not use non-commercial, registration-only, or no-redistribution datasets in release artifacts, CI downloads, or
  commercial demos without a separate review.
- Prefer CC BY 4.0 datasets for any fixture that may be shipped with the repository, but still verify whether derived
  samples are acceptable.
- If terms conflict or are ambiguous, keep only a reproducible recipe and require users to download the raw data
  themselves.
