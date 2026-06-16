# Laser and Photogrammetry Datasets for Future BA Support

This note records candidate datasets for adding laser points / LiDAR point clouds into PlaScan workflows.
The immediate engineering purpose is to prepare future tests for:

- laser point constraints in Bundle Adjustment;
- camera / LiDAR extrinsic sanity checks;
- sparse cloud registration against external control points;
- dense point cloud fusion and filtering quality checks;
- MVS depth and point cloud validation against laser-scanned ground truth.

## Recommended Shortlist

| Priority | Dataset | Sensors and geometry | Why it matters for PlaScan | license / access notes | First PlaScan use |
| --- | --- | --- | --- | --- | --- |
| 1 | [MUN-FRL](https://mun-frl-vil-dataset.readthedocs.io/) | UAV and aircraft visual-inertial-LiDAR data; RGB images, LiDAR point clouds, GPS/IMU logs, calibration files | Best first target for laser point constraints because it has synchronized aerial imagery, LiDAR, IMU/GNSS-style navigation data, and explicit calibration | Dataset files are documented as CC BY 4.0; publication may have a separate non-commercial license | Import camera / LiDAR calibration, create laser control points, test Bundle Adjustment priors |
| 2 | [Hessigheim 3D / H3D](https://www2.isprs.org/commissions/comm2/wg2/benchmark/) | UAV LiDAR point cloud plus UAV imagery and textured mesh products; high-density RGB-enriched point cloud | Strong photogrammetry target: UAV image reconstruction can be compared against a dense laser point cloud and mesh | Benchmark access requires checking current ISPRS / University of Stuttgart terms before redistribution | point cloud fusion, dense cloud QA, MVS depth validation |
| 3 | [NTU VIRAL](https://ntu-aris.github.io/ntu_viral_dataset/) | UAV platform with two 3D LiDARs, time-synchronized cameras, IMU, UWB ranging, calibration, and ground-truth resources | Useful for camera / LiDAR extrinsic verification and trajectory-aware BA experiments, especially when testing priors rather than only final clouds | Public dataset; verify current license on download page before using in CI or releases | sensor calibration import, pose priors, laser constraints in Bundle Adjustment |
| 4 | [UseGeo](https://usegeo.fbk.eu/home) | UAV-based multi-sensor geospatial dataset with image and LiDAR data | Good domain match for UAV photogrammetry, terrain products, and LiDAR-assisted dense reconstruction | Check project download terms before mirroring | terrain / DEM / DOM validation with LiDAR support |
| 5 | [DublinCity / NYU Dublin LiDAR](https://archive.nyu.edu/jspui/handle/2451/38659) | City-scale aerial laser scanning and photogrammetry data | Large external LiDAR reference for urban reconstruction and registration tests | Archive record lists CC BY 4.0 | large point cloud IO, control point extraction, city-scale fusion |
| 6 | [ETH3D](https://www.eth3d.net/) | High-resolution images and high-precision laser scan ground truth | Excellent for MVS accuracy and completeness evaluation, less UAV-specific | Dataset page lists CC BY-NC-SA 4.0 | MVS benchmark harness, laser ground truth comparison |
| 7 | [Tanks and Temples](https://www.tanksandtemples.org/) | Image/video reconstruction benchmark with industrial laser-scanned ground truth | Broad reconstruction QA target for fusion and meshing behavior | Check benchmark download terms before redistribution | dense cloud / mesh evaluation |
| 8 | [UAVStereo](https://github.com/rebecca0011/UAVStereo) | UAV stereo pairs; paper describes dense disparity generation from UAV imagery and LiDAR-supported meshes | Useful for depth estimation regression tests, but it is less direct for BA because it is stereo/depth oriented | Verify repository and download license before using | depth-map quality tests, stereo/MVS tuning |

## Dataset-to-Feature Mapping

| PlaScan feature | Primary datasets | Minimum data to import | Expected test artifact |
| --- | --- | --- | --- |
| Laser point constraints in Bundle Adjustment | MUN-FRL, NTU VIRAL | camera intrinsics/extrinsics, LiDAR-to-camera transform, sparse sampled LiDAR control points | BA report showing lower reprojection / laser residuals after optimization |
| LiDAR control points for known-pose SfM | MUN-FRL, H3D, UseGeo | image list, camera model, LiDAR point cloud, image-to-point correspondences or projected control candidates | `.plascan` project with laser control point residual table |
| point cloud fusion and filtering QA | H3D, DublinCity, ETH3D | reconstructed dense cloud and reference LiDAR cloud in a common frame | cloud-to-cloud distance statistics and color-preserving filtered PLY |
| MVS depth validation | H3D, ETH3D, Tanks and Temples, UAVStereo | reference laser cloud or depth/disparity ground truth and camera poses | per-frame depth error map and aggregate RMSE / completeness report |
| Terrain products with LiDAR reference | UseGeo, DublinCity, H3D | LiDAR ground surface reference, images, camera poses | DEM/DOM validation report against LiDAR-derived surface |

## First Implementation Path

1. Add a `LaserControlPoint` / `LaserObservation` data structure that stores world point, optional intensity/color, source cloud id, weight, and covariance.
2. Add an import path for PLY/LAS/LAZ control point samples without requiring them to become ordinary sparse SfM tracks.
3. Extend Bundle Adjustment with a separate residual block for camera-to-laser point constraints. Keep the feature-track residuals and laser residuals independently weighted.
4. Add a diagnostics report with per-camera reprojection residual, laser residual, and before/after pose delta.
5. Build the first integration test from a small MUN-FRL or H3D subset. The test should use a tiny sampled LiDAR cloud in-repo or generated from fixture data, not full raw downloads.

## Selection Notes

- MUN-FRL is the best first dataset because it is aerial, multi-sensor, and explicitly includes calibration-oriented data. It should drive BA and pose-prior design.
- H3D is the best photogrammetry-quality reference because the UAV LiDAR and imagery are intentionally paired for 3D reconstruction analysis.
- NTU VIRAL is valuable for sensor fusion design, but its robotics focus means it is better for pose/extrinsic tests than for final DEM/DOM products.
- ETH3D and Tanks and Temples are not UAV datasets, but they are useful guardrails for MVS and fusion correctness because laser-scanned ground truth is mature.
- DublinCity and UseGeo are good later-stage geospatial validation datasets once PlaScan has stronger large-cloud IO and coordinate reference handling.

## Download and Licensing Rules

- Do not commit raw dataset downloads to the repository.
- Keep only tiny derived fixtures in `testData/` when license terms allow it.
- Record source URL, license, checksum, and extraction script for every fixture.
- Do not use non-commercial datasets in release artifacts or commercial demos without a separate check.
- Prefer CC BY 4.0 datasets for CI fixtures and examples.
