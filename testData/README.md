# PlaScan Test Data Utilities

This directory contains small built-in fixtures and optional downloaded photogrammetry benchmark datasets.

## Built-in aerial stereo pair

`aerial_stereo_pair/` contains two consecutive 1600×1067 nadir aerial images, matching TSAI pinhole camera files,
WGS84 GNSS/IMU records, and ready-to-use `image_camera.lis` / `images.lis` inputs. The pair has an approximately
25.20 m acquisition baseline and strong verified feature overlap. See `aerial_stereo_pair/README.md` for coordinate
frame details, validation results, source attribution, and redistribution notes.

## Built-in TIFF RPC stereo pair

`rpc_stereo_pair/` contains two 8-bit Pléiades TIFF satellite images with complete RPC00B parameters embedded in
the GDAL `RPC` metadata domain. An explicit JSON export of all normalization values and 4×20 coefficients is included
for inspection. Both images use the same documented 1%–99% linear stretch from the upstream 16-bit containers. The
`Products/` subdirectory contains a PlaScan-generated 2 m georeferenced DEM, its quality rasters, and an RPC-orthorectified
RGBA DOM. See `rpc_stereo_pair/README.md` for geometry, validation, source, and licensing details.

## Download benchmark datasets

Use `download_photogrammetry_testdata.py` to download curated public benchmark data into
`testData/photogrammetry_benchmarks`.

```bash
python testData/download_photogrammetry_testdata.py --list
python testData/download_photogrammetry_testdata.py --dataset middlebury_dino_sparse_ring --extract
python testData/download_photogrammetry_testdata.py --dataset asp_dawn_fc_vesta_frame --extract
python testData/download_photogrammetry_testdata.py --dataset asp_lronac_csm_example --extract
python testData/download_photogrammetry_testdata.py --dataset eth3d_courtyard_highres --include-large
python testData/download_photogrammetry_testdata.py --dataset eth3d_facade_highres --include-large
python testData/download_photogrammetry_testdata.py --dataset eth3d_office_highres --include-large
```

Large datasets are skipped unless `--include-large` is passed.

For PlaScan's current planetary frame-camera path, prefer `asp_dawn_fc_vesta_frame`. It is a fixed NASA Ames Stereo
Pipeline solved example over Vesta from the digital Dawn FC2 orbital framing camera. The package contains two
1024×1024 ISIS cubes, matching USGS CSM Frame camera JSON files, an ASP recipe, and sample DEM/DRG outputs. All pixels
in each CSM Frame image share one acquisition time, camera center, and orientation, so this is not line-scan or
push-frame data. The packaged recipe does not require ISIS supporting data or an ISIS installation. Files are stored
under `testData/photogrammetry_benchmarks/asp_dawn_fc_vesta_frame/`.

The older `asp_lronac_csm_example` contains two cropped LRO NAC ISIS cubes, CSM camera JSON files, and sample DEM/DRG
outputs. LRO NAC is a pushbroom/line-scan sensor, so keep it as a dataset and interoperability fixture until PlaScan
implements a time-varying line-scan camera model; it is not a pinhole/frame-camera reconstruction regression.

For the current outdoor frame-camera regression, use `eth3d_courtyard_highres`. It downloads the official 38-view
Courtyard training scene into `testData/photogrammetry_benchmarks/eth3d_courtyard_highres/archives/`, including raw
and undistorted DSLR images, camera parameters, scan-evaluation data, and per-view depth ground truth (1,536,923,897
compressed bytes total). The downloader validates exact sizes and pinned SHA-256 hashes. The `.7z` archives require
7-Zip/p7zip for extraction.

For a larger outdoor architectural regression, use `eth3d_facade_highres`. It contains 76 building-facade views,
raw and undistorted DSLR inputs, camera parameters, scan-evaluation data, and depth ground truth. Its four archives
total 3,394,284,959 compressed bytes and are downloaded into
`testData/photogrammetry_benchmarks/eth3d_facade_highres/archives/`.

The `eth3d_office_highres` entry downloads the official 26-view Office training scene into
`testData/photogrammetry_benchmarks/eth3d_office_highres/archives/`. It includes undistorted DSLR images and camera
parameters, scan-evaluation ground truth, and per-view depth ground truth (883,139,523 compressed bytes total). The
downloader validates exact sizes and pinned SHA-256 hashes. Python's standard library does not extract `.7z` files;
use 7-Zip/p7zip after the download when extracted files are needed.

## Prepare the 100-camera Agisoft aerial model benchmark

`prepare_agisoft_aerial_subset.py` selects a deterministic, spatially connected
camera block from the full `agisoft_aerial_gcps` dataset. It filters missing
image/TSAI pairs, optionally rejects low-texture images, builds a symmetric
nearest-neighbor graph, and preserves the selected source order. When images
are resized, the script scales `fu`, `fv`, `cu`, and `cv` in each TSAI camera by
the exact output dimensions.

```powershell
.\.venv\Scripts\python.exe testData\prepare_agisoft_aerial_subset.py `
  --source testData\photogrammetry_benchmarks\agisoft_aerial_gcps\extracted\aerial_images_with_gcps `
  --output testData\photogrammetry_benchmarks\agisoft_aerial_gcps_100\prepared\plascan `
  --count 100 `
  --neighbors 8 `
  --minimum-texture-score 1000 `
  --max-image-dim 1600
```

The prepared dataset contains `Images/`, `cameras/`, `Metadata/`,
`image_camera.lis`, and `subset_manifest.json`. The manifest records every
selected camera and internal adjacency edge. The current fixture has 100
image/camera pairs and 376 internal edges in its symmetric 8-neighbor graph.

For a model-generation performance A/B, first generate fixed depth artifacts
with the explicit `aerial_terrain` scene profile. Depth estimation time is not
part of the model benchmark. Then run:

```powershell
.\.venv\Scripts\python.exe scripts\bench\run_model_generation_benchmark.py `
  --exe build\windows-vcpkg-cuda-release\bin\mesh_reconstruct_cli.exe `
  --depth-map-dir build\benchmark_runs\agisoft_aerial_gcps_100_depth_aerial_v1\headless.files\1\reconstruction\mvs `
  --settings-json scripts\bench\agisoft_aerial_gcps_100_mesh_settings.json `
  --output-dir build\benchmark_runs\agisoft_aerial_gcps_100_mesh_aerial_ab `
  --parallel-workers 30
```

The runner derives serial and parallel settings from the same no-interpolation
JSON, captures per-stage timings, verifies vertex/face counts and final PLY
SHA-256, and writes `benchmark_results.json` plus `benchmark_results.md`.

## LiDAR / laser-constrained photogrammetry candidates

The same downloader also keeps a curated registry of public datasets that may be useful when PlaScan adds
LiDAR points, laser scans, trajectories, or dense point clouds as constraints/reference data. Most of these entries
are intentionally marked as `manual` and `large`: the script records the official homepage and local manifest, but
does not silently download tens of gigabytes or data with registration/licensing steps.

```bash
python testData/download_photogrammetry_testdata.py --list
python testData/download_photogrammetry_testdata.py --category uav_lidar_fusion --dry-run
python testData/download_photogrammetry_testdata.py --workflow-tag ba_constraint_candidate --dry-run
python testData/download_photogrammetry_testdata.py --dataset mun_frl_vil --dry-run
```

Use tags in the `--list` output as quick guidance:

- `ba_constraint_candidate`: suitable for early experiments where LiDAR points, calibrated sensor poses, or
  trajectories participate in bundle adjustment or camera-LiDAR extrinsic checks.
- `lidar_fusion_validation`: suitable for comparing/fusing PlaScan image-based point clouds, DSM/DEM, or meshes
  against LiDAR/laser-scan reference data.

## Validate LiDAR inputs for laser-constrained BA

### Prepare the Tanks and Temples Ignatius object-ring benchmark

`prepare_tanks_ignatius_lidar_ba.py` converts the official COLMAP camera-to-world poses into the laser ground-truth
frame with `Ignatius_trans.txt`. It also downsamples the eight registered laser stations independently, estimates
local PCA normals with a KD-tree, orients each station's normals toward its scanner position, and writes one bounded
binary PLY for PlaScan's point-to-plane BA constraints.

```powershell
.\.venv\Scripts\python.exe testData\prepare_tanks_ignatius_lidar_ba.py `
  --dataset-root testData\photogrammetry_benchmarks\tanks_ignatius_quickstart `
  --voxel-size 0.005 `
  --k-neighbors 24 `
  --max-points-per-scan 100000 `
  --max-output-points 500000 `
  --workers -1 `
  --overwrite
```

The generated `prepared/plascan_lidar_ba/` directory contains:

- `cameras/000001.tsai` through `000263.tsai`, all expressed in the Ignatius laser frame;
- `image_camera.lis` and `images.lis` for headless matching/reconstruction workflows;
- `lidar/Ignatius_lidar_planes.ply` with `x/y/z`, `normal_x/normal_y/normal_z`, and `curvature` fields;
- `summary.json` with the applied similarity scale, point counts, normal statistics, and conservative BA defaults.

The official Tanks and Temples release does not include exact intrinsics. The adapter uses the benchmark's standard
initial pinhole estimate `fx=fy=0.7*image_width` (`1344 px` for Ignatius), centered principal point, and zero
distortion. Refine the camera model in a normal photogrammetry workflow when evaluating final accuracy.

In the GUI, use **Tools -> Import reference DEM/LiDAR**, choose **BA soft constraint (`ba_prior`)**, then run
**Reference terrain constraint re-adjustment** after a formal aerial triangulation. Point-cloud BA imports accept only
PLY files with real surface normals; LAS/LAZ/COPC/XYZ/CSV must first be converted. Do not enable the
"missing normals as height planes" option for an object-ring dataset.

For a headless smoke test, create a 60-frame list from the start of `image_camera.lis` (the segment completes an
approximately full orbit), run `match_photos_cli --reference-preselection`, then run `bundle_adjust_cli --ab-compare`
on a copy of the generated project. Start with:

```text
--laser-max-distance 0.05 --laser-max-curvature 0.2 --laser-max-samples 500000
--laser-sigma 0.0025 --laser-huber-delta 0.05 --refine-pose
```

`--laser-sigma` is converted to the statistical point-to-plane weight `1/sigma^2` (for 2.5 mm, `160000`).
Use `--laser-weight` only when an explicit weight is required; its default `0` selects sigma-based weighting.
The CLI writes compact per-track JSON by default, while the GUI keeps detailed per-observation residuals for its
feature-residual viewer.

Always keep a pure-image baseline, require a non-zero associated-track count, and compare both reprojection and
LiDAR RMS. A 60-frame Ignatius smoke run with the settings above retained 4,561 laser constraints: reprojection RMS
changed from `5.072 px` to `2.301 px`, point-to-plane RMS from `26.515 mm` to `18.405 mm`, and the absolute-distance
median from `21.160 mm` to `6.211 mm`. These numbers validate that the constraint is active; the benchmark intrinsics
are approximate, so they are not a final survey-accuracy claim. The A/B CLI writes the LiDAR result back into its
input project, so always use a disposable project copy.

After extracting the MUN-FRL lighthouse sample, use `validate_lidar_ba_inputs.py` to check whether the generated
PLY streams can be used directly by the current LiDAR point-to-plane BA prototype:

```bash
python testData/validate_lidar_ba_inputs.py \
  --dataset-root testData/photogrammetry_benchmarks/mun_frl_vil/lighthouse_benchmarking_bag/extracted \
  --summary-json build/mun_frl_lidar_ba_summary.json
```

The script only reads PLY headers. It reports which stream contains `x/y/z` plus normals and is therefore suitable
for the current `enable_laser_constraints` BA option. For the extracted MUN-FRL sample, `lidar/cloud_registered`
is the intended first BA constraint stream; `lidar/velodyne_points` is useful for later raw LiDAR fusion or
sensor-level constraints but needs normal estimation before point-to-plane BA.

The current GUI/service path accepts one PLY file at a time. Start with the recommended
`ba_constraint_cloud_path` from the JSON summary for smoke tests; merging or windowing per-frame LiDAR PLYs is a
follow-up preprocessing step.

Prepare a small real-data A/B benchmark manifest for comparing ordinary BA with LiDAR-constrained BA:

```bash
python testData/prepare_lidar_ba_ab_benchmark.py \
  --dataset-root testData/photogrammetry_benchmarks/mun_frl_vil/lighthouse_benchmarking_bag/extracted \
  --output-dir build/mun_frl_lidar_ba_ab_benchmark \
  --start-index 5 \
  --window-size 20 \
  --max-abs-dt-ms 100 \
  --merge-lidar
```

This writes `benchmark_plan.json`, `images.lis`, and `lidar_clouds.lis`. On the current extracted sample, the default
window selects 20 images and 10 unique `lidar/cloud_registered` PLY files. With `--merge-lidar`, those frames are
merged into `merged_lidar_cloud.ply`, and the LiDAR BA option `laser_constraint_cloud_path` is pointed at that merged
PLY.

The extracted MUN-FRL `cloud_registered` PLY files contain normal fields, but the observed sample has zero normals.
Estimate usable point-to-plane normals before running LiDAR-constrained BA:

```bash
python testData/estimate_lidar_normals.py \
  --input build/mun_frl_lidar_ba_ab_benchmark/merged_lidar_cloud.ply \
  --output build/mun_frl_lidar_ba_ab_benchmark/merged_lidar_cloud_normals.ply \
  --k-neighbors 16 \
  --summary-json build/mun_frl_lidar_ba_ab_benchmark/merged_lidar_cloud_normals_summary.json
```

If the external reference is a DEM/DSM rather than a point cloud with normals, convert an ESRI ASCII Grid (`.asc`)
into an XYZ-only PLY and use PlaScan's explicit height-plane mode:

```bash
python testData/dem_grid_to_height_ply.py \
  --input path/to/reference_dem.asc \
  --output build/lidar_height_constraints/reference_dem_height_planes.ply \
  --summary-json build/lidar_height_constraints/reference_dem_height_planes_summary.json

build/bin/bundle_adjust_cli \
  path/to/project.plascan \
  --output-dir build/ba_with_dem_height_planes \
  --laser-cloud build/lidar_height_constraints/reference_dem_height_planes.ply \
  --laser-missing-normals-as-height-planes \
  --ab-compare \
  --fail-on-quality-gate \
  --force
```

The converter writes only `x/y/z` vertices. It does not reproject or change CRS; the DEM/DSM grid coordinates must
already be in the same world/map frame as the PlaScan sparse points and cameras. This mirrors the conservative first
step of using external heights as soft BA constraints before adding full raster sampling or map-projection support.

For an end-to-end CLI BA smoke test, first run `match_photos_cli` on the selected window. CUDA SIFT features stay in
task memory and the CLI writes one binary `.pimatch` shard per image plus `match_photos_report.json`. The C++
`ImageMatchFile` reader remains the only binary parser; the project builder copies the report's
`image_match_files` index into project metadata.

Then package a temporary PlaScan project from the benchmark manifest, ROS camera info, odometry, and match shards:

```bash
python testData/prepare_mun_frl_lidar_ba_project.py \
  --benchmark-plan build/mun_frl_lidar_ba_ab_benchmark/benchmark_plan.json \
  --camera-info testData/photogrammetry_benchmarks/mun_frl_vil/lighthouse_benchmarking_bag/extracted/camera/camera_info_first.yaml \
  --trajectory testData/photogrammetry_benchmarks/mun_frl_vil/lighthouse_benchmarking_bag/extracted/trajectory/odometry.csv \
  --tf-static testData/photogrammetry_benchmarks/mun_frl_vil/lighthouse_benchmarking_bag/extracted/tf/tf_static_unique.csv \
  --camera-frame camera \
  --body-frame imu_link \
  --matches-dir build/mun_frl_lidar_ba_ab_benchmark/cli_match_run/assets/image_matches \
  --match-report build/mun_frl_lidar_ba_ab_benchmark/cli_match_run/reports/match_photos_report.json \
  --output-dir build/mun_frl_lidar_ba_ab_project \
  --project-name mun_frl_lidar_ba
```

`--tf-static` applies the ROS-style static transform chain to convert odometry body poses into camera poses before
writing PlaScan camera JSON. The extracted lighthouse sample has odometry `camera_init -> lio_body` and static
transforms including `camera -> imu_link` and `camera -> velodyne`. Current smoke-test evidence favors
`--body-frame imu_link` for image BA geometry, while `--body-frame velodyne` is useful as a diagnostic because it
can reduce LiDAR residuals at the cost of a worse image-only baseline.

Run ordinary BA versus LiDAR-constrained BA with the standalone CLI:

```bash
build/bin/bundle_adjust_cli \
  build/mun_frl_lidar_ba_ab_project/mun_frl_lidar_ba.plascan \
  --output-dir build/mun_frl_lidar_ba_ab_project/ba_ab_run_laser5m \
  --laser-cloud build/mun_frl_lidar_ba_ab_benchmark/merged_lidar_cloud_normals.ply \
  --ab-compare \
  --laser-max-distance 5 \
  --max-iterations 5 \
  --max-point-iterations 8 \
  --max-camera-iterations 3 \
  --threads 8 \
  --force
```

The first MUN-FRL smoke run is useful for exercising the data path, not for final accuracy claims. The default
`--laser-max-distance 1` rejected all tracks in the current 20-image window; `--laser-max-distance 3..5` produces
non-zero associations. Because the camera/LiDAR frame preparation is still approximate, compare reprojection RMS,
optimized track count, LiDAR residuals, and associated constraint count together.

After two BA runs produce JSON summaries, compare them with the fixed-scope evaluation helper:

```bash
python testData/compare_lidar_ba_ab_results.py \
  --baseline-json build/mun_frl_lidar_ba_ab_project/ba_ab_run_laser5m/baseline/ba_run_summary.json \
  --lidar-json build/mun_frl_lidar_ba_ab_project/ba_ab_run_laser5m/laser/ba_run_summary.json \
  --output-json build/mun_frl_lidar_ba_ab_project/ba_ab_run_laser5m/fixed_comparison.json \
  --output-md build/mun_frl_lidar_ba_ab_project/ba_ab_run_laser5m/fixed_comparison.md
```

The comparison report focuses on global reprojection RMS, optimized track count, common-valid-track RMS, LiDAR
point-to-plane RMS/median, associated LiDAR constraint count, and refined-camera center/rotation drift. On the current
20-image lighthouse smoke window with estimated normals:

| Pose chain | Laser radius | Result |
|---|---:|---|
| `camera_init -> lio_body` plus `camera -> imu_link` inverse | 1 m | 0 associated tracks; only validates no-LiDAR baseline |
| `camera_init -> lio_body` plus `camera -> imu_link` inverse | 3-5 m | image-only baseline is strong (`~0.533 px`), but LiDAR RMS increases after BA |
| `camera_init -> lio_body` plus `camera -> velodyne` inverse | 3-5 m | LiDAR RMS decreases, but image-only baseline is weaker (`~1.126 px`) |

Treat these as diagnostics. A final accuracy test still needs a better-confirmed `lio_body` to camera/LiDAR frame
definition, or a dataset slice with independently verified camera-LiDAR extrinsics in the same map frame.

For ASP-style final geometry validation, compare a PlaScan image-derived point cloud against a LiDAR/reference point
cloud with nearest-neighbor distance metrics:

```bash
python testData/compare_point_cloud_to_lidar.py \
  --source build/plascan_dense_or_sparse_cloud.ply \
  --reference build/lidar_reference_sample.ply \
  --output-json build/plascan_vs_lidar_quality.json \
  --nearest-neighbor-method auto \
  --max-rmse-m 0.50 \
  --max-p95-m 1.00 \
  --coverage-radius-m 1.00 \
  --min-reference-coverage-percent 75 \
  --fail-on-quality-gate
```

This first helper reads ASCII PLY files with `x/y/z` vertices and simple `.csv`, `.txt`, or `.xyz` point files whose
first three columns are `x y z`. Extra PLY fields such as intensity or classification are ignored, and one text header
row is tolerated. It uses exact nearest-neighbor distances from PlaScan points to the reference points, so use it on
small benchmark clouds or pre-windowed/downsampled LiDAR samples; large LAS/LAZ/COPC files should first be clipped or
converted by an external point-cloud tool. The JSON report includes source-to-reference mean, RMSE, median, p95, max,
plus reference LiDAR coverage within `--coverage-radius-m`. Use the coverage gate to catch partial overlap cases where
the reconstruction is close only to a small subset of the reference cloud. `--nearest-neighbor-method auto` keeps
small comparisons on the brute-force path and uses the built-in exact KD-tree path for larger reference samples; set
`brute` or `kd-tree` explicitly when debugging metric differences.

Candidate table, checked against official or dataset homepages on 2026-06-17:

| Dataset | Data types | Scale | Homepage / entry | License / restrictions | PlaScan use | Risks / notes |
|---|---|---:|---|---|---|---|
| H3D Hessigheim UAV LiDAR | UAV imagery, high-density UAV LiDAR, textured mesh, labels | LiDAR about 800 pts/m^2, multi-epoch | https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx | Download requires subscription/registration; redistribution terms need manual confirmation | `BA` maybe later, `fusion/validation` strong | Best UAV mapping match, but original camera/orientation availability must be confirmed after download |
| MUN-FRL VIL | Aerial RGB/mono cameras, Velodyne LiDAR, IMU, RTK/PPK GNSS, calibration | About 27-90 GB per main sequence; smaller workshop bags exist | https://mun-frl-vil-dataset.readthedocs.io/en/latest/ | Raw data/calibration are CC BY 4.0; cite paper | `BA` strong, `fusion/validation` strong | ROS bag parsing and time synchronization adapters are needed |
| NTU VIRAL | UAV stereo cameras, dual 3D LiDAR, IMU, UWB, ground truth, calibration | About 4-9.4 GB per sequence | https://ntu-aris.github.io/ntu_viral_dataset/ | CC BY-NC-SA 4.0; non-commercial academic use | `BA` strong, `fusion/validation` strong | Ground truth is measured at prism; official note warns about 0.4 m IMU-prism offset |
| USTC FLICAR | Multiple LiDARs, stereo/mono cameras, IMU, laser tracker ground truth | Aerial sequences about 25.5-121.1 GB | https://ustc-flicar.github.io/ | Public homepage/downloads, but license is unclear; needs manual confirmation | `BA` strong, `fusion/validation` strong | Aerial work robot/bucket-truck platform, not a UAV; very large sequences |
| UrbanScene3D | Large-scale aerial image sets, LiDAR scans, meshes, simulator data | About 1.43 TB, 128k+ images, 16 scenes | https://vcc.tech/UrbanScene3D | Non-commercial only; citation required; no redistribution or altered variants | `fusion/validation` strong | Huge dataset; use only selected scenes/proxy data for local tests |
| DublinCity | Airborne LiDAR, vertical/oblique aerial imagery, labeled point clouds | About 5.6 km^2 scanned; about 260M labeled points from 1.4B raw points | https://v-sense.scss.tcd.ie/dublincity/ | Academic use with citation; commercial use requires contact | `fusion/validation` strong | Original imagery/LiDAR are in NYU repository; camera orientation availability needs confirmation |
| ISPRS Vaihingen 3D Semantic Labeling | Airborne laser scanning point clouds, reflectance/return info, labels | Two ALS areas, ASCII XYZ/reference files | https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/3d-semantic-labeling.aspx | Terms of use need manual confirmation from ISPRS benchmark page | `fusion/validation` useful | Mainly ALS point cloud benchmark, not full photogrammetry block |
| USGS 3DEP + USDA NAIP pairing | Public airborne LiDAR catalog plus aerial orthoimagery | Area-dependent national catalog | https://www.usgs.gov/3d-elevation-program | 3DEP products are free and without use restrictions; NAIP terms must be checked per source | `fusion/validation` useful | Not a single ready-made benchmark; user must choose matching area/year/CRS |
| ETH3D MVS | DSLR images, camera/depth files, laser scan ground truth | High-res scenes from about 14-76 images; GT scans per scene | https://www.eth3d.net/datasets | Dataset terms need manual confirmation | `fusion/validation` useful | Not UAV/aerial; good for MVS accuracy regression |
| Tanks and Temples | Video/image sets, laser scanned training GT, COLMAP poses/alignment files | Training and benchmark scenes; per-scene downloads | https://www.tanksandtemples.org/download/ | Free for non-commercial purposes under dataset license | `fusion/validation` useful | Does not provide exact intrinsics; camera model may need optimization |
| KITTI Raw | Vehicle cameras, Velodyne LiDAR, GPS/IMU, calibration | Sequence-dependent | https://www.cvlibs.net/datasets/kitti/raw_data.php | CC BY-NC-SA 3.0; non-commercial | `BA` useful for algorithm units, `fusion/validation` useful | Ground vehicle data, not UAV/aerial; dynamic objects and driving geometry differ from photogrammetry |
| DTU MVS SampleSet | Multi-view images, calibration, structured-light reference geometry | SampleSet and Points are each about 6.3 GB | https://roboimagedata.compute.dtu.dk/?page_id=36 | Citeware; cite related papers | `BA` limited, `fusion/validation` useful | Indoor object-scale benchmark, not aerial/LiDAR |

Recommended near-term test sets:

1. `mun_frl_vil`: best first target for LiDAR/trajectory constraints in BA because it has synchronized cameras, LiDAR,
   IMU, RTK/PPK, and explicit CC BY 4.0 raw-data licensing.
2. `h3d_hessigheim_uav_lidar`: closest to PlaScan's UAV photogrammetry direction and strongest candidate for
   image-based point cloud versus UAV LiDAR fusion/quality checks.
3. `ntu_viral`: good second multi-sensor BA/extrinsic test after MUN-FRL; smaller per sequence than many mapping
   datasets, with calibration and ground truth, but non-commercial restrictions apply.

## Prepare benchmark camera files for PlaScan

Use `prepare_photogrammetry_benchmarks.py` after downloading and extracting supported datasets. It does not modify
the original benchmark files. It writes a PlaScan-ready view under each dataset:

```text
testData/photogrammetry_benchmarks/<dataset>/prepared/plascan/
  image_camera.lis
  cameras/*.tsai
  summary.json
```

Supported converters:

- Middlebury sparse-ring `*_par.txt`: converts `K * [R t]` into PlaScan camera-to-world `.tsai`.
- EPFL/Strecha `.camera`: converts `K[R^T|-R^T t]X` into PlaScan `.tsai`.

Run all supported converters:

```bash
python testData/prepare_photogrammetry_benchmarks.py --target-root testData/photogrammetry_benchmarks --all --overwrite
```

The generated `image_camera.lis` can be passed to PlaScan CLI tools that accept image/camera lists.

The same conversion core is also available through the built CLI and GUI Tools menu:

```bash
build/bin/camera_convert_cli --format auto \
  --input testData/photogrammetry_benchmarks/middlebury_dino_sparse_ring/extracted/dinoSparseRing \
  --output-dir build/camera_inputs/dino \
  --overwrite
```

In the GUI, use `工具 -> 相机格式转换...` to choose the input camera file/directory and output directory.
