# PlaScan Test Data Utilities

This directory contains small built-in fixtures and optional downloaded photogrammetry benchmark datasets.

## Download benchmark datasets

Use `download_photogrammetry_testdata.py` to download curated public benchmark data into
`testData/photogrammetry_benchmarks`.

```bash
python testData/download_photogrammetry_testdata.py --list
python testData/download_photogrammetry_testdata.py --dataset middlebury_dino_sparse_ring --extract
```

Large datasets are skipped unless `--include-large` is passed.

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
