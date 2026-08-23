#!/usr/bin/env python3
"""Download curated photogrammetry and LiDAR-adjacent benchmark datasets for PlaScan tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_ROOT = REPO_ROOT / "testData" / "photogrammetry_benchmarks"


@dataclass(frozen=True)
class Resource:
    filename: str
    url: str = ""
    manual_url: str = ""
    size_hint: str = ""
    large: bool = False
    note: str = ""
    expected_bytes: int = 0
    expected_sha256: str = ""


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    title: str
    category: str
    source_url: str
    license_note: str
    description: str
    resources: tuple[Resource, ...]
    workflow_tags: tuple[str, ...] = ()


DATASETS: dict[str, Dataset] = {
    "colmap_south_building": Dataset(
        dataset_id="colmap_south_building",
        title="COLMAP South Building",
        category="classic_sfm_mvs",
        source_url="https://demuc.de/colmap/datasets/",
        license_note="COLMAP 示例数据；使用时按 COLMAP 数据页和原始数据说明引用。",
        description="经典 SfM/MVS 建筑场景，适合验证特征、匹配、SfM 和稠密重建主链。",
        resources=(
            Resource(
                filename="south-building.zip",
                url="https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip",
                size_hint="large",
                large=True,
                note="高分辨率建筑影像和 COLMAP 示例工程；体量较大，默认 --all 会跳过。",
            ),
        ),
    ),
    "middlebury_temple_sparse_ring": Dataset(
        dataset_id="middlebury_temple_sparse_ring",
        title="Middlebury TempleSparseRing",
        category="calibrated_object_mvs",
        source_url="https://vision.middlebury.edu/mview/data/",
        license_note="Middlebury MVS benchmark；用于研究和评测时请引用 Middlebury MVS 数据集论文。",
        description="16 视角标定物体小数据，适合快速回归相机读入、三角化和 MVS 几何一致性。",
        resources=(
            Resource(
                filename="templeSparseRing.zip",
                url="https://vision.middlebury.edu/mview/data/data/templeSparseRing.zip",
                size_hint="4 MB",
            ),
        ),
    ),
    "middlebury_dino_sparse_ring": Dataset(
        dataset_id="middlebury_dino_sparse_ring",
        title="Middlebury DinoSparseRing",
        category="calibrated_object_mvs",
        source_url="https://vision.middlebury.edu/mview/data/",
        license_note="Middlebury MVS benchmark；用于研究和评测时请引用 Middlebury MVS 数据集论文。",
        description="16 视角标定物体小数据，和 TempleSparseRing 搭配覆盖不同形状与纹理。",
        resources=(
            Resource(
                filename="dinoSparseRing.zip",
                url="https://vision.middlebury.edu/mview/data/data/dinoSparseRing.zip",
                size_hint="4 MB",
            ),
        ),
    ),
    "epfl_rathaus_multiview": Dataset(
        dataset_id="epfl_rathaus_multiview",
        title="EPFL Strecha City Hall Leuven",
        category="classic_sfm_mvs",
        source_url="https://www.epfl.ch/labs/cvlab/data/data-strechamvs/",
        license_note="EPFL/CVLAB Strecha MVS 数据；原站声明仅供研究使用并要求致谢/引用。",
        description="带内外方位元素的真实建筑多视影像，适合验证标定导入和少视角重建。",
        resources=(
            Resource(
                filename="rathaus.tar.gz",
                url="https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/rathaus.tar.gz",
                size_hint="small",
            ),
        ),
    ),
    "eth3d_two_view_training": Dataset(
        dataset_id="eth3d_two_view_training",
        title="ETH3D Low-res Two-view Training",
        category="benchmark_depth_gt",
        source_url="https://www.eth3d.net/datasets",
        license_note="ETH3D benchmark；使用时请遵守 ETH3D 数据页和论文引用要求。",
        description="小体量两视图训练集，包含图像和真值，适合深度图/视差基础验证。",
        resources=(
            Resource(
                filename="two_view_training.7z",
                url="https://www.eth3d.net/data/two_view_training.7z",
                size_hint="13.6 MB",
                note="7z 归档可下载；标准库不自动解压，请用 7z/p7zip 解压。",
            ),
            Resource(
                filename="two_view_training_gt.7z",
                url="https://www.eth3d.net/data/two_view_training_gt.7z",
                size_hint="14.2 MB",
                note="7z 归档可下载；标准库不自动解压，请用 7z/p7zip 解压。",
            ),
        ),
    ),
    "eth3d_delivery_area_lowres": Dataset(
        dataset_id="eth3d_delivery_area_lowres",
        title="ETH3D Delivery Area Low-res Many-view",
        category="benchmark_depth_gt",
        source_url="https://www.eth3d.net/datasets",
        license_note="ETH3D benchmark；使用时请遵守 ETH3D 数据页和论文引用要求。",
        description="多视低分辨率场景，含深度/评估数据，适合 MVS 精度和完整性评估。",
        resources=(
            Resource(
                filename="delivery_area_rig_undistorted.7z",
                url="https://www.eth3d.net/data/delivery_area_rig_undistorted.7z",
                size_hint="0.2 GB",
                large=True,
                note="默认 --all 跳过；需要时加 --include-large。",
            ),
            Resource(
                filename="delivery_area_rig_depth.7z",
                url="https://www.eth3d.net/data/delivery_area_rig_depth.7z",
                size_hint="0.4 GB",
                large=True,
                note="默认 --all 跳过；需要时加 --include-large。",
            ),
        ),
    ),
    "eth3d_office_highres": Dataset(
        dataset_id="eth3d_office_highres",
        title="ETH3D Office High-res Multi-view Training",
        category="benchmark_depth_gt",
        source_url="https://www.eth3d.net/datasets",
        license_note="ETH3D benchmark；使用时请遵守 ETH3D 数据页和论文引用要求。",
        description="26 视角室内 DSLR 场景，包含无畸变图像、相机参数、扫描评估真值和深度真值。",
        resources=(
            Resource(
                filename="office_dslr_undistorted.7z",
                url="https://www.eth3d.net/data/office_dslr_undistorted.7z",
                size_hint="288,765,137 bytes (0.3 GB)",
                large=True,
                note="高分辨率无畸变图像和相机参数；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=288765137,
                expected_sha256="316c0c10c79cc173e4b5c26102fed3caedbde7e1865fd896ae15423f0f8cf04c",
            ),
            Resource(
                filename="office_dslr_scan_eval.7z",
                url="https://www.eth3d.net/data/office_dslr_scan_eval.7z",
                size_hint="153,021,449 bytes (0.1 GB)",
                large=True,
                note="用于官方高分辨率多视图表面评估；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=153021449,
                expected_sha256="25ff6d0f2d421896e8391ad0484ad918a1fab597c6dc37c85cfaa6675c5ece88",
            ),
            Resource(
                filename="office_dslr_depth.7z",
                url="https://www.eth3d.net/data/office_dslr_depth.7z",
                size_hint="441,352,937 bytes (0.4 GB)",
                large=True,
                note="每视图深度真值；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=441352937,
                expected_sha256="aa2e421079eff69a25332e41c81f2f7ecfea54de97bd30f49e2435cc7c700d8b",
            ),
        ),
    ),
    "eth3d_courtyard_highres": Dataset(
        dataset_id="eth3d_courtyard_highres",
        title="ETH3D Courtyard High-res Multi-view Training",
        category="benchmark_depth_gt",
        source_url="https://www.eth3d.net/datasets",
        license_note="ETH3D benchmark；CC BY-NC-SA 4.0，非商业使用，并遵守数据页和论文引用要求。",
        description="38 视角室外庭院 DSLR 场景，包含原始/无畸变图像、相机参数、扫描评估和深度真值。",
        resources=(
            Resource(
                filename="courtyard_dslr_undistorted.7z",
                url="https://www.eth3d.net/data/courtyard_dslr_undistorted.7z",
                size_hint="500,990,569 bytes (0.5 GB)",
                large=True,
                note="无畸变面阵图像和 PINHOLE 相机参数；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=500990569,
                expected_sha256="9dc3126363bc89b229e2b564c7725f1b858a52b46a5452ccfb9e9eb8e72ba26d",
            ),
            Resource(
                filename="courtyard_dslr_jpg.7z",
                url="https://www.eth3d.net/data/courtyard_dslr_jpg.7z",
                size_hint="425,492,376 bytes (0.4 GB)",
                large=True,
                note="原始畸变 DSLR 图像和 THIN_PRISM_FISHEYE 相机参数；用于导入边界校正测试。",
                expected_bytes=425492376,
                expected_sha256="91addd6e1f8be6d68ebac1c967ca934c06e8acaeee002d7cd9480fc223a37aef",
            ),
            Resource(
                filename="courtyard_dslr_scan_eval.7z",
                url="https://www.eth3d.net/data/courtyard_dslr_scan_eval.7z",
                size_hint="182,151,847 bytes (0.2 GB)",
                large=True,
                note="用于官方高分辨率多视图表面评估；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=182151847,
                expected_sha256="d1d335d6c0710e2a2ee8e3d1268d03cf6feb625cbff25c568a1eb90e90637ca7",
            ),
            Resource(
                filename="courtyard_dslr_depth.7z",
                url="https://www.eth3d.net/data/courtyard_dslr_depth.7z",
                size_hint="428,289,105 bytes (0.4 GB)",
                large=True,
                note="每视图深度真值；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=428289105,
                expected_sha256="c0a149fd43c3f3aaad2ea12cd070613ceb8b4835f115eeb2557cd8212c862639",
            ),
        ),
    ),
    "eth3d_facade_highres": Dataset(
        dataset_id="eth3d_facade_highres",
        title="ETH3D Facade High-res Multi-view Training",
        category="benchmark_depth_gt",
        source_url="https://www.eth3d.net/datasets",
        license_note="ETH3D benchmark；CC BY-NC-SA 4.0，非商业使用，并遵守数据页和论文引用要求。",
        description="76 视角室外建筑立面 DSLR 场景，包含原始/无畸变图像、相机参数、扫描评估和深度真值。",
        resources=(
            Resource(
                filename="facade_dslr_undistorted.7z",
                url="https://www.eth3d.net/data/facade_dslr_undistorted.7z",
                size_hint="1,252,088,400 bytes (1.2 GB)",
                large=True,
                note="无畸变面阵图像和 PINHOLE 相机参数；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=1252088400,
                expected_sha256="046e577388db0633eeb2d8d72da6a2de53857434b4a7a98e4c97485795f9ce82",
            ),
            Resource(
                filename="facade_dslr_jpg.7z",
                url="https://www.eth3d.net/data/facade_dslr_jpg.7z",
                size_hint="1,161,258,255 bytes (1.1 GB)",
                large=True,
                note="原始畸变 DSLR 图像和相机参数；用于导入边界校正测试。",
                expected_bytes=1161258255,
                expected_sha256="d9570f38fcb026b6a10946b83fa42b7bd7ff40ed7ead044aab4fd0364329a13b",
            ),
            Resource(
                filename="facade_dslr_scan_eval.7z",
                url="https://www.eth3d.net/data/facade_dslr_scan_eval.7z",
                size_hint="176,517,224 bytes (0.2 GB)",
                large=True,
                note="用于官方高分辨率多视图表面评估；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=176517224,
                expected_sha256="d686462d417fbea010d0021f918bec8b881444929ec06e3bd4e8a5230ebf59a8",
            ),
            Resource(
                filename="facade_dslr_depth.7z",
                url="https://www.eth3d.net/data/facade_dslr_depth.7z",
                size_hint="804,421,080 bytes (0.7 GB)",
                large=True,
                note="每视图深度真值；7z 归档需用 7z/p7zip 解压。",
                expected_bytes=804421080,
                expected_sha256="f44b013efed624fb90b5e10e651ef3dd750cf621017db021c5b7dfcff868131d",
            ),
        ),
    ),
    "tanks_ignatius_quickstart": Dataset(
        dataset_id="tanks_ignatius_quickstart",
        title="Tanks and Temples Ignatius",
        category="real_scene_video",
        source_url="https://www.tanksandtemples.org/tutorial/",
        license_note="Tanks and Temples 数据有专门许可；下载和使用前请阅读其 license 页面。",
        description="真实室外/室内重建 benchmark 训练场景，适合视频/抽帧、纹理和 mesh 评估。",
        resources=(
            Resource(
                filename="Ignatius.zip",
                url="https://storage.googleapis.com/t2-downloads/image_sets/Ignatius.zip",
                size_hint="large",
                large=True,
                note="Quickstart image set；默认 --all 跳过，确实需要时加 --include-large。",
            ),
            Resource(
                filename="Ignatius.ply",
                url="https://storage.googleapis.com/t2-training-gt-data/Ignatius/Ignatius.ply",
                size_hint="ground truth ply",
                large=True,
                note="训练集公开真值；默认 --all 跳过，确实需要时加 --include-large。",
            ),
            Resource(
                filename="download_t2_dataset.py",
                url="https://raw.githubusercontent.com/IntelVCL/TanksAndTemples/master/python_toolbox/download_t2_dataset.py",
                size_hint="small",
                note="官方 downloader，可用于下载更多 Tanks and Temples 场景。",
            ),
        ),
    ),
    "dtu_mvs_sampleset": Dataset(
        dataset_id="dtu_mvs_sampleset",
        title="DTU MVS SampleSet",
        category="benchmark_depth_gt",
        source_url="https://roboimagedata.compute.dtu.dk/?page_id=36",
        license_note="DTU Robot Image Data Sets；原站说明为 citeware，使用时需引用相关论文。",
        description="实验室高精度标定和结构光真值，适合严肃评估准确性但体量很大。",
        resources=(
            Resource(
                filename="SampleSet.zip",
                url="http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip",
                size_hint="6.3 GB",
                large=True,
            ),
            Resource(
                filename="Points.zip",
                url="http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip",
                size_hint="6.3 GB",
                large=True,
            ),
        ),
    ),
    "agisoft_aerial_gcps": Dataset(
        dataset_id="agisoft_aerial_gcps",
        title="Agisoft Aerial Images with GCPs",
        category="aerial_mapping",
        source_url="https://www.agisoft.com/downloads/sample-data/",
        license_note="Agisoft sample data；请按 Agisoft 示例数据页和教程要求使用。",
        description="航测影像、GCP 和 GNSS 偏移，适合 DEM/DOM、控制点和地理参考流程验证。",
        resources=(
            Resource(
                filename="aerial_images_with_gcps.zip",
                url="https://download.agisoft.com/datasets/aerial_images_with_gcps.zip",
                size_hint="444 images",
                large=True,
                note="航测完整示例，默认 --all 跳过。",
            ),
        ),
    ),
    "h3d_hessigheim_uav_lidar": Dataset(
        dataset_id="h3d_hessigheim_uav_lidar",
        title="Hessigheim 3D (H3D) UAV LiDAR and MVS",
        category="uav_lidar_fusion",
        source_url="https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx",
        license_note="Institute for Photogrammetry, University of Stuttgart benchmark；下载需订阅/注册，许可和再分发限制需人工确认。",
        description="UAV 同平台采集的高密度 LiDAR、影像和纹理网格，适合验证影像重建点云与 LiDAR 点云融合和精度检查。",
        resources=(
            Resource(
                filename="H3D_official_download",
                manual_url="https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx",
                size_hint="UAV LiDAR about 800 pts/m^2; multi-epoch benchmark",
                large=True,
                note="官方页面通过 Subscribe & Download Data 提供下载入口；下载前确认许可和坐标/相机数据内容。",
            ),
        ),
        workflow_tags=("ba_constraint_candidate", "lidar_fusion_validation"),
    ),
    "mun_frl_vil": Dataset(
        dataset_id="mun_frl_vil",
        title="MUN-FRL Visual-Inertial-LiDAR Dataset",
        category="aerial_vil_lidar",
        source_url="https://mun-frl-vil-dataset.readthedocs.io/en/latest/",
        license_note="原始数据和标定文件为 CC BY 4.0；论文许可另有说明，使用时需引用原论文。",
        description="直升机和 DJI M600 平台的同步相机、LiDAR、IMU、RTK/PPK GNSS 与标定，适合激光点/轨迹参与 BA 的近期原型测试。",
        resources=(
            Resource(
                filename="MUN_FRL_official_downloads",
                manual_url="https://mun-frl-vil-dataset.readthedocs.io/en/latest/",
                size_hint="sequences about 27-90 GB; workshop bags about 3.55-6 GB",
                large=True,
                note="官方表格提供 Google Drive 链接；优先选择 lighthouse 或 workshop bag 做小规模验证。",
            ),
        ),
        workflow_tags=("ba_constraint_candidate", "lidar_fusion_validation"),
    ),
    "ntu_viral": Dataset(
        dataset_id="ntu_viral",
        title="NTU VIRAL UAV Visual-Inertial-Ranging-LiDAR",
        category="uav_multi_sensor_lidar",
        source_url="https://ntu-aris.github.io/ntu_viral_dataset/",
        license_note="CC BY-NC-SA 4.0，仅非商业学术使用；商业用途需联系数据集作者。",
        description="UAV 双 3D LiDAR、双同步相机、IMU、UWB 和 ground truth，适合测试相机-LiDAR 外参、轨迹先验和激光约束。",
        resources=(
            Resource(
                filename="NTU_VIRAL_official_downloads",
                manual_url="https://ntu-aris.github.io/ntu_viral_dataset/",
                size_hint="sequences about 4-9.4 GB; calibration data about 49 MB to 0.96 GB",
                large=True,
                note="官方提醒 ground truth 位于 prism，和 IMU 存在 0.4 m offset；使用前必须处理该偏移。",
            ),
        ),
        workflow_tags=("ba_constraint_candidate", "lidar_fusion_validation"),
    ),
    "ustc_flicar": Dataset(
        dataset_id="ustc_flicar",
        title="USTC FLICAR Lidar-Inertial-Camera Dataset",
        category="aerial_work_robot_lidar",
        source_url="https://ustc-flicar.github.io/",
        license_note="已公开主页和下载入口，但未在已核验页面看到明确许可；需要人工确认。",
        description="高空作业平台采集的多 LiDAR、相机、IMU 和激光跟踪仪真值，适合高精度轨迹和多传感器约束研究。",
        resources=(
            Resource(
                filename="USTC_FLICAR_official_downloads",
                manual_url="https://ustc-flicar.github.io/datasets/",
                size_hint="aerial sequences about 25.5-121.1 GB",
                large=True,
                note="平台不是无人机而是升降臂/高空作业机器人；下载前确认许可、坐标系和标定文件。",
            ),
        ),
        workflow_tags=("ba_constraint_candidate", "lidar_fusion_validation"),
    ),
    "urbanscene3d": Dataset(
        dataset_id="urbanscene3d",
        title="UrbanScene3D",
        category="large_scale_aerial_lidar",
        source_url="https://vcc.tech/UrbanScene3D",
        license_note="公开页面声明仅限非商业用途，需引用论文，不允许传播数据或修改版本。",
        description="大规模城市高分影像、LiDAR scans、合成/真实城市场景和模拟器，适合航测路径规划、重建和点云验证。",
        resources=(
            Resource(
                filename="UrbanScene3D_official_downloads",
                manual_url="https://vcc.tech/UrbanScene3D",
                size_hint="about 1.43 TB; 128k+ images; 16 scenes",
                large=True,
                note="体量很大且非商业限制严格；建议只选单场景或 proxy/camera 数据做离线试验。",
            ),
        ),
        workflow_tags=("lidar_fusion_validation",),
    ),
    "dublincity_lidar_aerial": Dataset(
        dataset_id="dublincity_lidar_aerial",
        title="DublinCity Annotated LiDAR and Aerial Images",
        category="airborne_lidar_aerial",
        source_url="https://v-sense.scss.tcd.ie/dublincity/",
        license_note="学术研究需引用论文；商业应用需联系作者/团队。",
        description="Dublin 市区航空 LiDAR、垂直/倾斜航空影像和标注点云，适合城市尺度影像重建点云与 ALS 对比。",
        resources=(
            Resource(
                filename="DublinCity_official_downloads",
                manual_url="https://v-sense.scss.tcd.ie/dublincity/",
                size_hint="about 5.6 km^2 scanned; about 260M labeled points from 1.4B raw points",
                large=True,
                note="原始 LiDAR 和航空影像位于 NYU 数据仓库；相机外方位、影像块组织需要下载后确认。",
            ),
        ),
        workflow_tags=("lidar_fusion_validation",),
    ),
    "isprs_vaihingen_3d_semantic": Dataset(
        dataset_id="isprs_vaihingen_3d_semantic",
        title="ISPRS Vaihingen 3D Semantic Labeling",
        category="airborne_lidar_point_cloud",
        source_url="https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/3d-semantic-labeling.aspx",
        license_note="ISPRS benchmark；页面提供数据和条款入口，具体使用限制需按 terms of use 人工确认。",
        description="Vaihingen ALS 点云、反射强度/回波信息和语义标签，适合点云分类、融合后质量检查和空间精度抽检。",
        resources=(
            Resource(
                filename="ISPRS_Vaihingen_3D_official_download",
                manual_url="https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/3d-semantic-labeling.aspx",
                size_hint="two ALS areas; ASCII XYZ plus reflectance and return count",
                large=True,
                note="主要是 ALS 点云，不是完整相机-LiDAR同步航测包；适合验证点云产品而非 BA。",
            ),
        ),
        workflow_tags=("lidar_fusion_validation",),
    ),
    "usgs_3dep_naip_pairing": Dataset(
        dataset_id="usgs_3dep_naip_pairing",
        title="USGS 3DEP LiDAR paired with USDA NAIP imagery",
        category="public_aerial_lidar_catalog",
        source_url="https://www.usgs.gov/3d-elevation-program",
        license_note="USGS 3DEP 产品官方声明免费且无使用限制；NAIP 影像使用限制需按 USDA/数据门户条款逐项确认。",
        description="美国公开航空 LiDAR 和航空正射影像目录，适合选择同一区域进行 DOM/DSM、点云融合和精度验证。",
        resources=(
            Resource(
                filename="USGS_3DEP_NAIP_manual_selection",
                manual_url="https://www.usgs.gov/3d-elevation-program",
                size_hint="area-dependent; nationwide catalog",
                large=True,
                note="不是单一 benchmark；需要人工选定区域、年份、坐标系和 NAIP 影像匹配关系。",
            ),
        ),
        workflow_tags=("lidar_fusion_validation",),
    ),
    "kitti_raw_lidar_camera": Dataset(
        dataset_id="kitti_raw_lidar_camera",
        title="KITTI Raw Camera and Velodyne LiDAR",
        category="ground_mobile_lidar_camera",
        source_url="https://www.cvlibs.net/datasets/kitti/raw_data.php",
        license_note="CC BY-NC-SA 3.0，非商业用途；使用时需引用 KITTI 论文和遵守条款。",
        description="车载相机、Velodyne LiDAR、GPS/IMU 和标定，适合调试相机-LiDAR投影、外参和点到射线约束的基础算法。",
        resources=(
            Resource(
                filename="KITTI_raw_official_downloads",
                manual_url="https://www.cvlibs.net/datasets/kitti/raw_data.php",
                size_hint="sequence-dependent",
                large=True,
                note="地面车载数据，不代表 UAV/航空摄影测量；动态物体较多，适合算法单元验证。",
            ),
        ),
        workflow_tags=("ba_constraint_candidate", "lidar_fusion_validation"),
    ),
    "agisoft_depth_images": Dataset(
        dataset_id="agisoft_depth_images",
        title="Agisoft Depth Images",
        category="benchmark_depth_gt",
        source_url="https://www.agisoft.com/downloads/sample-data/",
        license_note="Agisoft sample data；请按 Agisoft 示例数据页和教程要求使用。",
        description="带深度传感器的 iPad Pro 影像，适合验证深度图/颜色保留和 RGB-D 辅助流程。",
        resources=(
            Resource(
                filename="depth_images.zip",
                url="https://download.agisoft.com/datasets/depth_images.zip",
                size_hint="29 images",
            ),
        ),
    ),
    "pix4d_quarry_mapper": Dataset(
        dataset_id="pix4d_quarry_mapper",
        title="Pix4D Quarry Example",
        category="aerial_mapping",
        source_url="https://support.pix4d.com/hc/en-us/articles/360000235126",
        license_note="Pix4D 示例项目；原站声明仅供个人/专业培训，商业或宣传用途需标注 Pix4D。",
        description="UAV 航测 quarry 场景，含影像、GCP 和 Pix4D project，适合航测端到端参考。",
        resources=(
            Resource(
                filename="example_quarry_2.0.zip",
                url="https://data.pix4d.com/misc/example_datasets/example_quarry_2.0.zip",
                size_hint="347 images",
                large=True,
                note="完整航测示例，默认 --all 跳过。",
            ),
        ),
    ),
    "asp_dawn_fc_vesta_frame": Dataset(
        dataset_id="asp_dawn_fc_vesta_frame",
        title="Ames Stereo Pipeline Dawn FC Vesta Frame-camera Example",
        category="planetary_stereo",
        source_url="https://stereopipeline.readthedocs.io/en/latest/examples/csm.html",
        license_note="ASP solved example 仓库为 Apache-2.0；Dawn/PDS 影像仍需保留任务归属并遵守引用要求。",
        description="灶神星 Dawn FC2 真正面阵双目，含 1024×1024 ISIS cubes、CSM Frame 相机和 DEM/DRG 成果。",
        resources=(
            Resource(
                filename="DawnFramingCamera_example.tar",
                url=(
                    "https://github.com/NeoGeographyToolkit/StereoPipelineSolvedExamples/releases/"
                    "download/DawnFC/DawnFramingCamera_example.tar"
                ),
                size_hint="13,189,120 bytes (12.6 MiB)",
                note="数字轨道面阵相机；示例配方无需 ISIS 支撑数据或 ISIS 安装。",
                expected_bytes=13189120,
                expected_sha256="fdb4f595175fd04ec4ec3a063d601c987bbc678d6ece6c98dbffc06fb8a506f7",
            ),
        ),
    ),
    "asp_lronac_csm_example": Dataset(
        dataset_id="asp_lronac_csm_example",
        title="Ames Stereo Pipeline LRO NAC CSM Example",
        category="planetary_stereo",
        source_url="https://stereopipeline.readthedocs.io/en/latest/tutorial.html",
        license_note="NASA/ASP solved example；按 ASP 文档和数据源说明使用。",
        description="月球 LRO NAC 双目和 CSM camera 示例，适合 PlaScan 行星影像/DEM 场景对照。",
        resources=(
            Resource(
                filename="LRONAC_example.tar",
                url="https://github.com/NeoGeographyToolkit/StereoPipelineSolvedExamples/releases/download/LRONAC/LRONAC_example.tar",
                size_hint="204,144,640 bytes (194.7 MiB)",
                note="月球 LRO NAC 立体影像、CSM 相机文件和示例 DEM/DRG；适合行星地表流程验证。",
                expected_bytes=204144640,
                expected_sha256="5307b5a42a829833339e6e3e2991f837f352c11232c1a54c99a20409ed282dfa",
            ),
        ),
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download curated photogrammetry and LiDAR-adjacent datasets into PlaScan testData.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT, help="download destination root")
    parser.add_argument("--list", action="store_true", help="list available datasets and exit")
    parser.add_argument("--dataset", action="append", default=[], help="dataset id to download; repeatable")
    parser.add_argument("--category", action="append", default=[], help="category to download; repeatable")
    parser.add_argument("--workflow-tag", action="append", default=[], help="workflow tag to download; repeatable")
    parser.add_argument("--all", action="store_true", help="select all datasets")
    parser.add_argument("--include-large", action="store_true", help="include resources marked as large")
    parser.add_argument("--dry-run", action="store_true", help="write manifests but do not download archives")
    parser.add_argument("--extract", action="store_true", help="extract zip/tar archives after download")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing archive files")
    parser.add_argument("--timeout", type=int, default=60, help="network timeout per request, in seconds")
    parser.add_argument("--retries", type=int, default=2, help="download retries per resource")
    return parser.parse_args(argv)


def select_datasets(
    dataset_ids: Iterable[str],
    categories: Iterable[str],
    workflow_tags: Iterable[str],
    include_all: bool,
) -> list[Dataset]:
    dataset_id_set = set(dataset_ids)
    category_set = set(categories)
    workflow_tag_set = set(workflow_tags)

    unknown_ids = sorted(dataset_id_set.difference(DATASETS.keys()))
    if unknown_ids:
        raise ValueError(f"Unknown dataset id(s): {', '.join(unknown_ids)}")

    known_categories = {dataset.category for dataset in DATASETS.values()}
    unknown_categories = sorted(category_set.difference(known_categories))
    if unknown_categories:
        raise ValueError(f"Unknown category/categories: {', '.join(unknown_categories)}")

    known_workflow_tags = {tag for dataset in DATASETS.values() for tag in dataset.workflow_tags}
    unknown_workflow_tags = sorted(workflow_tag_set.difference(known_workflow_tags))
    if unknown_workflow_tags:
        raise ValueError(f"Unknown workflow tag(s): {', '.join(unknown_workflow_tags)}")

    selected: list[Dataset] = []
    for dataset in DATASETS.values():
        if (
            include_all
            or dataset.dataset_id in dataset_id_set
            or dataset.category in category_set
            or workflow_tag_set.intersection(dataset.workflow_tags)
        ):
            selected.append(dataset)
    return selected


def list_datasets(target_root: Path = DEFAULT_TARGET_ROOT) -> None:
    print(f"target_root: {target_root}")
    print("available datasets:")
    for dataset in DATASETS.values():
        flags = []
        if any(resource.large for resource in dataset.resources):
            flags.append("large")
        if any(resource.manual_url and not resource.url for resource in dataset.resources):
            flags.append("manual")
        flags.extend(dataset.workflow_tags)
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        print(f"  {dataset.dataset_id:32s} {dataset.category:22s} {dataset.title}{flag_text}")


def resource_to_manifest(resource: Resource, status: str, path: Path | None = None) -> dict[str, object]:
    item = asdict(resource)
    item["status"] = status
    if path is not None:
        item["local_path"] = str(path)
    return item


def write_manifest(
    dataset: Dataset,
    dataset_dir: Path,
    mode: str,
    resource_entries: list[dict[str, object]],
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": dataset.dataset_id,
        "title": dataset.title,
        "category": dataset.category,
        "source_url": dataset.source_url,
        "license_note": dataset.license_note,
        "description": dataset.description,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_tags": list(dataset.workflow_tags),
        "resources": resource_entries,
    }
    manifest_path = dataset_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def recorded_resource_contracts(dataset_dir: Path) -> dict[str, tuple[int, str]]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid existing download manifest {manifest_path}: {exc}") from exc

    contracts: dict[str, tuple[int, str]] = {}
    resources = manifest.get("resources", [])
    if not isinstance(resources, list):
        raise RuntimeError(f"Invalid resources list in download manifest: {manifest_path}")
    for entry in resources:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("filename", "")).strip()
        digest = str(entry.get("sha256", "")).strip().lower()
        try:
            size_bytes = int(entry.get("size_bytes", 0))
        except (TypeError, ValueError):
            continue
        if filename and size_bytes >= 0 and len(digest) == 64:
            try:
                int(digest, 16)
            except ValueError:
                continue
            contracts[filename] = (size_bytes, digest)
    return contracts


def write_manual_instructions(dataset: Dataset, dataset_dir: Path, entries: list[dict[str, object]]) -> None:
    lines = [
        f"# {dataset.title}",
        "",
        f"dataset_id: {dataset.dataset_id}",
        f"source_url: {dataset.source_url}",
        f"license_note: {dataset.license_note}",
        "",
        "Resources requiring manual handling:",
    ]
    for entry in entries:
        url = entry.get("manual_url") or entry.get("url") or dataset.source_url
        note = entry.get("note") or ""
        lines.append(f"- {entry['filename']}: {url}")
        if note:
            lines.append(f"  note: {note}")
    (dataset_dir / "MANUAL_DOWNLOAD.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024.0
    return f"{size} B"


def parse_content_length(headers: object) -> int:
    get_header = getattr(headers, "get", None)
    if get_header is None:
        return 0

    try:
        value = get_header("Content-Length")
    except TypeError:
        return 0

    if not value:
        return 0

    try:
        parsed = int(value)
        return parsed if parsed >= 0 else 0
    except ValueError:
        return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resource_file(resource: Resource, path: Path) -> tuple[int, str]:
    size_bytes = path.stat().st_size
    digest = sha256_file(path)
    if resource.expected_bytes > 0 and size_bytes != resource.expected_bytes:
        raise ValueError(
            f"Size mismatch for {resource.filename}: expected "
            f"{resource.expected_bytes}, got {size_bytes}"
        )
    expected_sha256 = resource.expected_sha256.strip().lower()
    if expected_sha256 and digest.lower() != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {resource.filename}: expected "
            f"{expected_sha256}, got {digest}"
        )
    return size_bytes, digest


def download_url(resource: Resource, output_path: Path, timeout: int, retries: int, overwrite: bool) -> str:
    if output_path.exists() and not overwrite:
        try:
            validate_resource_file(resource, output_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Cached resource validation failed for {resource.filename}: {exc}"
            ) from exc
        print(f"exists: {resource.filename} -> {output_path}", flush=True)
        return "exists"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(resource.url, headers={"User-Agent": "PlaScan-testdata-downloader/1.0"})
    chunk_size = 64 * 1024

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                print(f"retry {attempt}/{retries}: {resource.filename}", flush=True)

            print(f"download: {resource.filename} <- {resource.url}", flush=True)
            with urllib.request.urlopen(request, timeout=timeout) as response, part_path.open("wb") as handle:
                total_size = parse_content_length(response.headers)
                downloaded = 0
                last_reported = 0
                report_step = min(1024 * 1024, total_size) if total_size > 0 else 1024 * 1024

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    handle.write(chunk)
                    downloaded += len(chunk)

                    should_report = downloaded - last_reported >= report_step
                    if total_size > 0:
                        should_report = should_report or downloaded >= total_size

                    if should_report:
                        if total_size > 0:
                            percent = min(100.0, downloaded * 100.0 / total_size)
                            print(
                                f"  {resource.filename}: {percent:.1f}% "
                                f"({format_bytes(downloaded)}/{format_bytes(total_size)})",
                                flush=True,
                            )
                        else:
                            print(f"  {resource.filename}: {format_bytes(downloaded)}", flush=True)
                        last_reported = downloaded

            if total_size > 0 and downloaded != total_size:
                raise ValueError(
                    f"Content-Length mismatch for {resource.filename}: "
                    f"expected {total_size}, got {downloaded}"
                )
            validate_resource_file(resource, part_path)
            part_path.replace(output_path)
            print(f"done: {resource.filename} -> {output_path}", flush=True)
            return "downloaded"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink()
            print(f"failed: {resource.filename}: {exc}", flush=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"Failed to download {resource.url}: {last_error}")


def validated_member_path(destination: Path, member_name: str) -> Path:
    if not member_name or "\0" in member_name:
        raise RuntimeError("Archive contains an empty or invalid member name")
    destination_resolved = destination.resolve()
    member_path = (destination / member_name).resolve()
    if destination_resolved not in [member_path, *member_path.parents]:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")
    return member_path


def safe_extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            validated_member_path(destination, member.name)
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Archive links are not allowed: {member.name} -> {member.linkname}"
                )
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported tar member type: {member.name}")
        tar.extractall(destination)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            validated_member_path(destination, member.filename)
            unix_mode = member.external_attr >> 16
            file_type = stat.S_IFMT(unix_mode)
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"Archive links are not allowed: {member.filename}")
            if not member.is_dir() and file_type not in (0, stat.S_IFREG):
                raise RuntimeError(f"Unsupported zip member type: {member.filename}")
        zipped.extractall(destination)


def extract_archive(archive: Path, destination: Path) -> str:
    suffixes = "".join(archive.suffixes[-2:]).lower()
    destination.mkdir(parents=True, exist_ok=True)

    if archive.suffix.lower() == ".zip":
        safe_extract_zip(archive, destination)
        return "extracted"

    if archive.suffix.lower() == ".tar" or suffixes in {".tar.gz", ".tar.bz2", ".tar.xz"}:
        safe_extract_tar(archive, destination)
        return "extracted"

    if archive.suffix.lower() == ".7z":
        return "skipped_extract_7z"

    return "skipped_extract_unknown"


def download_dataset(
    dataset: Dataset,
    target_root: Path,
    include_large: bool,
    dry_run: bool,
    extract: bool,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> None:
    dataset_dir = target_root / dataset.dataset_id
    recorded_contracts = {} if dry_run else recorded_resource_contracts(dataset_dir)
    resource_entries: list[dict[str, object]] = []
    manual_entries: list[dict[str, object]] = []

    for resource in dataset.resources:
        archive_path = dataset_dir / "archives" / resource.filename

        if dry_run:
            print(f"{dataset.dataset_id}: dry-run {resource.filename}", flush=True)
            entry = resource_to_manifest(resource, "dry_run")
            resource_entries.append(entry)
            if not resource.url:
                manual_entries.append(entry)
            continue

        if resource.large and not include_large:
            print(
                f"{dataset.dataset_id}: skip large {resource.filename} "
                f"(add --include-large to download)",
                flush=True,
            )
            entry = resource_to_manifest(resource, "skipped_large")
            resource_entries.append(entry)
            manual_entries.append(entry)
            continue

        if not resource.url:
            print(f"{dataset.dataset_id}: manual {resource.filename}", flush=True)
            entry = resource_to_manifest(resource, "manual")
            resource_entries.append(entry)
            manual_entries.append(entry)
            continue

        effective_resource = resource
        if not overwrite and resource.filename in recorded_contracts:
            recorded_size, recorded_sha256 = recorded_contracts[resource.filename]
            effective_resource = replace(
                resource,
                expected_bytes=resource.expected_bytes or recorded_size,
                expected_sha256=resource.expected_sha256 or recorded_sha256,
            )
        status = download_url(
            effective_resource,
            archive_path,
            timeout=timeout,
            retries=retries,
            overwrite=overwrite,
        )
        entry = resource_to_manifest(resource, status, archive_path)
        size_bytes, digest = validate_resource_file(resource, archive_path)
        entry["size_bytes"] = size_bytes
        entry["sha256"] = digest

        if extract:
            print(f"{dataset.dataset_id}: extract {resource.filename}", flush=True)
            extract_status = extract_archive(archive_path, dataset_dir / "extracted")
            entry["extract_status"] = extract_status

        resource_entries.append(entry)

    mode = "dry_run" if dry_run else "download"
    write_manifest(dataset, dataset_dir, mode, resource_entries)
    if manual_entries:
        write_manual_instructions(dataset, dataset_dir, manual_entries)

    print(f"{dataset.dataset_id}: wrote {dataset_dir / 'manifest.json'}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list or not (args.all or args.dataset or args.category or args.workflow_tag):
        list_datasets(args.target_root)
        return 0

    try:
        selected = select_datasets(args.dataset, args.category, args.workflow_tag, args.all)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not selected:
        print("No datasets selected.", file=sys.stderr)
        return 2

    failed: list[str] = []
    for dataset in selected:
        try:
            download_dataset(
                dataset=dataset,
                target_root=args.target_root,
                include_large=args.include_large,
                dry_run=args.dry_run,
                extract=args.extract,
                overwrite=args.overwrite,
                timeout=args.timeout,
                retries=args.retries,
            )
        except RuntimeError as exc:
            failed.append(dataset.dataset_id)
            print(f"{dataset.dataset_id}: error: {exc}", file=sys.stderr, flush=True)

    if failed:
        print(f"Failed datasets: {', '.join(failed)}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
