#!/usr/bin/env python3
"""Download curated photogrammetry benchmark datasets for PlaScan tests."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    title: str
    category: str
    source_url: str
    license_note: str
    description: str
    resources: tuple[Resource, ...]


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
                size_hint="tutorial sample",
            ),
        ),
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download curated photogrammetry datasets into PlaScan testData.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT, help="download destination root")
    parser.add_argument("--list", action="store_true", help="list available datasets and exit")
    parser.add_argument("--dataset", action="append", default=[], help="dataset id to download; repeatable")
    parser.add_argument("--category", action="append", default=[], help="category to download; repeatable")
    parser.add_argument("--all", action="store_true", help="select all datasets")
    parser.add_argument("--include-large", action="store_true", help="include resources marked as large")
    parser.add_argument("--dry-run", action="store_true", help="write manifests but do not download archives")
    parser.add_argument("--extract", action="store_true", help="extract zip/tar archives after download")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing archive files")
    parser.add_argument("--timeout", type=int, default=60, help="network timeout per request, in seconds")
    parser.add_argument("--retries", type=int, default=2, help="download retries per resource")
    return parser.parse_args(argv)


def select_datasets(dataset_ids: Iterable[str], categories: Iterable[str], include_all: bool) -> list[Dataset]:
    dataset_id_set = set(dataset_ids)
    category_set = set(categories)

    unknown_ids = sorted(dataset_id_set.difference(DATASETS.keys()))
    if unknown_ids:
        raise ValueError(f"Unknown dataset id(s): {', '.join(unknown_ids)}")

    known_categories = {dataset.category for dataset in DATASETS.values()}
    unknown_categories = sorted(category_set.difference(known_categories))
    if unknown_categories:
        raise ValueError(f"Unknown category/categories: {', '.join(unknown_categories)}")

    selected: list[Dataset] = []
    for dataset in DATASETS.values():
        if include_all or dataset.dataset_id in dataset_id_set or dataset.category in category_set:
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
        "resources": resource_entries,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
        return int(value)
    except ValueError:
        return 0


def download_url(resource: Resource, output_path: Path, timeout: int, retries: int, overwrite: bool) -> str:
    if output_path.exists() and not overwrite:
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

            part_path.replace(output_path)
            print(f"done: {resource.filename} -> {output_path}", flush=True)
            return "downloaded"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink()
            print(f"failed: {resource.filename}: {exc}", flush=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"Failed to download {resource.url}: {last_error}")


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if destination_resolved not in [member_path, *member_path.parents]:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(destination)


def extract_archive(archive: Path, destination: Path) -> str:
    suffixes = "".join(archive.suffixes[-2:]).lower()
    destination.mkdir(parents=True, exist_ok=True)

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
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

        status = download_url(resource, archive_path, timeout=timeout, retries=retries, overwrite=overwrite)
        entry = resource_to_manifest(resource, status, archive_path)

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

    if args.list or not (args.all or args.dataset or args.category):
        list_datasets(args.target_root)
        return 0

    try:
        selected = select_datasets(args.dataset, args.category, args.all)
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
