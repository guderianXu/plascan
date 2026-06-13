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
