# Third-party reconstruction diagnostics

These scripts build reproducible COLMAP/OpenMVS comparison inputs without
overwriting a PlaScan project or its current products.

`plascan_eth3d_surface_eval` provides deterministic ETH3D laser-scan to mesh
accuracy/completeness and topology metrics for TSDF/Visual Hull validation.
It is deliberately separate from the per-frame `plascan_eth3d_depth_eval` so
that depth improvements must also pass a final-model geometry check.

## Camera preparation

`prepare_middlebury_colmap.py` converts official Middlebury `K[R|t]` records
to a COLMAP text model. It is useful when evaluating against the dataset's
official camera coordinate system.

`prepare_plascan_colmap.py` exports the cameras in `mvs_manifest.json` together
with the sparse points and tracks in `sfm_sparse_points.json`. Use this route
to isolate dense depth and meshing differences while keeping PlaScan's SfM
result fixed.

Both scripts write `cameras.txt`, `images.txt`, `points3D.txt`, and a
ring-neighbour `patch-match.cfg` below the explicitly supplied output
directory.

## Depth and mesh comparison

`compare_depth_maps.py` compares PlaScan fast binary depth matrices with
COLMAP photometric or geometric depth maps by image name. The camera models
must be identical.

`compare_plascan_depth_runs.py` compares two PlaScan `mvs_manifest.json`
runs by `ref_index` and image basename. It reports pooled valid coverage,
mask IoU, relative depth differences, geometry support, inverse-depth spread,
acceptance transitions, and illegal values. Optional CLI thresholds turn the
report into a regression gate; a gate failure still writes the JSON report and
returns exit code 2. Large-run quantiles use a deterministic bounded reservoir,
while counts, means, maxima, and mask metrics remain exact.

`compare_mvs_depth_to_metashape.py` compares PlaScan depth with float depth
exported by `export_metashape_depth_maps.py`. In addition to absolute residuals
and mask overlap, it reports a diagnostic global scale factor, the residual
shape error after applying that factor, and the rigid-invariant scale implied
by matched camera-center pair distances. The scale-aligned values must not be
used as proof of absolute scale accuracy.

`summarize_mvs_depth_provenance.py` verifies that every final valid depth pixel
belongs to exactly one persisted provenance class and that every referenced
provenance artifact exists.

`audit_mvs_workspace_integrity.py` audits an `mvs_manifest.json` before depth
fusion. It summarizes frame completion/acceptance/fusion eligibility and
algorithm revisions, decodes every completed frame's support mask to verify its
path, dimensions, nonzero coverage, and all-white/all-black state, and reports
the persisted sparse absolute-depth residual distributions. Coverage differing
from `mask_coverage` by more than `1e-6` is an integrity failure by default.
The concise result is printed to the terminal; pass `--output` to retain the
complete per-frame JSON audit:

```powershell
.\.venv\Scripts\python.exe `
  scripts\validation\audit_mvs_workspace_integrity.py `
  E:\path\to\mvs_output\mvs_manifest.json `
  --output E:\path\to\validation\mvs_workspace_integrity.json
```

`compare_mesh_geometry.py` reports topology, triangle quality, bidirectional
point-to-surface distance, and normal agreement. Prefer binary PLY inputs;
parsing large textured OBJ files with `trimesh` is substantially slower. When
official Middlebury and PlaScan coordinates differ, pass both
`--reference-middlebury-par` and `--candidate-mvs-manifest` to estimate the
candidate-to-reference Sim(3) from corresponding camera centers rather than
from the surfaces being compared.

Run all scripts with the repository Python environment:

```powershell
.\.venv\Scripts\python.exe scripts\validation\prepare_plascan_colmap.py --help
.\.venv\Scripts\python.exe scripts\validation\compare_depth_maps.py --help
.\.venv\Scripts\python.exe scripts\validation\compare_plascan_depth_runs.py --help
.\.venv\Scripts\python.exe scripts\validation\compare_mvs_depth_to_metashape.py --help
.\.venv\Scripts\python.exe scripts\validation\compare_mesh_geometry.py --help
.\.venv\Scripts\python.exe scripts\validation\audit_mvs_workspace_integrity.py --help
```

Python dependencies used by the comparison scripts are `numpy`, OpenCV,
`Pillow`, `scipy`, `rtree`, and `trimesh`.

## ETH3D depth evaluation

`plascan_eth3d_depth_eval` scores a prediction generated for ETH3D's official
undistorted DSLR image against the raw laser depth. It implements COLMAP's
`THIN_PRISM_FISHEYE` equations, maps measured samples into the official
zero-distortion `PINHOLE` raster, and preserves depth on the corresponding
camera ray. The mapping uses the ETH3D/COLMAP corner-origin convention: raw
array element `[row,column]` has camera coordinate `(column+0.5,row+0.5)`.

The remap rejects invalid or depth-discontinuous raw samples, resolves target
pixel collisions deterministically, and performs a strict forward reprojection
check. Missing laser depth is reported as unobserved support rather than a
false-positive prediction. See `plascan_eth3d_depth_eval/README.md` for the
full convention and command contract. Native PlaScan depth grids can be scored
without full-resolution upsampling by pairing `--prediction-manifest`,
`--prediction-ref-index`, and the official `--undistorted-images` file. The
evaluator converts COLMAP's corner-origin K to PlaScan's raster-index K,
validates prediction path, frame name, fixed pose, acceptance, native-grid
state, and intrinsics, then fails closed on any provenance mismatch:

```bash
.venv/bin/python -m scripts.validation.plascan_eth3d_depth_eval --help
```

## Held-out textured-model evaluation

`evaluate_textured_model.py` loads the final OBJ/MTL/texture products and
rasterizes them into cameras that were not used to build the texture. It uses
a deterministic z-buffer, near-plane clipping, stable face IDs,
perspective-correct UV interpolation, and hardware texel-center texture
sampling.
Appearance is measured in linear sRGB only where rendered and reference
foreground pixels overlap; silhouette precision/recall/IoU are reported
separately so missing geometry cannot be hidden by a favorable color score.
The report also measures discontinuity across actual shared mesh edges,
sampled atlas occupancy/overlap, world-area-weighted UV Jacobian singular
values, anisotropy and conformal distortion, area stretch, and a
reference-relative sharpness ratio. Render, visibility, and error previews are
retained per view.

The input JSON must use schema `plascan.texture_eval.v1`, declare a non-empty
`training_images` list, and provide one or more held-out `views`. The evaluator
rejects path, byte-identical, and decoded-pixel overlap between the two sets.
Formal scoring requires every view to contain an `image`, same-size binary
foreground `mask`, and zero-distortion `camera_model`; relative paths resolve
against the manifest. `--allow-unmasked-diagnostic` is explicitly non-formal
and records that mode in the report. Reference images must already be
single-frame, untagged 8-bit RGB with EXIF orientation 1. Downscaling happens
in linear light, while masks use nearest-neighbour resizing. Camera fields are
`fx/fy/cx/cy`, nine row-major `rotation_world_to_camera` values, and a
three-value `camera_center`. For example:

```json
{
  "schema": "plascan.texture_eval.v1",
  "training_images": ["images/view_000.png"],
  "views": [
    {
      "id": "held_out_001",
      "image": "images/view_001.png",
      "mask": "masks/view_001.png",
      "camera_model": {
        "fx": 1200.0,
        "fy": 1200.0,
        "cx": 799.5,
        "cy": 599.5,
        "rotation_world_to_camera": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "camera_center": [0, 0, 0]
      }
    }
  ]
}
```

Run it with the repository runtime. `--max-dimension 0` keeps native image
resolution; the default 1600-pixel cap is intended for repeatable CPU smoke
comparisons.

```powershell
.\.venv\Scripts\python.exe `
  scripts\validation\evaluate_textured_model.py `
  --obj E:\run\textured_model.obj `
  --manifest E:\run\held_out_texture_eval.json `
  --output-dir E:\run\texture_evaluation
```

`prepare_middlebury_texture_holdout.py` creates a fixed Middlebury ring split,
training image/camera list, foreground masks, evaluation manifest, and SHA-256
preparation report without copying source images. Its default Dino protocol
holds out one-based views 4, 8, 12, and 16:

```bash
.venv/bin/python scripts/validation/prepare_middlebury_texture_holdout.py \
  --dataset-dir /data/dinoSparseRing \
  --camera-parameters /data/dinoSparseRing/dinoSR_par.txt \
  --input-list /runs/dino/image_camera_all.lis \
  --output-dir build/tmp/dino-texture-holdout
```

## Reproducible mesh-quality baseline

`run_mesh_quality_baseline.ps1` freezes the depth artifacts, effective settings,
mesh CLI, baseline mesh, and optional third-party reference mesh with SHA-256
records before reconstruction. It then produces mesh metrics, registered
geometry comparisons, a fixed-view contact sheet, and a hard-gate summary.
Pinned depth and settings hashes in `mesh_quality_scenes.json` make the runner
fail before reconstruction when a scene input drifts.

When zero-crossing diagnostics are enabled, each scene also writes
`products/boundary_attribution_debug.ply`. Interior vertices are dark gray;
boundary vertices use orange for extraction/postprocess, purple for support
gate rejection, red for absolute-TSDF rejection, yellow for surface-weight
rejection, magenta for depth-spread rejection, cyan for insufficient sources,
blue for no observation, and white for unclassified edges. The mesh statistics
record boundary counts after Marching Cubes, component filtering, weak-tip
trimming, topology cleanup, simplification, fallback simplification, and final
hole handling.

The default scene definitions are in `mesh_quality_scenes.json`. Paths support
`${REPO_ROOT}`, `${BUILD_DIR}`, and environment-variable tokens. Temple input
is intentionally supplied through `TEMPLE_DEPTH_DIR` instead of embedding a
machine-specific data path:

```powershell
$env:TEMPLE_DEPTH_DIR = 'E:\path\to\temple\mvs'
.\scripts\validation\run_mesh_quality_baseline.ps1 `
  -CandidateName cell_sheet_baseline `
  -ValidateOnly
```

Remove `-ValidateOnly` to reconstruct all selected scenes. The runner refuses
to overwrite an existing candidate directory. Use a new candidate name for
every experiment; `-ReuseExistingModel` is only for resuming an interrupted
report whose model already exists in that exact candidate directory.

The accepted MC33/native topology-safe validation profiles are:

- Dino: `mesh_gui_detail_topology_safe_dino_170k_v5.json` with
  `mesh_quality_scenes_gui_detail_topology_safe_dino_v5.json`.
- Temple: `mesh_mc33_topology_safe_temple_v4.json` with
  `mesh_quality_scenes_mc33_topology_safe_temple_v4.json`.

The frozen 2026-07-27 runs are
`task40_gui_detail_openmesh_dino_v5` and `task40_temple_openmesh_v8`.
Dino finished with 169,976 faces, 4 boundary edges, one component, a
20.86-degree adjacent-normal median, and 0.0008787 Chamfer-L1. Temple finished
with 60,060 faces, no boundary edges, one component, and an 11.59-degree
adjacent-normal median. Both strict gates passed; inspect each scene's
`contact_sheet.png` before accepting later parameter changes.

`mesh_gui_ultra_topology_safe.json` is the GUI-equivalent smoke profile: it only
sets the visible Ultra/240,000-face controls and leaves all TSDF advanced
options automatic. Its frozen supplemental output directories retain their
historical names:
`analysis/gui_ultra_openmesh_dino_240k` and
`analysis/gui_ultra_openmesh_temple_240k`. Dino produced 239,971 faces,
7 boundary edges, one component, a 16.79-degree normal median, and 0.0008693
Chamfer-L1 against the registered Metashape mesh. Temple produced 239,949
faces, 45 boundary edges, one component, and an 8.55-degree normal median.
These runs verify the GUI profile and provide the pre-migration baseline. New
runs select MC33, adaptive TGV, the six-voxel geometry profile, and PlaScan's
native topology-safe simplifier rather than the removed third-party path.

`mesh_gui_ultra_observed_topology_safe.json` is the observation-only counterpart.
It uses the same Ultra/240,000-face controls, disables interpolation, preserves
vertex colors, and enables zero-crossing attribution. Use it to distinguish
missing MVS depth from TSDF support-gate or extraction/post-processing holes;
do not accept a candidate solely because projected recall passes the quality
gate—inspect the aligned contact sheet and geometry-distance report as well.

## Depth/pose/fusion attribution

`run_depth_pose_fusion_ablation.ps1` freezes declared camera, depth, fusion,
and mesh artifacts before running A/B variants. The output
`ablation_manifest.json` records a SHA-256 fingerprint for every stage,
the exact executable and arguments, elapsed time, exit code, and fingerprints
of expected products. Existing output directories are never overwritten.

The JSON configuration contains `scenes[].stages` and `scenes[].variants`.
Stage values are arrays of files or directories. Variant arguments support
`${REPO_ROOT}`, `${BUILD_DIR}`, and `${VARIANT_OUTPUT}`. Start with `-DryRun`
to validate all paths and inspect the resolved commands:

```powershell
.\scripts\validation\run_depth_pose_fusion_ablation.ps1 `
  -Config E:\path\to\hyb2_depth_pose_ablation.json `
  -DryRun
```

Recommended variants are `current_pose_current_depth`,
`reference_pose_current_depth`, `current_pose_reference_depth`, and
`reference_pose_reference_depth`. The runner deliberately does not infer or
substitute third-party results; every artifact must be declared explicitly.

`analyze_depth_pose_alignment.py` is the read-only precursor to the pose A/B.
It reprojects PlaScan depth into the source views declared by
`mvs_manifest.json`, estimates bounded per-camera SE(3) corrections, and writes
only a diagnostic JSON. It does not edit the project, manifest, cameras, or
depth files:

```powershell
.\.venv\Scripts\python.exe `
  scripts\validation\analyze_depth_pose_alignment.py `
  --mvs-manifest E:\path\to\mvs_output\mvs_manifest.json `
  --output E:\path\to\validation\pose_metrics.json
```

## Historical mesh experiments

The active mesh-quality profiles remain directly in this directory. Superseded
July 2026 trial profiles are preserved under `experiments/2026-07/`; they are
not selected by the baseline runner and must be passed explicitly when replaying
an older experiment.
