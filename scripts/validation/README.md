# Third-party reconstruction diagnostics

These scripts build reproducible COLMAP/OpenMVS comparison inputs without
overwriting a PlaScan project or its current products.

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
.\.venv\Scripts\python.exe scripts\validation\compare_mesh_geometry.py --help
```

Python dependencies used by the comparison scripts are `numpy`, `Pillow`,
`scipy`, `rtree`, and `trimesh`.

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

The accepted MC33/OpenMesh validation profiles are:

- Dino: `mesh_gui_detail_openmesh_dino_170k_v5.json` with
  `mesh_quality_scenes_gui_detail_openmesh_dino_v5.json`.
- Temple: `mesh_mc33_openmesh_temple_v4.json` with
  `mesh_quality_scenes_mc33_openmesh_temple_v4.json`.

The frozen 2026-07-27 runs are
`task40_gui_detail_openmesh_dino_v5` and `task40_temple_openmesh_v8`.
Dino finished with 169,976 faces, 4 boundary edges, one component, a
20.86-degree adjacent-normal median, and 0.0008787 Chamfer-L1. Temple finished
with 60,060 faces, no boundary edges, one component, and an 11.59-degree
adjacent-normal median. Both strict gates passed; inspect each scene's
`contact_sheet.png` before accepting later parameter changes.

`mesh_gui_ultra_openmesh.json` is the GUI-equivalent smoke profile: it only
sets the visible Ultra/240,000-face controls and leaves all TSDF advanced
options automatic. Its frozen supplemental runs are
`analysis/gui_ultra_openmesh_dino_240k` and
`analysis/gui_ultra_openmesh_temple_240k`. Dino produced 239,971 faces,
7 boundary edges, one component, a 16.79-degree normal median, and 0.0008693
Chamfer-L1 against the registered Metashape mesh. Temple produced 239,949
faces, 45 boundary edges, one component, and an 8.55-degree normal median.
These runs verify that GUI defaults select MC33, adaptive TGV, the six-voxel
geometry profile, and OpenMesh rather than only the explicit validation JSON.
