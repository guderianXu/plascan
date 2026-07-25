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
