# ETH3D depth evaluator

This package evaluates a depth prediction made for ETH3D's official
undistorted DSLR images against the raw ground-truth depth maps. It implements
COLMAP's `THIN_PRISM_FISHEYE` model directly and inverse-warps the raw depth
into the official `PINHOLE` camera domain.

The mapping treats each raw array element `[row, column]` as a pixel center at
COLMAP coordinate `(column + 0.5, row + 0.5)`. It uses a safeguarded Newton
inversion with a strict forward reprojection check, projects the recovered ray
through the official undistorted camera, then subtracts the same half-pixel
before deterministic target-raster placement.

Invalid ETH3D values (`NaN`, positive or negative infinity, zero, and negative
finite values) remain invalid. The remap forward-projects only measured raw
samples and never fills holes. A measurement is rejected when any valid
four-connected neighbor makes the local relative depth spread exceed the
configured limit. This avoids blending or carrying samples across silhouettes
and occlusion boundaries. If multiple raw samples round to the same target
pixel, the one projected closest to that target pixel center wins. The report
includes the resulting target-center quantization-distance distribution.

Missing ETH3D laser depth means "unobserved", not "known empty". Predictions
outside the remapped ground-truth support are counted for diagnostics but are
not treated as false positives. Coverage is measured only over valid remapped
ground-truth samples.

Run from the repository root with the repository Python environment:

```bash
.venv/bin/python -m scripts.validation.plascan_eth3d_depth_eval \
  --raw-depth /data/office/ground_truth_depth/dslr_images/DSC_0219.JPG \
  --raw-cameras /data/office/dslr_calibration_jpg/cameras.txt \
  --undistorted-cameras /data/office/dslr_calibration_undistorted/cameras.txt \
  --prediction /runs/office/depth_0.bin \
  --prediction-format plascan-fast-matrix \
  --output build/tmp/eth3d-office/DSC_0219.json \
  --remapped-ground-truth-output build/tmp/eth3d-office/DSC_0219.npy
```

The command above preserves the original contract: without a prediction
manifest, the prediction must have the same dimensions and PINHOLE camera as
the official undistorted image. The report structure and remap behavior are
unchanged in this mode.

PlaScan may retain the final pyramid level at its native depth-grid size. To
evaluate that artifact directly, select its completed frame record from the
workspace manifest:

```bash
.venv/bin/python -m scripts.validation.plascan_eth3d_depth_eval \
  --raw-depth /data/office/ground_truth_depth/dslr_images/DSC_0219.JPG \
  --raw-cameras /data/office/dslr_calibration_jpg/cameras.txt \
  --undistorted-cameras /data/office/dslr_calibration_undistorted/cameras.txt \
  --undistorted-images /data/office/dslr_calibration_undistorted/images.txt \
  --prediction /runs/office/mvs/raw/depth_0.bin \
  --prediction-format plascan-fast-matrix \
  --prediction-manifest /runs/office/mvs/mvs_manifest.json \
  --prediction-ref-index 0 \
  --output build/tmp/eth3d-office/DSC_0219-native.json
```

`--prediction-manifest`, `--prediction-ref-index`, and
`--undistorted-images` must be supplied together. The evaluator requires a
`plascan.mvs.workspace.v2` manifest, a single matching completed frame,
matching root/frame algorithm revisions and config hashes, positive
`grid_width`/`grid_height`, and a finite undistorted PINHOLE `camera_model`.
The prediction path must equal the selected frame's `raw_depth_path`; the raw
GT and `ref_image` basenames must match; the referenced image must exist and is
fingerprinted independently from the raw GT; and the manifest pose must agree
with the matching official COLMAP `images.txt` record. That record's
`camera_id` must equal the selected official PINHOLE camera ID. The prediction
is read at that grid size, and raw ETH3D measurements are projected directly
into that camera instead of being remapped to full size first.

COLMAP stores the first pixel center at `(0.5, 0.5)`, while PlaScan/OpenCV
stores array element `[0,0]` at `(0,0)`. The manifest camera is therefore
converted back to COLMAP coordinates by adding `0.5` to `cx/cy` before any
projection. For each axis with
`scale = prediction_size / official_size`, strict validation requires

```text
f' = f * scale
c'_COLMAP = c_COLMAP * scale
c'_manifest = c_COLMAP * scale - 0.5
```

with a maximum absolute intrinsic residual of `1e-6` pixels. This applies
independently to odd widths and heights. A missing field, ambiguous/missing
reference index, unfinished record, distorted camera, path/name/pose mismatch,
or camera mismatch is a hard error rather than a fallback to the official
full-size camera.

For algorithm revision 40 and later, each frame must contain a boolean
`effective_native_final_depth_grid` and a complete
`pixel_domain_diagnostics` object. The evaluator requires the configured and
effective domains, requested/effective native-grid flags, raster and grid
dimensions, per-axis scales, geometric-mean linear scale, area scale,
`grid_matches_raster`, and the parameter-audit object. It verifies
`scale_x=grid_width/raster_width`,
`scale_y=grid_height/raster_height`, `area_scale=scale_x*scale_y`, and
`linear_scale=sqrt(area_scale)`. The diagnostic raster must equal the selected
official camera, its grid must equal the frame grid, and its effective flag
must equal the frame field. An effective native-grid flag always requires the
requested flag, even when the final grid happens to match the raster. A reduced
grid additionally requires both requested and effective flags to be true.
Revision-39 manifests remain
supported only for a full-size official grid; missing revision-40 pixel-domain
fields are not inferred. Accepted and fusion-eligible frames are the default
publishable scope; rejected or validation-only frames require
`--allow-non-publishable-frame` and are prominently marked diagnostic in the
report.

For causal quality-gate audits, `mvs_depth_reprocess_cli` can capture selected
references at four independent boundaries: `patchmatch_output`,
`cross_view_consistency`, `confidence_postprocess`, and `final_admission`.
Evaluate one captured depth with the same strict camera/pose/GT contract by
adding both stage arguments and pointing `--prediction` at that record's depth
artifact:

```bash
.venv/bin/python -m scripts.validation.plascan_eth3d_depth_eval \
  ...normal manifest-aware arguments... \
  --prediction /runs/office/mvs/stage_snapshots/ref_0000/cross_view_consistency_depth.bin \
  --stage-snapshot-manifest /runs/office/mvs/stage_snapshots/manifest.json \
  --stage-snapshot-stage cross_view_consistency
```

The stage manifest must be finalized, complete, explicitly non-authoritative,
and error-free. The evaluator binds its reference, original grid, scaled
PINHOLE camera, and pose back to the workspace frame; fingerprints and parses
the depth, confidence, and binary valid-mask payloads; requires mask support to
equal positive finite depth support; and remaps raw ETH3D GT directly into the
stage raster. Stage reports are always marked `diagnostic_stage_snapshot` and
cannot by themselves authorize publication or a threshold change.

Existing outputs are rejected by default. Use `--overwrite` only when the
replacement is intentional. Neither output may resolve to an input, including
the `ref_image` discovered from the manifest, even with `--overwrite`. Every
final output path is resolved to an absolute path exactly once at startup;
later path preparation and publication never re-resolve a newly introduced
symbolic link. Every input is copied once into a private, suffix-preserving
file snapshot. Camera parsing, manifest parsing, depth reads,
and pose reads use only those immutable snapshots. Reported source paths are
the original resolved paths, while reported sizes and SHA-256 hashes come from
the exact snapshot bytes used for scoring. The manifest snapshot is checked
against the parser result immediately; relative artifact paths are still
resolved from the original manifest directory. Before publication, source
content and file identity (including inode and modification/change timestamps)
are checked again, so detectable change-and-restore (ABA) mutations also abort.

The optional NPY and JSON report are both fully generated in same-directory
temporary files before anything is published. The report contains the hash of
the prepared final NPY and preserves the v1
`remapped_ground_truth_output` field as `string|null`; detailed size/hash
provenance is additive in `remapped_ground_truth_artifact`. Default publication
uses an atomic no-clobber link. For two outputs, the NPY is committed first and
the report last. If no-clobber report publication fails after the NPY commit,
the error names that auditable orphan and leaves it in place; it is not removed
through a check-then-unlink sequence that could race with another process.
Explicit overwrite keeps a recoverable copy of the previous NPY until the new
report commits and restores it on a caught report failure. Two independent
filesystem names cannot form a true crash-atomic transaction: a process or
machine crash in the narrow interval between the two commits can still require
rerunning with `--overwrite`. Publishing the report last prevents a successful
run from exposing a report that names an NPY which was never committed.

The report hashes every input, records both camera models and all remap
settings, separates invalid-value counts, and reports measured-sample coverage
plus absolute and ground-truth-relative depth error distributions. The depth
scalar is not rescaled: raw and undistorted pixels represent the same camera
ray and optical center. Manifest-aware reports additionally hash the manifest
and referenced image, and record the manifest schema, algorithm revision,
config hash, selected frame, official camera/pose, PlaScan manifest camera,
equivalent COLMAP prediction camera, per-axis scale, expected scaled camera,
pixel-domain validation, pose residuals, intrinsic residuals, frame acceptance,
and publishable/diagnostic scope.

Metrics are defined on the selected prediction grid. A reduced grid merges
more raw laser samples into one target pixel than a full-resolution grid, so
their remapped-GT sample sets are not identical. Use these values as
target-raster product quality; do not describe a full/native comparison as a
strict same-raw-ray A/B without an additional common-ray analysis.
