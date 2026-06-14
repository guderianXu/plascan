# Sparse Reconstruction Quality Design

Date: 2026-06-14

## Problem

The current GUI can present a pairwise triangulation result as if it were a normal aerial sparse
point cloud. On the `agisoft_aerial_gcps` project, the generated
`sparse_cloud_points.json` contains 57,026 exported points, and every exported point has
`track_len = 2`. The visual result is a line-like, noisy cloud rather than a coherent nadir UAV
surface reconstruction.

The root cause is architectural, not just a display issue. `BaInputBuilder` reads each match
sidecar and creates a new `BATrack` per matched point pair with exactly two observations.
`InitialSparsePointCloudTriangulator` accepts those two-view tracks and `TriangulationService`
exports them as `sparse_cloud.ply`. This is useful as a quick diagnostic preview, but it is not a
final SfM sparse reconstruction and should not feed DEM, DOM, dense reconstruction, or model
generation by default.

## Goals

- Preserve the fast pairwise triangulation workflow as a diagnostic preview.
- Stop treating pairwise triangulation preview clouds as final aerial sparse point clouds.
- Make formal sparse reconstruction results come from multi-view tracks and bundle adjustment.
- Add machine-readable quality metadata so GUI flows can warn, reject, or select results safely.
- Upgrade match sidecars so future sparse reconstruction can build stable multi-view tracks from
  original feature indices instead of only pixel coordinates.

## Non-Goals

- Remove the existing triangulation dialog or quick camera/match diagnostic workflow.
- Rewrite the whole incremental SfM pipeline in one pass.
- Delete or regenerate user project outputs under `testData/`, `.plascan` projects, or existing
  result directories.
- Claim dense reconstruction, DEM, DOM, or mesh quality is fixed until the formal sparse result is
  verified first.

## Proposed Approach

Use a staged design.

Stage 1 fixes semantics and safety. Pairwise triangulation remains available, but its metadata and
GUI label identify it as a preview. Downstream product workflows default to formal SfM/BA results
and warn or reject pairwise preview clouds.

Stage 2 fixes sparse reconstruction quality. Match sidecars gain original feature indices, a
multi-view track builder creates stable tracks across image pairs, and formal sparse cloud export
records track-length and BA quality statistics.

This is preferred over simply increasing triangulation thresholds. Filtering two-view points can
remove some outliers, but it cannot turn independent pairwise intersections into a multi-view SfM
cloud.

## Result Types

Every aerial triangulation style result should carry a `result_kind` field.

- `pairwise_triangulation_preview`: output from the existing two-view triangulation service. It is
  allowed for inspection and manual diagnostics.
- `sfm_sparse_reconstruction`: output from formal SfM/known-camera triangulation plus bundle
  adjustment. This is the default input for dense reconstruction, DEM, DOM, and model generation.
- `sparse_postprocess`: output from sparse point cleanup. It must retain provenance to the source
  result and should only be considered production-ready when its source is an
  `sfm_sparse_reconstruction`.

Existing legacy records without `result_kind` are treated conservatively. If their points sidecar
shows only two-view tracks or no BA fields, the GUI should classify them as preview/unknown and
avoid using them as final products without an explicit user confirmation.

## Quality Metadata

Sparse result records and `sparse_cloud_points.json` should include:

- `camera_count`
- `point_count`
- `track_len_histogram`
- `two_view_ratio`
- `median_track_len`
- `mean_reproj_px`
- `median_reproj_px`
- `min_tri_angle_deg`
- `ba_applied`
- `result_kind`
- `source_result_kind`
- `source_result_ref`, stored as the source result ID when available, otherwise the source output
  directory path

The first implementation can compute these fields from existing point sidecars and service result
objects. Later implementations should compute them directly from the SfM reconstruction data model
before export.

## Quality Gates

GUI workflows should use the metadata to decide whether a result is acceptable.

Formal downstream products require:

- `result_kind == sfm_sparse_reconstruction` or a postprocess result whose source is
  `sfm_sparse_reconstruction`.
- `ba_applied == true`.
- `camera_count >= 2`.
- `point_count > 0`.
- `two_view_ratio < 1.0` for formal downstream use. `two_view_ratio >= 0.8` is allowed only with a
  warning because it indicates weak multi-view support.
- A non-empty `track_len_histogram` with at least some tracks longer than 2 for UAV multi-view
  datasets.

If only preview clouds exist, the GUI should show a clear Chinese warning such as:

`当前结果是两视初始三角化预览云，不能作为正式航测稀疏点云。请先运行三维重建/空三。`

The warning should name the selected result and the failing quality fields when possible.

## Match Sidecar v2

The current LightGlue sidecar stores `matched_points0` and `matched_points1`, but not the original
feature indices. Multi-view track building needs stable feature identity, so new sidecars should
also store:

- `feature_format_version`
- `matched_indices0`
- `matched_indices1`
- `sp0_path`
- `sp1_path`
- `image0_path`
- `image1_path`

The existing point arrays remain for match visualization. Index arrays become the authoritative
input for track construction.

Backward compatibility:

- Sidecars without indices can still feed `pairwise_triangulation_preview`.
- Sidecars without indices cannot feed formal multi-view reconstruction unless a carefully tested
  fallback is added later.
- The GUI and logs should state when old sidecars prevent formal sparse reconstruction.

## Multi-View Track Builder

Add a focused core component, tentatively `MultiViewTrackBuilder`.

Inputs:

- Project image list and stable image IDs.
- Per-image feature files.
- Pairwise match sidecars with `matched_indices0` and `matched_indices1`.

Process:

- Represent each observation as `(image_id, keypoint_index)`.
- Add graph edges from pairwise matches.
- Extract connected components.
- Reject components that contain more than one keypoint from the same image. Component splitting is
  a later optimization and is out of scope for the first implementation.
- Build one `BATrack` per valid component with observations from all participating images.
- Keep track provenance and statistics for reports.

Outputs:

- `std::vector<BATrack>` or an equivalent core track type.
- Track statistics: total components, accepted tracks, rejected conflicts, track-length histogram,
  and image coverage.

The builder should be independent of Qt widgets. It can depend on core feature and match IO types,
but GUI code should call it through services/runners.

## Data Flow

Formal sparse reconstruction:

1. Feature extraction writes feature files with stable keypoint ordering.
2. Feature matching writes sidecar v2 files with point coordinates and feature indices.
3. `MultiViewTrackBuilder` creates multi-view tracks.
4. Known-camera triangulation or incremental SfM initializes 3D points.
5. Bundle adjustment optimizes cameras and points.
6. Formal sparse cloud export writes `sparse_cloud.ply`, `sparse_cloud_points.json`, and quality
   metadata.
7. Downstream workflows select this result by default.

Pairwise preview:

1. Existing `TriangulationService` reads sidecars.
2. It creates two-view tracks and exports a preview cloud.
3. The result is tagged as `pairwise_triangulation_preview`.
4. GUI allows inspection but warns before using it downstream.

## GUI Behavior

- Rename or relabel the current initial triangulation action to communicate that it is an initial
  preview, not final sparse reconstruction.
- In the project tree and result selectors, show result kind and quality status.
- When opening a sparse cloud whose `two_view_ratio` is high, show a non-blocking warning in the log
  panel and a visible status message near the result detail area.
- Dense reconstruction, DEM, DOM, and model generation selectors should prefer formal SfM results.
- If a preview cloud is selected explicitly for a downstream product, require an explicit warning
  confirmation and keep the default action pointed at formal SfM.

## Testing Strategy

Unit tests:

- `MultiViewTrackBuilder` merges A-B, A-C, and B-C matches into 3-view tracks.
- It rejects components containing duplicate observations from the same image.
- It reports a correct track-length histogram.

Service tests:

- `TriangulationService` writes `result_kind = pairwise_triangulation_preview` and quality metadata.
- Formal SfM export writes `result_kind = sfm_sparse_reconstruction`, `ba_applied = true`, and
  non-empty quality metadata.
- Old sidecars without indices are accepted only for pairwise preview.

GUI/project tests:

- Result selectors exclude preview clouds by default for DEM, DOM, dense reconstruction, and model
  generation.
- If only preview clouds exist, the GUI explains why formal downstream processing is blocked.
- Legacy records without `result_kind` are classified conservatively.

Dataset verification:

- Run the `agisoft_aerial_gcps` dataset through the formal workflow.
- Verify the formal sparse sidecar is not all `track_len = 2`.
- Compare point cloud shape visually against the current pairwise preview failure case.
- Record command lines, result paths, point counts, and track-length histograms in the final report.

## Phasing

### Phase 1: Semantics and Safety

- Add result-kind constants and quality metadata helpers.
- Tag pairwise triangulation outputs as preview.
- Add conservative legacy classification.
- Update downstream result selection to avoid preview clouds by default.
- Add GUI warnings and focused tests.

Acceptance criteria:

- The current two-view cloud is no longer presented as a normal final UAV sparse cloud.
- Downstream products do not silently consume pairwise preview clouds.
- Existing preview functionality remains available.

### Phase 2: Sidecar v2 and Track Builder

- Extend C++/Python/LightGlue match exporters to write matched feature indices.
- Add `MultiViewTrackBuilder`.
- Add unit tests with synthetic multi-view matches and conflict cases.
- Wire the builder into formal SfM/known-camera reconstruction.

Acceptance criteria:

- Multi-view tracks are built from sidecar v2 indices.
- Conflicting tracks are rejected or split deterministically.
- Formal sparse reconstruction has meaningful track-length statistics.

### Phase 3: Full Dataset Validation

- Run feature extraction, matching, formal SfM, and sparse export on
  `E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps`.
- Verify CUDA selection remains honored for feature extraction/matching where supported.
- Inspect the sparse cloud and quality report.
- Document any remaining reconstruction issues separately from GUI semantics.

Acceptance criteria:

- The formal sparse cloud no longer consists entirely of two-view tracks.
- GUI selects the formal result for downstream products.
- The final report explains any remaining quality limitations with concrete statistics.

## Open Decisions

- Exact threshold for warning or rejecting high `two_view_ratio` should start conservative and be
  made visible in settings later if needed.
- Component conflict handling starts with deterministic rejection. Splitting conflict components is
  not part of this design and should get its own spec if needed later.
- Existing no-index sidecars should not be promoted to formal reconstruction unless a robust
  coordinate-based fallback is designed and tested separately.
