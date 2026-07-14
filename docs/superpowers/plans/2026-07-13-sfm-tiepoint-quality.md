# SfM Tie Point Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce the configured tie-point limit for unknown-camera incremental SfM at the multiview-track level and introduce graded two-view-track quality gates.

**Architecture:** Add a focused correspondence-track thinning unit that reuses `MultiViewTrackBuilder`, then retain only original verified edges belonging to selected tracks before `CorrespondenceGraph::buildCorrespondences()`. Keep quality reporting independent and expose advisory versus blocking thresholds in `SfmQualityReport`.

**Tech Stack:** C++17, Qt6 JSON types, OpenCV-backed SfM core, GoogleTest, CMake/CTest.

## Global Constraints

- Preserve existing pairwise geometric verification; never synthesize unverified pair edges.
- Prefer longer tracks, then higher confidence, then stable input order.
- Do not add a compatibility field for the old known-pose-only option names.
- Use Chinese comments for non-obvious photogrammetric behavior.
- Build with `cmake --build build -j 24`.

---

### Task 1: Correspondence Graph Track Retention

**Files:**
- Modify: `src/core/sfm/graph/CorrespondenceGraph.h`
- Modify: `src/core/sfm/graph/CorrespondenceGraph.cpp`
- Test: `tests/test_correspondence_graph.cpp`

**Interfaces:**
- Produces: `std::vector<ImagePair> imagePairs() const`
- Produces: `std::size_t retainMatchesInTracks(const std::vector<Track> &tracks)` returning removed match count.

- [ ] Add a failing test with one accepted three-view track and one rejected two-view track.
- [ ] Run the focused test and verify the new API is missing.
- [ ] Implement stable pair enumeration and original-edge retention.
- [ ] Run the focused test and existing correspondence graph tests.

### Task 2: Unknown-Camera Input Track Thinning

**Files:**
- Create: `src/core/sfm/tracks/CorrespondenceTrackThinner.h`
- Create: `src/core/sfm/tracks/CorrespondenceTrackThinner.cpp`
- Modify: `src/core/sfm/CMakeLists.txt`
- Modify: `src/core/sfm/pipeline/IncrementalSfm.h`
- Modify: `src/core/sfm/pipeline/IncrementalSfm.cpp`
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Test: `tests/test_multiview_track_builder.cpp`
- Test: `tests/test_sfm_pipeline.cpp`

**Interfaces:**
- Consumes: `CorrespondenceGraph`, `SfmReconstruction`, and `CorrespondenceTrackThinningOptions`.
- Produces: `CorrespondenceTrackThinningResult thinCorrespondenceTracks(...)` with input, retained, pruned, and retained-match statistics.

- [ ] Add a failing test proving long tracks win under a per-image limit.
- [ ] Add a failing SfM-level test proving unknown-camera input invokes the limit.
- [ ] Run both focused tests and verify expected failures.
- [ ] Implement the focused thinning unit and invoke it before correspondence indexing.
- [ ] Rename known-pose-only option fields to generic track-limit names and update callers.
- [ ] Run focused and SfM pipeline tests.

### Task 3: Graded Sparse Quality Gate

**Files:**
- Modify: `src/core/sfm/quality/SfmQualityReport.h`
- Modify: `src/core/sfm/quality/SfmQualityReport.cpp`
- Test: `tests/test_sfm_quality_report.cpp`

**Interfaces:**
- Produces: `advisories` in the quality-gate JSON.
- Uses: advisory threshold `0.70` and blocking threshold `0.85` by default.

- [ ] Add failing tests for advisory-only and blocking two-view ratios.
- [ ] Run the quality report test and verify the expected failure.
- [ ] Implement advisory/blocking classification and stable JSON output.
- [ ] Run all quality report tests.

### Task 4: Regression Verification and Documentation

**Files:**
- Modify: `src/core/sfm/README.md`
- Modify: `src/core/aerial_triangulation/README.md`
- Modify: `docs/PROJECT_ARCHITECTURE.md`

- [ ] Build relevant targets with `cmake --build build -j 24`.
- [ ] Run `ctest --test-dir build --output-on-failure -R "CorrespondenceGraph|MultiViewTrack|SfmQuality|SfmPipeline|AerialTriangulation"`.
- [ ] Run the current `small_test` CLI workflow with tie-point limit 4000.
- [ ] Verify 9/9 registration, per-image track limit, quality JSON, RMS, runtime, and GPU memory behavior.
- [ ] Update module documentation with the verified behavior and remaining risks.
