# PlaPoint GPU Search, Features, and Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PlaPoint's production GPU brute-force neighbor paths and CPU-staged normal/filter paths with a cached uniform-grid index and device-resident algorithms built on PlaMatrix CUDA primitives.

**Architecture:** A reusable `GpuSpatialIndex` owns sorted finite-point indices and cell ranges for one cloud revision. Radius and KNN queries share this index; small workloads retain the existing brute-force kernel. Normal estimation, refinement, SOR, and RadiusOR consume device query results and PlaMatrix reduction/indexing APIs without downloading intermediate vectors.

**Tech Stack:** C++17, CUDA 13.x, CUB radix sort/run-length encode, PlaMatrix Release 1 APIs, OpenMP CPU references, GoogleTest, CMake.

---

## File Map

- Modify `include/plapoint/core/point_cloud.h`: expose monotonic point revision invalidation.
- Create `include/plapoint/gpu/spatial_index.h`: index, workspace, build, radius, and KNN APIs.
- Create `src/spatial_index_gpu.cu`: grid construction and indexed queries.
- Modify `include/plapoint/gpu/knn.h` and `src/knn_gpu.cu`: backend selection and shared result semantics.
- Modify `include/plapoint/search/kdtree.h`: cache and use the GPU spatial index.
- Create `include/plapoint/gpu/normal_estimation.h` and `src/normal_estimation_gpu.cu`.
- Create `include/plapoint/gpu/normal_refinement.h` and `src/normal_refinement_gpu.cu`.
- Modify `include/plapoint/features/normal_estimation.h` and `normal_refinement.h`.
- Modify `include/plapoint/gpu/filter_indices.h` and `src/filter_indices_gpu.cu`.
- Modify `include/plapoint/filters/statistical_outlier_removal.h` and `radius_outlier_removal.h`.
- Modify `include/plapoint/filters/preprocessing.h`: report actual backend and fallback reason.
- Modify `src/CMakeLists.txt` and `test/CMakeLists.txt`.
- Create `test/unit/search/spatial_index_gpu_test.cpp`.
- Create `test/unit/features/normal_estimation_gpu_test.cpp`.
- Create `test/unit/features/normal_refinement_gpu_test.cpp`.
- Extend `test/unit/filters/filter_indices_gpu_test.cpp` and preprocessing API tests.
- Modify GPU benchmarks and `README.md`; create `docs/gpu-search.md`.

### Task 1: Add point revision tracking and cache identity

**Files:**
- Modify: `test/unit/core/point_cloud_test.cpp`
- Modify: `include/plapoint/core/point_cloud.h`

- [ ] **Step 1: Write failing revision tests**

Exercise this contract:

```cpp
const auto initial = cloud.pointsRevision();
const auto& read_only = std::as_const(cloud).points();
EXPECT_EQ(cloud.pointsRevision(), initial);
auto& mutable_points = cloud.points();
EXPECT_GT(cloud.pointsRevision(), initial);
mutable_points(0, 0) = 3.0f;
```

Verify `setPoints`, move assignment, and replacing a cloud advance or replace cache identity. Read-only access and point-wise attribute mutations must not invalidate the point-position revision.

- [ ] **Step 2: Verify RED**

Run the PointCloud focused tests. Expected: `pointsRevision()` does not exist.

- [ ] **Step 3: Implement revision tracking**

Add a monotonic `std::uint64_t _points_revision` initialized to one. Increment it before returning mutable `points()` and after successful point replacement. Expose:

```cpp
std::uint64_t pointsRevision() const noexcept { return _points_revision; }
```

Handle overflow by wrapping zero to one; zero remains reserved for an unbound cache.

- [ ] **Step 4: Verify GREEN**

Run core, attribute, transfer, and point-view tests. Expected: pass.

### Task 2: Define and build the GPU uniform-grid index

**Files:**
- Create: `test/unit/search/spatial_index_gpu_test.cpp`
- Create: `include/plapoint/gpu/spatial_index.h`
- Create: `src/spatial_index_gpu.cu`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing construction tests**

Define this public shape:

```cpp
template <typename Scalar>
class GpuSpatialIndex {
public:
    void build(const PointCloud<Scalar, Device::GPU>& cloud,
               Scalar cell_size,
               cudaStream_t stream = 0);
    bool matches(const PointCloud<Scalar, Device::GPU>& cloud,
                 Scalar cell_size) const noexcept;
    int finitePointCount() const noexcept;
    int cellCount() const noexcept;
    Scalar cellSize() const noexcept;
};
```

Test empty input, invalid cell sizes, one point, repeated points, negative coordinates, cell-boundary values, non-finite points, deterministic rebuilds, and revision-based cache invalidation.

- [ ] **Step 2: Verify RED**

Build CUDA tests and run `ctest -R SpatialIndexGpu`. Expected: missing header/API.

- [ ] **Step 3: Implement deterministic grid construction**

The build stages are:

1. finite-mask and cell-coordinate kernel;
2. PlaMatrix mask compaction of source indices;
3. 64-bit Morton/hash key generation with overflow-checked origin normalization;
4. CUB stable radix sort of `(key, source_index)`;
5. CUB run-length encoding for unique keys and counts;
6. PlaMatrix exclusive scan for cell offsets.

Store cloud data pointer, point count, revision, scalar cell size, sorted indices, unique keys, offsets, and counts. Allocate replacements before swapping them into the live index.

- [ ] **Step 4: Verify GREEN**

Run construction tests for float/double and repeat each build twice. Expected: identical sorted keys and indices.

### Task 3: Implement indexed radius queries

**Files:**
- Modify: `test/unit/search/spatial_index_gpu_test.cpp`
- Modify: `include/plapoint/gpu/spatial_index.h`
- Modify: `src/spatial_index_gpu.cu`

- [ ] **Step 1: Write failing radius-query tests**

Expose count and bounded-neighbor forms:

```cpp
auto counts = index.radiusCountAsync(queries, radius, max_count, workspace, stream);
auto neighbors = index.radiusSearchAsync(queries, radius, max_neighbors, workspace, stream);
```

Verify against CPU brute force for random clouds, exact-radius equality, repeated points, query points outside bounds, non-finite queries, count saturation, and `(distance, source_index)` order.

- [ ] **Step 2: Verify RED**

Run `SpatialIndexGpu` tests. Expected: query methods absent.

- [ ] **Step 3: Implement radius query kernels**

Enumerate cells intersecting the query sphere's axis-aligned cell range, binary-search unique cell keys, reject candidate axes before squared distance, and maintain a bounded ordered result. Use squared distances internally and include points whose distance equals radius.

- [ ] **Step 4: Verify GREEN**

Run radius tests at query counts 1, 31, 32, 33, 1000 and both scalar types. Expected: CPU-equivalent results.

### Task 4: Implement adaptive indexed KNN

**Files:**
- Modify: `test/unit/search/spatial_index_gpu_test.cpp`
- Modify: `include/plapoint/gpu/spatial_index.h`
- Modify: `src/spatial_index_gpu.cu`
- Modify: `include/plapoint/gpu/knn.h`
- Modify: `src/knn_gpu.cu`
- Modify: `include/plapoint/search/kdtree.h`

- [ ] **Step 1: Write failing KNN and cache tests**

Test `k` from 1 through 32, sparse and dense grids, queries on cell faces, deterministic equal-distance ordering, pathological cell occupancy, mutable cloud invalidation, and equality with CPU KdTree/brute force.

Require backend reporting:

```cpp
enum class GpuNeighborBackend { BruteForce, UniformGrid };
GpuNeighborBackend lastNeighborBackend() const noexcept;
```

- [ ] **Step 2: Verify RED**

Run KdTree GPU tests. Expected: no indexed backend/reporting.

- [ ] **Step 3: Implement shell-expanding KNN**

Expand Chebyshev cell shells around each query. Stop when at least `k` candidates exist and the next shell's minimum possible distance exceeds the current kth result. For KNN without a caller radius, estimate cell size as the largest finite bounding-box extent divided by `cbrt(finite_point_count)`, clamped to a positive scalar epsilon; degenerate clouds use the smallest positive representable spacing derived from their finite coordinates. Fall back to existing brute force when coordinate quantization exceeds the 64-bit key range, the grid is pathological, `k > 32`, or the benchmark-derived small-work threshold selects it.

Cache one `GpuSpatialIndex` in `KdTree<Scalar, GPU>` and rebuild only when cloud pointer, point count, revision, or selected cell size changes.

- [ ] **Step 4: Verify GREEN**

Run all KdTree CPU/GPU and validation tests. Expected: deterministic agreement.

### Task 5: Implement device-resident normal estimation

**Files:**
- Create: `test/unit/features/normal_estimation_gpu_test.cpp`
- Create: `include/plapoint/gpu/normal_estimation.h`
- Create: `src/normal_estimation_gpu.cu`
- Modify: `include/plapoint/features/normal_estimation.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing GPU normal tests**

Test planes, rotated planes, spheres, duplicate points, fewer than three usable neighbors, non-finite points, deterministic signs, output reuse, stream execution, and CPU/GPU angular agreement.

Define:

```cpp
template <typename Scalar>
void estimateNormalsAsync(
    const PointCloud<Scalar, Device::GPU>& cloud,
    const GpuSpatialIndex<Scalar>& index,
    int k,
    DenseMatrix<Scalar, Device::GPU>& normals,
    NormalEstimationGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream);
```

- [ ] **Step 2: Verify RED**

Run `NormalEstimationGpu` tests. Expected: API absent.

- [ ] **Step 3: Implement the GPU pipeline**

Query KNN indices, launch one covariance accumulation per point into compact `[xx,xy,xz,yy,yz,zz]`, call PlaMatrix `symmetricEigh3x3Batched`, select the first eigenvector, normalize it, and apply deterministic sign selection. Invalid/degenerate neighborhoods produce zero normals, matching documented CPU behavior.

- [ ] **Step 4: Route the existing high-level API**

For GPU clouds, replace `pointsCpu()` and per-point CPU SVD with the new path. CPU template behavior stays unchanged.

- [ ] **Step 5: Verify GREEN**

Run normal estimation tests, CPU-only tests, and GPU integration tests. Expected: no host staging in the GPU source contract and numerical agreement.

### Task 6: Implement GPU normal refinement and orientation

**Files:**
- Create: `test/unit/features/normal_refinement_gpu_test.cpp`
- Create: `include/plapoint/gpu/normal_refinement.h`
- Create: `src/normal_refinement_gpu.cu`
- Modify: `include/plapoint/features/normal_refinement.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing smoothing/orientation tests**

Cover KNN averaging, zero-length sums, viewpoint flips, in-place cloud updates, attributes remaining unchanged, output/workspace reuse, and non-default streams.

- [ ] **Step 2: Verify RED**

Run focused tests. Expected: GPU implementation still stages through CPU.

- [ ] **Step 3: Implement device kernels**

Gather KNN normal components, accumulate in double, normalize, and write a scratch normal matrix before swapping into the cloud. Implement viewpoint orientation as one thread per point with finite checks.

- [ ] **Step 4: Verify GREEN**

Run CPU/GPU refinement and point-cloud attribute tests. Expected: pass.

### Task 7: Convert RadiusOR and SOR to indexed, device-resident paths

**Files:**
- Modify: `test/unit/filters/filter_indices_gpu_test.cpp`
- Modify: `test/unit/filters/statistical_outlier_removal_test.cpp`
- Modify: `test/unit/filters/radius_outlier_removal_test.cpp`
- Modify: `include/plapoint/gpu/filter_indices.h`
- Modify: `src/filter_indices_gpu.cu`
- Modify: `include/plapoint/filters/statistical_outlier_removal.h`
- Modify: `include/plapoint/filters/radius_outlier_removal.h`

- [ ] **Step 1: Write failing backend and no-staging tests**

Test CPU/GPU output equality, removed indices, finite/non-finite mixtures, `meanK` limits, radius equality, duplicate points, every point-wise attribute, scalar fields, stable ordering, and cached index reuse.

- [ ] **Step 2: Verify RED**

Run filter tests. Expected: backend remains brute force and SOR downloads mean-distance arrays.

- [ ] **Step 3: Implement RadiusOR on indexed counts**

Use `radiusCountAsync` with early saturation at `min_neighbors`, produce a keep mask, and call PlaMatrix compact/gather for positions and all point-wise attributes.

- [ ] **Step 4: Implement SOR with GPU reductions**

Use indexed KNN, compute finite mean distances on GPU, reduce sum/count and squared deviation through PlaMatrix, launch threshold masking, then compact/gather. Transfer only the scalar threshold when the synchronous compatibility API requires a host report.

- [ ] **Step 5: Verify GREEN**

Run all filters, preprocessing, point-cloud attribute, and CPU/GPU consistency tests. Expected: pass.

### Task 8: Calibrate Auto selection, benchmarks, and documentation

**Files:**
- Modify: `include/plapoint/filters/preprocessing.h`
- Modify: `benchmarks/plapoint_benchmark_gpu.h`
- Modify: `benchmarks/plapoint_benchmark_cpu.h`
- Modify: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Create: `docs/gpu-search.md`

- [ ] **Step 1: Add failing report/benchmark registration tests**

Require `ProcessingReport` to expose requested device, actual device, neighbor backend, fallback flag, and fallback reason. Add benchmark rows for brute-force/indexed KNN, radius count, normal estimation, normal smoothing, SOR, and RadiusOR.

- [ ] **Step 2: Verify RED**

Run preprocessing API and benchmark registration tests. Expected: fields/rows absent.

- [ ] **Step 3: Implement benchmark-derived selection policy**

Keep thresholds in one documented policy helper. Small query workloads use CPU or brute force; sufficiently large finite-radius workloads use the grid. Allocation failure in `Auto` records the reason and retries CPU; explicit GPU rethrows.

- [ ] **Step 4: Update documentation**

Document index complexity, `k <= 32`, deterministic ordering, cache invalidation, non-finite handling, Auto semantics, workspace lifetimes, and known pathological-grid fallback.

- [ ] **Step 5: Run full Release 2 validation**

Run:

```powershell
cmake --build build-cpu --parallel 8
ctest --test-dir build-cpu --output-on-failure
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda --output-on-failure
build-cuda\benchmarks\plapoint_benchmarks.exe --points 1000 --iterations 3
build-cuda\benchmarks\plapoint_benchmarks.exe --points 100000 --iterations 3
```

Expected: all tests pass; every new benchmark row reports a valid status and finite measured timing when CUDA is usable.
