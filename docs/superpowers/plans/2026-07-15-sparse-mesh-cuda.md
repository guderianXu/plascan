# PlaMatrix Sparse and PlaPoint Mesh CUDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production CPU/CUDA sparse operations to PlaMatrix and use them to provide device-resident Marching Cubes, HeightGrid hole filling, and Poisson reconstruction in PlaPoint.

**Architecture:** PlaMatrix exposes matching CPU and CUDA CSR operations plus solver reports and reusable workspaces. PlaPoint introduces explicit device field/grid types, keeps callback-based mesh APIs on CPU, and provides compatibility wrappers that download only final products. Poisson assembles one deterministic sparse system and selects the PlaMatrix CPU or GPU PCG backend.

**Tech Stack:** C++17, CUDA 13.x, cuSPARSE generic API, CUB, OpenMP, PlaMatrix Release 1 primitives, GoogleTest, CMake.

---

## File Map

PlaMatrix:

- Modify `CMakeLists.txt` and `include/plamatrix/core/error_check.h`: cuSPARSE dependency and checked errors.
- Create `include/plamatrix/sparse/sparse_ops.h`: COO conversion, SpMV, SpMM, workspace APIs.
- Create `src/sparse/sparse_ops_cpu.cpp` and `src/sparse/sparse_ops.cu`.
- Create `include/plamatrix/sparse/iterative_solver.h`: solver options/report/workspace.
- Create `src/sparse/iterative_solver_cpu.cpp` and `src/sparse/iterative_solver.cu`.
- Modify `include/plamatrix/sparse/coo_matrix.h`, `csr_matrix.h`, and `plamatrix.h`.
- Create `test/unit/sparse/sparse_ops_test.cpp` and `iterative_solver_test.cpp`.
- Modify benchmark and sparse API documentation.

PlaPoint:

- Create `include/plapoint/gpu/marching_cubes.h` and `src/marching_cubes_gpu.cu`.
- Create `test/unit/mesh/marching_cubes_gpu_test.cpp`.
- Modify `include/plapoint/gpu/height_grid.h`, `src/height_grid_gpu.cu`, and height-grid tests.
- Create `include/plapoint/mesh/poisson_system.h` and `src/poisson_system.cpp`.
- Modify `include/plapoint/mesh/poisson_reconstruction.h` and create `src/poisson_reconstruction.cpp`.
- Create `src/poisson_reconstruction_gpu.cu` when direct device assembly is enabled.
- Extend Poisson unit/quality/validation tests, benchmarks, README, and architecture docs.
- Modify `benchmark/benchmark_cases.cpp`, `benchmark/benchmark_cases.cu`, and `benchmark/benchmark_cases.h` in PlaMatrix.
- Modify `benchmarks/plapoint_benchmark_cpu.h`, `benchmarks/plapoint_benchmark_gpu.h`, and `benchmarks/plapoint_benchmarks.cpp` in PlaPoint.
- Modify PlaPoint `README.md`; create PlaPoint `docs/gpu-mesh.md`.

### Task 1: Add checked cuSPARSE integration

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `include/plamatrix/core/error_check.h`
- Modify: `test/unit/core/device_matrix_test.cpp`

- [ ] **Step 1: Write a failing checked-error test**

Under `PLAMATRIX_WITH_CUDA`, invoke `PLAMATRIX_CHECK_CUSPARSE(CUSPARSE_STATUS_INVALID_VALUE)` and assert the exception contains `cuSPARSE`, the expression, and source location. Under CPU-only mode, verify the macro remains compilable and reports unavailable backend status when invoked.

- [ ] **Step 2: Verify RED**

Build CPU and CUDA tests. Expected: the macro and cuSPARSE link target are absent.

- [ ] **Step 3: Add dependency and checked wrapper**

Link `CUDA::cusparse` only in CUDA builds. Add:

```cpp
inline void cusparseCheck(cusparseStatus_t status,
                          const char* file,
                          int line,
                          const char* expression);
#define PLAMATRIX_CHECK_CUSPARSE(call) \
    ::plamatrix::cusparseCheck((call), __FILE__, __LINE__, #call)
```

Map status through `cusparseGetErrorString` when available and retain a fallback switch for older toolkits.

- [ ] **Step 4: Verify GREEN**

Run focused error tests in CPU/CUDA builds. Expected: pass.

### Task 2: Define deterministic CPU sparse operations

**Files:**
- Create: `test/unit/sparse/sparse_ops_test.cpp`
- Create: `include/plamatrix/sparse/sparse_ops.h`
- Create: `src/sparse/sparse_ops_cpu.cpp`
- Modify: `include/plamatrix/sparse/coo_matrix.h`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing COO/CSR and multiplication tests**

Exercise:

```cpp
auto csr = plamatrix::cooToCsr(rows, cols, row_indices, col_indices, values);
auto y = plamatrix::spmv(csr, x);
auto C = plamatrix::spmm(csr, B);
```

Cover unsorted triplets, duplicate coordinates, empty matrices, empty rows, zero-valued duplicate sums, out-of-range coordinates, dimension mismatch, float/double, and 64-bit indices. Require sorting by row/column and summing duplicates in original insertion order.

- [ ] **Step 2: Verify RED**

Run `ctest -R SparseOps` in CPU build. Expected: APIs absent.

- [ ] **Step 3: Implement CPU conversion and products**

Use stable permutation sorting, one deterministic pass to combine duplicate coordinates, prefix row counts into `Index` offsets, and contiguous CSR loops for SpMV/SpMM. Provide allocating and output-reuse overloads.

- [ ] **Step 4: Route existing `COOMatrix::toCsr()`**

Delegate its CPU path to `cooToCsr` so duplicate semantics are shared. Preserve the existing `add()` API.

- [ ] **Step 5: Verify GREEN**

Run all COO/CSR and SparseOps tests. Expected: pass.

### Task 3: Add device COO-to-CSR, SpMV, and SpMM

**Files:**
- Modify: `test/unit/sparse/sparse_ops_test.cpp`
- Modify: `include/plamatrix/sparse/sparse_ops.h`
- Create: `src/sparse/sparse_ops.cu`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing CUDA sparse tests**

Add device-triplet input overload tests:

```cpp
auto csr_gpu = plamatrix::cooToCsrAsync(
    rows, cols, row_gpu, col_gpu, value_gpu, workspace, stream);
plamatrix::spmvAsync(csr_gpu, x_gpu, y_gpu, workspace, stream);
```

Verify CPU/GPU conversion identity, CSR `toGpu()`/`toCpu()` round trips, duplicate sums, empty matrices, non-default streams, output reuse, workspace reuse, float/double, and explicit unsupported-index-range errors.

- [ ] **Step 2: Verify RED**

Run CUDA SparseOps tests. Expected: device APIs absent.

- [ ] **Step 3: Implement device conversion**

Use stable device sorting by packed `(row, col)` keys, reduce duplicate keys by sum, and construct row offsets. Add explicit `CSRMatrix::toGpu()` and `CSRMatrix::toCpu()` transfers before routing compatibility callers through them. Use cuSPARSE 64-bit descriptors where supported. Keep original device triplet buffers unmodified.

- [ ] **Step 4: Implement cuSPARSE products**

Create `SparseOpsWorkspace` that caches cuSPARSE handle/stream binding, matrix/vector descriptors, dense descriptors, and grow-only temporary memory. Query buffer sizes before launch and synchronize only in synchronous wrappers.

- [ ] **Step 5: Verify GREEN**

Run CPU/CUDA SparseOps tests and no-CUDA behavior tests. Expected: pass.

### Task 4: Add CPU CG and PCG

**Files:**
- Create: `test/unit/sparse/iterative_solver_test.cpp`
- Create: `include/plamatrix/sparse/iterative_solver.h`
- Create: `src/sparse/iterative_solver_cpu.cpp`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing solver tests**

Define:

```cpp
struct IterativeSolverOptions {
    int maxIterations = 1000;
    double relativeTolerance = 1e-6;
    double absoluteTolerance = 0.0;
    bool useJacobiPreconditioner = true;
    bool requireConvergence = false;
};

struct IterativeSolverReport {
    bool converged = false;
    int iterations = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;
};
```

Test SPD diagonal, tridiagonal Poisson, multiple scales, zero RHS, invalid non-square systems, zero diagonal preconditioner, max-iteration non-convergence, and `requireConvergence` exceptions.

- [ ] **Step 2: Verify RED**

Run `ctest -R IterativeSolver` in CPU build. Expected: API absent.

- [ ] **Step 3: Implement deterministic CPU CG/PCG**

Use CSR SpMV and double dot-product accumulation. Stop when `||r|| <= max(absTol, relTol*||r0||)`. Treat an initially converged system as zero iterations. Jacobi uses reciprocal diagonal and rejects missing/near-zero diagonal entries.

- [ ] **Step 4: Verify GREEN**

Run solver and NumPy/SciPy reference tests. Expected: pass.

### Task 5: Add CUDA CG/PCG with reusable workspace

**Files:**
- Modify: `test/unit/sparse/iterative_solver_test.cpp`
- Modify: `include/plamatrix/sparse/iterative_solver.h`
- Create: `src/sparse/iterative_solver.cu`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing CUDA solver tests**

Test CPU/GPU solution and residual agreement, warm workspace reuse, non-default streams, zero-iteration convergence, non-convergence report, and explicit `requireConvergence` failure.

- [ ] **Step 2: Verify RED**

Run CUDA IterativeSolver tests. Expected: GPU overloads absent.

- [ ] **Step 3: Implement CUDA solver**

Use cuSPARSE SpMV, cuBLAS dot/axpy/scal, and device Jacobi application. `IterativeSolverWorkspace` owns all vectors and nested sparse workspace. The adaptive synchronous API copies the convergence scalar on its stream at each iteration. Add `cgFixedIterationsAsync` and `pcgFixedIterationsAsync` that submit exactly the requested iteration count without host convergence checks; `finalizeIterativeSolverReport` reads the residual only after the caller has synchronized.

Expose the fixed-iteration result boundary explicitly:

```cpp
struct AsyncIterativeSolverState {
    int submittedIterations = 0;
    DenseMatrix<double, Device::GPU> initialResidualSquared;
    DenseMatrix<double, Device::GPU> finalResidualSquared;
};

template <typename Scalar>
AsyncIterativeSolverState pcgFixedIterationsAsync(
    const CSRMatrix<Scalar, Device::GPU>& matrix,
    const DenseMatrix<Scalar, Device::GPU>& rhs,
    DenseMatrix<Scalar, Device::GPU>& solution,
    int iterations,
    IterativeSolverWorkspace<Scalar>& workspace,
    cudaStream_t stream);

IterativeSolverReport finalizeIterativeSolverReport(
    const AsyncIterativeSolverState& state,
    const IterativeSolverOptions& options);
```

`cgFixedIterationsAsync` has the same signature without Jacobi use. The caller must synchronize `stream` before finalization; finalization validates that synchronization through a recorded CUDA event and throws if the event is not complete. It computes the report from the two residual scalars and applies `requireConvergence` consistently with the adaptive API.

- [ ] **Step 4: Verify GREEN**

Run CPU/CUDA solver tests at 10, 1,000, and 100,000 unknowns. Expected: convergence and bounded residuals.

### Task 6: Add device-field CUDA Marching Cubes

**Files:**
- Create: `test/unit/mesh/marching_cubes_gpu_test.cpp`
- Create: `include/plapoint/gpu/marching_cubes.h`
- Create: `src/marching_cubes_gpu.cu`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing device-field tests**

Define:

```cpp
template <typename Scalar>
PointCloud<Scalar, Device::GPU> marchingCubes(
    const DenseMatrix<Scalar, Device::GPU>& field,
    int nx, int ny, int nz,
    const Vec3<Scalar>& min_corner,
    const Vec3<Scalar>& max_corner,
    Scalar iso,
    MarchingCubesGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream = 0);
```

Test empty surfaces, one-cell cases, plane, sphere, boundary intersections, invalid field size/bounds/resolution, float/double, deterministic repeated output, and CPU/GPU surface area and bounds.

- [ ] **Step 2: Verify RED**

Run `ctest -R MarchingCubesGpu`. Expected: API absent.

- [ ] **Step 3: Implement classify/scan/emit pipeline**

Classify one cube per thread using the existing lookup tables in device constant memory. Store triangle counts, invoke PlaMatrix exclusive scan, allocate exact output, and emit vertices/faces in cube linear-index order. Validate finite field values before publishing output.

- [ ] **Step 4: Verify GREEN**

Run Marching Cubes CPU/GPU unit and quality tests. Expected: topology is valid and quality metrics meet documented tolerances.

### Task 7: Introduce device-resident HeightGrid and GPU hole filling

**Files:**
- Modify: `test/unit/mesh/height_grid_gpu_test.cpp`
- Modify: `include/plapoint/gpu/height_grid.h`
- Modify: `src/height_grid_gpu.cu`

- [ ] **Step 1: Write failing device-grid tests**

Define `GpuHeightGrid<Scalar>` with device matrices for heights, validity, fill pass, and optional colors plus scalar bounds/step metadata. Add:

```cpp
GpuHeightGrid<Scalar> buildHeightGridDeviceAsync(
    const PointCloud<Scalar, Device::GPU>& cloud,
    const mesh::HeightGridOptions<Scalar>& options,
    HeightGridGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream);
void fillHolesAsync(GpuHeightGrid<Scalar>& grid,
                    int max_passes,
                    int min_neighbors,
                    int search_radius,
                    HeightGridGpuWorkspace<Scalar>& workspace,
                    cudaStream_t stream);
mesh::HeightGrid<Scalar> downloadHeightGrid(
    const GpuHeightGrid<Scalar>& grid,
    cudaStream_t stream = 0);
```

Test no holes, one/multiple passes, weighted values, colors, fill-pass metadata, radius/min-neighbor settings, empty grids, and CPU/GPU equality.

- [ ] **Step 2: Verify RED**

Run HeightGrid GPU tests. Expected: only CPU-readable wrapper exists.

- [ ] **Step 3: Refactor build output to device grid**

Make existing GPU aggregation populate `GpuHeightGrid`. Preserve the current `buildHeightGrid()` by calling device build and one final download.

- [ ] **Step 4: Implement ping-pong fill kernels**

Each pass reads one immutable grid and writes the other, including height, validity, color, and fill-pass metadata. Synchronous mode downloads one changed flag per pass for early termination. Async mode launches exactly `max_passes`; stable passes reproduce the same values.

- [ ] **Step 5: Verify GREEN**

Run all HeightGrid CPU/GPU tests. Expected: pass.

### Task 8: Extract deterministic Poisson sparse-system assembly

**Files:**
- Modify: `test/unit/mesh/poisson_reconstruction_test.cpp`
- Create: `include/plapoint/mesh/poisson_system.h`
- Create: `src/poisson_system.cpp`
- Create: `src/poisson_reconstruction.cpp`
- Modify: `include/plapoint/mesh/poisson_reconstruction.h`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing assembly and compatibility tests**

Define an internal/public-testable result:

```cpp
template <typename Scalar>
struct PoissonSystem {
    plamatrix::CSRMatrix<Scalar, Device::CPU> matrix;
    plamatrix::DenseMatrix<Scalar, Device::CPU> rhs;
    std::vector<int> leafNodes;
};
```

Test symmetric structure, positive diagonal, deterministic row ordering, finite RHS, identical reconstruction API output on existing fixtures, invalid normals, and depth limits.

- [ ] **Step 2: Verify RED**

Run Poisson tests. Expected: no extracted system API.

- [ ] **Step 3: Extract octree and system assembly**

Move octree construction, normal splatting, leaf collection, and six-neighbor Laplacian assembly from the header into focused source files. Emit COO triplets in leaf-row and neighbor-order sequence, then use PlaMatrix COO-to-CSR duplicate summation.

- [ ] **Step 4: Replace CPU Gauss-Seidel with PlaMatrix PCG**

Add processing options and report access while preserving `reconstruct()`:

```cpp
void setProcessingDevice(ProcessingDevice device);
void setSolverTolerance(double tolerance);
void setSolverIterations(int iterations);
const PoissonProcessingReport& lastReport() const noexcept;
```

CPU mode calls PlaMatrix CPU PCG and evaluates the solved field through the existing CPU Marching Cubes compatibility path.

- [ ] **Step 5: Verify GREEN**

Run Poisson unit, quality, and validation tests. Expected: existing public behavior remains within quality tolerances and report fields are populated.

### Task 9: Add GPU Poisson solve and device surface extraction

**Files:**
- Modify: `test/unit/mesh/poisson_reconstruction_test.cpp`
- Modify: `include/plapoint/mesh/poisson_reconstruction.h`
- Create: `src/poisson_reconstruction_gpu.cu`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing GPU Poisson tests**

Test CPU/GPU mesh bounds, area, watertightness metrics, solver convergence report, explicit GPU failure, Auto allocation/non-convergence fallback, deterministic output, float/double, and GPU output type.

- [ ] **Step 2: Verify RED**

Run CUDA Poisson tests. Expected: GPU backend absent.

- [ ] **Step 3: Implement GPU solve path**

Upload deterministic CSR/RHS, call PlaMatrix GPU PCG, evaluate the solved leaf field into a dense device scalar grid, and call device-field Marching Cubes. Download only final CPU outputs for the existing `reconstruct()` wrapper; expose a GPU-output overload for device pipelines.

- [ ] **Step 4: Implement Auto fallback/reporting**

Auto catches allocation and convergence failures, records exact reason, and retries CPU from the already assembled system. Explicit GPU propagates an exception containing dimensions, iteration count, and residual.

- [ ] **Step 5: Verify GREEN**

Run Poisson CPU/GPU and real reconstruction validation tests. Expected: pass or documented fixture-specific tolerance results.

### Task 10: Benchmark, document, and validate Release 3

**Files:**
- Modify: PlaMatrix `benchmark/benchmark_cases.cpp`
- Modify: PlaMatrix `benchmark/benchmark_cases.cu`
- Modify: PlaMatrix `benchmark/benchmark_cases.h`
- Modify: PlaMatrix `docs/api/sparse-matrix.md`
- Modify: PlaMatrix `docs/architecture.md`
- Modify: PlaPoint `benchmarks/plapoint_benchmark_cpu.h`
- Modify: PlaPoint `benchmarks/plapoint_benchmark_gpu.h`
- Modify: PlaPoint `benchmarks/plapoint_benchmarks.cpp`
- Modify: PlaPoint `README.md`
- Create: PlaPoint `docs/gpu-mesh.md`

- [ ] **Step 1: Add failing benchmark registration checks**

Require PlaMatrix rows `coo_to_csr`, `spmv`, `spmm`, `cg`, `pcg`; require PlaPoint rows `marching_cubes_field`, `height_grid_fill`, `poisson_solve`, and `poisson_end_to_end`.

- [ ] **Step 2: Verify rows are absent**

Run benchmark case-list tests. Expected: new rows missing.

- [ ] **Step 3: Implement benchmark matrices and reports**

Include cold allocation, warm workspace, kernel-only, transfer, iteration count, residual, and end-to-end timings. Use deterministic generated sparse Laplacians and sphere fields.

- [ ] **Step 4: Update documentation**

Document duplicate COO semantics, CSR index width, solver convergence, async lifetimes, device-field layout, HeightGrid download boundary, Poisson fallback behavior, and quality limitations.

- [ ] **Step 5: Run full cross-library verification**

Run PlaMatrix CPU/CUDA full builds and tests, then PlaPoint CPU/CUDA full builds and tests against the updated PlaMatrix. Run focused benchmarks at small and production-scale presets.

Expected: all suites pass; benchmark rows report finite timings and solver reports contain finite residuals.
