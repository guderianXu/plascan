# PlaMatrix CUDA Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable CPU/CUDA element-wise, reduction, indexing, compaction, and batched symmetric 3x3 eigendecomposition primitives to PlaMatrix.

**Architecture:** Public template APIs live under `include/plamatrix`; CPU and CUDA implementations use explicit float/double instantiation in focused source files. GPU allocating overloads delegate to output-reuse async overloads, while synchronous wrappers synchronize the supplied stream. Grow-only workspace classes own temporary matrices and preserve the existing no-CUDA behavior.

**Tech Stack:** C++17, CUDA 13.x, CUB device primitives, OpenMP, GoogleTest, CMake.

---

## File Map

- Create `include/plamatrix/dense/elementwise.h`: scalar and matrix element-wise public API.
- Create `src/dense/elementwise_cpu.cpp`: CPU/OpenMP implementations.
- Create `src/dense/elementwise.cu`: CUDA kernels and stream-aware wrappers.
- Create `include/plamatrix/ops/reduction.h`: reduction enums, results, workspace, and public API.
- Create `src/ops/reduction_cpu.cpp`: deterministic CPU reductions.
- Create `src/ops/reduction.cu`: CUB-backed CUDA reductions.
- Create `include/plamatrix/ops/indexing.h`: scan, gather, scatter, and compaction API.
- Create `src/ops/indexing_cpu.cpp`: CPU reference implementations.
- Create `src/ops/indexing.cu`: CUDA indexing and compaction.
- Create `include/plamatrix/ops/small_matrix.h`: batched symmetric 3x3 eigendecomposition API.
- Create `src/ops/small_matrix_cpu.cpp`: CPU reference Jacobi implementation.
- Create `src/ops/small_matrix.cu`: deterministic batched CUDA Jacobi kernel.
- Modify `include/plamatrix/plamatrix.h`: export new APIs.
- Modify `src/CMakeLists.txt`: compile CPU and conditional CUDA sources.
- Create `test/unit/dense/elementwise_test.cpp`.
- Create `test/unit/ops/reduction_test.cpp`.
- Create `test/unit/ops/indexing_test.cpp`.
- Create `test/unit/ops/small_matrix_test.cpp`.
- Modify `test/CMakeLists.txt`: register tests.
- Modify `benchmark/benchmark_cases.cpp`, `benchmark/benchmark_cases.cu`, and `benchmark/benchmark_cases.h`: add warm/cold rows.
- Modify `README.md`, `docs/api/dense-matrix.md`, `docs/api/linear-algebra.md`, and `docs/architecture.md`.

### Task 1: Define element-wise behavior through CPU tests

**Files:**
- Create: `test/unit/dense/elementwise_test.cpp`
- Create: `include/plamatrix/dense/elementwise.h`
- Create: `src/dense/elementwise_cpu.cpp`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing CPU API tests**

Add typed float/double tests that exercise this exact API:

```cpp
auto scaled = plamatrix::scalarMultiply(input, Scalar(2));
auto shifted = plamatrix::scalarAdd(input, Scalar(-1));
auto divided = plamatrix::scalarDivide(input, Scalar(2));
auto product = plamatrix::hadamardMultiply(input, other);
auto quotient = plamatrix::hadamardDivide(input, other);
auto absolute = plamatrix::absElements(input);
auto rooted = plamatrix::sqrtElements(non_negative);
auto clipped = plamatrix::clampElements(input, Scalar(-1), Scalar(1));
```

Cover empty matrices, mismatched Hadamard dimensions, scalar division by zero, invalid clamp bounds, and IEEE matrix-element division by zero.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
cmake --build build-cpu --target plamatrix_tests --parallel 8
ctest --test-dir build-cpu --output-on-failure -R "Elementwise"
```

Expected: compilation fails because `plamatrix/dense/elementwise.h` and the named APIs do not exist.

- [ ] **Step 3: Add the public API and CPU implementation**

Declare allocating CPU overloads plus this shared operation enum:

```cpp
enum class ElementwiseUnaryOp { Abs, Sqrt };

template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarMultiply(
    const DenseMatrix<Scalar, Device::CPU>& input, Scalar value);
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarAdd(
    const DenseMatrix<Scalar, Device::CPU>& input, Scalar value);
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> scalarDivide(
    const DenseMatrix<Scalar, Device::CPU>& input, Scalar value);
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> hadamardMultiply(
    const DenseMatrix<Scalar, Device::CPU>& lhs,
    const DenseMatrix<Scalar, Device::CPU>& rhs);
template <typename Scalar>
DenseMatrix<Scalar, Device::CPU> hadamardDivide(
    const DenseMatrix<Scalar, Device::CPU>& lhs,
    const DenseMatrix<Scalar, Device::CPU>& rhs);
```

Add corresponding `absElements`, `sqrtElements`, and `clampElements` declarations. Implement contiguous loops with `detail::shouldUseOpenMp(input.size())`, validate before allocating outputs, and explicitly instantiate float/double.

- [ ] **Step 4: Build and verify GREEN**

Run the focused test command again. Expected: all Elementwise CPU tests pass.

- [ ] **Step 5: Run existing dense tests**

Run:

```powershell
ctest --test-dir build-cpu --output-on-failure -R "DenseMatrix|DenseOps|Elementwise"
```

Expected: pass with no regression.

### Task 2: Add stream-aware CUDA element-wise operations

**Files:**
- Modify: `test/unit/dense/elementwise_test.cpp`
- Modify: `include/plamatrix/dense/elementwise.h`
- Create: `src/dense/elementwise.cu`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing GPU and async tests**

Test allocating, output-reuse, and async forms:

```cpp
plamatrix::DenseMatrix<Scalar, Device::GPU> output(rows, cols);
plamatrix::scalarMultiplyAsync(input_gpu, Scalar(2), output, stream);
PLAMATRIX_CHECK_CUDA(cudaStreamSynchronize(stream));
EXPECT_MATRIX_NEAR(output.toCpu(), expected, tolerance);
```

Verify operations launched on two non-default streams do not require a device-wide synchronization. Add CPU-only tests asserting GPU algorithm entry points throw the established no-CUDA exception.

- [ ] **Step 2: Verify RED in CUDA and CPU-only builds**

Run:

```powershell
cmake --build build-cuda --target plamatrix_tests --parallel 8
ctest --test-dir build-cuda --output-on-failure -R "Elementwise"
```

Expected: compilation fails because async/output-reuse overloads are absent.

- [ ] **Step 3: Implement CUDA overloads**

For each operation expose this pattern:

```cpp
template <typename Scalar>
void scalarMultiplyAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream);

template <typename Scalar>
void scalarMultiply(
    const DenseMatrix<Scalar, Device::GPU>& input,
    Scalar value,
    DenseMatrix<Scalar, Device::GPU>& output,
    cudaStream_t stream = 0);
```

Use one grid-stride kernel family, validate output shape before launch, call `PLAMATRIX_CHECK_CUDA(cudaGetLastError())`, and synchronize only in non-Async wrappers.

- [ ] **Step 4: Verify CUDA GREEN and CPU-only compatibility**

Run focused tests in `build-cuda` and `build-cpu`. Expected: both pass.

### Task 3: Define deterministic CPU reductions

**Files:**
- Create: `test/unit/ops/reduction_test.cpp`
- Create: `include/plamatrix/ops/reduction.h`
- Create: `src/ops/reduction_cpu.cpp`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing reduction tests**

Define and test:

```cpp
enum class ReductionAxis { All, Rows, Columns };

template <typename Scalar, Device Dev>
struct IndexedReductionResult {
    DenseMatrix<Scalar, Dev> values;
    DenseMatrix<Index, Dev> indices;
};

auto sums = plamatrix::sum(input, ReductionAxis::Columns);
auto means = plamatrix::mean(input, ReductionAxis::Rows);
auto minima = plamatrix::min(input, ReductionAxis::All);
auto argmax = plamatrix::argMax(input, ReductionAxis::Columns);
```

Cover column-major axis semantics, empty dimensions, equal-value lowest-index tie breaking, negative values, NaN propagation, and float accumulation against a double reference.

- [ ] **Step 2: Verify RED**

Build and run `ctest -R Reduction` in `build-cpu`. Expected: missing API compilation failure.

- [ ] **Step 3: Implement CPU reductions**

Return shapes are fixed as:

- `All`: `1 x 1`;
- `Rows`: `input.rows() x 1`, reducing columns;
- `Columns`: `1 x input.cols()`, reducing rows.

Use deterministic serial reduction inside each output lane and OpenMP only across independent lanes. For `All`, preserve source linear-index ordering for ties and NaNs.

- [ ] **Step 4: Verify GREEN**

Run Reduction tests and existing point-cloud covariance tests. Expected: pass.

### Task 4: Add CUB-backed GPU reductions and workspace reuse

**Files:**
- Modify: `test/unit/ops/reduction_test.cpp`
- Modify: `include/plamatrix/ops/reduction.h`
- Create: `src/ops/reduction.cu`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing GPU reduction tests**

Test CPU/GPU equivalence, output reuse, non-default streams, workspace growth, repeated warm calls, NaN behavior, and deterministic tie indices.

Use this workspace contract:

```cpp
class ReductionWorkspace {
public:
    void reserveBytes(std::size_t bytes);
    std::size_t capacityBytes() const noexcept;
    void* data() noexcept;
};
```

- [ ] **Step 2: Verify RED**

Run `ctest --test-dir build-cuda -R Reduction`. Expected: missing GPU overloads.

- [ ] **Step 3: Implement GPU reduction paths**

Use CUB `DeviceReduce` for `All`; use deterministic custom lane kernels for rows/columns. Represent indexed values as `(value, source_index)` and compare NaN first, then value, then lowest index. Query CUB temporary bytes before `reserveBytes`, then launch on the supplied stream.

- [ ] **Step 4: Verify GREEN and allocation reuse**

Run tests twice with the same workspace and assert its capacity does not increase on the second equal-sized call. Expected: pass.

### Task 5: Add CPU scan, gather, scatter, and mask compaction

**Files:**
- Create: `test/unit/ops/indexing_test.cpp`
- Create: `include/plamatrix/ops/indexing.h`
- Create: `src/ops/indexing_cpu.cpp`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing API tests**

Exercise:

```cpp
auto offsets = plamatrix::exclusiveScan(counts);
auto gathered = plamatrix::gatherRows(input, indices);
plamatrix::scatterRows(values, indices, output);
auto compacted = plamatrix::compactRows(input, keep_mask);
```

Test empty inputs, duplicate gather indices, deterministic duplicate scatter handling, out-of-range indices, mask length mismatch, stable compaction order, and preservation of every input column. When scatter destinations repeat, the lowest source row wins.

- [ ] **Step 2: Verify RED**

Build/run `ctest -R Indexing` in `build-cpu`. Expected: missing API.

- [ ] **Step 3: Implement CPU references**

Use `DenseMatrix<Index, CPU>` for scan/indices and `DenseMatrix<std::uint8_t, CPU>` for masks. The synchronous allocating `compactRows` returns exact-sized matrices:

```cpp
template <typename Scalar, Device Dev>
struct CompactRowsResult {
    DenseMatrix<Scalar, Dev> values;
    DenseMatrix<Index, Dev> sourceIndices;
};
```

Preserve ascending source-row order.

- [ ] **Step 4: Verify GREEN**

Run Indexing tests. Expected: pass.

### Task 6: Add CUDA indexing and compaction

**Files:**
- Modify: `test/unit/ops/indexing_test.cpp`
- Modify: `include/plamatrix/ops/indexing.h`
- Create: `src/ops/indexing.cu`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing GPU tests**

Cover CPU/GPU equivalence, stable ordering, capacity-sized output/workspace reuse, non-default stream execution, masks selecting none/all, deterministic duplicate scatter destinations, and int64 scan overflow detection.

- [ ] **Step 2: Verify RED**

Run CUDA Indexing tests. Expected: missing overloads.

- [ ] **Step 3: Implement CUDA operations**

Use CUB `DeviceScan::ExclusiveSum` and `DeviceSelect::Flagged`. Gather/scatter kernels operate row-wise over column-major input. Duplicate scatter destinations select the lowest source row through an owner-index pass. Add `IndexingWorkspace` with grow-only temporary storage and device selected-count/status values.

Add the CUDA-side `DenseMatrix<std::uint8_t, Device::GPU>` copy/fill instantiations required by keep masks; include an all-zero and all-one mask test so this support is exercised directly rather than only through compaction.

Expose the non-synchronizing compaction form as:

```cpp
template <typename Scalar>
void compactRowsAsync(
    const DenseMatrix<Scalar, Device::GPU>& input,
    const DenseMatrix<std::uint8_t, Device::GPU>& keep_mask,
    DenseMatrix<Scalar, Device::GPU>& capacity_output,
    DenseMatrix<Index, Device::GPU>& capacity_source_indices,
    DenseMatrix<Index, Device::GPU>& selected_count,
    IndexingWorkspace& workspace,
    cudaStream_t stream);
```

The two capacity outputs have `input.rows()` rows. The synchronous allocating wrapper downloads `selected_count` and returns exact-sized matrices.

- [ ] **Step 4: Verify GREEN**

Run CPU and CUDA Indexing tests. Expected: pass.

### Task 7: Add CPU batched symmetric 3x3 eigendecomposition

**Files:**
- Create: `test/unit/ops/small_matrix_test.cpp`
- Create: `include/plamatrix/ops/small_matrix.h`
- Create: `src/ops/small_matrix_cpu.cpp`
- Modify: `include/plamatrix/plamatrix.h`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write failing numerical tests**

Define:

```cpp
template <typename Scalar, Device Dev>
struct SymmetricEigh3x3Result {
    DenseMatrix<Scalar, Dev> eigenvalues;  // N x 3, ascending
    DenseMatrix<Scalar, Dev> eigenvectors; // N x 9, column-major per row
};

auto result = plamatrix::symmetricEigh3x3Batched(compact_matrices);
```

Test diagonal, rotated, rank-one, zero, repeated-eigenvalue, float/double, empty, and non-finite matrices. Verify `A*v=lambda*v`, orthonormality, ascending values, and deterministic eigenvector signs.

- [ ] **Step 2: Verify RED**

Run `ctest -R SmallMatrix` in CPU build. Expected: missing API.

- [ ] **Step 3: Implement deterministic CPU Jacobi**

Decode `[xx, xy, xz, yy, yz, zz]`, perform eight fixed cyclic Jacobi sweeps over `(0,1)`, `(0,2)`, `(1,2)`, sort eigenpairs ascending, normalize each vector, and flip each vector so its largest-magnitude component is non-negative. Resolve repeated eigenvalues by deterministic basis re-orthogonalization.

- [ ] **Step 4: Verify GREEN**

Run SmallMatrix tests and compare random symmetric matrices with existing `eigh` eigenvalues. Expected: pass.

### Task 8: Add CUDA batched symmetric 3x3 eigendecomposition

**Files:**
- Modify: `test/unit/ops/small_matrix_test.cpp`
- Modify: `include/plamatrix/ops/small_matrix.h`
- Create: `src/ops/small_matrix.cu`
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Write failing GPU tests**

Test at least 1, 31, 32, 33, and 4096 matrices, CPU/GPU eigen residual agreement, output reuse, non-default streams, explicit async status checking for non-finite input, and no allocation growth on repeated calls.

- [ ] **Step 2: Verify RED**

Run CUDA SmallMatrix tests. Expected: missing GPU overloads.

- [ ] **Step 3: Implement one-matrix-per-thread Jacobi kernel**

Load six values into registers, run the exact CPU sweep/order/sign rules, and store three values plus nine vector components. The async kernel records the lowest invalid row in a device workspace status and zeroes that row's outputs; callers inspect status after synchronizing. The synchronous wrapper synchronizes, checks status, and throws before returning outputs. Provide sync and async output-reuse overloads.

- [ ] **Step 4: Verify GREEN**

Run CPU/CUDA SmallMatrix tests. Expected: pass within scalar-specific residual tolerance.

### Task 9: Add benchmarks, documentation, and release validation

**Files:**
- Modify: `benchmark/benchmark_cases.cpp`
- Modify: `benchmark/benchmark_cases.cu`
- Modify: `benchmark/benchmark_cases.h`
- Modify: `README.md`
- Modify: `docs/api/dense-matrix.md`
- Modify: `docs/api/linear-algebra.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: Add benchmark registration tests or assertions**

Register `elementwise`, `reduction`, `compact`, and `eigh3x3_batch` cases. Each CUDA case records cold allocation, warm workspace, kernel-only event time, and transfer time.

- [ ] **Step 2: Verify benchmark rows are initially absent**

Run the benchmark case-list command and confirm the four new names are missing.

- [ ] **Step 3: Implement benchmark rows and update documentation**

Document exact shapes, axis semantics, NaN behavior, async lifetime rules, compact ordering, eigendecomposition layout, and CPU-only behavior.

- [ ] **Step 4: Run full Release 1 verification**

Run:

```powershell
cmake --build build-cpu --parallel 8
ctest --test-dir build-cpu --output-on-failure
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda --output-on-failure
build-cuda\benchmark\plamatrix_benchmark.exe --mode all --size small --case elementwise,reduction,compact,eigh3x3_batch
```

Expected: all tests pass and all benchmark rows report finite non-negative timings.

- [ ] **Step 5: Validate PlaPoint downstream compilation**

Configure PlaPoint against this PlaMatrix build in CPU-only and CUDA modes and build `plapoint_tests`. Expected: both compile before Release 2 starts.
