# PlaMatrix and PlaPoint CUDA Foundations Design

## Goal

Extend PlaMatrix and PlaPoint with production CUDA paths for six related capability groups while preserving the current CPU APIs, CPU-only builds, numerical semantics, and deterministic behavior.

The work is split into three dependency-ordered releases:

1. PlaMatrix CUDA primitives and batched 3x3 symmetric eigendecomposition.
2. PlaPoint scalable GPU spatial search, normal processing, and outlier filtering.
3. PlaMatrix sparse CUDA operations and PlaPoint CUDA mesh reconstruction.

## Compatibility Contract

- Existing public APIs remain source compatible.
- New synchronous GPU APIs wrap asynchronous implementations and synchronize before returning.
- Async APIs accept a caller-owned `cudaStream_t` and never synchronize implicitly.
- Validation that depends on device data writes a device status into the workspace. Callers check it explicitly after stream synchronization; synchronous wrappers perform that check before returning.
- Repeated operations accept reusable output matrices and workspace objects.
- `ProcessingDevice::Auto` may select CPU or GPU from input size, CUDA availability, and memory requirements.
- An explicit GPU request never silently falls back. CUDA, cuSOLVER, cuSPARSE, allocation, or range failures throw an operation-specific exception.
- CPU-only builds retain compilable declarations and explicit no-CUDA failure behavior for GPU algorithm entry points.
- Float and double are supported wherever the existing library supports both types.

## Ownership Boundary

PlaMatrix owns reusable numerical and data-parallel primitives:

- element-wise transforms;
- reductions and index reductions;
- prefix scan, gather, scatter, and mask compaction;
- batched small symmetric eigendecomposition;
- sparse matrix conversion, multiplication, and iterative solving;
- CUDA stream and workspace mechanics.

PlaPoint owns point-cloud and mesh semantics:

- spatial indexing and neighbor ordering;
- KNN and radius query policy;
- normal estimation and orientation;
- statistical and radius outlier definitions;
- Marching Cubes topology;
- HeightGrid hole filling;
- Poisson system assembly and mesh extraction.

PlaPoint must not add private replacements for new PlaMatrix primitives unless a benchmark demonstrates a domain-specific fused kernel is materially better. Such fused kernels still use PlaMatrix storage and stream conventions.

## Release 1: PlaMatrix CUDA Core

### Element-wise operations

Add GPU and CPU implementations for scalar add, scalar multiply/divide, Hadamard multiply/divide, absolute value, square root, and clamp. Each GPU operation has:

- allocating synchronous overload;
- output-reuse synchronous overload;
- allocating async overload;
- output-reuse async overload.

Dimension validation occurs before launch. Division by a zero scalar is rejected. Element-wise division follows IEEE floating-point behavior for zero matrix elements.

### Reductions and indexing

Add reusable APIs for:

- global and row/column `sum`, `mean`, `min`, `max`;
- global and row/column `argMin`, `argMax`;
- exclusive prefix scan over integer matrices;
- row gather and row scatter;
- byte-mask row compaction with selected-count output.

Async compaction writes into caller-provided capacity-sized matrices and stores the selected count on the device. The synchronous allocating wrapper reads that count and returns exact-sized matrices. Duplicate scatter destinations deterministically select the lowest source row.

Reduction accumulation uses double for float and double inputs where required by existing point-cloud numerical behavior. Equal-value index reductions return the lowest source index. Value reductions propagate NaN when any selected input is NaN; index reductions select the lowest NaN source index. PlaPoint excludes non-finite points with an explicit mask before invoking these primitives.

Workspace objects grow on demand and retain temporary device buffers. APIs validate that matrix sizes fit CUDA launch dimensions and selected backend index types.

### Batched symmetric 3x3 eigendecomposition

Add `symmetricEigh3x3Batched` for compact `N x 6` symmetric matrices stored as `[xx, xy, xz, yy, yz, zz]`. The output contains `N x 3` ascending eigenvalues and `N x 9` eigenvector matrices; each output row stores one 3x3 eigenvector matrix in column-major order. The implementation uses a deterministic fixed-sweep Jacobi kernel and normalizes eigenvectors before storing them.

Degenerate matrices produce finite orthonormal eigenvectors with deterministic sign selection. Empty input returns empty outputs. NaN or infinite input is rejected before result publication.

## Release 2: PlaPoint Search, Features, and Filters

### GPU spatial index

Add a reusable uniform-grid spatial index for float and double point clouds. The index contains:

- finite-point mask;
- integer cell coordinates;
- sorted point indices and cell keys;
- unique cell keys and cell ranges;
- input pointer, point count, build generation, and cell-size cache identity.

Radius queries inspect only intersecting cells. KNN expands cell shells until the current best distance proves that later shells cannot improve the result, with a bounded brute-force fallback for pathological grids. Results are sorted by distance and then source index to preserve deterministic ordering.

The existing brute-force KNN remains available for small inputs. `Auto` chooses the backend using benchmark-derived thresholds. Mutable point access invalidates the cached index.

### Normal estimation and refinement

The GPU pipeline remains device-resident:

1. query neighbor indices using the shared spatial index;
2. accumulate one compact covariance per point;
3. call PlaMatrix batched symmetric eigendecomposition;
4. select the smallest-eigenvalue vector;
5. apply deterministic sign or viewpoint orientation;
6. write normals directly to the GPU cloud.

Normal smoothing gathers neighbor normals, reduces and normalizes them on the caller stream. CPU behavior remains unchanged.

### Outlier filters

Radius outlier removal uses the shared radius search instead of scanning the entire cloud. Statistical outlier removal uses indexed KNN, PlaMatrix reductions for global mean and variance, and PlaMatrix mask compaction for output gathering.

Named scalar fields, colors, normals, intensities, texture coordinates, and other point-wise attributes retain the existing preservation rules. GPU outputs remain deterministic and use the same removed-index semantics as CPU outputs.

## Release 3: Sparse and Mesh CUDA

### PlaMatrix sparse backend

Link cuSPARSE only when CUDA is enabled. Add:

- device COO-to-CSR conversion;
- CPU reference COO-to-CSR conversion with identical ordering;
- CSR SpMV and SpMM;
- Conjugate Gradient and preconditioned Conjugate Gradient;
- Jacobi preconditioning;
- reusable descriptors and temporary buffers;
- synchronous and stream-aware async entry points.

COO conversion sorts by row and then column and deterministically sums duplicate coordinates. CPU SpMV, SpMM, CG, and PCG implementations provide the CPU-only path and numerical reference for CUDA tests.

The public sparse index type remains `Index`. Backend calls use 64-bit cuSPARSE indices when supported and reject unsupported ranges explicitly. Adaptive CG/PCG is a synchronous host-controlled API because convergence requires residual inspection. A fixed-iteration async API submits a caller-selected iteration count without host inspection and exposes a separate report-finalization call after synchronization. Solvers return convergence state, iteration count, and final residual. Explicit GPU requests throw on non-convergence only when configured to require convergence.

### CUDA Marching Cubes

Add a device-field API that accepts dimensions, bounds, iso value, and a contiguous GPU scalar field. The implementation performs:

1. cube classification;
2. per-cube triangle-count scan;
3. deterministic vertex and face emission;
4. optional vertex welding as a separate stage.

The existing callback-based CPU API remains unchanged because arbitrary `std::function` callbacks cannot execute on the device.

### HeightGrid hole filling

Add ping-pong CUDA hole-fill buffers for heights, validity, fill-pass metadata, and optional colors. Each pass records whether any cell changed. The synchronous API stops early when no cell changes; the async API executes the requested maximum passes without a host synchronization inside the loop.

### Poisson reconstruction

Replace the current Gauss-Seidel solve boundary with explicit sparse system assembly. CPU mode uses the same assembled system with a CPU solver. GPU mode uploads or directly assembles CSR data and calls PlaMatrix PCG.

Poisson reports convergence, iteration count, and residual. `Auto` may fall back to CPU after allocation failure or non-convergence; explicit GPU mode reports the failure. Surface extraction uses the new device-field Marching Cubes path when the solved field is on the GPU.

## Error Handling

- Every CUDA runtime, cuBLAS, cuSOLVER, and cuSPARSE call uses checked wrappers.
- Error messages include operation, scalar type, dimensions, stream-sensitive stage, and backend status where available.
- No CUDA API silently narrows `Index` to `int`.
- Workspace growth provides the strong exception guarantee: an existing usable workspace remains valid if a larger allocation fails.
- Async APIs document caller lifetime requirements for inputs, outputs, workspace, and streams.
- Cancellation remains a PlaPoint orchestration concern and is checked between multi-stage launches, not from inside numerical kernels.

## Testing Strategy

All behavior changes follow red-green-refactor development.

PlaMatrix tests cover:

- CPU and GPU element-wise equivalence;
- reduction axes, tie-breaking, empty matrices, and non-finite values;
- scan, gather, scatter, and compaction edge cases;
- batched 3x3 eigen residuals and orthogonality, including repeated eigenvalues;
- COO-to-CSR ordering and duplicate handling;
- SpMV/SpMM reference results;
- CG/PCG convergence and non-convergence reports;
- async stream execution and workspace reuse;
- CPU-only build behavior.

PlaPoint tests cover:

- indexed KNN/radius results against CPU brute force;
- duplicate points, cell boundaries, sparse cells, non-finite points, and deterministic ties;
- normal CPU/GPU angular agreement;
- smoothing and viewpoint orientation;
- SOR and RadiusOR output and attribute preservation;
- Marching Cubes topology and geometric quality against CPU output;
- HeightGrid hole-fill values, colors, and fill-pass metadata;
- Poisson CPU/GPU quality and convergence reporting.

## Benchmark and Auto Selection

Benchmarks report cold-start, warm-workspace, kernel-only, transfer, and end-to-end timings. Required size sweeps include small inputs where CPU should win and production-scale inputs where GPU should win.

`Auto` thresholds are constants derived from checked-in benchmark methodology, not machine-specific benchmark outputs. A user can always override `Auto` with explicit CPU or GPU selection.

## Delivery Gates

Each release must satisfy all of the following before the next release depends on it:

- PlaMatrix CPU-only build and full tests pass;
- PlaMatrix CUDA build and full tests pass;
- PlaPoint CPU-only build and full tests pass;
- PlaPoint CUDA build and full tests pass;
- downstream PlaPoint compilation succeeds against the updated PlaMatrix;
- every benchmark row required by the corresponding release plan exists and produces valid output;
- README and API documentation describe actual behavior and limitations.

Commits and pushes are performed only when explicitly requested by the user.
