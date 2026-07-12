# PlaMatrix / PlaPoint / PlaScan Vector Math Consolidation Design

## Goal

Remove the duplicated three-dimensional vector implementations across PlaMatrix, PlaPoint, and PlaScan while preserving PlaScan's existing `std::array<double, 3>` public data contracts.

## Current State

- PlaMatrix exposes `plamatrix::Vec3<T>` from `ops/point_cloud.h`, but the type is only a three-field parameter object used by rotation and rigid-transform functions.
- PlaPoint's mesh processing header defines another private `detail::Vec3` together with private `cross()` and `norm()` implementations.
- PlaScan's `src/common/math/Vec3Ops.h` implements `dot()`, `norm()`, `cross()`, `subtract()`, `isFinite()`, and `normalize()` for `std::array<double, 3>`.
- PlaScan's `src/common/math/Vec.h` is not included or referenced by production code or tests.
- The `plascan_common_math` CMake interface target has no consumers. Bundle adjustment and SfM reach the headers through raw `${CMAKE_SOURCE_DIR}/src/common` include paths.

## Chosen Architecture

PlaMatrix owns reusable small-vector representation and arithmetic. PlaPoint owns point-cloud and mesh algorithms and consumes PlaMatrix vector arithmetic. PlaScan owns photography-specific data contracts and converts at module boundaries when it needs generic vector operations.

The dependency direction remains:

```text
PlaScan algorithms -> PlaPoint -> PlaMatrix
PlaScan BA/SfM ----------------> PlaMatrix
```

PlaScan must not wrap isolated camera centers, normals, or bundle-adjustment parameters in `plapoint::PointCloud`; a point-cloud container is the wrong abstraction for scalar geometry operations.

## PlaMatrix API

Create `include/plamatrix/ops/vector.h` and move `Vec3<T>` into it. The header remains CPU/GPU-backend independent and provides:

- value initialization for all components;
- construction from three scalar values;
- construction from `std::array<T, 3>` and conversion back to it;
- `+`, `-`, scalar `*`, scalar `/`, `+=`, and `-=`;
- `dot()`, `cross()`, `squaredNorm()`, `norm()`, `normalized()`, and `isFinite()`.

All operations are header-only templates. Division and normalization do not silently invent fallback directions. `normalized(epsilon)` returns the original vector when its norm is no greater than `epsilon`, matching PlaScan's current behavior. The default epsilon is derived from `std::numeric_limits<T>::epsilon()` and callers may provide a domain-specific threshold.

`ops/point_cloud.h` includes `ops/vector.h` and continues exposing `Vec3<T>` to existing consumers, so current source compatibility is preserved. The umbrella `plamatrix/plamatrix.h` also includes the new header directly.

## PlaPoint Migration

`mesh_processing.h` replaces its private `detail::Vec3` with `plamatrix::Vec3<long double>`. Point extraction, edge construction, cross products, accumulated normals, and norm calculations use the PlaMatrix API.

Public PlaPoint APIs and point-cloud storage do not change. Existing mesh-processing tests provide behavioral regression coverage; a focused test verifies that degenerate-face removal and normal-related processing still produce the same result after migration.

## PlaScan Migration

BundleAdjust and SfM retain `std::array<double, 3>` in public structures, CUDA host worksets, camera APIs, and serialization boundaries. The four current `Vec3Ops.h` consumers convert arrays with `plamatrix::Vec3<double>(array)` and convert results back with `toArray()` only where a resulting vector is required.

The affected targets link directly to `plamatrix::plamatrix`. They must not rely on PlaPoint's transitive dependency to make PlaMatrix headers visible.

After all consumers migrate:

- delete `src/common/math/Vec3Ops.h`;
- delete the unused `src/common/math/Vec.h`;
- remove the unused `plascan_common_math` interface target;
- remove obsolete raw common include paths only where they are no longer needed for other common headers;
- update `docs/PROJECT_ARCHITECTURE.md` so it no longer lists the deleted math module.

## Compatibility And Error Handling

- Existing brace initialization such as `Vec3<double>{1.0, 2.0, 3.0}` remains valid.
- Existing PlaMatrix rotation and rigid-transform APIs retain their signatures.
- Non-finite checks remain component-wise and return `false` for NaN or infinity.
- Normalizing a near-zero vector preserves the input, matching PlaScan's current `Vec3Ops` contract.
- No PlaScan public API, file format, CUDA structure, or point-cloud metadata contract changes.

## Testing

Follow test-driven development in dependency order:

1. Add PlaMatrix unit tests for construction, array conversion, arithmetic, dot/cross, norm, near-zero normalization, and finite checks; verify they fail before implementing `ops/vector.h`.
2. Run the full PlaMatrix CPU test suite after the implementation.
3. Add or tighten PlaPoint mesh-processing regression coverage, migrate the implementation, and run the PlaPoint CPU test suite.
4. Migrate PlaScan BundleAdjust and SfM consumers, then build their targets and run the related BA/SfM tests.
5. Configure and build the CUDA-enabled dependency chain to ensure the header-only vector API is accepted by the CUDA 13.1 toolchain.

## Out Of Scope

- Replacing every `std::array<double, 3>` in PlaScan.
- Redesigning PlaPoint's `PointCloud` storage.
- Adding dynamic-size vectors, SIMD specialization, or GPU kernels for scalar `Vec3` arithmetic.
- Refactoring unrelated local vector calculations elsewhere in PlaScan.
