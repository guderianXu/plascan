# Dependency policy

PlaCamera keeps camera-model evaluation and standard file APIs separate from image
processing and coordinate-reference-system parsing. Dependency-backed readers live in
optional components, which keeps the core portable and prevents an adapter's ABI from
becoming the library's public ABI.

## Version 0.2 dependency set

| Scope | Dependency | Linkage | Reason |
|---|---|---|---|
| Runtime core | C++20 standard library | Required | Arrays, ownership, strings, errors, and numerical primitives are sufficient for current camera equations. |
| Runtime core | `placoordinate::types` 0.1+ | Required, public | Shares frame identifiers and time references across cameras, control points, SfM, DEM/DOM, and LiDAR without pulling CRS adapters into camera evaluation. |
| External references | `placoordinate::transform` 0.1+ | Optional, public | Explicit frame and unit resolution for `placamera::reference`; absent from the base camera target. |
| Canonical state and ISD decoding | nlohmann/json | Optional, private | Used only by `placamera::state`; public APIs exchange strings and opaque state rather than JSON DOM types. |
| RPC raster files | GDAL | Optional, private | Used only by `placamera::gdal` to open rasters and sidecars; public APIs expose PlaCamera and standard-library types. |
| Build/install | CMake 3.21+ | Required at build time | Target export, relocatable package config, and consumer discovery. |
| Unit tests | GTest | Optional, private | Used only when `PLACAMERA_BUILD_TESTS=ON`; never appears in installed headers or link interfaces. |

The core deliberately does not require Qt, OpenCV, GDAL, Eigen, PlaMatrix,
PlaBundle, Boost, fmt, spdlog, PROJ, a JSON implementation, or csmapi. General
frame graphs, CRS conversion, uncertainty propagation, and coordinate-context
persistence remain in sibling PlaCoordinate components; PlaCamera depends only
on its dependency-free types target.

## Optional and planned components

| Component | Candidate dependency | Intended responsibility | Public boundary |
|---|---|---|---|
| `placamera::state` | nlohmann/json | Implemented deterministic JSON definition/instance codecs, strict validation, and schema upgrades. MessagePack is not implemented. | Strings and opaque byte payloads; no `nlohmann::json` in public APIs. |
| `placamera::reference` | `placoordinate::transform` | Typed external GNSS/IMU/POS observations, resolution, pose priors, and geometry comparison. | PlaCamera and PlaCoordinate value types; no Qt or application image UUID. |
| `placamera::gdal` | GDAL | Open RPC/RPB rasters, read image dimensions and RPC00B metadata, and construct typed RPC instances. | `std::filesystem::path` and PlaCamera types; no dependency types in public headers. |
| `placamera_csm` | csmapi | Import/export or wrap models through actual CSM `Model`, `RasterGM`, and plugin contracts. | CSM types only in this adapter target. |
| `placamera_spice` | CSPICE | Build line-scan trajectories and frame transforms from kernels/ISD. | Produces validated PlaCamera trajectory state; evaluation remains kernel-free. |
| Application project IO | libtiff/libzip | Locate images, unpack project containers, and publish application-specific resources. | Stays in the application; PlaCamera owns reusable camera-file IO. |
| Image utilities | OpenCV | Remap/undistort image pixels and generate maps. | Consumes camera results; not part of the mathematical model. |

Optional ecosystem adapters must default to off, use private linkage where
possible, and carry their own install-time dependency discovery. The state
component is enabled by default but imported only when a downstream package
requests `COMPONENTS state`. The reference and GDAL components default off and are imported
with `COMPONENTS reference` and `COMPONENTS gdal`. A user that links only `placamera::placamera`
needs only `placoordinate::types`, never GDAL, nlohmann/json, or PlaCoordinate's
JSON/GDAL components.

## Libraries considered but not selected

- **Eigen/PlaMatrix**: unnecessary for 2x2 and 3x3 operations in the current
  model kernels. Explicit fixed-size math makes conventions reviewable and
  avoids exporting another library's types.
- **GeographicLib**: does not add a unique v0.1 capability. RPC uses a small,
  tested conversion for its explicitly configured reference ellipsoid, while
  arbitrary CRS and datum transformations belong to PROJ.
- **Qt JSON**: would couple a reusable numerical library to PlaScan's GUI
  stack. State codecs must remain UI-independent.
- **A mandatory CSM base class**: would force CSM's ABI and lifecycle on every
  user. Compatibility should be verified in a real optional adapter instead.

## Acceptance rules for a new dependency

A dependency may become mandatory only when all of these are true:

1. it provides functionality needed by every supported camera model;
2. an equivalent small, testable implementation would be riskier to own;
3. it works with GCC, MSVC, and Apple Clang and has a compatible license;
4. it has a stable CMake target and install story;
5. its types do not unnecessarily leak into the public model contract.
