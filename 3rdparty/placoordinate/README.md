# PlaCoordinate

PlaCoordinate is a C++20 library for explicit coordinate semantics in planetary
and photogrammetric software. It owns typed frame and CRS identifiers, immutable
coordinate contexts, solver-frame scale state, static rigid frame graphs, unit
conversion, canonical hashes, and optional CRS/JSON adapters.

## Components

- `placoordinate::types`: dependency-free frame, CRS, time, context, solver scale,
  and SHA-256 types.
- `placoordinate::transform`: static parent-frame graphs; point, vector, and rotation
  conversion; metre/kilometre conversion; transform-chain hashes.
- `placoordinate::gdal`: optional GDAL-backed WKT normalization, explicit axis order,
  non-ballpark CRS transforms, and diagonal uncertainty propagation.
- `placoordinate::presets`: optional GDAL-backed factories for geographic CRS,
  Gauss-Kruger, Mercator, lunar Equirectangular, and north/south lunar Polar
  Stereographic definitions.
- `placoordinate::state`: optional nlohmann JSON persistence for complete coordinate
  contexts.

The core does not depend on Qt, OpenCV, PlaScan, or camera models. Dynamic SPICE
frames, grid-resource management, and full covariance propagation are outside the
0.1 scope.

## Build and test

```bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLACOORDINATE_BUILD_TESTS=ON
cmake --build build/release
ctest --test-dir build/release --output-on-failure
```

The optional adapters can be disabled; the presets component requires GDAL:

```bash
cmake -S . -B build/core \
  -DPLACOORDINATE_BUILD_GDAL=OFF \
  -DPLACOORDINATE_BUILD_PRESETS=OFF \
  -DPLACOORDINATE_BUILD_STATE=OFF
```

Install and verify the exported package:

```bash
cmake --install build/release --prefix build/install
cmake -S examples/consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/consumer
```

Downstream projects use component targets directly:

```cmake
find_package(placoordinate CONFIG REQUIRED COMPONENTS types transform)
target_link_libraries(my_target PRIVATE placoordinate::transform)
```

Projection presets are consumed explicitly because they require GDAL:

```cmake
find_package(placoordinate CONFIG REQUIRED COMPONENTS presets)
target_link_libraries(my_target PRIVATE placoordinate::presets)
```

```cpp
#include <placoordinate/presets/ProjectionPresets.h>

placoordinate::LunarEquirectangularParameters parameters;
parameters.centralMeridianDegrees = 0.0;
auto moon_eq = placoordinate::makeLunarEquirectangularSpatialReference(
    placoordinate::SpatialReferenceId("moon-eq"),
    placoordinate::CoordinateFrameId("moon-eq-frame"),
    parameters);
```

All angular parameters use east-positive degrees and all offsets/radii use metres;
vertical semantics remain explicit and configurable on every projected preset.
Lunar presets use a configurable sphere with a default mean radius of 1,737,400 m.
Gauss-Kruger defaults to the CGCS2000 ellipsoid and a 500,000 m false easting, but
requires the central meridian to be supplied explicitly. Lunar `EQ` means spherical
Equirectangular/Equidistant Cylindrical, not conformal Mercator. Lunar `POLA` is
selected explicitly as north- or south-polar stereographic.

## PlaScan integration

PlaScan includes `placoordinate/...`, uses the `placoordinate` namespace, and links
the component targets directly. The former PlaScan `coordinate_system` forwarding
layer and legacy CMake targets have been removed.

## License

MIT. See [LICENSE](LICENSE).
