# Changelog

## Unreleased

- Add the optional `placoordinate::presets` component with validated factories for
  generic geographic CRS, CGCS2000 Gauss-Kruger, WGS84 Mercator, lunar
  Equirectangular, and north/south lunar Polar Stereographic definitions.
- Add configurable reference ellipsoids, the 1,737,400 m lunar mean sphere, strict
  projection parameter validation, numerical round-trip tests, and an installed
  presets consumer example.

## 0.1.0 - 2026-09-20

- Extract typed frame, CRS, time, coordinate-context, and solver-frame definitions from PlaScan.
- Add static frame-graph transforms with unit conversion and stable transform-chain hashes.
- Add optional GDAL-backed CRS normalization, coordinate transformation, and diagonal uncertainty propagation.
- Add optional nlohmann JSON coordinate-context persistence.
- Add standalone CMake build, tests, install exports, and an external consumer example.
