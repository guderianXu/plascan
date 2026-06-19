# Terrain Module

`src/core/terrain` turns dense clouds, meshes, or depth-derived products into formal DEM/DOM terrain artifacts.
The module now records product metadata and quality rasters instead of treating DEM/DOM as one-off temporary
exports.

## DEM Aggregation

- `DemGridAggregator` bins samples into grid cells and supports mean, median, min, max, count, standard
  deviation, NMAD, P80, confidence-weighted, and inverse-error-weighted aggregation.
- `DemGridData` carries elevation, color, point count, confidence, coverage, and error-related matrices when
  available.
- Confidence is used as a positive weight. Triangulation or source error can be used as an inverse weight for
  cells where quality metadata exists.

## Product Manifest

- `TerrainProductManifest` stores the paths and properties of terrain products for GUI and project metadata.
- A DEM run can write `dem.tif`, `dem_error.tif`, `dem_count.tif`, `dem_confidence.tif`, and
  `dem_coverage.tif`.
- Product records include `dem_path`, `dom_path`, `error_path`, `count_path`, `confidence_path`,
  `coverage_path`, projection, grid resolution, aggregation mode, and preview path.

## Mosaic And DOM

- `DemMosaic` combines multiple DEM tiles with first, last, mean, median, min, max, confidence-weighted, or
  inverse-error-weighted blending.
- Tiled processing keeps large terrain products from requiring a single in-memory DEM mosaic.
- `DomGenerator` checks DEM coverage before texturing and can weight DOM color selection by view quality,
  confidence, and image sharpness.

## GUI Expectations

- The project tree should show DEM, DOM, error, count, confidence, and coverage artifacts as separate child
  nodes where available.
- Terrain products are sorted by natural filename order and refreshed from project metadata.

## Focused Tests

Useful filters:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainDemDom|DemGridAggregator|DemMosaic|TerrainProductManifest|DemQualityRasters" --output-on-failure
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DataTreeWidgetTest\.DemSectionShowsQualityRasterProducts" --output-on-failure
```
