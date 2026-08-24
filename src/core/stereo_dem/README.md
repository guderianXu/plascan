# RPC Stereo DEM And DOM

`src/core/stereo_dem` completes the georeferenced satellite-stereo product chain that precedes the existing
`terrain` raster writers.

## Processing chain

```text
RPC TIFF pair
  -> mutual SIFT matching
  -> fundamental-matrix RANSAC
  -> nonlinear RPC forward intersection in WGS84 ECEF
  -> robust ellipsoidal-height filtering
  -> local WGS84 UTM point cloud
  -> terrain::DemGenerator / DemDomIO
  -> dem.tif and quality rasters
  -> DEM grid to WGS84
  -> RPC projection into source images
  -> blended DOM GeoTIFF with alpha coverage
```

Frame cameras use the classical collinearity equations. RPC images use their rational-polynomial projection in
the same multi-view reprojection objective; they are never converted into a fictitious fixed-centre camera.

`RpcStereoDemGenerator` writes a Float32 DEM, a preview, a PLY raster point cloud, error/count/confidence/coverage
rasters, and a JSON processing report. Heights are WGS84 ellipsoidal metres. The horizontal CRS is the WGS84 UTM
zone selected from the median accepted stereo point.

`RpcDomGenerator` reads the generated DEM, transforms every valid grid centre back to WGS84 longitude/latitude,
projects it into each RPC image, performs bilinear sampling, and writes a georeferenced RGB+alpha TIFF. The first
version uses weighted-equal image blending and does not yet perform seamline or occlusion optimisation.

The `rpc_stereo_products_cli` command runs both stages:

```powershell
rpc_stereo_products_cli `
  --left testData/rpc_stereo_pair/Images/img_01.tif `
  --right testData/rpc_stereo_pair/Images/img_02.tif `
  --output testData/rpc_stereo_pair/Products `
  --resolution 2.0
```
