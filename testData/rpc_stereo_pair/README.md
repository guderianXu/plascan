# TIFF + RPC 卫星立体像对

这是从 S2P（Satellite Stereo Pipeline）上游测试数据中提取的一个小型 Pléiades 卫星立体像对。
两片都是保留原始传感器几何的 8 位单波段 TIFF，RPC00B 参数直接嵌入 TIFF 的 GDAL `RPC`
metadata domain，可用于 PlaScan 的 RPC 读取、投影、反投影、像对匹配和双 RPC 前方交会测试。

## 文件

- `Images/img_01.tif`：1024×1024，单波段 `uint8`；
- `Images/img_02.tif`：1031×1102，单波段 `uint8`；
- `Metadata/rpc_parameters.json`：从两张 TIFF 无损展开的 RPC 参数，便于人工检查和非 GDAL 程序读取；
- `Metadata/upstream_s2p_config.json`：上游 S2P 测试所用的处理区域和参数；
- `images.lis`：相对路径影像清单；
- `manifest.json`：来源版本、文件大小、SHA-256 和验证结果；
- `LICENSE.s2p.txt`：上游 S2P 仓库的 BSD-3-Clause 许可证文本。

## RPC 内容

每张 TIFF 的 `RPC` 域均包含 16 个标准键：

- 影像归一化：`LINE_OFF`、`SAMP_OFF`、`LINE_SCALE`、`SAMP_SCALE`；
- 地面归一化：`LAT_OFF`、`LONG_OFF`、`HEIGHT_OFF`、`LAT_SCALE`、`LONG_SCALE`、
  `HEIGHT_SCALE`；
- 四组 20 项系数：`LINE_NUM_COEFF`、`LINE_DEN_COEFF`、`SAMP_NUM_COEFF`、
  `SAMP_DEN_COEFF`；
- 误差字段：`ERR_BIAS`、`ERR_RAND`。

影像没有地图投影 CRS，也没有可直接作为正射图使用的仿射地理变换。像素与 WGS84
经度、纬度、椭球高之间的关系完全由 RPC 模型定义，这是原始卫星立体影像的正常形式。
区域大致位于留尼汪岛（约 55.65°E、21.23°S）。

## 8 位转换

上游 TIFF 以 `uint16` 容器保存，两个影像的主要有效灰度集中在较窄范围。当前测试数据使用两片共享的
1%–99% 线性拉伸转换为 8 位：输入值 `112` 映射到 `0`，输入值 `406` 映射到 `255`，范围外截断，
范围内按 `round((value - 112) / (406 - 112) * 255)` 计算。共享阈值保持左右片的相对亮度关系。
转换参数也写入每个 TIFF 的 `PLASCAN_*` 默认 metadata；RPC 和嵌入的 GML 内容保持不变。

PlaScan 可通过 `loadRpcCameraFromRaster()` 直接读取 TIFF 内嵌 RPC，不需要额外的 RPB/XML 旁车文件。
空中三角测量会把全 RPC00B 批次分流到固定传感器模型的 RPC 地面点空三；针孔 BA、固定光心极线
校正和针孔 MVS 不得把这组 RPC 静默降级成针孔相机。RPC 稀疏几何使用 `RpcCameraModel` 和
`intersectRpcObservations()`，没有 GCP 时只优化地面点并保留厂商 RPC。

使用 GDAL 检查：

```powershell
gdalinfo -mdd RPC Images\img_01.tif
gdalinfo -mdd RPC Images\img_02.tif
```

## 验证结果

- 两片均可由 GDAL/rasterio 作为单波段 `uint8` `GTiff` 打开，灰度范围为 0–255；
- 每片均有完整的 10 个归一化参数和 4×20 个 RPC 系数；
- 在 `HEIGHT_OFF = 1295 m` 上执行 RPC 像点—地面—像点往返，误差约为半个像素；该半像素来自
  GDAL 的像素中心约定；
- OpenCV 5.0.0 SIFT 检查得到 2540 组比值筛选匹配和 2423 个基础矩阵 RANSAC 内点，内点率
  95.39%，确认 8 位转换后的左右片仍有强重叠。

## 已生成的 DEM / DOM

`Products/` 是 PlaScan 的 `rpc_stereo_products_cli` 直接使用本目录两片影像生成的回归产品：

- `dem.tif`：单波段 Float32、高程单位米、WGS84 椭球高，投影为 EPSG:32740；
- `dom.tif`：四波段 RGBA、8 位、带相同 EPSG:32740 地理参考；
- `stereo_points.ply`：RPC 双像前方交会后的栅格点云；
- `dem_error.tif`、`dem_count.tif`、`dem_confidence.tif`、`dem_coverage.tif`：直接观测质量栅格；
- `dem_preview.png`、`dom_preview.png` 与两个 JSON 报告：快速检查和可追溯记录。

生成命令：

```powershell
rpc_stereo_products_cli `
  --left Images\img_01.tif `
  --right Images\img_02.tif `
  --output Products `
  --resolution 2.0
```

本次结果使用 20,000 个 SIFT 特征上限，得到 3,577 组双向比值匹配、3,452 个基础矩阵内点和
3,452 个通过 RPC 前方交会及高程过滤的地面点；中值重投影误差为 0.255 px。DEM 为 261×270、
2 m 网格，范围为 359666.96875–360188.96875 m E、7651463–7652003 m N，高程范围为
2267.841–2377.307 m。3,169 个网格有直接立体观测，其余有效高程由 terrain 模块的鲁棒栅格聚合和
孔洞填充得到，因此这是一版稀疏同名点约束的摄影测量 DEM，不应把插值区误解为逐像素密集匹配精度。
DOM 与 DEM 严格共格网，有效像元 70,214/70,470，覆盖率 99.637%。

## 来源与许可提示

- 上游仓库：https://github.com/centreborelli/s2p
- 固定版本：`75ea354dc7de658bf696eabf11944e92c65412c5`
- 原始目录：`tests/data/input_pair/`
- S2P 仓库许可证：BSD-3-Clause，文本见 `LICENSE.s2p.txt`。

上游仓库直接分发这两个测试 TIFF，但没有在数据目录中单独说明 Pléiades 原始影像的权利范围。
本地测试时应保留来源和许可证文件；若要把卫星影像重新发布到公开数据集、安装包或第三方镜像，
应另行确认原始影像的再分发权限，不能仅凭软件仓库许可证推定商业卫星影像权利。
