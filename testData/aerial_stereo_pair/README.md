# 航空对地立体像对

这是从 Agisoft “Aerial images (with GCPs)”公开样例中选取的一个连续航带像对，数据提供方为
GeoScan LLC。两张影像由 Sony DSC-RX1R 航空相机垂直对地拍摄，内容为林地和农田，可用于快速验证
特征提取、像对匹配、相对定向、三角化和小规模 MVS。

## 内容

- `Images/`：两张 1600×1067 JPEG 测试影像。它们由原始 6000×4000 影像等比例缩小得到。
- `cameras/`：与缩小影像严格对应的 ASP/Vision Workbench TSAI 针孔相机文件，包含内参、Brown
  畸变参数以及局部重建坐标系下的相机中心 `C` 和旋转矩阵 `R`。
- `Metadata/Cameras_WGS84.txt`：原始相机 GNSS/IMU 记录，坐标为 WGS84 纬度、经度和椭球高。
- `Metadata/GNSS_offset.txt`：原样保留的数据集 GNSS 天线偏心量。
- `image_camera.lis`：影像与 TSAI 相机一一对应的 PlaScan 输入清单。
- `images.lis`：仅影像路径的输入清单。
- `manifest.json`：来源、缩放、像对质量和 SHA-256 完整性信息。

## 坐标与相机说明

两张影像的 GNSS 位置水平距离约为 25.18 m，三维距离约为 25.20 m。WGS84 记录用于地理定位或
参考预选；TSAI 中的 `C`/`R` 是已解算的局部重建坐标，不应直接当作经纬度或米制地图坐标。
本像对提供的是面阵针孔相机模型，不包含 RPC。

缩放采用 `scale_x = 1600 / 6000`、`scale_y = 1067 / 4000`。TSAI 的 `fu`、`fv`、`cu`、`cv`
已经分别按这两个实际比例更新，不需要再次缩放。

## 已验证的像对质量

使用 OpenCV 5.0.0 SIFT（最多 8000 特征/片）、0.75 Lowe 比值和 1.5 px 基础矩阵 RANSAC 阈值检查：

- 比值筛选匹配：1539；
- 基础矩阵内点：1496；
- 内点率：97.2%；
- 内点在 8×6 网格中的覆盖：左片 39/48，右片 42/48。

这些数字用于确认像对重叠和几何一致性，不作为固定的算法精度门限。

## 来源与使用提示

- 官方样例页：https://www.agisoft.com/downloads/sample-data/
- 官方下载包：https://download.agisoft.com/datasets/aerial_images_with_gcps.zip
- 数据归属：GeoScan LLC（原包 `info.txt`）

Agisoft 样例页公开提供下载，但该页面没有列出独立的再分发许可证。若要将这两张影像提交到公开仓库、
发布包或第三方数据镜像，请先向数据权利方确认再分发条款；在本地开发和测试中保留本说明与来源信息。
