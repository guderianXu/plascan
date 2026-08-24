# Terrain Module

`src/core/terrain` 把密集点云、网格、深度产品和已解算相机转换为正式 DEM/DOM 地形成果。
DEM/DOM、质量栅格、有效参数和覆盖统计都会进入项目结果记录，而不是只作为临时导出文件。

## DEM Aggregation

- `DemGridAggregator` 将样本聚合到规则网格，支持 mean、median、min、max、count、standard
  deviation、NMAD、P80、confidence-weighted 和 inverse-error-weighted。
- `DemGridData` 在数据可用时携带高程、颜色、点数、置信度、覆盖率和误差矩阵。
- 置信度作为正权重；存在三角化或源误差时，可按逆误差对网格单元加权。

## Product Manifest

- `TerrainProductManifest` 保存 GUI 和项目元数据消费的地形产品路径与属性。
- 一次 DEM 任务可写出 `dem.tif`、`dem_error.tif`、`dem_count.tif`、
  `dem_confidence.tif` 和 `dem_coverage.tif`。
- 产品记录包含 `dem_path`、`dom_path`、`error_path`、`count_path`、
  `confidence_path`、`coverage_path`、投影、网格分辨率、聚合模式和预览路径。

## DEM Mosaic And Legacy DOM

- `DemMosaic` 支持 first、last、mean、median、min、max、confidence-weighted 和
  inverse-error-weighted 的多 DEM tile 融合。
- 同网格 mosaic 的逐像元融合可使用 CUDA 或 OpenCL；GPU kernel 同时输出高程、有效/覆盖
  掩膜、贡献计数、平均置信度和平均三角化误差。median 在设备上执行确定性的次序统计，
  不会为每个像元分配可变长设备内存。
- 分块处理避免大范围 DEM mosaic 必须一次性全部驻留内存。
- `DomGenerator` 保留旧的对齐影像合成及纹理网格 DOM 能力；项目内“生成正射影像”入口在
  有项目相机元数据时走下述 `OrthoProjector` 相机反投影链路。

## Orthophoto Workflow

### 模块与数据流

```text
MapProjectDialog
  -> ProjectTerrainProductsManager / GuiTaskRunner
  -> TerrainPipeline::estimateOrthoProduct() 或 generateOrthoProduct()
  -> OrthoGenerationOptions
  -> OrthoProjector::planOutputGrid()
  -> OrthoProjector::buildImageInputs()
  -> OrthoProjector::project()
  -> DemDomIO::writeDomGeoTiff()
  -> project_results.ortho_results[]
```

- `OrthoGenerationOptions` 负责从 JSON 解析、校验和回写已生效参数；旧的单值
  `resolution` 仍可作为 X/Y 像元大小兼容输入。
- `DemDomIO::readDemMetadata()` 只读取 DEM 尺寸、地理变换和投影，用于对话框估算，
  不加载高程波段。
- `OrthoProjector::planOutputGrid()` 将用户区域与 DEM 外边界相交，解析最终 X/Y 像元、
  宽高、范围和预计内存。默认像素上限为 100,000,000，超限、无交集或非法范围都会明确失败。
- `OrthoProjector::project()` 在输出网格中心采样 DEM 高程，将三维点投影到已选择的有效相机，
  双线性采样影像，并记录有效表面、直接影像覆盖和孔洞填充掩膜。
- `TerrainPipeline` 把后台阶段和百分比回传 GUI，并在安全检查点响应取消；任务完成后把
  已解析参数、网格、相机贡献数和覆盖统计写入 `ortho_results`。

### CPU / CUDA / OpenCL 后端

- `compute_backend` 接受 `auto`、`cpu`、`cuda`、`opencl`，`compute_device_index` 为 `-1`
  时扫描并选择对应后端的首个兼容 GPU。默认 `auto` 按 CUDA、OpenCL、CPU 顺序解析。
- CUDA/OpenCL kernel 完成输出像元的 DEM 双线性采样、Brown-Conrady 相机投影、项目蒙版判断、
  影像双线性采样、视角/边缘/锐度权重，以及 mosaic、weighted-average、first-valid 融合。
  相机影像和参数在一次投影内打包上传，输出影像与掩膜在 kernel 完成后一次下载。
- 显式 `cuda` 或 `opencl` 不会静默改用 CPU：构建未包含后端、设备索引无效、驱动错误、
  内存不足或 kernel 失败都会直接返回错误。`auto` 才会继续尝试下一后端并记录原因。
- OpenCL 正射路径使用双精度世界坐标，设备必须提供 `cl_khr_fp64` 或 `cl_amd_fp64`；这避免
  大地坐标转成单精度后丢失亚像元精度。DEM mosaic kernel 只使用单精度，不要求 FP64，
  因而可使用更广泛的 OpenCL GPU。
- 点云着色 DOM 与无相机参数的兼容 DOM 当前仅有 CPU 路径；显式 GPU 会失败，`auto` 会记录
  实际 CPU 后端及回退原因。
- `ghost_filter` 的每像元可变长鲁棒中值筛选目前保留在 CPU。显式 GPU 加该选项会明确失败；
  `auto` 会回退 CPU。孔洞连通域和颜色传播也在投影 kernel 完成后运行于 CPU。
- `OrthoProjectionResult::computeExecution` 和流水线 JSON 的 `compute_backend`、
  `compute_device_index`、`compute_device_name`、`compute_fallback_reason` 报告实际执行设备，
  不以请求值冒充实际值。
- GPU kernel 作为一个有界批次提交；取消在上传前、kernel 返回后和后处理期间检查，不能抢占
  已经提交的单个 kernel。CPU-only 构建始终保留完整功能。

### 当前支持范围

| 维度 | 当前支持 | 尚未实现 |
|---|---|---|
| 投影 | `dem_grid`，地理/本地坐标跟随 DEM 网格与 WKT | 平面、圆柱 |
| 表面 | `dem` | 模型、点云等其他表面 |
| 颜色来源 | `images` | 模型纹理、分类或其他栅格 |
| 混合 | `mosaic`、`weighted_average`、`first_valid` | 全局接缝线优化 |

GUI 会显示“平面”“圆柱”和“完善接缝线”以明确后续扩展位置，但这些控件当前禁用，也不会
提交伪参数。核心遇到非 `dem_grid`、非 `dem` 或非 `images` token 时同样立即报错。

### 输出网格与区域

- `pixel_size` 模式支持独立的 `pixel_size_x` / `pixel_size_y`；未显式给值时核心沿用 DEM
  X/Y 像元，而 GUI 会先读取 DEM 并显示真实生效值。
- `maximum_dimension` 模式保持 DEM 的 X/Y 像元比例，把最长边限制到指定像素数；该模式不会
  为小于限制的 DEM 强制上采样。
- `bounds_enabled` 可指定 `min_x/min_y/max_x/max_y`，实际范围始终裁剪到 DEM 外边界。
- 估算结果包含最终像元、边界、`width`、`height` 和 `estimated_memory_bytes`；GUI 同步显示
  总尺寸和预计处理内存。

### 颜色与覆盖处理

- `mosaic` 为每个输出像元选择视角、影像边缘距离与可选锐度综合权重最高的候选；
  `weighted_average` 按该权重融合全部候选；`first_valid` 按输入顺序采用首个有效候选。
- `color_correction` 以各影像平均亮度的中值为目标做有界增益校正；`sharpness_weighting`
  （GUI 文案为“启用离焦过滤”）按拉普拉斯锐度降低模糊影像权重。
- `ghost_filter` 在多候选模式下使用颜色中值和鲁棒离差移除明显离群候选。
- `use_project_masks` 读取项目影像记录中的 `mask_path`，非零蒙版像素不参与采样。
- `fill_holes` 只填充 DEM 有效表面内部、面积不超过 `hole_fill_max_area` 的小连通孔洞，
  使用 `hole_fill_radius` 邻域颜色传播；它不会填充输出边界外或大面积无覆盖区域。
- 若所有有效 DEM 像元都没有相机影像覆盖，任务以“正射投影没有产生有效像素”失败，不会
  写出一张全黑影像。`coverage_ratio` 统计孔洞填充前的直接影像覆盖比例。

### 输出

- `.tif/.tiff` 使用 `DemDomIO::writeDomGeoTiff()` 写出与最终网格一致的地理变换和 DEM
  投影 WKT。OpenCV 内部 BGR 会转换为 GeoTIFF 的 R/G/B 波段顺序，并追加有效覆盖 Alpha，
  设置对应 color interpretation，使 GIS 能区分无覆盖区与真实黑色。
- `.png` 可作为非地理参考图像输出；需要保留网格范围和坐标参考时应使用 GeoTIFF。
- `ortho_projector_v1` 当前未实现逐相机 DEM 遮挡/z-buffer 判定；陡峭地形可能出现被遮挡表面
  采到前景颜色的情况，应结合覆盖 Alpha 和质量检查复核。

## GUI Expectations

- “创建正射影像”对话框应显示真实 DEM 坐标系、X/Y 像元、边界、输出宽高和内存估算。
- 有效相机影像默认勾选；缺少相机参数的影像保留在列表中但禁用。界面同时显示项目蒙版
  就绪数量，没有蒙版时“使用项目蒙版”禁用。
- 长任务必须在后台执行，参数运行期间锁定，同时保留阶段进度、安全取消和失败后重试。
- 项目树应把 DEM、DOM、error、count、confidence 和 coverage 成果显示为独立子节点，
  并按自然文件名排序、从项目元数据刷新。

## Focused Tests

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_ortho_generation test_map_project_dialog
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "OrthoGeneration|OrthoGridPlanner|OrthoProjector|MapProjectDialog" --output-on-failure
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "TerrainDemDom|DemGridAggregator|DemMosaic|TerrainProductManifest|DemQualityRasters" --output-on-failure
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DataTreeWidgetTest\.DemSectionShowsQualityRasterProducts" --output-on-failure
```
