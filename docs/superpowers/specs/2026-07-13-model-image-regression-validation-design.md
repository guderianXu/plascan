# 模型影像回归验收设计

## 目标

为 PlaScan 建立可重复的模型质量验收链路，分别覆盖 Dino 环绕摄影和无人机对地摄影。
验收不能只检查模型文件是否生成，而要从未参与重建的相机视角渲染模型，与原始影像进行
轮廓、覆盖、结构和外观对比；存在 Metashape 参考点云时，再补充三维几何距离统计。

## 验收范围

- Dino：使用 `middlebury_dino_sparse_ring` 原始影像与相机，按环绕角度留出约 20% 视角。
- 无人机：使用 `agisoft_aerial_gcps_small` 的 3x3 九张影像与 TSAI 相机，按空间位置留出
  中心和边缘代表视角；全量 444 张只在九图验收通过后运行。
- 验收对象包括深度图生成的模型、模型的颜色，以及可选的密集点云或 Metashape 参考点云。

## 总体方案

采用 A+C 组合：

1. A，影像空间回归：模型从留出相机视角进行 CPU 离屏光栅化，输出原图、模型渲染、
   半透明叠加和误差热图，并计算图像质量指标。
2. C，参考几何回归：若提供 Metashape 点云，先用项目坐标或显式 Sim3/ICP 对齐，再计算
   双向最近邻距离、覆盖率和百分位误差。

渲染和指标计算放在 `src/core/qc`，不依赖 Qt GUI、OpenGL 或当前窗口状态。CLI 与未来 GUI
报告只调用同一个核心服务。

## 数据划分

### Dino

- 按相机中心绕目标的方位角排序。
- 每五个连续视角留出一个，保证留出视角覆盖完整环绕范围。
- 留出视角不参与深度估计、融合和模型生成，只用于验收渲染。

### 无人机九图

- 九张影像按 TSAI 相机中心投影到主平面并划分为 3x3 网格。
- 留出中心格一张和边缘格一张；若会破坏最小重建连通性，则只留出中心格一张。
- 留出影像不参与模型生成，但相机参数用于最终渲染。

划分结果写入报告，包含训练和留出影像文件名，保证后续回归使用同一集合。

## 核心接口

新增 `ModelImageQualityEvaluator`：

```cpp
struct ModelImageQualityOptions
{
    QString meshPath;
    QVector<ModelValidationView> validationViews;
    QString referenceCloudPath;
    QString outputDirectory;
    int maximumRenderDimension = 1600;
    bool alignReferenceCloud = false;
};

struct ModelImageQualityResult
{
    bool ok = false;
    QVector<ModelViewQuality> views;
    ModelGeometryQuality geometry;
    QJsonObject summary;
    QString error;
};

class ModelImageQualityEvaluator
{
public:
    ModelImageQualityResult evaluate(const ModelImageQualityOptions &options) const;
};
```

`ModelValidationView` 包含原始影像路径、`PositiveDepthCameraModel`、前景掩膜策略和视图标识。
`ModelViewQuality` 至少记录有效覆盖率、轮廓 IoU、边缘 P50/P90 偏差、前景 SSIM、前景 PSNR
和渲染耗时。

## 离屏渲染

- 读取 PLY 三角网格的顶点、面、顶点颜色和法线。
- 使用 `PositiveDepthCameraModel::projectWithDepth()` 投影三角形。
- CPU tile-based z-buffer 光栅化，按图块并行，避免写入冲突。
- 颜色优先使用顶点色插值；没有颜色时输出独立的几何法线渲染，但外观指标标记为不可用。
- 渲染背景为无效值，不用黑色替代，避免 Dino 黑背景提高 SSIM。
- 输出 `source.png`、`render.png`、`overlay.png`、`error_heatmap.png` 和 `valid_mask.png`。

## 图像指标

- 有效覆盖率：模型有效像素与参考前景像素的交集占参考前景面积比例。
- 轮廓 IoU：模型有效掩膜与参考前景掩膜的交并比。
- 边缘偏差：双方轮廓距离变换的对称距离，报告 P50 和 P90 像素。
- 前景 SSIM/PSNR：仅在交集有效区域计算，并在比较前做一次稳健的亮度/对比度线性校正。
- 漂浮面率：模型投影落在参考背景区域的像素占模型有效像素比例。

Dino 参考前景由暗背景阈值、最大连通分量和孔洞填充生成。无人机不使用前景分割，覆盖率以
有效影像范围为基准，结构误差使用灰度梯度和道路/地块边缘进行评估。

## 参考几何指标

- 读取待测模型顶点和 Metashape 参考点云。
- 若两个数据已在同一工程坐标系，直接比较；只有显式指定时才进行 Sim3/ICP 对齐。
- 使用 PlaPoint KDTree 计算模型到参考、参考到模型的双向最近邻距离。
- 报告 RMSE、P50、P84、P95、双向覆盖率和异常点比例。
- 距离阈值由参考点云中位点间距归一化，同时保留原始坐标单位结果。

## 通过标准

### Dino

- 所有留出视角轮廓 IoU 中位数不低于 0.90。
- 有效覆盖率中位数不低于 0.90。
- 边缘 P90 中位数不高于 3 像素（在最大边 1600 像素的渲染尺度下）。
- 前景 SSIM 中位数不低于 0.75。
- 最大网格连通分量占比不低于 0.85，且不得存在占包围盒对角线 5% 以上的漂浮分量。

### 无人机九图

- 所有留出视角有效覆盖率中位数不低于 0.90。
- 结构边缘 P90 中位数不高于 3 像素。
- 有效区域 SSIM 中位数不低于 0.70。
- 模型不得出现跨航带漂浮面、明显折叠或主要连通分量低于 0.85。

任何硬性几何门控失败时，即使外观指标达到阈值也判定失败。

## CLI

新增 `model_quality_cli`，核心参数：

```text
--mesh <model.ply>
--project <project.plascan> 或 --image-camera-list <image_camera.lis>
--validation-split angular-20|grid-center-edge|all
--reference-cloud <optional.ply>
--output-dir <directory>
--max-render-dim 1600
--align-reference-cloud
```

退出码：0 表示达标，2 表示输入或解析错误，3 表示成功完成评估但质量不达标。

## 输出

- `model_quality_report.json`：完整配置、数据划分、逐视图指标、几何指标和最终判定。
- `model_quality_views.csv`：逐视图指标，便于回归比较。
- `comparisons/<view>/`：原图、渲染、叠加、热图和掩膜。
- `contact_sheet.png`：留出视角的四联图总览，供人工复核。

## 错误处理

- 相机无效、影像缺失、网格没有面、模型完全不在相机前方时立即返回明确错误。
- 网格没有颜色时继续做几何验收，但外观指标显示 `unavailable`，不能据此判通过。
- 参考点云缺少或坐标系不一致时，不静默对齐；几何 C 项标记为跳过并说明原因。
- 单个视图失败不会隐藏其它视图结果，但最终状态判失败。

## 测试与数据验证

- 单元测试：投影、z-buffer 遮挡、颜色插值、掩膜指标、边缘距离和判定阈值。
- 合成测试：已知彩色平面/立方体从指定相机渲染，结果与解析解比较。
- Dino：运行留出视角 A 验收，并输出可视化接触表。
- 无人机九图：运行留出视角 A 验收；若存在 Metashape 参考点云，再运行 C 验收。
- 九图不达标时不启动 444 张全量模型重建，先定位空三、深度图、融合或网格中的首个失败阶段。

## 非目标

- 本轮不以神经渲染、视图合成或图像生成替代真实网格质量。
- 不通过放宽阈值、隐藏无效区域或只选择最好视角让失败模型通过。
- 不要求不同曝光和阴影下逐像素完全相等；“几乎一致”定义为几何轮廓、结构位置、覆盖和
  稳健校正后的外观指标同时达到上述阈值。
