# PlaScan 多视图纹理映射 v4 完整实施计划

> 状态：已实施（2026-07-30）
> 日期：2026-07-29
> 适用入口：工作流程 → 生成纹理
> 当前基线：`camera_projected_atlas_v3`
> 目标算法：`camera_projected_atlas_v4`
> 主构建目录：`E:\code\plascan\build\windows-vcpkg-cuda-release`

## 实施结果

已完成 `camera_projected_atlas_v4` 主链：原图/证据相机分离、七点可见性检查、
top-K 候选、ICM 标签优化、小孤岛合并、连通投影 chart、确定性 MaxRects 打包、
逐纹素多视图融合、鬼影过滤、曝光校正、纹理空间填充、锐化、取消检查以及
OBJ/MTL/PNG 临时文件提交。GUI 已开放实际生效的影像下采样、混合模式、孔洞填充、
重影过滤、焦外过滤、色彩校正、边距和锐化参数，稳定英文配置键由
`textureConfigFromSettings()` 统一解析。

首版为控制模块数量，将数据契约集中在 `TextureMapper.h` 与
`TextureMappingV4Internal.h`，将光度校正合入输入预处理，将后处理和输出合入
`TextureAtlasBaker.cpp`；职责边界与本计划一致，但未机械拆成每项一个公开类。
当前为单页 PNG 图集；超过安全内存估算的 16384 配置会在分配前失败。

验证结果（2026-07-30）：

- `cmake --build . --target test_mesh_reconstructor plascan_gui -j 4`：通过。
- `test_mesh_reconstructor.exe --gtest_filter=TextureMapperTest.*`：8/8 通过。
- `test_gui_project_utils.exe`：596 项通过，1 项因未设置
  `PLASCAN_OBJ_BENCHMARK_PATH` 跳过。
- `test_mesh_reconstructor.exe`：172 项中 171 项通过；唯一失败为既有
  `DepthTsdfSurfaceBuilderTest.BuildsFiniteSurfaceFromConfidenceWeightedPlane`
  对 TSDF 进度文案的断言，不涉及纹理代码，单独复跑结果一致。

## 1. 目标

为 PlaScan 增加可实际用于行星表面和常规摄影测量模型的多视图纹理生成能力。新实现必须：

1. 从已定向影像、相机、深度、置信度和支持掩模中生成 OBJ、MTL 和纹理图。
2. 保持颜色影像的有效分辨率，不再把每张完整影像平均缩入固定网格后再引用。
3. 通过深度一致性、遮挡、视角、投影分辨率和影像质量选择可靠纹理来源。
4. 让相邻三角面尽量使用一致的主相机，减少碎片化纹理块和高频接缝。
5. 支持“最佳视角”“Natural”“加权平均”三种真实生效的混合模式。
6. 支持图块打包、边界扩张、受证据约束的孔洞处理和可选色彩校正。
7. 在 GUI 中只开放已经由核心实现且能够持久化、复现的参数。
8. 输出足够的统计和诊断产物，使纹理覆盖、接缝、回退和失败原因可审计。
9. 保留当前 v3 作为阶段性回退路径，直到 v4 通过合成测试和真实数据非回归。

## 2. 当前基线与主要缺陷

当前实现位于：

- `src/core/mesh/TextureMapper.h`
- `src/core/mesh/TextureMapper.cpp`
- `src/core/mesh/CameraTextureMapper.cpp`
- `src/core/mesh/ModelWorkflowService.cpp`
- `src/gui/dialogs/TextureMappingDialog.*`
- `src/gui/project/manager/ProjectModelManager.cpp`

`camera_projected_atlas_v3` 已经具备：

- PLY/OBJ 网格读取。
- 按相机投影生成逐面 UV。
- 基于深度、置信度、支持掩模和视角的候选相机评分。
- 严格候选失败后的宽松回退。
- 基于顶点票数的局部相机一致性调整。
- 单张 PNG 图集、OBJ 和 MTL 输出。
- 映射面、回退面、一致性调整面和未映射面统计。

当前主要缺陷：

1. 每张完整源影像被缩放到规则图集单元，相机数量越多，单张影像可用分辨率越低。
2. 颜色影像在 `DepthTsdfSurfaceBuilder::loadFrames()` 中被直接缩放到深度图尺寸，
   无法利用原始影像细节。
3. 面—相机搜索复杂度接近 `O(faceCount × viewCount)`，没有候选上限和粗筛索引。
4. 深度绝对容差使用 `meshDiagonal / 64`，大型行星场景可能产生过宽容差。
5. 视图一致性只根据共享顶点票数局部修正，没有显式的相邻面能量和接缝颜色代价。
6. 图集没有真正的 chart 裁剪和打包，大量纹理面积被未使用的影像区域占据。
7. GUI 中的 Natural、影像下采样、孔洞填充、重影过滤和色彩校正尚未映射到核心行为。
8. 没有曝光校正、局部清晰度评分、颜色异常值剔除和接缝质量报告。
9. 没有图集占用率、有效纹素密度、接缝长度和失败分类等质量指标。
10. 无深度源时会回退到顶点色平面烘焙，但 GUI 没有明确告知真实算法发生变化。

## 3. 非目标和安全边界

- v4 第一阶段不引入新的第三方 Graph Cut、UV 展开或图集打包库。
- 不修改 `3rdparty/plapoint` 和 `3rdparty/plamatrix` 子模块。
- 不用纹理填充掩盖网格本身的大型几何孔洞。
- 不把深度不一致或落在支持掩模外的背景影像强行投到模型上。
- 不默认修改科学影像的辐射特性；自动色彩校正必须可关闭且默认保守。
- 不在 GUI 中展示尚未由核心消费的“假参数”。
- 不覆盖已有 `textured_model.obj` 和纹理图后才报告失败；输出必须先写临时文件并原子替换。
- 不在主线程执行纹理候选计算、图集打包或纹理烘焙。
- 不要求 v4 首版支持任意导入模型的多材质合并；先支持 PlaScan 当前单网格主流程。
- 不把 16384 纹理视为无条件安全选项；执行前必须进行内存预算。

## 4. 验收指标

### 4.1 功能验收

- “工作流程 → 生成纹理”可以从项目最新有效模型启动独立纹理任务。
- 有相机与深度证据时使用 `camera_projected_atlas_v4`。
- 无深度证据时必须明确提示并要求用户确认是否使用顶点色回退，不得静默降级。
- 成功结果包含 OBJ、MTL、至少一张纹理图和 JSON 质量报告。
- 重新打开 `.plascan` 项目后仍可定位并显示带纹理模型。
- 任务支持进度、取消、错误提示和项目切换保护。

### 4.2 质量验收

在具有有效影像支持的三角面集合上：

- 严格证据映射率不低于 v3。
- 背景或掩模外采样在合成测试中为 0。
- 未映射面比例不高于 v3；若提高，报告必须证明是更严格的错误采样拒绝。
- 相机标签小孤岛面积占比低于 1%，或较 v3 至少下降 80%。
- 图集有效纹素占用率目标不低于 65%。
- 中位有效纹素密度不低于 v3，目标提升至少 2 倍。
- Natural 模式的接缝 Lab 色差 P95 较“最佳视角”下降至少 30%。
- 关闭色彩校正时，单视图区域颜色不得发生全局增益或偏移。
- 相同输入、配置和线程策略下，标签、chart 布局和结果统计可重复。

### 4.3 性能验收

- 面候选存储只保留前 `K` 个视图，默认 `K=4`。
- 200k 面、100 视图的候选阶段不得分配 `faceCount × viewCount` 的稠密浮点矩阵。
- 8192 RGBA 图集模式峰值内存目标低于 2 GiB。
- 16384 模式必须根据实际缓冲数量预估内存；超出安全预算时在启动前失败。
- 所有长循环按固定间隔检查取消标志。

## 5. 总体架构

保留 `TextureMapper` 作为稳定门面，将当前超过单一职责的实现拆为以下模块：

| 模块 | 新文件 | 职责 |
|---|---|---|
| 数据契约 | `TextureMappingTypes.h` | 配置、输入视图、候选、chart、统计和结果 |
| 输入准备 | `TextureSourcePreprocessor.h/.cpp` | 分辨率坐标、影像降采样、清晰度图、曝光采样 |
| 网格邻接 | `TextureMeshAdjacency.h/.cpp` | 面几何、共享边、连通关系和中位边长 |
| 可见性评分 | `TextureVisibilityEvaluator.h/.cpp` | 投影、掩模、深度、遮挡和 top-K 候选 |
| 标签优化 | `TextureViewLabelOptimizer.h/.cpp` | 主相机初始化、ICM、孤岛清理和接缝代价 |
| chart 构建 | `TextureChartBuilder.h/.cpp` | 按相机与连通域生成投影 chart |
| 图集打包 | `TextureAtlasPacker.h/.cpp` | MaxRects、缩放、页数和占用率 |
| 光度校正 | `TexturePhotometricCalibrator.h/.cpp` | 重叠采样、鲁棒增益和偏移求解 |
| 纹理烘焙 | `TextureAtlasBaker.h/.cpp` | 最佳视角、Natural、加权平均和纹素掩模 |
| 后处理 | `TextureAtlasPostprocessor.h/.cpp` | 小孔、边界扩张、可选锐化 |
| 输出 | `TextureModelExporter.h/.cpp` | OBJ、MTL、PNG、JSON 和原子替换 |
| 编排 | `CameraTextureMapper.cpp` | 阶段编排、进度、取消和 v3/v4 选择 |

`TextureMapper.cpp` 继续负责没有相机数据时的顶点色回退，不能与相机纹理主链混在同一个实现文件。

## 6. 核心数据契约

### 6.1 枚举

```cpp
enum class TextureMappingMode
{
    AutoProjective,
    KeepExistingUv
};

enum class TextureBlendMode
{
    BestView,
    Natural,
    WeightedAverage
};

enum class TextureHoleFillMode
{
    Disabled,
    TextureSpaceSmallHoles,
    NeighborViewRecovery
};
```

字符串序列化使用稳定的英文键：

- `auto_projective`
- `keep_existing_uv`
- `best_view`
- `natural`
- `weighted_average`
- `disabled`
- `texture_space_small_holes`
- `neighbor_view_recovery`

GUI 文案可以中文化，但项目 JSON、CLI 和报告不得依赖界面显示字符串。

### 6.2 配置

```cpp
struct TextureMappingConfig
{
    int textureSize = 8192;
    int maximumTexturePages = 1;
    int imageDownscale = 2;
    int padding = 8;
    int maximumCandidateViews = 4;
    int maximumBlendedViews = 3;
    int labelOptimizationPasses = 6;
    int minimumChartFaces = 8;

    float minimumConfidence = 0.25f;
    float minimumViewCosine = 0.20f;
    float relativeDepthTolerance = 0.005f;
    float edgeLengthDepthTolerance = 2.0f;
    float labelSmoothness = 0.35f;
    float labelColorPenalty = 0.50f;
    float coherentReplacementRatio = 0.65f;
    float ghostLabThreshold = 20.0f;
    float sharpeningStrength = 1.0f;

    TextureMappingMode mappingMode = TextureMappingMode::AutoProjective;
    TextureBlendMode blendMode = TextureBlendMode::Natural;
    TextureHoleFillMode holeFillMode =
        TextureHoleFillMode::TextureSpaceSmallHoles;

    bool enableGhostFilter = true;
    bool enableOutOfFocusFilter = false;
    bool enableColorCorrection = false;
    bool useAssignedImages = false;
    bool keepUnmapped = true;
    bool enableV4 = true;

    std::function<void(const std::string &, int)> progressFn;
    std::function<bool()> isCancelled;
};
```

配置解析必须集中在 `ModelWorkflowService::textureConfigFromSettings()`，并完成：

- 数值范围约束。
- 枚举字符串校验。
- 参数组合校验。
- 实际生效配置回写到结果报告。
- 旧版设置兼容迁移。

### 6.3 输入视图

不能继续假设颜色图、深度图和相机投影天然处于同一分辨率。

```cpp
struct TextureSourceView
{
    int sourceIndex = -1;
    QString imagePath;
    Camera colorCamera;
    Camera evidenceCamera;

    cv::Mat colorBgr;
    cv::Mat depth;
    cv::Mat confidence;
    cv::Mat depthValidMask;
    cv::Mat supportMask;
    cv::Mat sharpnessMap;

    float qualityWeight = 1.0f;
    float exposureGain = 1.0f;
    cv::Vec3f colorGain{1.0f, 1.0f, 1.0f};
    cv::Vec3f colorBias{0.0f, 0.0f, 0.0f};
    bool assigned = true;
};
```

要求：

- `colorCamera` 投影到 `colorBgr` 坐标。
- `evidenceCamera` 投影到深度和掩模坐标。
- 如果相机类支持按比例缩放内参，输入准备阶段显式构建两个相机。
- 如果不能可靠恢复颜色相机，任务应失败并指出相机文件和影像尺寸，不得猜测比例。

现有 `MeshColorView` 暂时保留供网格顶点着色使用；v4 使用独立的 `TextureSourceView`，
避免继续扩大颜色器和纹理器之间的隐式耦合。

### 6.4 候选、标签和 chart

```cpp
struct TextureFaceViewCandidate
{
    int viewIndex = -1;
    float totalScore = 0.0f;
    float depthScore = 0.0f;
    float angleScore = 0.0f;
    float resolutionScore = 0.0f;
    float centerScore = 0.0f;
    float sharpnessScore = 0.0f;
    bool strictEvidence = false;
};

struct TextureFaceAssignment
{
    int primaryViewIndex = -1;
    QVector<TextureFaceViewCandidate> candidates;
    bool relaxed = false;
    bool optimized = false;
};

struct TextureChart
{
    int chartIndex = -1;
    int primaryViewIndex = -1;
    QVector<int> faceIndices;
    QRect sourceBounds;
    QRect atlasBounds;
    float atlasScale = 1.0f;
    int texturePage = 0;
};
```

候选采用每面固定容量的小向量或 `std::array + count`，不得为每个面创建大量堆对象。

### 6.5 结果与统计

扩展 `TextureMappingResult`：

```cpp
struct TextureMappingStatistics
{
    int faceCount = 0;
    int strictMappedFaceCount = 0;
    int relaxedMappedFaceCount = 0;
    int neighborRecoveredFaceCount = 0;
    int unmappedFaceCount = 0;
    int optimizedFaceCount = 0;
    int chartCount = 0;
    int texturePageCount = 0;
    int sourceViewCount = 0;
    int usedViewCount = 0;

    std::uint64_t candidateEvaluationCount = 0;
    std::uint64_t rejectedProjectionCount = 0;
    std::uint64_t rejectedMaskCount = 0;
    std::uint64_t rejectedDepthCount = 0;
    std::uint64_t rejectedAngleCount = 0;
    std::uint64_t rejectedResolutionCount = 0;
    std::uint64_t rejectedSharpnessCount = 0;
    std::uint64_t rejectedColorOutlierCount = 0;

    double atlasOccupancy = 0.0;
    double medianTexelDensity = 0.0;
    double seamLength = 0.0;
    double seamLabP50 = 0.0;
    double seamLabP95 = 0.0;
    double peakMemoryEstimateMiB = 0.0;
};
```

结果同时记录：

- 实际算法名和版本。
- 请求配置与实际生效配置。
- 输入网格、影像、相机和证据指纹。
- 每张纹理图路径。
- 报告 JSON 路径。
- 是否发生 v3 或顶点色回退。
- 每个阶段耗时。

## 7. 详细算法

### 7.1 输入发现与预检

1. 解析项目最新模型记录，取得网格路径和对应深度工作区。
2. 从深度工作区 manifest 发现相机、参考影像、深度、置信度和支持掩模。
3. 校验每个输入视图：
   - 颜色图为 `CV_8UC3`。
   - 深度和置信度为 `CV_32FC1`。
   - 掩模为 `CV_8UC1`。
   - 深度相关矩阵尺寸一致。
   - 颜色相机尺寸与颜色图一致。
   - 证据相机尺寸与深度图一致。
4. 丢弃无颜色影像或无有效相机的视图，并记录原因。
5. 少于 1 个有效视图时失败；Natural 模式少于 2 个视图时自动降为 BestView，并写入报告。
6. 读取网格并检查面索引、退化面和法向。
7. 计算中位边长、网格包围盒和内存预算。
8. 所有输出先写入任务临时目录。

### 7.2 影像预处理

1. 按 `imageDownscale` 生成颜色工作图，同时缩放 `colorCamera` 内参。
2. 生成灰度图和低分辨率清晰度图：

   ```text
   sharpness = localVariance(Laplacian(gray))
   ```

3. 清晰度图按 32×32 或 64×64 网格存储，避免保存全分辨率浮点图。
4. 对掩模边缘计算距离场，用于纹理融合时的边缘衰减。
5. 只在开启色彩校正时收集重叠颜色样本。
6. 预处理结果按视图独立，可用 OpenMP 并行。

### 7.3 网格邻接与几何缓存

为每个面缓存：

- 三个顶点索引。
- 质心。
- 单位法线。
- 三维面积。
- 三条边长。
- 相邻面索引。

共享边使用排序后的顶点对作为键。非法或非流形边写入统计，不阻塞纹理生成，但这些边不参与
普通二面邻接平滑。

计算全局中位边长 `h`，用于深度容差：

```text
tau = max(
    edgeLengthDepthTolerance * h,
    relativeDepthTolerance * abs(cameraDepth),
    2 * depthSigma)
```

没有可用 `depthSigma` 时只使用前两项。禁止继续使用 `meshDiagonal / 64` 作为默认容差。

### 7.4 候选相机粗筛

首版不构建复杂空间索引，先做低成本粗筛：

1. 质心必须位于相机前方。
2. 质心投影位于颜色图和证据图范围内。
3. 面法线与观察方向余弦大于最小值。
4. 三个顶点投影后的包围框与影像相交。
5. 投影三角形面积大于最小像素阈值。

粗筛通过后才执行 7 点严格证据检查。

如果真实数据性能不足，再增加第二阶段优化：

- 每个相机先用视锥体筛选网格包围盒。
- 将相机按观察方向和位置构建简单分桶。
- 对超大网格分块处理。

### 7.5 严格可见性与质量评分

对三角形三个顶点、三个边中点和质心共 7 个样本：

1. 投影到证据相机。
2. 检查支持掩模和深度有效掩模。
3. 读取深度和置信度。
4. 计算网格相机深度与观测深度残差。
5. 使用局部容差 `tau` 检查一致性。

任一采样点超出支持掩模或深度残差严重超限时，严格候选失败。

质量分数：

```text
total =
    frameQuality
    * depthScore^2
    * angleScore^4
    * resolutionScore
    * centerScore
    * sharpnessScore
```

其中：

- `depthScore = exp(-0.5 * (residual / tau)^2)`
- `angleScore = clamp(dot(normal, viewDirection), 0, 1)`
- `resolutionScore` 根据投影三角形像素面积归一化
- `centerScore` 根据投影位置到影像中心的归一化半径衰减
- `sharpnessScore` 从清晰度图双线性采样并归一化

如果网格面朝向尚不可信，可在预检阶段检测整体法向一致性；只有检测失败时才允许临时使用
`abs(dot)`，并在报告中写入警告。

每个面只保留分数最高的 `maximumCandidateViews` 个严格候选。

### 7.6 宽松候选与邻域恢复

严格候选为空时，不能立即使用当前 v3 的无深度回退。

恢复顺序：

1. 尝试相邻面主相机，但仍要求：
   - 三角形完整投影。
   - 支持掩模通过。
   - 使用更宽但有上限的深度容差。
2. 尝试普通宽松候选：
   - 仍检查全部支持掩模。
   - 深度只允许最多 2 个采样点缺失。
   - 其余采样点必须满足宽松深度容差。
3. 仍失败则标记未映射。

宽松候选必须单独计数，不能混入严格映射率。

### 7.7 主相机标签优化

以最高质量候选初始化每个面的主相机。

优化能量：

```text
E(labels) =
    sum(unaryCost(face, label))
    + labelSmoothness * sum(pairwiseCost(faceA, faceB))
```

单项代价：

```text
unaryCost = -log(totalScore + epsilon)
```

邻接代价：

```text
0, same label
sharedEdgeWeight * (1 + labelColorPenalty * edgeColorDifference), different label
```

首版使用确定性的 ICM：

1. 固定面遍历顺序。
2. 每个面只尝试自己的 top-K 候选和相邻面当前标签。
3. 只有候选对该面仍满足可见性时才能采用。
4. 执行 6 轮或直到一轮变更率低于 0.1%。
5. 新标签分数不得低于原标签的 `coherentReplacementRatio`。

之后执行孤岛清理：

- 按相机标签查找面连通域。
- 小于 `minimumChartFaces` 的区域尝试并入边界占比最大的相邻标签。
- 合并前重新检查该标签对所有面的投影和支持掩模。
- 不能安全合并时保留原区域。

### 7.8 chart 生成

chart 的定义是“同一主相机下的面连通域”。

对每个 chart：

1. 把所有三角形顶点投影到主相机颜色图。
2. 计算像素包围框。
3. 裁剪到影像范围。
4. 增加 `padding`。
5. 根据 `imageDownscale` 和全局 atlas 缩放计算最终图块尺寸。
6. 保存从源影像坐标到 chart 局部坐标的仿射变换。

chart 内继续使用相机投影 UV，不做 LSCM，因此不会引入额外几何拉伸。

### 7.9 MaxRects 图集打包

实现确定性 MaxRects：

- 按图块最大边、面积、chartIndex 稳定排序。
- 默认使用 Best Short Side Fit。
- 每个图块包含 padding。
- 保持 chart 方向，不旋转；避免方向变化影响调试和复现。

单页放不下时：

1. 如果 `maximumTexturePages > 1`，按页继续打包。
2. 否则估算统一缩放因子：

   ```text
   scale = sqrt(availableAtlasArea / requestedChartArea) * 0.92
   ```

3. 应用缩放后重新打包。
4. 缩放低于安全阈值时失败并提示用户提高纹理大小、增加页数或提高影像下采样。

统计：

- 请求总图块面积。
- 实际有效图块面积。
- padding 面积。
- atlas 占用率。
- 统一缩放因子。

### 7.10 纹理烘焙

每个 atlas chart 独立栅格化。对每个覆盖纹素：

1. 找到所在三角形和重心坐标。
2. 恢复三维位置和法线。
3. 从该面的候选视图中采样颜色。
4. 应用可选曝光和色彩校正。
5. 根据混合模式生成最终颜色。

#### BestView

- 只采样主相机。
- 使用双线性插值。
- 最快且最锐利，用作基线和回退。

#### WeightedAverage

- 采样最多 `maximumBlendedViews` 个严格候选。
- 权重使用候选分数、掩模距离和局部清晰度。
- 直接计算加权均值。

#### Natural

1. 主相机决定 chart 和 UV。
2. 采样主相机及最多 2 个其他严格候选。
3. 将颜色转换到 Lab。
4. 计算加权中值和 MAD。
5. 开启重影过滤时，剔除：

   ```text
   LabDistance > max(ghostLabThreshold, 2.5 * MAD)
   ```

6. 对剩余颜色执行鲁棒加权平均。
7. 如果颜色分歧仍过大，回退主相机。
8. 掩模边缘根据距离场降低权重，减少背景污染。

首版不实现拉普拉斯金字塔多频段融合；只有 Natural 的鲁棒融合仍无法达到接缝验收指标时，
再增加多频段阶段。

### 7.11 自动色彩校正

只在用户启用时执行。

1. 从同时被两个相机严格观察的面中均匀采样三维点。
2. 跳过阴影边缘、过曝、欠曝和颜色异常值。
3. 建立相机重叠图。
4. 选择质量最高且连接度高的相机作为锚点。
5. 鲁棒求解每相机的颜色增益和偏移：

   ```text
   corrected = gain * source + bias
   ```

6. 限制：
   - `gain ∈ [0.7, 1.4]`
   - `bias ∈ [-20, 20]`
7. 重叠图不连通时，每个分量独立锚定，并在报告中告警。

默认只做亮度增益和温和色偏修正，不能把科学影像自动拉成相同色调。

### 7.12 焦外影像过滤

开启后：

- 使用预计算局部清晰度图。
- 候选局部清晰度低于该视图 P10 且另有合格候选时拒绝。
- 如果某面只有一个可见视图，不因焦外过滤直接变成未映射，改为降低权重并记录回退。

### 7.13 孔洞处理和 padding

区分：

1. chart 内部因栅格离散产生的 1–2 像素空洞。
2. chart 边缘需要的纹理 padding。
3. 整个三角面无可靠影像。

处理：

- 小纹素孔洞使用 push-pull 或最近有效纹素传播。
- padding 使用逐轮 8 邻域扩张，最多 `padding` 轮。
- 未映射面只允许邻域相机恢复或默认色，不允许从 atlas 邻近位置任意复制。
- `keepUnmapped=false` 时未映射面写黑色保留块；不能删除几何面。

### 7.14 锐化

在所有融合和孔洞处理完成后执行：

- 只处理有效纹理区域。
- 在 Lab 的 L 通道使用受限 unsharp mask。
- `sharpeningStrength=0` 完全不修改。
- 最大值限制为 2.0。
- padding 区域不单独锐化，使用相邻有效纹素传播结果。

### 7.15 输出与原子提交

任务临时目录：

```text
products/.texture_task_<uuid>/
```

临时产物：

- `textured_model.obj`
- `textured_model.mtl`
- `textures/model_texture_0.png`
- 可选 `textures/model_texture_N.png`
- `texture_mapping_report.json`

全部成功后再移动或替换正式产物。取消或失败时：

- 删除本任务临时目录。
- 保留旧纹理产物和旧项目记录。
- 不把旧结果标记为本次成功产物。

## 8. GUI 参数映射

`TextureMappingDialog` 的控件与核心配置一一对应：

| GUI | JSON 键 | 核心字段 | 首次开放阶段 |
|---|---|---|---|
| 纹理类型：纹理映射 | `textureType` | 固定入口类型 | 已有 |
| 源数据：图像 | `sourceData` | 固定为项目影像 | 已有 |
| 映射模式 | `mappingMode` | `mappingMode` | 阶段 4 |
| 混合模式 | `blendMode` | `blendMode` | 阶段 5 |
| 纹理大小 | `textureSize` | `textureSize` | 已有 |
| 影像下采样 | `imageDownscale` | `imageDownscale` | 阶段 1 |
| 每步保存项目 | `saveEachStep` | 工作流保存策略 | 阶段 6 |
| 孔洞填充 | `holeFillMode` | `holeFillMode` | 阶段 5 |
| 重影过滤器 | `ghostFilter` | `enableGhostFilter` | 阶段 5 |
| 焦外影像过滤器 | `outOfFocusFilter` | `enableOutOfFocusFilter` | 阶段 5 |
| 自动色彩校正 | `colorCorrection` | `enableColorCorrection` | 阶段 5 |
| 使用指定影像 | `useAssignedImages` | `useAssignedImages` | 后续阶段 |
| 转移纹理 | `transferTexture` | 暂不支持 | 保持禁用 |
| 纹理填充边距 | `padding` | `padding` | 已有 |
| 锐化强度 | `sharpeningStrength` | `sharpeningStrength` | 阶段 5 |
| 保留无纹理区域 | `keepUnmapped` | `keepUnmapped` | 已有 |

规则：

- 只有对应阶段完成并通过测试后才启用控件。
- GUI 显示值、JSON 保存值和核心生效值必须一致。
- 不再用中文显示文本作为算法枚举值。
- 对自动降级、内存限制和页数调整显示实际采用值。

## 9. 项目持久化与报告

保留：

```json
"texture_mapping_settings": {}
```

模型结果记录扩展：

```json
{
  "textured": true,
  "texture_algorithm": "camera_projected_atlas_v4",
  "texture_mapping_version": 4,
  "model_obj": "...",
  "model_mtl": "...",
  "texture_pages": ["..."],
  "texture_report": "...",
  "texture_size": 8192,
  "texture_page_count": 1,
  "texture_chart_count": 42,
  "texture_atlas_occupancy": 0.73,
  "texture_strict_mapped_faces": 188400,
  "texture_relaxed_mapped_faces": 4200,
  "texture_unmapped_faces": 600,
  "texture_effective_settings": {},
  "texture_input_fingerprint": "..."
}
```

需要同步检查：

- `.plascan` 归档是否包含多张纹理图和报告。
- 项目移动、另存和重新打开时路径是否正确重写。
- 清理旧模型时是否只删除当前模型关联的纹理资源。
- 数据树和模型查看器是否优先使用带纹理 OBJ。

## 10. 分阶段实施任务

### 阶段 0：冻结 v3 基线与测试夹具

修改：

- `tests/test_mesh_reconstructor.cpp`
- `tests/CMakeLists.txt`
- 新增 `tests/texture/TextureTestFixtures.h/.cpp`
- `scripts/validation/README.md`
- 新增 `scripts/validation/compare_texture_mapping.py`

任务：

1. 固化现有四个 `TextureMapperTest`。
2. 增加 v3 输出统计快照。
3. 新建独立的 `test_texture_mapping` 测试目标，后续 v4 测试不继续扩大
   `test_mesh_reconstructor.cpp`。
4. 合成影像、相机、深度和网格由测试运行时写入临时目录，仓库不提交大体量二进制夹具。
5. 实现纹理报告比较脚本：
   - 映射面比例。
   - chart/标签连通域。
   - atlas 占用率。
   - 接缝 Lab P50/P95。
   - 掩模外颜色泄漏。
6. 准备两相机平面、遮挡台阶、颜色偏差和掩模背景四组合成夹具。

阶段门：

- 当前 v3 测试全部通过。
- 比较脚本能读取报告并对两个结果输出差异。

### 阶段 1：数据契约和颜色/证据分辨率解耦

新增：

- `src/core/mesh/TextureMappingTypes.h`
- `src/core/mesh/TextureSourcePreprocessor.h/.cpp`
- `tests/test_texture_source_preprocessor.cpp`

修改：

- `TextureMapper.h`
- `MeshColorizer.h`
- `DepthTsdfSurfaceBuilder.h/.cpp`
- `ModelWorkflowService.cpp`
- `src/core/mesh/CMakeLists.txt`

任务：

1. 引入新配置枚举和稳定序列化键。
2. 构建 `TextureSourceView`。
3. 保留颜色影像工作分辨率，不再无条件缩到深度尺寸。
4. 显式构建颜色和证据相机。
5. 实现输入尺寸和内存预检。
6. 保持旧 `MeshColorView` 行为不变，避免顶点着色回归。

阶段门：

- 同一世界点在颜色图和证据图中的缩放投影均正确。
- 不同颜色/深度分辨率的合成测试通过。
- 错误相机尺寸产生明确失败信息。

### 阶段 2：可见性评分和 top-K 候选

新增：

- `TextureMeshAdjacency.h/.cpp`
- `TextureVisibilityEvaluator.h/.cpp`
- `tests/test_texture_visibility_evaluator.cpp`

修改：

- `CameraTextureMapper.cpp`
- `src/core/mesh/CMakeLists.txt`

任务：

1. 构建面几何和邻接缓存。
2. 实现局部深度容差。
3. 实现粗筛、7 点严格检查和受限宽松候选。
4. 实现角度、分辨率、中心和清晰度评分。
5. 每面只保留 top-K。
6. 输出拒绝原因统计。

阶段门：

- 遮挡面不能选择被遮挡相机。
- 掩模内部跨越背景的三角形必须拒绝。
- 大场景尺度变化不导致容差按包围盒失控。
- top-K 与全排序参考结果一致。

### 阶段 3：全局一致标签优化

新增：

- `TextureViewLabelOptimizer.h/.cpp`
- `tests/test_texture_view_label_optimizer.cpp`

任务：

1. 实现面标签能量。
2. 实现确定性 ICM。
3. 实现共享边颜色代价。
4. 实现小孤岛检测和安全合并。
5. 保留优化前后标签用于调试。

诊断输出：

- `face_labels_initial.csv`，仅诊断模式。
- `face_labels_optimized.csv`，仅诊断模式。
- 标签连通域统计。

阶段门：

- 合成棋盘式初始标签可收敛为连续区域。
- 不可见的邻居标签不能被强制传播。
- 相同输入重复运行标签完全一致。

### 阶段 4：投影 chart 与图集打包

新增：

- `TextureChartBuilder.h/.cpp`
- `TextureAtlasPacker.h/.cpp`
- `tests/test_texture_chart_builder.cpp`
- `tests/test_texture_atlas_packer.cpp`

任务：

1. 按主相机和面连通域生成 chart。
2. 计算裁剪图块和局部 UV。
3. 实现不旋转的确定性 MaxRects。
4. 实现单页统一缩放。
5. 为多页预留结果结构，但首轮可保持最大页数为 1。
6. 计算占用率和有效纹素密度。

阶段门：

- chart UV 全部位于分配矩形内。
- 任意两个 chart 的含 padding 矩形不重叠。
- 打包结果不依赖哈希容器迭代顺序。
- 同等输入下有效纹素密度显著高于 v3 全图平铺。

### 阶段 5：纹理烘焙、Natural 和高级选项

新增：

- `TextureAtlasBaker.h/.cpp`
- `TexturePhotometricCalibrator.h/.cpp`
- `TextureAtlasPostprocessor.h/.cpp`
- `tests/test_texture_atlas_baker.cpp`
- `tests/test_texture_photometric_calibrator.cpp`
- `tests/test_texture_atlas_postprocessor.cpp`

任务：

1. 实现三角形栅格化与重心坐标恢复。
2. 实现 BestView。
3. 实现 WeightedAverage。
4. 实现 Natural 鲁棒融合。
5. 实现重影异常值过滤。
6. 实现焦外评分开关。
7. 实现自动色彩校正。
8. 实现小孔、padding 和受限锐化。
9. 将已实现控件从禁用改为可用。

阶段门：

- 三种模式产生可区分且符合预期的结果。
- 开关关闭时完全不执行对应处理。
- 两视曝光偏差合成测试中，校正后接缝下降且单视图细节不丢失。
- 移动物体颜色异常值不会被平均成重影。

### 阶段 6：工作流、GUI、持久化和取消

修改：

- `ModelWorkflowService.h/.cpp`
- `ProjectModelManager.cpp`
- `TextureMappingDialog.h/.cpp/.ui`
- `src/gui/project/data/ProjectData.cpp`
- `src/gui/project/data/ProjectFilesManager.cpp`
- `src/gui/project/archive/PlascanArchive.cpp`
- 模型查看和数据树相关代码
- `tests/test_project_data.cpp`
- `tests/test_gui_project_utils.cpp`

任务：

1. 完成所有 GUI JSON 到核心配置映射。
2. 显示输入模型、影像数量、预计纹理页数和内存。
3. 在任务启动前验证有效模型和纹理源。
4. 传播取消标志。
5. 使用临时目录和原子提交。
6. 持久化多纹理资源和报告。
7. 成功提示显示实际算法、覆盖率、页数和未映射面。
8. 重新打开项目后显示纹理模型。

阶段门：

- GUI 显示值与报告中的有效配置一致。
- 取消任务不会覆盖旧纹理结果。
- 项目切换后旧任务不能写入新项目。
- 项目归档移动后所有纹理路径仍有效。

### 阶段 7：性能、真实数据验证和默认切换

修改：

- `CameraTextureMapper.cpp`
- `scripts/validation/compare_texture_mapping.py`
- `docs/PROJECT_ARCHITECTURE.md`
- 新增 `src/core/mesh/README.md` 中纹理章节；若无 README 则创建
- `CHANGELOG.md`，仅在准备发版时更新

任务：

1. 对候选阶段和 atlas bake 使用 OpenMP。
2. 保证并行阶段输出确定性。
3. 增加分阶段耗时和峰值内存估算。
4. 在 Dino、Temple、Hyb2 和至少一个行星表面数据集上对比 v3/v4。
5. 通过全部验收后，将 v4 设为 GUI 默认。
6. 保留内部 `enableV4=false` 回退一个发布周期。

阶段门：

- 合成测试、模块测试和项目持久化测试全部通过。
- 三个现有数据集无背景泄漏和明显覆盖回归。
- 行星表面数据集达到纹素密度与接缝指标。
- v4 错误率和取消行为达到 GUI 交付要求。

## 11. 测试矩阵

| 测试 | 输入 | 主要断言 |
|---|---|---|
| 单相机平面 | 1 面、1 相机 | UV、颜色、OBJ/MTL 正确 |
| 双相机不同分辨率 | 颜色 2× 深度 | 两坐标系投影一致 |
| 遮挡台阶 | 前后两层面 | 后层不能采样前层颜色 |
| 掩模跨越 | 面内部穿过背景 | 整面严格候选被拒绝 |
| 斜视角 | 正视和擦边视角 | 正视相机得分更高 |
| 投影分辨率 | 远近相机 | 高纹素密度相机优先 |
| 焦外过滤 | 清晰/模糊两视图 | 模糊视图降权或拒绝 |
| 标签棋盘 | 人工候选分数 | 小孤岛被清理 |
| 标签不可见保护 | 邻居标签不可投影 | 不发生非法传播 |
| chart 连通域 | 同相机多区域 | 生成多个独立 chart |
| atlas 打包 | 多尺寸矩形 | 无重叠、确定性、占用率正确 |
| Natural 曝光差 | 两视图亮度不同 | 接缝色差下降 |
| 重影过滤 | 一视图异常颜色 | 异常样本被拒绝 |
| 小孔填充 | 纹理内 1–2 px 空洞 | 小孔填满且不越过 chart |
| 未映射面 | 无任何支持 | 使用明确默认色并统计 |
| 取消 | 每个长阶段触发取消 | 无正式产物被覆盖 |
| 项目移动 | 带纹理 `.plascan` | OBJ/MTL/PNG/报告均可恢复 |

现有测试必须继续通过：

- `TextureMapperTest.ReadsPlyMeshFacesForTextureMapping`
- `TextureMapperTest.CameraAtlasUsesPerFaceProjectedUvWithoutPlanarOverlap`
- `TextureMapperTest.CameraAtlasSuppressesIsolatedFaceCameraSwitches`
- `TextureMapperTest.CameraAtlasDoesNotSampleBackgroundAcrossMaskedFaceInterior`

## 12. 验证命令

配置和构建：

```powershell
cmake -S . -B build\windows-vcpkg-cuda-release `
  -G Ninja `
  -DBUILD_TESTS=ON `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE=C:\BuildTools\VC\vcpkg\scripts\buildsystems\vcpkg.cmake

cmake --build build\windows-vcpkg-cuda-release `
  --target plascan_gui test_mesh_reconstructor test_texture_mapping `
  -j 4
```

纹理相关测试：

```powershell
ctest --test-dir build\windows-vcpkg-cuda-release `
  --output-on-failure `
  -R "Texture|MeshReconstructor|ProjectData|GuiProject"
```

直接运行：

```powershell
build\windows-vcpkg-cuda-release\tests\test_mesh_reconstructor.exe `
  --gtest_filter=TextureMapperTest.*

build\windows-vcpkg-cuda-release\tests\test_texture_mapping.exe
```

真实数据 A/B：

```powershell
.\.venv\Scripts\python.exe scripts\validation\compare_texture_mapping.py `
  --baseline <v3-report.json> `
  --candidate <v4-report.json> `
  --output <comparison.json>
```

全量测试：

```powershell
ctest --test-dir build\windows-vcpkg-cuda-release --output-on-failure
```

若全量测试出现
`TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 的 `dom_png not found`，
按项目已知历史失败单独报告，不能把全量测试描述为全部通过。

## 13. 进度与取消协议

建议阶段百分比：

| 阶段 | 进度 |
|---|---:|
| 输入发现与预检 | 0–5 |
| 影像预处理 | 5–15 |
| 网格邻接 | 15–20 |
| 候选相机评分 | 20–45 |
| 标签优化 | 45–58 |
| chart 构建 | 58–65 |
| atlas 打包 | 65–70 |
| 光度校正 | 70–76 |
| 纹理烘焙 | 76–92 |
| padding、孔洞与锐化 | 92–96 |
| 输出与项目记录 | 96–100 |

每个阶段：

- 开始时报告真实阶段名。
- 大循环每 1024～4096 面或每个 chart 检查取消。
- 取消返回独立状态，不使用普通算法失败文案。
- GUI 收到取消后不显示“纹理映射失败”。

## 14. 风险与回退

### 风险 1：原图相机和深度相机内参比例不可靠

缓解：

- 显式保存和验证两套相机。
- 在 manifest 中记录影像尺寸和内参适用尺寸。
- 无法验证时停止任务。

### 风险 2：Natural 逐纹素多视投影过慢

缓解：

- 每面最多保留 4 候选，每纹素最多混合 3 视图。
- chart 独立并行。
- BestView 始终作为快速模式。
- 后续再考虑 GPU，不作为首版前置。

### 风险 3：8192/16384 图集内存过高

缓解：

- 启动前预算所有图像、mask 和临时缓冲。
- 清晰度和距离场使用低分辨率。
- 避免为每个候选创建全尺寸权重图。
- 超预算时要求降低纹理大小或增加下采样。

### 风险 4：标签平滑牺牲局部清晰度

缓解：

- 设定 `coherentReplacementRatio`。
- 新标签必须对当前面有效。
- 报告因平滑改变的面数量。
- 保留 BestView 无平滑诊断模式。

### 风险 5：色彩校正破坏科学颜色

缓解：

- 默认关闭。
- 记录每相机增益和偏移。
- 严格限制参数范围。
- 报告校正前后统计。

### 风险 6：v4 影响现有模型生成流程

缓解：

- 独立 `enableV4` 开关。
- v3 测试不删除。
- 先只用于独立“生成纹理”入口，再接入“生成模型”的可选纹理阶段。
- v4 失败不自动覆盖旧产物。

## 15. 完成定义

只有以下条件全部满足，才能把计划标记为完成：

- 所有计划模块已实现并纳入 `src/core/mesh/CMakeLists.txt`。
- GUI 中所有启用控件都映射到真实核心行为。
- v4 输出报告包含配置、输入指纹、覆盖、chart、atlas、接缝和耗时统计。
- 合成单元测试、纹理集成测试、项目移动/归档测试通过。
- Dino、Temple、Hyb2 和行星数据集完成 v3/v4 A/B。
- 背景泄漏、接缝、占用率和纹素密度达到验收指标。
- 取消和失败不会覆盖旧结果。
- GUI、核心、项目持久化和文档行为一致。
- `docs/PROJECT_ARCHITECTURE.md` 和纹理模块说明已同步。
- 构建与测试结果、已知问题记录在对应版本文档。

## 16. 推荐执行顺序

严格按以下顺序实施：

1. 阶段 0：基线与评价器。
2. 阶段 1：输入坐标和分辨率契约。
3. 阶段 2：候选评分。
4. 阶段 3：标签优化。
5. 阶段 4：chart 和图集。
6. 阶段 5：Natural 与高级选项。
7. 阶段 6：GUI、项目与取消。
8. 阶段 7：性能、真实数据和默认切换。

不能先启用 GUI 高级选项，再补核心实现；也不能跳过阶段 0 的评价器直接凭截图调整算法。
