# StereoDenseCloudPipeline 设计文档

## 概述

新增 `StereoDenseCloudPipeline`，实现双视图密集点云生成流程，输出与 ASP `run-PC.tif` 一致的 4 波段 TIF 点云和 PLY 点云。

核心思路：DISK+LightGlue 稀疏匹配 → 极线校正 → PatchMatch CUDA 密集匹配 → 子像素精化 → 视差滤波 → 逐像素三角化 → 双格式输出。

## 流程总览

```
leftImage.tif + rightImage.tif + left.tsai + right.tsai
    │
    ▼
Step 1: DISK+LightGlue 稀疏特征匹配 (CUDA)
    │
    ▼
Step 2: 极线校正 (EpipolarRectifier)
    │
    ▼
Step 3: PatchMatch CUDA 密集匹配 (epipolarRectified=true)
    │
    ▼
Step 4: 子像素精化 (抛物线拟合)
    │
    ▼
Step 5: 视差滤波 (中值 + 左右一致性 + 噪斑去除)
    │
    ▼
Step 6: 逐像素视差三角化 (双射线交汇, 多线程+CUDA)
    │
    ▼
Step 7: 输出 TIF (4波段 Float32 + POINT_OFFSET) + PLY
```

## 模块划分

| 模块 | 文件 | 职责 | 复用/新增 |
|------|------|------|-----------|
| 稀疏匹配 | `LightGlueMatcher` | DISK+LightGlue 提取匹配点 | 复用 |
| 极线校正 | `EpipolarRectifier` | 计算 H 矩阵，校正图像对 | 复用 |
| 密集匹配 | `PatchMatchCUDA` | 校正后图像逐像素深度估计 | 复用 |
| 子像素精化 | `SubpixelRefiner` | 抛物线拟合精化视差 | 新增 ~150行 |
| 视差滤波 | `DisparityFilter` | 中值+噪斑+左右一致性 | 新增 ~200行 |
| 视差三角化 | `DisparityTriangulator` | 逐像素双射线交汇 | 新增 ~250行 |
| TIF 输出 | `PointCloudTifIO` | GDAL 4波段 Float32 写出 | 新增 ~150行 |
| PLY 输出 | `PointCloudIO` | 已有 PLY 写出 | 复用 |
| Pipeline | `StereoDenseCloudPipeline` | 串联所有步骤 | 新增 ~300行 |

新增文件约 5 个（含 .h/.cpp），总新增代码约 1000-1100 行。

## Pipeline 接口与配置

### StereoPipelineConfig

```cpp
struct StereoPipelineConfig
{
    // 特征匹配
    std::string featureAlgorithm = "disk+lightglue";
    float matchScoreThreshold = 0.2f;

    // PatchMatch 密集匹配
    PatchMatchConfig patchMatch;  // 复用现有结构，默认 epipolarRectified=true

    // 子像素精化
    int subpixelMode = 1;         // 0=关闭, 1=抛物线, 2=仿射窗口
    int subpixelKernelSize = 21;

    // 视差滤波
    int medianFilterSize = 3;
    int speckleSize = 60;
    float speckleRange = 3.0f;
    bool leftRightCheck = true;
    float lrThreshold = 1.0f;

    // 三角化
    float maxTriangulationError = 0.001f;  // 与 ASP 参数对齐
    bool useCudaTriangulation = true;

    // 输出
    bool outputTif = true;
    bool outputPly = true;
    int numThreads = 0;  // 0=自动
};
```

### StereoPipelineResult

```cpp
struct StereoPipelineResult
{
    std::string tifPath;
    std::string plyPath;
    int totalPoints = 0;
    int validPoints = 0;
    float coveragePercent = 0.f;
    double medianTriError = 0.0;
    std::string errorMsg;
};
```

### StereoDenseCloudPipeline 公开接口

```cpp
class StereoDenseCloudPipeline : public QObject
{
    Q_OBJECT
public:
    // 傻瓜式接口：只需影像路径 + 相机路径
    bool run(
        const std::string &leftImagePath,
        const std::string &rightImagePath,
        const std::string &leftCameraPath,
        const std::string &rightCameraPath,
        const std::string &outputDir,
        StereoPipelineResult *result = nullptr);

    // 高级接口：传入已加载的对象
    bool run(
        const cv::Mat &leftImage,
        const cv::Mat &rightImage,
        const Camera &leftCamera,
        const Camera &rightCamera,
        const std::string &outputDir,
        StereoPipelineResult *result = nullptr);

    void setConfig(const StereoPipelineConfig &cfg);
    void cancel();

signals:
    void progressChanged(QString stage, float ratio);
    void finished(bool success);
};
```

设计要点：
- 默认参数对齐 ASP stereo.default（maxTriangulationError=0.001）
- 傻瓜式接口自动加载 .tsai 相机文件
- Qt 信号报告进度，兼容现有 GUI 框架

## 各步骤详细数据流

### Step 1-2: 稀疏匹配 + 极线校正

```
leftImg (gray) ──┐
                 ├─→ DISK+LightGlue (CUDA) ──→ matchPairs[(u1,v1), (u2,v2)]
rightImg (gray) ─┘
                                                    │
                                                    ▼
                                          EpipolarRectifier::rectify()
                                                    │
                                                    ▼
                                          RectifiedPair { rectLeft, rectRight,
                                                          H1, H2, H1inv,
                                                          rectCamL, rectCamR }
```

- 匹配点用于估计基础矩阵 F，计算极线校正单应矩阵
- 现有 EpipolarRectifier 已实现此逻辑
- 匹配点不足（<50）时回退到相机参数直接计算极线校正

### Step 3: PatchMatch CUDA 密集匹配

```
rectLeft, rectRight, rectCamL, rectCamR
         │
         ▼
PatchMatchCUDA::estimate(
    refImg=rectLeft, srcImgs=[rectRight],
    refCam=rectCamL, srcCams=[rectCamR],
    config={epipolarRectified=true})
         │
         ▼
depthMap (CV_32F) + confidenceMap (CV_32F)
         │
         ▼
深度→视差转换: disp(u,v) = fu * baseline / depth(u,v)
```

- epipolarRectified=true 让 PatchMatch 偏向水平传播
- 深度图转视差图是简单公式变换

### Step 4: 子像素精化 (SubpixelRefiner)

对每个有效像素，取整数视差 d 附近的 NCC cost 值，拟合抛物线求极值：

```
d_sub = d - 0.5 * (cost[d+1] - cost[d-1]) / (cost[d+1] - 2*cost[d] + cost[d-1])
```

PatchMatch 本身产生浮点深度，此步骤在深度→视差转换后进一步精化。

### Step 5: 视差滤波 (DisparityFilter)

1. 中值滤波 (3×3)：去除椒盐噪声
2. 左右一致性检查：对左右视差图交叉验证，差异 > 1.0px 的标记无效
3. 噪斑去除：连通域面积 < 60 的标记无效

### Step 6: 视差三角化 (DisparityTriangulator)

对每个有效像素 (u, v)：

```
leftPixel  = H1_inv * (u, v)           // 反变换回原始左图坐标
rightPixel = H2_inv * (u + disp, v)    // 反变换回原始右图坐标

ray1 = camL.undistortPixel(leftPixel)  → 归一化方向 → 世界射线
ray2 = camR.undistortPixel(rightPixel) → 归一化方向 → 世界射线

point3D = midpointIntersection(ray1, ray2)
error   = rayMissDistance(ray1, ray2)

if error < maxTriangulationError:
    output[u,v] = {X - offset_X, Y - offset_Y, Z - offset_Z, error}
```

- 批量处理，多线程（std::thread pool）+ 可选 CUDA kernel
- 复用 Camera::undistortPixel 做反畸变
- POINT_OFFSET 取点云质心，与 ASP 一致

### Step 7: 输出

**TIF 格式（PointCloudTifIO）：**
- GDAL 写 4 波段 Float32，LZW 压缩
- 元数据 POINT_OFFSET = "X Y Z"（点云质心）
- 无效像素写 0
- 尺寸与校正后左图一致

**PLY 格式：**
- 过滤 error > threshold 的点
- 用现有 PointCloudIO::writePlyPointCloud 写出
- 包含 XYZ + RGB（从左图采样颜色）

## 测试策略

### 测试数据

使用现有 `data/stereo_test_20260426/`：
- 左图: 20260413T174329163_NAS_PAN_L2b.tif
- 右图: 20260413T174419164_NAS_PAN_L2b.tif
- 相机: ba-tsai_*.tsai
- ASP 参考: run-PC.tif

### 测试程序

新增 `stereo_pipeline_test.cpp`：

1. 加载测试影像 + .tsai 相机
2. 调用 StereoDenseCloudPipeline::run（傻瓜式接口）
3. 读取输出 PC.tif (GDAL) 和 PC.ply
4. 读取 ASP run-PC.tif 作为参考
5. 逐像素对比 XYZ 差值
6. 验证 TIF 元数据和 PLY 一致性

### 验证指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| 有效点覆盖率 | ≥ 80% of ASP | PatchMatch 和 SGM 特性不同 |
| XYZ 中位数偏差 | < 0.001 | 与 ASP maxTriangulationError 对齐 |
| XYZ P95 偏差 | < 0.01 | 允许边缘区域较大偏差 |
| TIF 波段数 | = 4 | X, Y, Z, error |
| PLY 点数 | = TIF 有效像素数 | 双格式一致性 |

## 文件结构

```
src/core/mvs/
├── StereoDenseCloudPipeline.h      # Pipeline 主类声明
├── StereoDenseCloudPipeline.cpp    # Pipeline 实现（串联各步骤）
├── SubpixelRefiner.h               # 子像素精化声明
├── SubpixelRefiner.cpp             # 子像素精化实现
├── DisparityFilter.h               # 视差滤波声明
├── DisparityFilter.cpp             # 视差滤波实现
├── DisparityTriangulator.h         # 视差三角化声明
├── DisparityTriangulator.cpp       # 视差三角化实现
├── PointCloudTifIO.h               # TIF 点云 IO 声明
├── PointCloudTifIO.cpp             # TIF 点云 IO 实现
├── (已有文件不变)
│
src/core/terrain/tests/
├── stereo_pipeline_test.cpp        # 新增测试程序
```
