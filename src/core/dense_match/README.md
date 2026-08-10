# 密集匹配模块 (dense_match)

支持 CPU/OpenMP 与可选 CUDA 代价计算的立体密集匹配模块，参考 ASP
(Ames Stereo Pipeline) 和 OpenCV SGBM 实现。模块统一输出定义在左参考影像上的视差。

## 目录结构

```
dense_match/
├── DenseMatchTypes.h        # 枚举类型: 代价函数/算法/子像素模式/结果结构体
├── DenseMatchConfig.h       # 参数配置结构体 (所有算法参数集中管理)
├── CostFunctions.h          # 代价函数与结构化 CostVolume API (CPU + CUDA)
├── CostFunctions.cpp        # CPU 代价函数实现 (AD/SD/NCC/Census/TernaryCensus)
├── CostFunctions.cu         # CUDA kernel: 代价卷 GPU 计算
├── BlockMatcher.h           # WTA 块匹配器声明
├── BlockMatcher.cpp         # 块匹配器实现 (CPU/CUDA 自动调度)
├── SgmMatcher.h             # SGM/MGM 半全局匹配器声明
├── SgmMatcher.cpp           # 半全局匹配器实现 (8方向路径聚合 + CPU/CUDA 调度)
├── SubpixelRefiner.h        # 子像素视差精化声明
├── SubpixelRefiner.cpp      # 抛物线拟合子像素精化
├── DisparityValidator.h     # 视差验证 (L-R 一致性/中值滤波/Speckle)
├── DisparityValidator.cpp   # 视差验证实现
├── DenseMatchService.h      # 服务层: 编排完整匹配流水线
├── DenseMatchService.cpp    # 服务层实现 (加载→匹配→验证→保存, 含性能日志)
├── opencv/
│   ├── OpenCVSgbmWrapper.h  # OpenCV SGBM 封装声明
│   └── OpenCVSgbmWrapper.cpp # OpenCV SGBM 封装实现 (算法对比用)
├── tests/
│   ├── CostFunctionTest.cpp       # 代价、范围、有效性及 CPU/CUDA 一致性
│   ├── BlockMatcherTest.cpp       # WTA、正负视差、无证据候选与 CPU/CUDA 一致性
│   ├── SgmMatcherTest.cpp         # SGM 路径状态、方向数、遍历顺序与无证据候选
│   ├── SubpixelRefinerTest.cpp    # 子像素边界、合法邻项和精度
│   ├── DisparityValidatorTest.cpp # L-R、有效性中值滤波和影像支持掩码
│   └── DenseMatchIntegrationTest.cpp # 服务层和端到端语义
├── CMakeLists.txt            # 构建配置 (CUDA 自动检测)
└── README.md                 # 本文档
```

## 架构概况

```
ProjectPointCloudWorkflowController
    │ 解析工作流程配置并调度匹配对
    ▼
DenseMatchService
    │ 加载影像 → 分支调度匹配器 → 视差验证 → 保存
    ▼
┌────────────┐  ┌──────────────┐
│BlockMatcher│  │  SgmMatcher  │  ← 选择一种
│ (WTA块匹配) │  │ (SGM/MGM聚合) │
└─────┬──────┘  └──────┬───────┘
      │                │
      ▼                ▼
┌──────────────────────────────┐
│  CostFunctions                 │
│  computeCostVolume() [CPU]     │
│  computeCostVolumeCUDA() [GPU] │ ← useCuda=true 时走 GPU
│  CostVolume: 范围 + 代价 + 假设有效掩码 │
└──────────────────────────────┘
          │
          ▼
    SubpixelRefiner (匹配器内，可插拔)
          │
          ▼
    DisparityValidator (后处理/L-R/影像支持)
```

## 文件职责详述

### 类型层 (无依赖)
- **DenseMatchTypes.h** — 定义 `CostFunction`(5种)、`StereoAlgorithm`(4种)、`SubpixelMode`(4种) 枚举和 `DisparityResult` 输出结构体
- **DenseMatchConfig.h** — 集中管理所有算法参数 (视差范围、核大小、SGM 惩罚、CUDA 开关等)，带合理默认值

### 视差与有效性约定

- 输出视差定义在左参考影像上：`d = x_left - x_right`，因此左像素 `x` 在右图采样 `x - d`。
- 所有搜索范围均为 `[minDisparity, maxDisparity)`：下界包含、上界不包含。零视差和负视差只要在范围内且有证据，都是合法结果。
- 几何越界和“无证据”假设使用有限的 `kInvalidCost` 哨兵；`CostVolume` 同时保存逐假设有效掩码。WTA、置信度、子像素及 SGM 聚合均跳过哨兵，不能把它当成低代价候选。
- `Census` 的 1×1 窗口没有比较项，`Ternary Census` 也可能因全部比较项落入容差带而没有证据；这两种情况在 CPU/CUDA 上都返回 `kInvalidCost`。
- 反向匹配范围为 `[1 - maxDisparity, 1 - minDisparity)`，L-R 一致性检查使用 `abs(d_lr + d_rl)`。
- 后处理中值滤波以原始匹配掩码决定中心有效性，只统计掩码有效且有限的邻域值；无可用样本时保留中心值。它不会把无效区域中用于占位的零带入合法的零/负视差。

### 代价计算层
- **CostFunctions.h/cpp** — CPU 实现 5 种代价函数。`computeCostVolume()` 用 OpenMP 并行计算完整代价卷 (H×W×D)，并携带范围与显式有效性信息
- **CostFunctions.cu** — 5 种代价函数的 CUDA device 实现 (`adCostDev`/`sdCostDev`/`nccCostDev`/`censusCostDev`/`ternaryCensusCostDev`)。`computeCostVolumeCUDA()` 在 GPU 上并行计算代价卷，并保持与 CPU 相同的边界和无效值语义

### 匹配器层
- **BlockMatcher.h/cpp** — WTA 块匹配。`compute()` 内部根据 `_config.useCuda` 自动选择 CPU 或 CUDA 代价卷
- **SgmMatcher.h/cpp** — SGM 半全局匹配。最多 8 方向路径聚合 (Hirschmüller 算法)，每个方向独立遍历路径并复用 O(D) 滚动状态；同样支持 CUDA 代价卷自动调度

### 精化/验证层
- **SubpixelRefiner.h/cpp** — 子像素视差精化。`Parabola` 模式用三点抛物线拟合获得亚像素精度。`None`/`AffineBayes`/`LucasKanade` 保留扩展点
- **DisparityValidator.h/cpp** — `checkLRConsistency()` 左右一致性检查、有效性感知的 `medianFilter()`、`speckleFilter()` 连通域去噪和影像支持检查

### 服务层
- **DenseMatchService.h/cpp** — 编排完整流水线: 加载影像 → 匹配器 → 验证 → 保存 TIFF。内置 `printf` 性能诊断日志 (图像尺寸、算法、CUDA 状态、耗时)
- GUI 通过 `ProjectPointCloudWorkflowController` 协调深度估计与点云融合；它不是独立的“稠密重建管理器”。核心模块不依赖具体对话框。

### OpenCV 封装
- **opencv/OpenCVSgbmWrapper.h/cpp** — 封装 `cv::StereoSGBM::create()`，用于与自研 CUDA 算法对比精度和速度

## 数据流

```
输入: 左影像 (CV_8UC1) + 右影像 (CV_8UC1) + DenseMatchConfig
  │
  ├─ 1. CostFunctions::computeCostVolume()
  │     → CostVolume (范围 + H×W×D float32 + 显式假设有效掩码)
  │    [GPU: CostFunctions::computeCostVolumeCUDA() — DM_ENABLE_CUDA + useCuda=true]
  │
  ├─ 2a. BlockMatcher: WTA 逐像素最小值
  │  或
  │  2b. SgmMatcher: 8方向路径聚合 + WTA
  │     → DisparityResult (视差图 int→float, 置信度, 有效掩码)
  │
  ├─ 3. 匹配器内 SubpixelRefiner::refine() → 子像素 float 视差图
  │
  └─ 4. DisparityValidator::validate() → 有效性感知中值滤波 + L-R/影像支持掩码
      → 最终 DisparityResult

输出: TIFF 浮点视差图
```

## 构建配置

```cmake
# core/CMakeLists.txt 中通过 plascan_core_add_optional_module 注册
plascan_core_add_optional_module(dense_match "DenseMatch")

# GUI 中条件链接 (GuiCoreLinking.cmake)
if(TARGET dense_match)
    target_link_libraries(plascan_gui PRIVATE dense_match)
endif()
```

CUDA 通过 `check_language(CUDA)` 自动检测。可用时定义 `DM_ENABLE_CUDA=1`，编译 `.cu` 文件。

## CUDA 使用

```cpp
DenseMatchConfig cfg;
cfg.useCuda    = true;   // 运行时开关
cfg.cudaDevice = 0;      // GPU 设备 ID

BlockMatcher bm(cfg);
// compute() 内部:
// #ifdef DM_ENABLE_CUDA
//   if (useCuda) → computeCostVolumeCUDA()  ← GPU 计算代价卷
//   else         → computeCostVolume()      ← CPU + OpenMP
```

**注意**: CUDA 加速仅用于代价卷计算 (计算密度最大的步骤)。路径聚合 (SGM) 和 WTA 仍在 CPU 上执行，因为这部分计算量相对较小。

CPU 与 CUDA 必须遵守同一左参考、半开视差范围、平坦 NCC 和 Ternary/Census 无证据语义；CUDA 编译不会启用可能改变这些数值分支的 `--use_fast_math`。

## 测试

```bash
cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --target test_dense_match_unit
python ../scripts/env/run_tests.py --test-dir . --output-on-failure \
  -R 'CostFunction|BlockMatcher|SgmMatcher|SubpixelRefiner|DisparityValidator|DenseMatch'
```

当前测试覆盖：正/零/负视差、半开范围和边界掩码、五种代价函数、有限无效哨兵、Census/Ternary 无证据候选、WTA 与置信度、SGM 1/4/8 方向独立状态及遍历顺序、子像素合法邻项、L-R 反向范围、有效性感知中值滤波、端到端服务语义，以及可用 CUDA 设备上的五种代价函数和 BlockMatcher CPU/CUDA 一致性。

## 持久化与兼容性

`dense_match` 不维护内部持久缓存、工作区格式或可复用代价卷。`saveDisparity()` 只在调用方明确请求时写出最终浮点 TIFF。因此调整视差、有效性或代价语义不需要升级缓存版本；已有输出若要采用新语义，需要由上层工作流显式重新计算。

## GUI 集成

旧的“重建 → 密集匹配”独立对话框已移除。密集匹配作为模型生成和地形产品工作流程的
内部阶段，由项目管理层统一调度、取消、记录进度和写入结果，避免用户在多个重叠入口
间维护不一致参数。

## 扩展点

1. **新增代价函数**: 在 `CostFunction` 枚举加新值 → `CostFunctions.cpp` 加 static 函数 → `CostFunctions.cu` 加 `__device__` 函数和 switch case
2. **新增匹配算法**: 在 `StereoAlgorithm` 枚举加新值 → 新建 Matcher 类 → `DenseMatchService::process()` 加分支
3. **替换 MVS 管线**: 将 `DepthMapGenerator` 中的 PatchMatch 调用替换为 `DenseMatchService::process()`
4. **CUDA 路径聚合**: 在 `CostFunctions.cu` 中加 SGM path aggregation kernel (目前代价卷在 GPU 算, 聚合在 CPU)
