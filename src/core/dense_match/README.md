# 密集匹配模块 (dense_match)

基于 CUDA 加速的立体密集匹配模块，参考 ASP (Ames Stereo Pipeline) 和 OpenCV SGBM 实现。

## 目录结构

```
dense_match/
├── DenseMatchTypes.h        # 枚举类型: 代价函数/算法/子像素模式/结果结构体
├── DenseMatchConfig.h       # 参数配置结构体 (所有算法参数集中管理)
├── CostFunctions.h          # 代价函数 API 声明 (CPU + CUDA)
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
│   ├── CostFunctionTest.cpp       # 代价函数单元测试 (7 项)
│   ├── BlockMatcherTest.cpp       # 块匹配器测试 (3 项)
│   ├── SgmMatcherTest.cpp         # SGM 测试 (2 项)
│   ├── SubpixelRefinerTest.cpp    # 子像素测试 (3 项)
│   ├── DisparityValidatorTest.cpp # 验证器测试 (4 项)
│   └── DenseMatchIntegrationTest.cpp # 集成测试 (3 项)
├── CMakeLists.txt            # 构建配置 (CUDA 自动检测)
└── README.md                 # 本文档
```

## 架构概况

```
DenseMatchDialog (GUI)
    │ 用户配置参数 → JSON
    ▼
ReconstructionWorkflowController
    │ 创建进度追踪, 启动异步任务
    ▼
ProjectManager::startDenseMatchAsyncWithProgress()
    │ 遍历匹配对, 逐个调用 DenseMatchService
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
│  CostFunctions              │
│  computeCostVolume() [CPU]  │
│  computeCostVolumeCUDA() [GPU] │ ← useCuda=true 时走 GPU
└──────────────────────────────┘
          │
          ▼
    DisparityValidator (后处理)
    SubpixelRefiner   (子像素, 可插拔)
```

## 文件职责详述

### 类型层 (无依赖)
- **DenseMatchTypes.h** — 定义 `CostFunction`(5种)、`StereoAlgorithm`(4种)、`SubpixelMode`(4种) 枚举和 `DisparityResult` 输出结构体
- **DenseMatchConfig.h** — 集中管理所有算法参数 (视差范围、核大小、SGM 惩罚、CUDA 开关等)，带合理默认值

### 代价计算层
- **CostFunctions.h/cpp** — CPU 实现 5 种代价函数。`computeCostVolume()` 用 OpenMP 并行计算完整代价卷 (H×W×D)
- **CostFunctions.cu** — 5 种代价函数的 CUDA device 实现 (`adCostDev`/`sdCostDev`/`nccCostDev`/`censusCostDev`/`ternaryCensusCostDev`)。`computeCostVolumeCUDA()` 在 GPU 上并行计算代价卷

### 匹配器层
- **BlockMatcher.h/cpp** — WTA 块匹配。`compute()` 内部根据 `m_cfg.useCuda` 自动选择 CPU 或 CUDA 代价卷
- **SgmMatcher.h/cpp** — SGM 半全局匹配。8 方向路径聚合 (Hirschmüller 算法)，支持 SGM/MGM 两种模式。同样支持 CUDA 代价卷自动调度

### 精化/验证层
- **SubpixelRefiner.h/cpp** — 子像素视差精化。`Parabola` 模式用三点抛物线拟合获得亚像素精度。`None`/`AffineBayes`/`LucasKanade` 保留扩展点
- **DisparityValidator.h/cpp** — `checkLRConsistency()` 左右一致性检查、`medianFilter()` 中值滤波、`speckleFilter()` 连通域去噪

### 服务层
- **DenseMatchService.h/cpp** — 编排完整流水线: 加载影像 → 匹配器 → 验证 → 保存 TIFF。内置 `printf` 性能诊断日志 (图像尺寸、算法、CUDA 状态、耗时)

### OpenCV 封装
- **opencv/OpenCVSgbmWrapper.h/cpp** — 封装 `cv::StereoSGBM::create()`，用于与自研 CUDA 算法对比精度和速度

## 数据流

```
输入: 左影像 (CV_8UC1) + 右影像 (CV_8UC1) + DenseMatchConfig
  │
  ├─ 1. CostFunctions::computeCostVolume()  → CostVolume (H×W×D float32)
  │    [GPU: CostFunctions::computeCostVolumeCUDA() — DM_ENABLE_CUDA + useCuda=true]
  │
  ├─ 2a. BlockMatcher: WTA 逐像素最小值
  │  或
  │  2b. SgmMatcher: 8方向路径聚合 + WTA
  │     → DisparityResult (视差图 int→float, 置信度, 有效掩码)
  │
  ├─ 3. SubpixelRefiner::refine() → 子像素 float 视差图
  │
  └─ 4. DisparityValidator::validate() → 中值滤波 + 掩码标记
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

## 测试

```bash
cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --target test_dense_match_unit
./src/core/dense_match/test_dense_match_unit
```

22 项测试覆盖: 代价函数一致性、块匹配正确性、SGM 聚合、子像素精度、L-R 验证、端到端集成。

## GUI 集成

### 对话框
- **DenseMatchDialog.h/cpp** — 业务逻辑 (影像加载、匹配对刷新、参数收集/恢复)
- **DenseMatchDialogUi.cpp** — 纯 UI 构建 (setupUi, 左右分栏 QSplitter)

### 主窗口
- 进度条通过 `MainWindow::m_dmProgressBar` 显示在状态栏右下角
- 每对影像 5 步进度 × 150ms 轮询刷新

### 菜单
- 位置: 重建 → 密集重建 → 密集匹配...

## 扩展点

1. **新增代价函数**: 在 `CostFunction` 枚举加新值 → `CostFunctions.cpp` 加 static 函数 → `CostFunctions.cu` 加 `__device__` 函数和 switch case
2. **新增匹配算法**: 在 `StereoAlgorithm` 枚举加新值 → 新建 Matcher 类 → `DenseMatchService::process()` 加分支
3. **替换 MVS 管线**: 将 `DepthMapGenerator` 中的 PatchMatch 调用替换为 `DenseMatchService::process()`
4. **CUDA 路径聚合**: 在 `CostFunctions.cu` 中加 SGM path aggregation kernel (目前代价卷在 GPU 算, 聚合在 CPU)
