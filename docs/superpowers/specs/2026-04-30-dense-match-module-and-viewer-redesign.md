# 密集匹配模块与统一连接点查看器 设计文档

日期：2026-04-30
状态：已确认
TDD：是

## 概述

1. 将「连接点查看」从 重建→稀疏重建 移至 工具 菜单
2. 在 重建→密集重建→深度图估计 前新增「密集匹配...」菜单
3. 重构连接点查看器为统一查看器，Tab 切换稀疏/密集匹配
4. 新建 `core/dense_match/` 模块，参考 ASP StereoPipeline 实现 CUDA 加速密集匹配算法
5. 同时封装 OpenCV SGBM 作为对比选项

## 一、菜单重组

### 目标结构

```
工具
├── 重叠度获取
├── 前方交汇精度检验
│   ├── 检测交汇
│   └── 查看结果
├── 手动点云剔除
├── 连接点查看              ← 从 重建→稀疏重建 移入
└── 查看工作流程报告...

重建
├── 稀疏重建
│   ├── 特征点提取
│   ├── 创建连接点
│   /* 连接点查看 已移走 */
│   ├── 构建观测网络模型...
│   ├── 初始化相机位姿...
│   ├── 生成初始稀疏点云...
│   ├── 光束法平差优化...
│   └── 稀疏点云后处理...
├── 密集重建
│   ├── 密集匹配...           ← 新增（深度图估计的前置步骤）
│   ├── 深度图估计...
│   ├── 深度图融合生成密集点云...
│   └── 密集点云后处理...
└── 模型生成
    ├── 网格重建...
    ├── 纹理映射...
    └── 模型导出...
```

### MainMenu 变更清单

| 变更 | 文件 |
|------|------|
| + `QAction *m_denseMatchAct` 成员 | `MainMenu.h` |
| + `QAction *denseMatchAction() const` 访问器 | `MainMenu.h` / `MainMenu.cpp` |
| 从 sparseReconMenu 移除 `m_viewMatchesAct` | `MainMenu.cpp` |
| 在 denseReconMenu 最前面插入 `m_denseMatchAct` | `MainMenu.cpp` |
| 在 toolsMenu 中添加 `m_viewMatchesAct`（在手动点云剔除之后） | `MainMenu.cpp` |

## 二、统一连接点查看器

### 改造 MatchViewerDialog

现有 `MatchViewerDialog` 仅支持稀疏特征匹配连线查看。改造为带 `QTabWidget` 的统一查看器：

```
┌─────────────────────────────────────────────────────────┐
│ 匹配查看：img001.tif <-> img002.tif                      │
├─────────────────────────────────────────────────────────┤
│ [同步缩放] [适应窗口] [重置] [+] [-] │ [显示选项区...]   │
├─────────────────────────────────────────────────────────┤
│ ┌── 稀疏匹配 ──────────┬── 密集匹配 ──────────┐         │
│ │                      │                       │         │
│ │  左影像 + 特征点      │  左影像 + 视差热力图  │         │
│ │  + 关键点连线         │  叠加                 │         │
│ │                      │                       │         │
│ │  右影像 + 特征点      │  右影像               │         │
│ │  + 关键点连线         │                       │         │
│ └──────────────────────┴───────────────────────┘         │
├─────────────────────────────────────────────────────────┤
│ 总匹配点数：1234                                        │
└─────────────────────────────────────────────────────────┘
```

### Tab 对比

| 特性 | 稀疏匹配 Tab | 密集匹配 Tab |
|------|------------|------------|
| 数据源 | `.match` 关键点对文件 | 视差图 `.tif` 或深度图 |
| 叠加层 | `MatchLineOverlay` | `DisparityHeatmapOverlay` |
| 工具栏选项 | 颜色/宽度/透明度/最大显示/端点/彩虹/内点过滤 | 色彩映射/透明度/视差范围/无效值过滤 |
| 共享控件 | 同步缩放、适应窗口、重置、放大、缩小 | 同左 |

### 新增/变更文件

| 文件 | 说明 |
|------|------|
| `MatchViewerDialog.h/cpp` | 添加 QTabWidget 和双模式支持 |
| `DisparityHeatmapOverlay.h/cpp` | 新建，视差热力图叠加渲染 |
| `DualImageViewer.h/cpp` | 支持叠加层运行时切换 |

### 接口

```cpp
// 现有构造 → 默认打开稀疏 Tab
MatchViewerDialog(const QString &imgA, const QString &imgB,
                  const QString &matchFile, QWidget *parent);

// 新增静态工厂方法 → 默认打开密集 Tab
static MatchViewerDialog* forDenseMatch(
    const QString &imgA, const QString &imgB,
    const QString &disparityFile, QWidget *parent);

// 设置项目路径（记忆化）
void setProjectPath(const QString &plascanPath);
```

## 三、密集匹配模块 `core/dense_match/`

### 目录结构

```
src/core/dense_match/
├── CMakeLists.txt
├── DenseMatchTypes.h          # 枚举、公共类型
├── DenseMatchConfig.h         # 参数配置结构体
├── CostFunctions.h            # 代价函数声明
├── CostFunctions.cpp          # CPU 参考实现
├── CostFunctions.cu           # CUDA 实现
├── BlockMatcher.h             # BM 块匹配
├── BlockMatcher.cpp
├── BlockMatcher.cu
├── SgmMatcher.h               # SGM/MGM
├── SgmMatcher.cpp
├── SgmMatcher.cu
├── SubpixelRefiner.h          # 子像素精化
├── SubpixelRefiner.cpp
├── SubpixelRefiner.cu
├── DisparityValidator.h       # L-R 一致性检查、置信度过滤
├── DisparityValidator.cpp
├── DenseMatchService.h        # 服务层
├── DenseMatchService.cpp
│
├── tests/
│   ├── CMakeLists.txt
│   ├── CostFunctionTest.cpp
│   ├── BlockMatcherTest.cpp
│   ├── SgmMatcherTest.cpp
│   ├── SubpixelRefinerTest.cpp
│   ├── DisparityValidatorTest.cpp
│   └── DenseMatchIntegrationTest.cpp
│
└── opencv/
    ├── OpenCVSgbmWrapper.h
    └── OpenCVSgbmWrapper.cpp
```

### 公共类型 (`DenseMatchTypes.h`)

```cpp
enum class CostFunction
{
    AbsoluteDifference       = 0,  // 绝对差
    SquaredDifference        = 1,  // 平方差
    NormalizedCrossCorr      = 2,  // NCC
    CensusTransform          = 3,  // Census 变换
    TernaryCensusTransform   = 4   // 三值 Census
};

enum class StereoAlgorithm
{
    BlockMatch       = 0,  // CUDA 块匹配
    SemiGlobalMatch  = 1,  // CUDA SGM
    MoreGlobalMatch  = 2,  // CUDA MGM（参考 ASP asp_mgm）
    OpenCV_SGBM      = 3   // OpenCV SGBM（对比用）
};

enum class SubpixelMode
{
    None         = 0,  // 不做子像素精化
    Parabola     = 1,  // 抛物线拟合
    AffineBayes  = 2,  // 仿射 + 贝叶斯加权
    LucasKanade  = 3   // LK 光流精化
};

struct DisparityResult
{
    cv::Mat disparity;    // 浮点视差图
    cv::Mat confidence;   // 置信度图 (0-1)
    cv::Mat validMask;    // 有效像素掩码（L-R 一致性检查后）
};
```

### 配置结构 (`DenseMatchConfig.h`)

```cpp
struct DenseMatchConfig
{
    StereoAlgorithm algorithm  = StereoAlgorithm::MoreGlobalMatch;
    CostFunction    costFunc   = CostFunction::CensusTransform;
    SubpixelMode    subpixel   = SubpixelMode::Parabola;

    int   minDisparity      = 0;
    int   maxDisparity      = 256;

    int   corrKernelW       = 15;
    int   corrKernelH       = 15;

    int   p1                = 8;     // SGM 小惩罚
    int   p2                = 32;    // SGM 大惩罚
    int   sgmDirections     = 8;     // 4 或 8
    int   sgmCollarSize     = 512;

    int   pyramidLevels     = 2;

    int   subpixelKernelW   = 21;
    int   subpixelKernelH   = 21;

    float lrCheckThreshold  = 1.0f;
    int   medianFilterSize  = 3;

    bool  useCuda           = true;
    int   cudaDevice        = 0;
    int   numThreads        = 4;

    std::string leftImagePath;
    std::string rightImagePath;
    std::string outputDisparityPath;
};
```

## 四、算法设计

### 4.1 代价函数 (CostFunctions)

5 种代价函数均实现 CPU 和 CUDA 两个版本：

- **AD (AbsoluteDifference)**: `C(p,d) = |I_L(p) - I_R(p-d)|`
- **SD (SquaredDifference)**: `C(p,d) = (I_L(p) - I_R(p-d))²`
- **NCC**: 窗口归一化互相关，`C(p,d) = 1 - NCC(win_L, win_R)`，范围 [0,2]
- **Census**: 像素级 Census 变换 → 汉明距离
- **TernaryCensus**: 引入容差 τ 的三值 Census

CUDA kernel 策略：每个 thread block 处理一个 32×32 瓦片，共享内存缓存左窗口，视差循环内联。

### 4.2 块匹配 (BlockMatcher)

朴素 WTA 块匹配，作为 baseline：

```
对每个像素 p，每个视差 d ∈ [minDisp, maxDisp]:
  聚合代价 = sum over corrKernel of C(q, d)
disp(p) = argmin 聚合代价
conf(p) = (第二小 - 最小) / 最小
```

### 4.3 SGM/MGM (SgmMatcher) — 核心算法

参考 ASP `asp_mgm`：

**SGM 能量函数**：E(D) = Σ C(p, D_p) + Σ P1·1[|ΔD|=1] + Σ P2·1[|ΔD|>1]

**路径聚合（8 方向）**：
```
L_r(p, d) = C(p, d) + min(
    L_r(p-r, d),
    L_r(p-r, d±1) + P1,
    min_k L_r(p-r, k) + P2
) - min_k L_r(p-r, k)
```

**MGM 改进**：同时利用当前扫描线前一个像素和上一扫描线像素的代价，形成更完整的 2D 平滑约束。

CUDA 策略：每条路径一个 kernel，`__syncthreads()` 瓦片内同步，分两步：先算全部代价，再路径聚合，最后 WTA。

### 4.4 子像素精化 (SubpixelRefiner)

- **Parabola**: `d_sub = d - (C(d+1)-C(d-1)) / (2(C(d+1)+C(d-1)-2C(d)))`
- **AffineBayes**: 仿射对齐 + 贝叶斯后验加权
- **LucasKanade**: 光流迭代

### 4.5 视差验证 (DisparityValidator)

- L-R 一致性检查（分别计算 L→R 和 R→L，差值 > 阈值则无效）
- 中值滤波去噪
- Speckle 连通域过滤

### 4.6 服务层流水线 (DenseMatchService)

```
输入: 左影像, 右影像, DenseMatchConfig
  │
  ├─ 影像加载 & 灰度转换
  ├─ [可选] Census 变换预处理
  ├─ 算法分支:
  │   ├─ BlockMatch  → BlockMatcher::compute()
  │   ├─ SGM/MGM     → SgmMatcher::compute()
  │   └─ OpenCV SGBM → OpenCVSgbmWrapper::compute()
  ├─ 子像素精化 (SubpixelRefiner)
  ├─ 视差验证 (DisparityValidator)
  └─ 输出: DisparityResult
```

## 五、密集匹配对话框 (DenseMatchDialog)

### UI 布局

```
┌──────────────────────────────────────────────────────┐
│ 密集匹配                                             │
├──────────────────────────────────────────────────────┤
│ 输入                                                  │
│  影像对选择:  [img001__img002              ▼]        │
│  AT 结果:     [AT_20260430_143052           ▼]        │
│  输出路径:    [/project/dense_match/       ...]       │
├──────────────────────────────────────────────────────┤
│ ┌ 算法参数 ──────────────────────────────────────┐   │
│ │ 匹配算法:  [MGM (More Global Match)      ▼]    │   │
│ │ 代价函数:  [Census Transform              ▼]    │   │
│ │ 子像素:    [Parabola Fitting              ▼]    │   │
│ │ 最小视差:  [0    ]  最大视差:  [256  ]           │   │
│ │ 核大小:    [15   ]  (W x H)                      │   │
│ └──────────────────────────────────────────────────┘   │
│ ┌ SGM/MGM 参数   (可折叠) ───────────────────────┐   │
│ │ P1 (小惩罚):      [8    ]                        │   │
│ │ P2 (大惩罚):      [32   ]                        │   │
│ │ 路径方向数:       [8    ]                        │   │
│ │ 金字塔层数:       [2    ]                        │   │
│ └──────────────────────────────────────────────────┘   │
│ ┌ 系统参数       (可折叠) ───────────────────────┐   │
│ │ ☑ 使用 CUDA    设备: [GPU 0              ▼]     │   │
│ │ CPU 线程数:     [4    ]                          │   │
│ │ ☐ 同时运行 OpenCV SGBM 对比                      │   │
│ └──────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────┤
│ [运行]  [取消]  [恢复默认]   进度: [==========] 45%  │
└──────────────────────────────────────────────────────┘
```

### 交互逻辑

- 算法切换时联动控件启用状态（OpenCV SGBM → CUDA 设备置灰）
- "OpenCV 对比"勾选时同时运行两种算法，输出到不同文件
- 预设档位：快速(BM)/标准(MGM)/精细(MGM+AffineBayes)
- 参数持久化到 `project_dialog.json`

### 新增文件

| 文件 | 说明 |
|------|------|
| `DenseMatchDialog.h` | 密集匹配参数对话框声明 |
| `DenseMatchDialog.cpp` | 对话框实现（参考 `FeatureMatchingDialog` 的分组折叠模式） |

## 六、测试策略 (TDD)

### 单元测试

| 测试 | 验证内容 |
|------|---------|
| `CostFunctionTest` | 5 种代价 CPU/CUDA 结果一致性（容差 1e-5），边界输入（全 0、全 255） |
| `BlockMatcherTest` | 合成水平平移影像对，WTA 视差与 ground truth 误差 < 1px |
| `SgmMatcherTest` | 已知视差合成数据，P1/P2 惩罚不引入额外误差 |
| `SubpixelRefinerTest` | 亚像素偏移合成数据（0.25px 步长），误差 < 0.1px |
| `DisparityValidatorTest` | L-R 一致性检测率 > 95%，噪声误过滤率 < 5% |

### 集成测试

| 测试 | 验证内容 |
|------|---------|
| `DenseMatchIntegrationTest` | CPU vs CUDA 端到端一致性；对标定影像对输出视差图与 Middlebury ground truth 比较 |

### TDD 流程

1. 先写测试 → 编译失败（红）
2. 实现最小代码 → 测试通过（绿）
3. 重构优化 → 保持测试通过
4. 每次提交前跑全量测试，确保无回归

## 七、文件变更汇总

### 新建

| 路径 | 行数上限 |
|------|---------|
| `src/core/dense_match/CMakeLists.txt` | 30 |
| `src/core/dense_match/DenseMatchTypes.h` | 60 |
| `src/core/dense_match/DenseMatchConfig.h` | 80 |
| `src/core/dense_match/CostFunctions.h` | 40 |
| `src/core/dense_match/CostFunctions.cpp` | 200 |
| `src/core/dense_match/CostFunctions.cu` | 250 |
| `src/core/dense_match/BlockMatcher.h` | 40 |
| `src/core/dense_match/BlockMatcher.cpp` | 180 |
| `src/core/dense_match/BlockMatcher.cu` | 200 |
| `src/core/dense_match/SgmMatcher.h` | 50 |
| `src/core/dense_match/SgmMatcher.cpp` | 300 |
| `src/core/dense_match/SgmMatcher.cu` | 350 |
| `src/core/dense_match/SubpixelRefiner.h` | 40 |
| `src/core/dense_match/SubpixelRefiner.cpp` | 150 |
| `src/core/dense_match/SubpixelRefiner.cu` | 180 |
| `src/core/dense_match/DisparityValidator.h` | 30 |
| `src/core/dense_match/DisparityValidator.cpp` | 120 |
| `src/core/dense_match/DenseMatchService.h` | 40 |
| `src/core/dense_match/DenseMatchService.cpp` | 200 |
| `src/core/dense_match/opencv/OpenCVSgbmWrapper.h` | 30 |
| `src/core/dense_match/opencv/OpenCVSgbmWrapper.cpp` | 100 |
| `src/core/dense_match/tests/CMakeLists.txt` | 30 |
| `src/core/dense_match/tests/CostFunctionTest.cpp` | 180 |
| `src/core/dense_match/tests/BlockMatcherTest.cpp` | 150 |
| `src/core/dense_match/tests/SgmMatcherTest.cpp` | 200 |
| `src/core/dense_match/tests/SubpixelRefinerTest.cpp` | 150 |
| `src/core/dense_match/tests/DisparityValidatorTest.cpp` | 120 |
| `src/core/dense_match/tests/DenseMatchIntegrationTest.cpp` | 180 |
| `src/gui/widgets/DisparityHeatmapOverlay.h` | 50 |
| `src/gui/widgets/DisparityHeatmapOverlay.cpp` | 200 |
| `src/gui/dialogs/DenseMatchDialog.h` | 120 |
| `src/gui/dialogs/DenseMatchDialog.cpp` | 400 |

### 修改

| 路径 | 变更 |
|------|------|
| `src/gui/menu/MainMenu.h` | + `m_denseMatchAct` 成员，+ `denseMatchAction()` 访问器 |
| `src/gui/menu/MainMenu.cpp` | 移动 m_viewMatchesAct 到 toolsMenu，添加 m_denseMatchAct 到 denseReconMenu |
| `src/gui/main_window/MainWindow.cpp` | 连接 denseMatchAction 到 DenseMatchDialog 打开槽 |
| `src/gui/dialogs/MatchViewerDialog.h/cpp` | 添加 QTabWidget 和密集匹配 Tab 支持 |
| `src/gui/widgets/DualImageViewer.h/cpp` | 支持叠加层切换 |
| `src/gui/config/settings/DialogSettingKeys.h` | + DenseMatch 键 |
| `src/core/CMakeLists.txt` | + `add_subdirectory(dense_match)` |

## 八、向后兼容

- `MatchViewerDialog` 原有构造函数行为不变（默认打开稀疏 Tab）
- 现有 `.match` 格式不变
- 现有 MVS 管线不受影响，后续可用新模块替换其中的匹配步骤
- 密集匹配模块输出标准视差图 TIFF，与现有 `DepthMapGenerator` 兼容

## 九、后续集成路径

当新模块充分验证后，MVS 管线替换路径：
1. `DepthMapGenerator::compute()` 内部调用 `DenseMatchService` 代替现有匹配
2. `PatchMatchCUDA` 逐步废弃，由 `SgmMatcher::compute(mgm=true)` 替代
3. `EpipolarRectifier` 的输出直接接入 `DenseMatchService`
