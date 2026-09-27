# PlaBundle 独立库拆分实施计划

> 日期：2026-09-20
> 状态：PB0–PB6 与 PB8 本地 submodule 接入已完成；PB7 数值求解迁移已接线，完整主工程门禁待复核
> 产品名：PlaBundle
> CMake package / C++ namespace：`plabundle`
> 正式 target：`plabundle::plabundle`

## 1. 目标

把 PlaScan 当前 `src/core/bundle_adjust` 中成熟的面阵光束法平差能力拆成一个可以独立配置、构建、
安装和复用的 C++20 库，同时保持 PlaScan 当前 CPU/CUDA/OpenCL 数值语义、后端选择、质量门控、
取消行为和用户可见报告不变。

PlaBundle v0.1 的定位是：

> A C++ bundle and control-network adjustment library powered by PlaMatrix.

它是摄影测量领域库，不宣称替代 Ceres 的通用非线性最小二乘能力，也不宣称提供 GTSAM 式通用因子图。
第一版聚焦当前已经生产使用的 frame-pinhole / Brown-Conrady 联合 BA；推扫线阵、ISIS PVL 和时变姿轨
作为后续可选组件接入，不阻塞 v0.1。

## 2. 完成定义

第一阶段全部完成时必须同时满足：

1. `3rdparty/plabundle` 是一个独立顶层 CMake 工程，可在不配置 PlaScan、Qt、OpenCV、GDAL 或
   PlaPoint 的情况下完成 configure/build/test/install。
2. 对外公开头只使用 `plabundle` namespace，不包含 `xjw`、PlaScan camera/project/coordinate 类型，
   不泄漏 Qt 或 PlaPoint。
3. `find_package(plabundle CONFIG REQUIRED)` 与
   `target_link_libraries(app PRIVATE plabundle::plabundle)` 可以在安装树消费。
4. PlaBundle 只依赖 C++20、PlaMatrix 和可选 OpenMP；CUDA/OpenCL 能力从 PlaMatrix target 继承。
5. PlaScan 的 SfM、LiDAR、CLI 和 GUI 服务全部切换到 PlaBundle；旧 `bundle_adjust` 求解实现被删除，
   不保留长期双轨或转发壳。
6. CPU 参考结果、约束统计、状态、取消/失败保护与当前基线一致；CUDA/OpenCL 继续满足跨后端质量门。
7. PlaScan 现有项目格式、CLI 参数、GUI 文案和 BA JSON 报告字段保持兼容。
8. 独立库测试、安装消费测试、PlaScan 集成测试、相关全量测试和文档门禁均通过。

物理创建独立 GitHub 仓库、转换为 git submodule、commit、push、tag 和 Release 不属于隐含授权；
只有用户另行明确授权后才执行。

## 3. 当前基线与主要耦合

当前 `src/core/bundle_adjust` 约 8.9k 行，已经包含：

- 问题、选项、结果、状态和后端选择公共契约；
- frame-pinhole / Brown-Conrady 投影与解析 Jacobian；
- 相机位姿、三维点和九参数分组共享内参联合优化；
- GCP、LiDAR 点到面、独立激光测距、比例尺、位姿和相机平面约束；
- gauge、输入验证、结果质量门和取消/失败保护；
- PlaMatrix CPU sparse Schur、CUDA/OpenCL Schur-PCG；
- 自适应相机模型参数释放和后端 benchmark。

现有独立性缺口：

- 公共 API 直接暴露 PlaScan `FramePinholeNumericState`；该类型携带 camera instance、image、frame、
  capture time 和 typed instance 回写能力，不是纯 BA 数值状态。
- CMake 公开依赖 `camera` / `camera_models_frame_pinhole`，私有依赖 PlaScan common。
- `BAOptions` 同时混放算法选项、问题约束、固定块、标定组和运行控制。
- 核心 namespace 仍是 `xjw`，公开字段中大量诊断名绑定 PlaScan/PlaMatrix 现有报告布局。
- 纯库测试由 PlaScan 根测试工程注册；真实数据 benchmark loader 依赖 Qt 和 TSAI IO。
- 推扫线阵 BA 与 ISIS 控制网仍位于 `src/core/lidar`，和 frame BA 尚未共享统一公共问题模型。

## 4. 目标依赖边界

```text
PlaScan GUI / CLI / SfM / LiDAR workflows
                  │
                  ▼
      plascan_plabundle_adapter
      - typed camera identity
      - coordinate/session context
      - project report/write-back
                  │
                  ▼
           plabundle::plabundle
      - numeric camera/problem/constraints
      - BA nonlinear driver and quality gates
      - backend policy and diagnostics
                  │
                  ▼
           plamatrix::plamatrix
      - block normal equations / Schur / PCG
      - CPU/CUDA/OpenCL runtime and kernels

PlaPoint ──► PlaScan LiDAR association ──► PlaBundle constraints
```

边界规则：

- PlaBundle 不读取图片、工程文件、TSAI、PLY、PVL 或 Qt JSON。
- PlaBundle 不负责特征匹配、track 构建、LiDAR 最近邻/法线、SfM 局部/全局/分层调度。
- PlaBundle 只接受已经位于同一数值坐标系中的相机、点和约束；CRS/frame 解析由 PlaScan adapter 完成。
- PlaBundle 不猜测单位、轴序、正深度、range 单/往返语义或虚拟像点类型；非法输入 fail closed。
- PlaScan adapter 保留 typed identity 并按原索引回写，PlaBundle 不持有项目对象或外部生命周期。

## 5. 独立库目录与 target

第一阶段在 PlaScan 内新增普通源码目录，暂不修改 `.gitmodules`：

```text
3rdparty/plabundle/
├── CMakeLists.txt
├── CMakePresets.json
├── LICENSE
├── README.md
├── CHANGELOG.md
├── cmake/
│   └── plabundleConfig.cmake.in
├── include/plabundle/
│   ├── backend.h
│   ├── camera.h
│   ├── constraints.h
│   ├── problem.h
│   ├── options.h
│   ├── result.h
│   ├── solver.h
│   └── version.h
├── src/
│   ├── solver.cpp
│   ├── validation.cpp
│   ├── projection.cpp
│   ├── quality.cpp
│   ├── adaptive_camera_model.cpp
│   ├── internal/
│   └── plamatrix/
├── test/
├── benchmark/
└── examples/
```

首版只导出一个正式库 target：

```cmake
add_library(plabundle ...)
add_library(plabundle::plabundle ALIAS plabundle)
```

构建选项：

- `PLABUNDLE_BUILD_TESTS`
- `PLABUNDLE_BUILD_BENCHMARKS`
- `PLABUNDLE_BUILD_EXAMPLES`
- `PLABUNDLE_ENABLE_OPENMP`

不增加独立的 `PLABUNDLE_WITH_CUDA` / `PLABUNDLE_WITH_OPENCL` 开关。PlaBundle 根据
`plamatrix::plamatrix` 导出的 compile definitions 和 targets 编译对应能力，避免两套配置发生矛盾。
安装配置使用 `find_dependency(plamatrix CONFIG)`；OpenMP/CUDA/OpenCL 的传递依赖继续由 PlaMatrix package 管理。

## 6. v0.1 公共 API

### 6.1 相机数值状态

PlaBundle 自己定义不带项目身份的 `FrameCamera`：

```cpp
namespace plabundle
{

struct FrameCamera
{
    std::array<double, 9> cameraToWorldRotation;
    std::array<double, 3> cameraCenter;
    double focalXPixels;
    double focalYPixels;
    double principalXPixel;
    double principalYPixel;
    double k1;
    double k2;
    double k3;
    double p1;
    double p2;
    int uAxisSign;
    int vAxisSign;
    bool depthAxisFlipped;
};

}
```

最终字段会按当前投影契约补齐显式默认值和验证，但不得加入 PlaScan ID、path、frame、time、typed instance、
Qt 或 IO 成员。输入和结果相机严格保持向量顺序，外部身份映射由调用方持有。

PlaScan 新增 Qt-free adapter：

```text
src/core/plabundle_adapter/
├── CMakeLists.txt
├── FrameCameraAdapter.h/.cpp
└── tests/
```

adapter 从 `FramePinholeNumericState` 构造 `FrameCamera`，并把求解结果写回原 typed state 副本；
批量写回先在副本上完整验证，任一相机失败都不会部分修改调用方。转换前后通过随机点投影、正深度、
轴方向、Brown 参数、身份保持和冻结求解 parity 测试。PB2 曾用 Problem/Result adapter 做迁移期双跑，
PB5 完成后已删除这两层兼容转换，生产消费者直接构造 `plabundle::Problem` 并读取结构化 `Result`。

### 6.2 Problem / Options 职责

`Problem` 只保存待求解数据与结构：

- cameras、tracks、observations；
- fixed cameras / fixed points；
- calibration group 和稳定内参参考；
- GCP、laser plane、laser range、scale bar、pose prior、camera plane constraints；
- gauge/anchor 结构和调用方已经解析的单位语义。

`Options` 只保存求解策略：

- 后端请求、设备索引和 Auto 阈值；
- 最大迭代、阻尼、Armijo/线性求解设置；
- 允许优化的内参参数 mask、边界、弱先验和自适应模型策略；
- 鲁棒阈值、点过滤和质量门策略；
- 线程数、取消 token 和进度回调。

当前由 `enable*` 开关控制的约束，在 PlaScan service/SfM 问题构建边界只在开关启用时加入 `Problem`；PlaBundle 内看到
非空约束就必须求解或明确返回 unsupported，不允许静默忽略。这样保持 PlaScan 兼容，同时让独立 API
避免“数据存在但 enable=false”的歧义。

### 6.3 Solver / Result

公开入口统一为：

```cpp
plabundle::Solver solver;
plabundle::Result result = solver.solve(problem, options);
```

契约：

- 输入为 const，任何失败或取消都不修改调用方数据；
- `Result::usable()` 是结果是否可提交的唯一门；
- invalid input、unsupported、backend unavailable、cancelled、no convergence 和 numerical failure
  使用稳定枚举，不通过解析错误字符串判断；
- refined cameras / points 与输入索引一一对应；
- 通用质量统计与 PlaMatrix 后端诊断分层存放，PlaScan adapter 继续写出现有 JSON 字段；
- v0.x 明确不保证 ABI 稳定，但保证同一 minor 内保存的机器可读枚举名稳定。

取消首选 `std::stop_token`；进度回调接收结构化 iteration summary。PlaScan 现有 atomic/Qt 取消状态在 adapter
边界桥接，迁移测试必须证明取消时机和“不发布部分结果”语义不变。

## 7. 分阶段实施

### PB0：冻结基线和可复现证据

状态：已完成。基线保存在 `build/tmp/plabundle-bootstrap/baseline/`，供 PB3/PB4 重放。

任务：

1. 重新检查 `git status --short`、submodule、相关构建/测试进程。
2. 把本次临时证据放到 `build/tmp/plabundle-extraction/baseline/`，记录：
   - BA 相关源文件 SHA-256；
   - 当前 CMake target 和链接依赖；
   - 纯 BA 定向测试结果；
   - 一组固定 synthetic frame BA 的完整机器可读结果；
   - 可用时记录 CPU/CUDA/OpenCL 同数据 benchmark。
3. 区分本任务前既有失败和本任务引入失败；当前共享 dirty 工作区的无关失败不能静默归因给 PlaBundle。

验收：基线文件可重放，且没有修改生产代码。

### PB1：搭建独立工程和安装消费门禁

状态：已完成。Linux/GCC 独立构建、测试、安装和安装树 consumer 均已验证。

任务：

1. 创建 `3rdparty/plabundle` 顶层工程、MIT LICENSE、README、版本 `0.1.0` 和最小公开头。
2. 创建 `plabundle` / `plabundle::plabundle` target、build/install include、package config 和 version config。
3. 增加顶层/子项目两种 option 默认值，避免作为 PlaScan 子目录时自动构建测试和 benchmark。
4. 增加最小 consumer example，并从安装前缀用 `find_package(plabundle CONFIG REQUIRED)` 编译运行。
5. 在 `cmake/PlascanPackages.cmake` 中于 PlaMatrix 后、`src/core` 前接入 PlaBundle；此阶段尚不切换生产调用。

验收：

- 独立空壳工程可在 Linux/GCC 下 configure/build/test/install；
- 安装消费 smoke test 通过；
- PlaScan 原有 `bundle_adjust` 仍构建，用户行为不变。

### PB2：建立纯数值公共契约和 PlaScan adapter

状态：已完成。公共契约、严格输入校验、Qt-free adapter 和投影 parity 已接入；生产求解入口未切换。

任务：

1. 先写 `FrameCamera`、observations/tracks/constraints、`Problem`、`Options`、`Result`、backend/status 测试。
2. 将当前混在 `BAOptions` 中的问题数据重新归位到 `Problem`，但由 adapter 保持原开关行为。
3. 新增 `plascan_plabundle_adapter` target，负责：
   - `FramePinholeNumericState` ↔ `FrameCamera`；
   - 旧 BA 输入结构 → 新 `Problem`；
   - 新 `Result` → PlaScan 相机/点/报告字段。
4. 增加随机化投影 parity、正深度、pose delta、内参/畸变和非法状态测试。
5. 增加公开头自包含测试和禁止依赖扫描：`Qt`、`xjw`、`PlaScan`、`PlaPoint`、项目绝对路径必须为零。

验收：公共契约和 adapter 测试通过；尚不切换正式求解器。

### PB3：迁移 CPU 参考求解器

状态：已完成。Linux/GCC 下独立 OpenMP 开/关配置各 36/36，通过安装树 consumer、固定 PB0
synthetic 新旧求解 parity、PlaScan adapter 11/11 与旧生产 BA 93/93 回归；生产入口仍未切换。

迁移顺序严格从底到顶：

1. projection / analytic Jacobian；
2. problem validation 和 active block 映射；
3. Brown 共享内参组、边界和结果发布；
4. GCP/LiDAR/scale/pose/camera-plane 约束；
5. normal equation assembly、online point Schur 和点回代；
6. PlaMatrix CPU sparse/dense Schur 驱动；
7. quality/filter/backend fallback；
8. adaptive camera model。

实施约束：

- 先做语义机械迁移和 namespace/type 替换，不在同一切片改算法或阈值；
- 大文件拆分只在 parity 通过后进行，避免同时改变组织和数值路径；
- 用 PlaBundle 内部 C++20 并行 helper 替代 `SafeWorkerGroup`，不公开线程实现类型；
- 删除未使用的 PlaScan log 依赖，不复制无消费者的日志设施；
- 所有异常在线程边界汇聚，外部返回稳定 failure status；`bad_alloc` 等不可恢复异常可继续抛出。

测试迁移：

- projection/Jacobian 数值差分；
- backend selection；
- point-only、fixed pose、full refinement；
- grouped focal/full Brown；
- fixed tracks/cameras 和 gauge；
- 所有物方约束；
- quality gate、invalid input、cancel/no-mutation；
- 自适应参数 mask；
- 1/多线程确定性或有界数值一致性。

验收：新旧 CPU 求解器在固定输入上的 status、可用性、有效点集合和报告结构一致；相机/点/约束 RMS
满足冻结容差，且没有性能数量级回退。

### PB4：迁移 CUDA/OpenCL 和 Auto 后端策略

状态：已完成。Linux/GCC 下 CPU-only、CUDA-only 和 OpenCL-only 独立配置均通过；真实
RTX 4060 已验证 CUDA/OpenCL device assembly/PCG、Auto、全约束 parity 和质量回退。
OpenCL 设备索引与 FP64 能力 fail-closed，CPU/CUDA/OpenCL 安装包均通过外部 consumer。

任务：

1. 迁移 PlaMatrix backend 映射、设备可用性、fallback 和质量门。
2. 复用 PlaMatrix 导出的 CUDA/OpenCL compile definitions 和传递依赖，不独立寻找另一套 SDK。
3. 迁移 Schur workspace、device assembly、mixed precision、PCG 容差和诊断统计。
4. 保持当前 Auto policy version、相机/观测阈值和 CUDA 优先顺序。
5. 将现有源码字符串契约改为行为测试；PlaScan 仅保留默认设置和报告接线契约。

验收：

- CPU-only build 不包含 CUDA/OpenCL 必需依赖；
- 显式不可用后端返回 `BackendUnavailable`，禁止回退时绝不偷偷执行 CPU；
- 允许回退时原因、used backend 和 quality gate 可观测；
- 可用设备上 CPU/CUDA/OpenCL 同数据结果通过现有 parity 门；
- benchmark 分别报告 setup/assembly/linear/back-substitution/total，不把 kernel 时间冒充端到端时间。

### PB5：PlaScan 一次性切换并删除旧实现

状态：已完成。SfM、LiDAR、航三、CLI、GUI service/controller 和契约测试已切到
`plabundle::*`；旧 `src/core/bundle_adjust` 求解源码、target、兼容 adapter、重复测试和 benchmark
均已删除。稳定 CLI/JSON 字段保留，frame camera 回写通过最小 Qt-free adapter 保持 typed identity。

生产消费者按依赖顺序迁移：

1. `src/core/sfm`：input builder、coordinator、hierarchical BA、adaptive model 调度；
2. `src/core/lidar`：普通 LiDAR 关联输出和行星 laser range adapter；
3. `src/core/aerial_triangulation`；
4. `src/cli/reconstruction`；
5. GUI `BundleAdjustService` / controller / project workflow；
6. 根测试和 Python source contracts。

切换规则：

- 所有直接消费者在同一阶段改为 `plabundle::*` 和 `plascan_plabundle_adapter`；
- 不建立 `using BAOptions = ...`、旧聚合头或 `BundleAdjust::optimizePoints()` 转发壳；
- PlaScan 稳定 JSON/CLI 字段由 service/CLI 边界显式映射，不把外部格式稳定性强加给 PlaBundle C++ 字段名；
- 全部消费者和测试通过后删除 `src/core/bundle_adjust` 旧求解源码及旧 CMake target；
- `docs/PROJECT_ARCHITECTURE.md`、README、模型/CLI 文档和边界 baseline 同步更新。

验收：

- `rg` 不再发现生产调用旧 `BundleAdjust*` 公共接口；
- PlaScan 只存在一个 BA 算法实现；
- CLI/GUI 输出与旧项目兼容；
- 相关核心、CLI、GUI、边界测试通过。

### PB6：独立测试、benchmark 和发布准备

状态：进行中。Linux/GCC 独立构建/安装消费、CPU/CUDA/OpenCL 数值 parity 和 deterministic
cold/warm benchmark 已具备；本轮继续收紧生产切换后的回归、文档和 benchmark 驱动。

独立 PlaBundle 测试矩阵：

| 维度 | 必须覆盖 |
| --- | --- |
| 编译器 | 当前原生平台；CI 规划 GCC、MSVC、Apple Clang |
| 后端 | CPU 必过；CUDA/OpenCL 只在真实设备可用时报告通过 |
| 构建形态 | 顶层、`add_subdirectory`、安装后 `find_package` |
| 共享/静态 | 至少验证默认静态；发布前补 `BUILD_SHARED_LIBS` smoke |
| 数值 | projection/Jacobian、全部约束、gauge、内参、质量门、取消 |
| 数据规模 | 小型 dense、常规 sparse、大型 iterative Schur |
| 性能 | cold/warm、setup/solve/total、CPU/GPU，不设置跨机器绝对时间门 |
| 依赖 | 无 Qt/OpenCV/GDAL/PlaPoint/PlaScan/xjw 泄漏 |

版本与文档：

- 初始版本 `0.1.0`，公开 API 处于活跃开发状态；
- README 包含五分钟构建、最小 frame BA、约束示例、backend 说明和限制；
- CHANGELOG 记录从 PlaScan 提取的来源、行为兼容范围和已知限制；
- 保留 MIT 版权文本；不复制测试数据中无明确再分发许可的大文件。

### PB7：line-scan 数值求解与后续 ISIS 可选组件

状态：第一步数值迁移已完成本地接线。PlaBundle 新增 `plabundle::linescan::Problem/Options/Result/solve`，
负责相机 6DoF 偏差、三维点、固定/约束激光落点、PlaMatrix Schur/LM、取消、后端选择与质量门；
PlaScan 继续持有 typed `placamera::LineScanModel`、逐行投影回调、ISD/PVL 解析和报告身份映射。
尚未提供 PlaBundle 原生线阵 trajectory/投影，也未支持混合网络或 Fixed/Constrained 控制点。
该阶段不阻塞 frame BA 独立发布，后续任务包括：

1. 定义纯数值 `plabundle::linescan` 相机/trajectory/problem，不暴露 PlaScan `LineScanInstance`。
2. PlaScan/USGSCSM adapter 负责 ISD、frame/time/坐标上下文转换。
3. 迁移当前 line-scan camera correction、控制点、range 和 PlaMatrix assembly。
4. 将 ISIS PVL/control network IO 做成可选 `plabundle::isis` 组件；求解核心不依赖文件格式。
5. 增加 frame/line-scan 混合网络、时刻、像素中心、虚拟 measure、Fixed/Constrained/Free 控制点测试。

是否将其放在同一仓库的额外 target，等 frame v0.1 API 稳定后再决定；不提前抽象通用 factor graph。

### PB8：物理独立仓库和 submodule（已完成本地接入）

状态：独立仓库已完成初始导入并通过独立构建、测试、安装和 consumer 验证；PlaScan 已将
`3rdparty/plabundle` 转换为固定 submodule，远端为 `https://github.com/guderianXu/plabundle.git`，当前指针为
`f87f7d8`。新克隆验证需在父仓库提交该 submodule 指针后执行。

前置条件：PB0-PB6 全绿且用户确认 API。

候选步骤：

1. 创建独立 `guderianXu/plabundle` 仓库；
2. 保留 PlaBundle 目录历史或以明确的 initial import 记录来源；
3. 配置独立 CI、issue/release 模板；
4. 发布 `v0.1.0`；
5. 将 PlaScan `3rdparty/plabundle` 转换为固定 submodule；
6. 更新 `.gitmodules`、source-deps、架构文档和发布说明。

父仓库的 submodule 元数据已准备好；PlaBundle 远端尚未打 tag/Release，版本发布另行处理。

## 8. 验证命令

实施时以当前原生 Linux/GCC preset 为准，示例命令如下；实际目录必须使用配置脚本返回的 preset 路径。

### 独立 CPU 构建

```bash
cmake -S 3rdparty/plabundle -B build/tmp/plabundle-extraction/cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DPLABUNDLE_BUILD_TESTS=ON \
  -DPLABUNDLE_BUILD_BENCHMARKS=ON \
  -DCMAKE_PREFIX_PATH=<plamatrix-install-prefix>
cmake --build build/tmp/plabundle-extraction/cpu --parallel
ctest --test-dir build/tmp/plabundle-extraction/cpu --output-on-failure
```

### 安装消费测试

```bash
cmake --install build/tmp/plabundle-extraction/cpu \
  --prefix build/tmp/plabundle-extraction/install
cmake -S 3rdparty/plabundle/examples/consumer \
  -B build/tmp/plabundle-extraction/consumer \
  -DCMAKE_PREFIX_PATH=<plamatrix-prefix>\;build/tmp/plabundle-extraction/install
cmake --build build/tmp/plabundle-extraction/consumer --parallel
```

### PlaScan 定向验证

```bash
python3 scripts/env/configure_with_env.py --source-deps
cmake --build build/linux-source-release --target \
  plabundle plascan_plabundle_adapter test_plabundle_adapter \
  test_bundle_adjust_lidar_constraints sfm_core sfm_project lidar \
  aerial_triangulation bundle_adjust_cli gui_project test_gui_project_utils
python3 scripts/env/run_tests.py \
  --test-dir build/linux-source-release --output-on-failure \
  -R 'BundleAdjust|PlaBundle|Sfm|AerialTriangulation|Lidar|PlanetaryLineScan'
```

### 最终门禁

- 修改过的 C++ 文件运行仓库 `.clang-format`；
- Python 契约脚本运行 `python -m py_compile`；
- `git diff --check`；
- PlaBundle 公开依赖扫描；
- 当前原生平台全量测试；
- 若用户明确要求 push，再执行 AGENTS.md 规定的全量本地门禁并等待对应 CI。

不能用 Linux 单平台结果声称 MSVC/Apple Clang 已通过；独立库发布前应分别取得对应平台结果。

## 9. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 当前工作区有大量并行未提交修改 | 每阶段重查 status/process，只修改清单内文件；PB0 保存本次基线 hash，不回滚他人改动。 |
| 相机旋转、轴方向或正深度约定漂移 | PlaScan state ↔ PlaBundle camera 随机投影 parity；所有后端共享同一投影实现。 |
| API 清理与算法迁移同时导致难以定位回归 | 先建立 contracts/adapter，再按底到顶机械移植；性能重构延后。 |
| 临时存在两份源码造成修复分叉 | 临时新实现不接生产入口；达到 parity 后一次性切换并删除旧实现，不保留长期 feature flag。 |
| GPU package 在独立安装后找不到运行时 | 安装消费矩阵验证 CPU/CUDA/OpenCL；依赖只经 PlaMatrix Config 传播。 |
| 约束从 Options 移到 Problem 后被遗漏 | adapter 行为测试逐项验证 enable/disabled、计数、RMS 和 unsupported 路径。 |
| line-scan/ISIS 扩大首版范围 | PB7 先迁移数值求解，ISIS IO、混合网络和完整控制点语义仍单独评审。 |
| benchmark 被 Qt/TSAI 绑定 | 独立库使用生成 fixture/BAL；PlaScan 保留真实 TSAI 集成重放。 |
| 过早创建远端仓库或 submodule | PB8 明确要求单独授权，当前只创建本地普通源码目录。 |

## 10. 第一轮实现切片

用户已确认并完成第一轮 PB0-PB2：

1. 冻结当前 BA 基线；
2. 创建可独立安装的 PlaBundle 空壳；
3. 定义纯数值公共契约；
4. 实现 PlaScan frame camera adapter 和 projection parity 测试；
5. 不迁移求解器、不删除旧模块、不改变任何生产调用。

下一轮通过评审后再进入 CPU 求解器迁移。第一轮差异只验证“库边界和 API 是否正确”，没有同时承担数值回归风险。
