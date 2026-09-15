# Core 渐进重构

这轮已完成原计划的六类职责提取：影像/源计划/单帧估计、工件发布与恢复、
同步 MVS 服务、模型工作流分解、TSDF 阶段化及统一执行控制。
保持 GUI/CLI 入口、项目格式、质量阈值和 recovered 生产算法不变；
不整体搬迁顶层目录，也不把仍使用 QtCore 数据的算法声称为 Qt-free。

## 依赖与使用入口

| 职责 | 接口 / target | 边界 |
| --- | --- | --- |
| 深度帧、金字塔数据 | `DepthFrameResult.h`、`DepthPyramidTypes.h` / `mvs_contracts` | QtCore/OpenCV/相机数据；无生成器、后台调度 |
| 融合前同步处理 | `depth_processing/DepthPostprocessor.h` / `mvs_depth_processing` | 置信度、离群、小连通域及质量统计；不写工件 |
| 深度后端、读取、重放 | 原后端接口 / `mvs_backend` | 算法及存储适配；不依赖流程服务或 QtConcurrent |
| 同步深度流程 | `MvsPipelineService.h` / `mvs_pipeline` | 调用后端和私有阶段；由调用者选择线程 |
| Qt 异步深度任务 | `gui/project/tasks/DepthMapTask.h` / `gui_project` | GUI 专用信号、重复启动拒绝、取消、持有并 join future |
| 网格重建算法 | `DepthTsdfSurfaceBuilder.h` 等 / `meshing_algorithms` | 算法和 recovered 资源；不依赖模型工作流 |
| 模型业务编排 | `ModelWorkflowService.h` / `model_workflow` | 输入、参数、质量策略、取色/纹理及成果发布 |
| 通用控制 | `task_runtime/WorkflowExecution.h` / `task_runtime` | 纯 C++ 取消标志、归一化进度和成功/失败/取消结果 |

下层 target 不反向链接 `mvs`、`meshing` 或 QtConcurrent。
网格公开头文件使用的 PlaPoint 类型通过 PUBLIC 依赖明确提供，
不再依靠 GUI 或大测试目标偶然传递包含路径。

### MVS 阶段

`src/core/mvs/pipeline/` 内按实际职责拆出：

- 影像准备、掩膜、prepared raster 发布及资源策略。
- 稀疏可见性、源计划、深度范围、hint/seed、支撑域先验。
- 单帧估计、后端桥接、融合帧组装、批次/流式一致性、候选恢复。
- 工件 IO、保存队列、manifest/证据发布，以及 recovered 场景生产编排。

私有头文件只用于阶段之间共享实现类型和声明，不是新增公共算法 API。
旧生成器及其静态转发入口已删除，同步服务中的重复后处理转发也已删除。
新代码处理深度数据应包含数据或后处理头文件；执行流程应直接链接 mvs_pipeline。

`execute()` 同步执行，回调在执行线程调用；同一个服务拒绝重叠执行。
输入、配置、events 必须在执行前设置，执行期间不得修改或销毁服务。
取消可从另一线程请求。同步执行不隐式清除已置位的共享取消标志；
新任务通过显式 `resetCancellation()` 重置，Qt 适配器在启动 worker 前重置。
适配器在任务运行期间拒绝修改输入/配置/输出目录，析构先请求取消再等待 worker。

### 模型工作流

`src/core/mesh/workflow/` 分离了生成契约、参数解析、输入准入、轨道默认策略、
表面质量/降噪、取色/纹理阶段、成果发布，以及点云/深度/纹理入口。

深度产品入口仍只接受显式 `recovered_ooc`，需要 `recovered_model_input`，
不接受 recovered UV/OBJ 导出，也不回退旧算法。
原入口在恒真 recovered 分支后的旧 TSDF/Visual Hull/稀疏支架代码原本不可达；
现已删除 `LegacyDepthModelStages.cpp` 及其私有声明和链接入口。
源码契约测试改为约束当前 recovered 生产路径，不再检查已删除的不可达分支。
公开旧参数/质量策略函数仍保留，测试和显式算法验证入口继续可用。

### TSDF 阶段

`src/core/mesh/tsdf/` 分离读取、输入规划、布局、观测、恢复、
统计序列化、体素积分、支持域恢复和曲面后处理。
主流程仍依次执行积分 → 支持域恢复 → 等值面 → 清理 → 简化 → 最终质量检查。
短路返回保留原失败/取消顺序，不会继续写出失败阶段的结果。

私有阶段状态以引用传递既有数组及统计，不复制 TSDF/weight 等大体素缓冲；
OpenMP 积分计数保留在积分阶段局部，原 reduction 与运算顺序不变。
网格修复、补洞和 QEM 回退仍保留原拓扑/边界/包围盒门禁。
流式一致性事务、TSDF 支持域算法、统计字段表等高内聚文件仍可能超过
400 行；没有为满足行数而拆散事务或更改算法。

### 统一控制与未完成迁移

四种模型请求只保留 `execution`：共享取消标志或外部任务取消检查、UTF-8 stage 和
0..1 ratio。GUI/CLI 调用方已迁移，旧公开 `isCancelled`/`progress` 字段及双回调桥接已删除。
私有 `bindWorkflowCallbacks` 将唯一控制转换为现有算法需要的回调；子请求传递相同的控制，
不会重复报告进度。非有限/越界 ratio 归一化。
TSDF options 的旧回调与 `execution` 双接口尚未迁移，属于明确保留的兼容债务，
不能称为全项目兼容层已清除。
结果适配通过明确布尔取消字段分类，不依赖中文错误字符串猜测取消。
这提供协作取消，不宣称暂停、checkpoint 恢复或任意算法瞬时中断能力。
调用方回调应快速返回，不抛异常、不在执行回调内销毁服务。

## 保持的行为契约

- Accepted 帧才能参与融合；ValidationOnly 可作为一致性来源；失败帧不能由准入枚举单独提升。
- streaming 释放大像素数据但保留支撑、相机、证据身份和诊断；终态释放也清理支撑。
  共享读取者的数据不被强制失效，两种释放均可重复调用。
- 稀疏支撑只降低域外置信度，不删除深度；低置信保留仍要求真实几何证据。
- 后处理维持置信度 → 局部离群 → 小连通域顺序、像素域缩放和移除比例回退。
- 几何来源位序、工件身份、manifest 审计、准入和 recovered 保存/通知顺序不变。

边界基线显式迁移 target 名称及提取后的 QtCore 声明。
扫描器把实现私有头文件也按 header 计数，因此基线条目增加不等于新增 GUI 依赖。
Qt Widgets / GUI include 和反向流程依赖禁令未放宽；
WorkflowExecution 保持纯 C++，阶段仍有 QtCore 数据和内部 IO。

## 验证

新增独立行为测试只链接对应同步库，不依赖 Qt 生成器：

- WorkflowExecutionContract：进度归一化、取消合并、结果/错误定位。
- MvsPipelineServiceContract：同线程执行、完成一次、共享预取消、重复执行拒绝、显式重置。
- ModelWorkflowExecutionContract / TsdfWorkflowExecutionContract：
  各入口取消、外部任务取消、嵌套进度只报告一次、输入失败与成功结果契约。
- 原数据/后处理、深度工件重放、MVS、模型/TSDF、GUI 和仿真测试继续保留。

`tests/CoreImplementationBundles.h` 只为原有源码结构契约指定拆分后的显式文件集；
不是行为测试替身。边界测试另检查实际 CMake 依赖、纯 C++ 控制、
TSDF 后处理顺序和非拥有数组引用，以及产品不调用旧验证分支。

原生 Windows 验证入口：

```powershell
cmake --build --preset windows-source-release --parallel 8
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_core_boundaries.py'
.venv/Scripts/python.exe scripts/validation/check_core_boundaries.py
git diff --check
```

Linux/GCC、macOS/Apple Clang 必须在各自环境验证，不能从 MSVC 结果推定。
本轮原生 MSVC Release 完整构建及全量 CTest 已验证：3159 项登记，3141 项通过、
17 项跳过、1 项禁用基准，0 失败。跳过包括 11 项未配置 MC33 的关联测试、
3 项外部 Metashape/OBJ/真实网格夹具、1 项可选 TLS 网络验证及 2 项 offscreen 菜单交互。
6 个相关 Python 模块合计 64 个单测通过；边界扫描 728 条审查冻结项、0 禁止依赖。
CI YAML 解析及 12 个 build-test shell step 的 bash -n 通过；远端 CI 尚未执行。
以上数量记录阶段提取后的验证；兼容清理后的最终验证与剩余项见
[`COMPATIBILITY_CLEANUP.md`](COMPATIBILITY_CLEANUP.md)。本次按用户要求推送后不跟踪 CI。
粗/中/精三维物体与地形仿真生成/评分测试保留；源码重构与完整真实重建精度验证
是不同门禁，不能把评分脚本单测说成全部重建链路已完成端到端实测。

## CI Python 运行时修复

Linux build-test 在统一 .venv 安装 NumPy/Pillow/SciPy/rasterio/trimesh，再从固定 OpenCV 5.0.0
submodule 构建绑定并安装到同一环境，不混用系统 OpenCV 4 或不同 wheel。
安装 python3-dev，显式提供 Python/NumPy/site 路径；在主构建前检查 prefix、
5.0.0 版本、PNG/TIFF 内存编解码。headless/GUI 配置指定相同 Python3_EXECUTABLE。
Windows 独立 run_tests.py 在没有显式 PROJ_DATA/PROJ_LIB 时，从实际构建 CMakeCache 的
vcpkg 安装目录及 triplet 绑定存在的 proj.db，保持与构建 PROJ DLL 配套；不硬编码开发机路径。
远端 CI 状态只能依据发布这些改动后的 GitHub Actions 运行结果。
