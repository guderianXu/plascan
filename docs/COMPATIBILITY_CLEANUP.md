# 兼容层清理记录

本次清理代码重构造成的转发壳和重复 API，并同步调用方、架构文档及测试。
它不是所有工程格式、历史工件和第三方依赖的统一迁移；下表明确区分已删除与未迁移。
本次包含前序未提交的 core 阶段提取和 CI Python 环境修复，按用户要求推送后不跟踪 CI。

## 已删除

| 兼容项 | 当前入口与影响 |
| --- | --- |
| core 的 `DepthMapGenerator.h/.cpp` 与 `mvs` target | CLI 直接调用同步 `MvsPipelineService`；异步生命周期移至 `gui/project/tasks/DepthMapTask`，不再属于 core API |
| 生成器的静态转发与同步服务的四个后处理转发 | 调用方直接使用真实实现接口；像素后处理统一调用 `DepthPostprocessor` |
| `meshing` INTERFACE 链接别名 | 业务调用方链接 `model_workflow`；质量评估链接 `meshing_algorithms` |
| 不可达的旧模型生产分支 | 删除 `LegacyDepthModelStages.cpp`、私有声明和链接入口；只保留明确的 recovered 生产路径，不增加自动回退 |
| 密集匹配 `DenseMatchConfig::useCuda` | CLI、算法与测试统一使用 `computeBackend`；显式加速器失败不得回退，Automatic 仍按真实设备可用性选择 |
| TensorRT `TensorRtEngineBuildRequest::fixedKeypointCount` 与形状补全分支 | LoMa-R 调用方直接填写六种 `inputShapes`，构建器只处理通用输入契约 |
| 四种模型请求的旧 `isCancelled`/`progress` 与双回调桥接 | 公开请求只携带 `execution`；私有 `bindWorkflowCallbacks` 只转换唯一控制为算法回调，不接受旧请求字段 |

旧 C++ 接口和 target 不再可用，仓库内调用方已迁移。
GUI 的 future 持有、取消、析构等待和重复启动拒绝仍保留，这是必要的线程职责，不是旧算法兼容壳。

## 确认仍保留、尚未迁移

| 剩余项 | 定位 | 为什么没有直接删除 |
| --- | --- | --- |
| TSDF 旧回调与 execution 双接口及取消合并 | `src/core/mesh/DepthTsdfSurfaceBuilder.h/.cpp`、`task_runtime/WorkflowExecution.h` 的 `combineWorkflowCancellation` | 显式 TSDF 算法及相关测试仍使用旧回调；需要单独迁移 options 与所有算法阶段，不能称为已完成 |
| BA 的 LegacyCpu 枚举 / legacy_cpu 参数映射 | `src/core/bundle_adjust/BundleAdjustTypes.h`、`BundleAdjust.cpp`、`src/cli/reconstruction/cli_bundle_adjust.cpp` | 仍接受旧工程/CLI 的名称，实际映射 PlaMatrix CPU；旧独立求解器并不存在 |
| BA 无效旧参数与对比命名 | `BundleAdjustOptions.h`、`tools/ba_backend_benchmark.cpp` | 如 huberDelta、旧 dense Schur 阈值、compareAutoBackendWithLegacy；需要迁移配置、基准输出和相应测试 |
| 工程配置字段默认补全、缺 UUID 补齐、旧 shared image store 输入 | `src/common/project/ProjectConfigManager.cpp`、`ProjectSessionModel.h/.cpp` | 与已有工程内容读取有关，删除前需明确数据迁移方案；不意味着任意旧 .plascan 容器都可读取 |
| GUI 会话兼容门面 | `src/gui/project/services/ProjectSessionFacade.h/.cpp` | 仍被 ProjectManager 等调用，包含元数据归一化、空会话检查等业务行为；尚未将调用方统一至新的会话接口，不能作为纯空壳删除 |
| 地形成果 JSON 双字段写入与读取 | `src/core/terrain/TerrainProductManifest.cpp` | dem_tif/dom_png/depth_png 与规范字段并存；GUI 元数据及旧记录仍消费别名，需要联合迁移读写方 |
| 旧深度工件重放 / 源计划 / PatchMatch 诊断路径 | `src/core/mvs/MvsWorkspaceReplay.cpp`、`MvsSourcePlanner.h/.cpp`、`PatchMatchEstimator.cpp` | 并非当前 recovered 正式生产器；显式重放、诊断和算法测试仍有使用，不能按名字删除整套实现 |
| Python 旧环境变量及生成配置读取 | `src/common/runtime/PythonRuntimeLocator.cpp` | PLASCAN_PYTHON 作为 PLASCAN_PYTHON_EXECUTABLE 的旧别名，另读取 build/env/plascan-env.json；需要同步环境与安装入口 |

这是本次源码审查确认的剩余清单，不是通过关键词自动推断的“全仓无兼容层”证明。
后续优先迁移 TSDF 控制，再处理 BA 参数；工程与成果字段应先提供显式迁移工具，
完成新格式读写与旧数据验收后再删除兼容读取。

## 必须保留的真实边界

- `recovered_depth/src/metalign_compat.cpp` 实现 recovered 深度路径需要的几何/影像适配，
  不是旧 PlaScan 接口的转发壳；删除会破坏当前算法链接与运行。
- 相机/深度工件的身份、版本、质量与场景校验是有效性门禁，不因名字包含 compatibility 就删除。
- MSVC/GCC、Qt、OpenCV、CUDA/TensorRT 和模型资源适配属于实际平台/依赖差异；
  本次不删除必要的平台边界或模型文件。

## 验证

原生平台：Windows/MSVC，使用现有 `windows-source-release` Release preset
（当前构建树启用 CUDA、TensorRT 和 OpenCL），没有以本机结果代替 Linux/GCC 验证。

执行入口：

```powershell
cmake --build --preset windows-source-release --parallel 8
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure
.venv/Scripts/python.exe -m py_compile scripts/env/run_tests.py tests/test_core_boundaries.py tests/test_repo_hygiene.py tests/test_run_tests.py
.venv/Scripts/python.exe -m unittest tests.test_core_boundaries tests.test_run_tests tests.test_repo_hygiene
.venv/Scripts/python.exe scripts/validation/check_core_boundaries.py
git diff --check
```

源码边界测试防止旧头文件、旧 target、后处理转发、CLI 事件循环和公开模型双回调重新引入。
更新边界基线时审查实际差异：移除 core 生成器的 Qt 调度耦合，
公开模型请求中的四个 QString 回调移至私有算法转换边界，未放宽禁止 GUI 依赖的规则。
最终验证结果：

- MSVC Release 完整构建通过；最后仅修改 CLI 源码契约测试后，完整增量构建再次通过。
- 全量 CTest：3159 项登记、3141 项通过、17 项跳过、1 项禁用基准、0 失败，55.63 秒。
  跳过为 11 项未配置 MC33 的关联测试、3 项外部 Metashape/OBJ/真实网格数据、
  1 项可选 TLS 网络测试及 2 项 offscreen 菜单交互；禁用项为可选 CUDA 对比基准。
- 相关模型/MVS/控制/CLI/边界定向测试 28 项通过；相关 Python 单测 47 项通过，py_compile 通过。
- 边界扫描：707 条冻结项、0 禁止依赖；git diff --check 通过。
- 首次全量中的 MeshReconstructCliGTest.UsesSharedModelWorkflowEntry 因仍断言旧 meshing
  target 失败，更新为 model_workflow 并增加 execution.progress 检查后，定向及全量重跑均通过。
- Linux/GCC、macOS/Apple Clang 和人工 GUI 交互未执行，本机结果不代表这些验证通过。

本次未新增需要保留的 build/tmp 实验目录；正式构建树及其测试日志不属于一次性清理对象。
按用户要求不等待、不查看、不修复本次推送后的 CI，不将未知 CI 状态记为通过。
