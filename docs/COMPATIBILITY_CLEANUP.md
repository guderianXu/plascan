# 兼容层清理记录

本次清理代码重构造成的转发壳和重复 API，并同步调用方、架构文档及测试。
下表区分已删除与未迁移。旧工程不再静默升级：缺失/重复影像 UUID、缺失配置、
旧共享源影像和地形成果别名会被拒绝，需要新建项目重新导入或重新生成成果。
不提供兼容读取或自动迁移，不修改用户原始影像和旧工程文件。

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
| 旧工程配置默认补全与影像 UUID 修复 | 删除 mergeWithDefaults 与 ensureImageUuids；打开和恢复只校验，新建/正常导入仍生成默认配置和 UUID |
| 旧共享源影像注册接口 | addValidatedExternalImages 仅接收外部源影像；拒绝 shared 类型和共享源路径；新便携导出明确使用 packaged 类型，可重新打开与再次导出 |
| 地形成果字段别名 | 读写方统一 dem_path、dom_path、preview_path；拒绝 dem_tif/dom_png 及 DEM 的 depth_png/depth_preview_png；MVS depth_png 工件字段保留 |
| TSDF 旧回调与取消合并 | DepthTsdfOptions 仅保留 execution，全部算法阶段和测试已迁移；删除 combineWorkflowCancellation，保留真实 TSDF 算法及诊断快照 |
| BA 旧后端与无效参数 | 删除 LegacyCpu/legacy_cpu、旧点/相机迭代上限、影像 Huber、有限差分/阻尼/步长及无效 dense Schur 参数；GUI/CLI/基准和配置同步删除，拒绝旧字段和名称 |
| BA 无条件 CPU 对照求解 | 删除 compareAutoBackendWithLegacy 与对比路径；只在状态或质量门控拒绝时运行当前 CPU 回退，不删除当前 PlaMatrix 求解器 |

旧 C++ 接口和 target 不再可用，仓库内调用方已迁移。
GUI 的 future 持有、取消、析构等待和重复启动拒绝仍保留，这是必要的线程职责，不是旧算法兼容壳。

## 确认仍保留、尚未迁移

| 剩余项 | 定位 | 为什么没有直接删除 |
| --- | --- | --- |
| GUI 会话兼容门面 | `src/gui/project/services/ProjectSessionFacade.h/.cpp` | 仍被 ProjectManager 等调用，包含元数据归一化、空会话检查等业务行为；尚未将调用方统一至新的会话接口，不能作为纯空壳删除 |
| 旧深度工件重放 / 源计划 / PatchMatch 诊断路径 | `src/core/mvs/MvsWorkspaceReplay.cpp`、`MvsSourcePlanner.h/.cpp`、`PatchMatchEstimator.cpp` | 并非当前 recovered 正式生产器；显式重放、诊断和算法测试仍有使用，不能按名字删除整套实现 |
| Python 旧环境变量及生成配置读取 | `src/common/runtime/PythonRuntimeLocator.cpp` | PLASCAN_PYTHON 作为 PLASCAN_PYTHON_EXECUTABLE 的旧别名，另读取 build/env/plascan-env.json；需要同步环境与安装入口 |
| 相机标定旧字段读取 | `src/gui/dialogs/camera/CameraCalibrationData.cpp` | legacyCamera/copyLegacyParameter 仍读取比较记录中的旧 before/after 扁平字段；本轮 BA 求解器清理未迁移此展示格式 |
| 模型减面旧 targetFaces 字段 | `src/gui/dialogs/reconstruction/GenerateModelDialog.cpp`、`application/WorkflowSettingsDialog.cpp` | 仍将旧面数推导为 faceCountMode/faceCountCustom；属于参数 UI 迁移，不是 TSDF 求解器接口 |
| BA 基准 seconds 输出别名 | `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp`、`scripts/bench/run_ba_backend_benchmark.py` | seconds 与 api_wall_seconds 同时输出；本轮删除旧求解器设置，但未更改现有 CSV 指标格式 |

这是本次源码审查确认的剩余清单，不是通过关键词自动推断的“全仓无兼容层”证明。
本轮完成 TSDF 控制和 BA 求解器接口清理；上述会话、诊断、环境、展示字段及基准输出格式仍保留。

## 必须保留的真实边界

- `recovered_depth/src/metalign_compat.cpp` 实现 recovered 深度路径需要的几何/影像适配，
  不是旧 PlaScan 接口的转发壳；删除会破坏当前算法链接与运行。
- 相机/深度工件的身份、版本、质量与场景校验是有效性门禁，不因名字包含 compatibility 就删除。
- PlanetaryLineScanBundleAdjustOptions::maxDenseSchurCameras 参与当前线阵 BA 的实际求解；
  与已删除的 BAOptions 同名无效字段不同，不能作为旧接口删除。
- MSVC/GCC、Qt、OpenCV、CUDA/TensorRT 和模型资源适配属于实际平台/依赖差异；
  本次不删除必要的平台边界或模型文件。

## 前序 core 清理验证（不代表本次工程字段清理）

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
前序提交按用户要求未跟踪推送后的 CI，不将未知 CI 状态记为通过。

## 本次工程兼容与字段清理验证

Windows/MSVC Release 完整构建通过：

```powershell
cmake --build --preset windows-source-release --parallel 8
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure -R 'Project|Terrain|DemDom|WorkspaceSection'
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure
.venv/Scripts/python.exe -m py_compile scripts/validation/generate_synthetic_terrain_dataset.py scripts/validation/synthetic_e2e_metrics.py tests/test_synthetic_e2e.py
.venv/Scripts/python.exe -m unittest tests.test_synthetic_e2e tests.test_generate_synthetic_terrain_dataset tests.test_core_boundaries tests.test_repo_hygiene
.venv/Scripts/python.exe scripts/validation/check_core_boundaries.py
git diff --check
```

- 定向测试：391 项登记，390 通过、1 外部 Metashape 数据缺失跳过，0 失败；
  修正属性面板后，相关拒绝旧数据、便携包跨 Chunk 打开与 DEM/DOM 分辨率 5 项再次通过。
- 全量重跑：3163 项登记，3145 通过、17 跳过、1 禁用，0 失败，55.24 秒。
  跳过原因与前序一致：11 项缺 MC33、3 项外部数据、1 项可选 TLS、
  2 项 offscreen 菜单交互；禁用为可选 CUDA 对比基准。
- Python 单测 50 项通过，py_compile 通过；边界扫描 707 条冻结项、0 禁止依赖。
- 测试覆盖缺失/非法/重复 UUID、旧 shared 输入、旧 ui 配置、旧地形字段、
  无效清单格式、拒绝快照时保留活动项目以及新便携导出包跨 Chunk 打开。
- 首轮全量发现 SelectionPropertiesWidgetTest.ShowsDemAndDomPixelResolution 失败：
  正射记录的参考 dem_path 被误当成果路径；调整为优先 output_path/dom_path 后定向与全量通过。
- 原始影像、旧工程、testData、模型资源及子模块未修改；未创建本次一次性 build/tmp 目录。
- 本轮未执行 commit、push、tag、Release 或 CI 操作；Linux/GCC、macOS 与人工 GUI 交互未验证。

## 本次 BA / TSDF 兼容清理验证

保留当前 PlaMatrix CPU/CUDA/OpenCL 联合 BA 和真实 TSDF 重建算法；删除的是旧兼容接口，
不是有效求解器。TSDF 进度与取消仅通过 execution；算法内部仍使用自己的原生回调签名，
由唯一控制契约转换，不接受或合并旧公开字段。

```powershell
cmake --build --preset windows-source-release --parallel 8
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure -R 'BundleAdjust|Tsdf|WorkflowExecution|ProjectConfigManager'
python scripts/env/run_tests.py --test-dir build/windows-source-release --output-on-failure
.venv/Scripts/python.exe -m py_compile scripts/bench/run_ba_backend_benchmark.py tests/test_core_boundaries.py tests/test_ba_cuda_contracts.py src/cli/reconstruction/tests/test_bundle_adjust_cli_contracts.py
.venv/Scripts/python.exe -m unittest tests.test_core_boundaries tests.test_repo_hygiene tests.test_run_tests tests.test_ba_cuda_contracts
.venv/Scripts/python.exe src/cli/reconstruction/tests/test_bundle_adjust_cli_contracts.py
.venv/Scripts/python.exe scripts/bench/run_ba_backend_benchmark.py --help
.venv/Scripts/python.exe scripts/validation/check_core_boundaries.py
git diff --check
```

- Windows/MSVC Release 完整构建通过，后续增量构建通过。
- BA/TSDF/控制/配置定向测试 299 项登记，297 通过、2 项缺 MC33 跳过，0 失败，9.95 秒。
- 全量 CTest 3166 项登记，3148 通过、17 跳过、1 禁用，0 失败，54.31 秒；
  跳过原因与前序一致，不将跳过和禁用项记为通过。
- Python 单测 55 项通过，另 CLI 契约 4 项通过；py_compile 和脚本 --help 通过。
- 实际 BA CLI --help 不再列出八个旧参数，逐项调用返回 unknown argument；
  实际基准程序拒绝 --max-dense-schur-cameras 和 legacy_cpu。
- 初轮定向失败为两个过时契约断言：旧 CPU 映射提示和旧 Auto 策略迁移分支；
  更新为当前 CPU 后端、无兼容回退及当前独立规模参数契约后，定向与全量重跑通过。
- 删除未生效的 HuberDeltaSensitivity 测试，保留实际观测权重、数值后端一致性和内参测试。
  新增旧 BA 字段/后端拒绝、纯 TSDF 取消检查、Auto BA 取消后不进行质量回退测试。
- 边界扫描 707 条冻结项、0 禁止依赖；仅将 TSDF 公开头的 QString 次数从 11 减至 10，
  对应已删除进度回调，未放宽任何禁止规则；git diff --check 通过。
- 本次迁移辅助程序位于 build/tmp/ba-tsdf-cleanup，验证后已删除；没有需要保留的临时产物。
- 原始工程、testData、模型和子模块未修改；未执行 commit、push、tag、Release 或 CI 操作。
  Linux/GCC、macOS 与人工 GUI 交互未执行，不以本机结果代替验证。
