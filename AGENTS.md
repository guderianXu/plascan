# AGENTS.md

本文件给后续 agent 使用，优先级低于用户当前指令和系统/开发者指令。目标是在 PlaScan 项目里改代码、验证功能、维护文档和处理模型资源时保持一致做法。

## 项目定位

PlaScan 是面向行星表面影像的摄影测量处理系统，主线是从多视角影像生成稀疏点云、密集点云、网格、DEM 和 DOM。项目包含：

- C++20 / CMake 的核心库、CLI 工具和 Qt6 GUI。
- OpenCV、TensorRT、GDAL、libtiff、libzip、OpenMP、GTest 等依赖。
- CUDA 可选加速，主要用于深度学习特征、匹配、MVS 和 dense match。
- Python 脚本用于模型导出、深度学习特征提取和辅助处理。
- `3rdparty/plapoint`、`3rdparty/plamatrix`、Qt、OpenCV、GDAL、AprilTag 和 PoissonRecon 等固定 git submodule。

先读现有实现、测试和文档，再改动。优先延续当前模块边界、命名方式和 UI 行为，避免无关重构。

## 工作区约束

- 开始改动前先检查工作区状态：
  ```bash
  git status --short
  git submodule status --recursive
  ```
- 可能存在 GUI、构建、测试或长时间运行的处理任务。Linux 需要确认时使用：
  ```bash
  pgrep -af 'plascan|feature_match_cli|match_photos_cli|dense_match_cli|triangulate_cli|cmake --build|ctest' || true
  ```
  macOS 使用：
  ```bash
  pgrep -fl 'plascan|feature_match_cli|match_photos_cli|dense_match_cli|triangulate_cli|cmake --build|ctest' || true
  ```
  Windows PowerShell 使用：
  ```powershell
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'plascan|feature_match_cli|match_photos_cli|dense_match_cli|triangulate_cli|cmake|ctest' } |
    Select-Object ProcessId, Name, CommandLine
  ```
- 不要随意删除、覆盖或重生成 `testData/`、`.plascan` 工程文件、`resources/models/`。需要清理磁盘或重建数据时先确认路径和用途。
- Agent 自行创建的一次性编译验证、算法测试中间文件、转换文件、日志和实验产物统一放在 `build/tmp/<task-name>/` 下，不要散落在仓库根目录、源码目录或 `testData/` 中。正式 CMake preset 构建树 `build/<preset>/`、环境配置 `build/env/` 和发布产物不属于临时目录。
- 任务结束时清理本次在 `build/tmp/` 中创建且不再需要的内容。清理前确认没有相关进程正在使用，只删除本次任务明确创建的路径；保留仍需复现、对比或排查问题的实验数据，并在最终说明中写明保留位置和原因。不要以“定期清理”为由删除其他任务或用户创建的内容。
- 子模块目录内的改动要特别谨慎。除非任务明确要求，不要重置、更新或提交 `3rdparty/plapoint`、`3rdparty/plamatrix` 的指针或内部 dirty state。
- 仓库可能已有用户改动。只修改与当前任务相关的文件，不回滚未授权改动。

## 代码规范

### C++

- 使用 C++20，4 空格缩进，不使用 tab，行宽尽量不超过 120 字符。
- 花括号使用 Allman 风格，左大括号独占一行。
  ```cpp
  if (ok)
  {
      runPipeline();
  }
  else
  {
      reportError();
  }
  ```
- 一个文件尽量不超过 400 行，嵌套尽量不超过 4 层；这是可维护性提示，不是机械门禁。生成代码、第三方适配、算法常量表或拆分后反而破坏内聚性的文件可以例外，并在改动说明中解释原因。
- 类和结构体使用 PascalCase，函数和方法使用 camelCase，局部变量使用 snake_case。
- 私有成员变量统一使用前导下划线 `_` 加 lowerCamelCase，例如 `_projectManager`、`_isRunning`、`_outputDir`。不要再为新增私有成员使用 `m_` 前缀或裸名。
- 公开成员、局部变量、全局变量和命名空间级函数不要使用前导下划线。避免 `__name` 和 `_Name` 这类 C++ 保留命名形式；Qt Designer 自动生成对象、第三方 API 适配字段或既有外部格式字段可保留原命名。
- 旧代码中已有的 `m_` 私有成员不做机械式全仓重命名。只有旧成员直接参与本次改动，且迁移不会扩大审查和回归风险时，才改为 `_` 前缀；不要把批量命名迁移混入功能修复，并同步更新受影响测试。
- 延续现有命名空间和目录边界。`src/core` 不应新增对 GUI 的依赖；GUI 通过 service、runner 或 task 调用核心能力。
- Qt GUI 中不要阻塞主线程。耗时处理放到已有 runner、manager、task 或 worker 模式中，并保持取消、进度、错误提示可用。
- 参数对话框应显示真实生效值；自动推导的模型、输出路径和设备选择也要能被用户看见。
- include 顺序优先为标准库、第三方库、项目头文件。涉及 Qt 与 Torch 宏冲突时，沿用项目里的兼容头文件做法。
- C/C++ 格式以仓库根目录 `.clang-format` 为准。只格式化本次修改的文件或代码段，不对无关文件做全量格式化；提交前检查格式化结果和 `git diff --check`。
- Windows/MSVC 与 Linux/GCC 都是一等支持目标。新增或修改的 C++、CUDA 和 CMake 代码必须同时符合两套工具链，不能依赖某个编译器的宽松扩展、整数底层类型巧合、静态库链接顺序或未声明的传递依赖。
- 跨编译器问题优先从类型、接口、模块职责和 CMake target 依赖关系上解决。除非第三方 API 或平台能力确实不同，不要用 `_MSC_VER`、`__GNUC__` 等条件分支或额外兼容层掩盖设计问题；确需平台分支时应缩小边界并写明原因。
- IO、模型加载、图像读取、CUDA 设备选择等失败路径要返回明确错误，不要静默降级。

### Python

- 项目统一复用仓库根目录 `.venv/` 作为本地 Python 开发环境。先检查并复用现有 `.venv/`；仅在环境缺失或依赖不完整时运行：
  ```bash
  python scripts/env/setup_python_runtime.py --device cpu
  ```
  Windows CUDA 开发机可用：
  ```powershell
  python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
  ```
  不要为单个 build 目录临时创建新的 Python 虚拟环境；只有 CI、打包或明确隔离需求才通过 `--runtime-dir` 指定其它位置。
- 使用 `pathlib.Path`、`argparse` 和结构化读写接口，避免硬编码本机绝对路径。
- 脚本入口保持 `parse_args()`、`main()` 结构，错误信息写清缺失路径、参数和建议修复方式。
- 深度学习脚本要明确区分 CPU/CUDA、模型路径、输入输出目录和阈值参数。
- `scripts/workflows/extract_features.py` 等运行时脚本依赖 torch、cv2 和相关模型库；验证失败时要说明缺少的 Python 包或环境。
- 新增长期保留的脚本时，同步补充脚本用途、输入输出和依赖说明；一次性探索脚本不要长期留在根目录。

## 构建与验证

### 编译器与标准入口

- 项目代码必须同时兼容 Windows/MSVC 和 Linux/GCC，但本地只要求完成当前原生平台的构建和测试门禁：Windows 使用 MSVC，Linux 使用 GCC，macOS 使用仓库现有 Apple Clang preset。Windows 不需要为了提交或推送额外启动 WSL/GCC；Linux/GCC 可由 CI 或专门环境补充。
- 只能报告实际完成的平台验证。不能用单个平台结果声称 MSVC、GCC 或 Apple Clang 均已通过；修复明确的跨平台、编译器或链接问题时，应在相关平台分别验证，无法执行的平台要写明原因。
- 优先使用 `CMakePresets.json` 和 `scripts/env/configure_with_env.py`，不要用依赖当前目录、默认生成器或 `nproc` 的临时命令替代标准入口。正式构建树默认位于 `build/<preset>/`；需要更换整套构建目录时使用统一脚本的 `--build-dir <path>`，不要手工拆散源码依赖和主工程路径。环境未初始化时按 `scripts/env/README.md` 配置 `.venv` 和 vcpkg，不要为单次构建创建另一套环境。

当前原生平台的 CPU Release 配置、构建和测试命令如下：

Linux：

```bash
python3 scripts/env/configure_with_env.py --source-deps --build --test
```

macOS Apple Silicon：

```bash
python scripts/env/configure_with_env.py --source-deps --build --test
```

Windows PowerShell：

```powershell
python scripts\env\configure_with_env.py --source-deps --build --test
```

CUDA、TensorRT、OpenCL 或打包任务使用 `CMakePresets.json` 中对应的专用 preset，不要在 CPU preset 上临时堆叠一组难以复现的 `-D` 参数。

### 验证等级

- 开发过程中先构建受影响 target，并运行最相关的测试以快速反馈。已有构建树可通过统一入口筛选测试，例如：
  ```bash
  python scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure -R 'SuperPoint|Feature|Match|DenseMatch|Mvs|Sfm|Terrain|Gui'
  ```
  Windows 或 macOS 将 `--test-dir` 替换为当前实际 preset 目录。也可直接运行已有测试二进制，但它只能作为定向反馈，不能替代要求的全量测试。
- 不需要 commit 时，验证范围与改动风险匹配即可。用户要求 commit 时，在提交前完成受影响目标、相关测试和静态检查，并重新检查差异。
- 用户明确要求 push，且改动涉及 C++、CUDA、CMake 或测试代码时，除定向测试外还必须在当前原生平台运行可执行的本地全量测试：
  ```bash
  python scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure
  ```
  使用当前实际 preset 目录；脚本默认使用全部逻辑线程。开发阶段的 `-R`、`--gtest_filter` 或单个测试程序不能作为 push 前的唯一测试依据。
- 本地门禁存在失败、未解释跳过、测试发现错误或运行环境缺失时，不得 push。先修复代码或环境并重新验证；确因外部依赖无法完成时，停止推送并向用户说明具体阻塞，只有获得明确许可后才能例外处理。
- push 后检查该 commit 对应的 GitHub Actions / required checks 并等待全部完成。CI 失败时读取日志，在原任务范围内修复后重新完成本地验证和 push；若修复需要扩大权限或范围，或因服务、额度、密钥等外部原因受阻，停止并明确报告。

### 按改动类型选择验证

- 仅文档：至少运行 `git diff --check`，并人工核对修改过的路径、命令、链接和行为描述；无需编译项目。
- C/C++、CUDA、CMake：格式化本次修改的代码，构建受影响 target，运行相关测试；push 前按上述门禁运行本地全量测试。
- Python：至少对所有修改过的脚本运行 `python -m py_compile <script>`，再按风险运行单元测试或脚本 `--help`。依赖 torch、cv2、kornia、LightGlue 等包的脚本必须使用依赖完整的 `.venv` 执行；缺少依赖时不能声称脚本验证通过。
- GUI：除编译外运行相关 GUI 测试；需要界面交互或图形环境而无法自动验证的部分，应说明人工验证步骤或未验证风险。
- 模型、资源和打包：验证实际查找路径、文件名、安装布局及对应 preset；不要把“文件存在”当作模型可加载或安装包可运行的充分验证。

## 模型与资源

- 预训练模型默认放在 `resources/models/`，也可能通过 `PLASCAN_MODEL_DIR` 指定。
- GUI 中自动选择的模型路径应显示给用户，且实际运行配置要与显示路径一致。
- 不要删除或替换模型文件，除非任务明确要求并说明来源。
- 修改模型查找逻辑时，同时考虑 CPU/CUDA 文件、Release 下载模型、源码树运行和安装后运行。
- 修改 `docs/models/README.md`、`README.md` 或模型导出脚本时，保持模型文件名、算法名和 GUI/CLI 行为一致。

## 模块约定

- `src/core`：放算法、数据模型和业务流程，不依赖 Qt Widgets 或 `src/gui`。需要呈现进度、错误或取消状态时通过稳定接口向上层暴露。
- `src/gui`：保持对话框、任务 runner、项目服务和主窗口职责分离。UI 文案使用中文，错误信息要能定位到路径、算法或参数。
- `src/cli`：CLI 参数应与核心配置一致，错误输出适合脚本调用和日志排查。
- `src/common`：只放跨模块通用能力，不把业务流程塞进 common。
- `tests` 和各模块内的 `tests`：测试按被测模块组织，优先使用稳定的最小夹具，不依赖开发机绝对路径、执行顺序或上一次运行残留；测试产生的临时数据遵循 `build/tmp/` 规则。
- `scripts`：长期脚本按环境、构建、模型或工作流职责放入对应子目录，提供 `--help`、明确输入输出并避免硬编码本机路径；一次性实验脚本及产物放入 `build/tmp/<task-name>/`，不要长期留在源码树。

## 文档同步

- 新增、删除或移动核心文件后，对照并更新 `docs/PROJECT_ARCHITECTURE.md`。
- 修改构建、依赖、模型、CLI 或 Docker 流程时，同步检查 `README.md`、`docs/models/README.md` 和 `docker/` 相关脚本说明。
- 版本更新文档集中维护：
  - `CHANGELOG.md`：根目录版本变更索引，按版本倒序记录每个版本的主要新增、优化、修复、验证和已知问题。
  - `docs/releases/RELEASE_PROCESS.md`：发版流程、分支规则、tag 规则、GitHub Release 模板和检查清单。
  - `docs/releases/vX.Y.Z*.md`：单个版本的详细说明，包括变更背景、影响范围、验证命令、测试数据结果和遗留风险。
  - 修改版本号、tag、Release 内容或阶段性功能说明时，同步更新以上文档；目录或文件不存在时先创建。
- 文档要写当前真实行为，不保留已经失效的命令、路径或测试数量。

## Git 与提交

- 只有用户明确要求时才 commit。commit、push、创建或切换分支、创建 tag、创建或修改 GitHub Release 是不同操作，不能互相推定授权；用户只说“提交”不代表允许 push，只有明确要求“提交并推送”或单独要求 push 时才修改远端状态。
- 除非用户主动、明确要求创建或使用分支，否则不要创建新分支，也不要切换到其它分支；
  默认在当前分支完成修改、验证和提交。不能仅因为任务类型、开发惯例或工具建议而自行开分支。
- 分支管理建议：
  - `main`：稳定可运行版本，只合并已经构建和测试验证过的代码。
  - `release/vX.Y.Z`：发版准备分支，用于版本号、文档、打包和最终验证。
- 只有用户明确要求发版或创建 tag 时才执行版本操作。版本 tag 使用语义化版本，并在快速迭代期使用预发布后缀：
  ```bash
  git tag -a v0.2.1-alpha.1 -m "PlaScan v0.2.1-alpha.1"
  git push origin v0.2.1-alpha.1
  ```
  不要每个 commit 都打 tag；完成用户可感知的一组功能、优化或修复后再打 tag。
- 推送版本 tag 后必须创建或更新同名 GitHub Release，不能只保留 GitHub Tags 页面里的默认短说明。Release 正文必须让用户直接看懂这个版本更新了什么，至少包含：新增、优化、修复、验证、已知问题；没有对应内容时写“无”或“未发现”，不要留空。
- GitHub Release 说明按固定结构编写：新增、优化、修复、验证、已知问题。验证项要写具体命令和通过/失败结果；若使用 `gh release create/edit`，正文优先来自 `docs/releases/vX.Y.Z*.md` 或同步内容，避免只写 `PlaScan vX.Y.Z`。
- 若用户授权修改版本、tag 或 Release 内容，同步维护 `CHANGELOG.md` 和 `docs/releases/` 下的对应版本文档；没有相关文件时应先创建。
- commit 前必须重新检查 `git status --short`，确认提交内容只包含当前任务相关文件。
- commit 和 push 前按“构建与验证”中的对应门禁执行，不得把 GitHub CI 当作本地基本验证的替代品。push 后可使用 `gh run list --commit <sha>`、`gh run watch <run-id> --exit-status` 或 `gh pr checks --watch` 跟踪检查，不得只确认 push 成功就结束任务。
- 若需要 commit，先检查当前仓库作者信息；缺失或不正确时仅在本仓库配置：
  ```bash
  git config user.email "guderian_xu@henu.edu.cn"
  git config user.name "guderianXu"
  ```
- 不要擅自修改远端地址、重写历史或强推。即使用户已授权普通 push，强推仍需要单独、明确授权。

## 沟通方式

- 对用户用中文简洁说明做了什么、验证了什么、还剩什么风险。
- 涉及失败、跳过验证、依赖缺失或历史问题时，给出具体命令、测试名和错误要点。
- 涉及路径时写绝对路径或项目相对路径，避免只说“这里”“那个文件”。
- 不要把临时 PID、一次性估算、私密 token 或 sudo 密码写进文档。
- 最终说明按任务实际情况覆盖以下信息，不需要为空项凑格式：
  - 修改了哪些文件及其用户可见效果。
  - 实际执行的构建、测试、格式或静态检查命令，以及通过、失败或跳过结果。
  - 未执行验证的具体原因、剩余风险和建议的后续动作。
  - `build/tmp/` 中保留的实验数据路径和保留原因，或确认已清理本次临时产物。
  - 是否执行 commit、push、tag 或 Release；若已 push，列出 commit 和 CI/required checks 状态。
