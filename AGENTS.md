# AGENTS.md

本文件给后续 agent 使用，优先级低于用户当前指令和系统/开发者指令。目标是在 PlaScan 项目里改代码、验证功能、维护文档和处理模型资源时保持一致做法。

## 项目定位

PlaScan 是面向行星表面影像的摄影测量处理系统，主线是从多视角影像生成稀疏点云、密集点云、网格、DEM 和 DOM。项目包含：

- C++20 / CMake 的核心库、CLI 工具和 Qt6 GUI。
- OpenCV、TensorRT、GDAL、libtiff、libzip、OpenMP、GTest 等依赖。
- CUDA 可选加速，主要用于深度学习特征、匹配、MVS 和 dense match。
- Python 脚本用于模型导出、深度学习特征提取和辅助处理。
- `3rdparty/plapoint` 和 `3rdparty/plamatrix` 两个 git submodule。

先读现有实现、测试和文档，再改动。优先延续当前模块边界、命名方式和 UI 行为，避免无关重构。

## 工作区约束

- 开始改动前先检查工作区状态：
  ```bash
  git status --short
  git submodule status --recursive
  ```
- 可能存在 GUI、构建、测试或长时间运行的处理任务。需要确认时使用：
  ```bash
  pgrep -af 'plascan|feature_match_cli|match_photos_cli|dense_match_cli|triangulate_cli|cmake --build|ctest' || true
  ```
- 不要随意删除、覆盖或重生成 `testData/`、`.plascan` 工程文件、`resources/models/`。需要清理磁盘或重建数据时先确认路径和用途。
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
- 一个文件尽量不超过 400 行，嵌套尽量不超过 4 层。超过时优先拆分真实职责，不做机械式拆分。
- 类和结构体使用 PascalCase，函数和方法使用 camelCase，局部变量使用 snake_case。
- 私有成员变量统一使用前导下划线 `_` 加 lowerCamelCase，例如 `_projectManager`、`_isRunning`、`_outputDir`。不要再为新增私有成员使用 `m_` 前缀或裸名。
- 公开成员、局部变量、全局变量和命名空间级函数不要使用前导下划线。避免 `__name` 和 `_Name` 这类 C++ 保留命名形式；Qt Designer 自动生成对象、第三方 API 适配字段或既有外部格式字段可保留原命名。
- 旧代码中已有的 `m_` 私有成员不做机械式全仓重命名；修改相关类时，在不扩大风险的前提下逐步迁移到 `_` 前缀，并同步更新对应测试。
- 延续现有命名空间和目录边界。`src/core` 不应新增对 GUI 的依赖；GUI 通过 service、runner 或 task 调用核心能力。
- Qt GUI 中不要阻塞主线程。耗时处理放到已有 runner、manager、task 或 worker 模式中，并保持取消、进度、错误提示可用。
- 参数对话框应显示真实生效值；自动推导的模型、输出路径和设备选择也要能被用户看见。
- include 顺序优先为标准库、第三方库、项目头文件。涉及 Qt 与 Torch 宏冲突时，沿用项目里的兼容头文件做法。
- Windows/MSVC 与 Linux/GCC 都是一等支持目标。新增或修改的 C++、CUDA 和 CMake 代码必须同时符合两套工具链，不能依赖某个编译器的宽松扩展、整数底层类型巧合、静态库链接顺序或未声明的传递依赖。
- 跨编译器问题优先从类型、接口、模块职责和 CMake target 依赖关系上解决。除非第三方 API 或平台能力确实不同，不要用 `_MSC_VER`、`__GNUC__` 等条件分支或额外兼容层掩盖设计问题；确需平台分支时应缩小边界并写明原因。
- IO、模型加载、图像读取、CUDA 设备选择等失败路径要返回明确错误，不要静默降级。

### Python

- 项目统一复用仓库根目录 `.venv/` 作为本地 Python 开发环境。需要 Python 依赖时先运行或复用：
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

### 编译器支持

- 项目必须同时支持 Windows 上的 MSVC 和 Linux/WSL 上的 GCC。任何涉及 C++、CUDA、链接关系或构建系统的改动，都应在两种编译器下验证受影响目标；公共库或核心 CMake 关系发生变化时，应尽量完成两边的默认全目标构建。
- 优先复用已经配置好的 MSVC 与 GCC 构建目录，避免无必要地重复安装依赖。两边都可用时，只验证其中一个编译器不能视为跨平台验证完成。
- 若当前环境缺少其中一套工具链、SDK 或第三方依赖，必须在最终说明中列出未验证的编译器、具体缺失项和已执行的另一侧验证，不能声称 MSVC/GCC 均已通过。
- 修复跨平台构建问题时，应补充能覆盖公共语义的测试，并分别运行相关测试；编译器、链接器或平台专属测试仅在确有专属行为时使用。

改 C++ 后至少在 `build` 目录编译验证：

```bash
cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j$(nproc)
```

按改动范围优先跑相关测试，再决定是否跑全量：

```bash
ctest --output-on-failure -R 'SuperPoint|Feature|Match|DenseMatch|Mvs|Sfm|Terrain|Gui'
ctest --output-on-failure
```

也可直接运行单个测试二进制，例如：

```bash
./tests/test_gui_project_utils
```

改 Python 脚本后至少做语法检查：

```bash
python -m py_compile scripts/workflows/extract_features.py
```

需要运行依赖 torch、cv2、kornia 或 LightGlue 的脚本时，先初始化 `.venv`，再优先使用 `.venv` 中的解释器，例如 Windows：

```powershell
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
.\.venv\Scripts\python.exe scripts\models\export_lightglue_tensorrt.py --help
```

若需要运行脚本本体，必须使用同时包含 torch、cv2 和相关模型依赖的 Python 环境。不要在依赖缺失时声称脚本验证通过。

## 模型与资源

- 预训练模型默认放在 `resources/models/`，也可能通过 `PLASCAN_MODEL_DIR` 指定。
- GUI 中自动选择的模型路径应显示给用户，且实际运行配置要与显示路径一致。
- 不要删除或替换模型文件，除非任务明确要求并说明来源。
- 修改模型查找逻辑时，同时考虑 CPU/CUDA 文件、Release 下载模型、源码树运行和安装后运行。
- 修改 `docs/models/README.md`、`README.md` 或模型导出脚本时，保持模型文件名、算法名和 GUI/CLI 行为一致。

## 模块约定

- `src/gui`：保持对话框、任务 runner、项目服务和主窗口职责分离。UI 文案使用中文，错误信息要能定位到路径、算法或参数。
- `src/cli`：CLI 参数应与核心配置一致，错误输出适合脚本调用和日志排查。
- `src/common`：只放跨模块通用能力，不把业务流程塞进 common。

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

- 每一次修改代码后都要 commit。
- 分支管理建议：
  - `main`：稳定可运行版本，只合并已经构建和测试验证过的代码。
  - `release/vX.Y.Z`：发版准备分支，用于版本号、文档、打包和最终验证。
- 版本 tag 使用语义化版本，并在快速迭代期使用预发布后缀：
  ```bash
  git tag -a v0.2.1-alpha.1 -m "PlaScan v0.2.1-alpha.1"
  git push origin v0.2.1-alpha.1
  ```
  不要每个 commit 都打 tag；完成用户可感知的一组功能、优化或修复后再打 tag。
- 推送版本 tag 后必须创建或更新同名 GitHub Release，不能只保留 GitHub Tags 页面里的默认短说明。Release 正文必须让用户直接看懂这个版本更新了什么，至少包含：新增、优化、修复、验证、已知问题；没有对应内容时写“无”或“未发现”，不要留空。
- GitHub Release 说明按固定结构编写：新增、优化、修复、验证、已知问题。验证项要写具体命令和通过/失败结果；若使用 `gh release create/edit`，正文优先来自 `docs/releases/vX.Y.Z*.md` 或同步内容，避免只写 `PlaScan vX.Y.Z`。
- 若修改了版本、tag 或 Release 内容，同步维护 `CHANGELOG.md` 和 `docs/releases/` 下的对应版本文档；没有相关文件时应先创建。
- commit 前必须重新检查 `git status --short`，确认提交内容只包含当前任务相关文件。
- 若需要 commit，Git 作者信息使用：
  ```bash
  git config user.email "guderian_xu@henu.edu.cn"
  git config user.name "guderianXu"
  ```
- 按 `CLAUDE.md` 要求，commit 后需要 push 到 GitHub；除非用户另有指令，不要擅自改远端、重写历史或强推。

## 沟通方式

- 对用户用中文简洁说明做了什么、验证了什么、还剩什么风险。
- 涉及失败、跳过验证、依赖缺失或历史问题时，给出具体命令、测试名和错误要点。
- 涉及路径时写绝对路径或项目相对路径，避免只说“这里”“那个文件”。
- 不要把临时 PID、一次性估算、私密 token 或 sudo 密码写进文档。
