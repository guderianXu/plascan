# Windows CUDA build scripts

本目录用于在 Windows 原生环境下构建 PlaScan，不使用 WSL。当前推荐入口是
`build_windows_cuda.ps1`。

脚本目标是固定一套自洽的 Windows CUDA Release 构建环境：

- 主构建目录：`E:\code\plascan\build\windows-vcpkg-cuda-release`
- vcpkg 依赖：`E:\code\plascan\build\windows-vcpkg-cuda-release\vcpkg_installed`
- CUDA：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
- LibTorch：`E:\code\plascan\build\env\libtorch-cu130\libtorch`

脚本会设置统一的 `PATH`、`Torch_DIR`、`PLASCAN_TORCH_DIR`、`CMAKE_PREFIX_PATH`、
`CUDA_PATH` 和 Qt plugin 环境，并同步 Qt platform plugins 与 LibTorch 运行时 DLL，
避免 CMake/CTest 混用旧 CPU LibTorch、旧 vcpkg 或其它项目的依赖。

## 前置条件

默认路径需要存在：

- `C:\BuildTools\Common7\Tools\VsDevCmd.bat`
- `C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`
- `C:\BuildTools\VC\vcpkg`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
- `E:\code\plascan\build\env\libtorch-cu130\libtorch`
- `E:\code\plascan\build\windows-vcpkg-cuda-release\vcpkg_installed\x64-windows`

如果路径不同，用参数覆盖，见“参数速查”。

默认不会自动运行 vcpkg manifest install；脚本会先检查自己的
`vcpkg_installed\x64-windows` 里是否已有 Qt6、OpenCV、GDAL、GTest、libzip 和 TIFF。
需要自动补依赖时再显式加 `-InstallDeps`。

## 常用命令

在 PowerShell 中从任意目录运行均可：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -Jobs 8
```

只编译 GUI：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -Target plascan_gui -Jobs 8
```

只重新配置 CMake，不编译：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -ConfigureOnly
```

只编译，不重新配置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -BuildOnly -Jobs 8
```

清理根 `build` 目录里的旧 Linux/WSL CMake 残留，并清理当前 CUDA 构建配置后重新构建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -CleanRootCache -CleanConfigure -Jobs 8
```

构建并运行指定测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -BuildOnly -RunTests -CTestRegex "PatchMatch|MvsPipeline" -Jobs 8
```

构建并运行全部 CTest：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -BuildOnly -RunTests -Jobs 8
```

## 输出位置

主要可执行文件：

- GUI：`E:\code\plascan\build\windows-vcpkg-cuda-release\bin\plascan_gui.bin.exe`
- CLI：`E:\code\plascan\build\windows-vcpkg-cuda-release\bin`
- 测试：`E:\code\plascan\build\windows-vcpkg-cuda-release\tests`

脚本会把 Qt platform plugins 同步到：

- `E:\code\plascan\build\windows-vcpkg-cuda-release\bin\platforms`
- `E:\code\plascan\build\windows-vcpkg-cuda-release\tests\platforms`

## 参数速查

| 参数 | 作用 |
| --- | --- |
| `-SourceDir <path>` | 源码目录，默认脚本目录上两级，即项目根目录。 |
| `-BuildDir <path>` | 构建目录，默认 `build\windows-vcpkg-cuda-release`。必须位于项目 `build` 目录下。 |
| `-VcpkgRoot <path>` | vcpkg 根目录，默认 `C:\BuildTools\VC\vcpkg`。 |
| `-VsDevCmd <path>` | VS Build Tools 环境脚本。 |
| `-CMakeExe <path>` | CMake 可执行文件路径。 |
| `-CudaRoot <path>` | CUDA Toolkit 根目录，默认 CUDA 13.1。 |
| `-TorchRoot <path>` | LibTorch 根目录，默认 `build\env\libtorch-cu130\libtorch`。 |
| `-Target <name>` | 只构建指定 CMake target，例如 `plascan_gui`。 |
| `-CTestRegex <regex>` | `ctest -R` 过滤表达式。 |
| `-Jobs <n>` | 并行构建线程数，默认 CPU 核心数。 |
| `-ConfigureOnly` | 只配置，不构建。 |
| `-BuildOnly` | 只构建，不配置。 |
| `-RunTests` | 构建后运行 CTest。 |
| `-CleanConfigure` | 清理当前构建目录中除 `vcpkg_installed` 外的 CMake/编译产物。 |
| `-CleanRootCache` | 清理根 `build` 目录旧 CMake/Ninja/Linux 残留。 |
| `-InstallDeps` | 允许 CMake/vcpkg 自动执行 manifest install。 |
| `-SkipVsDevCmd` | 跳过加载 VS Build Tools 环境，仅在当前 shell 已配置好编译器时使用。 |

## 排错

### 命令行中文输出乱码

如果命令行里出现 `鍥惧儚`、`棰勫姞杞` 这类乱码，不要改 C++ 宽字符串。
这是 Windows 终端代码页没有按 UTF-8 解码原生程序输出。先在当前 PowerShell
里加载环境脚本，再运行 GUI 或 CLI：

```powershell
. E:\code\plascan\build\env\plascan-env.ps1
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\plascan_gui.bin.exe
```

`build_windows_cuda.ps1` 和 `plascan-env.ps1` 会设置：

- `chcp 65001`
- PowerShell `[Console]::InputEncoding` / `[Console]::OutputEncoding`
- `PYTHONUTF8=1`
- `PYTHONIOENCODING=utf-8`

### Qt platform plugin 弹窗

现象：

```text
This application failed to start because no Qt platform plugin could be initialized.
```

脚本会设置：

- `QT_QPA_PLATFORM=offscreen`
- `QT_PLUGIN_PATH=<build>\vcpkg_installed\x64-windows\Qt6\plugins`
- `QT_QPA_PLATFORM_PLUGIN_PATH=<build>\vcpkg_installed\x64-windows\Qt6\plugins\platforms`

并复制 `qwindows.dll`、`qoffscreen.dll`、`qminimal.dll` 到 `bin\platforms` 和
`tests\platforms`。如果手动运行测试或 GUI，也要保证这些目录存在。

### 运行测试出现 `0xc0000139`

通常是输出目录混入旧 CPU LibTorch DLL，或 CUDA 版 LibTorch 的 cuDNN、CUPTI、
cuBLAS 等延迟加载 DLL 没有进入输出目录。脚本会把当前 CUDA LibTorch 的完整
DLL 运行时同步到 `bin`、`tests` 和已有测试输出目录。
优先用本脚本重新构建，不要直接手工拼 PATH。

### CMake 又引用旧目录

检查：

```powershell
Select-String E:\code\plascan\build\windows-vcpkg-cuda-release\CMakeCache.txt `
  -Pattern "windows-vcpkg-release|sat_sim_cuda|build/env/libtorch/libtorch"
```

正常情况下不应有输出。若出现旧路径，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\code\plascan\scripts\build_win\build_windows_cuda.ps1 -CleanConfigure -ConfigureOnly
```

### vcpkg ICU/pkg-config 构建失败

如果使用 `-InstallDeps` 触发 vcpkg 安装，可能遇到本机 vcpkg/ICU 的
`icu-i18n.pc` 查找失败。日常构建默认不加 `-InstallDeps`，而使用当前 CUDA 构建目录
自己的 `vcpkg_installed`。需要重建依赖时，先确认 vcpkg baseline 和二进制缓存状态。

## 建议

- 日常只使用本脚本构建 Windows CUDA 版本。
- 不要直接在 `E:\code\plascan\build` 根目录运行 `cmake ..`。
- 不要把 `windows-vcpkg-release`、`sat_sim_cuda` 或旧 CPU LibTorch 加进 PATH。
- 清理空间时优先清旧包、benchmark 输出和 vcpkg `blds/pkgs` 缓存；不要删当前
  `windows-vcpkg-cuda-release\vcpkg_installed\x64-windows`，除非准备重新补依赖。
