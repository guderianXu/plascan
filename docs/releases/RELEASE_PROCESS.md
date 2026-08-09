# PlaScan Release Process

本流程用于记录 PlaScan 的分支、tag 和 GitHub Release 更新方式。

## 分支

- `main`：稳定可运行分支，所有对外可见版本必须从这里打 tag。
- `develop`：可选集成分支，用于中期汇总多个功能。
- `feature/<name>`：功能开发分支。
- `fix/<name>`：缺陷修复分支。
- `release/vX.Y.Z`：可选发版准备分支，用于最终文档、版本号和打包验证。

小规模迭代允许直接使用 `main` 加 `feature/*` / `fix/*`，但发布前必须把变更合入 `main`。

## Tag 规则

- 使用语义化版本：`vX.Y.Z`。
- 快速迭代和未稳定版本使用预发布后缀：`vX.Y.Z-alpha.N`、`vX.Y.Z-beta.N`、`vX.Y.Z-rc.N`。
- tag 必须从 `main` 创建，并优先使用 annotated tag。
- 与应用版本独立的大型模型资产使用 `models-vX.Y.Z` tag，并创建为非 Latest Release；客户端目录固定
  Release tag、文件大小和 SHA-256，发布后不得覆盖同名资产，更新模型必须递增模型版本。

示例：

```bash
git switch main
git pull --ff-only origin main
git tag -a v1.1.6 -m "PlaScan v1.1.6"
git push origin main
git push origin v1.1.6
```

## Release 文档

每次打 tag 前同步维护：

- `CHANGELOG.md`
- `docs/releases/vX.Y.Z*.md`

版本文档至少包含：

- 新增
- 优化
- 修复
- 验证
- 已知问题

验证项必须写具体命令和结果，不能只写“测试通过”。

## GitHub Release

GitHub Release 说明从对应 `docs/releases/vX.Y.Z*.md` 摘要生成，并保留：

- 版本号和日期
- 主要变更
- 验证命令
- 已知问题
- 下载或打包说明

## Windows 一键打包

Windows CUDA 发布构建树必须先由标准脚本初始化；全新环境或依赖缺失时运行：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps -Jobs 8
. scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch
```

代码迭代期间使用增量 smoke 安装树：

```powershell
cmake --workflow --preset windows-package-smoke
```

结果位于 `build/windows-vcpkg-cuda-release/package-smoke/PlaScan`。该流程会增量构建 GUI、安装 Runtime
组件并执行 ONNX 校验，但不压缩安装器。它不会删除 staging 中已经失效的文件；依赖删除或安装布局
变化后，应先清理这个精确目录再重新运行。smoke 结果只用于安装后功能验证，不能代替正式发布制品。
vcpkg、CUDA、cuDNN、Qt 或工具链发生变化时，必须先重跑 `build_windows_cuda.ps1` 完成运行时同步；
workflow 只负责源码增量构建与安装，不替代依赖准备脚本。

使用 CMake/CPack 3.27+ 和 Inno Setup 6 生成正式分卷安装器：

```powershell
cmake --workflow --preset windows-package-release
```

结果位于 `build/windows-vcpkg-cuda-release/packages/release`。该流程与 smoke 共用同一个增量 CUDA
构建目录，但会额外执行完整 Inno Setup 压缩及分卷门禁。

## Linux 一键打包

Linux 发布基线为 Ubuntu 24.04 x86_64、CMake 3.25+、Ninja 和 vcpkg。通用 CPU/OpenCL 包的增量
rootfs 与正式 DEB 分别运行：

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --workflow --preset linux-package-smoke
cmake --workflow --preset linux-package-deb
```

rootfs 位于 `build/linux-vcpkg-package-release/package-smoke/rootfs`，DEB 与 SHA-256 位于
`build/linux-vcpkg-package-release/packages/release`。正式包必须通过 staging 内动态库闭包、Qt
XCB/offscreen 插件、GDAL/PROJ 数据、可迁移 RUNPATH、模型哈希和 DEB 内容门禁；不得通过关闭
`PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME` 绕过失败。

当前影像匹配生产后端需要 CUDA/TensorRT。需要安装后直接运行 LightGlue 时，构建机先准备 CUDA 13.1、
TensorRT 10.15+、ONNX parser 开发库和 NVIDIA Ubuntu 软件源，再运行：

```bash
cmake --workflow --preset linux-package-cuda-smoke
cmake --workflow --preset linux-package-cuda-deb
```

CUDA 制品位于 `build/linux-vcpkg-cuda-package-release/packages/release`，包名为 `plascan-cuda`。
该包声明 CUDA/TensorRT 精确主版本依赖，不捆绑 NVIDIA 驱动；目标机必须有兼容驱动并启用 NVIDIA
Ubuntu 仓库。`plascan` 与 `plascan-cuda` 使用相同安装路径并声明互斥，发布时必须明确标注适用环境。

Linux 发布额外检查：

- `dpkg-deb --field/--contents` 确认包名、版本、`amd64`、依赖、桌面入口及两份 ONNX，且不含 headers、
  静态库或 CMake exports；
- 在没有源码、vcpkg 和构建目录的环境通过 `/usr/bin/plascan` 启动 GUI，并确认所有 ELF/Qt plugins
  的 `ldd` 无 `not found`、RUNPATH 不含构建机绝对路径；
- 使用包内 U2Net 完成一次 CPU 掩模；CUDA 包还要从包内 LightGlue ONNX 首次生成 engine、完成匹配并
  验证缓存复用；安装树保持只读且不包含预生成 `.engine`；
- 安装和卸载后确认桌面数据库与图标缓存可正常刷新。

## CPack 模型门禁

对外发布的可掩模、可匹配安装包必须保持 `PLASCAN_BUNDLE_ONNX_MODELS=ON`，并在打包前准备
`models-v1.1.0` 中的 U2Net 与 LightGlue ONNX。`cmake --install`/CPack 的大小和 SHA-256 校验必须
通过；模型缺失或校验失败时不得关闭门禁后继续发布。

发布验证至少包括：

- 解包 ZIP/TGZ/DEB；对 INNOSETUP 执行静默安装或检查 CPack staging，确认两份 ONNX 位于约定路径且哈希与
  `docs/models/models-v1.1.0.sha256` 一致，并包含两份模型 notice 和 `Apache-2.0.txt`；
- 确认安装包不含任何本机生成的 `.engine`；
- Windows INNOSETUP 的 `.exe` 与每个 `-N.bin` 分卷都必须小于 2 GiB，并与
  `-INNOSETUP.sha256` 一起显式上传；发布前逐项核对清单，避免漏传分卷，也不要通配整个包目录
  而误传 ZIP 或旧制品；
- 安装器如需 Authenticode 签名，应在 Inno Setup/ISCC 编译阶段完成；若在 CPack 完成后重新签名
  `.exe`，必须重新生成并复核 `-INNOSETUP.sha256`；
- 在干净的 CUDA/TensorRT 环境从内置 LightGlue ONNX 完成首次 engine 构建和一对影像匹配，随后验证
  缓存复用，且安装树没有新增文件；
- 使用内置 U2Net 至少完成一次 CPU 蒙版推理；Windows CUDA 发布包还需执行现有 CUDA 部署测试。
