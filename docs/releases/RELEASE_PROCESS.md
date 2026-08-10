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
git tag -a v1.1.7 -m "PlaScan v1.1.7"
git push origin main
git push origin v1.1.7
```

## Release 文档

每次打 tag 前同步维护：

- `CHANGELOG.md`
- `docs/releases/vX.Y.Z*.md`

模型 tag `models-vX.Y.Z` 还必须同步维护：

- `docs/releases/models-vX.Y.Z.md`
- `docs/models/models-vX.Y.Z.sha256`
- `docs/models/README.md` 中的来源版本、精确字节数、SHA-256、许可和部署契约
- 客户端 `ModelAssetCatalog` 与 CPack 安装门禁中的同一组不可变资产元数据

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

## 模型 Release

模型 Release 与应用版本独立，必须设为非 Latest，并遵循以下边界：

- 不得覆盖或重新上传已发布的同名资产；任何字节变化都必须递增 `models-vX.Y.Z`；
- 每个 Release 文档列出它实际包含的完整资产白名单，不得让客户端假定新 Release 重发全部历史模型；
- `models-v1.1.0` 继续唯一承载 U2Net、LightGlue 和 LoMa-R；`models-v1.2.0` 只承载
  `BiRefNet_dynamic_1024.onnx` 与 `BiRefNet_dynamic_1024.provenance.json`；
- 上传前后分别核对文件字节数和 SHA-256，并确认 GitHub Release 资产列表没有遗漏、旧文件或 `.engine`；
- ONNX 与 provenance/manifest 必须来自同一次已验证导出，禁止手工改写 sidecar 或为了匹配目录而重命名。

`models-v1.2.0` 当前资产契约：

| 资产 | 字节数 | SHA-256 |
|------|-------:|---------|
| `BiRefNet_dynamic_1024.onnx` | 972558911 | `3af7fe29f80be80e12595671293c877af6767cae71566a8765face68965f0742` |
| `BiRefNet_dynamic_1024.provenance.json` | 1688 | `9e100509b59aedfeabd0aabc7277009b0d620803b27f482abb2e28220de8d4ff` |

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
vcpkg、CUDA、TensorRT、Qt 或工具链发生变化时，必须先重跑 `build_windows_cuda.ps1` 完成运行时同步；
workflow 只负责源码增量构建与安装，不替代依赖准备脚本。

使用 CMake/CPack 3.27+ 和 Inno Setup 6 生成正式分卷安装器：

```powershell
cmake --workflow --preset windows-package-release
```

结果位于 `build/windows-vcpkg-cuda-release/packages/release`。该流程与 smoke 共用同一个增量 CUDA
构建目录，但会额外执行完整 Inno Setup 压缩及分卷门禁。Inno 默认使用
`PLASCAN_INNO_LZMA_BLOCK_THREADS=4` 对 LZMA2 大文件分块并行压缩；发布机内存不足时可降为 `2`，不得
通过关闭分卷、模型哈希或 `.engine` 禁入门禁换取速度。

## Linux 一键打包

Linux 发布基线为 Ubuntu 24.04 x86_64、CMake 3.25+、GCC/G++、`gfortran`、Ninja、`pkg-config`、
`patchelf` 和 vcpkg。先准备宿主构建依赖，再运行通用 CPU/OpenCL 包的增量 rootfs 与正式 DEB：

```bash
sudo apt install build-essential gfortran ninja-build pkg-config patchelf
```

打包命令为：

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --workflow --preset linux-package-smoke
cmake --workflow --preset linux-package-deb
```

rootfs 位于 `build/linux-vcpkg-package-release/package-smoke/rootfs`，DEB 与 SHA-256 位于
`build/linux-vcpkg-package-release/packages/release`。正式包必须通过 staging 内动态库闭包、Qt
XCB/offscreen 插件、GDAL/PROJ 数据、可迁移 RUNPATH、模型哈希和 DEB 内容门禁；不得通过关闭
`PLASCAN_VERIFY_LINUX_PACKAGE_RUNTIME` 绕过失败。

当前影像匹配、U2Net GPU 后端和 BiRefNet Dynamic 需要 CUDA/TensorRT。构建机先准备 CUDA 13.1、
TensorRT 10.15.1
CUDA 13.1 变体、ONNX parser 开发库、完整 Builder/runtime 和 NVIDIA Ubuntu 软件源，再运行：

```bash
cmake --workflow --preset linux-package-cuda-smoke
cmake --workflow --preset linux-package-cuda-deb
```

CUDA 制品位于 `build/linux-vcpkg-cuda-package-release/packages/release`，包名为 `plascan-cuda`。
该包声明 CUDA/TensorRT 精确主版本依赖，不捆绑 NVIDIA 驱动；目标机必须有兼容驱动并启用 NVIDIA
Ubuntu 仓库。`plascan` 与 `plascan-cuda` 使用相同安装路径并声明互斥，发布时必须明确标注适用环境。

Linux 发布额外检查：

- `dpkg-deb --field/--contents` 确认包名、版本、`amd64`、依赖和桌面入口；通用包包含 U2Net/LightGlue
  两份 ONNX，CUDA 包另含 BiRefNet ONNX/provenance，且均不含 headers、静态库或 CMake exports；
- 在没有源码、vcpkg 和构建目录的环境通过 `/usr/bin/plascan` 启动 GUI，并确认所有 ELF/Qt plugins
  的 `ldd` 无 `not found`、RUNPATH 不含构建机绝对路径；
- 使用包内 U2Net 完成一次 CPU 掩模；CUDA 包还要从包内 LightGlue ONNX 首次生成 engine、完成匹配并
  验证缓存复用，并从包内 BiRefNet ONNX 完成首次蒙版推理和第二进程缓存复用；安装树保持只读且不包含
  预生成 `.engine`；
- 安装和卸载后确认桌面数据库与图标缓存可正常刷新。

## CPack 模型门禁

对外发布的可掩模、可匹配安装包必须保持 `PLASCAN_BUNDLE_ONNX_MODELS=ON`，并在打包前准备
`models-v1.1.0` 中的 U2Net 与 LightGlue ONNX。Windows CUDA 安装包还必须保持
`PLASCAN_BUNDLE_LOMA_R_MODELS=ON`，准备 LoMa-R 两个共享 ONNX 和三个 K 档 manifest。
Windows/Linux CUDA 安装包必须保持 `PLASCAN_BUNDLE_BIREFNET_DYNAMIC=ON`，并从 `models-v1.2.0`
准备 BiRefNet ONNX 和 provenance；不得在 v1.2.0 中寻找或复制 v1.1.0 的原有模型。
`cmake --install`/CPack 的大小和 SHA-256 校验必须通过；模型缺失或校验失败时不得关闭门禁后继续发布。

发布验证至少包括：

- 解包 ZIP/TGZ/DEB；对 INNOSETUP 执行静默安装或检查 CPack staging。通用包确认 U2Net/LightGlue；
  CUDA 包另确认 BiRefNet ONNX/provenance；Windows CUDA 包还确认 LoMa-R 五文件包。原有七项资产的
  哈希必须与 `docs/models/models-v1.1.0.sha256` 一致，BiRefNet 两项必须与
  `docs/models/models-v1.2.0.sha256` 一致，并包含相应 notice、Apache-2.0 和 BiRefNet MIT 许可；
- 确认安装包不含任何本机生成的 `.engine`；
- Windows INNOSETUP 的 `.exe` 与每个 `-N.bin` 分卷都必须小于 2 GiB，并与
  `-INNOSETUP.sha256` 一起显式上传；发布前逐项核对清单，避免漏传分卷，也不要通配整个包目录
  而误传 ZIP 或旧制品；
- 安装器如需 Authenticode 签名，应在 Inno Setup/ISCC 编译阶段完成；若在 CPack 完成后重新签名
  `.exe`，必须重新生成并复核 `-INNOSETUP.sha256`；
- 在干净的 CUDA/TensorRT 环境分别从内置 LightGlue 和 LoMa-R ONNX 完成首次 engine 构建和一对影像
  匹配，随后验证缓存复用，且所有 engine 只写入用户模型缓存，安装树没有新增文件；
- 使用内置 U2Net 至少完成一次 OpenCV CPU 蒙版推理；Windows CUDA 发布包还必须执行
  `build_windows_cuda.ps1 -Target test_mask_generation -RunU2NetTensorRtDeploymentTest`，确认 OpenCV ABI
  不含 `cuda/cudnn/dnn-cuda`、安装树没有 cuDNN 或预生成 engine，并在无外部 SDK PATH 的全新缓存中
  从 ONNX 完成 TensorRT engine 首次构建和真实推理。
- Windows CUDA 发布包还必须执行
  `build_windows_cuda.ps1 -Target test_mask_generation -RunBiRefNetTensorRtDeploymentTest`；第一次进程必须
  从内置 ONNX 新建 engine，第二次进程必须报告缓存复用，两个进程都只能写隔离的用户缓存。
  `models-v1.2.0` 基线已在 RTX 4060 Laptop 8 GiB / TensorRT 10.15 / FP16 通过：首次构建加推理
  `2631483 ms`（43 分 51 秒），第二进程复用 `33573 ms`，engine `540031644` bytes；后续 Release 仍须
  在其实际 staging 上重跑，不能沿用历史结果代替当前制品验证。
