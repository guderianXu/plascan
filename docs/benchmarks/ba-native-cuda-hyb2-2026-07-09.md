# BA native CUDA hyb2 验证记录（2026-07-09）

## 输入

- 项目：`E:/code/test/hyb2/hyb2.plascan`
- BA 输入规模：`cameras=211`，`tracks=25499`，`observations=554132`
- 命令公共参数：`--threads 32 --max-iterations 1 --max-point-iterations 1 --max-camera-iterations 1 --force`
- 构建产物：`E:/code/plascan/build/windows-vcpkg-cuda-release/bin/bundle_adjust_cli.exe`
- 当前 native CUDA 能力边界：固定相机投影，GPU 上优化三维点块；相机 Schur/PCG 更新尚未作为已完成能力发布。

## 结果摘要

| 后端请求 | 实际后端 | GPU | 回退 | 有效 track 比例 | 优化 track | RMS before | RMS after | setup(s) | solve(s) | total(s) | native 观测 | native 接受步 | 结论 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `legacy_cpu` | `legacy_cpu` | 否 | 否 | 0.7742 | 19741 | 11.806893 | 0.882593 | 0.000 | 0.055532 | 0.055532 | 0 | 0 | CPU legacy 仍是最快基线 |
| `native_cuda` | `native_cuda` | 是 | 否 | 0.7954 | 20282 | 11.806893 | 0.688198 | 0.046887 | 0.091462 | 0.138349 | 554132 | 25489 | 质量优于 legacy，但单轮总耗时更高 |
| `auto` | `native_cuda` | 是 | 否 | 0.7954 | 20282 | 11.806893 | 0.688198 | 0.038898 | 0.145856 | 0.184755 | 554132 | 25489 | 通过质量门控后选择 native CUDA |

`auto` 的 `total(s)` 包含 native CUDA 候选和 legacy 对照质量门控成本，因此比显式 `native_cuda` 更高。

## 关键判断

- 当前 hyb2 数据上，native CUDA 首期点块求解可以启用 GPU，并且 1 轮后 RMS 和有效 track 比例优于 legacy。
- 当前速度还没有超过 legacy：native CUDA 的 GPU solve 约 `0.091 s`，legacy 总耗时约 `0.056 s`。
- native CUDA 报告 `ba_native_cuda_active_observations=554132`、`ba_native_cuda_accepted_steps=25489`，说明完整 BA 观测工作集进入了 CUDA 点块求解。
- `ba_native_cuda_pcg_iterations` 当前为 0，符合首期边界；后续相机 Schur/PCG 更新落地后再作为完整 CUDA 全局 BA 加速结论。

## 复现命令

```powershell
$env:CUDNN_ROOT_DIR='D:\NVIDIA\CUDNN\v9.24.0.43_cuda13'
$env:QT_QPA_PLATFORM='offscreen'
$env:PATH='E:\code\plascan\build\windows-vcpkg-cuda-release\vcpkg_installed\x64-windows\bin;E:\code\plascan\build\windows-vcpkg-cuda-release\vcpkg_installed\x64-windows\tools\Qt6\bin;D:\NVIDIA\CUDNN\v9.24.0.43_cuda13\bin;E:\code\plascan\build\windows-vcpkg-cuda-release\bin;E:\code\plascan\build\windows-vcpkg-cuda-release\tests;E:\code\plascan\build\env\libtorch-cu130\libtorch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;' + $env:PATH

E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe `
  E:\code\test\hyb2\hyb2.plascan `
  --output-dir E:\code\plascan\build\ba-bench\hyb2-native-cuda-1iter `
  --ba-backend native_cuda `
  --threads 32 `
  --max-iterations 1 `
  --max-point-iterations 1 `
  --max-camera-iterations 1 `
  --force

E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe `
  E:\code\test\hyb2\hyb2.plascan `
  --output-dir E:\code\plascan\build\ba-bench\hyb2-native-auto-1iter `
  --ba-backend auto `
  --threads 32 `
  --max-iterations 1 `
  --max-point-iterations 1 `
  --max-camera-iterations 1 `
  --force
```
