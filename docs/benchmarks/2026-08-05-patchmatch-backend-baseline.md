# PatchMatch 跨后端改造基线（2026-08-05）

本文记录跨厂商计算后端改造前的可重复 CUDA 质量与性能基线。基准使用
`tests/test_patchmatch_cpu.cpp` 中的合成正视平面，理论深度为 10，避免依赖本机外部测试数据。

## 环境

- 系统：Windows 11 专业版 10.0.26200（build 26200）
- CPU：Intel Core i7-13700H，14 核 20 线程
- 内存：63.63 GiB
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB
- NVIDIA 驱动：610.62
- CUDA：13.1（compute capability 8.9）
- CMake：3.31.6-msvc6
- 编译器：MSVC 19.44 / CUDA 13.1 nvcc
- 基线提交：`58ae29d`

## 构建与回归

```powershell
& scripts\build_win\build_windows_cuda.ps1 `
  -BuildOnly -Target test_patchmatch_cpu -Jobs 8 `
  -EnableOpenCvDnnCuda 0 -EnableCeresCudaBa 0

build\windows-vcpkg-cuda-release\tests\test_patchmatch_cpu.exe `
  --gtest_brief=1
```

结果：5/5 通过。构建显式关闭与本目标无关、且本机开发包未安装的 OpenCV DNN CUDA 和
Ceres CUDA feature；MVS PatchMatch 仍使用 CUDA 13.1 编译和执行。

## 性能基准

```powershell
build\windows-vcpkg-cuda-release\tests\test_patchmatch_cpu.exe `
  --gtest_also_run_disabled_tests `
  --gtest_filter=PatchMatchCudaBenchmarkTest.DISABLED_CompareParallelAndLegacySweepAfterWarmup `
  --gtest_repeat=7 --gtest_brief=1
```

输入为 640×480 灰度影像，焦距 520、基线 1、视差 52、搜索范围 5–15、4 次迭代，
理论深度为 10。统计区域为中部 426×320，共 136320 像素。

| 指标 | CUDA 棋盘格传播 | CUDA 传统四向 sweep |
|---|---:|---:|
| 7 次耗时中位数 | 5.028 ms | 137.146 ms |
| 有效像素 | 136320 / 136320 | 136320 / 136320 |
| 平均深度 | 9.925758 | 10.000049 |
| 深度 RMSE | 0.555886 | 0.012295 |
| 平均置信度 | 0.989937 | 0.992608 |

棋盘格路径相对传统 sweep 的中位数加速约为 27.3 倍。性能数值仅用于同一机器、同一构建配置下的
回归对比，不作为其它硬件的绝对性能承诺。

## 基线过程中发现的问题

GPU 灰度图缓存原先仅使用 `cv::Mat.data`、尺寸、步长和降采样因子作为键。测试进程重复释放和
分配影像时，OpenCV 可能复用同一主机地址，使不同影像错误命中旧 GPU 缓存，表现为第三次运行
开始深度 RMSE 从 0.555886 跳到 4.264899。

缓存条目现在持有输入 `cv::Mat` 的浅引用，在对应 GPU 缓存释放前保持底层 OpenCV 分配有效，
防止地址被另一张影像复用。修复后连续 7 次的有效像素、平均深度、置信度和 RMSE 完全一致。

## 后端验收基线

后续 CPU、CUDA、OpenCL 或 Vulkan 后端至少需要记录：

- 实际 backend、设备 ID/UUID、设备名称和驱动版本；
- 上传、计算、下载和总耗时；
- 有效像素、平均深度、深度 RMSE 和平均置信度；
- 峰值设备内存、缓存命中率、失败和回退原因。

跨厂商后端不要求逐位一致，但必须通过合成几何阈值，并在航测与环绕物体数据上验证最终点云、
网格和 DEM 指标没有不可解释的退化。
