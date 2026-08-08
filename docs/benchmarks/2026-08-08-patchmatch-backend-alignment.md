# PatchMatch CUDA/OpenCL 语义对齐验收（2026-08-08）

## 目标

验证 MVS 算法修订 25 的 CUDA/OpenCL 对齐不会牺牲 Hyb2 质量，并确认显式 OpenCL 在 AMD 核显上
全程单进程、单设备执行槽运行。建模阶段严格使用 `interpolation=disabled`。

## 对齐内容

- CUDA 与 OpenCL 使用相同的确定性逆深度粗搜和 13 点局部细化。
- `numIterations` 统一映射为 32–96 个深度样本和 2–4 次棋盘传播。
- 传播统一包含邻域平面、邻域法向、随机法向、随机深度及组合候选。
- CPU/CUDA/OpenCL 在没有显式 hint 半径时统一使用深度的 5%，并对任意来源数执行光度唯一性判定。
- CUDA/OpenCL 掩膜平面测试要求有效覆盖率差不超过 2%，交集深度相对误差中位数小于 2%。

曾验证 30% 默认 hint 搜索半径会使 AMD OpenCL 的过滤后融合帧从 13/14 降至 11/14；恢复 5% 后
重新达到 13/14。因此生产配置保留窄搜索窗，不通过放宽融合门限掩盖歧义深度。

## Hyb2 深度结果

两次正式重放都使用 `highest`、4 个来源、7 个 CPU 像素线程、1 个 GPU 帧 worker、0 个 CPU 帧
worker。OpenCL 显式选择设备 1（AMD `gfx1036`），CUDA 显式选择 CUDA 后端。

| 后端 | 过滤后可融合帧 | 平均置信度 | 平均掩膜内有效率 | 平均覆盖率 | 批次耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD OpenCL | 13/14 | 0.850322 | 0.961470 | 0.127206 | 269.50 s |
| CUDA | 14/14 | 0.852062 | 0.964361 | 0.127584 | 18.70 s |

OpenCL 日志报告 `cuda_devices=0 opencl_devices=1 physical_gpu_workers=1`，所有主要 kernel 链 duty
接近 100%，没有 CPU/CUDA 回退。结果目录：

- `E:/code/test/hyb2/validation_revision25_aligned_v3_opencl_amd_depth_20260808`
- `E:/code/test/hyb2/validation_revision25_aligned_cuda_depth_20260808`

## 禁用插值模型结果

| 后端深度 | 顶点 | 面 | 组件 | 边界边 | 非流形边 | 最终完整性中位/P10/最低 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AMD OpenCL | 69,522 | 139,040 | 1 | 0 | 0 | 0.9073 / 0.8877 / 0.8831 |
| CUDA | 69,314 | 138,624 | 1 | 0 | 0 | 0.9081 / 0.8801 / 0.8778 |

两模型都 watertight、绕序一致且 Euler 数为 2。以 100,000 个双向表面采样比较，Chamfer-L1 为
0.00016529（包围盒对角线的 0.111%），P95 为 0.00050472（0.338%），采样 Hausdorff 为
0.00314316（2.105%）。六视角图未见大面积空洞或轮廓分叉：

- `E:/code/test/hyb2/validation_revision25_aligned_opencl_vs_cuda_no_interp_20260808.png`
- `E:/code/test/hyb2/validation_revision25_aligned_opencl_vs_cuda_geometry_20260808.json`

## 验证命令

```powershell
cmake --build build/windows-vcpkg-cuda-release --target `
  mvs_depth_reprocess_cli test_patchmatch_mask_aware test_patchmatch_cpu --parallel 8

build/windows-vcpkg-cuda-release/src/core/mvs/test_patchmatch_mask_aware.exe
build/windows-vcpkg-cuda-release/tests/test_patchmatch_cpu.exe
```

完整 MSVC 构建通过；`ctest --output-on-failure` 在 200.45 秒内完成，1913 个已启用测试全部通过，
其中 10 个因环境或显式条件跳过，另有 1 个 CUDA benchmark 保持 disabled。
