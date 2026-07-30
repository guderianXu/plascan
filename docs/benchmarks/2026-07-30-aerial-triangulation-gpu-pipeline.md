# 空三 GPU 流水线基准（2026-07-30）

## 环境

- CPU：AMD Ryzen 9 9950X3D，16 核 32 线程
- GPU：NVIDIA GeForce RTX 5080，16303 MiB
- NVIDIA 驱动：591.86
- 构建：Windows Release、CUDA 13.1、LibTorch CUDA、32 路并行编译

## 几何阶段微基准

构建：

```powershell
cmake --build build/windows-vcpkg-cuda-release `
  --target aerial_geometry_benchmark --parallel 32
```

运行：

```powershell
build/windows-vcpkg-cuda-release/bin/aerial_geometry_benchmark.exe `
  8000 128 120 2000 40000 16
```

参数依次为每影像关键点数、描述子维度、影像对数、每对几何匹配数、三角化点数和 PnP 影像数。

| 阶段 | 结果 |
|------|------|
| 完整特征读取 | 4100.49 ms |
| 仅关键点几何读取 | 200.30 ms，20.47 倍加速 |
| 120 对 USAC-MAGSAC | 179.00 ms |
| 16 次 PnP RANSAC | 20.82 ms |
| 4 万点串行三角化 | 4.33 ms |
| 4 万点线程批量三角化 | 5.21 ms，慢于串行 |

结论：几何阶段此前的主要瓶颈是为 F/E/H 验证反复读取 Nx128 描述子，而不是计算本身。
默认流程已经改为只读关键点，并对影像对做最多 8 路 CPU 并发。USAC、PnP 和三角化的规模不足以
抵消 CUDA 启动、数据打包和传输开销，因此不接入默认 GPU 路径。

## dino 全链路回归

连接点：

```powershell
match_photos_cli.exe `
  --input E:/code/test/dino/dino_images_for_ba_validation.lis `
  --output-dir build/aerial-validation/dino_gpu_pipeline_20260730_2055 `
  --quality cuda --device cuda `
  --feature-algorithm sift --match-algorithm lightglue `
  --keypoint-limit 40000 --tiepoint-limit 4000 `
  --pair-mode exhaustive --exclude-fixed-tie-points
```

结果：

- 16 张影像全部重新提取 CUDA SIFT。
- 120/120 个影像对完成 LightGlue CUDA 匹配，累计原始匹配 12879 个。
- 16 GB 显存自动解析为 3 个独立 LightGlue worker，没有发生 OOM。
- 几何验证使用 8 个 CPU worker，38 对通过，形成 2599 条多视连接点轨迹。

空三复用上述连接点，不重新匹配：

```powershell
aerial_triangulation_cli.exe `
  --input E:/code/test/dino/dino_images_for_ba_validation.lis `
  --output-dir build/aerial-validation/dino_gpu_sfm_20260730_2058 `
  --assets-dir build/aerial-validation/dino_gpu_pipeline_20260730_2055/headless.files/1/assets `
  --quality high --device cuda --threads 32 `
  --reference-preselection --reference-mode sequence `
  --generic-preselection --reset-alignment `
  --adaptive-camera-model-fitting --exclude-fixed-tie-points `
  --no-auto-generate-missing-matches
```

结果：

- 注册影像：16/16
- 稀疏点：2794
- 平均重投影误差：0.521099 px
- 正式 BA：7856 个观测，`ceres_cpu`，约 0.070 s
- MVS 质量门控：`ok`
- 空三总耗时：34.772 s，其中包含无先验广域焦距候选搜索

正式 BA 在该规模下选择 CPU 是预期行为。GPU 只加速 Ceres 的线性代数部分，小问题的模型建立、
主机到设备传输和 CUDA 初始化会抵消求解收益；Auto 只有在相机数和观测数达到门槛时才选择
`ceres_cuda`。

## 验收结论

- GPU 资源用于计算密集且可批处理的 CUDA SIFT 和 LightGlue。
- CPU 与 GPU 通过有界队列和独立 worker 重叠，不共享非线程安全的 TorchScript matcher。
- 缓存复用、蒙版、固定连接点过滤、候选对顺序和输出顺序保持原语义。
- 几何验证、PnP、三角化和小型 BA 不以 GPU 利用率为目标强制迁移；默认选择以端到端耗时和重建质量为准。

## 构建与测试

- `match_photos_cli`、`aerial_triangulation_cli`、`bundle_adjust_cli`、
  `aerial_geometry_benchmark` 和相关核心库均以 `--parallel 32` 构建成功。
- 连接点任务 25 项、并发策略 8 项、特征准备队列 2 项、空三工作流 21 项、
  SfM 40 项和 BA/CUDA 38 项测试通过。
- GUI 对象和依赖库已编译；最后一次 `plascan.exe` 链接因正在运行的 GUI 占用目标文件而返回
  `LNK1104`，关闭当前 GUI 后重新执行增量链接即可。
- 全仓历史契约测试仍有与本次 GPU 优化无关的失败：ProjectSession 迁移后的 CLI 报告路径
  已改为 chunk 目录，但旧测试仍检查输出根目录；另有 IO/GUI 源码字符串契约未同步到当前重构。
