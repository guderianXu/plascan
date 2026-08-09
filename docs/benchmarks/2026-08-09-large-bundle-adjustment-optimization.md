# Agisoft 100 相机大规模光束法平差优化（2026-08-09）

## 输入与范围

- 输入清单：`testData/photogrammetry_benchmarks/agisoft_aerial_gcps_100/prepared/plascan/image_camera.lis`
- 清单 SHA-256：`3E7EFF1DDA401F792839DF4F0CFD38F364D83E877DE23A3C8D214A8FC010C402`
- 影像/相机：100/100，影像尺寸 1600 x 1067
- BA 重放：88,104 tracks、261,589 个真实像点观测
- 平台：Windows/MSVC Release、Ryzen 9 9950X3D、32 线程、Ceres 2.2.0
- 参数：联合优化相机位姿，固定 0 号相机，最多 20 次迭代，Huber 与点过滤参数保持一致

数据目录中的 `GCPs_WGS84.txt` 有 17 个物方坐标，但没有逐影像像点量测，工程也没有
`marker_set.json`。因此本次不能构造 GCP BA 约束或 GCP 残差门禁；A/B 只比较同一真实连接点网络的
速度、重投影质量和输出结构。

## 实现优化

- Ceres 可用路径不再重复扫描全量 BA 问题。
- 逐轨直接筛选原始观测，移除有效观测深拷贝、逐轨 `std::set` 及对应堆分配。
- 每台相机只生成一份稳定 `ProjectionCamera` 快照，残差只保存指针，不再为每条观测复制相机模型。
- 普通影像重投影残差共享一个无状态 Huber loss，消除逐观测小对象分配。
- 迭代回调不读取参数块，关闭 `update_state_every_iteration` 的全状态复制。
- `ba_backend_benchmark` 新增真实 `sfm_sparse_points.json` + TSAI 重放、重复次数和 Schur 阈值参数；
  真实模式禁止后端回退和 legacy 对照求解。

## 固定问题 A/B

同一进程加载一次数据，每轮重新复制相机和 tracks；下表为 5 次运行的中位数。API 墙钟是跨后端可比
口径，内部 setup/solve/total 用于分析同一 Ceres 后端。

| 指标 | 优化前 | 优化后 | 改善 |
|---|---:|---:|---:|
| Ceres setup | 0.260508 s | 0.155176 s | -40.4% |
| Ceres solve | 1.999787 s | 1.824404 s | -8.8% |
| Ceres internal total | 2.252324 s | 1.971683 s | -12.5% |
| 公共 API 墙钟 | 2.484245 s | 2.135195 s | -14.0% |
| RMS before | 0.5743401084 px | 0.5743401084 px | 一致 |
| RMS after | 0.1968729905 px | 0.1968729905 px | 一致 |
| 实际求解器 | dense_schur_cpu | dense_schur_cpu | 一致 |
| Ceres 迭代 | 9 | 9 | 一致 |

复现命令：

```powershell
build/windows-vcpkg-cuda-release/bin/ba_backend_benchmark.exe `
  --dataset-json `
  build/benchmark_runs/aerial_ba_20260809_agisoft_gcps_100_baseline/headless.files/1/reconstruction/sparse/sfm_sparse/sfm_sparse_points.json `
  --camera-list testData/photogrammetry_benchmarks/agisoft_aerial_gcps_100/prepared/plascan/image_camera.lis `
  --backend ceres_cpu --iterations 20 --threads 32 --repetitions 5 --refine-pose
```

## 求解器规划对照

优化前对相同输入各运行 3 次；中位数如下。100 个可变相机时 Dense Schur 仍是合理默认值，当前数据
不支持下调 `maxDenseSchurCameras=200`。

| 求解器 | internal total | API 墙钟 | RMS after |
|---|---:|---:|---:|
| Dense Schur | 2.252 s | 2.484 s | 0.1968729905 px |
| Sparse Schur | 2.286 s | 2.527 s | 0.1968729905 px |
| Iterative Schur | 11.667 s | 11.940 s | 0.1968728009 px |

强制 Sparse 使用 `--max-dense-schur-cameras 1 --max-sparse-schur-cameras 2000`；强制 Iterative
再把 `--max-sparse-schur-cameras` 设为 1。

## 正式空三调用链复核

复用 1,035 个匹配对和几何验证缓存，完整重建阶段两次都注册 100/100 相机并生成 88,104 个点：

| 指标 | 优化前 | 优化后 | 改善 |
|---|---:|---:|---:|
| 全局 BA internal total | 2.007718 s | 1.589780 s | -20.8% |
| SfM duration | 14.090 s | 12.345 s | -12.4% |
| BA observations | 261,583 | 261,583 | 一致 |
| BA RMS | 0.366257 -> 0.183408 px | 0.366257 -> 0.183408 px | 一致 |
| 稀疏平均重投影 | 0.2643375351 px | 0.2643375351 px | 一致 |

`aerial_triangulation_cli` 在两次运行完成稀疏结果和报告后，均在最终 Chunk 资源封装阶段遇到既有的
`提交工程资源失败`。该失败发生在 BA/SfM 计时之后，A/B 的失败阶段和输出结构相同；本次没有把独立的
项目写回问题混入 BA 性能改动。
