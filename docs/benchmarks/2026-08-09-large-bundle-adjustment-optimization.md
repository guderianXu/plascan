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
- 自适应相机模型不再依赖“对地/环拍”硬分类，而是利用逐轨点 Schur 消元后的归一化内参相关性、
  典型扰动响应及几何/像面覆盖证据，为 9 个共享内参形成 PlaScan 启发式可靠性评分，并冻结未达阈值的参数。
- Ceres 共享内参支持逐参数掩码；冻结参数保持每台相机的原值，低阶模型不再进入无效的高阶
  自标定阶段。
- `ba_backend_benchmark` 新增真实 `sfm_sparse_points.json` + TSAI 重放、重复次数和 Schur 阈值参数；
  真实模式禁止后端回退和 legacy 对照求解，并支持 `fixed|full|adaptive` 相机模型 A/B。

## 固定问题 A/B

同一进程加载一次数据，每轮重新复制相机和 tracks；下表为 5 次运行的中位数。API 墙钟是跨后端可比
口径，内部 setup/solve/total 用于分析同一 Ceres 后端。

重放 JSON 是正式空三的生成物，不纳入 Git（137,819,165 bytes；SHA-256
`3D569CE402B1C9FECD01BF629186A9BE723A2AF0ACE6E9B6E26DD7E5404303B1`）。产物不存在时先从仓库根目录运行：

```powershell
build/windows-vcpkg-cuda-release/bin/aerial_triangulation_cli.exe `
  --input testData/photogrammetry_benchmarks/agisoft_aerial_gcps_100/prepared/plascan/image_camera.lis `
  --output-dir build/benchmark_runs/aerial_ba_20260809_agisoft_gcps_100_baseline `
  --quality high --device cuda --reference-mode source-code --algorithm-id sift_lightglue `
  --keypoint-limit 40000 --tiepoint-limit 4000 --reference-preselection --no-reset-alignment `
  --adaptive-camera-model-fitting --threads 32 --force
```

该命令会生成下述 `headless.files/.../sfm_sparse_points.json`；模型、驱动或匹配缓存版本变化时应重新记录
产物哈希，不应把不同连接点网络的计时混入同一 A/B。

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

## 自适应相机模型 A/B

同一最终 Release 二进制、同一空闲机器状态下，完整 Brown-Conrady 9 参数模型与自适应模型各运行
5 次。下表使用第 2–5 次热运行的中位数；“策略 + BA API”包含模型评估与掩码应用，不含数据加载。

| 指标 | 完整 9 参数模型 | 自适应模型 | 改善/差异 |
|---|---:|---:|---:|
| 实际模型 | `f+aspect+cx+cy+k1+k2+k3+p1+p2` | `f+k1` | 本次策略冻结 7 个未达可靠性阈值的参数 |
| 参与 BA 相机 | 100 | 100 | 一致；自适应评估确认 100/100 活动 |
| 模型策略开销 | 0 s | 0.059074 s | 约占“策略 + BA API” 0.83% |
| Ceres solve | 10.604717 s | 6.689746 s | -36.9% |
| 公共 API 墙钟 | 10.982075 s | 7.074230 s | -35.6% |
| 模型策略 + BA API | 10.982075 s | 7.133304 s | **-35.0%** |
| RMS before | 0.5743401084 px | 0.5743401084 px | 一致 |
| RMS after | 0.1961197223 px | 0.1967353954 px | +0.0006156731 px（+0.314%） |
| 20 轮终止状态 | 5/5 `no_convergence` | 5/5 `success` | 自适应模型稳定收敛 |

该航测块的活动相机光轴近似平行，策略保留焦距与一阶径向畸变，降低主点、高阶径向和切向畸变
与航高、姿态及场景结构互相补偿的风险。速度收益主要来自共享内参自由度由 9 降至 2，以及不再执行
不必要的高阶自标定阶段；终止状态改善表明该数据上的数值行为更稳定，但本基准未计算法方程条件数。
策略本身只需约 0.06 秒。隔离后端基准为避免回退干扰计时而关闭后端质量门；两组均报告
`solutionUsable=true`、`camera_model_applied=true`，88,104 个轨迹全部有效。

在生产链路语义修复后的最终 Release 二进制上再次运行 5+5 次，模型选择、RMS 和终止状态完全一致；
该批次热运行中位数为完整模型 12.491 s、自适应模型 7.360 s（策略 + BA API，-41.1%）。由于同机
并行验证任务造成计时波动，主表保留较早空闲批次的保守 -35.0% 结果，不用更有利的复核批次替换。

复现命令仅需在上一节命令末尾分别追加：

```powershell
--camera-model full
--camera-model adaptive
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
