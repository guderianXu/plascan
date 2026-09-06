# PlaMatrix 原生块稀疏直接求解对比（2026-09-04）

## 实现边界

本次没有接入 CHOLMOD、SuiteSparse、BLAS、LAPACK 或其它新增第三方数值库。`SparseCpu` 位于 PlaMatrix 内部，
复刻参考 BA 的“Schur 约化—稀疏排序—符号分析—数值 Cholesky—三角回代”结构，但数据结构和数值内核均为
PlaMatrix 自己实现：

- 以固定尺寸的相机/内参参数块构造无向邻接图，并执行确定性最小度消元；
- 在符号阶段生成填充块、CSR 输入值映射和 trailing-update 目标；
- 在数值阶段执行块 LLT、panel 右三角求解和尾部块更新；
- 在 workspace 中复用排列、填充图、映射、因子存储和 RHS 缓冲，同一拓扑只更新数值；
- 保留旧 `DenseCpu` 作为数值对照，CUDA/OpenCL 继续使用原有设备 Schur-PCG。

## 测试环境与口径

- Linux/GCC Release，CPU-only PlaMatrix 独立构建；基准使用 `OMP_NUM_THREADS=8`。
- 160 个 6 维 primary 块、8,000 个 3 维 eliminated 块，每点连接 3 台相机。
- DenseCpu 与原生 SparseCpu 使用完全相同的法方程、阻尼 `0.1` 和收敛容差。
- 各路径预热一次，之后交错执行 7 次；表中为稳态中位数。
- 解一致性按两条路径最终 primary 增量的最大绝对差检查，门限为 `1e-8`。

## 结果

| 路径 | 总 wall 中位数 | Schur 装配 | 数值分解 | 三角回代 | 相对 DenseCpu |
|---|---:|---:|---:|---:|---:|
| DenseCpu | 20.677 ms | 2.414 ms | 14.581 ms | 0.898 ms | 1.00x |
| PlaMatrix 原生 SparseCpu | 10.547 ms | 5.764 ms | 2.746 ms | 0.067 ms | **1.96x** |

首次稀疏符号分析耗时 1.612 ms；后续同拓扑求解复用该结构。两条路径的解向量最大绝对差在六位小数输出下为
`0.000000`，并通过 `<= 1e-8` 的程序断言。

原生稀疏路径目前的主要剩余成本已经从分解转移到完整对称 CSR 装配：虽然数值分解比 DenseCpu 快约 5.31 倍，
但稀疏装配约为 DenseCpu 的 2.39 倍，因此端到端收益约 1.96 倍。下一轮若继续优化，应优先让块稀疏因子直接消费
Schur block slots，减少标量 CSR 展开和遍历，而不是再替换求解器。

以上表格是原生 `SparseCpu` 接入时冻结的 S1 合成基线。后续参考轨迹优化没有改变该求解器的外部接口，
但进一步改进了真实 BA 的零阻尼有效子空间、迭代精化和 Schur 装配；当前真实夹具结果见下节。

## South Building 参考轨迹优化结果

以 Phase358/Phase536 捕获夹具为硬门禁，本轮又完成以下自研优化：

- 对固定宽度参数块中完全无效的坐标做等价消除，使有效子系统可以从零阻尼开始；
- 严格残差复核失败时复用同一因子执行至多 3 次迭代精化；
- 常见 `9x3` cross transform 和 3 项点积使用固定尺寸内核与连续 workspace；
- block slot 先用哈希去重，再排序恢复确定性编号；slot term 表改为两遍计数和 prefix-sum 连续存储。

Phase536 三次内部求解中位数由约 `1.188 s` 降到约 `0.934 s`，提升约 `21.4%`；最终 cost 与内参逐行一致。
双方 1 线程、3 次外部 wall 中位数如下：

| 夹具 | 参考 BA | 修复前 PlaScan | 当前 PlaScan | PlaScan 自身提升 |
|---|---:|---:|---:|---:|
| Phase358（2 相机/3,801 点） | 0.21 s | 0.47 s | 0.26 s | 1.81x |
| Phase536（29 相机/17,019 点） | 0.90 s | 5.12 s | 1.07 s | 4.79x |

当前 Phase358 的 `f/k1/k2` 与参考复刻分别相差约 `1.6e-6/1.7e-9/6.1e-10`；Phase536 与参考结束或
后续阶段输入分别相差约 `4.1e-9/6.3e-9/5.6e-9`。Phase536 的 Schur 装配仍约占内部求解主体，原生 LLT
分解只占约 `0.009 s`，因此下一轮仍应优化 block-to-CSR 数值展开，不能把主要时间归因于 Cholesky 本身。

## PlaScan 接线验证

PlaScan 的 160 相机、8,000 tracks、24,000 observations 联合 BA 实际运行报告：

- 求解器：`sparse_cholesky_native_cpu`；
- RMS：`6.541682075` 降至 `0.0343742902`；
- Schur pattern：构建 1 次、复用 4 次；
- 符号分析：构建 1 次、复用 4 次；
- 无后端回退。

CPU-only PlaMatrix 全量 286 项测试通过。独立 CUDA 构建排除未能编译的既有 benchmark 后 524/524 通过，
其中 CUDA/OpenCL Schur 测试实际执行；该 benchmark 的失败是 NVCC 在未修改的 `release3_cases.cu` 中将
默认化的 `IterativeSolverWorkspace<float>` 构造函数判为 deleted。PlaScan BA/SfM/line-scan 定向 137/137 通过。
标准 BA 程序的动态链接和直接 `NEEDED` 检查均未发现 CHOLMOD、SuiteSparse、BLAS、LAPACK 或 OpenBLAS。
