# PlaMatrix CUDA BA 专项优化（2026-09-04）

## 实现

本轮没有新增第三方库，也没有改动 BA 残差、局部参数化、LM/Armijo 接受规则或质量门。优化集中在
PlaMatrix 自研 CUDA Schur-PCG：

- 对常见 3 维消元块，先在 GPU 上计算每条 `inverse * cross_row`，Schur 标量项由重复的 3×3 双循环
  改为 3 项点积；其它块尺寸继续使用通用 kernel。
- CUDA 自适应 PCG 每轮仍在设备端检查容差并记录首个收敛轮次；主机每 8 轮读取一次状态。命中后设备冻结
  solution/residual，使同批次后续已排队轮次成为数值 no-op，不越过首个收敛解。
- 将 PCG 的 solution/residual 两次 AXPY 合为一个 kernel，并将 direction 的 SCAL+AXPY 合为一个 kernel，
  减少相机 Schur 向量较小时的 launch 开销。

CPU 与 OpenCL 求解流程没有改动。公共 `convergenceCheckInterval` 默认仍为 1；只有 CUDA Schur 内部明确使用 8。

## 合成 BA 真机 A/B

环境为 Linux/GCC Release、Intel i7-13700H（20 逻辑线程）和 RTX 4060 Laptop GPU。固定内参，联合优化
相机位姿与点，10 次非线性预算，严格 `1e-12` 线性容差，禁用回退；每个进程先执行一次冷启动，表中是随后
3 次热启动的 wall 中位数。所有记录均为 `gpu=true`、`fallback=false`。

| 规模 | 优化前 wall | 优化后 wall | wall 提升 | 优化前线性阶段 | 优化后线性阶段 |
|---|---:|---:|---:|---:|---:|
| 80 相机 / 3,000 点 / 24,000 观测 | 0.1916 s | 0.1634 s | 14.7% | 0.0974 s | 0.0642 s |
| 256 相机 / 10,000 点 / 80,000 观测 | 0.6265 s | 0.5310 s | 15.2% | 0.3359 s | 0.2070 s |
| 512 相机 / 20,000 点 / 160,000 观测 | 1.4306 s | 1.2366 s | 13.6% | 0.7551 s | 0.5503 s |

三组 final cost 在当前输出精度下始终为 `48.31155563`、`160.7575468`、`321.3686064`。浮点执行次序改变后，
累计 PCG 迭代由 `1288/3677/6713` 变为 `1294/3673/6689`，但求解状态、RMS、接受规则和最终 cost 不变。
CUDA assembly 是异步提交的，报告中的部分 kernel 等待时间记入后续 D2D/linear 区间，因此以完整 wall 判断收益，
不能把 `linear_solve_seconds` 的变化全部解释为 PCG 算术加速。

## South Building Phase536 同输入验证

双方读取完全相同的 29 相机、17,019 点、106,020 观测捕获夹具。当前 CPU SparseCpu 与 CUDA 均成功，
无回退；CUDA 明确运行在 RTX 4060：

| 指标 | CPU SparseCpu | CUDA Schur-PCG |
|---|---:|---:|
| final cost | `11607.289064323028` | `11607.289064322920` |
| f | `2485.077250069588` | `2485.077250069438` |
| k1 | `-0.0365434473834433` | `-0.0365434456039105` |
| k2 | `0.0217625740719421` | `0.0217625703949288` |
| total | 0.951 s | 0.959 s |

CUDA 与 CPU 的 final cost 相差约 `1.1e-10`，标定参数差异在 `4e-9` 内；两者与参考后续阶段标定均在
`1e-8` 量级。该夹具只有 29 台相机，GPU 初始化、PCG 和小块准备抵消了设备 Schur 装配收益，因此不能用于
宣称 CUDA 比 CPU 更快；它用于证明真实输入质量没有因优化回退。

## 调度结论

新的交叉扫描在 64–160 相机区间没有形成可稳定下调阈值的单调优势，Phase536 也再次证明小相机网络应使用
CPU。因此保留现有 Auto 的保守 CUDA 常规门槛和高密度覆盖规则；显式选择 CUDA 不受该门槛限制。后续若继续
优化，优先减少法方程到设备的数值上传和 Schur 结果 D2D 交接，不应改变当前已对齐的非线性策略。

## 验证

- PlaMatrix CPU-only 全量 286/286 通过；CUDA/OpenCL 库与设备测试 526/526 通过。
- PlaScan BA、共享/自适应内参、投影、SfM 和 line-scan 定向测试 157/157 通过。
- 受影响 Release target 与 `ba_backend_benchmark` 构建通过；本轮修改文件的 clang-format dry-run、
  主仓库和 PlaMatrix 子模块 `git diff --check` 均通过。
- CUDA 独立构建的 9 个 benchmark 注册测试未纳入 526 项门禁；它们依赖本任务前已存在且未构建的 benchmark
  可执行文件，不影响本轮算法库和真机设备用例。
