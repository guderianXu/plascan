# PlaScan 深度估计、融合与网格深度渲染真实数据验证

更新日期：2026-08-14

状态：revision 34 的三类深度正确性结论仍为“有条件通过”，默认参数暂不调整；补充的丝川 222 v40
闭合模型已通过实跑拓扑与深度完整性门

对应计划：`docs/plans/2026-08-12-depth-map-rendering-correctness-optimization.md`

## 1. 结论

本轮 revision 34 没有发现负深度、非有限深度、工件来源缺失或存储深度无法回读的问题。Hyb2 的严格
paired A/B 表明有效覆盖和多视支持显著增加，但相对 revision 27 的共同有效区深度差 P95 为 `1.7387%`，
未达到预先设置的 `1%` 旧结果稳定性门槛。

这个稳定性失败不能直接解释为精度回退。对同一批 Metashape 2.3.1 原生深度进行无 ICP、无 Sim(3) 的
逐像素比较后，revision 34 相对 revision 27 的 pooled 相对误差变化为：

- P50：`1.4071% -> 1.3890%`；
- P90：`2.1223% -> 1.4562%`；
- P99：`10.1176% -> 1.5153%`；
- 平均有效 mask IoU：`0.87918 -> 0.88524`。

因此，revision 34 相对旧 PlaScan 结果的主要尾部变化更接近独立参考，而不是随机漂移。原始深度比显示的
约 `1.39%` 比例差来自两套相机坐标尺度：Metashape/PlaScan 相机中心成对距离比例中位数为 `1.014386`，
深度所需对齐比例为 `1.014085`，两者只差 `0.0297%`。按该尺度做诊断性归一化后，形状残差 P90 从
`0.8000%` 降到 `0.2396%`，P99 从 `8.8348%` 降到 `1.0411%`。这消除了“PlaScan 固有 1.39% 深度
尺度错误”的证据，但尺度归一化结果仍不能替代同一坐标系下的绝对真值验收。

还确认了两项未收口问题：

- Hyb2 OpenCL 批次耗时由 `170.022 s` 增至 `178.067 s`，回退 `4.73%`；
- Temple 当前只有 `1/16` 帧为核心 accepted，另有 15 帧为低权重 `validation_only`，需要用真值或固定
  融合模型继续校准 `fusion_postprocess_coverage_loss`，不能仅凭覆盖率放宽门限。

所以本轮按“**深度正确性有条件通过、发布默认值门禁未通过**”处理：保留 revision 34 的语义修复和验证
工具，不回退到 revision 27；同时不调整 checkerboard、支持阈值或默认后端参数。

## 2. 验证范围和环境

### 2.1 场景

| 场景 | 用途 | 规模 | 后端 | 基线性质 |
|---|---|---:|---|---|
| Middlebury Temple sparse ring | 近景、遮挡、物体边界 | 16 × 640×480 | CUDA | revision 11 历史 sanity；13/16 帧源图计划不同 |
| Hayabusa2 Hyb2 | 行星弱纹理、多视支持、深度尾部 | 14 × 1024×1024 | AMD OpenCL | revision 27 严格 paired A/B |
| Agisoft aerial 3×3 | Brown 畸变、锁定相机、全流程回读与融合 | 9 × 6000×4000 | CUDA | 无旧 PlaScan 基线，功能与工件契约验证 |
| Temple 固定网格 | 真实相机下的网格深度光栅化非回退 | 16 个视角 | CPU renderer | 同一网格、相机和输出尺寸的历史重放 |

Agisoft 输入相机的 Brown 参数为 `k1=-0.0407083`、`k2=-0.176266`、`k3=0.246339`、
`p1=-7.0789e-05`、`p2=1.7325e-04`。按 6000×4000 四角估算，最大畸变位移约 `105.27 px`，因此不是
可忽略畸变样本。但该数据没有硬边界工程 mask，不能单独证明 mask 去畸变边界达到 1 px 门槛。

### 2.2 本机环境

- Windows 11 Pro `10.0.26200`；
- AMD Ryzen 9 9950X3D，16 核 / 32 线程；
- 93.65 GiB 可见内存；
- NVIDIA GeForce RTX 5080，16,303 MiB，驱动 `591.86`；
- OpenCL：Advanced Micro Devices `gfx1036`；
- MSVC Release + CUDA 构建目录：`build/windows-vcpkg-cuda-release`。

关键可执行文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `mvs_depth_reprocess_cli.exe` | `BD10D747D7A3FAFC22E99F857442A28DD9F018C0CE7AE671E2B24F2B56898925` |
| `reconstruct_pipeline_cli.exe` | `B144DB1DCF63BB27FBDE6D6D47991A8A4B564F799D4AD32A7D72EE4B61D32FBE` |
| `model_quality_cli.exe` | `13AD0D2D1C2CE13DF1C1AD0F1BBE6E2AC390FFC0C8AD0CDAD0804449294ECC93` |

输入清单 SHA-256：Temple `31416327554C5EFE2E1D5C7741B3182D37CBDC30240EE80B560E86B7ACD9690C`；
Hyb2 revision 27 `4F550AB993B8894DC67B1329F1FF3BA6FB4784D69BB01E0BFAB916A5FEDD5691`；
Agisoft list `69B711BE73655CD849B41351374AA0206A42B159E3D8F3A9D347E94C1194D266`。

## 3. Hyb2 严格 paired A/B

### 3.1 可比性审计

revision 27 与 revision 34 均为 14 个唯一 `ref_index` 和 14 个同名影像。逐帧检查以下字段全部相同：

- `ref_image`、`camera_model`、1024×1024 网格；
- `source_indices`、`source_images`、`source_plan`；
- `quality=highest`、`scene_profile=orbital_object`、`depth_filter=moderate`；
- OpenCL device 1、`source_views=6`、`threads=30`、`gpu_frame_workers=2`；
- 同一稀疏点云和 22 个 verified pairs。

因此可以归因到 revision 27 到 revision 34 的整体算法变化，但不能把结果再拆分归因到某一个独立补丁。

### 3.2 PlaScan 对 PlaScan

| 指标 | revision 27 | revision 34 | 变化 |
|---|---:|---:|---:|
| 完成帧 | 14/14 | 14/14 | 无失败 |
| 有效深度像素 | 1,845,481 | 1,936,503 | +91,022 |
| full-grid coverage | 0.1257134 | 0.1319138 | +0.6200 pp |
| mean confidence | 0.805479 | 0.843985 | +0.038506 |
| accepted / validation / rejected | 2 / 4 / 8 | 5 / 9 / 0 | 拒绝帧清零 |
| 有效像素中 support≥2 | 90.7541% | 99.8909% | +9.1368 pp |
| 有效像素中 support≥3 | 89.1254% | 99.7096% | +10.5842 pp |
| 批次耗时 | 170.022 s | 178.067 s | **+4.73%** |
| revision 34 峰值 working set | - | 1.179 GiB | 仅记录新运行 |

逐像素比较：

- valid-mask IoU：`0.9501337`；
- 共同有效像素：`1,842,638`；旧版独有 `2,843`，新版独有 `93,865`；
- 旧有效像素保留率：`99.846%`；
- 相对深度差 P50/P90/P95/P99：`0.00532% / 1.0912% / 1.7387% / 41.1864%`；
- revision 34 的负深度与非有限深度均为 0；14/14 帧 coverage 均未下降。

第 8 帧相对 revision 27 的 P95 为 `6.444%`，但它相对 Metashape 原生深度的 P99 为 `1.496%`。因此将
它保留为视觉/真值复核重点，不把“偏离旧版”直接标成错误。

使用以下门禁运行 `compare_plascan_depth_runs.py`：coverage 最多下降 2 pp、聚合 mask IoU 至少 0.90、
共同有效区深度差 P95 不超过 1%、非法深度为 0。coverage、IoU 和非法值三项检查通过，只有 1% 深度
稳定门槛失败。
报告：
`build/validation/depth_render_real_20260812/hyb2_rev27_vs_rev34.depth_ab.json`。

### 3.3 Metashape 原生深度独立参考

Metashape 2.3.1 从相同影像导出的原生 float depth 有 12/14 个同名视角，缺少 ref 1、3。比较按影像标签
和 1024×1024 像素直接进行，不使用 ICP 或 Sim(3)。它是独立算法参考，不是激光真值；两套相机坐标的
单位尺度并不完全相同，因此 raw depth 指标只适合比较同一参考下 revision 27→34 的相对变化。

| pooled 指标 | revision 27 | revision 34 | 变化 |
|---|---:|---:|---:|
| 相对误差 P50 | 1.40714% | 1.38895% | -0.01819 pp |
| 相对误差 P90 | 2.12230% | 1.45622% | -0.66608 pp / -31.38% |
| 相对误差 P99 | 10.11758% | 1.51527% | -8.60231 pp / -85.02% |
| candidate/reference depth ratio P50 | 0.985929 | 0.986111 | 反映参考坐标尺度差 |
| 12 帧平均 mask IoU | 0.879179 | 0.885244 | +0.006065 |
| reference-only fraction | 0.39304% | 0.05814% | 缺失明显减少 |

为区分全局尺度和局部形状，报告额外给出诊断性全局尺度对齐：将 candidate 乘以 pooled depth ratio P50
的倒数。revision 34 所需比例为 `1.014085`。作为独立交叉检查，12 个匹配相机中心产生 66 个成对距离，
Metashape/PlaScan 比例 P10/P50/P90 为 `1.011257 / 1.014386 / 1.019250`；深度对齐比例相对相机中心
P50 只差 `-0.0297%`。

| 尺度对齐后形状残差 | revision 27 | revision 34 |
|---|---:|---:|
| P50 | 0.04774% | 0.04562% |
| P90 | 0.80002% | 0.23957% |
| P95 | 1.15118% | 0.49594% |
| P99 | 8.83476% | 1.04107% |
| mean | 0.46945% | 0.10595% |

对应报告：

- `build/validation/depth_render_real_20260812/hyb2_rev27_vs_metashape_depth.json`；
- `build/validation/depth_render_real_20260812/hyb2_rev34_vs_metashape_depth.json`；
- `build/validation/depth_render_real_20260812/hyb2_rev34_vs_metashape_diagnostics/contact.png`。

另外将两版深度投影到已经注册的 Metashape 网格：revision 34 的跨帧 P50/P90 中位残差为
`0.2388% / 0.5859%`，revision 27 不作为该网格的历史固定报告。本地补算的 revision 15 对应值为
`0.2364% / 0.6074%`。该网格早先通过 scale-aware ICP 对齐到 PlaScan revision 15，只能作为形状参考，
不能用于独立绝对尺度结论。

## 4. Temple 近景验证

### 4.1 MVS 历史 sanity

revision 34 使用当前 pair audit 后只有 3/16 帧保持与 revision 11 完全相同的源图计划，因此以下结果只能
证明没有整体崩溃，不能作为严格算法 A/B：

| 指标 | revision 11 | revision 34 |
|---|---:|---:|
| 完成帧 | 16/16 | 16/16 |
| full-grid coverage | 0.2385500 | 0.2417816 |
| 有效像素 | 1,172,521 | 1,188,405 |
| mean confidence | 0.805758 | 0.818642 |
| accepted / validation | 7 / 9 | 1 / 15 |
| manifest 帧耗时均值 | 641 ms | 462 ms |

revision 34 批次耗时 `7.165 s`，外部 wall `7.990 s`，峰值 working set `0.517 GiB`。负深度和非有限
深度均为 0。15 个 validation-only 帧中，14 个原因为 `fusion_postprocess_coverage_loss`，另 1 个原因为
`excessive_adaptive_geometry_conflict`；这属于需要继续校准的质量门控变化，不能用 coverage 增加掩盖。

历史对比报告：
`build/validation/depth_render_real_20260812/temple_rev11_vs_rev34_historical.depth_ab.json`。

### 4.2 固定真实网格的 renderer 重放

使用同一 Temple PLY 网格、同一历史 MVS 相机、16 个 640×480 视角运行当前 `model_quality_cli`。当前报告与
2026-07-25 历史报告相比，coverage、IoU、双向 edge P90 和 16/16 个逐帧有效 mask 完全一致。修正后的
透视属性插值使 16 个 `render.png` 均发生预期变化，但 SSIM 变化很小：中位数
`0.46934235 -> 0.46937071`，单视角绝对变化最大 `5.91e-5`。

历史与当前共同指标：coverage 中位数 `0.985993`、IoU 中位数 `0.857398`、edge P90 中位数 `11.5 px`。
CLI 返回 3 是因为这份历史模型本身未通过既有 IoU/edge/SSIM 质量门槛；报告中
`error` 为空且 16 个视角全部成功渲染。因此本项证明真实网格的轮廓与颜色输出没有回退，不证明该旧网格
本身已经达到发布质量，也不替代 reciprocal-Z 的解析合成测试。

第一次重放后 `model_quality_cli.exe` 又被重新链接，最终结论只采用重链接后的第二次结果。当前输出：
`build/validation/depth_render_real_20260812/temple_mesh_renderer_rev34_current`。

## 5. Agisoft Brown 畸变全流程

锁定输入 TSAI 相机并运行 CUDA SIFT、CUDA MVS、`aerial_terrain`、`mild` filter。最终深度工件为完整
6000×4000，结果如下：

- SfM 注册 `9/9`，稀疏点 `8,008`，过滤后稀疏点 `4,123`，平均重投影误差 `0.58191 px`；
- MVS 完成 `9/9`，coverage mean/min/max 为 `0.490798 / 0.383866 / 0.612876`；
- 有效深度 `106,012,372`，mean confidence `0.755611`；
- 4 accepted / 5 validation-only / 0 rejected；
- 负深度和非有限深度均为 0；
- MVS 内部耗时 `61.601 s`，深度-only pipeline `68.776 s`，外部 wall `97.004 s`；
- 峰值 working set `7.763 GiB`。

随后在同一隔离输出目录运行融合。日志显示深度估计 `success=0 failed=0 skipped=9`，确认 revision 34 深度
被回读而不是重算；几何支持证据在融合后处理阶段仍可用。融合输出：

- 原始稠密点云 `1,944,525` 点；
- 精炼点云 `1,898,824` 点；
- pipeline 内部耗时 `31.524 s`，其中 MVS/融合阶段 `26.290 s`。

权威 pipeline report 的 `status=ok` 且进程 exit code 为 0。外层计时包装器在
`agisoft_distorted_rev34_cuda.fusion.run.json` 中误写了 `status=failed`，所以该字段不作为验收依据。

本场景证明明显 Brown 畸变影像可完成锁定相机的深度生成、工件回读和流式融合，但由于没有工程 mask 和
独立地形真值，不能据此宣称 image/mask/depth 域边界已经在真实数据上达到 1 px，也不能给出绝对 DEM 精度。

## 6. 深度来源与工件完整性

`summarize_mvs_depth_provenance.py` 对三组 revision 34 manifest 均返回 `passed=true`：

| 场景 | 帧 | 有效像素 | 实测比例 | 受锚插值比例 | 未分类 | 工件存在 |
|---|---:|---:|---:|---:|---:|---:|
| Temple | 16 | 1,188,405 | 88.4063% | 11.5937% | 0 | 16/16 |
| Hyb2 | 14 | 1,936,503 | 92.3282% | 7.6718% | 0 | 14/14 |
| Agisoft | 9 | 106,012,372 | 100% | 0% | 0 | 9/9 |

报告位于 `build/validation/depth_render_real_20260812/*.provenance.json`。这里的“实测”包括 native、targeted、
cross-view measured 和 residual PatchMatch；插值只统计 anchored interpolation。

## 7. 新增验证工具和验证命令

新增 `scripts/validation/compare_plascan_depth_runs.py` 及
`scripts/validation/plascan_depth_compare/`，支持：

- manifest 最终帧配对和相对工件路径；
- CV_32FC1/CV_16UC1 Fast Matrix 严格读取；
- coverage、mask IoU、相对深度、几何支持、逆深度离散度和 acceptance 转移；
- 有界、确定性的 pooled 分位数采样；
- coverage、IoU、深度差和非法值可选门禁。

同时扩展 `compare_mvs_depth_to_metashape.py`，把全局比例偏差和尺度对齐后的局部形状残差分开报告。

本轮脚本验证：

```powershell
python -m py_compile scripts/validation/compare_plascan_depth_runs.py
$comparisonModules = Get-ChildItem scripts/validation/plascan_depth_compare -Filter *.py
foreach ($module in $comparisonModules) {
    python -m py_compile $module.FullName
}
python -m py_compile scripts/validation/compare_mvs_depth_to_metashape.py
python -m pytest tests/test_compare_plascan_depth_runs.py `
  tests/test_compare_mvs_depth_to_metashape.py -q
```

结果：包内 8 个 Python 文件语法检查全部通过，pytest 为 `9 passed`。

核心运行参数保存在：

- `build/validation/depth_render_real_20260812/temple_rev34_cuda.run.json`；
- `build/validation/depth_render_real_20260812/hyb2_rev34_opencl_amd.run.json`；
- `build/validation/depth_render_real_20260812/agisoft_distorted_rev34_cuda.run.json`；
- `build/validation/depth_render_real_20260812/agisoft_distorted_rev34_cuda.fusion.run.json`。

## 8. 丝川 222 环拍闭合模型 v40-v46 补充验证

### 8.1 问题定位与修复路径

补充验证使用日本小行星探测任务“丝川”地面模拟影像项目 222。深度源为
`D:\plascan-validation\222_rev37_lowest_full_v1`，模型运行目录为
`D:\plascan-validation\222_rev37_sparse_scaffold_v40\model_runs\20260813T204332337Z-91e2350e-186c-48fb-baed-b8f3b8cff6ed`。
旧问题不是单纯的网格显示错误：局部深度不能稳定提供完整的小行星全局闭合载体，而未经质量筛选的稀疏
PLY 又会把短轨迹、大重投影误差和小三角化角的错误点带进表面重建，造成卫星碎片、错误拓扑或与丝川
双叶外形不一致的实体。

v40 实际选择的算法为 `orbital_sparse_scaffold_screened_poisson`，处理顺序如下：

1. 把正式 SfM PLY 与逐点质量 sidecar JSON 作为不可拆分的配对输入，流式读取 sidecar；逐点检查数量和
   坐标对齐，默认仅保留 `track_len >= 3`、`rms_reproj_px <= 1.5`、三角化角 `>= 5°` 的点。
2. 执行有限值、径向和统计离群点过滤、体素降采样，并根据鲁棒中心生成朝外法向；稀疏点不注入 TSDF。
3. 使用固定版本的官方 PoissonRecon Screened Poisson，而不是 PlaPoint 的旧通用 Poisson/PCG 路径；
   `pointWeight` 必须大于 0，避免静默退化为非 screened Poisson。
4. 只保留 Poisson 网格的最大面连通分量，删除所有卫星分量。
5. 若最大分量仍不是单连通闭合 genus-0 二流形，强制执行保守体素化、六邻域外部洪泛和递增半径闭运算。
   只有体素实体为单分量且 Euler 数为 1、边界网格为单分量闭合二流形且 Euler 数为 2 时才允许继续。
6. 已选择该路径后，PLY/sidecar 配对不完整或错位、质量过滤后点数不足、Screened Poisson 或体素拓扑
   修复失败，以及最终深度完整性或质量门失败，都会停止写出，不再静默退回原始 PLY 或错误的旧载体。

### 8.2 v40 真实运行结果

| 阶段/指标 | v40 结果 |
|---|---:|
| 注册项目影像 / accepted / qualified primary 深度帧 | 222 / 69 / 65 |
| 辅助桥选择 | primary 图原有 5 个分量；选择 ref `36, 67, 179, 205` 后连通 |
| 稀疏输入点 | 140,858 |
| 质量规则拒绝 / 统计离群点拒绝 | 71,715 / 2,513 |
| 过滤和降采样后骨架点 | 47,254 |
| 官方 Screened Poisson 输出 | 164,623 顶点 / 328,318 面 |
| Poisson 连通性 | 281 个分量；最大分量占面 `97.7315%` |
| 最大连通分量过滤 | 删除 280 个卫星分量；保留 160,343 顶点 / 320,870 面 |
| 过滤后拓扑 | 1 个分量，Euler `-92`，仍不满足闭合 genus-0 要求 |
| 体素拓扑修复 | 已执行并采用，闭运算半径 2；修复后 1 个分量、Euler `2` |
| 最终写出网格 | 92,608 顶点 / 185,212 面，4,630,580 bytes |
| 最终拓扑 | 单分量、闭合二流形、Euler `2`、genus `0` |
| 最终缺陷计数 | 开放边 0、非流形边 0、非流形顶点 0，最大分量面占比 1.0 |
| 最终完整性 aggregate / median / P10 / minimum | `79.5201%` / `78.7822%` / `72.5940%` / `65.5510%` |
| 缺口边界 minimum recall | `74.0434%` |
| 完整性采样 | 300,582 点中解释 239,023 点；最终门通过 |
| 模型核心耗时 | 75.685 s |
| 外层进程观测峰值内存 | 约 38 GB（近似采样，运行 JSON 未持久化精确峰值字段） |

报告中的 `post_simplification_face_count=119999` 是中间阶段计数，不是最终交付网格；最终 PLY 和报告根级
`vertex_count/face_count` 均为 `92608/185212`。通用最终门字段虽然记录
`final_topology_quality_gate_policy=observation_only`，但稀疏全局载体在交接前已经通过自身严格闭合拓扑门，
而且最终实际统计仍满足严格条件：`final_topology_quality_input_strict_gate_passed=true`、开放边和非流形均为
0、Euler 为 2。最终表面去噪候选没有被接受（`final_surface_denoising_accepted=false`），未用平滑结果掩盖
拓扑失败。

### 8.3 无配准诊断载体交叉检查

外部几何审计位于 `build/codex-doming-verify/validation/222_v40_vs_oracle_geometry.json`，对 v40 最终 PLY
与 v29 诊断载体 `sparse_poisson_voxel_0p01_close1.ply` 各采样 100,000 点，且没有 ICP/Sim(3) 对齐：

| 方向 | 距离 P50 | 距离 P95 | 距离不大于 0.05 的比例 |
|---|---:|---:|---:|
| v40 → v29 诊断载体 | 0.005246 | 0.074378 | 92.962% |
| v29 诊断载体 → v40 | 0.003845 | 0.104872 | 91.675% |

双向均值 Chamfer 为 `0.015749`（均为工作区坐标单位）。两者包围盒对角线分别为 `3.31018` 和
`3.33602`，且都是单分量闭合网格。v29 只是内部诊断载体，不是独立真值；这些数字只能说明 v40 没有
再次坍缩成相机环或无关大块，不能据此宣称丝川表面的绝对几何精度。

### 8.4 v46 平滑闭合交付复验

在 v40 的形状修复基础上，v46 把拓扑体素分辨率提高到 224，并加入三级 fail-closed 表面策略：先对
闭合占据体构造 `[1, 2, 1]^3` 局部平均场，以半整数阈值避开等值点退化，再执行 MC33 平滑提取；若
MC33 候选拓扑正确但三角形质量未过门，则先进行不改连通关系的受保护翻边，再执行最多两轮、每轮位移
不超过局部平均边长 5% 的切向质量优化。每个候选都必须重新通过单分量、闭合二流形、Euler `2`、
genus `0` 和严格三角形质量门；任一步失败仍退回精确体素边界。真实运行目录为
`D:\plascan-validation\222_rev37_sparse_scaffold_v46_smooth_tangent\model_runs\20260813T214055781Z-e918940a-2992-4e06-aa98-43baf48f0207`。

| 阶段/指标 | v46 结果 |
|---|---:|
| 实际算法 / 有效补全模式 | `orbital_sparse_scaffold_screened_poisson` / `sparse_scaffold_completion` |
| 最终门策略 / 是否通过 | `strict` / true |
| 体素分辨率 / 体素尺寸 / 闭运算半径 | 224 / `0.00998297` / 1 |
| MC33 平滑候选 | 已尝试并采用；未使用 cell-boundary 回退 |
| MC33 三角质量优化 | 12 轮翻边，78,859 次；切向微调最多 2 轮 |
| 优化前高 / 极端长宽比面比例 | `1.62921%` / `0.766573%` |
| 优化后高 / 极端长宽比面比例 | `0.412740%` / `0.007116%` |
| 最终额外去噪 | 候选未通过事务验收，未改变已通过门禁的 MC33 网格 |
| 最终写出网格 | 126,474 顶点 / 252,944 面 |
| 最终拓扑 | 1 个分量、开放边 0、非流形边/顶点 0、Euler `2`、genus `0` |
| 最终完整性 aggregate / median / P10 / minimum | `79.0150%` / `78.3666%` / `72.2255%` / `64.7093%` |
| 缺口边界 minimum recall | `73.0725%` |
| 完整性采样 | 300,582 点中解释 237,505 点；最终门通过 |
| 模型核心耗时 | 81.815 s |

外部几何审计文件为
`build/codex-doming-verify/validation/222_v46_smooth_tangent_vs_oracle_geometry.json`，三视图为
`build/codex-doming-verify/validation/222_v46_smooth_tangent_vs_oracle_render.png`。在没有 ICP/Sim(3) 对齐的
100,000 点双向采样中，v46 与 v29 内部诊断载体的 Chamfer-L1 为 `0.015720`，候选到诊断载体和反向的
距离 P50 分别为 `0.003248`、`0.003536`；两者包围盒对角线分别为 `3.30487`、`3.33602`。三视图确认
双叶轮廓、颈部和长轴方向一致，最终模型没有再次重建成相机环、中心碎片或无关大块。

另使用通用恐龙数据集式图像质量工具对 45 个近景视角做过诊断：轮廓覆盖中位数为 `99.9967%`，IoU
为 `0.76849`、SSIM 为 `0.69166`，但边缘误差门失败。这批自然背景近景中目标经常充满画面或出框，
不存在可供通用轮廓门稳定判断的完整前景剪影，因此该结果保留为工具适用性限制，不替代本节的严格
拓扑门、逐帧深度召回和无配准几何交叉检查。

### 8.5 当前 222 批次的低主帧骨架辅助复验

2026-08-14 对用户当前项目目录
`E:\code\test\tw2-222\222.files\1\mvs_output` 做了原位完整性审计：222 帧全部完成，18 帧为
`accepted && fusion_eligible` 主帧，145 帧为验证帧，59 帧拒绝；222 个支持掩码、163 个必须的几何来源
掩码和 222 份稀疏绝对深度残差均完整可读，审计为零错误、零警告。单独依赖 TSDF 时 18 帧仍低于
环拍覆盖下限 56 帧；当且仅当同源稀疏 PLY 与逐点质量 sidecar JSON 成对存在且骨架补全启用时，GUI
与核心工作流允许这些深度帧作为局部几何证据继续运行，稀疏骨架负责全局形状。缺少任一骨架文件、
禁用骨架补全或主帧少于 2 帧时仍按原策略失败关闭。

GUI 的“禁用插值”是深度观测策略，不是禁用同源空三骨架。无论插值由自然背景策略自动关闭，还是由
对话框显式保存为 `disabled`，只要上述骨架配对和低主帧保护条件成立，都必须允许稀疏骨架承载全局
形状；否则 GUI 会在基础 TSDF 的多分量最终门失败，而 CLI 默认设置却能成功，形成入口行为不一致。

真实运行目录为
`D:\plascan-validation\222_current_batch_sparse_fallback_v47\model_runs\20260814T153230364Z-22f671ed-1c50-4612-b4f7-81a12284dda5`。

| 指标 | 当前 18 主帧批次结果 |
|---|---:|
| 低主帧骨架辅助放行 | 已采用；实际载入 18，深度独立下限 56 |
| 实际算法 | `orbital_sparse_scaffold_screened_poisson` |
| 空三输入 / 质量过滤后 | 140,858 / 47,254 点 |
| 最终写出网格 | 126,474 顶点 / 252,944 面 |
| 最终拓扑 | 单分量、闭合二流形、Euler `2`、genus `0` |
| 最终缺陷计数 | 开放边 0、非流形边/顶点 0 |
| 最终完整性 aggregate / median / P10 / minimum | `83.9826%` / `94.9454%` / `60.5424%` / `54.3817%` |
| 缺口边界 minimum recall | `54.3817%`；最终门通过 |
| 运行结果 | 成功，约 73 s |

该 PLY 的 SHA-256 为
`4DD8D7094FBA27B8D28C6F495E6FAD0B87EC7E5AAA52B18B7281DD5633A2A18E`，与 v46 已验证交付模型逐字节
一致。无配准的 100,000 点双向检查得到 Chamfer-L1 `0.01572044`；候选到诊断载体及反向的距离 P50
分别为 `0.00324782`、`0.00353566`。报告和三视图位于
`build/codex-doming-verify/validation/222_current_v47_vs_oracle_geometry.json` 与
`build/codex-doming-verify/validation/222_current_v47_vs_oracle_render.png`。三视图确认当前批次输出保持丝川
的长轴、双叶轮廓和颈部，没有形成相机环、中心块或方盒。

### 8.6 结论与剩余风险

v46 已解决本次“整体形状完全对不上”的载体级故障：最终输出恢复丝川双叶全局形状，采用平滑 MC33
曲面，并通过严格拓扑、三角形质量和深度完整性门。仍需保留以下风险：

- 官方 Screened Poisson 加体素拓扑修复的外层进程峰值约 38 GB，尚不适合低内存机器；需要继续做内存
  分段、分辨率预算和峰值遥测。
- v46 已消除精确 cell-boundary 的方块阶梯，但细节仍受约 `0.01` 工作区单位体素和稀疏骨架精度限制。
  后续平滑或重网格仍必须受位移、深度完整性和拓扑事务共同约束。
- 配置目标为 120,000 面，但拓扑安全的 v46 最终网格为 252,944 面；当前优先保形和闭合，减面仍需
  单独优化，不能直接复用会破坏闭合门的普通简化结果。
- 本轮只有 222 一个 Windows/MSVC 真实项目，且诊断载体不是独立真值；仍需加入更多环拍小天体、
  Linux/GCC 复跑和有测量真值的几何验收。

这项补充验证不改变本文对 revision 34 深度默认参数的“有条件通过”结论；它验证的是深度不完整时的
环拍全局闭合载体和 fail-closed 交付路径。

## 9. 未覆盖范围和下一步

1. **参考坐标尺度**：Hyb2 的 raw depth 比例已经由相机中心尺度差解释到 `0.03%` 内。后续公开基线必须先
   冻结同一相机坐标和单位；不把诊断性比例乘法写入生产深度，也不把尺度归一化结果冒充绝对精度。
2. **OpenCL 性能**：复测并定位 revision 34 的 `+4.73%` 批次耗时回退，正确性门禁通过后再优化。
3. **Temple 门控**：用固定融合模型或独立网格真值校准 `fusion_postprocess_coverage_loss`，不能直接放宽阈值。
4. **畸变边界**：增加带硬边界 mask 和独立几何真值的真实畸变场景，补齐 1 px image/mask/depth 域验收。
5. **大世界坐标**：当前真实稀疏 PLY 读取仍是 float，不能安全把完整场景平移到 `1e8` 做等价复跑；NUM-01
   由合成 PatchMatch 测试覆盖，真实大坐标输入链仍是风险。
6. **公开开源后端**：本机未安装 COLMAP/OpenMVS 可执行文件，本轮没有直接运行第三方开源 MVS；Metashape
   只作为已有独立深度参考。后续应冻结 COLMAP workspace 后补同相机 A/B。
7. **平台**：本轮仅验证 Windows/MSVC/CUDA/OpenCL，没有 Linux/GCC 实跑。

基于以上结果，revision 34 可以保留并继续优化，但还不满足“调整默认 PatchMatch/checkerboard 参数”或
“宣布真实数据验证全部通过”的条件。
