# 环拍角度缺口感知的深度融合与分扇区质量计划

> 状态：待实施
> 日期：2026-07-28
> 适用范围：匹配对审计 → MVS 选源 → 跨视深度证据 → TSDF 融合 → 网格完整性
> 首要数据集：Hyb2
> 非回归数据集：Dino、Temple
> 主构建目录：`E:\code\plascan\build\windows-vcpkg-cuda-release`

## 1. 目标

解决环拍数据存在不均匀视角间隔时，深度图表面看似完整、但大量像素无法在
TSDF 中获得两个独立相机支持，最终形成整片网格空洞的问题。

本计划的核心不是全局降低质量阈值，而是：

1. 让当前影像集合的匹配对质量可审计，不再把缺少验证统计的回退源当成已验证源。
2. 在 MVS 和 TSDF 中显式识别角度缺口及其影响扇区。
3. 只在缺口扇区恢复具有强几何证据的单视图表面，普通扇区继续使用双相机支持门。
4. 区分原生深度、跨视修补深度和仅用于验证的宽基线深度。
5. 用逐扇区完整性门阻止“全局中位数通过、局部仍缺掉一整面”的模型交付。

## 2. 非目标与安全约束

- 不把 `minimumDistinctCameraSupport` 全局降为 1。
- 不把 `minimumGeometrySupportCount` 全局从 4 降到 3。
- 不把 75° 宽基线影像直接作为普通 PatchMatch 主源。
- 不允许跨视修补像素独立创建新表面。
- 不启用无轮廓、无深度、无可见性证据的大孔洞填充。
- 不用更高面数、全局拉普拉斯平滑或纹理遮盖几何空洞。
- 不删除或覆盖 Hyb2、Dino、Temple 的深度图和模型产物。
- 每个实验使用独立输出目录，默认行为只有通过三数据集回归后才能更新。

## 3. Hyb2 当前证据基线

数据来源：

- `E:\code\test\hyb2\mvs_output\mvs_manifest.json`
- `E:\code\test\hyb2\assets\reports\reconstruction_quality_report.json`
- `E:\code\test\hyb2\mvs_output\products\model_from_mesh.ply`

### 3.1 匹配、相机和选源

| 指标 | 当前值 |
|---|---:|
| 注册影像 | 12/12 |
| BA RMS | 0.3834 px |
| 平均重投影误差 | 0.5211 px |
| 稀疏点 | 2,332 |
| 平均轨迹长度 | 2.989 |
| 每帧 MVS 源视图 | 3–4 |
| 已验证源对 | 0 |
| 源对类型 | 全部为 `track_geometry_backfill` |

选中的常规相邻源角度约为 28°，二阶相邻源约为 56°。第 9、10 帧之间存在
47° 缺口，相关二阶组合达到约 75°，导致第 8–11 帧只能选择 3 个安全源。

### 3.2 深度完整性与跨视支持

| 指标 | 常规帧 | 弱扇区 |
|---|---:|---:|
| 蒙版内有效深度 | 95.9%–98.9% | 第 10 帧 89.9% |
| 至少双视图深度证据 | 93.6%–95.0% | 第 9 帧 80.5% |
| 至少四份几何证据 | 51.9%–54.6% | 第 9 帧 15.9%，第 10 帧 21.1% |
| 跨视确认票 | 64.3%–67.9% | 第 9 帧 53.6%，第 10 帧 55.5% |
| 遮挡票 | 14.6%–22.1% | 第 9 帧 35.5%，第 10 帧 32.0% |

最终深度图没有大型内部孔洞。缺值主要连接轮廓和深度突变边界，并且无效像素
的平均图像梯度高于有效像素，因此不能归因于单纯的低纹理匹配失败。

### 3.3 TSDF 与模型结果

| 指标 | 当前值 |
|---|---:|
| 多视图支持样本 | 2,659,562 |
| 单相机样本拒绝 | 2,129,746 |
| 最终边界边 | 10,312 |
| 第 9 帧网格召回率 | 0.4268 |
| 第 10 帧网格召回率 | 0.5681 |
| 其余帧召回率 | 0.7383–0.8256 |
| 全局中位召回率 | 0.7891 |
| 全局质量门 | 通过 |
| 严格拓扑门 | 未通过 |

结论：主要故障是角度缺口扇区内的深度无法稳定落入同一个 TSDF 表面，而不是
深度图普遍缺值。缺少匹配对验证统计会降低可审计性，但不是大孔洞的唯一原因。

## 4. 设计原则

### 4.1 三类深度证据

所有进入融合的深度像素必须标记为以下一种：

1. `native_confirmed`：原生估计且具有跨视确认，可正常参与 TSDF。
2. `repaired_anchored`：由跨视插值恢复，必须依附已有表面，不能独立创建表面。
3. `wide_baseline_validation`：由宽基线视图提供，仅用于验证、可见性和冲突否决。

现有 `crossViewRepairedMask`、`geometrySupportCount`、
`geometrySourceMask` 和逆深度离散度继续作为像素级依据，不新增无法追踪来源的
“已修补”布尔值。

### 4.2 缺口扇区不是全局宽松模式

角度缺口恢复必须同时满足：

- 场景为 `orbital_object`。
- 最大相邻角度与中位相邻角度之比超过可配置阈值，初始值 1.5。
- 当前体素投影落入缺口两侧相机共同可见的目标轮廓。
- 没有强自由空间冲突。
- 仅对缺口影响扇区生效，其他体素继续要求双相机支持。

### 4.3 修补像素只能扩展已证实表面

`repaired_anchored` 像素不得作为单视图恢复种子。它只能在以下条件下扩展：

- 邻域已有多视图支持的零等值面。
- 扩展距离不超过 1–2 个体素。
- 与邻域法线和深度变化连续。
- 来源相机集合与锚点至少有一个共同源。

## 5. 文件与职责

### MVS 和匹配对审计

- 修改 `src/core/mvs/MvsSourcePlanner.h/.cpp`
- 修改 `src/core/mvs/DepthMapGenerator.cpp`
- 修改 `src/core/mvs/MvsWorkspaceManifest.h/.cpp`
- 修改 `src/core/mvs/tests/test_mvs_source_planner.cpp`
- 修改 `src/core/mvs/tests/test_mvs_pipeline.cpp`

### 角度缺口与 TSDF 融合

- 修改 `src/core/mesh/DepthFusionFramePolicy.h/.cpp`
- 修改 `src/core/mesh/DepthTsdfSurfaceBuilder.h/.cpp`
- 修改 `src/core/mesh/ModelWorkflowService.cpp`
- 修改 `tests/test_mesh_reconstructor.cpp`

### 完整性质量门与报告

- 修改 `src/core/mesh/DepthMeshCompleteness.h/.cpp`
- 修改 `src/core/qc/ProcessingBaselineManager.h/.cpp`
- 修改 `src/gui/project/manager/ProjectModelManager.cpp`
- 修改 `src/core/mvs/README.md`
- 修改 `docs/PROJECT_ARCHITECTURE.md`

### 验证

- 修改 `scripts/validation/run_mesh_quality_baseline.ps1`
- 新增 Hyb2 场景配置到 `scripts/validation/mesh_quality_scenes.json`
- 使用 `scripts/validation/render_mesh_comparison.py`

## 6. 实施任务

### Task 1：冻结 Hyb2 诊断基线

- [ ] 为每帧保存原生有效率、修补像素率、几何支持直方图和逆深度离散度分位数。
- [ ] 为每个相邻相机保存角度、共享轨迹、验证内点和源视图等级。
- [ ] 为最终网格保存逐帧召回率、逐扇区召回率和边界归因。
- [ ] 把当前 47° 缺口、10,312 条边界边和第 9/10 帧召回率写入基线。

验收：

- 同一输入重复运行时诊断值稳定。
- 报告能直接指出最差扇区和对应 `ref_index`。

### Task 2：补齐当前匹配对几何审计

- [ ] 区分“缺少验证统计”和“几何验证失败”，不得都映射为低质量。
- [ ] 对当前影像集合重新生成或计算匹配对内点、覆盖和重投影一致性。
- [ ] `source_plan` 写入 `verification_status`、内点率、覆盖率和缺失原因。
- [ ] 源视图不足目标数量时写入 `source_shortfall_reason`。
- [ ] 保留相机/稀疏轨迹回退，但在 GUI 和报告中明确标成回退。

验收：

- Hyb2 被使用的常规源对不再全部显示 `verified_pair_geometry=0`。
- 删除或重新引用影像后，不读取旧影像集合的匹配验证结果。
- 验证失败的源对不能因共享轨迹数量大而自动升级为主源。

### Task 3：角度缺口分析和帧角色

- [ ] 扩展 `DepthFusionFramePolicy`，输出中位间隔、最大缺口、缺口比值和缺口两侧帧。
- [ ] 为每帧标记 `normal_sector`、`gap_boundary` 或 `gap_opposite`。
- [ ] 对 75° 宽基线候选只赋予 `wide_baseline_validation` 角色。
- [ ] 将角度缺口和帧角色写入 TSDF 统计 JSON。

单元测试：

- 均匀 12 相机环拍不得产生缺口扇区。
- 28° 中位间隔加 47° 缺口必须识别第 9、10 帧。
- 非环拍和航空场景不得启用该策略。

### Task 4：缺口感知的 MVS 选源

- [ ] 常规 PatchMatch 继续使用约 28°/56° 的已验证源。
- [ ] 当安全主源不足 4 个时，记录短缺而不是无条件加入 75° 源。
- [ ] 宽基线源通过几何和光度门后，只参与一致性验证与遮挡判断。
- [ ] 为缺口两侧帧提高互补轮廓覆盖权重，避免所有源集中在同一侧。
- [ ] 保持每帧显存和计算预算有上限。

验收：

- 第 8–11 帧的选源报告包含安全主源和宽基线验证源的明确分组。
- 宽基线验证源不能直接增加原生有效深度计数。
- Temple 薄柱和真实开口不能因宽基线源而被错误封闭。

### Task 5：受约束的缺口扇区单视图恢复

- [ ] 保持全局 `minimumDistinctCameraSupport=2`。
- [ ] 新增缺口扇区专用的单视图候选，不修改普通 `isSampleSupported()` 语义。
- [ ] 候选必须是 `native_confirmed`，禁止修补像素作为种子。
- [ ] 初始约束：
  - 置信度不低于 0.70。
  - `geometrySupportCount >= 3`。
  - `geometrySourceMask` 至少包含两个不同源。
  - 逆深度相对离散度不高于 0.008。
  - 26 邻域中至少 6 个多视图支持样本。
  - 至少两个相机轮廓同意占据。
  - 不存在自由空间否决。
- [ ] 单视图候选权重上限不超过普通多视图样本的 0.45。
- [ ] 只允许沿已有表面扩展，不允许生成孤立组件。

回滚门：

- 新增非流形边。
- 新增独立组件。
- Temple 真实开口被封闭。
- Dino/Temple Chamfer-L1 退化超过 2.5%。

### Task 6：锚定的边界表面延伸

- [ ] 仅处理归因于 `insufficient_distinct_camera_support` 的开放边界。
- [ ] 用现有零等值面锚点、轮廓共识和深度梯度预测 1–2 体素延伸。
- [ ] `repaired_anchored` 像素只能参与该阶段。
- [ ] 与自由空间、前后表面冲突或深度离散度过大的候选必须拒绝。
- [ ] 输出考虑、接受和各类拒绝计数。

验收：

- Hyb2 第 9/10 扇区边界减少。
- 不使用无证据的多边形大孔填充。
- 新增表面与深度观测距离必须落在完整性容差内。

### Task 7：分扇区完整性质量门

- [ ] `DepthMeshCompleteness` 按相机方位角划分扇区。
- [ ] 输出每个扇区的样本数、召回率、最差帧和边界边归因。
- [ ] 保留全局 median/P10，同时增加最差有效扇区门。
- [ ] 缺少足够样本的扇区标记为不可验证，不能静默记为通过。
- [ ] GUI 错误信息显示最差扇区和相关影像编号。

Hyb2 第一阶段门：

- 全局中位召回率不低于 0.78。
- P10 不低于 0.58。
- 第 9 帧召回率从 0.4268 提高到至少 0.60。
- 第 10 帧召回率从 0.5681 提高到至少 0.65。
- 最终边界边从 10,312 至少下降 30%。

### Task 8：GUI 与诊断可见性

- [ ] 深度图节点显示原生、修补和弱支持像素比例。
- [ ] 模型任务进度显示当前最差扇区召回率。
- [ ] 质量报告列出匹配对验证缺失、角度缺口和恢复启用状态。
- [ ] 不新增阻塞式完成弹窗。
- [ ] 所有耗时诊断继续在模型 worker 中运行。

### Task 9：三数据集回归与默认值决策

- [x] Hyb2 使用当前 12 张影像和现有深度进行 A/B。
- [x] Hyb2 重新生成一次深度，确认匹配对审计和选源生效。
- [x] Dino 检查耳朵、脖颈和底座，不允许出现新桥接。
- [x] Temple 检查柱子、柱间真实开口和底座，不允许错误封孔。
- [x] 生成统一正面、侧面、顶面 contact sheet。
- [ ] 只有全部质量门通过后，才允许把缺口恢复设为轨道场景自动策略。

### 2026-07-28 实施验证记录

- Hyb2 匹配审计：66 对候选中 13 对通过几何验证，0 对被证明失败，53 对为匹配为空或数量不足。
  证据不足不再被错误当作几何失败。
- 12 视图环拍高质量源池保持四个近邻源；高质量 1024×1024 深度金字塔恢复
  `4→2→1` 全分辨率末层。后续发现 GUI 通用质量预设仍会把配置值扩大到六源，因此稀疏环拍
  四源现改为场景上限而非推荐下限。算法修订号更新为 10，修订 9 的六源缓存不会透明复用。
- Hyb2 新深度/新网格：12 帧全部融合，单组件，P10/中位/最低召回
  `0.6289/0.8625/0.5442`；第 9/10 帧为 `0.5442/0.6123`，仍低于本计划的
  `0.60/0.65` 最终目标，因此缺口恢复自动策略尚未视为完成。
- Hyb2 照片投影：IoU `0.9246`、边缘 P90 `15.50 px`、SSIM `0.4413`；
  冻结旧深度对照为 `0.9306/15.13 px/0.3986`。严格 3 px 边缘和 0.75 SSIM 门仍未通过。
- GUI 等价复验显式传入七源，日志确认自动收敛为
  `source_pool=4 (configured=7)`。第 9/10 帧深度一致性保留率为 `0.9077/0.8303`，
  12 帧中位数 `0.9755`。两帧原始深度和最终 PLY 与已验收四源回归哈希一致，因此网格召回
  恢复为 `0.5442/0.6123`；TSDF 融合 12 帧并输出单组件，不再触发截图中的
  `0.3333/0.359` 缺口扇区完整性失败。
- Dino 当前深度回归：16 帧、单组件、436 条边界边，IoU `0.9366`、SSIM `0.8709`；
  耳朵、颈部、底座和真实开口无新增桥接。
- Temple 当前深度回归：16 帧、单组件、864 条边界边，IoU `0.9077`、SSIM `0.6059`；
  柱间真实开口保持。工作区产物已从冻结清单的 161 个变为 274 个，因此冻结哈希脚本按设计拒绝
  伪装成同输入基线；本次使用独立输出和照片投影 contact sheet 验证。
- Hyb2 像素级孔洞诊断确认：PatchMatch 原始深度接近完整，大块空洞主要由跨视一致性过滤产生；
  旧锚定插值又会因连通域略超固定像素上限、或通过狭窄通道接触轮廓而整块拒绝。修订 11 改为
  保留 4 像素轮廓保护带，只对支持蒙版内部核心执行锚定逆深度插值；项目蒙版开口不参与。
  第 9/10 帧蒙版内有效率由 `85.4%/80.0%` 提高到 `98.9%/99.2%`，12 帧均达到
  `98.2%–99.2%`。新网格保持单组件，照片投影中位覆盖率/IoU 从 GUI 等价基线
  `0.9633/0.9257` 提升到 `0.9745/0.9348`；SSIM 从 `0.4313` 小幅降为 `0.4256`，
  严格边缘与 SSIM 门仍未通过，因此后续仍需优化表面细节，不得把本次完整性修复描述为最终质量达标。
- 测试通过：MVS pipeline 66、source planner 18、workspace manifest 19、depth pyramid 3、
  MVS types 17、rectifier 9、match photo task 23。`test_mesh_reconstructor` 为 152/156；
  四个失败均在 `ConsistentIsoSurfaceExtractorTest` 的既有合成闭合拓扑用例，未改动或隐藏。

## 7. 测试与构建命令

编译：

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release `
  --target test_mvs_source_planner test_mvs_pipeline `
           test_mesh_reconstructor test_cli_contracts mesh_reconstruct_cli `
  -j 3
```

定向测试：

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release `
  -R "test_mvs_source_planner|test_mvs_pipeline|test_mesh_reconstructor|test_cli_contracts" `
  --output-on-failure
```

质量回归：

```powershell
powershell -ExecutionPolicy Bypass -File `
  E:\code\plascan\scripts\validation\run_mesh_quality_baseline.ps1 `
  -BuildDirectory E:\code\plascan\build\windows-vcpkg-cuda-release
```

所有 Python 可视化和统计脚本必须使用：

```powershell
E:\code\plascan\.venv\Scripts\python.exe
```

## 8. 实施顺序与阶段提交点

1. Task 1–3：只增加审计、缺口识别和报告，不改变几何结果。
2. Task 4：改进缺口帧选源，重新生成 Hyb2 深度并冻结新基线。
3. Task 5：加入默认关闭的受约束单视图恢复，做 Hyb2 A/B。
4. Task 6：处理仍可归因的开放边界。
5. Task 7–8：把分扇区质量门和 GUI 诊断接入正式流程。
6. Task 9：通过 Hyb2、Dino、Temple 后再决定轨道场景默认值。

每一阶段都必须保留独立输出和统计。任何阶段出现桥接、组件增加或 Temple
真实开口被封闭，都应回滚该候选策略，不得用后续平滑掩盖。

## 9. 完成定义

本计划只有同时满足以下条件才算完成：

- 当前影像集合的匹配对验证状态完整且不引用已移除影像。
- 角度缺口及受影响帧能在报告和 GUI 中直接定位。
- Hyb2 第 9/10 扇区达到阶段召回率目标。
- Hyb2 不再出现截图中的大面积开放区域。
- Dino 和 Temple 无新桥接、无真实开口误封、无显著几何退化。
- 所有定向测试通过，主 Windows 构建目录可生成 GUI 和 CLI。
- 架构文档、MVS README、验证配置和质量报告字段同步更新。
