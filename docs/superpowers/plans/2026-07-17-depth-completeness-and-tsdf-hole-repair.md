# 蒙版感知深度完整性与 TSDF 孔洞修复实施计划

**目标：** 保持“深度图直接生成 TSDF 三维模型”的主链，不引入密集点云中间步骤；解释 Level 3 不可用的真实原因，阻止最终深度后处理无提示地制造内部空缺，并让模型插值设置真正作用于 TSDF 小孔修复。

**原则：** 先把每个阶段丢失的像素和 TSDF 拒绝原因量化，再做受证据约束的修复。不得用全局膨胀、闭运算、无条件深度插值或大范围网格封口掩盖问题；门洞、拱门、窗洞、轮廓边界等真实开口必须保留。

**执行约束：** 仅使用 Windows 原生 PowerShell 和 `E:\code\plascan\build\windows-vcpkg-cuda-release`。MVS 测试只放在 `src/core/mvs/tests`。Python 只用 `E:\code\plascan\.venv`。不 reset、checkout、clean、回滚，不删除现有项目、构建产物或回归结果，不 commit、不 push。

---

## 2026-07-19 Temple 收敛结果

- 质量评估修复了 Dino 前景掩膜错误：大门洞/柱间开口保持背景，只填对象内部的小噪点孔。旧评估把
  真实开口填成前景，导致约 `80 px` 的伪边缘 P90；校正后同一 Ultra 模型为 coverage `0.94723`、
  IoU `0.88895`、edge P90 `13.02 px`、SSIM `0.61767`。
- 新增逐视图缺失阶段归因和 `missing_stage.png`，区分 support mask 外、无有效深度、几何支持不足、
  以及有可靠深度但未形成网格。校正掩膜后，剩余缺失以几何支持不足和 TSDF 未成面为主。
- 环拍对象增加一致性过滤后的跨视补回：要求三个不同源视图、1.5% 深度簇、0.8 px 投影半径和局部
  深度一致。补回像素不参与帧准入评分，避免低质量帧被补回结果提升为 accepted。Temple 保持
  13 accepted / 3 validation-only，补回 `4193` 像素；320 模型 coverage `0.95077→0.95115`、IoU
  `0.87530→0.88454`、edge P90 `12.83→12.23 px`、SSIM `0.60935→0.61084`。
- Ultra TSDF 使用断层感知亚像素 2x2 采样，2% 之外的另一深度模式不参与平均。同深度 A/B 中
  coverage `0.84286→0.84526`、IoU `0.82612→0.82753`、edge P90 `80.34→80.07 px`，边界边减少
  约 2.7%；该处旧 P90 使用的是尚未修正开口的掩膜，仅用于同配置 A/B。
- 最新 Temple Ultra 结果为 coverage `0.94975`、IoU `0.88763`、edge P90 `12.48 px`、SSIM
  `0.61545`，单一连通分量。支持 3 的同深度变体只改善 `0.18 px` P90，却增加约 2.5% 开放边界，
  因此仍保留支持 4 为默认。严格 IoU `0.90`、P90 `3 px`、SSIM `0.75` 门槛尚未全部达成。

---

## 已确认基线与根因边界

### 2026-07-18 毛边收敛补充

- 近景检查确认毛边主要是低视角支持的深度有效边界进入 TSDF 后形成的连通薄片，既有连通域过滤
  无法删除它们；小孔填补开关对毛边外观和质量指标几乎没有影响。
- 超高质量曾改为至少三相机支持，并在内存中对最终深度有效边界收缩两像素；后续深度边缘回归证明
  三相机门槛会删除整段薄结构，最终恢复为双相机支持，同时保留两像素收缩和弱边界尖片清理。
  高质量保持双相机支持和一像素收缩。原始深度图及用户数据不被改写。
- 开放网格边界只执行一次、最大位移 `0.35` 体素的平滑，内部顶点不参与。Temple 384 回归的边界边
  从两相机基线 `139751` 降至 `115466`，coverage `0.8132`、IoU `0.8040`、edge P90 `80.12 px`、
  SSIM `0.5703`。
- 缩窄到五体素截断带虽把边界降至 `87462`，但 coverage 同时降到 `0.7524`，会扩大柱体缺失，故未
  固化；仅提高累计权重和 aggressive 置信度预设没有改变本数据的网格。

### 1. Level 3 不是点击故障，而是本次没有生成

- Temple 深度网格为 `640 x 480`。
- `src/core/mvs/DepthPyramidPolicy.cpp` 要求每层短边至少为 `160` 像素。
- Level 3 使用 1/4 分辨率时短边只有 `120`，因此策略只激活 Level 1、Level 2。
- 当前 16 帧均有 Level 1、Level 2，Level 3 为 0 帧；不能强行把不存在的 Level 3 文件显示出来。
- 需要修复的是 UI 解释：Level 3 保持禁用，但必须显示“当前 640×480 深度只生成 2 层；Level 3 的 1/4 短边 120 小于 160”的原因。Level 2/Level 1 和总叠加按钮不能被连带禁用。

### 2. 模型空缺首先来自最终深度，不是连通域清理

- Temple 项目蒙版占整图 `20.05%–30.82%`，平均 `26.03%`；这会有意排除大部分背景和地面。因此，与包含地面的 Metashape 模型比较时必须使用相同蒙版。
- 最终深度在有效项目蒙版内的覆盖率为 `68.56%–94.22%`，平均 `86.38%`；内部缺失真实存在。
- Level 1 输出汇总有 `1,279,186` 个有效像素，最终质量分析前只剩 `1,112,477` 个，损失 `166,709`，即 `13.03%`；单帧损失最高 `30.82%`。
- 后续已记录的置信度/散斑后处理仍保留 `98.92%`，所以主要损失发生在金字塔结果与最终质量分析之间。当前最可疑的是 `DepthMapGenerator.cpp` 输出前的 `removeLocalDepthOutliers()`，但在增加逐阶段计数前不把它当作唯一结论。
- 计划制定时的 TSDF 网格曾记录 `component_count=1`、`largest_component_face_ratio=1.0`；后续把
  `support_mask_path` 与最终有效深度掩膜分离后，诊断暴露出两个大主体分量，因此不能再仅凭旧结果
  断言连通域清理与空缺无关。

### 3. TSDF 支持门槛和插值契约存在缺口

- `DepthTsdfOptions.minimumVoxelWeight` 默认是 `1.0`，修复后的 `minimumDistinctCameraSupport` 默认是 `2`；
  显式设置为 `1` 时仍保留强单次观测路径。
- 实际单次观测权重是“像素置信度 × 帧平均置信度”；平均置信度约 `0.9` 时，强单次观测通常仍小于 `1.0`，于是配置写着允许单视图支持，实际却经常需要两次观测。
- Temple 中 14 帧只有 2 个源视图，2 帧有 3 个源视图，薄结构和弱纹理区域更容易因上述隐式双视图门槛消失。
- 生成模型对话框会保存 `interpolation=enabled`，但 `depth_tsdf` 分支没有使用 `ReconstructionConfig.fillHoles`。这不是参数调优问题，而是明确的设置契约缺失。
- 已有 `detail::fillSmallBoundaryHoles()` 可复用，但只能用于有上限的小边界环，不能默认填所有孔。

### 4. 最近一次同配置模型质量基线

- coverage：`0.7590`
- IoU：`0.7350`
- edge P90：`69.90 px`
- SSIM：`0.18257`

这些数值只和相同相机、渲染背景、蒙版、模型分辨率及评估脚本的结果比较。

## 2026-07-17 实施结果

- Level 3 可用性已从单一布尔值升级为“状态 + 原因”。Temple 640×480 只生成 Level 1/2，Level 3
  明确提示 1/4 短边 120 小于 160；其余层级和总叠加按钮不被连带禁用。
- 新增 `DepthCompletenessMetrics` 和逐阶段 manifest 统计。Temple 第二次诊断回归的蒙版内平均覆盖为
  `0.974426`，最低帧 `0.788588` 被降为 `validation_only`；小内部孔像素从 33,988 降为 8,914，
  减少 `73.8%`。
- 主要深度损失实际发生在跨视图一致性阶段：旧实现用工程总帧数 16 选择严格策略，而不是当前帧的
  实际 source views。最终策略为 1 源视图只移除明确矛盾，2 源视图要求至少一张在 10% 内确认，
  3 个以上来源使用 5% 确认阈值；遮挡证据不再被误判为矛盾。
- `valid_mask_path` 现在表示最终有效深度，`support_mask_path` 表示项目/内容允许区域；TSDF 默认排除
  `validation_only` 帧，并为强单次观测设置独立权重门槛。掩膜外自由空间雕刻改为显式可选、默认关闭。
- TSDF 分量诊断增加每个分量的面数和包围盒。4–5.5 体素截断带会把 Temple 沿世界 Y 切成两个
  大分量；7.5 体素得到单一连通体。普通插值限制为最多 16 条边、直径最多 4 体素的小闭环。
- Temple 环绕对象的源视角上限从 35°调整为 47°，航测仍为 35°。47°回归把每帧来源提高到 2–5，
  深度蒙版内平均覆盖为 `0.938001`、最低 `0.868078`，最终为 12 accepted / 4 validation-only。
  默认 7.5 体素 TSDF 模型为单一分量，coverage `0.80717`、IoU `0.78111`、edge P90 `80.82 px`、
  SSIM `0.56732`。覆盖、轮廓重叠和外观已改善，但结构边缘仍未达到 65 px 目标，质量优化仍未完成。
- 将对象上限扩大到 50°会引入一个 17,672 面的内部浮层；因此拒绝 48–49°来源，并把默认小分量
  面数比例从 2.0% 调整为 2.5%。9 体素截断未改善边缘指标；旧的无限自由空间融合下，强制双相机
  支持也没有收益，后续只有与有限自由空间共同使用时才设为默认。
- 回归脚本会验证 `.lis` 首条有效记录确实包含存在的影像与 `.tsai`，不再误选 image-only probe；
  质量评估使用 MVS workspace 相机，并汇总完整性、TSDF 支持、分量和补洞统计。
- UAV9 防回归发现全图航测帧被错误套用 Temple 的 0.80 约束蒙版门槛；现已限定该门槛只作用于
  project/content 蒙版，并为航测边缘/内部帧保留原有一致性阈值。
- 最新 UAV9 完整回归为 pipeline `ok`、9/9 completed、7 accepted、2 validation-only、0 rejected；
  直接读取九个 v3 `SFTB` 特征文件得到每张 SIFT 均为 40000，`over_limit=0`。低一致性航测帧降为
  validation-only，只有低于 0.20/0.25 的真正崩溃才拒绝。
- 模型外观链已拆成两个明确产物：PLY 使用带网格遮挡、深度/视角一致性和颜色离群拒绝的鲁棒顶点色；
  OBJ 使用原始相机影像逐面投影图集，不再采用会让不同表面重叠的全局平面 UV。Temple 相机图集
  `1,137,405` 个面全部获得映射，其中 `25,485` 个为显式记录的保守回退。
- 超高质量 TSDF 默认剥离一轮“全部顶点均为弱相机支持”的开放边界尖片，高质量档默认不启用且两档
  均可显式覆盖。Temple 候选面 `12,345`，最终边界边从 `115,466` 降至 `107,948`，coverage/IoU/SSIM
  仅分别变化为 `0.8126`/`0.8036`/`0.6018`；edge P90 仍约 `80.10 px`，未达到 `65 px` 完成目标。
- 深度金字塔父层传播改用图像引导加权中位深度，避免在断层两侧平均出不存在的中间表面；跨视检查
  改为 3x3 投影邻域搜索，并用相对深度容差、真实相机基线和 3 px 数值余量共同形成往返投影包络。
  固定 1.5 px 往返门槛会把全部 Temple 帧降级，已通过真实回归否决。
- 新 Temple 深度回归保持 15 accepted / 1 validation-only，蒙版内平均覆盖 `0.958636`。384 双相机
  TSDF + 弱边界清理得到 coverage `0.83451`、IoU `0.82304`、edge P90 `79.67 px`、SSIM `0.60312`，
  相比此前超高档四项均改善。单相机实验虽将 P90 降至 `77.69 px`，但边界升至 `169744` 并出现
  柱间浮片，故未固化。
- 新 UAV9 CUDA depth-only 防回归为 pipeline `ok`、9/9 completed、8 accepted、1 validation-only、
  0 rejected，优于旧基线的 7/2/0；九个 v3 `SFTB` 文件均为 40000 个 SIFT 关键点，`over_limit=0`。
- 最终深度产物新增 `raw_geometry_support_path`：每个有效像素记录参考帧自身加上通过深度邻域和往返
  投影验证的源视图数，不再把 PatchMatch 的候选来源数误当成真实跨视支持。旧 workspace 缺少该图时
  按零确认处理，继续要求两个实际相机观测。
- Ultra TSDF 只为“总几何支持至少 4（参考帧 + 3 个独立源视图）且观测权重至少 0.85”的单帧体素
  开放补全；High 默认关闭。新 Temple 同深度 A/B 中，保守补全恢复 `239167` 个样本，保持单一分量，
  coverage `0.84040→0.84286`、IoU `0.82478→0.82612`、edge P90 `80.41→80.34 px`、SSIM
  `0.60371→0.60582`。支持门槛 3 虽恢复 `451731` 个样本，但开放边界增长更多，故未选为默认。
- 仅使用强几何像素重纳 `validation_only` 帧会把 coverage/SSIM 降到 `0.84053/0.60289`；恢复被两像素
  边界侵蚀移除的强几何像素也会增加开放边界且不改善 P90，两个实验均未固化。剩余约 80 px 的 P90
  主要来自少数视角的大块缺面，而不是统一的一像素边缘偏移。

## 2026-07-18 柱体完整度继续优化结果

- Temple 项目蒙版只描述外轮廓，柱间真实黑色开口仍被误标为前景。新增暗背景环拍细化：只在项目
  有效区内部结合原始灰度挖出暗开口，保护 4 像素外轮廓带，并要求至少保留 75% 原有效区；航测、
  亮背景和近全图内容不进入该分支。Temple 正式接受帧从 12 增至 15，大内部无效区从 51,796 降至
  15,139 像素，最低蒙版内覆盖为 `0.8763`。
- 将三源以上一致性从 5% 放宽到 7.5% 虽得到 16/16 accepted，却使 coverage 降至 `0.7776`、SSIM
  降至 `0.5088`，因此已恢复 5%；这证明不能靠放宽深度一致性填柱体。
- 根因进一步定位到 TSDF 长距离自由空间：任一错误的远背景深度都会以饱和 `+1` 抵消其他相机看到
  的近处细柱。纯窄带融合虽然得到 coverage `0.9950`、IoU `0.9070`，但产生大量窗口内薄片，予以拒绝。
- 最终采用最大 36 体素正自由空间距离，并默认要求两个不同相机支持。320 分辨率模型为单一分量、
  896,601 面、边界边 79,824，coverage `0.83729`、IoU `0.81848`、edge P90 `81.50 px`、SSIM
  `0.55727`；384 分辨率超高档为单一分量、1,305,672 面，coverage `0.83913`、IoU `0.82039`、
  edge P90 `80.03 px`、SSIM `0.56549`。柱体与横梁明显更连续，结构边缘仍未达到 65 px 目标。

---

## 完成标准

### UI 与项目状态

- 工作区继续只显示一个不可直接打开的聚合“深度图”节点，用来表示深度已生成并支持删除；不恢复逐张深度图标签页。
- 深度可视化仍只通过照片上的“叠加显示深度信息”按钮进入。
- 对当前照片分别计算 Final、Level 1、Level 2、Level 3 可用性；缺一层不得禁用其余层或总按钮。
- Level 3 不存在时动作禁用，并在 tooltip/status tip 给出基于活动金字塔与分辨率的具体原因。
- 删除聚合深度节点只删除项目中的深度记录和对应深度产物，不删除源照片、蒙版或其他用户数据；保留现有回归测试。

### 深度完整性

- 每帧记录有效蒙版像素数、蒙版内有效数、蒙版内覆盖率、各后处理阶段前后有效数、阶段保留率、小内部孔像素、边界相连缺失像素、大内部开口像素以及修复来源。
- 所有进入 TSDF 的 `accepted` 帧，输出前破坏性过滤的保留率不得低于 `0.90`。低于此值时必须经过证据约束修复、保守回退，或降为 `validation_only`；不得静默当作正常帧。
- Temple 蒙版内平均有效覆盖目标不低于 `0.90`，最低单帧目标不低于 `0.80`。
- 小内部孔像素相对当前基线减少至少 `50%`；大门洞、拱门开口以及连接蒙版轮廓的缺失区域不得被自动填充。
- 有效蒙版外的深度始终保持为零。

### TSDF 与网格

- `interpolation=disabled`：不运行边界孔填充。
- `interpolation=enabled`：只填受边界边数和物理直径双重限制的小闭合环。
- `interpolation=extrapolated`：允许更大的显式上限，但仍不得填大门洞或开放轮廓。
- 强单次观测可在 `minimumDistinctCameraSupport=1` 时进入 TSDF；低置信度单次观测仍被拒绝；显式设置为 2 时必须保持双相机支持要求。
- JSON 结果记录所有 TSDF 拒绝计数、单视图/多视图支持样本数、实际阈值、填孔数及填孔前后边界统计。
- Temple 网格继续保持非空、单个主连通体，不能因补孔新增漂浮组件。

### 回归质量

- 硬性无回退线：coverage 不低于 `0.7590`，IoU 不低于 `0.7350`，edge P90 不高于 `69.90 px`，SSIM 不低于 `0.18257`。
- 质量优化完成目标：coverage 至少 `0.78`，IoU 至少 `0.75`，edge P90 不高于 `65 px`，SSIM 至少 `0.20`。
- 若深度完整性改善但模型指标未达目标，报告真实结果并继续定位，不通过扩大盲填孔范围来“过指标”。

---

## 文件范围

### 新增

- `src/core/mvs/DepthCompletenessMetrics.h`
- `src/core/mvs/DepthCompletenessMetrics.cpp`
- `src/core/mvs/tests/test_mvs_depth_completeness.cpp`

### 修改

- `src/core/mvs/CMakeLists.txt`
- `src/core/mvs/DepthMapGenerator.h/.cpp`
- `src/core/mvs/DepthFrameQualityGate.h/.cpp`
- `src/core/mvs/MvsQualityReport.h/.cpp`
- `src/core/mvs/MvsWorkspaceManifest.h/.cpp`
- `src/core/mvs/tests/test_mvs_pipeline.cpp`
- `src/core/mvs/tests/test_mvs_workspace_manifest.cpp`
- `src/core/mvs/tests/test_mvs_depth_pyramid.cpp`
- `src/core/mesh/DepthTsdfSurfaceBuilder.h/.cpp`
- `src/core/mesh/ModelWorkflowService.cpp`
- `tests/test_mesh_reconstructor.cpp`
- `src/gui/views/DepthOverlayData.h/.cpp`
- `src/gui/widgets/DepthOverlayController.h/.cpp`
- `src/gui/widgets/CanvasWidget.h/.cpp`
- `src/gui/menu/MainMenu.h/.cpp`
- `src/gui/main_window/MainWindow.cpp`
- `tests/test_gui_project_utils.cpp`
- `scripts/validation/run_depth_overlay_regression.ps1`
- `src/core/mvs/README.md`
- `docs/PROJECT_ARCHITECTURE.md`

只在实现时确认确实需要的文件中改动；不做机械重构，不触碰无关 dirty 文件。

---

## Task 1：先补可归因的逐阶段诊断

### 1.1 增加蒙版感知指标组件

在 `DepthCompletenessMetrics` 中定义并独立测试以下数据：

```cpp
struct DepthCompletenessMetrics
{
    int maskPixelCount = 0;
    int validWithinMaskCount = 0;
    float validWithinMaskRatio = 0.0f;
    int invalidWithinMaskCount = 0;
    int smallInteriorHoleCount = 0;
    int smallInteriorHolePixelCount = 0;
    int largeInteriorOpeningPixelCount = 0;
    int boundaryConnectedInvalidPixelCount = 0;
};
```

分类规则：

- 先把输入蒙版归一为“255 表示允许深度”的 `CV_8U` 有效蒙版。
- 缺失集合是 `effective_mask != 0 && depth <= 0`。
- 缺失连通域若有像素与有效蒙版外部 8 邻接，归为轮廓相连缺失，永不自动修复。
- 不触及轮廓且面积不超过 `max(64, round(maskPixelCount * 0.002))` 的连通域归为小内部孔。
- 其余内部连通域归为大开口，只统计、不自动修复。

先写合成测试：实心蒙版、边界缺口、小内部孔、大拱门孔、空深度、尺寸不匹配。测试必须证明小孔和真实开口不会混为一类。

### 1.2 记录每个破坏性阶段的计数

在 `DepthMapGenerator` 为最终 Level 1 保存只读阶段快照或计数：

1. 金字塔选中结果；
2. 重新应用项目蒙版后；
3. 稀疏支撑软约束后；
4. 输出前局部离群过滤后；
5. 置信度过滤后；
6. 小组件/散斑过滤后；
7. 最终修复后。

每个阶段记录 `valid_count`、`valid_within_mask_count`、相对上一阶段的移除数和保留率。把目前未持久化的输出前 `removeLocalDepthOutliers()` 移除数单独写入，不再和后续 `depth_postprocess` 混在一起。

### 1.3 持久化并验证恢复兼容性

向深度 artifact、`depth_quality` 和 workspace manifest 增加：

- `mask_pixel_count`
- `valid_within_mask_count`
- `valid_within_mask_ratio`
- `pre_output_filter_valid_count`
- `post_output_filter_valid_count`
- `output_filter_retention_ratio`
- `small_internal_hole_pixel_count`
- `large_internal_opening_pixel_count`
- `boundary_connected_invalid_pixel_count`
- `restored_from_prefilter_count`
- `restored_from_parent_level_count`

旧项目没有这些字段时用 `-1`/空值表示不可用，不能伪造为 0。扩展 `test_mvs_workspace_manifest.cpp` 验证新字段往返和旧 manifest 读取。

### 1.4 质量门接入完整性

扩展 `DepthFrameQualityInput`，增加 `validWithinMaskRatio` 和 `outputFilterRetentionRatio`：

- `retention < 0.75`：Rejected，原因 `destructive_output_filter_collapse`；
- `0.75 <= retention < 0.90`：ValidationOnly，原因 `output_filter_coverage_loss`；
- `validWithinMaskRatio < 0.80`：至少 ValidationOnly，原因 `insufficient_mask_normalized_coverage`；
- 旧项目或无有效蒙版时不应用这些新门槛，保持向后兼容。

先在 `test_mvs_pipeline.cpp` 写失败测试，再实现质量门。

---

## Task 2：让 Level 3 禁用状态可解释

### 2.1 持久化活动金字塔契约

每帧记录：

- `pyramid_requested_level_count`
- `pyramid_active_level_count`
- `pyramid_minimum_short_side`
- `pyramid_degraded_reason`

Temple 应明确得到 `requested=3`、`active=2`、`minimum_short_side=160`，原因包含 `480 / 4 = 120 < 160` 的等价信息。保留现有 `fallback_reason` 兼容字段，但 UI 优先使用专用金字塔原因。

### 2.2 从布尔值升级为“可用性 + 原因”

在 `DepthOverlayData` 增加轻量状态：

```cpp
enum class DepthOverlayAvailabilityCode
{
    Available,
    NotComputedForResolution,
    NotPersisted,
    ArtifactMissing,
    NoDepthRecord
};

struct DepthOverlayAvailability
{
    bool available = false;
    DepthOverlayAvailabilityCode code{};
    QString reason;
};
```

`DepthOverlayController` 针对当前照片和每个层级返回该状态。`CanvasWidget`/`MainWindow`/`MainMenu` 传递原因，菜单动作禁用时把原因写入 tooltip 和 status tip。不能只显示笼统的“当前层不可用”。

### 2.3 GUI 回归测试

在 `tests/test_gui_project_utils.cpp` 覆盖：

- 640×480、active=2：Final/L1/L2 可用，L3 禁用且原因正确；
- Level 3 记录存在但文件缺失：原因是 artifact missing，不伪称分辨率不足；
- Level 2 缺失：只禁用 Level 2，总按钮和 Final/L1 仍可用；
- 从不可用层切换照片后，动作状态和已选层恢复正确；
- 聚合深度节点仍存在且不可直接激活，删除行为不回归。

不改变 `DepthPyramidPolicy` 的 160 像素安全下限，也不为 Temple 强制生成伪 Level 3。

---

## Task 3：修复最终深度阶段制造的小孔

### 3.1 把输出前局部过滤拆成“候选”和“决策”

保留 `removeLocalDepthOutliers()` 的公开兼容入口，但内部拆为：

1. 生成局部离群候选掩码；
2. 对候选做父层/置信度/支持证据验证；
3. 只清零未被证据保护的候选；
4. 返回候选数、实际移除数、受保护数及保留率。

证据保护必须同时满足：

- 候选仍在有效项目蒙版内；
- 预过滤深度、置信度有效；
- 多视支持达到当前源视图数允许的安全下限；
- 与上采样父层深度在相对阈值内一致；
- 邻域跨越明显深度断层时不进行保护或插值。

这样保留被中值滤波误伤的薄柱和纹理细节，但不会简单撤销所有离群过滤。

### 3.2 仅修复有几何证据的小内部孔

局部过滤后，对 Task 1 分类出的“小内部孔”执行两级修复：

1. 优先恢复预过滤候选中仍有高置信度和多视支持的原深度；
2. 仅在原深度不可用时，使用上采样父层深度，并要求父层置信度、支持数和边界一致性通过。

修复后的置信度必须降权，并记录来源。以下区域一律不修：

- 与项目蒙版边界相连的缺失；
- 超过小孔面积上限的内部开口；
- 深度断层两侧差异过大的孔；
- 父层和邻域互相矛盾的区域；
- 有效蒙版外区域。

如果整帧输出过滤保留率仍低于 0.90，不进行整图盲回填：选择更安全的父层结果需经过同一蒙版归一化质量门，否则把该帧标为 `validation_only`。

### 3.3 合成测试

在 `src/core/mvs/tests/test_mvs_depth_completeness.cpp` 和 `test_mvs_pipeline.cpp` 验证：

- 单像素/小斑块错误孔在父层一致时修复；
- 大拱门保持为零；
- 轮廓边界缺口保持为零；
- 深度断层上的小孔不跨边插值；
- 低置信度或单视图弱支持不恢复；
- 修复不在蒙版外写入深度；
- 一帧过滤损失 30% 时不会仍以 Accepted 静默进入 TSDF。

---

## Task 4：修正 TSDF 支持门槛，不靠全局降阈值

### 4.1 消除“配置允许单视图、实际隐式要求双视图”的矛盾

保留累计权重数组，同时记录每个体素的最大单次观测权重和观测次数。默认支持判定改成显式两路：

```text
多视图路：distinct_support >= 2 且 accumulated_weight >= minimumVoxelWeight
强单视图路：minimumDistinctCameraSupport == 1
          且 distinct_support == 1
          且 max_observation_weight >= minimumSingleObservationWeight
```

初始安全值在测试和 Temple 数据上校准：

- 累计权重门槛继续默认 `1.0`；
- 强单次观测门槛：mild `0.60`、moderate `0.70`、aggressive `0.80`；
- 显式 `tsdfMinimumVoxelWeight` 和新增 `tsdfMinimumSingleObservationWeight` 优先于自动值；
- 用户显式设置 `tsdfMinimumDistinctCameraSupport=2` 时完全禁用强单视图路。

这允许高置信度薄结构保留，同时避免把所有弱单视图噪声放进网格。

### 4.2 完整统计 TSDF 拒绝原因

扩展 `DepthTsdfStatistics` 和 `statisticsToJson()`，至少输出：

- `rejected_projection_count`
- `rejected_support_mask_count`
- `rejected_depth_valid_count`
- `rejected_depth_count`
- `rejected_confidence_count`
- `single_view_supported_sample_count`
- `multi_view_supported_sample_count`
- `rejected_accumulated_weight_count`
- `rejected_single_observation_weight_count`
- `effective_minimum_voxel_weight`
- `effective_minimum_single_observation_weight`
- `effective_minimum_distinct_camera_support`

当前已经在内存中计数但未写入 JSON 的字段必须全部落盘，便于回归报告区分“深度本身缺失”和“TSDF 门槛拒绝”。

### 4.3 TSDF 合成测试

在 `tests/test_mesh_reconstructor.cpp` 增加：

- 强单次观测在 support=1 时生成支持样本；
- 低置信度单次观测不生成支持样本；
- support=2 时仍需累计权重通过；
- 显式 minimumDistinctCameraSupport=2 禁用单视图路；
- 所有新增统计字段出现在 JSON 中；
- 默认设置不产生额外漂浮组件。

---

## Task 5：让模型“插值”设置真正作用于 depth_tsdf

### 5.1 增加 TSDF 小边界环填充选项

向 `DepthTsdfOptions` 增加：

- `fillSmallBoundaryHoles`
- `maximumHoleBoundaryEdges`
- `maximumHoleDiameterVoxels`

`ModelWorkflowService::depthTsdfOptionsFromSettings()` 显式映射：

- `disabled`：关闭填孔；
- `enabled`：保守边数/直径上限；
- `extrapolated`：使用更大但仍有硬上限的边数/直径；
- `maxHoleSize` 参与计算上限，最终有效值写入结果 JSON。

不得复用 Poisson 的 `holeFillPasses` 假装 TSDF 已处理；TSDF 必须有自己的真实选项和统计。

### 5.2 在正确的清理顺序中填孔

TSDF Marching Cubes 后按以下顺序：

1. 去退化面；
2. 去小漂浮连通体；
3. 焊接重合顶点；
4. 统计边界环；
5. 若设置允许，调用受边数限制的 `fillSmallBoundaryHoles()`，并额外执行物理直径检查；
6. 再次去退化面并重算法线；
7. 重新统计连通性和边界。

记录 `boundary_loop_count_before/after`、`filled_boundary_hole_count`、`added_hole_fill_face_count`。大边界环和开放边界不填。

### 5.3 网格测试

增加合成网格测试：

- disabled 时小孔保持；
- enabled 时小闭合孔填上；
- 大拱门环超过边数或物理直径上限，保持开放；
- extrapolated 只扩大显式上限，不填开放轮廓；
- 填孔后无退化面、法线有限、连通体数量不增加；
- 对话框保存的 interpolation 三种值在 TSDF options 中逐一可见。

---

## Task 6：回归脚本和报告闭环

### 6.1 扩展 Temple 报告

`scripts/validation/run_depth_overlay_regression.ps1` 在现有去重、退出码和模型质量解析基础上，汇总：

- 每帧 active pyramid level 和不可用原因；
- 蒙版内覆盖率的 min/mean/max；
- 各阶段有效数与最差过滤保留率；
- 小孔、大开口、边界缺失、父层恢复数；
- 每帧源视图数；
- TSDF 单视图/多视图支持和全部拒绝计数；
- 填孔前后边界环与新增面数；
- 模型 coverage、IoU、edge P90、SSIM。

脚本只写入新的时间戳回归目录，不覆盖或删除现有 `E:\code\test\temple` 产物。

### 6.2 同蒙版视觉检查

固定检查至少三类图：

- 原照片 + 项目蒙版；
- Level 2 与最终 Level 1 的有效掩码；
- 修复前后同相机位姿模型截图。

与 Metashape 比较时明确区分：由 PlaScan 项目蒙版排除的地面不算算法孔洞；蒙版内部缺失才进入本任务指标。

### 6.3 UAV9 防回归

Temple 达标后再跑既有 UAV 9 图回归，确认：

- 每张图 SIFT 关键点不超过 40000；
- 不因单视图 TSDF 支持策略引入地形漂浮片；
- 蒙版为空/全图有效时新完整性指标仍正确；
- 模型和深度质量不低于保存的 UAV 基线。

---

## Task 7：构建与验证顺序

当前 `E:\code\plascan\build\windows-vcpkg-cuda-release\bin\plascan.exe` 正由 PID 50140 运行。实现阶段可以先改代码和构建不锁定的测试目标，但在链接 `plascan_gui` 前必须由用户正常关闭应用；不强制结束进程，不删除被占用产物。

### 7.1 测试先行构建

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release -j 3 --target `
  test_mvs_depth_completeness `
  test_mvs_depth_pyramid `
  test_mvs_pipeline `
  test_mvs_workspace_manifest `
  test_mesh_reconstructor `
  test_gui_project_utils `
  test_cli_contracts
```

### 7.2 定向测试

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release\tests `
  -C Release --output-on-failure `
  -R "DepthCompleteness|MvsDepthPyramid|MvsPipeline|MvsWorkspace|DepthTsdf|MeshWorkflow|DepthOverlay|DepthWorkspace|CliContract"
```

### 7.3 用户指定的重建测试

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release -j 3 --target `
  test_aerial_triangulation_workflow `
  test_mvs_depth_pyramid `
  test_mvs_types `
  test_mvs_pipeline `
  test_mvs_rectifier_unit `
  test_cli_contracts
```

逐个运行并简短记录通过数和失败测试名，不粘贴大段日志。

### 7.4 GUI 构建和手工验证

用户正常关闭 PID 50140 后：

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release -j 3 --target plascan_gui
```

打开 Temple 验证聚合深度节点、删除入口、叠加按钮、照片切换、Final/L1/L2 和带原因禁用的 L3。

### 7.5 Temple 与 UAV 回归

使用既有 `run_depth_overlay_regression.ps1` 参数和 CUDA/cuDNN 环境，新增时间戳输出目录。先比较深度完整性和 TSDF 诊断，再比较模型质量；只有两类指标都通过才判定任务完成。

---

## 实施顺序与停线条件

1. 先完成 Task 1、Task 2，只增加诊断和 UI 原因，不改变深度数值。
2. 用新诊断重跑一次 Temple，精确确认 `166,709` 像素分别在哪个阶段丢失。
3. 只有证据确认后才执行 Task 3；若主要损失不是输出前局部过滤，按实际阶段调整修复点，不把补丁硬塞进 `removeLocalDepthOutliers()`。
4. 深度完整性通过后执行 Task 4，再单独测量 TSDF 支持变化。
5. 最后执行 Task 5 的小孔填充；这样可以区分“深度恢复”与“网格封小孔”各自贡献。
6. 任一阶段出现大开口被填、蒙版外产生深度、漂浮组件增加或质量跌破硬性无回退线，立即停止扩大修复范围，保留诊断并回到该阶段的判定规则。
7. 全程不提交、不推送 GitHub。
