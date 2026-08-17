# PlaScan 深度图估计、融合与渲染正确性优化计划

更新日期：2026-08-12

状态：第一轮正确性优化、合成回归和本机真实数据验证已完成；OpenCL 性能、Temple 门控和后续阶段实施中

适用范围：`src/core/mvs`、`src/core/qc`、深度图 GUI 覆盖层、对应 CLI/GUI 与回归测试

## 1. 背景与结论

PlaScan 当前 MVS 主流程已经具备成熟开源方案常见的主体结构：多尺度 PatchMatch、平面假设、
多源光度代价、跨视几何一致性、深度修复、质量门控和点云融合。与 COLMAP PatchMatch Stereo、
OpenMVS Dense Reconstruction 和 AliceVision DepthMap 的公开实现相比，主要问题不是“缺少一个全新的
深度网络”，而是若干坐标、数值和证据语义没有在各阶段保持一致。这些问题会在会聚相机、大地坐标、
畸变镜头、深度断层和低纹理场景中放大，足以造成系统性深度偏差、孔洞、边界污染或质量门控误判。

本轮按“先保证几何正确，再比较精度和性能”的顺序实施。任何提速、补洞或默认参数调整，都不能绕过
本计划中的坐标契约、无效值契约和真实多视证据门禁。

### 1.1 对照基线与采用原则

- [COLMAP dense reconstruction](https://colmap.github.io/tutorial.html#dense-reconstruction)：以
  `image_undistorter -> patch_match_stereo -> stereo_fusion` 明确区分工作像素域、深度估计和几何融合；PlaScan
  本轮据此检查“去畸变图像/掩膜同域”和“最终融合只消费几何一致证据”。
- [COLMAP MVS source](https://github.com/colmap/colmap/tree/main/src/colmap/mvs)：作为 PatchMatch、深度图融合、
  相机投影与输入校验的代码级对照，不机械复制其面向普通近景坐标的默认参数。
- [OpenMVS MVS source](https://github.com/cdcseacave/openMVS/tree/develop/libs/MVS)：作为深度工件、邻接视图、
  深度过滤和多视融合一体化流程的对照；PlaScan 保留自身的行星场景质量门控与流式缓存设计。
- [AliceVision depthMap source](https://github.com/alicevision/AliceVision/tree/develop/src/aliceVision/depthMap)：作为
  多尺度深度/相似度图、GPU 深度估计和后处理职责划分的对照。

这些项目证明 PlaScan 的总体路线是合理的；本轮采纳的是可验证的几何/证据契约，不以“参数看起来相近”作为
正确性证明。行星大坐标、工程蒙版、缓存复用和 CPU/CUDA/OpenCL 三后端一致性仍需 PlaScan 自己的专门测试。

## 2. 优化目标与非目标

### 2.1 目标

- 在原始相机、去畸变相机和极线校正相机之间保持同一个三维点及其 `Z_cam` 深度语义。
- 在百万米级世界坐标下保持与局部坐标等价的相对位姿和 PatchMatch 结果。
- 深度滤波不让无效零值进入统计，且对米、毫米等整体尺度变换保持协变。
- 每像素支持数表示真实通过门限的视图或几何证据，不再用候选源数量代替。
- 质量门控只消费实际存在的证据；跨视一致性尚未计算时不得伪造分数。
- 保存、重新加载、融合和 GUI 展示始终使用与深度图同域的相机和图像。
- CPU、CUDA、OpenCL 后端在统一夹具上满足明确精度门限，性能优化不以降低精度为代价。

### 2.2 非目标

- 本轮不把传统 PatchMatch 整体替换为单目或端到端神经深度网络。
- 不以插值、网格补洞或视觉平滑掩盖几何不一致。
- 不删除或重生成 `testData/`、`.plascan` 项目和 `resources/models/`。
- 不进行与本问题无关的 GUI、MVS 超大文件或模块目录机械重构。

## 3. 已确认问题与优先级

| ID | 优先级 | 问题 | 直接后果 | 主要位置 |
|---|---:|---|---|---|
| GEO-01 | P0 | 极线校正后只把深度标量按像素搬回原图 | 相机旋转时 `Z_cam` 系统性错误 | `EpipolarRectifier`、`DepthMapGenerator` |
| GEO-02 | P0 | 去畸变只作用于图像，没有同步重映射有效掩膜 | mask、纹理和深度坐标错位 | `MvsImagePreprocessor`、`DepthMapGenerator` |
| NUM-01 | P0 | CPU/CUDA 先把大世界平移降为 float，再计算相对位姿 | 行星尺度下小基线丢失 | `PatchMatchCPU`、`PatchMatchCUDA` |
| FIL-01 | P0 | median/bilateral 把无效零深度当真实样本，且固定绝对深度 sigma | 边界塌陷、孔洞污染、单位敏感 | 三个 PatchMatch 后端 |
| EVD-01 | P1 | `supportCount` 写成源图数量 | 融合把候选视图误当真实支持 | `DepthPyramidEstimator`、`DepthMapFusion` |
| EVD-02 | P1 | 几何检查前用置信度推算“多视一致性” | 质量门控可自我拒绝本应验证的帧 | `DepthMapGenerator`、`DepthFrameQualityGate` |
| ART-01 | P1 | 保存工件后融合可能改用当前工程相机 | 缓存重用结果与首次运行不一致 | `DepthFrameUtils`、融合帧构造 |
| REN-01 | P1 | 网格深度在屏幕空间线性插值 | 斜三角形遮挡和深度值错误 | `ModelMeshRenderer` |
| REN-02 | P1 | GUI 与保存预览采用不同分位数、方向和坐标底图 | 同一深度结果显示不一致 | `DepthOverlayData`、预览保存逻辑 |
| PM-01 | P1 | CUDA checkerboard 搜索预算显著低于 legacy 路径 | 合成平面已有可重复精度回退 | `PatchMatchCUDA`、配置默认值 |
| DEM-01 | P2 | 深度直接转 DEM 时把图像行列当规则世界 XY | 透视/斜视场景地形几何错误 | `DemGeneratorFromDepth` |
| IO-01 | P2 | 浮点深度显示和部分产物缺少完整单位/域元数据 | 跨工具交换时容易误解释 | workspace manifest、GUI |

## 4. 不可破坏的公共契约

### 4.1 深度和像素域

- 深度统一定义为当前 `cameraModel` 坐标系中的正向 `Z_cam`，不是射线距离、世界高程或视差。
- 无效深度统一为 `0`；任何统计和滤波必须同时检查 `isfinite(depth) && depth > 0` 与有效掩膜。
- 像素坐标使用 OpenCV 像素中心约定。原始畸变域、去畸变域和校正域必须由显式元数据区分。
- 深度从相机 A 转到相机 B 时，必须执行 A 反投影、世界变换和 B 投影，不能复制标量。
- `zNear/zFar` 必须属于执行 PatchMatch 的工作相机，且包围该相机下全部有效先验深度。

### 4.2 多视证据

- “候选源数”“光度支持数”“几何支持数”和“可观测源数”是四个不同量，字段和报告不得混用。
- 未计算的指标用明确的 unavailable 状态表示，不能由另一指标线性推算。
- 融合使用的支持门限必须对应真实逐像素证据；预筛选只决定是否进入几何检查，不能充当最终判定。

### 4.3 数值和尺度

- 世界坐标到局部相机坐标的相减和相对位姿组合在 double 中完成，只把已局部化的结果下转换为 float。
- 深度平滑参数使用相对深度、逆深度或由有效深度统计量归一化的量；将场景整体放大 `s` 后，输出也应放大
  `s`，有效掩膜和支持关系保持不变。
- 后端优化必须以同一参考语义为准。CUDA/OpenCL 不得为了吞吐量暗中减少搜索范围或质量迭代预算。

## 5. 分阶段实施

## 阶段 0：固定基线、测试夹具和缓存边界

1. 保存当前 CPU、CUDA、OpenCL 的配置、覆盖率、均值、RMSE、耗时和显存/内存峰值。
2. 建立最小确定性夹具：
   - 旋转的校正相机和平面；
   - 径向畸变图像及硬边界 mask；
   - 原点与平移 `1e8` 后完全等价的小基线相机组；
   - 带孔洞和 1:10 深度断层的深度图；
   - 单源、两源、冲突源和遮挡源支持图；
   - 非恒定 Z 的斜三角形。
3. 将深度算法修订、滤波语义和工件坐标域纳入 workspace hash/manifest。旧结果允许查看，但在新流程中必须
   显式失效或迁移，不能被无提示复用。

阶段门槛：每个已确认问题都有修复前失败、修复后通过的行为测试；测试不依赖随机种子和 GPU 调度顺序。

## 阶段 1：修复几何坐标契约

### 1.1 极线校正深度回投

- 新增明确的 `remapDepthBetweenCameras` 或等价职责：对校正域有效像素，用校正相机反投影为三维点，变换到
  原始参考相机，投影到原始像素，并写入原始 `Z_cam`。
- 多个校正像素落到同一原始像素时使用 z-buffer 选择最近有效表面；空隙保持无效，不做未经授权的插值。
- 置信度、支持数和 provenance 使用同一像素映射；它们是标量属性，不执行深度坐标变换。
- 从稀疏点或原深度范围端点的三维包络计算校正相机 `zNear/zFar`，增加有限安全边界并验证 `0 < near < far`。
- 旧的只按 homography 搬运深度函数重命名为 scalar-map 或限制为 mask/confidence，防止再次误用。

验收：旋转校正相机 roundtrip 的三维误差和原始 `Z_cam` 相对误差均不超过 `1e-5`（double 参考；最终 float
输出允许 `5e-5`）。纯平移/同旋转场景保持当前结果。

### 1.2 图像与掩膜同步去畸变

- 预处理接口同时接收图像和 valid mask，并复用完全相同的 map；图像按既有插值，mask 使用
  `INTER_NEAREST`、边界值 0。
- 返回结构携带工作图像、工作 mask、工作相机和像素域；禁止调用者混搭原始 mask 与去畸变相机。
- 支持 mask、有效深度 mask、稀疏提示和调试预览都必须选择同一工作域。
- GUI 如果叠加在原始传感器图像上，则把深度/属性明确重映射回原始畸变域；否则底图也使用去畸变图像。

验收：合成径向畸变边界的 image/mask 轮廓偏差不超过 1 像素，mask 只有 0/255，不出现插值灰边。

## 阶段 2：PatchMatch 数值稳定性和后处理

### 2.1 大坐标相对位姿

- 提取共享的 double 精度相对位姿构造逻辑：`R_rel = R_src * R_ref^T`，平移由 double 相机中心或 double
  world-to-camera translation 组合；只在得到局部量后转换为后端 float 数据。
- CPU、CUDA、OpenCL 使用同一公式、方向和单元测试，删除各自易漂移的重复实现或至少共享验证 helper。
- 日志在基线退化为零、非有限或远小于标称基线时给出明确诊断，而不是继续生成全空/全平深度。

验收：所有相机中心整体平移 `1e8` 后，相对位姿误差不超过 `1e-6`；合成深度有效 mask 完全一致，深度
RMSE 不超过场景深度的 `1e-5`。

### 2.2 无效值感知且尺度稳定的滤波

- median 只统计有效邻域；有效样本不足时保留中心值或标为无效，不能由零值投票覆盖有效中心。
- bilateral 在归一化逆深度或相对深度上计算 range 权重，只对有效邻域归一化；空间 sigma 仍使用像素单位。
- 深度断层处使用相对差门限，禁止前后景互相渗透；滤波后更新或保守降低置信度，不能沿用过高原置信度。
- 三个后端共享参数含义和边界行为。若某后端暂时无法正确实现，安全回退是禁用该滤波并记录原因，而不是
  继续使用会污染深度的旧实现。

验收：有效值不会被无效零值变成 0；空洞不会被普通平滑擅自填充；场景整体缩放 1000 倍后，归一化结果
MAE 小于 `1e-5`；1:10 断层两侧误差均小于 1%。

## 阶段 3：真实支持度和两级质量门控

### 3.1 支持计数语义

- PatchMatch 代价聚合返回获胜假设真正通过门限的 source count，必要时保存紧凑 view bitset。
- 将当前“所有源数量”字段改名为 `candidateSourceCount`；只在兼容读旧工件时保留解释。
- 跨视验证产出的 `geometrySupportCount` 作为最终融合的主要支持字段，光度支持只作为补充质量信息。
- 计数上限、无源、单源、多源冲突和超过 255 个源的饱和/存储行为必须明确测试。

### 3.2 两级质量门控

- pre-geometry gate 只检查结构性失败：尺寸/相机无效、有限深度覆盖、光度置信、支持区域和资源异常。
- 跨视一致性在几何阶段实际计算；不存在时为 unavailable，而不是 `confidence * 1.15`。
- final gate 使用实际几何覆盖、支持分布、冲突率、遮挡率和分位数。统计分别记录 prefilter candidate、
  observable、accepted 和 rejected，不能用不可观测像素稀释冲突率。
- 被 pre-gate 拒绝但结构上可运行的帧可进入 validation-only 池，避免它永远无法获得几何证据。

验收：低光度但几何一致的帧能进入最终判定；高置信但几何冲突的帧被拒绝；帧遍历顺序不影响结果。

## 阶段 4：工件、融合和渲染语义

### 4.1 工件相机和路径

- 深度工件保存完整工作相机、原始相机、像素域、深度单位、算法修订和映射信息。
- 从工件构造融合帧时优先使用工件内相机；只有旧 schema 明确缺失时才采用受控兼容路径并记录警告。
- GUI/Controller 对相对 artifact path 统一以项目文件目录解析，并把已解析的绝对路径传给加载器。
- 读取工件时验证相机尺寸、深度矩阵尺寸、mask 尺寸和坐标域一致，不一致立即失败。

### 4.2 网格深度渲染

- 三角形深度采用 reciprocal-Z 插值：`Z = 1 / sum(lambda_i / Z_i)`；透视属性使用相同分母校正。
- 增加 near-plane clipping 或在光栅化前裁剪穿过近裁面的三角形，避免整片丢弃或产生无穷深度。
- GUI 和保存预览统一颜色方向、分位数策略、无效色、单位和图例。批次比较允许锁定同一显示范围。
- 非有限值在统计、着色和保存前统一排除。

验收：斜三角形解析深度误差不超过 `1e-5`；遮挡顺序与解析投影一致；同一工件在 GUI 和保存预览中的
近远颜色、有效 mask 和范围一致。

## 阶段 5：后端精度、性能和默认策略

- CPU 初始化统一采用逆深度或与 GPU 等价的采样分布，近景/远景获得相近相对分辨率。
- 将 `numIterations` 拆为清晰的初始化采样预算和传播迭代数；checkerboard 路径不得把 16 次迭代静默压成 4 次。
- 为 plane/fronto-parallel、斜平面、深度阶跃、低纹理和多源遮挡建立 CPU/CUDA/OpenCL parity。
- 在质量门槛满足后比较 legacy 和 checkerboard 的吞吐、峰值显存和确定性，再决定默认值。
- 默认值变更必须有真实场景 A/B，不能只依据合成 coverage 或单次耗时。

验收建议：

- 无噪声正面平面有效区 RMSE 小于深度范围的 `0.5%`，均值偏差小于 `0.2%`。
- CPU/CUDA/OpenCL 的共同有效区相对深度 P95 差异小于 `1%`，有效率差异小于 2 个百分点。
- checkerboard 精度不低于 legacy 门槛；若未达到，保持默认关闭或自动回退。
- 固定种子重复三次，摘要指标差异不超过浮点容差。

## 阶段 6：DEM 和长期数据格式收口

- 深度转 DEM 必须先按相机模型反投影为世界点，再投影/栅格化到目标 CRS；不能把图像行列直接当世界 XY。
- 输出 CRS、垂直基准、深度单位和 nodata 进入元数据，并由 GDAL 读取回验。
- 对正射、斜视、旋转相机和非方形像素分别建立 DEM 夹具。
- 完成旧 workspace 工件迁移策略、兼容期和清理工具；迁移必须可恢复，不能覆盖唯一旧结果。

## 6. 验证矩阵

开发中先构建受影响测试，避免触碰正在运行的 GUI 可执行文件：

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mvs_pipeline
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_patchmatch_cpu
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_model_image_quality
```

定向回归：

```powershell
python scripts/env/run_tests.py `
  --test-dir build/windows-vcpkg-cuda-release `
  --output-on-failure `
  -R 'Mvs|PatchMatch|Depth|Geometry|Fusion|Overlay|MeshRenderer'
```

CUDA 设备可用时额外运行 CPU/CUDA parity、checkerboard 与 legacy 精度用例。准备提交或推送前，按仓库门禁
完成 Windows/MSVC 全量可执行测试；Linux/GCC 由对应本地环境或 GitHub required checks 验证，不能把单平台
结果表述为双平台通过。

### 6.1 真实数据 A/B

至少选择以下三类现有、可重复场景，固定输入、相机、源图选择和随机种子：

1. 小型近景模型：检查边缘、斜面和遮挡。
2. 行星/轨道影像：检查大坐标、弱纹理、辐射差异和长基线。
3. 带明显镜头畸变的地面数据：检查 image/mask/depth 域一致性。

记录每帧和批次的有效覆盖、光度支持、几何支持、冲突/遮挡、深度 RMSE 或重投影误差、融合点数、
完整性、峰值内存和运行时间。正确性阶段不得出现以下回退：

- 几何一致像素 P95 重投影误差上升超过 `0.1 px`；
- 人工真值/激光可用时，深度 RMSE 上升超过 1%；
- 无解释的有效覆盖下降超过 2 个百分点；
- 非有限值、负深度或 mask 外有效深度数量非零。

## 7. 发布、回退和可观测性

- 每个阶段独立提升算法或工件 revision，避免旧缓存和新语义混用。
- 新路径先保留可控开关；回退只能回到上一个语义完整的实现，不能回到已确认会破坏深度的滤波或标量搬运。
- 日志和质量报告至少包含像素域、相机来源、near/far、候选/光度/几何支持分布、无效原因、后端、滤波方式
  和算法修订。
- 任一帧出现非有限相机、无效深度范围、零化基线或工件尺寸不一致时快速失败并给出文件/帧/参数，禁止静默
  降级为全空结果。
- 默认参数只有在合成门禁和三类真实数据 A/B 均通过后才调整。

## 8. 实施顺序与当前进度

1. **已完成第一轮实现与合成回归**：GEO-01、GEO-02、NUM-01、FIL-01、EVD-02、ART-01、REN-01。
2. **已完成最终融合证据收口**：融合输入与门限只使用含参考帧的真实
   `geometrySupportCount`；内存、stored/GUI 和 CLI 流式入口统一，revision 34 缺公共几何证据时拒绝复用。
   PatchMatch 原 `supportCount` 暂保留为候选来源诊断，不再参与最终融合；真正的逐像素光度支持仍属于后续
   kernel ABI 优化，不在本轮冒险扩展三个后端。
3. **已完成质量证据可用性修复**：几何检查前不再用光度置信度伪造多视一致性；连续证据统计排除
   reference-only 像素，纯冲突观测仍计入可观测集合。
4. **已完成渲染第一轮**：reciprocal-Z 与透视属性插值、穿越相机平面的 near-plane clipping、投影坐标
   转整数前的有限性和视口夹紧均已实现；深度覆盖层相对路径已统一解析。
5. **真实数据 A/B 已完成第一轮**：Hyb2 严格 paired A/B 的覆盖和独立参考尾部精度改善，但旧结果稳定性
   P95 为 1.7387%，且 OpenCL 批次耗时回退 4.73%；Temple 和 Agisoft 的真值范围仍有限。因此暂不调整
   PM-01 checkerboard 搜索预算或生产默认值。
6. **后续阶段**：冻结第三方参考的共同坐标尺度，恢复 OpenCL 性能，校准 Temple 融合门控，补真实畸变
   hard-mask/GT，然后实施 REN-02、DEM-01、IO-01 和旧工件迁移。

### 8.1 2026-08-12 第一轮验证记录

- MSVC 构建通过：MVS 核心、`reconstruct_pipeline_cli`、PatchMatch、极线/预处理、质量证据、stored 回放、
  模型深度渲染和 GUI 项目工具相关目标。
- `test_mvs_pipeline`：95/95 通过。
- `test_mvs_streaming_depth_fusion`：4/4 通过。
- `test_mvs_workspace_manifest`：25/25 通过。
- `test_model_image_quality`：17/17 通过。
- `test_patchmatch_cpu`：10/10 通过，另有 1 项既有 CUDA 精度测试仍为 disabled。
- `test_mvs_image_preprocessor`：4/4 通过；`test_mvs_rectifier_unit`：13/13 通过；
  `test_mvs_adaptive_geometry_evidence_policy`：10/10 通过。
- `src/core/mvs` CTest：397/397 通过，覆盖 MVS、Depth、Geometry、Fusion、PatchMatch CPU/CUDA/OpenCL
  对齐与缓存策略等已登记用例。
- `test_gui_project_utils`：470 通过、3 跳过；跳过原因为外部 OBJ benchmark 数据未配置及 Qt offscreen
  不支持原生弹出菜单，与本轮算法改动无关。
- 深度 TSDF、融合帧策略和深度网格定向回归：80 通过、1 跳过；跳过项需要未配置的 MC33 依赖。
- 三类真实数据与固定网格 renderer 验证已在 8.2 节补充；Linux/GCC 和仓库全量测试仍未执行。仓库根级
  测试发现阶段还会被与本轮无关的运行时 DLL 缺失阻断。因此不得把本轮结果表述为完成全部优化或跨平台验证。

每一步先提交行为测试与最小实现，再跑定向构建/测试。若某项依赖尚未可用，保持显式关闭并记录阻塞，
不通过放宽门限、填洞或伪造证据宣称完成。

### 8.2 2026-08-12 真实数据验证记录

完整记录见 `docs/benchmarks/2026-08-12-depth-rendering-real-data-validation.md`。

- Hyb2 revision 27→34 是 14 帧同相机、同源图、同参数的严格 A/B：coverage `+0.6200 pp`，非法深度为 0，
  mask IoU `0.95013`；相对旧版深度 P95 `1.7387%`，未通过 1% 稳定性目标。
- 相对 Metashape 2.3.1 原生深度，pooled P90 `2.1223%→1.4562%`、P99
  `10.1176%→1.5153%`；诊断性尺度对齐后的形状 P90 `0.8000%→0.2396%`。Metashape/PlaScan 相机
  中心距离尺度 `1.014386` 与深度对齐比例 `1.014085` 只差 `0.0297%`，原始 1.39% 深度比不是 PlaScan
  固有尺度错误的证据。
- Temple 16/16 和 Agisoft Brown 畸变 9/9 均完成。Agisoft revision 34 深度成功回读并融合为
  1,944,525 个原始点、1,898,824 个精炼点；三组 provenance 均无未分类有效像素。
- 固定 Temple 网格和 16 个真实相机重放后，有效 mask、coverage、IoU 和 edge 指标与历史结果完全一致；
  透视属性修正改变了颜色图，SSIM 中位数仅变化 `+2.84e-5`。reciprocal-Z 数值正确性仍由解析合成测试承担。
- Hyb2 OpenCL 批次耗时 `170.022 s→178.067 s`，回退 4.73%；在恢复性能并补齐参考坐标门禁前不改变默认策略。

## 9. 完成定义

- P0/P1 问题均有独立回归测试，修复前可复现、修复后稳定通过。
- 原始、去畸变、校正、保存、重载和融合阶段的深度/相机/像素域契约一致。
- 大坐标、不同深度单位、孔洞和深度断层夹具满足本计划门槛。
- 支持计数和质量报告来自真实逐像素证据，未计算指标不会被伪造。
- CPU/CUDA/OpenCL 达到统一精度门槛；默认加速路径有可重复的速度收益且没有显著质量回退。
- 相关 MSVC 定向测试和本地全量门禁通过；Linux/GCC/CUDA 的实际验证范围被准确记录。
- `docs/PROJECT_ARCHITECTURE.md`、workspace/模型说明和发布文档与最终行为一致。
