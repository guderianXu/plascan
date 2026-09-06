# PlaMatch-HCT 匹配算法接入 PlaScan 的模块计划

> 状态：CPU/CUDA/OpenCL 主链、任务缓存、coarse 预选、批量匹配、CLI/GUI 已实施
>
> 日期：2026-09-03
>
> 正式名称：`PlaMatch-HCT`；算法 ID：`plamatch_hct`
>
> 算法来源：用户自有且已经可以正常运行的
> `/home/guderian/code/metashape_code/对齐照片`

实施说明：为优先保持用户已验证算法的行为，核心 `features.cpp`、`matching.cpp` 以隔离的
`plamatch_hct/vendor/` 源码边界接入，PlaScan 侧通过 OpenCV 图像桥、`IFeaturePayload` 和算法门面适配；
没有把参考工程的几何、重建或相似变换模块带入。CPU 使用预建 HCTree；CUDA/OpenCL 使用任务级
full/coarse 描述子批量驻留，显式设备失败不做静默回退。

当前验证结果：CPU 基线在仓库双图每图 2,000 点时得到 957 个初始匹配、925 个 USAC 内点和 925 条轨迹；
GPU 接入后使用 1,000 点/1,600 px 的同一航测双图，CUDA 与 OpenCL 均得到 241 个初始匹配、237 个 USAC
内点，CPU 为 242/238。五图 CUDA 任务的 coarse batch 从 10 对保留 4 对，正式 batch 完成 10 对匹配。

## 1. 直接结论

算法主体应加入：

```text
src/core/image_matching/plamatch_hct/
```

这里负责从影像生成 MLDB 特征，以及使用 HCTree、双向 ratio、唯一性门控和局部一致性完成特征匹配。

与 PlaScan 工作流的衔接放在：

```text
src/core/matchphototask/
```

这里不重复实现算法，只负责特征缓存、候选像对、并行调度、结果缓存、几何验证和轨迹构建。

参考工程中的 `geometry.cpp`、`reconstruction.cpp` 和 `similarity.cpp` 不应随匹配算法一起放进
`image_matching`。PlaScan 已有几何验证、SfM 和轨迹模块，本次只接入 `features.cpp` 与 `matching.cpp`
对应的能力。

## 2. 模块分配

| 参考工程能力 | PlaScan 目标模块 | 说明 |
|---|---|---|
| LoG 金字塔和关键点检测 | `src/core/image_matching/plamatch_hct/` | 属于算法特征前端 |
| 36-bin 主方向 | `src/core/image_matching/plamatch_hct/` | 与 detector identity 一起保留 |
| 498-bit/64-byte MLDB | `src/core/image_matching/plamatch_hct/` | 以 `CV_8U` 描述子接入 |
| 2,048/40,000 两阶段空间选点 | `src/core/image_matching/plamatch_hct/` | 同时生成 coarse/full 两套特征流 |
| CPU HCTree | `src/core/image_matching/plamatch_hct/` | 按影像建树一次，多像对复用 |
| CUDA/OpenCL Hamming | `src/core/image_matching/plamatch_hct/gpu/` | 使用条件编译，不影响 CPU 构建 |
| 双向 top-2、0.8 ratio、目标唯一性 | `src/core/image_matching/plamatch_hct/` | 输出 PlaScan `MatchResult` |
| detector/orientation 合并 | `src/core/image_matching/plamatch_hct/` | 先按原算法合并，再适配通用结果 |
| 局部空间一致性 | `src/core/image_matching/plamatch_hct/` | 在 PlaScan USAC 前执行 |
| generic 粗像对筛选和森林削减 | `src/core/matchphototask/pair_selection/` | 属于任务的候选像对规划 |
| reference/sequential 预选 | 复用现有 `PairSelector` | 不在算法目录重复实现 |
| 特征和索引生命周期 | `src/core/matchphototask/runtime/` | 任务级缓存和内存预算 |
| USAC/MAGSAC 几何验证 | 复用 `GeometryVerifyStage` | 不移植参考工程几何模块 |
| 轨迹生成 | 复用 `TrackBuildStage` | 不移植参考工程重建模块 |
| GUI/CLI 选择入口 | `src/gui`、`src/cli` | 只传算法 ID 和配置 |

## 3. `image_matching/plamatch_hct` 内部结构

建议按职责拆分，避免把参考工程的大型 `features.cpp`、`matching.cpp` 原样合成两个超长文件：

```text
src/core/image_matching/plamatch_hct/
├── PlaMatchHctAlgorithm.h/.cpp
├── PlaMatchHctFeatureExtractor.h/.cpp
├── PlaMatchHctFeaturePayload.h/.cpp
├── GaussianPyramid.h/.cpp
├── LogKeypointDetector.h/.cpp
├── HistogramOrienter.h/.cpp
├── MldbDescriptor.h/.cpp
├── SpatialKeypointSelector.h/.cpp
├── BinaryHcTreeIndex.h/.cpp
├── PlaMatchHctFeatureMatcher.h/.cpp
├── LocalConsistencyFilter.h/.cpp
└── gpu/
    ├── PlaMatchHctDescriptorAccelerator.h
    ├── PlaMatchHctCudaBackend.cu
    └── PlaMatchHctOpenClBackend.cpp
```

核心类职责：

- `PlaMatchHctAlgorithm`：实现 `IImageMatchingAlgorithm`，作为注册表看到的唯一算法门面。
- `PlaMatchHctFeatureExtractor`：组织检测、方向扩展、MLDB 和两阶段选点。
- `PlaMatchHctFeaturePayload`：保存通用 `FeatureSet` 无法表达的算法私有数据。
- `BinaryHcTreeIndex`：对应参考实现的 `CpuDescriptorIndex`，构建后只读。
- `PlaMatchHctFeatureMatcher`：执行两个方向的查询、ratio、唯一性和方向行合并。
- `LocalConsistencyFilter`：执行 5/10 邻域、局部/全局半径和共同邻居检查。
- GPU backend：只实现描述子比较和批量匹配，不承担任务编排。

建议注册信息：

```text
algorithm ID: plamatch_hct
display name: PlaMatch-HCT（空间一致性二进制匹配）
input model: ReusableFeatures
stable feature IDs: true
```

完成 CPU/CUDA/OpenCL、双图和多图验证后，`plamatch_hct` 已设为 GUI、CLI、MatchPhotosTask 与空三流程的默认算法；
`auto_sift` 继续作为可显式选择的兼容算法。

## 4. 特征数据如何接入

PlaScan 当前 `FeatureSet` 只有关键点、分数和描述子；完整 PlaMatch-HCT 匹配还需要：

- `detector_id`：方向扩展前的检测点身份。
- `source_id`：方向扩展后的描述子行身份。
- `laplacian_sign`：描述子匹配分区。
- `coarse_keypoints`：独立选出的 2,048 点粗匹配流。
- full/coarse HCTree 索引。

建议给 `FeatureSet` 增加一个通用、只读的算法扩展接口：

```text
IFeaturePayload
├── schemaId()
├── approximateBytes()
└── virtual destructor

FeatureSet
└── shared_ptr<const IFeaturePayload> payload

PlaMatchHctFeaturePayload : IFeaturePayload
├── detectorIds
├── sourceIds
├── laplacianSigns
├── coarseKeypoints/coarseDescriptors
├── fullIndex
└── coarseIndex
```

这样做比把 PlaMatch-HCT 专用字段直接塞入所有算法共用的 `FeatureSet` 更干净，也不用用 `cv::KeyPoint::class_id`
偷偷编码业务语义。`FeatureSet::isConsistent()` 和 `approximateBytes()` 需要把 payload 的校验与内存计入。

full/coarse 索引建议在特征提取结束时构建，或通过 `std::once_flag` 延迟构建。无论采用哪种方式，都必须做到
每幅影像只构建一次，不能在每个像对的 `matchFeatures()` 中重复建树。

## 5. 与 MatchPhotosTask 的串联

现有流程无需推翻，调整后仍保持：

```text
FeatureStage
    ↓
算法专用 generic 预选 + 现有 reference/sequential 预选
    ↓
PairSelector
    ↓
MatchingStage
    ↓
GeometryVerifyStage
    ↓
GuidedMatchStage（是否支持另行决定）
    ↓
TrackBuildStage
```

具体修改点如下。

### 5.1 FeatureStage

- 通过注册表创建 `plamatch_hct`。
- 把 PlaScan 已读取的灰度影像、原图尺寸、缩放比例和有效蒙版传给 `extract()`。
- `extract()` 返回 full 特征，coarse 特征和 HCTree 通过 payload 一起进入任务缓存。
- 关键点 ID 使用 full 特征最终输出顺序，保证 `.pifeature`、`.pimatch` 和轨迹中的 ID 一致。
- `featureSchemaVersion` 单独递增，不能复用 ORB/SIFT 的特征 schema。

### 5.2 MatchPhotosFeatureCache

- 继续以规范化影像路径缓存 `shared_ptr<const FeatureSet>`。
- `PlaMatchHctFeaturePayload` 跟随 `FeatureSet` 存活，因此不需要再建立一套全局算法缓存。
- `approximateBytes()` 必须统计 full/coarse 描述子、树节点和 GPU staging buffer。
- 缓存仍限定在一次 MatchPhotosTask 内，任务结束自动释放。

### 5.3 统一 generic 像对预选

参考算法的 coarse 预选不放入通用 `overlap` 库，建议新增：

```text
src/core/matchphototask/pair_selection/PlaMatchHctPairPreselector.h/.cpp
```

它从任务特征缓存读取正式算法结果：PlaMatch-HCT 直接使用
`PlaMatchHctFeaturePayload::coarseFeatures`；其它算法对正式浮点描述子做空间均衡选点和确定性
512-bit 排序签名映射。两种输入都执行：

1. coarse HCTree 匹配。
2. 最低粗匹配数量门控。
3. 临时轨迹及 shared/novel 统计。
4. 最多 20 轮的严格最大生成森林削减。
5. 生成标准 `PairSelectionResult` 候选信号。

在 `ImageMatchingAlgorithmDescriptor` 中增加“提供原生 coarse preselection”的能力字段，
`MatchPhotosAlgorithmPlan` 只传递该能力。`MatchPhotosTask` 的所有 Auto 模式都调用统一预选器；
能力字段只决定复用原生 coarse payload 还是正式浮点描述子适配器。

后续源码恢复确认了目标的真实候选语义，因此正式实现已改为以“对齐照片”代码为准：generic-only 使用全组合；
启用 reference 时，coarse 阶段使用目标的循环 ±24 候选图，再运行 PlaMatch-HCT 森林逻辑。Source/Estimated
随后按最近相机中心增补，坐标集合为空时使用索引邻域；普通目录适配器没有 sequence group 元数据，Sequential
增补为 no-op。所有算法的 Auto 路径不再把旧 PlaScan overlap/词汇召回作为 coarse 硬上限。

### 5.4 MatchingStage

- CPU 路径直接调用 `PlaMatchHctAlgorithm::matchFeatures()`；它从 payload 读取预建 HCTree。
- 输出先保留原算法的方向行合并与局部一致性结果，再转换成 PlaScan `MatchResult`。
- PlaScan 要求每个 source/target 至多出现一次；转换时按照原算法的稳定排序规则建立双向索引。
- 结果继续由现有 `ImageMatchRepository` 写入 `.pimatch`。
- 算法版本、backend、关键点上限、ratio 和影响输出的常量必须进入配置指纹。

GPU 批量路径没有强行塞进逐 pair 接口，统一算法接口已增加可选 batch 方法：

```text
IImageMatchingAlgorithm::matchFeatureBatch(...)
```

`MatchingStage` 检测到该能力后，把冷缓存像对组成批次交给 PlaMatch-HCT；其他算法仍走现有逐 pair 接口。这样可以复用
参考实现的 focal batch、设备 worker 和描述子驻留策略，而不会影响 SIFT/LightGlue。批量接口同时接收协作式取消检查，
在设备子批次之间停止继续提交计算；已经提交的 GPU kernel 不会被强制中断。

原始匹配缓存指纹包含解析后的 backend 和设备名称，避免 CPU、CUDA、OpenCL 或不同设备之间误复用存在数值差异的结果。

## 6. 配置与界面

算法已经可用，因此接入时直接保留其已验证默认参数：

- HCTree：8 棵树、4 分支、叶阈值 200、最多 1,000 checks。
- ratio：严格 0.8。
- coarse/full 上限：2,048 / 40,000。
- 局部一致性：5/10 邻域和至少 3 个共同邻居。

首版只向普通用户暴露：

- 算法：`PlaMatch-HCT（空间一致性二进制匹配）`。
- 设备：自动、CPU、CUDA、OpenCL。
- 关键点上限和每百万像素上限。
- generic/reference/sequential 预选开关。

树数量、分支数和局部一致性内部常量先不放到 GUI。它们属于算法版本的一部分，避免用户组合出未经验证的参数集。

GUI 下拉框继续从 `ImageMatchingRegistry::descriptors()` 自动填充。CLI 当前有算法名称的静态帮助文本，实施时需同步
加入 `plamatch_hct`，并让参数校验最终以注册表为准。

## 7. 分阶段实施顺序

### 阶段 A：CPU 算法核心

1. 创建 `image_matching/plamatch_hct` 目录和 CMake 源文件清单。
2. 接入 LoG、方向、MLDB 和两阶段选点。
3. 增加 `PlaMatchHctFeaturePayload` 和 full/coarse 数据校验。
4. 接入 CPU HCTree、方向合并和局部一致性。
5. 注册 `plamatch_hct`，但暂不加入 GUI 默认选项。

完成标准：算法级测试与参考程序在固定输入上的特征数、描述子、匹配对和顺序一致。

### 阶段 B：MatchPhotosTask 集成

1. FeatureStage 缓存完整 PlaMatch-HCT payload。
2. MatchingStage 使用预建索引，不重复建树。
3. 完成 `MatchResult`、`.pifeature`、`.pimatch` 和缓存指纹适配。
4. 复用 PlaScan 现有几何验证与轨迹构建。

完成标准：双图和小型多图任务可从影像输入走到有效轨迹，缓存冷热运行结果一致。

### 阶段 C：算法专用像对预选

1. 新增 `PlaMatchHctPairPreselector`。
2. 接入 coarse 特征流、临时轨迹和最大生成森林。
3. 在专用预选器内按参考顺序合并 Source/Estimated/Sequential；manual-only 显式绕过自动链。
4. 所有算法的 Auto 模式统一输出 `PairSelectionResult`；非 PlaMatch 算法复用正式特征生成 coarse 视图。

完成标准：South Building 的 coarse 候选与参考一致，三种参考模式和空集回退均有测试，人工指定像对不被改写。

### 阶段 D：CUDA/OpenCL 批量匹配（已完成）

1. 接入设备枚举和后端选择。
2. 增加 batch matching 可选接口。
3. 复用 full/coarse 描述子设备驻留和 16×16 tiled Hamming。
4. 明确 GPU 失败策略，禁止未提示的结果语义切换。

完成标准：CPU/GPU 接受匹配集合满足已有差分标准，设备内存有上限，多 worker 无共享状态错误。

### 阶段 E：GUI、CLI 与生产启用

1. GUI 显示算法、真实 backend 和最终生效参数。
2. CLI 增加算法 ID 与设备参数。
3. 更新模块 README、项目架构和模型/设备说明。
4. 在真实行星数据上与 `auto_sift`、`sift_lightglue`、`loma_r` 做全流程对比。

默认启用已按用户决策完成。注册相机数、有效轨迹、重投影误差、耗时和峰值内存的大规模对比继续作为
跨平台发布验收项，不阻止当前 GUI/CLI 使用 PlaMatch-HCT 默认值。

## 8. 测试计划

### 算法差分测试

- Gaussian/DoG、关键点、方向峰和 MLDB 描述子。
- coarse/full 两阶段选择结果及行顺序。
- HCTree 结构、top-2 索引、Hamming 距离和严格 ratio 边界。
- 方向行合并、目标唯一性和局部一致性输出。
- CPU、CUDA、OpenCL 的匹配集合一致性。

### PlaScan 单元测试

- 新算法注册、能力描述和运行配置。
- payload 类型、完整性、复制生命周期和内存统计。
- 每幅影像只建一次索引。
- 空输入、错误描述子类型、重复点、同距离和 ratio 临界值。
- 多 worker 并发查询的确定性。
- 算法版本和参数变化使旧匹配缓存失效。

### 集成测试

- FeatureStage → 预选 → MatchingStage → GeometryVerifyStage → TrackBuildStage。
- manual/reference/sequential/generic 各预选组合。
- 蒙版、缩放坐标和每百万像素关键点限制。
- `.pifeature`/`.pimatch` 写入、读取和匹配查看器显示。
- GUI、双图 CLI、Match Photos CLI 和空三 CLI 使用同一算法 ID。
- Linux/GCC 与 Windows/MSVC 构建；CUDA/OpenCL 分别做有设备和无设备路径测试。

## 9. 预计修改文件

实施时的主要文件范围是：

```text
src/core/image_matching/CMakeLists.txt
src/core/image_matching/FeatureSet.h/.cpp
src/core/image_matching/ImageMatchingAlgorithm.h
src/core/image_matching/ImageMatchingRegistry.cpp
src/core/image_matching/plamatch_hct/**

src/core/matchphototask/CMakeLists.txt
src/core/matchphototask/algorithm/MatchPhotosAlgorithmPlan.h/.cpp
src/core/matchphototask/pair_selection/PlaMatchHctPairPreselector.h/.cpp
src/core/matchphototask/runtime/MatchPhotosFeatureCache.h/.cpp
src/core/matchphototask/stages/FeatureStage.cpp
src/core/matchphototask/stages/MatchingStage.cpp
src/core/matchphototask/task/MatchPhotosTask.cpp

src/gui/dialogs/application/WorkflowSettingsDialog.cpp
src/cli/features/cli_feature_match.cpp
src/cli/features/cli_match_photos.cpp
src/cli/workflows/ReconstructionCliOptions.cpp
src/cli/workflows/cli_aerial_triangulation.cpp
```

同时增加对应核心、任务、CLI 和 GUI 测试，并在实现完成后更新 `docs/PROJECT_ARCHITECTURE.md`。

## 10. 最终边界

这次接入的定义是：

- 完整使用你的 LoG + orientation + MLDB 特征前端。
- 完整使用你的 HCTree、双向匹配、方向合并和局部一致性。
- 接入你的 coarse generic pair reduction，但放入 PlaScan 的候选像对阶段。
- 继续使用 PlaScan 已有 USAC、轨迹管理和 SfM，避免两套重建链并存。
- 算法主体不依赖 Qt Widgets，GUI/CLI 不包含算法实现。

上述边界已经按阶段 A-E 接入；当前 Linux/GCC 环境完成了 CPU、CUDA、OpenCL、CLI、GUI、缓存和任务级
批处理验证。Windows/MSVC 与更大规模行星数据集的性能、内存和结果质量对比仍作为发布前验证项。
