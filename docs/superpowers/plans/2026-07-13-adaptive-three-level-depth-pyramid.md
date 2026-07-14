# PlaScan 自适应三级深度金字塔实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有两层 PatchMatch 深度估计升级为可诊断、可流式运行的三级深度金字塔，并分别对环拍物体和
航测地形执行场景化源视图、置信度、过滤与帧级质量门控。

**Architecture:** 新增纯核心策略组件 `DepthPyramidPolicy`、`MvsSceneClassifier` 和
`DepthFrameQualityGate`，避免继续扩大 `DepthMapGenerator.cpp`。`DepthMapGenerator` 使用这些组件编排
Level 3/2/1，父层传递深度中心和不确定半径；manifest 保存逐层摘要，GUI 只展示最终生效配置和诊断产物。

**Tech Stack:** C++17、OpenCV、Qt6、CUDA PatchMatch、CMake、GoogleTest。

## Global Constraints

- Level 1/2/3 默认 downsample 为 `D / 2D / 4D`，且每层短边不低于 160 像素。
- 最终质量 `highest/high/medium/low/lowest` 分别对应 `D=1/2/4/8/16`。
- 父层只提供 `depthCenter + depthRadius`，低置信区域必须允许重新搜索，不能硬锁定错误深度。
- 低内存路径仍必须执行 source 邻域多视一致性，不能静默跳过质量检查。
- 默认只持久化 Level 1 与逐层摘要；Debug 配置才保存 Level 2/3 原始栅格。
- 当前工作区已有大量用户改动；只增量修改本计划文件，不重置、不覆盖、不机械格式化其他文件。
- 本轮不创建 commit；只有用户明确要求提交时才执行 Git commit/push。

---

### Task 1: 统一深度质量 profile 与最终生效配置

**Files:**
- Modify: `src/gui/dialogs/DepthMapEstimateDialog.cpp`
- Modify: `src/gui/project/support/ProjectDenseWorkflowConfig.h`
- Modify: `src/gui/project/support/ProjectDenseWorkflowConfig.cpp`
- Test: `tests/test_gui_project_utils.cpp`

**Interfaces:**
- Produces: `DepthQualityProfile`、`depthQualityProfileId()`、`depthQualityDownsample()`。
- Produces: `DenseGenerationSettings::qualityProfile` 使用稳定 ID，不再使用界面文本。

- [x] **Step 1: 写失败测试，覆盖五档质量和显式参数优先级**

```cpp
TEST(DepthQualityProfileTest, MapsStableIdsToFinalDownsample)
{
    using namespace xjw::gui::project;
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Highest), 1);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::High), 2);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Medium), 4);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Low), 8);
    EXPECT_EQ(depthQualityDownsample(DepthQualityProfile::Lowest), 16);
}

TEST(DepthQualityProfileTest, ExplicitSettingsAreNotRaisedByDefaultProfile)
{
    QJsonObject json;
    json[QStringLiteral("qualityProfile")] = QStringLiteral("medium");
    json[QStringLiteral("minViews")] = 3;
    json[QStringLiteral("confidence")] = 0.25;
    const auto settings = xjw::gui::project::denseGenerationSettingsFromJson(json);
    EXPECT_EQ(settings.minViews, 3);
    EXPECT_FLOAT_EQ(settings.patchMatchConfidence, 0.25f);
}
```

- [x] **Step 2: 运行测试并确认因接口缺失而失败**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils -j 8`

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_gui_project_utils.exe --gtest_filter=DepthQualityProfileTest.*`

Expected: 编译失败，提示 `DepthQualityProfile` 或 `depthQualityDownsample` 未定义。

- [x] **Step 3: 实现稳定 profile ID 和五档映射**

```cpp
enum class DepthQualityProfile
{
    Highest,
    High,
    Medium,
    Low,
    Lowest
};

QString depthQualityProfileId(DepthQualityProfile profile);
DepthQualityProfile depthQualityProfileFromId(const QString &profile_id);
int depthQualityDownsample(DepthQualityProfile profile);
```

Dialog 的 combo item data 固定写入 `highest/high/medium/low/lowest`，`collectSettings()` 同时写入
`qualityProfile` 和由 profile 推导的最终 `resScale`。`applyDenseQualityProfile()` 只补充 JSON 未显式提供的字段，
不得用 `std::max` 覆盖用户显式值。

- [x] **Step 4: 运行 profile 和 GUI 合同测试**

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_gui_project_utils.exe --gtest_filter=DepthQualityProfileTest.*:DepthMapEstimateDialog*`

Expected: 所有匹配测试通过。

---

### Task 2: 新增三级层级调度与场景自动判定

**Files:**
- Create: `src/core/mvs/DepthPyramidPolicy.h`
- Create: `src/core/mvs/DepthPyramidPolicy.cpp`
- Create: `src/core/mvs/MvsSceneClassifier.h`
- Create: `src/core/mvs/MvsSceneClassifier.cpp`
- Modify: `src/core/mvs/MvsTypes.h`
- Modify: `src/core/CMakeLists.txt`
- Test: `tests/test_mvs_pipeline.cpp`

**Interfaces:**
- Produces: `DepthPyramidConfig makeDepthPyramidConfig(const PatchMatchConfig &, int, int)`。
- Produces: `MvsSceneClassification classifyMvsScene(const std::vector<CameraView> &, const SparseCloud &)`。

- [x] **Step 1: 写失败测试，覆盖层级缩放和场景分类**

```cpp
TEST(DepthPyramidPolicyTest, BuildsStrictThreeLevelSchedule)
{
    xjw::mvs::PatchMatchConfig base;
    base.downsampleFactor = 1;
    const auto pyramid = xjw::mvs::makeDepthPyramidConfig(base, 800, 640);
    ASSERT_EQ(pyramid.levels.size(), 3u);
    EXPECT_EQ(pyramid.levels[0].level, 3);
    EXPECT_EQ(pyramid.levels[0].patchMatch.downsampleFactor, 4);
    EXPECT_EQ(pyramid.levels[1].patchMatch.downsampleFactor, 2);
    EXPECT_EQ(pyramid.levels[2].patchMatch.downsampleFactor, 1);
}

TEST(MvsSceneClassifierTest, DetectsAerialCameraLayout)
{
    const auto views = makeDownLookingGridViews(3, 3);
    const auto sparse = makePlanarSparseCloud();
    EXPECT_EQ(xjw::mvs::classifyMvsScene(views, sparse).profile,
              xjw::mvs::MvsSceneProfile::AerialTerrain);
}
```

- [x] **Step 2: 运行测试并确认新类型缺失**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mvs_pipeline -j 8`

Expected: 编译失败，提示 `makeDepthPyramidConfig` 和 `classifyMvsScene` 未定义。

- [x] **Step 3: 实现三级策略类型**

```cpp
enum class MvsSceneProfile { Auto, OrbitalObject, AerialTerrain, Custom };
enum class DepthFilterMode { Mild, Moderate, Aggressive };

struct DepthPyramidLevelConfig
{
    int level = 1;
    PatchMatchConfig patchMatch;
    int minSupportViews = 2;
    float radiusScale = 1.0f;
};

struct DepthPyramidConfig
{
    std::array<DepthPyramidLevelConfig, 3> levels;
    MvsSceneProfile sceneProfile = MvsSceneProfile::Auto;
    DepthFilterMode filterMode = DepthFilterMode::Moderate;
    bool saveIntermediateLevels = false;
};
```

层级构造从最终 `D` 生成 `4D/2D/D`，当短边限制使两个层级相同时，降低粗层 downsample 直到三层严格递减；
图像不足以形成三层时返回两层并在 `DepthPyramidConfig::degradedReason` 记录原因。

- [x] **Step 4: 实现场景分类器**

分类器计算相机中心 PCA、平均光轴与稀疏云中心夹角、稀疏云平面厚度比及俯视一致率；只有俯视一致率
`>=0.75` 且平面厚度比 `<=0.20` 时判为 `AerialTerrain`，其余回退 `OrbitalObject`。结果记录四项指标和说明。

- [x] **Step 5: 运行核心测试**

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_mvs_pipeline.exe --gtest_filter=DepthPyramidPolicyTest.*:MvsSceneClassifierTest.*`

Expected: 所有匹配测试通过。

---

### Task 3: 实现父层深度中心、不确定半径和边缘感知传播

**Files:**
- Create: `src/core/mvs/DepthPyramidPropagation.h`
- Create: `src/core/mvs/DepthPyramidPropagation.cpp`
- Modify: `src/core/CMakeLists.txt`
- Test: `tests/test_mvs_pipeline.cpp`

**Interfaces:**
- Produces: `DepthSearchPrior propagateDepthPrior(const DepthLevelResult &, const cv::Mat &, cv::Size)`。
- Consumes: Task 2 的 `DepthPyramidLevelConfig`。

- [x] **Step 1: 写失败测试，证明边缘不被最近邻跨越且低置信区域扩大搜索**

```cpp
TEST(DepthPyramidPropagationTest, PreservesDepthStepAndExpandsLowConfidenceRadius)
{
    const auto parent = makeTwoPlaneDepthLevel(8, 6, 10.0f, 20.0f, 0.9f, 0.2f);
    const cv::Mat guide = makeVerticalEdgeGuide(16, 12);
    const auto prior = xjw::mvs::propagateDepthPrior(parent, guide, guide.size());
    EXPECT_LT(prior.center.at<float>(6, 7), 12.0f);
    EXPECT_GT(prior.center.at<float>(6, 9), 18.0f);
    EXPECT_GT(prior.radius.at<float>(6, 12), prior.radius.at<float>(6, 3));
}
```

- [x] **Step 2: 运行测试并确认函数缺失**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mvs_pipeline -j 8`

Expected: 编译失败，提示 `propagateDepthPrior` 未定义。

- [x] **Step 3: 实现数据类型和传播算法**

```cpp
struct DepthLevelResult
{
    int level = 1;
    int downsampleFactor = 1;
    cv::Mat depth;
    cv::Mat confidence;
    cv::Mat supportCount;
    cv::Mat uncertainty;
    cv::Mat validMask;
};

struct DepthSearchPrior
{
    cv::Mat center;
    cv::Mat radius;
    cv::Mat validMask;
};
```

中心使用联合双边上采样；半径为 `max(parent_uncertainty, local_gradient_radius)`，再按
`1 / max(confidence, 0.1)` 放大。无父深度区域保持 mask=0，由下一级 PatchMatch 使用全局区间或稀疏种子重启。

- [x] **Step 4: 运行传播和现有 sparse hint 回归测试**

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_mvs_pipeline.exe --gtest_filter=DepthPyramidPropagationTest.*:MvsPipelineTest.ProjectedSparseSamplesFeedHintAndSupportReuse:MvsPipelineTest.SparseSeedDepthOverlayDoesNotPropagateAcrossFineHint`

Expected: 所有匹配测试通过。

---

### Task 4: 将三级估计接入 DepthMapGenerator

**Files:**
- Create: `src/core/mvs/DepthPyramidEstimator.h`
- Create: `src/core/mvs/DepthPyramidEstimator.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.h`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/CMakeLists.txt`
- Test: `tests/test_mvs_pipeline.cpp`
- Test: `tests/test_source_contracts.cpp`

**Interfaces:**
- Produces: `DepthPyramidResult DepthPyramidEstimator::estimate(const DepthPyramidRequest &)`。
- `DepthMapGenerator::computeDepthForView()` 消费结果的 Level 1，并把逐层摘要放入 `DepthFrameResult`。

- [x] **Step 1: 写失败测试，验证三级执行顺序和中间层不会长期驻留**

```cpp
TEST(DepthPyramidEstimatorTest, RunsCoarseMiddleFineAndReturnsFinalLevel)
{
    RecordingPatchMatchBackend backend;
    xjw::mvs::DepthPyramidEstimator estimator(&backend);
    const auto result = estimator.estimate(makeSyntheticPyramidRequest());
    EXPECT_EQ(backend.downsampleCalls(), (std::vector<int>{4, 2, 1}));
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.finalLevel.level, 1);
    EXPECT_EQ(result.levelSummaries.size(), 3u);
}
```

- [x] **Step 2: 运行测试并确认 estimator 缺失**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mvs_pipeline -j 8`

Expected: 编译失败，提示 `DepthPyramidEstimator` 未定义。

- [x] **Step 3: 抽象 PatchMatch 后端并实现三级编排**

`IPatchMatchBackend::estimate()` 接收参考图、源图、相机、全局深度区间、层级配置和可选 `DepthSearchPrior`。
生产实现复用 `PatchMatchDepthEstimator::estimate()`；测试实现只记录调用。每一级完成后立刻计算摘要，除非
`saveIntermediateLevels=true`，否则父层像素仅保留到下一级 prior 构造完成。

- [x] **Step 4: 替换旧 coarse/fine 分支**

删除 `computeDepthForView()` 中直接 `coarseCfg/fineCfg` 两趟调用，改为构建 `DepthPyramidRequest` 并调用 estimator。
保留 CUDA OOM 重试、取消点、极线校正和稀疏投影缓存；新增取消阶段名称“Level 3/2/1 PatchMatch 后”。

- [x] **Step 5: 运行 estimator、CPU PatchMatch 和源合同测试**

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_mvs_pipeline.exe --gtest_filter=DepthPyramidEstimatorTest.*:MvsPipelineTest.PatchMatchToDenseCloudEndToEnd`

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_source_contracts.exe --gtest_filter=MvsSchedulerContractTest.*`

Expected: 所有匹配测试通过，旧两层字符串合同更新为三级接口合同。

---

### Task 5: 校准置信度、过滤档位和帧级质量门控

**Files:**
- Create: `src/core/mvs/DepthFrameQualityGate.h`
- Create: `src/core/mvs/DepthFrameQualityGate.cpp`
- Modify: `src/core/mvs/MvsQualityReport.h`
- Modify: `src/core/mvs/MvsQualityReport.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/CMakeLists.txt`
- Test: `tests/test_mvs_pipeline.cpp`

**Interfaces:**
- Produces: `DepthFrameQualityDecision evaluateDepthFrame(const DepthFrameQualityInput &)`。
- Produces: `DepthFrameQualityDecision::Accepted/ValidationOnly/Rejected`。

- [x] **Step 1: 写失败测试覆盖错误满幅、低覆盖稳定主体和深度边界未收敛**

```cpp
TEST(DepthFrameQualityGateTest, RejectsSearchBoundaryCollapse)
{
    auto input = makeQualityInput();
    input.depthAtSearchBoundaryRatio = 0.72f;
    const auto decision = xjw::mvs::evaluateDepthFrame(input);
    EXPECT_EQ(decision.status, xjw::mvs::DepthFrameAcceptance::Rejected);
    EXPECT_THAT(decision.reasons, testing::Contains("depth_search_boundary_collapse"));
}
```

- [x] **Step 2: 运行测试并确认 gate 缺失**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mvs_pipeline -j 8`

Expected: 编译失败，提示 `evaluateDepthFrame` 未定义。

- [x] **Step 3: 实现校准置信度和过滤档位**

最终置信度按 `photometric * support * uniqueness * geometry * texture` 计算，每项 clamp 到 `[0,1]`。
`Mild/Moderate/Aggressive` 分别设置最小组件面积 `8/24/64`，局部相对深度阈值 `0.35/0.25/0.15`，
最小一致视图 `2/3/4`，但不得超过实际 source 数。

- [x] **Step 4: 实现帧级决策并阻止拒绝帧进入融合**

Decision 记录覆盖率、支持数 P50、置信度 P10/P50/P90、最大组件占比、边界深度比例、稀疏残差和原因列表。
`Rejected` 不写入融合队列；`ValidationOnly` 只能作为 source 一致性检查邻帧。

- [x] **Step 5: 运行质量测试**

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_mvs_pipeline.exe --gtest_filter=DepthFrameQualityGateTest.*:MvsPipelineTest.MvsQualityReport*`

Expected: 所有匹配测试通过。

---

### Task 6: 实现低内存流式 source 邻域一致性

**Files:**
- Create: `src/core/mvs/DepthConsistencyCache.h`
- Create: `src/core/mvs/DepthConsistencyCache.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/DepthFrameUtils.cpp`
- Modify: `src/core/CMakeLists.txt`
- Test: `tests/test_mvs_pipeline.cpp`
- Test: `tests/test_source_contracts.cpp`

**Interfaces:**
- Produces: `DepthConsistencyCache::loadNeighborhood(ref_index, source_indices, memory_budget)`。
- Consumes: Task 5 的 `DepthFrameQualityDecision`。

- [x] **Step 1: 写失败测试验证 LRU 预算和流式/常驻结果一致**

```cpp
TEST(DepthConsistencyCacheTest, EvictsByBudgetAndMatchesResidentConsistency)
{
    FakeDepthFrameStore store(makeThreeConsistentFrames());
    xjw::mvs::DepthConsistencyCache cache(&store, store.singleFrameBytes() * 2);
    const auto streaming = runStreamingConsistency(cache, 0, {1, 2});
    const auto resident = runResidentConsistency(store.frames(), 0, {1, 2});
    EXPECT_EQ(streaming.validMask, resident.validMask);
    EXPECT_LE(cache.peakBytes(), store.singleFrameBytes() * 2);
}
```

- [x] **Step 2: 运行测试并确认 cache 缺失**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mvs_pipeline -j 8`

Expected: 编译失败，提示 `DepthConsistencyCache` 未定义。

- [x] **Step 3: 实现线程安全 LRU 和邻域检查**

缓存 key 使用 `refIndex + configHash`，value 包含 depth/confidence/camera/source IDs；加载前检查预算，按最近最少使用
淘汰。单帧大于预算时允许临时独占，并在完成后立即释放。取消标志在每次 IO、重投影行块和写回前检查。

- [x] **Step 4: 删除低内存质量降级分支**

`DepthMapGenerator::crossCheckDepthConsistency()` 在常驻和流式模式下调用同一像素一致性核心；删除“内存不足跳过
全量一致性”的路径和日志。进度显示 `多视一致性：X/Y`。

- [x] **Step 5: 运行流式一致性和取消测试**

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_mvs_pipeline.exe --gtest_filter=DepthConsistencyCacheTest.*`

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_source_contracts.exe --gtest_filter=MvsSchedulerContractTest.DepthConsistency*`

Expected: 所有匹配测试通过，源码中不再含跳过一致性的质量降级文案。

---

### Task 7: 扩展 manifest、工作区树和深度诊断视图

**Files:**
- Modify: `src/core/mvs/MvsWorkspaceManifest.h`
- Modify: `src/core/mvs/MvsWorkspaceManifest.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `src/gui/widgets/DataTreeWidget.cpp`
- Modify: `src/gui/widgets/DisparityHeatmapOverlay.cpp`
- Test: `tests/test_mvs_pipeline.cpp`
- Test: `tests/test_gui_project_utils.cpp`

**Interfaces:**
- Produces: manifest schema 2 的 `levels`、`scene_profile`、`filter_mode`、`acceptance`、`quality_summary`。
- GUI 消费 metadata，不重新扫描目录推断层级状态。

- [x] **Step 1: 写失败测试覆盖 schema 2 round-trip 和旧 schema 单层读取**

```cpp
TEST(MvsWorkspaceManifestTest, RoundTripsDepthLevelSummaries)
{
    xjw::mvs::MvsDepthFrameRecord record;
    record.refIndex = 4;
    record.levels = makeLevelSummaries({4, 2, 1});
    record.sceneProfile = QStringLiteral("orbital_object");
    const auto restored = xjw::mvs::MvsDepthFrameRecord::fromJson(record.toJson());
    EXPECT_EQ(restored.levels.size(), 3);
    EXPECT_EQ(restored.sceneProfile, QStringLiteral("orbital_object"));
}
```

- [x] **Step 2: 运行测试并确认新字段缺失**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mvs_pipeline -j 8`

Expected: 编译失败，提示 `levels` 或 `sceneProfile` 不存在。

- [x] **Step 3: 实现 schema 2 与旧记录迁移**

旧记录读取为仅含 Level 1 的摘要；`makeMvsDepthConfigHash()` 纳入 pyramid、scene、filter 和输入签名。
配置 hash 不一致时禁止复用 raw depth/confidence。

- [x] **Step 4: 接入工作区树和诊断视图**

每个深度帧子项显示 `L1/L2/L3` 摘要；查看器模式为“所有级别、Level 1、Level 2、Level 3、置信度、
支持数、不确定度”。没有保存中间原始图时显示摘要，不伪造图像。

- [x] **Step 5: 运行 manifest 和 GUI 测试**

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_mvs_pipeline.exe --gtest_filter=MvsWorkspaceManifestTest.*`

Run: `build/windows-vcpkg-cuda-release/tests/Release/test_gui_project_utils.exe --gtest_filter=*DepthMap*:*DataTree*`

Expected: 所有匹配测试通过。

---

### Task 8: CLI、Dino/UAV A/B 和模型—影像回归验收

**Files:**
- Modify: `src/cli/cli_reconstruct_pipeline.cpp`
- Create: `scripts/validation/run_depth_pyramid_regression.ps1`
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/mvs/README.md`
- Test: `tests/test_cli_contracts.cpp`

**Interfaces:**
- CLI 新增 `--mvs-quality`、`--mvs-scene-profile`、`--mvs-depth-filter`、`--mvs-save-levels`。
- 脚本输出配置、逐帧质量、组件统计和模型—影像回归报告路径。

- [x] **Step 1: 写失败 CLI 合同测试**

```cpp
TEST(CliContractsTest, ReconstructPipelineExposesDepthPyramidOptions)
{
    const QString source = readSourceFile(QStringLiteral("src/cli/cli_reconstruct_pipeline.cpp"));
    EXPECT_TRUE(source.contains(QStringLiteral("--mvs-quality")));
    EXPECT_TRUE(source.contains(QStringLiteral("--mvs-scene-profile")));
    EXPECT_TRUE(source.contains(QStringLiteral("--mvs-depth-filter")));
    EXPECT_TRUE(source.contains(QStringLiteral("--mvs-save-levels")));
}
```

- [x] **Step 2: 运行测试并确认参数缺失**

Run: `cmake --build build/windows-vcpkg-cuda-release --config Release --target test_cli_contracts -j 8`

Expected: 测试失败并指出四个参数缺失。

- [x] **Step 3: 接入 CLI 与回归脚本**

CLI 和 GUI 都调用 `buildDepthGenConfig()` 与 `DepthPyramidEstimator`，禁止复制另一套参数映射。脚本分别执行 Dino、
UAV 9 图，并调用现有模型质量 CLI 生成 JSON；失败时返回非零退出码。

- [x] **Step 4: 运行核心和 GUI 回归测试**

Run: `ctest --test-dir build/windows-vcpkg-cuda-release -C Release --output-on-failure -R "Mvs|Depth|Gui|Cli"`

Expected: 相关测试全部通过。

2026-07-14 实测：Windows CUDA Release 下重新构建 `plascan_gui`、`reconstruct_pipeline_cli` 和聚焦测试目标；
`test_mvs_pipeline` 42/42、`test_mvs_rectifier_unit` 9/9、`test_mvs_workspace_manifest` 16/16、
`test_cli_contracts` 24/24、`test_source_contracts` 81/81 通过。

- [ ] **Step 5: 运行 Dino 质量验收**

Run: `powershell -ExecutionPolicy Bypass -File scripts/validation/run_depth_pyramid_regression.ps1 -Dataset Dino -Quality high`

Expected: 16/16 相机注册；至少 14 帧深度被接受；最大模型组件面数占比 `>=0.90`；浮片面数占比 `<0.02`；
留出视角轮廓 IoU `>=0.80`。

2026-07-14 深度子门槛：`mvs/mvs_manifest.json` 记录 16/16 帧为 `accepted`，深度帧门槛通过；现有
visual-hull/网格路径尚未通过完整模型连通性与模型—影像回归门槛，因此本步骤保持未完成。

- [ ] **Step 6: 运行 UAV 9 图质量验收**

Run: `powershell -ExecutionPolicy Bypass -File scripts/validation/run_depth_pyramid_regression.ps1 -Dataset Uav9 -Quality high`

Expected: 9/9 相机注册；至少 8 帧深度被接受；最大模型组件面数占比 `>=0.90`；与旧实现相比，相同误差阈值
下对 Metashape 局部参考点云覆盖率提高至少 20%，P50/P84/P95 距离均改善。

2026-07-14 深度子门槛：`mvs/mvs_manifest.json` 记录 8 帧 `accepted`、1 帧 `validation_only`、0 帧
`rejected`，达到至少 8 帧正式深度的门槛。边缘帧按最多 5 个有效邻图使用 0.50 一致性门槛，内部帧仍使用
0.55；模型组件和 Metashape 点云距离门槛仍待网格/融合阶段复测，
因此本步骤保持未完成。

- [x] **Step 7: 更新架构和 MVS 文档**

记录三级数据流、场景判定指标、manifest schema、流式一致性内存上界、CLI 示例和验收报告位置；文档只写实际
验证通过的结果。

---

## Completion Gate

- 所有新增行为都有先失败后通过的自动测试记录。
- Windows CUDA Release 构建成功。
- MVS/Depth/GUI/CLI 相关测试通过。
- Dino 与 UAV 9 图同时达到 Task 8 的硬门槛后，才允许运行 444 图长链。
- 未经用户明确要求，不创建 commit、tag 或 GitHub Release。
