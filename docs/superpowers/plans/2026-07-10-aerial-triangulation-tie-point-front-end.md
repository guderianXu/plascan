# Aerial Triangulation Tie-Point Front-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让“空中三角测量”在没有相机文件时使用与“创建连接点”一致的 SIFT + LightGlue 前端，避免全量匹配仍因特征过少导致匹配图断开。

**Architecture:** 保留 `AerialTriangulationWorkflow` 作为 UI/CLI 参数解析层，把用户的关键点限制、每百万像素限制、质量档位和重新生成连接点语义显式传入 `AerialTriangulationService`。`AerialTriangulationService` 不再用独立的稀疏 SIFT preset 决定空三连接点密度，而是复用与 `MatchPhotosTask` 一致的 SIFT 阈值、关键点预算和缓存签名；SfM 后端只消费已经通过几何验证的连接点结果。

**Tech Stack:** C++17, Qt6, OpenCV SIFT/CUDA SIFT, LightGlue TorchScript, CMake, GTest, CLI verification on `E:/code/test/temple`.

---

## File Structure

- Modify: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationService.h`
  - 增加空三连接点前端的显式配置字段：关键点上限、每百万像素关键点上限、是否使用连接点密集 SIFT 阈值、前端缓存版本。
- Modify: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationWorkflow.cpp`
  - 把 UI/CLI 的 `keypointLimit` 原值传入服务，不再只写入 `skeletonFeatureMaxKeypoints`。
  - 无相机/通用预选场景默认使用连接点密集前端；质量档位只影响图像缩放、二阶段预算和 SfM 阈值，不再把 SIFT 点压到几百个。
- Modify: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationService.cpp`
  - 增加 `siftDetectionThresholdForAerialTiePoints()`，与 `MatchPhotosTask` 的非 fast 阈值 `0.0005f` 对齐。
  - 特征提取时用 `opts.tiePointFeatureMaxKeypoints` / 每百万像素预算解析 `ExtractorConfig::maxKeypoints`。
  - 缓存 sidecar 写入前端版本和关键参数；发现旧低密度缓存时自动重提特征/重算匹配。
  - 质量报告增加每张影像关键点统计，方便 GUI/CLI 直接诊断“特征过少”。
- Modify: `E:/code/plascan/tests/test_aerial_triangulation_workflow.cpp`
  - 增加 workflow 层参数映射测试。
  - 更新低质量测试：低质量仍可缩放图像和降低二阶段预算，但不能丢掉用户请求的 40000 关键点上限。
- Modify: `E:/code/plascan/tests/test_source_contracts.cpp`
  - 增加源码契约测试，防止空三再次回退到独立的高 SIFT 阈值。
- Modify: `E:/code/plascan/src/cli/cli_aerial_triangulation.cpp`
  - 确认 report JSON 输出新增字段；CLI 不需要新增参数，但要暴露已解析关键点预算。
- Modify: `E:/code/plascan/src/core/aerial_triangulation/README.md`
  - 记录空三前端与创建连接点前端的统一规则。

---

### Task 1: Workflow 参数映射测试

**Files:**
- Modify: `E:/code/plascan/tests/test_aerial_triangulation_workflow.cpp`
- Modify after failing test: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationService.h`
- Modify after failing test: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationWorkflow.cpp`

- [ ] **Step 1: Write the failing test**

Add this test after `LowQualityUsesReducedImageScaleAndConservativeBudgets`:

```cpp
TEST(AerialTriangulationWorkflowCoreTest, LowQualityKeepsRequestedTiePointFeatureBudget)
{
    auto options = makeBaseOptions();
    options.quality = QStringLiteral("low");
    options.keypointLimit = 40000;
    options.tiepointLimit = 4000;
    options.featureAlgorithm = QStringLiteral("sift");
    options.matchAlgorithm = QStringLiteral("lightglue");

    const auto resolved = xjw::gui::AerialTriangulationWorkflow::resolveConfig(options);

    EXPECT_EQ(resolved.serviceOptions.quality, 0);
    EXPECT_EQ(resolved.serviceOptions.featureAlgorithm, QStringLiteral("sift"));
    EXPECT_EQ(resolved.serviceOptions.matchAlgorithm, QStringLiteral("lightglue"));
    EXPECT_EQ(resolved.serviceOptions.tiePointFeatureMaxKeypoints, 40000);
    EXPECT_EQ(resolved.serviceOptions.tiePointKeypointLimitPerMegapixel, 0);
    EXPECT_TRUE(resolved.serviceOptions.useTiePointDenseSift);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("resolved_keypoint_budget")).toInt(), 40000);
    EXPECT_EQ(resolved.resolvedSettings.value(QStringLiteral("skeleton_keypoint_budget")).toInt(), 10000);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_aerial_triangulation_workflow -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_aerial_triangulation_workflow.exe --gtest_filter=AerialTriangulationWorkflowCoreTest.LowQualityKeepsRequestedTiePointFeatureBudget --gtest_brief=1
```

Expected: compile fail because `tiePointFeatureMaxKeypoints`, `tiePointKeypointLimitPerMegapixel`, and `useTiePointDenseSift` do not exist.

- [ ] **Step 3: Add service option fields**

In `AerialTriangulationServiceOptions`, after `skeletonFeatureMaxKeypoints`, add:

```cpp
    /// 连接点前端的每张影像关键点上限。空三自动补齐连接点时必须使用这个预算，
    /// 不能被低质量 SfM 骨架预算覆盖；<=0 表示沿用质量档位。
    int                 tiePointFeatureMaxKeypoints = 40000;

    /// 每百万像素关键点限制。>0 时与 tiePointFeatureMaxKeypoints 共同取较小值。
    int                 tiePointKeypointLimitPerMegapixel = 0;

    /// 是否使用创建连接点同款密集 SIFT 阈值。无相机/通用预选空三默认开启。
    bool                useTiePointDenseSift = true;
```

- [ ] **Step 4: Map workflow fields**

In `AerialTriangulationWorkflow::resolveConfig`, replace the current resolved keypoint budget assignment block:

```cpp
    service.skeletonFeatureMaxKeypoints = scaledLimit(options.keypointLimit, preset.budgetScale);
```

with:

```cpp
    service.tiePointFeatureMaxKeypoints = std::max(0, options.keypointLimit);
    service.tiePointKeypointLimitPerMegapixel = 0;
    service.useTiePointDenseSift = featureAlgorithm == QStringLiteral("sift") &&
        matchAlgorithm == QStringLiteral("lightglue");
    service.skeletonFeatureMaxKeypoints = scaledLimit(options.keypointLimit, preset.budgetScale);
```

Then replace:

```cpp
    resolved.resolvedSettings.insert(QStringLiteral("resolved_keypoint_budget"),
                                     service.skeletonFeatureMaxKeypoints);
```

with:

```cpp
    resolved.resolvedSettings.insert(QStringLiteral("resolved_keypoint_budget"),
                                     service.tiePointFeatureMaxKeypoints);
    resolved.resolvedSettings.insert(QStringLiteral("skeleton_keypoint_budget"),
                                     service.skeletonFeatureMaxKeypoints);
```

- [ ] **Step 5: Update existing low-quality expectation**

Keep this expectation unchanged:

```cpp
EXPECT_EQ(resolved.serviceOptions.skeletonFeatureMaxKeypoints, 10000);
```

Add:

```cpp
EXPECT_EQ(resolved.serviceOptions.tiePointFeatureMaxKeypoints, 40000);
EXPECT_TRUE(resolved.serviceOptions.useTiePointDenseSift);
```

- [ ] **Step 6: Run workflow tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_aerial_triangulation_workflow -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_aerial_triangulation_workflow.exe --gtest_brief=1
```

Expected: all `AerialTriangulationWorkflowCoreTest` tests pass.

---

### Task 2: 空三 SIFT 前端与创建连接点密度对齐

**Files:**
- Modify: `E:/code/plascan/tests/test_source_contracts.cpp`
- Modify: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationService.cpp`

- [ ] **Step 1: Write the failing source contract**

Add this test near the existing aerial triangulation source contract tests:

```cpp
TEST(SourceContractsTest, AerialTriangulationUsesDenseSiftForTiePointGeneration)
{
    const QString source = readSourceFile(
        QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));

    EXPECT_TRUE(source.contains(QStringLiteral("siftDetectionThresholdForAerialTiePoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("0.0005f")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.tiePointFeatureMaxKeypoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("opts.tiePointKeypointLimitPerMegapixel")));
    EXPECT_FALSE(source.contains(QStringLiteral("extractorCfg.maxKeypoints  = presets.featureMaxKeypoints")));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_source_contracts -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_source_contracts.exe --gtest_filter=SourceContractsTest.AerialTriangulationUsesDenseSiftForTiePointGeneration --gtest_brief=1
```

Expected: fail because the function and option usages are not present yet.

- [ ] **Step 3: Add helper functions**

In the anonymous namespace of `AerialTriangulationService.cpp`, after `presetsForLevel`, add:

```cpp
float siftDetectionThresholdForAerialTiePoints(const AerialTriangulationServiceOptions &opts,
                                               const QualityPresets &presets)
{
    if (opts.featureAlgorithm.trimmed().toLower() == QStringLiteral("sift") &&
        opts.useTiePointDenseSift)
    {
        // 与“创建连接点”模块保持一致。空三前端需要密集连接点，低质量只应影响速度，
        // 不应把 SIFT 压到每张几十到几百个点。
        return 0.0005f;
    }
    return presets.featureDetectionThreshold;
}

int resolveTiePointFeatureMaxKeypoints(const AerialTriangulationServiceOptions &opts,
                                       const QualityPresets &presets,
                                       int imageWidth,
                                       int imageHeight)
{
    int limit = opts.tiePointFeatureMaxKeypoints > 0
        ? opts.tiePointFeatureMaxKeypoints
        : presets.featureMaxKeypoints;

    if (opts.tiePointKeypointLimitPerMegapixel > 0 && imageWidth > 0 && imageHeight > 0)
    {
        const double megapixels = static_cast<double>(imageWidth) *
            static_cast<double>(imageHeight) / 1000000.0;
        const int perMegapixelLimit = std::max(
            1,
            static_cast<int>(std::ceil(megapixels *
                                       static_cast<double>(opts.tiePointKeypointLimitPerMegapixel))));
        limit = limit > 0 ? std::min(limit, perMegapixelLimit) : perMegapixelLimit;
    }

    return limit;
}
```

- [ ] **Step 4: Use helper during extraction**

In the feature extraction block around `ExtractorConfig extractorCfg`, replace:

```cpp
            extractorCfg.maxKeypoints  = presets.featureMaxKeypoints;
            extractorCfg.detThreshold  = presets.featureDetectionThreshold;
```

with:

```cpp
            extractorCfg.maxKeypoints = resolveTiePointFeatureMaxKeypoints(
                opts,
                presets,
                grayImage.cols,
                grayImage.rows);
            extractorCfg.detThreshold = siftDetectionThresholdForAerialTiePoints(opts, presets);
```

Use the actual local image variable name in that block. If the current code uses `imgGray`, `gray`, or another name, use that variable; do not reload the image just for dimensions.

- [ ] **Step 5: Add Chinese log for effective front-end settings**

Near the existing SFM feature extraction logs, add:

```cpp
    LOG_INFO(QStringLiteral("SFM 连接点前端: %1 + %2，关键点上限 %3，每百万像素 %4，SIFT阈值 %5")
                 .arg(featureAlgorithm,
                      matchAlgorithm)
                 .arg(opts.tiePointFeatureMaxKeypoints)
                 .arg(opts.tiePointKeypointLimitPerMegapixel)
                 .arg(static_cast<double>(siftDetectionThresholdForAerialTiePoints(opts, presets)),
                      0,
                      'f',
                      6));
```

- [ ] **Step 6: Run source contract**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_source_contracts -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_source_contracts.exe --gtest_filter=SourceContractsTest.AerialTriangulationUsesDenseSiftForTiePointGeneration --gtest_brief=1
```

Expected: PASS.

---

### Task 3: 匹配缓存签名和旧低密度缓存失效

**Files:**
- Modify: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `E:/code/plascan/tests/test_source_contracts.cpp`

- [ ] **Step 1: Write cache contract test**

Add this source contract:

```cpp
TEST(SourceContractsTest, AerialTriangulationMatchCacheTracksTiePointFrontendVersion)
{
    const QString source = readSourceFile(
        QStringLiteral("src/core/aerial_triangulation/AerialTriangulationService.cpp"));

    EXPECT_TRUE(source.contains(QStringLiteral("tie_point_frontend_version")));
    EXPECT_TRUE(source.contains(QStringLiteral("tie_point_feature_max_keypoints")));
    EXPECT_TRUE(source.contains(QStringLiteral("tie_point_keypoint_limit_per_megapixel")));
    EXPECT_TRUE(source.contains(QStringLiteral("dense_sift_threshold")));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_source_contracts -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_source_contracts.exe --gtest_filter=SourceContractsTest.AerialTriangulationMatchCacheTracksTiePointFrontendVersion --gtest_brief=1
```

Expected: FAIL.

- [ ] **Step 3: Add frontend signature constants**

In `AerialTriangulationService.cpp` anonymous namespace, add:

```cpp
constexpr int kAerialTiePointFrontendVersion = 2;
```

- [ ] **Step 4: Write signature into sidecar/settings**

Where `.match` sidecar or `MatchFileRecord::settings` is created, add:

```cpp
settings.insert(QStringLiteral("tie_point_frontend_version"), kAerialTiePointFrontendVersion);
settings.insert(QStringLiteral("tie_point_feature_max_keypoints"), opts.tiePointFeatureMaxKeypoints);
settings.insert(QStringLiteral("tie_point_keypoint_limit_per_megapixel"),
                opts.tiePointKeypointLimitPerMegapixel);
settings.insert(QStringLiteral("dense_sift_threshold"),
                static_cast<double>(siftDetectionThresholdForAerialTiePoints(opts, presets)));
```

- [ ] **Step 5: Invalidate incompatible cache**

Where existing sidecar compatibility is checked, require these fields to match:

```cpp
const bool frontendCompatible =
    sidecar.value(QStringLiteral("tie_point_frontend_version")).toInt(-1) ==
        kAerialTiePointFrontendVersion &&
    sidecar.value(QStringLiteral("tie_point_feature_max_keypoints")).toInt(-1) ==
        opts.tiePointFeatureMaxKeypoints &&
    sidecar.value(QStringLiteral("tie_point_keypoint_limit_per_megapixel")).toInt(-1) ==
        opts.tiePointKeypointLimitPerMegapixel &&
    std::abs(sidecar.value(QStringLiteral("dense_sift_threshold")).toDouble(-1.0) -
             static_cast<double>(siftDetectionThresholdForAerialTiePoints(opts, presets))) < 1e-9;
```

If `frontendCompatible` is false, treat the match cache as missing and regenerate it. Preserve the existing algorithm/model/device compatibility checks.

- [ ] **Step 6: Run cache contract**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_source_contracts -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_source_contracts.exe --gtest_filter=SourceContractsTest.AerialTriangulationMatchCacheTracksTiePointFrontendVersion --gtest_brief=1
```

Expected: PASS.

---

### Task 4: Report 中输出关键点统计，定位匹配失败原因

**Files:**
- Modify: `E:/code/plascan/src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `E:/code/plascan/src/cli/cli_aerial_triangulation.cpp`
- Modify: `E:/code/plascan/tests/test_cli_contracts.cpp`

- [ ] **Step 1: Extend CLI contract**

In `PhotogrammetryWorkflowCliGTest.AerialTriangulationCliAcceptsImageOnlyListForDryRun`, after reading `service_options`, assert the new diagnostic keys are present in dry-run settings:

```cpp
EXPECT_TRUE(serviceOptions.contains(QStringLiteral("tie_point_feature_max_keypoints")));
EXPECT_TRUE(serviceOptions.contains(QStringLiteral("tie_point_keypoint_limit_per_megapixel")));
```

- [ ] **Step 2: Run CLI contract to verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_cli_contracts -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_cli_contracts.exe --gtest_filter=PhotogrammetryWorkflowCliGTest.AerialTriangulationCliAcceptsImageOnlyListForDryRun --gtest_brief=1
```

Expected: FAIL because the fields are missing.

- [ ] **Step 3: Add service options to CLI report JSON**

In `cli_aerial_triangulation.cpp`, where `service_options` JSON is built, add:

```cpp
serviceOptionsJson.insert(QStringLiteral("tie_point_feature_max_keypoints"),
                          service.tiePointFeatureMaxKeypoints);
serviceOptionsJson.insert(QStringLiteral("tie_point_keypoint_limit_per_megapixel"),
                          service.tiePointKeypointLimitPerMegapixel);
serviceOptionsJson.insert(QStringLiteral("use_tie_point_dense_sift"),
                          service.useTiePointDenseSift);
```

Use the actual local variable names in the report builder.

- [ ] **Step 4: Add feature count diagnostics to service result**

After feature extraction completes, compute min/max/average counts using `FeatureFileIO::peekCount(featurePath)` over generated or reused `.sift` files and insert into `result.sfmDiagnostics`:

```cpp
QJsonObject featureStats;
featureStats.insert(QStringLiteral("count"), featureCounts.size());
featureStats.insert(QStringLiteral("min_keypoints"), minCount);
featureStats.insert(QStringLiteral("max_keypoints"), maxCount);
featureStats.insert(QStringLiteral("avg_keypoints"), avgCount);
featureStats.insert(QStringLiteral("frontend"), QStringLiteral("sift-lightglue"));
result.sfmDiagnostics.insert(QStringLiteral("feature_keypoint_stats"), featureStats);
```

If the local result object is not available in the extraction helper, return the stats from the helper and attach it in `AerialTriangulationService::run()`.

- [ ] **Step 5: Run CLI contract**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_cli_contracts -j 20
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_cli_contracts.exe --gtest_filter=PhotogrammetryWorkflowCliGTest.AerialTriangulationCliAcceptsImageOnlyListForDryRun --gtest_brief=1
```

Expected: PASS.

---

### Task 5: Temple 数据回归验证

**Files:**
- No source changes in this task.
- Output directories under `E:/code/test/temple/.plascan_tmp/`.

- [ ] **Step 1: Clean only this task's output directories**

Run:

```powershell
$dirs = @(
  'E:/code/test/temple/.plascan_tmp/codex_at_dense_frontend_low',
  'E:/code/test/temple/.plascan_tmp/codex_at_dense_frontend_highest'
)
foreach ($dir in $dirs) {
  if (Test-Path $dir) {
    Remove-Item -LiteralPath $dir -Recurse -Force
  }
}
```

- [ ] **Step 2: Run low quality full-pair AT**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\aerial_triangulation_cli.exe `
  --input E:\code\test\temple\temple_images_for_probe.lis `
  --output-dir E:\code\test\temple\.plascan_tmp\codex_at_dense_frontend_low `
  --project E:\code\test\temple\.plascan_tmp\codex_at_dense_frontend_low\headless.plascan `
  --device cuda `
  --quality low `
  --keypoint-limit 40000 `
  --tiepoint-limit 4000 `
  --no-generic-preselection `
  --force
```

Expected:
- Report JSON exists at `E:/code/test/temple/.plascan_tmp/codex_at_dense_frontend_low/aerial_triangulation_cli_report.json`.
- `service_options.tie_point_feature_max_keypoints == 40000`.
- `registered_images == 16`.
- `points3d > 1000`.
- `assets/reports/matching_quality_report.csv` has `120` rows.

- [ ] **Step 3: Parse low quality report**

Run:

```powershell
$report = Get-Content -Raw E:/code/test/temple/.plascan_tmp/codex_at_dense_frontend_low/aerial_triangulation_cli_report.json | ConvertFrom-Json
"registered=$($report.registered_images)/$($report.image_count) points=$($report.points3d) reproj=$($report.mean_reproj_error)"
$csv = Import-Csv E:/code/test/temple/.plascan_tmp/codex_at_dense_frontend_low/assets/reports/matching_quality_report.csv
"pairs=$($csv.Count) matched=$(($csv | Where-Object status -eq 'matched').Count)"
```

Expected output shape:

```text
registered=16/16 points=<greater than 1000> reproj=<finite number>
pairs=120 matched=<greater than 40>
```

- [ ] **Step 4: Run highest quality as guardrail**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\aerial_triangulation_cli.exe `
  --input E:\code\test\temple\temple_images_for_probe.lis `
  --output-dir E:\code\test\temple\.plascan_tmp\codex_at_dense_frontend_highest `
  --project E:\code\test\temple\.plascan_tmp\codex_at_dense_frontend_highest\headless.plascan `
  --device cuda `
  --quality highest `
  --keypoint-limit 40000 `
  --tiepoint-limit 4000 `
  --no-generic-preselection `
  --force
```

Expected:
- `registered_images == 16`.
- `points3d >= 5000` or no worse than the current baseline `5798` by more than 15%.

- [ ] **Step 5: Record before/after numbers**

Add the measured values to the final implementation note:

```text
Temple low before: 9/16 registered, 258 points, 32 matched pairs.
Temple low after:  <fill measured result>.
Temple highest before: 16/16 registered, 5798 points, 59 matched pairs.
Temple highest after: <fill measured result>.
```

---

### Task 6: Build, focused tests, documentation

**Files:**
- Modify: `E:/code/plascan/src/core/aerial_triangulation/README.md`
- Optional modify if architecture docs are stale: `E:/code/plascan/docs/PROJECT_ARCHITECTURE.md`

- [ ] **Step 1: Update aerial triangulation README**

Add a section:

```markdown
## 连接点前端

空中三角测量在缺少已有连接点时会自动补齐 SIFT + LightGlue 连接点。
该前端与“创建连接点”模块保持同一关键点密度策略：

- SIFT 连接点检测阈值默认使用 `0.0005`，避免低纹理目标每张图只提取几十到几百个点。
- `keypoint_limit` 是每张影像连接点前端的真实上限；低/中/高质量只影响缩放、二阶段骨架预算和 SfM 阈值。
- `skeleton_keypoint_budget` 仅用于两阶段 SfM 骨架，不会覆盖最终连接点提取预算。
- 匹配缓存包含 `tie_point_frontend_version`、关键点预算和 SIFT 阈值；参数变化会自动重算。
```

- [ ] **Step 2: Build focused targets**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_aerial_triangulation_workflow test_source_contracts test_cli_contracts aerial_triangulation_cli match_photos_cli -j 20
```

Expected: build succeeds.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_aerial_triangulation_workflow.exe --gtest_brief=1
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_source_contracts.exe --gtest_filter=*AerialTriangulation*:*TiePoint* --gtest_brief=1
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_cli_contracts.exe --gtest_brief=1
```

Expected: all focused tests pass.

- [ ] **Step 4: Run temple regression**

Run the commands from Task 5. Expected: low quality full-pair AT registers all `16/16` images.

- [ ] **Step 5: Check git status**

Run:

```powershell
git status --short
```

Expected: only files from this plan are modified, except any pre-existing dirty submodule state that was already present before this work.

---

## Self-Review Checklist

- Spec coverage:
  - 空三全量匹配仍失败的根因是特征密度过低：Task 1/2 directly addresses it.
  - 低质量不再把连接点前端限制到几百点：Task 1/2 covers it.
  - 旧缓存不能污染新结果：Task 3 covers it.
  - GUI/CLI 后续能看到真实诊断：Task 4 covers it.
  - `temple` 数据必须验证：Task 5 covers it.
- Placeholder scan:
  - No `TBD`, `TODO`, or unspecified “add tests” steps.
  - Steps include exact file paths and commands.
- Type consistency:
  - New fields are consistently named:
    - `tiePointFeatureMaxKeypoints`
    - `tiePointKeypointLimitPerMegapixel`
    - `useTiePointDenseSift`
  - JSON keys are consistently named:
    - `tie_point_feature_max_keypoints`
    - `tie_point_keypoint_limit_per_megapixel`
    - `use_tie_point_dense_sift`
    - `tie_point_frontend_version`

