# Generic Overlap Pair Graph Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在没有相机参数时，让 Generic/BoW 预选输出一个可控、尽量连通、适合 SfM 的影像候选 pair 图。

**Architecture:** 在 `src/core/overlap` 新增独立的 `OverlapPairGraphPlanner`，输入 BoW 候选分数和可选序列先验，输出带来源和诊断的 pair 图。`VocabularyOverlapRetriever` 只负责视觉词汇相似度计算；pair 图连通性、top-K、component 补边和闭环序列补边由新模块负责。`matchphototask` 和 GUI 通过 `VocabularyOverlapResult::acceptedPairs` 消费规划后的结果。

**Tech Stack:** C++17, Qt-free core module, OpenCV-independent graph utility, GTest, CMake.

---

## File Structure

- Create: `src/core/overlap/OverlapPairGraphPlanner.h`
  - 定义 `OverlapPairGraphPlannerOptions`、`OverlapPairGraphEdge`、`OverlapPairGraphPlan` 和 `OverlapPairGraphPlanner::plan()`。
- Create: `src/core/overlap/OverlapPairGraphPlanner.cpp`
  - 实现 per-image top-K、mutual/one-way top-K、BoW component bridge、sequence bridge、ring sequence bridge、诊断统计。
- Modify: `src/core/overlap/VocabularyOverlapRetriever.h`
  - 增加配置项：`minPairsPerImage`、`connectComponents`、`useSequenceFallback`、`sequenceWindow`、`closeSequenceLoop`、`componentBridgeMaxPairs`。
- Modify: `src/core/overlap/VocabularyOverlapRetriever.cpp`
  - 保留 BoW 分数计算；把 accepted pair 选择交给 `OverlapPairGraphPlanner`。
- Modify: `src/core/overlap/CMakeLists.txt`
  - 把新文件加入 `overlap` 静态库。
- Modify: `src/core/matchphototask/pair_selection/PairSelector.cpp`
  - 保留现有兜底桥接，但优先消费 retriever 已经修复过的 `acceptedPairs`。
- Modify: `src/core/matchphototask/task/MatchPhotosTask.cpp`
  - 根据创建连接点参数把 sequence/ring/connectivity 选项传给 `VocabularyOverlapConfig`。
- Modify: `src/gui/dialogs/VocabularyOverlapDialog.cpp`
  - 如后续需要，在独立词汇重叠工具里显示连通性诊断；第一阶段可只复用默认配置。
- Test: `tests/test_overlap_pair_graph_planner.cpp`
- Modify: `tests/test_vocabulary_overlap_retriever.cpp`
- Modify: `tests/CMakeLists.txt`

---

## Task 1: Add Core Pair Graph Planner Types

**Files:**
- Create: `src/core/overlap/OverlapPairGraphPlanner.h`
- Create: `src/core/overlap/OverlapPairGraphPlanner.cpp`
- Modify: `src/core/overlap/CMakeLists.txt`
- Test: `tests/test_overlap_pair_graph_planner.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing tests for top-K and connectivity**

Add `tests/test_overlap_pair_graph_planner.cpp` with cases:

```cpp
TEST(OverlapPairGraphPlannerTest, KeepsAtLeastMinPairsPerImageFromOneWayTopK);
TEST(OverlapPairGraphPlannerTest, BridgesDisconnectedBowComponentsWithBestCrossComponentEdge);
TEST(OverlapPairGraphPlannerTest, FallsBackToSequenceBridgeWhenBowHasNoCrossComponentScore);
TEST(OverlapPairGraphPlannerTest, AddsRingClosureForSequenceLoop);
```

Use small synthetic score matrices:

```cpp
// component A: 0-1-2, component B: 3-4-5, best cross edge: 2-3
// expected accepted edges include 2-3 with source "bow_component_bridge".
```

- [ ] **Step 2: Verify tests fail before implementation**

Run:

```powershell
cmd /c ""C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_overlap_pair_graph_planner --parallel 32 && E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_overlap_pair_graph_planner.exe"
```

Expected: build fails because `OverlapPairGraphPlanner.h` or symbols do not exist.

- [ ] **Step 3: Implement planner API**

Define:

```cpp
namespace xjw {

enum class OverlapPairGraphSource
{
    BowMutualTopK,
    BowOneWayTopK,
    BowComponentBridge,
    SequenceBridge,
    SequenceLoop
};

struct OverlapPairGraphInputEdge
{
    int indexA = -1;
    int indexB = -1;
    double bowScore = 0.0;
    int sharedWordCount = 0;
    int geometricInliers = 0;
};

struct OverlapPairGraphPlannerOptions
{
    int imageCount = 0;
    int topK = 8;
    int minPairsPerImage = 4;
    double minSimilarity = 0.05;
    bool mutualTopK = true;
    bool keepOneWayTopK = true;
    bool connectComponents = true;
    bool useSequenceFallback = true;
    int sequenceWindow = 1;
    bool closeSequenceLoop = true;
    int componentBridgeMaxPairs = 0;
};

struct OverlapPairGraphEdge
{
    int indexA = -1;
    int indexB = -1;
    double bowScore = 0.0;
    int sharedWordCount = 0;
    int geometricInliers = 0;
    std::vector<OverlapPairGraphSource> sources;
};

struct OverlapPairGraphPlan
{
    std::vector<OverlapPairGraphEdge> edges;
    int componentCountBeforeRepair = 0;
    int componentCountAfterRepair = 0;
    int bowBridgeCount = 0;
    int sequenceBridgeCount = 0;
    std::string detail;
};

class OverlapPairGraphPlanner
{
public:
    static OverlapPairGraphPlan plan(const std::vector<OverlapPairGraphInputEdge> &edges,
                                     const OverlapPairGraphPlannerOptions &options);
};

} // namespace xjw
```

- [ ] **Step 4: Implement minimal graph repair**

Implementation rules:

- normalize pair index so `indexA < indexB`;
- build per-image sorted neighbor list by `bowScore`;
- accept mutual top-K first;
- accept one-way top-K until each image has at least `minPairsPerImage` if available;
- compute connected components over accepted edges;
- if disconnected, add best available BoW edge crossing each component boundary;
- if still disconnected and `useSequenceFallback=true`, add sequence-window crossing edges;
- if `closeSequenceLoop=true`, add edges between sequence head/tail within `sequenceWindow`.

- [ ] **Step 5: Run planner tests**

Run the same command as Step 2.

Expected: all planner tests pass.

---

## Task 2: Route Vocabulary Overlap Through Planner

**Files:**
- Modify: `src/core/overlap/VocabularyOverlapRetriever.h`
- Modify: `src/core/overlap/VocabularyOverlapRetriever.cpp`
- Modify: `tests/test_vocabulary_overlap_retriever.cpp`

- [ ] **Step 1: Add failing retriever tests**

Add tests:

```cpp
TEST(VocabularyOverlapRetrieverTest, PlannerKeepsOneWayTopKWhenMutualTopKWouldDisconnect);
TEST(VocabularyOverlapRetrieverTest, PlannerReportsConnectivityRepairInDetail);
```

Expected behavior:

- `acceptedPairs` contains planner-repaired pairs;
- `result.detail` contains `components_before` and `components_after`;
- existing tests for dense and inverted index parity still pass.

- [ ] **Step 2: Extend config**

Add to `VocabularyOverlapConfig`:

```cpp
int minPairsPerImage = 4;
bool keepOneWayTopK = true;
bool connectComponents = true;
bool useSequenceFallback = true;
int sequenceWindow = 1;
bool closeSequenceLoop = true;
int componentBridgeMaxPairs = 0;
```

- [ ] **Step 3: Convert BoW candidates to planner input**

After `result->candidates` is populated, build `std::vector<OverlapPairGraphInputEdge>`.

For each planner edge, copy the corresponding candidate into `acceptedPairs`, set `accepted=true`, and set `rejectReason.clear()`.

For candidates not accepted, keep them in `candidates`, set `accepted=false`, and set `rejectReason="not_selected_by_pair_graph_planner"` unless they already have a geometry rejection reason.

- [ ] **Step 4: Preserve geometry check**

If `geometryCheck=true`, geometry rejection still wins:

- planner may propose a pair;
- if geometric inliers `< minInliers`, it remains rejected;
- after geometry rejection, run a second lightweight repair pass only over geometrically accepted candidate edges if needed.

This prevents the final `acceptedPairs` from claiming graph connectivity that RANSAC has already invalidated.

- [ ] **Step 5: Run retriever tests**

Run:

```powershell
cmd /c ""C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_vocabulary_overlap_retriever --parallel 32 && E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_vocabulary_overlap_retriever.exe"
```

Expected: all vocabulary retriever tests pass.

---

## Task 3: Connect MatchPhotos Defaults To No-Camera Planner

**Files:**
- Modify: `src/core/matchphototask/task/MatchPhotosTask.cpp`
- Modify: `src/core/matchphototask/task/MatchPhotosOptions.h`
- Modify: `src/core/matchphototask/tests/test_match_photos_task.cpp`

- [ ] **Step 1: Add failing source/behavior tests**

Assert that `makeVocabularyConfig()` maps MatchPhotos options:

```cpp
EXPECT_EQ(config.topK, std::max(8, options.pairPolicy.sequenceWindow * 2));
EXPECT_EQ(config.minPairsPerImage, std::max(4, options.pairPolicy.sequenceWindow));
EXPECT_TRUE(config.connectComponents);
EXPECT_TRUE(config.useSequenceFallback);
EXPECT_TRUE(config.closeSequenceLoop);
```

- [ ] **Step 2: Update config mapping**

Set no-camera defaults:

- `topK = max(8, sequenceWindow * 2)`;
- `minPairsPerImage = max(4, sequenceWindow)`;
- `connectComponents = true`;
- `useSequenceFallback = true`;
- `sequenceWindow = max(1, pairPolicy.sequenceWindow)`;
- `closeSequenceLoop = true`.

- [ ] **Step 3: Keep PairSelector bridge as defense-in-depth**

Do not remove `PairSelector` bridge repair yet. It catches:

- manually supplied disconnected pair lists;
- older overlap JSON loaded from disk;
- future non-BoW preselection sources.

- [ ] **Step 4: Run MatchPhotos tests**

Run:

```powershell
cmd /c ""C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_match_photos_task test_matchphotos_pair_selector --parallel 32 && E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_match_photos_task.exe && E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_matchphotos_pair_selector.exe"
```

Expected: both binaries pass.

---

## Task 4: Improve Diagnostics And Reports

**Files:**
- Modify: `src/core/overlap/VocabularyOverlapRetriever.cpp`
- Modify: `src/gui/dialogs/VocabularyOverlapDialog.cpp`
- Modify: `src/core/matchphototask/task/MatchPhotosTask.cpp`

- [ ] **Step 1: Add planner detail fields**

Include in `VocabularyOverlapResult::detail`:

```text
components_before=<n> components_after=<n> bow_bridges=<n> sequence_bridges=<n> accepted=<n> candidates=<n>
```

- [ ] **Step 2: Persist pair source in overlap JSON**

When GUI writes vocabulary overlap candidates, add:

```json
"source_types": ["bow_mutual_top_k", "bow_component_bridge"]
```

This makes later review possible when users ask why a pair was matched.

- [ ] **Step 3: Surface progress text**

During Create Tie Points:

- after BoW retrieval: `Generic 预选: 构建候选图...`;
- after planner repair: `Generic 预选: 候选图 4 -> 1 个连通分量, 补边 3 对`;
- during matching: existing pair progress remains unchanged.

- [ ] **Step 4: Run GUI/source contract tests**

Run:

```powershell
cmd /c ""C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_gui_project_utils test_source_contracts plascan_gui --parallel 32"
```

Expected: build succeeds and relevant tests pass.

---

## Task 5: Validate On `E:/code/test/hyb2`

**Files:**
- Runtime project data only; do not commit generated `.match` or `.plascan` artifacts.

- [ ] **Step 1: Regenerate tie points**

Run from GUI or CLI path used by the project:

- quality: high/highest;
- feature: SIFT CUDA;
- matcher: LightGlue;
- generic preselection enabled;
- reference preselection disabled;
- reset current alignment enabled.

- [ ] **Step 2: Inspect generated reports**

Check:

```powershell
Get-Content E:\code\test\hyb2\assets\latest_tie_points.json
Get-Content E:\code\test\hyb2\assets\reports\matching_quality_report.json
```

Expected:

- `latest_tie_points.json` contains more than the old 616 planned/consumed pairs when graph repair adds bridges;
- `actual_match_graph.componentCount` should drop compared with old value `4`;
- if still disconnected, pending/failed bridge pairs should be visible with reasons.

- [ ] **Step 3: Run SfM**

Expected:

- SfM uses all compatible `.match` caches, including those outside the initial restricted plan;
- registered image ratio should improve beyond the previous `95/211`;
- if MVS quality gate still blocks, report should identify remaining disconnected or weak components.

---

## Self-Review

- Spec coverage: no-camera Generic/BoW planning, top-K, connectivity repair, sequence fallback, SfM consumption of all compatible pairs, diagnostics, and tests are covered.
- Placeholder scan: no TBD/TODO placeholders are present.
- Type consistency: planner types use `OverlapPairGraph*`; vocabulary retriever maps from `VocabularyOverlapPairResult` to planner inputs and back.
