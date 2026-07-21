# Aerial Triangulation Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将空中三角测量重构为唯一复用 `matchphototask` 的无 GUI 核心工作流，建立独立 CMake target，并把项目寻址与项目元数据工具迁移到 `common/project`。

**Architecture:** GUI 和 CLI 只调用 `xjw::aerial_triangulation::AerialTriangulationWorkflow`。工作流先根据重置状态和缓存完整性调用 `xjw::matchphotos::MatchPhotosTask` 准备连接点，再把只读的特征、匹配、轨迹和相机先验交给空三重建管线；空三管线不再拥有任何特征提取器或匹配器。项目路径、元数据、相机序列化和匹配目录扫描属于 `plascan_common_project`，不得依赖 GUI。

**Tech Stack:** C++17、CMake、Qt6 Core、OpenCV、现有 `matchphototask`、`sfm`、`bundle_adjust`、GTest。

---

## 1. 最终依赖方向

```text
plascan_gui / CLI executables
          |
          v
aerial_triangulation
          |
          +----> matchphototask ----> feature_extractors / feature_match / overlap
          |
          +----> sfm -------------> bundle_adjust / camera / intersection
          |
          +----> plascan_common_project

plascan_common_project ----> Qt6::Core / camera
```

禁止出现以下反向依赖：

```text
src/core -> src/gui
src/common -> src/gui
sfm -> aerial_triangulation
matchphototask -> aerial_triangulation
```

## 2. 最终目录结构

```text
src/common/project/
├── ProjectIO.h
├── ProjectIO.cpp
├── ProjectMetadata.h
├── ProjectMetadata.cpp
├── ProjectCameraIO.h
├── ProjectCameraIO.cpp
├── ProjectMatchCatalog.h
├── ProjectMatchCatalog.cpp
├── SparseResultQuality.h
├── SparseResultQuality.cpp
└── test/
    ├── CMakeLists.txt
    ├── test_project_io.cpp
    ├── test_project_metadata.cpp
    ├── test_project_camera_io.cpp
    └── test_project_match_catalog.cpp

src/core/aerial_triangulation/
├── CMakeLists.txt
├── README.md
├── model/
│   ├── AerialTriangulationOptions.h
│   ├── AerialTriangulationResult.h
│   └── AerialTriangulationResolvedConfig.h
├── workflow/
│   ├── AerialTriangulationWorkflow.h
│   ├── AerialTriangulationWorkflow.cpp
│   ├── AerialTriangulationPipeline.h
│   └── AerialTriangulationPipeline.cpp
├── preparation/
│   ├── TiePointPreparation.h
│   ├── TiePointPreparation.cpp
│   ├── MatchResultCatalog.h
│   ├── MatchResultCatalog.cpp
│   ├── ReconstructionPrerequisiteReport.h
│   └── ReconstructionPrerequisiteReport.cpp
├── reconstruction/
│   ├── SfmAttemptRunner.h
│   ├── SfmAttemptRunner.cpp
│   ├── SfmObservationGraphPolicy.h
│   ├── SfmObservationGraphPolicy.cpp
│   ├── GuidedRematchCoordinator.h
│   ├── GuidedRematchCoordinator.cpp
│   ├── SfmPairPlanner.h
│   └── SfmMatchDiagnostics.h
├── search/
│   ├── AdaptiveFocalSearch.h
│   ├── AdaptiveFocalSearch.cpp
│   ├── SfmSearchPolicy.h
│   └── SfmSearchPolicy.cpp
├── reporting/
│   ├── AerialTriangulationResultWriter.h
│   ├── AerialTriangulationResultWriter.cpp
│   ├── QualityReportWriter.h
│   └── QualityReportWriter.cpp
└── tests/
    ├── CMakeLists.txt
    ├── test_aerial_triangulation_workflow.cpp
    ├── test_aerial_triangulation_pipeline.cpp
    ├── test_sfm_attempt_runner.cpp
    ├── test_adaptive_focal_search.cpp
    ├── test_match_result_catalog.cpp
    └── test_reconstruction_prerequisites.cpp
```

所有公开类型使用：

```cpp
namespace xjw::aerial_triangulation
{
// ...
}
```

不得保留 `xjw::gui` 别名、转发头文件或旧类兼容层。

## 3. 唯一工作流数据流

```text
AerialTriangulationWorkflow::run
  1. resolveConfig
  2. 判断是否重置对齐、连接点是否缺失或缓存签名是否失效
  3. 需要准备时调用 MatchPhotosTask::run
  4. MatchResultCatalog 扫描并固定本次使用的匹配 variant
  5. ReconstructionPrerequisiteReport 检查图连通性和最小观测条件
  6. AerialTriangulationPipeline 调用 SfmAttemptRunner
  7. 必要时执行 GuidedRematchCoordinator 或 AdaptiveFocalSearch
  8. ResultWriter 写出稀疏点云、相机结果和质量报告
  9. 返回结构化结果，由 GUI/CLI 更新项目状态
```

`AerialTriangulationPipeline` 的输入必须是已经准备好的连接点资产，不允许包含以下字段：

```cpp
featureAlgorithm
matchAlgorithm
autoGenerateMissingMatches
cudaParallelPairs
featureMaxImageDim
tiePointFeatureMaxKeypoints
tiePointKeypointLimitPerMegapixel
useTiePointDenseSift
baOnly
```

这些字段只属于 `AerialTriangulationWorkflowOptions` 和 `MatchPhotosOptions`。空三结果也不再返回 `newFeatureFiles`、`newMatchFiles`、`failedPairs`；连接点产物由 `MatchPhotosResult` 原样承载。

---

### Task 1: 用行为测试锁定 common/project 迁移

**Files:**
- Create: `src/common/project/test/CMakeLists.txt`
- Create: `src/common/project/test/test_project_io.cpp`
- Create: `src/common/project/test/test_project_metadata.cpp`
- Create: `src/common/project/test/test_project_camera_io.cpp`
- Create: `src/common/project/test/test_project_match_catalog.cpp`
- Modify: `src/common/CMakeLists.txt`

- [ ] **Step 1: 为项目目录规则编写失败测试**

覆盖 `.plascan` 根目录、`assets/ip`、`assets/matches`、`assets/masks`、标记 sidecar、相对资源路径解析及 Unicode 路径。

```cpp
TEST(ProjectIOTest, ResolvesCanonicalProjectDirectories)
{
    const QString project = QDir(_temp.path()).filePath(QStringLiteral("工程.plascan"));
    EXPECT_EQ(ProjectIO::projectAssetsDir(project),
              QDir(_temp.path()).filePath(QStringLiteral("assets")));
    EXPECT_EQ(ProjectIO::ipmatchOutputDir(project),
              QDir(_temp.path()).filePath(QStringLiteral("assets/matches")));
}
```

- [ ] **Step 2: 为元数据和相机转换编写失败测试**

验证顶层/嵌套 `project_files` 两种格式、路径 token 解析、Camera JSON 往返和 TSAI 解析。

- [ ] **Step 3: 为匹配对目录扫描编写失败测试**

验证 `.match`、sidecar、`no_match_pairs.json` 的规范化、去重和稳定排序。

- [ ] **Step 4: 运行测试并确认迁移前失败**

```powershell
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure -R "ProjectIO|ProjectMetadata|ProjectCameraIO|ProjectMatchCatalog"
```

Expected: 新 target 或新命名空间尚不存在，测试编译失败。

### Task 2: 将项目通用能力迁移到 common/project

**Files:**
- Move: `src/gui/project/io/ProjectIO.h` -> `src/common/project/ProjectIO.h`
- Move: `src/gui/project/io/ProjectIO.cpp` -> `src/common/project/ProjectIO.cpp`
- Split: `src/gui/project/support/ProjectSupportUtils.h/.cpp`
- Create: `src/common/project/ProjectMetadata.h/.cpp`
- Create: `src/common/project/ProjectCameraIO.h/.cpp`
- Create: `src/common/project/ProjectMatchCatalog.h/.cpp`
- Modify: `src/common/CMakeLists.txt`
- Modify: all callers returned by `rg -l "ProjectIO.h|ProjectSupportUtils.h" src tests`
- Delete: `src/gui/project/io/ProjectIO.h/.cpp`
- Delete: `src/gui/project/support/ProjectSupportUtils.h/.cpp`

- [ ] **Step 1: 把纯路径函数迁移为 common API**

```cpp
namespace xjw::common::project
{
class ProjectIO
{
public:
    static QString projectRootFromPlascan(const QString &plascan_path);
    static QString projectAssetsDir(const QString &plascan_path);
    static QString ipfindOutputDir(const QString &plascan_path);
    static QString ipmatchOutputDir(const QString &plascan_path);
    static QString findFeatureForImage(const QString &plascan_path,
                                       const QString &image_path);
    static QString findMaskForImage(const QString &plascan_path,
                                    const QString &image_path);
};
}
```

- [ ] **Step 2: 按职责拆分 ProjectSupportUtils**

`ProjectMetadata` 负责项目 JSON 和影像 token；`ProjectCameraIO` 负责 Camera/TSAI；`ProjectMatchCatalog` 负责项目级匹配记录。不得把原 `ProjectSupportUtils` 整体换目录后继续保留。`plascan_common_project` 可以依赖 `camera`，但 `camera` 不得反向依赖 `plascan_common_project`，以免形成静态库循环依赖。

- [ ] **Step 3: 扩充 plascan_common_project target**

```cmake
add_library(plascan_common_project STATIC
  project/ProjectIO.cpp
  project/ProjectMetadata.cpp
  project/ProjectCameraIO.cpp
  project/ProjectMatchCatalog.cpp
  project/SparseResultQuality.cpp
)

target_link_libraries(plascan_common_project PUBLIC
  Qt6::Core
  camera
)
```

- [ ] **Step 4: 一次性迁移所有 include 和命名空间**

统一改为 `xjw::common::project`，删除 `xjw::gui::project` using 声明和旧 include directory。

- [ ] **Step 5: 删除旧文件并运行测试**

```powershell
ctest --test-dir build -C Release --output-on-failure -R "ProjectIO|ProjectMetadata|ProjectCameraIO|ProjectMatchCatalog|GuiProjectUtils"
```

Expected: common/project 测试通过，`rg "src/gui/project/io|ProjectSupportUtils" src/core src/cli` 无结果。

### Task 3: 建立独立 aerial_triangulation target

**Files:**
- Create: `src/core/aerial_triangulation/CMakeLists.txt`
- Modify: `src/core/CMakeLists.txt`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `src/gui/cmake/GuiCoreLinking.cmake`
- Modify: `src/cli/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 添加 target 失败契约测试**

测试配置必须能链接 `aerial_triangulation`，且 GUI/CLI 不再直接列出模块 `.cpp`。

- [ ] **Step 2: 创建静态库 target**

```cmake
add_library(aerial_triangulation STATIC ${AERIAL_TRIANGULATION_SOURCES})
target_compile_features(aerial_triangulation PUBLIC cxx_std_17)
target_include_directories(aerial_triangulation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aerial_triangulation PUBLIC
  Qt6::Core
  plascan_common_project
  matchphototask
  sfm
  bundle_adjust
  camera
  control_points
)
```

- [ ] **Step 3: GUI、CLI 和测试只链接 target**

从 `GuiSources.cmake`、`src/cli/CMakeLists.txt`、`tests/CMakeLists.txt` 删除所有 `aerial_triangulation/*.cpp` 和 GUI ProjectIO 源文件清单。

- [ ] **Step 4: 多线程全量编译**

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --config Release --parallel $env:NUMBER_OF_PROCESSORS
```

Expected: 同一个空三实现只编译进 `aerial_triangulation` 一次。

### Task 4: 统一模型和命名空间

**Files:**
- Create: `src/core/aerial_triangulation/model/AerialTriangulationOptions.h`
- Create: `src/core/aerial_triangulation/model/AerialTriangulationResult.h`
- Create: `src/core/aerial_triangulation/model/AerialTriangulationResolvedConfig.h`
- Modify: all files under `src/core/aerial_triangulation`
- Modify: GUI/CLI callers and aerial tests

- [ ] **Step 1: 添加公共 API 编译测试**

```cpp
using xjw::aerial_triangulation::AerialTriangulationOptions;
using xjw::aerial_triangulation::AerialTriangulationResult;
using xjw::aerial_triangulation::AerialTriangulationWorkflow;
```

- [ ] **Step 2: 把输入、解析结果和输出拆成独立 DTO**

DTO 只保存数据，不执行 IO、匹配或 SfM。字段命名继续使用现有持久化 JSON key，避免破坏项目文件格式。

- [ ] **Step 3: 全量迁移命名空间**

将 `xjw::gui::AerialTriangulation*` 改为 `xjw::aerial_triangulation::*`，不保留 namespace alias。

- [ ] **Step 4: 运行编译和 API 测试**

```powershell
ctest --test-dir build -C Release --output-on-failure -R "AerialTriangulation"
```

### Task 5: 让 matchphototask 成为连接点唯一所有者

**Files:**
- Create: `src/core/aerial_triangulation/preparation/TiePointPreparation.h/.cpp`
- Modify: `src/core/aerial_triangulation/workflow/AerialTriangulationWorkflow.cpp`
- Modify: `src/core/aerial_triangulation/model/AerialTriangulationOptions.h`
- Modify: `src/core/aerial_triangulation/model/AerialTriangulationResult.h`
- Modify: `src/cli/cli_aerial_triangulation.cpp`
- Modify: `src/cli/cli_reconstruct_pipeline.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/gui/project/manager/ProjectCameraSetupManager.cpp`
- Test: `src/core/aerial_triangulation/tests/test_aerial_triangulation_workflow.cpp`

- [ ] **Step 1: 写工作流顺序测试**

测试必须证明：重置或缺失缓存时 MatchPhotos 先执行；失败时 SfM 不执行；缓存完整时可跳过生成；SfM 永远收到 `autoGenerateMissingMatches` 不存在的准备完毕输入。

```cpp
EXPECT_EQ(events, (QStringList{QStringLiteral("matchphotos"),
                               QStringLiteral("sfm")}));
```

- [ ] **Step 2: 实现 TiePointPreparation**

它只负责把工作流参数转换成 `MatchPhotosOptions/Context` 并调用 `MatchPhotosTask::run`，不复制任何提取或匹配算法。

- [ ] **Step 3: 迁移所有直接 Service 调用**

以下入口全部改为 `AerialTriangulationWorkflow::run`：

```text
src/cli/cli_aerial_triangulation.cpp
src/cli/cli_reconstruct_pipeline.cpp
src/gui/main_window/MenuWorkflowController.cpp
src/gui/project/manager/ProjectCameraSetupManager.cpp
```

- [ ] **Step 4: 删除空三服务中的连接点生成代码**

删除 Phase 1/Phase 2、`baOnly`、特征提取器/匹配器 include、no-match 写入和 SIFT/BF fallback。Guided rematching 若需要新增匹配，必须调用 `matchphototask::GuidedMatchStage` 或其公开核心接口。

- [ ] **Step 5: 删除连接点兼容结果字段**

GUI/CLI 从 `workflowResult.tiePointResult` 读取新特征、新匹配和失败信息；空三重建结果只描述相机、稀疏点、BA 和质量。

- [ ] **Step 6: 执行唯一入口检查**

```powershell
rg -n "AerialTriangulationService::run" src
rg -n "ExtractorFactory|LightGlueMatcher|TraditionalMatcher" src/core/aerial_triangulation
```

Expected: 两条命令均无结果。

### Task 6: 拆分 AerialTriangulationService

**Files:**
- Delete: `src/core/aerial_triangulation/AerialTriangulationService.h`
- Delete: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Create/Modify: `workflow/`, `reconstruction/`, `search/`, `reporting/` 中最终结构文件
- Test: `src/core/aerial_triangulation/tests/*.cpp`

- [ ] **Step 1: 提取单次 SfM 尝试**

`SfmAttemptRunner::run` 只负责读取已选匹配、构建 observation graph、调用 `IncrementalSfm` 并返回内存结果，不写项目文件。

- [ ] **Step 2: 提取自适应焦距搜索**

`AdaptiveFocalSearch` 接收候选列表和 `SfmAttemptRunner`，负责粗筛、评分、正式重放。测试固定候选结果，验证评分和选择，不运行真实 CUDA。

- [ ] **Step 3: 提取引导重匹配协调器**

协调器只生成需要补充的 pair 和几何约束，然后构造 `MatchPhotosOptions`（`enableGuidedMatching=true`）及受限的 `PairSelectionInput`，统一调用 `MatchPhotosTask::run`，再决定是否接受重建增益。它不得直接调用 `GuidedMatchStage`，确保连接点生成始终只有一个公开入口。

- [ ] **Step 4: 提取结果写出和质量报告**

`AerialTriangulationResultWriter` 写 PLY/相机结果；`QualityReportWriter` 负责 JSON。二者使用显式输出目录，不读取 GUI ProjectManager。

- [ ] **Step 5: 用 Pipeline 组合各阶段**

```cpp
class AerialTriangulationPipeline
{
public:
    AerialTriangulationResult run(const PreparedAerialTriangulationInput &input) const;
};
```

Pipeline 只编排阶段、进度、取消和错误传播；单文件目标控制在 400 行左右。

- [ ] **Step 6: 删除旧 Service 文件并完整编译**

```powershell
cmake --build build --config Release --parallel $env:NUMBER_OF_PROCESSORS
```

### Task 7: 用行为测试替换源码字符串测试

**Files:**
- Modify: `tests/test_source_contracts.cpp`
- Modify: `tests/test_gui_project_utils.cpp`
- Move/Rewrite: aerial tests into `src/core/aerial_triangulation/tests/`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 删除对实现文本的断言**

删除读取 `AerialTriangulationService.cpp`、检查函数名/字符串和要求 GUI CMake 直接列出 `.cpp` 的测试。

- [ ] **Step 2: 增加配置解析测试**

覆盖最高/低精度、照片序列、参考预选、重置对齐、蒙版模式、关键点与连接点限制。

- [ ] **Step 3: 增加工作流编排测试**

通过 fake tie-point runner 和 fake pipeline 验证调用顺序、失败短路、进度映射、取消传播和结果合并。

- [ ] **Step 4: 增加最小集成测试**

使用仓库 Middlebury dino/temple 小数据运行 MatchPhotos -> SfM，至少断言匹配图连通、注册数量不回退、输出质量 JSON 可解析。

- [ ] **Step 5: 运行针对性测试**

```powershell
ctest --test-dir build -C Release --output-on-failure -R "Project|MatchPhotos|AerialTriangulation|Sfm|BundleAdjust|CliContract|GuiProjectUtils"
```

### Task 8: 文档和最终验收

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/aerial_triangulation/README.md`
- Modify: `src/core/matchphototask/README.md`
- Modify: `src/core/sfm/README.md`

- [ ] **Step 1: 更新模块职责文档**

明确 `matchphototask` 唯一生成连接点，`sfm` 只提供算法，`aerial_triangulation` 只编排，`common/project` 定义项目格式与寻址。

- [ ] **Step 2: 执行架构静态检查**

```powershell
rg -n "src/gui|ProjectManager|QWidget|QMessageBox" src/core/aerial_triangulation src/common/project
rg -n "xjw::gui::AerialTriangulation|AerialTriangulationService" src tests
rg -n "AerialTriangulation.*\.cpp" src/gui/cmake src/cli/CMakeLists.txt tests/CMakeLists.txt
```

Expected: 均无不允许的旧依赖或源码直编译记录。

- [ ] **Step 3: 执行全量构建与测试**

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --config Release --parallel $env:NUMBER_OF_PROCESSORS
ctest --test-dir build -C Release --output-on-failure
```

若历史 `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` 仍因 `dom_png not found` 失败，单独记录，不得描述为全量通过。

- [ ] **Step 4: 用 CLI 做 GUI 等价回归**

```powershell
build/bin/Release/aerial_triangulation_cli.exe --help
build/bin/Release/aerial_triangulation_cli.exe <现有 temple/dino 参数>
```

比较 GUI 与 CLI 的解析配置、连接点准备结果、注册相机数、稀疏点数和 BA RMS，要求两者调用同一工作流且结果一致。

## 4. 删除清单

重构完成后必须删除，而不是转发：

```text
src/gui/project/io/ProjectIO.h/.cpp
src/gui/project/support/ProjectSupportUtils.h/.cpp
src/core/aerial_triangulation/AerialTriangulationService.h/.cpp
xjw::gui::AerialTriangulation* 命名空间
Service 内 Phase 1/Phase 2 特征与匹配实现
GUI/CLI CMake 中直接列出的 aerial_triangulation 源文件
针对旧 Service 源码文本的 contract tests
```

## 5. 验收标准

1. GUI、空三 CLI、一键重建 CLI 都只调用 `AerialTriangulationWorkflow`。
2. `matchphototask` 是特征、匹配、几何验证、轨迹和连接点缓存的唯一生成模块。
3. `aerial_triangulation` 中没有特征提取器和匹配器实例化代码。
4. `src/core` 和 `src/common` 不包含对 `src/gui` 的 include 或源文件依赖。
5. `aerial_triangulation` 是独立 target，消费者只链接库。
6. 所有空三类型位于 `xjw::aerial_triangulation`，没有兼容别名。
7. 旧 7097 行 Service 被删除，阶段实现拆成可独立测试的小文件。
8. 测试验证行为和数据结果，不再依赖生产源码字符串。
9. temple/dino 回归中 GUI 与 CLI 使用相同配置时注册数量和质量指标一致。
