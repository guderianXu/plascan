# String Operation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将五类重复字符串规则收口到正确的 project、CLI、feature_match 和 model 模块，并保持现有文件格式兼容。

**Architecture:** `string_utils` 保持 std-only。Qt 路径和影像对协议进入 `common/project`，列表分词进入 CLI common，特征后缀进入 `AlgorithmCompat`，模型文件名进入 `common/model`。

**Tech Stack:** C++17、Qt6 Core、CMake、GTest、CTest。

---

### Task 1: 统一影像 token 语义

**Files:**
- Modify: `src/common/project/ProjectMetadata.h`
- Modify: `src/common/project/ProjectMetadata.cpp`
- Modify: `src/common/project/test/test_project_metadata.cpp`
- Modify: `src/gui/dialogs/ForwardIntersectionCheckDialog.cpp`
- Modify: `src/gui/widgets/DualImageViewer.cpp`
- Modify: `src/gui/widgets/MatchValidityAnalyzer.cpp`
- Modify: `src/gui/dialogs/MatchPairSelectorDialog.cpp`
- Modify: `src/core/matchphototask/runtime/MatchPhotosRuntime.cpp`

- [ ] **Step 1: 添加失败测试**

测试以下公共接口：

```cpp
QString normalizedImageToken(const QString &token);
QString imageBaseToken(const QString &token);
bool imageTokensReferToSameImage(const QString &lhs, const QString &rhs);
```

覆盖大小写、反斜杠、文件名和 stem 匹配。

- [ ] **Step 2: 运行 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_common_project_metadata --parallel 3
```

Expected: 编译失败，提示新接口尚未声明。

- [ ] **Step 3: 实现接口并迁移调用方**

实现统一 case-folded token，并删除各调用文件中的同名私有路径匹配函数。

- [ ] **Step 4: 运行 GREEN**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_common_project_metadata test_gui_project_utils test_match_photos_task --parallel 3
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R 'ProjectMetadata|GuiProjectUtils|MatchPhotos'
```

### Task 2: 统一影像对 key 编解码

**Files:**
- Modify: `src/common/project/ProjectMatchCatalog.h`
- Modify: `src/common/project/ProjectMatchCatalog.cpp`
- Modify: `src/common/project/test/test_project_match_catalog.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/gui/dialogs/MatchPairSelectorDialog.cpp`
- Modify: `src/gui/dialogs/FeaturePairPlanner.cpp`

- [ ] **Step 1: 添加失败测试**

测试以下接口：

```cpp
QString encodeImagePairKey(const QString &first,
                           const QString &second,
                           const QString &separator);
QString canonicalImagePairKey(const QString &left,
                              const QString &right,
                              const QString &separator);
bool decodeImagePairKey(const QString &key,
                        const QString &separator,
                        QString *first,
                        QString *second);
```

覆盖排序、保序编码、三种分隔符和非法字段数。

- [ ] **Step 2: 运行 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_common_project_match_catalog --parallel 3
```

Expected: 编译失败，提示 key API 尚未声明。

- [ ] **Step 3: 实现并替换手工拼接/split**

保留现有换行、`__` 和 `|` 格式，只统一 canonical、编码和解码规则。

- [ ] **Step 4: 运行 GREEN**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_common_project_match_catalog test_feature_pair_planner test_gui_project_utils --parallel 3
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R 'ProjectMatchCatalog|FeaturePairPlanner|GuiProjectUtils'
```

### Task 3: 统一 CLI 列表分词

**Files:**
- Modify: `src/cli/cli_photogrammetry_common.h`
- Modify: `src/cli/cli_photogrammetry_common.cpp`
- Modify: `src/cli/cli_reconstruct_pipeline.cpp`
- Modify: `src/cli/CMakeLists.txt`
- Create: `tests/test_cli_photogrammetry_common.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 添加失败测试**

公开并测试：

```cpp
bool parsePhotogrammetryListLine(const QString &line,
                                 QStringList *parts,
                                 QString *errorMessage);
```

覆盖 shell 空白、CSV、双引号转义、未闭合引号和尾随逗号。

- [ ] **Step 2: 运行 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_cli_photogrammetry_common --parallel 3
```

Expected: 目标或接口不存在。

- [ ] **Step 3: 实现公共入口并删除 reconstruct 私有副本**

两个 reconstruction CLI 目标加入 `cli_photogrammetry_common.cpp`，`readImageCameraList` 调用公共解析器。

- [ ] **Step 4: 运行 GREEN**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_cli_photogrammetry_common reconstruct_pipeline_cli three_d_reconstruction_cli test_cli_contracts --parallel 3
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R 'CliPhotogrammetryCommon|CliContracts'
```

### Task 4: 统一特征后缀和算法映射

**Files:**
- Modify: `src/core/feature_match/AlgorithmCompat.h`
- Modify: `src/core/feature_match/tests/test_algorithm_compat.cpp`
- Modify: `src/gui/dialogs/FeatureMatchingDialog.cpp`
- Modify: `src/gui/dialogs/InitCameraPoseDialog.cpp`
- Modify: `src/gui/project/manager/ProjectCameraSetupManager.cpp`

- [ ] **Step 1: 添加失败测试**

补充裸后缀、路径、去重列表和以下接口：

```cpp
QString featureAlgorithmForSuffix(const QString &pathOrSuffix);
QStringList normalizedFeatureSuffixes(const QStringList &suffixes);
```

- [ ] **Step 2: 运行 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_algorithm_compat --parallel 3
```

Expected: 编译失败，提示新接口不存在或裸后缀行为不符。

- [ ] **Step 3: 实现并删除 GUI 私有映射**

保留 `.dsk/.alk/.sp/.sift/.orb/.akz/.dedode` 的既有映射。

- [ ] **Step 4: 运行 GREEN**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_algorithm_compat test_gui_project_utils --parallel 3
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R 'AlgorithmCompat|GuiProjectUtils'
```

### Task 5: 统一特征模型候选目录

**Files:**
- Create: `src/common/model/FeatureExtractorModelCatalog.h`
- Create: `src/common/model/FeatureExtractorModelCatalog.cpp`
- Create: `src/common/model/test/CMakeLists.txt`
- Create: `src/common/model/test/FeatureExtractorModelCatalog_tests.cpp`
- Modify: `src/common/CMakeLists.txt`
- Modify: `src/gui/dialogs/FeatureExtractionDialog.cpp`
- Modify: `src/gui/tasks/FeatureExtractionRunner.cpp`
- Modify: `docs/PROJECT_ARCHITECTURE.md`

- [ ] **Step 1: 添加失败测试**

测试：

```cpp
QStringList featureExtractorModelCandidates(const QString &algorithm, bool useCuda);
bool isManagedFeatureExtractorModelPath(const QString &path);
```

确认 CUDA 候选后跟 CPU 和通用回退。

- [ ] **Step 2: 运行 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_feature_extractor_model_catalog --parallel 3
```

Expected: 目标或实现不存在。

- [ ] **Step 3: 实现 catalog 并迁移对话框与 runner**

删除两处模型候选名和托管前缀副本，统一使用 catalog。

- [ ] **Step 4: 运行 GREEN**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_feature_extractor_model_catalog test_gui_project_utils --parallel 3
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R 'FeatureExtractorModelCatalog|GuiProjectUtils'
```

### Task 6: 集成验证

**Files:**
- Verify all files above.

- [ ] **Step 1: 构建受影响目标**

```powershell
cmake --build build/windows-vcpkg-cuda-release --parallel 3 --target plascan_common_project plascan_common_model reconstruct_pipeline_cli three_d_reconstruction_cli test_algorithm_compat test_gui_project_utils
```

- [ ] **Step 2: 运行相关测试**

```powershell
$vcpkg = (Resolve-Path 'build/windows-vcpkg-cuda-release/vcpkg_installed/x64-windows/bin').Path
$torch = (Resolve-Path 'build/env/libtorch-cu130/libtorch/lib').Path
$env:PATH = "$vcpkg;$torch;$env:PATH"
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R 'CommonProject|ProjectMetadata|ProjectMatchCatalog|CliPhotogrammetryCommon|CliContracts|AlgorithmCompat|FeatureExtractorModelCatalog|FeaturePairPlanner|GuiProjectUtils|MatchPhotos'
```

- [ ] **Step 3: 检查重复定义和补丁质量**

```powershell
rg -n 'bool parseShellTokens|QString normalizeFeatureSuffix|QString normalizedImageToken' src
git diff --check
git diff --cached --quiet
```

Expected: 仅保留公共实现，无空白错误，没有暂存改动。根据用户约束不提交。
