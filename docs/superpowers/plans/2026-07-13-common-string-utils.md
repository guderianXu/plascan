# Common String Utils Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 明确现有宽松浮点数提取 API 的真实语义，新增三个无 locale 依赖的 ASCII 字符串原语，并迁移仓库中已确认的重复实现。

**Architecture:** `plascan_common_string_utils` 保持为只依赖 C++17 标准库的静态库。`StringParsing` 负责按子串尽力提取 double，`StringTransform` 负责 ASCII 小写、ASCII 空白裁剪和 ASCII 大小写无关后缀比较；业务模块只复用基础原语，领域校验继续留在原模块。

**Tech Stack:** C++17、CMake 3.16+、GoogleTest/CTest、`std::regex`、`std::string_view`。

**Repository constraint:** 用户已明确同意在当前 `main` 工作区原地实施。工作区有大量既有改动；只修改本计划列出的相关代码，不暂存、不提交、不回滚其它改动。

---

## File map

- Modify: `src/common/string_utils/StringParsing.h/.cpp` — 将宽松提取接口重命名为 `extractDoublesFromText`，保持行为不变。
- Create: `src/common/string_utils/StringTransform.h/.cpp` — 三个 ASCII 字符串原语。
- Modify: `src/common/CMakeLists.txt` — 编译新的转换实现。
- Create: `src/common/string_utils/test/StringParsing_tests.cpp` — 锁定宽松提取契约。
- Create: `src/common/string_utils/test/StringTransform_tests.cpp` — 覆盖三个 ASCII 原语。
- Create: `src/common/string_utils/test/CMakeLists.txt` — 模块内注册两个测试目标。
- Modify: `tests/CMakeLists.txt` — 删除已迁出的字符串测试目标。
- Modify: `src/core/camera/Camera.cpp`, `src/core/camera/CameraFormatConverter.cpp` — 使用解析与转换公共接口。
- Modify: `src/core/mvs/DepthMapGenerator.cpp`, `SparseCloudPreprocessor.cpp`, `SparseCloudValidator.cpp`, `CMakeLists.txt` — 移除重复小写/后缀实现。
- Modify: `src/core/feature_extractors/FeatureData.cpp`, `tradition/TraditionalFeatureExtractor.cpp`, `CMakeLists.txt` — 复用 ASCII 小写。
- Modify: `src/core/feature_match/tradition/TraditionalFeatureMatcher.cpp`, `CMakeLists.txt` — 复用 ASCII 小写。
- Modify: `src/cli/cli_feature_extract.cpp`, `src/cli/CMakeLists.txt` — 复用 ASCII 小写并声明直接依赖。
- Modify: `docs/PROJECT_ARCHITECTURE.md` — 登记 `StringTransform`。

### Task 1: Write failing public-contract tests

**Files:**
- Modify: `tests/test_common_string_parsing.cpp`
- Create: `tests/test_common_string_transform.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Rename parsing test calls before production code changes**

Replace every `parseDoublesFromLine` call with `extractDoublesFromText`. Rename `PreservesEmbeddedIdentifierDigitBehavior` to `ExtractsDigitEmbeddedInIdentifier` and keep the expected `{1.0, 0.25}` behavior. This makes the public-contract test fail until the production declaration is renamed.

- [ ] **Step 2: Add focused transform tests**

Create `tests/test_common_string_transform.cpp`:

```cpp
#include <string>

#include <gtest/gtest.h>

#include "string_utils/StringTransform.h"

namespace
{

using xjw::common::string_utils::asciiLowerCopy;
using xjw::common::string_utils::endsWithAsciiIgnoreCase;
using xjw::common::string_utils::trimAsciiWhitespace;

TEST(CommonStringTransformTest, LowercasesOnlyAsciiUppercaseLetters)
{
    std::string input;
    input.push_back(static_cast<char>(0xC3));
    input.push_back(static_cast<char>(0x84));
    input += "AZ-9";

    std::string expected;
    expected.push_back(static_cast<char>(0xC3));
    expected.push_back(static_cast<char>(0x84));
    expected += "az-9";
    EXPECT_EQ(asciiLowerCopy(input), expected);
}

TEST(CommonStringTransformTest, TrimsAsciiWhitespaceAtBothEnds)
{
    EXPECT_EQ(trimAsciiWhitespace("\t alpha beta \r\n"), "alpha beta");
    EXPECT_EQ(trimAsciiWhitespace("\t\r\n\f\v "), "");
    EXPECT_EQ(trimAsciiWhitespace(""), "");
}

TEST(CommonStringTransformTest, MatchesSuffixIgnoringAsciiCase)
{
    EXPECT_TRUE(endsWithAsciiIgnoreCase("cloud.PLY", ".ply"));
    EXPECT_TRUE(endsWithAsciiIgnoreCase("cloud.ply", ""));
    EXPECT_FALSE(endsWithAsciiIgnoreCase("ply", ".ply"));
    EXPECT_FALSE(endsWithAsciiIgnoreCase("cloud.xyz", ".ply"));
}

} // namespace
```

- [ ] **Step 3: Register the transform test target**

Add after `test_common_string_parsing` in `tests/CMakeLists.txt`:

```cmake
add_executable(test_common_string_transform test_common_string_transform.cpp)
target_link_libraries(test_common_string_transform PRIVATE
    plascan_common_string_utils
    GTest::gtest_main
)
target_include_directories(test_common_string_transform PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
)
gtest_discover_tests(test_common_string_transform)
```

- [ ] **Step 4: Verify the red state**

Run:

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel --target test_common_string_parsing test_common_string_transform
```

Expected: compilation fails because `extractDoublesFromText` and `string_utils/StringTransform.h` do not exist. The failure must be caused by these missing APIs, not an unrelated environment error.

### Task 2: Implement the minimal common APIs

**Files:**
- Modify: `src/common/string_utils/StringParsing.h`
- Modify: `src/common/string_utils/StringParsing.cpp`
- Create: `src/common/string_utils/StringTransform.h`
- Create: `src/common/string_utils/StringTransform.cpp`
- Modify: `src/common/CMakeLists.txt`

- [ ] **Step 1: Rename and document the parsing API**

Declare and define:

```cpp
bool extractDoublesFromText(const std::string &text, std::vector<double> &out);
```

Keep the existing regular expression, output clearing, exception handling and return semantics unchanged. Update the Doxygen text to state explicitly that this is best-effort numeric-substring extraction, identifier digits may be extracted, and out-of-range matches are ignored.

- [ ] **Step 2: Add the transform header**

Create `StringTransform.h` with C++17 declarations:

```cpp
#pragma once

#include <string>
#include <string_view>

namespace xjw::common::string_utils
{

std::string asciiLowerCopy(std::string_view text);
std::string trimAsciiWhitespace(std::string_view text);
bool endsWithAsciiIgnoreCase(std::string_view text, std::string_view suffix);

} // namespace xjw::common::string_utils
```

Document that only ASCII `A-Z` and the six ASCII whitespace characters have special treatment.

- [ ] **Step 3: Add the minimal locale-independent implementation**

Implement private helpers that compare characters directly against `'A'`/`'Z'` and explicitly recognize ` `, `\t`, `\n`, `\r`, `\f`, `\v`. `asciiLowerCopy` returns an owned copy, `trimAsciiWhitespace` returns an owned substring, and `endsWithAsciiIgnoreCase` compares without allocating a lowercase copy.

- [ ] **Step 4: Compile the new source**

Add `string_utils/StringTransform.cpp` to `plascan_common_string_utils` in `src/common/CMakeLists.txt`. Do not add Qt or other link dependencies.

- [ ] **Step 5: Verify the green state**

Run:

```powershell
cmake --build build --parallel --target test_common_string_parsing test_common_string_transform
ctest --test-dir build --output-on-failure -R "CommonStringParsing|CommonStringTransform"
```

Expected: both targets build and all focused tests pass.

### Task 3: Migrate camera and MVS callers

**Files:**
- Modify: `src/core/camera/Camera.cpp`
- Modify: `src/core/camera/CameraFormatConverter.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/SparseCloudPreprocessor.cpp`
- Modify: `src/core/mvs/SparseCloudValidator.cpp`
- Modify: `src/core/mvs/CMakeLists.txt`

- [ ] **Step 1: Migrate camera parsing and text normalization**

In `Camera.cpp`, include `StringTransform.h`, replace the using-declaration and all calls with `extractDoublesFromText`, replace manual left trim with `trimAsciiWhitespace(line)`, and replace the character loop with `asciiLowerCopy(s)`. Preserve `startsWithKey` and the existing `k1/k2/k3/p1/p2` delimiter slicing.

In `CameraFormatConverter.cpp`, include `StringTransform.h`, delete local `trim` and `toLower`, and replace their uses with `trimAsciiWhitespace` and `asciiLowerCopy`. Keep `normalizedFormatName` and its underscore-to-hyphen domain rule local.

- [ ] **Step 2: Migrate MVS suffix and error-message normalization**

In both sparse-cloud files, include `StringTransform.h`, delete local `endsWithIgnoreCase`, and call `endsWithAsciiIgnoreCase`. Remove `<cctype>` only where it becomes unused.

In `DepthMapGenerator.cpp`, replace the manual lowercase construction in the CUDA out-of-memory classifier with `asciiLowerCopy(message)`.

- [ ] **Step 3: Link MVS directly to the utility**

Add `plascan_common_string_utils` to the `PRIVATE` section of `target_link_libraries(mvs ...)`.

- [ ] **Step 4: Build and run focused regressions**

Run:

```powershell
cmake --build build --parallel --target test_camera_unit test_camera_tsai test_camera_format_converter mvs
ctest --test-dir build --output-on-failure -R "CameraUnitTest|CameraTsaiLoaderTest|CameraFormatConverterTest|Mvs"
```

Expected: affected targets build. Report any broad-regex historical failure separately.

### Task 4: Migrate feature and CLI callers

**Files:**
- Modify: `src/core/feature_extractors/FeatureData.cpp`
- Modify: `src/core/feature_extractors/tradition/TraditionalFeatureExtractor.cpp`
- Modify: `src/core/feature_extractors/CMakeLists.txt`
- Modify: `src/core/feature_match/tradition/TraditionalFeatureMatcher.cpp`
- Modify: `src/core/feature_match/tradition/CMakeLists.txt`
- Modify: `src/cli/cli_feature_extract.cpp`
- Modify: `src/cli/CMakeLists.txt`

- [ ] **Step 1: Replace extractor lowercase copies**

Include `StringTransform.h`. Replace both `FeatureData.cpp` transform loops and the lowercase step inside `TraditionalFeatureExtractor::normalizeAlgorithmName` with `asciiLowerCopy`. Keep all algorithm classification and fallback behavior unchanged.

- [ ] **Step 2: Replace matcher lowercase copy**

Include `StringTransform.h` and use `asciiLowerCopy` inside `TraditionalFeatureMatcher::normalizeAlgorithmName`. Keep matcher whitelist and fallback behavior unchanged.

- [ ] **Step 3: Replace CLI lowercase copy**

Include `StringTransform.h` and construct `requestedAlgo` with `asciiLowerCopy(algo)`. Keep the DeDoDe special case unchanged.

- [ ] **Step 4: Add direct target dependencies**

Add `plascan_common_string_utils` privately to `feature_extractors_traditional` and `feature_match_traditional`. Add it to the `feature_extract_cli` library list in `src/cli/CMakeLists.txt` because the CLI source directly calls the API.

- [ ] **Step 5: Build affected targets and run related tests**

Run:

```powershell
cmake --build build --parallel --target feature_extractors_traditional feature_match_traditional feature_extract_cli
ctest --test-dir build --output-on-failure -R "Feature|Match"
```

Expected: affected targets build. If the broad test regex includes unrelated failures, list them rather than masking them.

### Task 5: Documentation and final verification

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Verify: every file in the file map

- [ ] **Step 1: Document the expanded module**

Change the common tree entry to:

```text
├── string_utils/
│   ├── StringParsing.h/cpp # 文本中的 double 数值子串尽力提取
│   └── StringTransform.h/cpp # ASCII 小写、空白裁剪和后缀比较
```

- [ ] **Step 2: Confirm old duplication is gone**

Run:

```powershell
rg -n "parseDoublesFromLine|bool endsWithIgnoreCase|std::tolower\(" src tests
```

Expected: no matches in the migrated scope. Any remaining `std::tolower` outside the declared files must be reviewed and reported, not changed automatically.

- [ ] **Step 3: Run whitespace diagnostics**

Run `git diff --check` limited to the file map. Expected: no whitespace errors introduced by this task.

- [ ] **Step 4: Perform fresh affected build and tests**

Run:

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel --target test_common_string_parsing test_common_string_transform test_camera_unit test_camera_tsai test_camera_format_converter mvs feature_extractors_traditional feature_match_traditional feature_extract_cli
ctest --test-dir build --output-on-failure -R "CommonString|CameraUnitTest|CameraTsaiLoaderTest|CameraFormatConverterTest|Mvs|Feature|Match"
```

Read the full exit codes and failure count before making any completion claim.

- [ ] **Step 5: Audit final scope without staging or committing**

Run `git status --short` and inspect focused diffs for all file-map paths. Confirm the implementation did not stage, commit, or overwrite unrelated user changes.

### Task 6: Relocate string tests beside the module

**Files:**
- Create: `src/common/string_utils/test/StringParsing_tests.cpp`
- Create: `src/common/string_utils/test/StringTransform_tests.cpp`
- Create: `src/common/string_utils/test/CMakeLists.txt`
- Delete: `tests/test_common_string_parsing.cpp`
- Delete: `tests/test_common_string_transform.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/common/CMakeLists.txt`
- Modify: `docs/PROJECT_ARCHITECTURE.md`

- [ ] **Step 1: Move the test sources without changing behavior**

Copy the parsing tests verbatim to `StringParsing_tests.cpp` and the transform tests verbatim to
`StringTransform_tests.cpp`, then remove the two old root-level test files.

- [ ] **Step 2: Register tests in the module directory**

Create `src/common/string_utils/test/CMakeLists.txt` that finds GTest, includes `GoogleTest`, defines
the existing `test_common_string_parsing` and `test_common_string_transform` targets, links
`plascan_common_string_utils` and `GTest::gtest_main`, and calls `gtest_discover_tests` for both.

- [ ] **Step 3: Connect module tests and remove global registration**

Under `if(BUILD_TESTS)` in `src/common/CMakeLists.txt`, add:

```cmake
add_subdirectory(string_utils/test)
```

Remove only the two string-test target blocks from `tests/CMakeLists.txt`; preserve all unrelated tests.

- [ ] **Step 4: Update architecture documentation**

Add the `test/` directory and its three files below `string_utils/` in `docs/PROJECT_ARCHITECTURE.md`.

- [ ] **Step 5: Reconfigure and verify relocated tests**

Run:

```powershell
cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON
cmake --build build/windows-vcpkg-cuda-release --parallel --target test_common_string_parsing test_common_string_transform
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R "^CommonString"
```

Expected: both targets build from `src/common/string_utils/test`, and all 12 tests pass.
