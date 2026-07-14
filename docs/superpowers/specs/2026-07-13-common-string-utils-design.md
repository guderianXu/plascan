# Common String Utils 扩展设计

## 目标

在现有 `src/common/string_utils` 模块中补充跨模块复用的标准字符串原语，并消除仓库中已经确认的重复实现。模块继续只依赖 C++17 标准库，不引入 Qt、OpenCV 或业务模块依赖。

本次同时收紧现有浮点数提取 API 的命名和文档，使“按数值子串尽力提取”的真实行为对调用方可见，但不改变相机文件的既有解析结果。

## 方案比较

### 方案 A：保留行为并明确命名，补充三个 ASCII 原语（采用）

- 将 `parseDoublesFromLine` 重命名为 `extractDoublesFromText`。
- 保留正则扫描、标识符数字可能被匹配、越界值被忽略的既有行为。
- 新增 `asciiLowerCopy`、`trimAsciiWhitespace` 和 `endsWithAsciiIgnoreCase`。
- 只迁移已经确认的标准 `std::string` 重复实现。

优点是行为风险最低，同时让接口名称与实际语义一致；缺点是调用方若需要严格数值解析，仍要使用领域内的严格解析器。

### 方案 B：把现有解析器改为严格 token 解析（不采用）

让 `k1` 之类的标识符整体被忽略，并要求每个 token 完整转换为 `double`。该方案契约更严格，但会改变已有容错行为，还需要重新定义 `R=1.0`、逗号分隔和单位后缀等输入，超出本次去重范围。

### 方案 C：同时保留宽松和严格两套解析 API（暂不采用）

该方案扩展性最好，但仓库当前没有第二个适合使用严格公共 API 的调用点。提前引入会增加未被真实需求验证的接口，因此遵循 YAGNI 暂不实现。

## 模块结构

```text
src/common/string_utils/
├── StringParsing.h
├── StringParsing.cpp
├── StringTransform.h
├── StringTransform.cpp
└── test/
    ├── CMakeLists.txt
    ├── StringParsing_tests.cpp
    └── StringTransform_tests.cpp
```

`plascan_common_string_utils` 继续作为独立静态库，并公开 `src/common` 作为 include 根目录。

## 公共接口

### 尽力提取数值

```cpp
bool extractDoublesFromText(const std::string &text, std::vector<double> &out);
```

行为契约：

- 调用开始时清空 `out`。
- 支持带正负号的整数、小数和科学计数法。
- 按数值子串扫描，因此 `k1` 会产生数值 `1`。
- 单个匹配项转换越界时忽略该项，继续处理后续匹配项。
- 至少提取到一个 `double` 时返回 `true`。

该接口只用于允许宽松提取的文本。`CameraFormatConverter.cpp` 中要求固定字段数量并抛出错误的 `parseNumericLine` 保持在原模块，不改用此接口。

### ASCII 转换和比较

```cpp
std::string asciiLowerCopy(std::string_view text);
std::string trimAsciiWhitespace(std::string_view text);
bool endsWithAsciiIgnoreCase(std::string_view text, std::string_view suffix);
```

- 大小写转换只处理 `A-Z`，其它字节保持原样，结果不受当前 locale 影响。
- ASCII 空白集合明确为 ` `、`\t`、`\n`、`\r`、`\f`、`\v`。
- 空后缀始终匹配；后缀长于文本时返回 `false`。
- API 使用 `std::string_view` 接收只读输入；需要生成结果的函数返回拥有所有权的 `std::string`。

## 调用迁移范围

### 迁移到公共原语

- `SparseCloudPreprocessor.cpp` 和 `SparseCloudValidator.cpp` 中完全重复的 `endsWithIgnoreCase`。
- `Camera.cpp`、`CameraFormatConverter.cpp`、`FeatureData.cpp`、`TraditionalFeatureExtractor.cpp`、`TraditionalFeatureMatcher.cpp`、`DepthMapGenerator.cpp` 和 `cli_feature_extract.cpp` 中手写的标准字符串小写转换。
- `CameraFormatConverter.cpp` 的完整 trim，以及 `Camera.cpp` 的行首空白处理；相机字段识别和参数赋值规则不变。

领域函数仍保留原职责。例如 `normalizeAlgorithmName` 继续负责算法白名单和回退，只把其中的 ASCII 小写步骤替换为公共原语。

### 明确不迁移

- `CameraFormatConverter.cpp::parseNumericLine`：严格字段数量和异常语义不同。
- CLI 的 `parseShellTokens`、`parseCsvTokens`、`parseListLine`：依赖 Qt 且包含 CLI 错误语义，应在 `src/cli` 内另行合并。
- `MarkerCsv.cpp` 的 CSV 与有限浮点数解析：属于控制点文件格式逻辑。
- `normalizedImageToken`、`imageTokenMatches` 等：属于路径和影像身份语义，应放到专门的路径辅助模块。
- Qt `QString::trimmed()`、`toLower()`、`split()`：直接使用 Qt 能力，不为它们增加标准字符串包装层。

## 测试与验证

字符串模块测试与实现就近维护在 `src/common/string_utils/test/`，由该目录的
`CMakeLists.txt` 注册独立测试目标。根目录 `tests/CMakeLists.txt` 不再维护字符串模块测试。
测试覆盖：

- 数值提取接口的新名称以及原有宽松行为；
- ASCII 大小写转换，标点和非 ASCII 字节保持不变；
- ASCII trim 的空文本、全空白、左右空白和内部空白；
- 大小写无关后缀匹配的混合大小写、空后缀和过长后缀。

迁移完成后构建所有受影响目标，运行公共字符串、相机、MVS、特征提取和特征匹配相关测试，并使用 `git diff --check` 检查本次涉及文件。若宽泛测试正则选中与本任务无关的历史失败，必须单独报告，不得描述为全量通过。

## 兼容性与风险控制

- `parseDoublesFromLine` 是刚新增且只有仓库内部调用的接口，本次直接重命名，不保留容易继续误用的兼容别名。
- 相机解析继续跳过 `k1/k2/k3/p1/p2` 键名后再提取数值，保证行为不变。
- ASCII API 不承担 Unicode 大小写转换；Qt 字符串继续使用 Qt 自带接口。
- 不修改用户已有的其它工作区改动，不提交、不暂存、不重排无关代码。

## 构建依赖收口

任何直接编译 `FeatureData.cpp` 的目标都必须显式链接 `plascan_common_string_utils`。不能依赖其它业务库偶然传递该依赖，否则目标可能分别出现找不到 `string_utils/StringTransform.h` 或无法解析 `asciiLowerCopy` 的编译、链接错误。

Windows 下由 vcpkg 提供动态库的 QC 测试使用 `gtest_discover_tests(... DISCOVERY_MODE PRE_TEST)`。普通构建只负责编译和链接，不在链接后的构建步骤立即启动测试进程；测试仍由 CTest 在测试阶段发现和运行。
