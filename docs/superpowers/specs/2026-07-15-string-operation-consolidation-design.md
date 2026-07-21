# 字符串操作职责收口设计

## 目标

在不扩大 `plascan_common_string_utils` 职责的前提下，消除当前工作树中五类重复字符串规则：影像 token 匹配、匹配对 key 编解码、CLI 列表分词、特征后缀映射和特征模型文件名目录。

## 模块边界

- `src/common/string_utils` 继续只提供不依赖 Qt 的 ASCII 与数值子串原语，本次不增加 Qt 依赖。
- `src/common/project/ProjectMetadata.*` 维护影像 token 的路径、文件名和 stem 匹配语义。
- `src/common/project/ProjectMatchCatalog.*` 维护无序影像对的 canonical 化，以及换行、`__`、`|` 三种既有 key 格式的编解码。
- `src/cli/cli_photogrammetry_common.*` 维护带引号、转义和 CSV 语义的列表行解析。
- `src/core/feature_match/AlgorithmCompat.h` 维护特征后缀归一化及后缀到特征算法的映射。
- `src/common/model/FeatureExtractorModelCatalog.*` 维护 SuperPoint、DISK、ALIKED 模型候选名和托管模型识别。

## 行为约束

### 影像 token

路径比较先 trim、统一分隔符并 `QDir::cleanPath`，比较时使用 `QString::toCaseFolded()`；若完整 token 不同，再按文件名和 `completeBaseName()` 比较。保留现有“仅 stem 也可匹配影像”的宽松行为。

### 匹配对 key

canonical API 对两个非空且不同的 token 排序；编码 API可选择保留输入顺序。解码要求恰好两个非空字段。现有三种分隔符保持不变，避免项目文件和 UI 设置兼容性变化。

### CLI 列表解析

保持现有反斜杠转义、单双引号、双引号 CSV 转义、尾随空单元格以及中文错误信息。`cli_reconstruct_pipeline.cpp` 删除私有副本，调用公共 CLI 接口。

### 特征后缀

公共接口同时接受 `.dsk`、`dsk` 和 `image.dsk`，统一返回带点的小写后缀；提供去重列表和后缀到特征算法的唯一映射。

### 模型目录

CUDA 请求按 CUDA、CPU、无设备后缀顺序回退；CPU 请求按 CPU、无设备后缀顺序查找。对话框显示与 runner 实际解析必须使用同一候选列表。

## 验证

每类规则先补模块测试并确认旧实现下失败，再实现最小公共接口并迁移调用方。最终构建 `plascan_common_project`、相关 CLI、`test_algorithm_compat`、GUI 测试目标和模型 catalog 测试，并运行针对性 CTest。

不修改第三方代码，不提交、不暂存，也不回滚工作区中的其它改动。
