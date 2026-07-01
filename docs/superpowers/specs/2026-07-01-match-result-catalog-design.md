# Match Result Catalog and Viewer Redesign

## 背景

PlaScan 当前的匹配查看器直接扫描项目 `assets/matches/*.match` 文件。若同一影像对用多个算法生成过匹配结果，例如 `lightglue`、`sift_bf_l2`、`sift_flann`，主列表会显示多行。这对调试算法有价值，但对普通用户不友好：用户关注的是“这两张影像是否有对应关系、质量如何”，不是底层生成了多少个算法文件。

空中三角测量链路当前不是自动选择“匹配数最多”的结果，而是按本次 SfM 配置中的 `feature_algorithm + match_algorithm` 查找匹配缓存。候选 `.match` 文件还必须通过同名 `.match.json` sidecar 检查，包括特征算法、匹配算法、特征文件路径和 V2 matched index 字段。这个行为可复现，但界面没有把“本次空三实际使用哪个匹配算法”清楚展示出来。

## 目标

1. 匹配查看器主列表按影像对聚合，同一当前影像到同一目标影像只显示一行。
2. 每个影像对默认展示几何验证后有效内点数最多的匹配结果。
3. 进入详细匹配查看后，用户可以切换查看同一影像对下不同匹配算法的结果。
4. 空中三角测量继续默认使用用户当前选择的 `feature_algorithm + match_algorithm`，保持可复现。
5. 空三界面、日志和报告明确显示本次使用的匹配算法和可用匹配缓存情况。

## 非目标

- 不在第一阶段改变 SfM 默认策略为“每对自动选择最佳算法”。
- 不把多个算法的匹配结果直接混合成一个 `.match` 输入 SfM。
- 不改变 `.match` 二进制文件格式。
- 不删除已有历史匹配文件，也不迁移用户工程中的旧数据。

## 核心设计

新增一个轻量的匹配结果目录层，建议命名为 `MatchResultCatalog`。它不负责生成匹配，只负责发现、解析、聚合和排序已有匹配结果。

`MatchResultCatalog` 的输入：

- 当前项目 `.plascan` 路径。
- 当前项目影像列表。
- `assets/matches/*.match` 和同名 `.match.json` sidecar。
- 项目元数据中的 `ipmatch_results` 作为兜底来源。
- `assets/overlap/vocabulary_overlap_pairs.json` 或 `.lis` 作为未匹配重叠候选来源。

`MatchResultCatalog` 的输出：

- 按标准化影像对聚合的 `MatchPairGroup`。
- 每组包含多个 `MatchVariant`，每个 variant 对应一个算法结果。
- 每个 variant 至少记录：
  - `feature_algorithm`
  - `match_algorithm`
  - `match_file`
  - `sidecar_file`
  - `total_matches`
  - `inlier_matches`
  - `outlier_matches`
  - `mtime`
  - `is_compatible_with_current_sfm_config`
  - `failure_reason`

默认最佳结果选择规则：

1. 优先选择 `inlier_matches` 最大的 variant。
2. 若 `inlier_matches` 相同，选择 `total_matches` 更大的 variant。
3. 若仍相同，选择修改时间更新的 variant。
4. 若缺少 sidecar 或无法读取几何内点信息，则该 variant 可展示，但默认排序低于有明确内点统计的 variant。

## GUI 行为

### 匹配对选择器

`MatchPairSelectorDialog` 主表从“算法结果列表”改为“影像对列表”：

- 图像
- 最佳算法
- 有效内点
- 总匹配
- 可用算法数
- 状态

状态示例：

- `已匹配`
- `仅重叠候选`
- `匹配文件缺 sidecar`
- `与当前空三配置不兼容`

当同一影像对存在多个算法结果时，主表只显示最佳 variant，并在“可用算法数”列显示数量。

### 详细匹配查看器

`MatchViewerDialog` 增加一个算法结果下拉框。下拉项显示：

`SIFT + BF-L2：有效 955 / 总 1200`

切换下拉框时重新加载对应 `.match` 文件，并更新状态栏。若当前只传入单个 `.match` 文件，保持兼容旧调用方式。

### 空三提示

空三参数对话框和运行日志应明确显示：

- 当前 SfM 特征算法。
- 当前 SfM 匹配算法。
- 可复用匹配对数量。
- 因算法不兼容被排除的匹配结果数量。
- 缺 sidecar 或缺 matched index 导致不可用于 SfM 的匹配结果数量。

## SfM 行为

第一阶段不改变 SfM 默认选择策略。SfM 仍使用当前配置指定的 `feature_algorithm + match_algorithm`。这样结果可复现，也避免同一重建中不同 pair 混用算法导致 track 质量不可控。

后续可增加一个显式选项：

`匹配结果来源 = 当前算法 / 每对自动选择最佳内点`

该选项默认保持“当前算法”。只有用户显式选择“每对自动选择最佳内点”时，SfM 才按 `MatchResultCatalog` 的 best variant 读取不同算法结果。这个后续选项需要单独增加质量门槛和报告字段，不纳入本次第一阶段实现。

## 错误处理

- `.match` 存在但 sidecar 缺失：GUI 可展示总匹配数，但标记为不可用于正式 SfM track 合并。
- sidecar 算法与文件名算法不一致：以 sidecar 为准，并在诊断中提示。
- sidecar 缺少 V2 matched index：GUI 可看，SfM 不使用。
- 多个 variant 指向同一文件：去重。
- 图像路径无法解析到项目影像：在 GUI 中标记为“影像缺失”，不打开详细查看器。

## 测试计划

1. 单元测试 `MatchResultCatalog`：
   - 同一影像对多个算法聚合为一组。
   - 按有效内点数选择最佳结果。
   - 缺 sidecar 的 variant 排序低于有 sidecar 的 variant。
   - 当前 SfM 配置兼容性判断正确。

2. GUI 结构测试：
   - `MatchPairSelectorDialog` 主表不再因多个算法重复显示同一影像对。
   - 主表显示最佳算法、有效内点、总匹配和可用算法数。
   - `MatchViewerDialog` 支持多 variant 下拉切换。

3. SfM 回归测试：
   - 当前配置为 `sift + sift_bf_l2` 时，只使用兼容的 SIFT BF-L2 结果。
   - 存在 LightGlue 或 SIFT FLANN 结果时，不会被默认混入 SfM。
   - 日志和报告记录可复用、不可兼容和不可用于 SfM 的匹配结果数量。

4. 数据验证：
   - 使用 `agisoft_aerial_gcps_small` 9 张数据，生成多个算法匹配结果后检查 GUI 聚合显示。
   - 使用相同数据跑 `--stop-after-sfm`，确认注册数、稀疏点数和重投影误差与当前配置一致。

## 交付边界

第一批实现只覆盖查看器聚合、多算法切换、SfM 诊断显式化，不改变 SfM 默认匹配选择策略。自动按每对最佳内点混用算法作为后续独立功能处理。
