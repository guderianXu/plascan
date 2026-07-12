# 统一空中三角测量工作流设计

## 目标

让 GUI 与 `aerial_triangulation_cli` 使用同一条“连接点准备 -> SfM -> BA -> 结果写回”核心工作流。
同一组影像、参数、蒙版和项目数据必须解析出相同的连接点配置、候选影像对及 SfM 配置，
避免 CLI 复用旧缓存成功而 GUI 重建连接点失败的假等价验证。

## 当前问题

- GUI 在 `MenuWorkflowController` 中先独立执行 `MatchPhotosTask`，完成后再次进入
  `AerialTriangulationWorkflow`。
- CLI 直接进入 `AerialTriangulationWorkflow`，可由 `AerialTriangulationService` 内部补匹配，
  但该路径与 GUI 的 `MatchPhotosTask` 参数映射不同。
- “重置当前对齐”在 GUI 中会清理匹配并强制重提特征；此前 CLI 回归使用
  `--no-auto-generate-missing-matches`，实际复用了已有特征和匹配。
- GUI 默认蒙版模式为 `keypoints`，此前 CLI 回归使用 `none`。

## 统一边界

`AerialTriangulationWorkflow` 成为空三完整用例边界，按固定顺序执行：

1. 解析一次用户参数，生成连接点配置和 SfM 配置。
2. 根据 `resetAlignment` 和缓存策略决定是否执行连接点准备。
3. 需要准备时，由核心层适配器调用 `MatchPhotosTask`。
4. 连接点成功后，将实际生成的候选对、特征和匹配交给
   `AerialTriangulationService`，服务内部禁止再次生成匹配。
5. 执行增量 SfM、缺口重试和 BA。
6. 合并连接点阶段与 SfM 阶段的结果、诊断和进度。

GUI 和 CLI 只负责收集输入、提供项目路径/蒙版路径以及展示结果，不再分别实现连接点参数映射。

## 配置模型

在 `AerialTriangulationWorkflowOptions` 中增加：

- 项目资源目录、特征目录、匹配目录。
- 影像到蒙版路径映射。
- 是否准备连接点。
- 是否强制重建连接点。
- 可选的手工候选对。

`resolveConfig()` 同时返回：

- `MatchPhotosOptions` 和 `MatchPhotosContext` 所需的解析结果。
- `AerialTriangulationServiceOptions`。
- 可写入报告的统一 `resolvedSettings`。

质量等级、设备、SIFT/LightGlue、关键点限制、连接点限制、指导匹配、蒙版阶段、序列窗口和
固定连接点策略只能在该解析函数中映射一次。

## 重置与缓存语义

- `resetAlignment=true`：必须清理当前算法对应的连接点缓存，强制重提特征并重建匹配；
  不能被 `autoGenerateMissingMatches=false` 覆盖。
- `resetAlignment=false` 且连接点完整：复用缓存，跳过 `MatchPhotosTask`。
- `resetAlignment=false` 且连接点缺失、允许自动补齐：执行 `MatchPhotosTask`，复用兼容特征并补齐匹配。
- 只复用缓存的 CLI 模式保留，但报告必须写入 `tie_point_preparation=skipped_reuse_only`，
  不得再作为 GUI 重置流程的等价回归。

缓存清理只作用于当前项目的特征、匹配、连接点及相关 no-match 记录，不清理影像、蒙版或其它成果。

## 结果与进度

工作流结果增加连接点准备结果摘要，包括：

- 是否执行、是否强制重建。
- 特征文件数、匹配文件数、几何验证通过对数和轨迹数。
- 实际候选对数量及匹配图统计。
- 失败阶段和明确错误信息。

总进度采用固定区间：连接点准备 `0-35%`，SfM/BA `35-100%`。取消标志由两个阶段共享。
连接点准备失败时不进入 SfM；项目切换时 GUI 不写回结果。

## GUI 迁移

- 删除 `MenuWorkflowController` 中“连接点完成后递归启动 SfM”的双阶段编排。
- GUI 一次性构造 `AerialTriangulationWorkflowOptions` 并在后台运行完整工作流。
- 成功后继续使用 `replaceImageCameras()` 原子替换对齐状态，使概览与照片树显示本轮真实注册数。
- 新特征、新匹配和连接点记录从统一工作流结果写回项目。

## CLI 迁移

- `aerial_triangulation_cli` 默认使用统一连接点准备阶段。
- CLI 的参考模式、序列窗口、蒙版目录、关键点/连接点限制和设备参数映射到同一 Workflow 配置。
- `--no-auto-generate-missing-matches` 仅在未重置时表示只复用缓存；与
  `--reset-alignment` 同时出现时，重置语义优先并在报告中明确记录。

## 测试

1. 配置测试：GUI 与 CLI 等价输入解析出相同的连接点和 SfM 参数。
2. 工作流测试：验证连接点准备一定先于服务运行，失败时短路，成功时合并结果。
3. 重置测试：验证重置时强制重提特征、清理旧匹配且服务内部不二次生成。
4. GUI 契约测试：控制器不再直接执行独立 `MatchPhotosTask` 空三前处理。
5. CLI 契约测试：默认报告包含连接点准备摘要及实际生效参数。
6. `dino` 端到端测试：在全新目录中不复用特征/匹配，使用与 GUI 相同的最高精度、照片序列、
   关键点蒙版、40000 关键点和 4000 连接点配置，要求真实注册 16/16，并验证第二次运行结果稳定。

## 不在本次范围

- 不修改 SIFT、LightGlue、SfM 或 BA 的算法实现。
- 不改变用户现有蒙版文件。
- 不把 GUI 类型或 Qt Widget 依赖引入核心模块。
- 不通过插值伪造最终相机位姿。
