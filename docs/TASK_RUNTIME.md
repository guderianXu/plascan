# PlaScan 任务运行时与迁移边界

本文记录 `task_runtime` 首版的稳定契约、现有工作流能力审计，以及 GUI/自动化接入方式。它描述的是当前真实实现，
不是把现有取消标志包装成可恢复任务的功能承诺。

## 当前架构

`src/core/task_runtime` 是不依赖 Qt 的任务领域层。`TaskDefinition` 描述任务、依赖、优先级、项目代次和资源需求；
`TaskRunSnapshot` 描述一次运行的状态、revision、进度、检查点、结果和错误。`TaskScheduler` 负责：

- priority + 稳定 FIFO 排队、依赖 DAG 校验和缺失依赖阻塞；
- CPU slot、加速器 slot、项目读写 lease 的准入和释放；
- 协作式暂停、恢复、取消、队列相对移动，以及带 revision 的并发修改保护；
- 结构化事件订阅、项目 generation 发布门禁、终态历史清理；
- 把执行委托给按 `kind` 注册的 `ITaskExecutor`，不认识 GUI 对话框或具体摄影测量算法。

`src/gui/runtime/TaskRuntimeService` 是 Qt 适配层。它把项目 path、Chunk 和 generation 映射为 epoch guard，向 Work Pane
和 Browser Agent 提供同一份 JSON snapshot/命令入口，并把 journal 放在：

```text
<project>.files/task_runtime/<chunk-id>.journal
```

journal 当前 schema 为 v2，以临时文件加替换方式写入；保存队列、运行状态、检查点引用、结果摘要和结构化错误。
载入时，遗留的 Running、PauseRequested、CancelRequested 会转换为 Interrupted，绝不会伪装成仍在执行。

## 状态与命令语义

主要路径为 `Queued/Blocked -> Running -> Succeeded/Failed/Cancelled`。支持暂停的 executor 在安全点收到 pause 后，
保存检查点并返回 `Paused`；scheduler 随即释放 worker 和资源 lease。`resume` 将同一个 RunId 重新排队、重新申请资源，
并把检查点传回 executor。系统不会挂起线程、冻结 CUDA kernel 或序列化 C++ 栈。

只有任务声明的能力会暴露给 UI/Agent：

- `canPause`：executor 能在已声明安全点结束当前执行片段；
- `canCheckpoint`：Paused/Interrupted 后有可验证的业务检查点；
- `canReorder`：仅 Queued/Blocked 可移动，不能抢占 Running；
- `canCancel`：任务会协作轮询取消，完成中的不可中断第三方调用仍可能有延迟。

命令可携带 snapshot revision。revision 过期时返回 `revision_conflict`，调用方应刷新后再决定，避免两个自动化客户端
互相覆盖顺序或状态。

## 工作流能力审计与迁移矩阵

| 工作流 | 现有控制 | 首版对外能力 | 安全暂停/恢复边界 | 当前迁移状态 |
|---|---|---|---|---|
| 项目打开、保存、影像列表加载 | Qt runner/future，部分 generation 门禁 | 不暂停；是否取消沿用旧入口 | 归档事务结束后才可停 | 未迁移；继续由既有生命周期控制器管理 |
| 影像导入 | 异步导入和项目写事务 | 暂不暂停 | 一批文件复制及元数据提交完成后 | 未迁移；缺少批次 checkpoint 契约 |
| 特征提取与匹配 | `MatchPhotosContext::cancelFlag`，逐对/逐阶段轮询 | 可取消；暂不恢复 | 影像特征或像对持久化完成后 | 未迁移；需要定义缓存签名和完成集合 |
| 空中三角测量/BA | 共享 atomic cancel flag | 可取消；暂不恢复 | SfM attempt 或 BA 阶段边界 | 未迁移；第三方求解调用内部不可安全挂起 |
| MVS 逐影像深度 | cancel flag、`DepthComputeScheduler`、`MvsWorkspaceManifest` | 目标为帧边界暂停/恢复 | 单帧成果原子保存且 manifest `markCompleted` 后 | 基础设施已具备，生产 executor 尚未接入 |
| 深度融合/稠密点云 | streaming cancel flag | 可取消；暂不恢复 | fusion window 或完整输出临时文件边界 | 未迁移；没有可重放窗口 checkpoint |
| 网格、纹理、模型 | task lifecycle + cancel flag/generation | 能力随旧任务而定，不声称可暂停 | 第三方阶段完成与临时成果发布边界 | 未迁移；需逐算法证明可恢复性 |
| DEM/DOM/全球地形产品 | manager/runner + cancel flag | 能力随旧任务而定，不声称可暂停 | 独立瓦片/产品事务完成后 | 未迁移；尚无统一瓦片 manifest |

这里的“未迁移”意味着 Work Pane 仍展示旧任务历史，但暂停、恢复、重排按钮保持禁用；它们不会被转换成虚假的
scheduler-managed 任务。首个生产迁移对象仍应是 MVS 逐帧深度，因为现有 manifest 已记录配置 hash、算法 revision 和
完成帧，能够在帧边界建立可验证的 checkpoint。

## 接入 executor

生产 executor 必须在不持有业务锁或未提交项目事务时调用 `context.control.pollAtSafePoint()`。收到 Pause 时应先原子保存
业务 manifest，再通过 `context.saveCheckpoint()` 报告仅包含 schema、位置、输入签名和完成单元数的轻量引用，最后返回
`TaskExecutionStatus::Paused`。收到 Cancel 时清理本次未发布的临时文件并返回 Cancelled。Succeeded 前必须检查
`context.isProjectEpochCurrent()`；scheduler 还会在接收结果时再次检查，阻止旧项目代次发布。

不要在 executor 内使用 `waitIfPaused()` 长期占住 worker；该方法只用于无法立即改造的兼容执行体。新的可检查点任务应
返回 Paused，让 scheduler 释放所有 lease。

## GUI 与自动化接口

Work Pane 对 scheduler-managed 行显示上移、下移、暂停/恢复和取消，并按 snapshot capabilities/state 启用。
旧任务行继续可见，但不显示不存在的能力。Browser Agent 复用同一个 `TaskRuntimeService`：

```bash
python3 scripts/dev/browser_agent.py task-command --action pause --run-id <run-id> --revision <revision>
python3 scripts/dev/browser_agent.py task-command --action resume --run-id <run-id> --revision <revision>
python3 scripts/dev/browser_agent.py task-command --action move_before --run-id <run-id> \
  --reference-run-id <other-run-id> --revision <revision>
```

任务查询继续使用 Agent 的 `inspect --view tasks`、`query`、`wait` 和 `assert`。在线命令通过本机受令牌保护的调试桥进入
持有 scheduler 的 GUI 进程；不允许通过直接编辑 journal 冒充在线控制。

## 尚未完成的边界

- 尚无生产摄影测量 `ITaskExecutor` 注册，因此现有业务按钮不会自动进入新队列；当前 executor 覆盖来自核心和 GUI 集成测试。
- 尚未提供通用 `submit` 调试 RPC。任意 payload 提交需要先建立受限任务 kind/schema 注册表和权限校验。
- 项目切换会请求 scheduler shutdown，并依赖 executor 协作退出；对忽略取消的第三方调用还没有安全的有界 drain。
- journal schema v1 不自动迁移到 v2。此功能尚未发布过，当前选择明确拒绝未知 schema，避免误恢复。
- 没有操作系统线程挂起、CUDA kernel 冻结、跨进程/跨机器 worker、多用户调度或分布式资源租约。

后续迁移每个真实工作流时，应同时补齐 checkpoint 失配、取消清理、项目切换、恢复不重复发布以及 GUI/Agent 能力测试，
通过后才把该任务标为 scheduler-managed。
