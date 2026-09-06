# Agent-first PlaScan GUI 调试

该开发模式的第一使用者是编码 Agent。真实 PlaScan Qt 进程运行在独立 Xvfb 中，Agent 通常通过稳定的 JSON
协议检查状态和执行语义操作；只有布局、绘制或输入链路需要确认时，才打开 Debug Hub/noVNC 看像素画面。
算法、CUDA/TensorRT、GDAL 和工程 IO 仍由本机 Qt 进程执行，浏览器页面不复制业务逻辑。

调试桥仅在 `PLASCAN_BROWSER_TEST=1` 且一次性令牌、本机套接字都有效时启动。普通 PlaScan 会话不会开启该接口。

## 准备与启动

完整隔离运行器当前只支持 Linux。Windows 和 macOS 可以编译调试桥，但尚无对应桌面启动器。

```bash
sudo apt-get update
sudo apt-get install -y xvfb x11vnc novnc websockify openbox
python3 scripts/dev/browser_gui.py --json doctor
test -x build/linux-source-release/bin/plascan
```

没有 GUI 构建产物时使用标准入口：

```bash
python3 scripts/env/configure_with_env.py --source-deps --build
```

Agent 调试 South Building 的推荐入口：

```bash
python3 scripts/dev/browser_gui.py --json start --fixture south_building
python3 scripts/dev/browser_agent.py capabilities
python3 scripts/dev/browser_agent.py inspect
```

`south_building` 默认创建 `build/tmp/browser-gui/runs/<run-id>/case/` 沙箱：复制工程入口、可变元数据和约
215 MiB 共享影像；约 32 GiB 的 assets/MVS 资源创建同尺寸、同时间戳的稀疏占位文件。这样工程能完成完整索引
校验，实际临时占用约 218 MiB，并且表单保存不会写回源夹具。占位文件不含真实派生内容，所以 Agent 策略会拒绝
导入、保存、删除、运行工作流等项目写操作；需要真实计算时使用完整副本：

```bash
python3 scripts/dev/browser_gui.py --json start \
  --fixture south_building --copy-project --allow-large-project-copy
```

完整副本约 32 GiB，只有磁盘空间足够且确实要运行计算时才应创建。直接用 `--project` 打开真实工程不会自动隔离；
Agent 表单和项目写操作都需要显式 `--allow-project-write`。

## Agent 协议

`browser_agent.py` 的输出始终是机器可读 JSON；错误包含稳定 `error.code`、消息和可行时的建议操作。

```bash
# 低 token 摘要；不返回完整控件树、日志和产物数组
python3 scripts/dev/browser_agent.py inspect

# 分页/过滤查询，避免一次吞入完整 UI 树
python3 scripts/dev/browser_agent.py inspect --view artifacts --query mesh --offset 0 --limit 25
python3 scripts/dev/browser_agent.py query --query actionWorkflowAerialTriangulation
python3 scripts/dev/browser_agent.py describe --target actionWorkflowAerialTriangulation

# 按稳定 objectName 操作，不使用屏幕坐标
python3 scripts/dev/browser_agent.py act \
  --target actionWorkflowAerialTriangulation --operation activate

# 复合条件等待与断言
python3 scripts/dev/browser_agent.py wait --timeout 30 --condition \
  '{"all":[{"path":"project.image_count","equals":123},{"modal_present":false}]}'
python3 scripts/dev/browser_agent.py assert --condition \
  '{"not":{"recent_error_contains":"fatal"}}'

# 先整体校验，再依次应用；失败时对已应用字段做尽力回滚
python3 scripts/dev/browser_agent.py form --values \
  '{"m_qualityCombo":1,"m_resetAlignmentCheck":false}'

# 只在 revision 改变时输出一行 JSONL
python3 scripts/dev/browser_agent.py watch --duration 30 --interval 0.5

# 关闭当前模态框、取消具名任务、生成诊断包
python3 scripts/dev/browser_agent.py close-dialog
python3 scripts/dev/browser_agent.py cancel-task --target <task-status-objectName>
python3 scripts/dev/browser_agent.py diagnose
```

`inspect --since <revision>` 返回 `unchanged` 和字段级 `changes`。本地 history 只保留最近 8 个稳定状态。
可分页视图是 `controls`、`logs`、`artifacts`、`tasks`，另有 `summary` 和 `modal`。

路径条件支持 `exists`、`equals`、`not_equals`、`contains`、`gt/gte/lt/lte`，并可用 `all`、`any`、`not`
嵌套。兼容条件包括 `object_name`、`visible`、`enabled`、`text_contains`、`modal_present`、
`modal_title_contains`、`project_open`、`project_dirty`、`image_count`、`task_count` 和
`recent_error_contains`。

表单批处理不是数据库事务：所有目标和操作会在首次写入前校验，运行中失败则尽力按逆序恢复旧值；若 Qt 控件在
回滚期间销毁或拒绝操作，仍可能只完成部分恢复。项目写操作使用保守名称识别，不能替代完整副本。

## HTTP Agent API 与可视化 Debug Hub

启动结果中的 `url` 是 `http://127.0.0.1:6080/?token=<一次性令牌>`。页面 `<head>` 和 `body` 都声明了
`/api/agent/capabilities`，便于自动发现。除 `/api/health` 和静态文件外，API 必须携带 URL token 或
`X-PlaScan-Debug-Token`。

- `GET /api/agent/capabilities`
- `GET /api/agent/summary`、`/modal`
- `GET /api/agent/controls|logs|artifacts|tasks?query=...&offset=0&limit=25`
- `GET /api/events`：SSE 只发送 compact summary 和 revision
- `/api/snapshot`、`/api/ui-tree`、`/api/screenshot`：低层诊断接口

Hub 内嵌真实 noVNC 画面，适合最后确认布局、绘制和鼠标/键盘链路。常规状态判断优先使用 Agent CLI/API，减少
截图识别误差和 token 消耗。

## 声明式场景

可复用回归仍可使用场景运行器；它与 Agent CLI 共用同一套条件判断：

```bash
python3 scripts/dev/browser_gui_scenario.py \
  scripts/dev/browser_gui_scenarios/south_building_open.json
```

失败时会在当前 run 的 `diagnostics/scenario-<时间>/` 保存结果、完整快照、脱敏运行状态和可用时的窗口截图。

## 安全边界与故障定位

- Xvfb 默认使用 `:91`；每个 run 有独立 `profile/`、`logs/`、本机套接字和 ready 文件。
- HTTP/noVNC/VNC 只监听环回地址；HTTP 写请求上限 64 KiB，Qt RPC 上限 256 KiB。
- Qt 桥只允许固定操作和唯一具名控件，不接受任意 Qt 方法、文件路径或 shell 命令。
- `stop` 仅终止状态文件中 PID、启动时间和命令标识都匹配的进程，并只清理匹配该进程 PID 的工程锁。
- 稀疏夹具是 Agent 调试沙箱，不是完整测试数据副本，也不适合启动摄影测量计算。
- noVNC 无独立密码，只适合本机开发，不能通过端口转发或反向代理暴露。

每次运行位于 `build/tmp/browser-gui/runs/<run-id>/`。`logs/` 保存各进程日志，`profile/` 是隔离 XDG 状态，
`diagnostics/` 保存 `summary.json`、完整快照、脱敏 runtime、replay 信息和 Qt PNG。状态异常时先执行：

```bash
python3 scripts/dev/browser_gui.py --json status
python3 scripts/dev/browser_agent.py diagnose
python3 scripts/dev/browser_gui.py --json stop
```
