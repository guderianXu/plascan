# GUI/CLI 共享模型生成入口设计

## 目标

为模型生成增加一个无 GUI 依赖的命令行工具，并保证 CLI 与 Qt GUI 使用同一个核心入口、同一份
`QJsonObject` 参数映射和同一条深度图/点云分发逻辑。CLI 用于可重复地验证 Dino 等数据，避免测试
绕过 GUI 实际生效的网格清理参数。

## 架构

在 `ModelWorkflowService` 中新增 `ModelBuildRequest` 与 `buildModel()`。该入口接收源数据类型、深度图
目录、可复用点云、输出目录、GUI 设置 JSON 和进度回调，内部调用
`reconstructionConfigFromModelSettings()`，再分发到 `buildMeshFromDepthMaps()` 或
`buildMeshAndOptionalTexture()`。

`ProjectModelManager` 只负责解析项目中的源路径和异步生命周期，然后调用 `buildModel()`。新的
`mesh_reconstruct_cli` 负责解析 CLI 参数和 JSON 文件，也调用 `buildModel()`，不复制参数映射。

## CLI

```powershell
mesh_reconstruct_cli.exe `
  --source-data depth_maps `
  --depth-map-dir E:/code/test/dino/mvs_output `
  --dense-cloud E:/code/test/dino/mvs_output/dense_cloud.ply `
  --output-dir E:/code/test/dino/model_cli `
  --settings-json E:/code/test/dino/project_dialog.json `
  --settings-key generate_model
```

CLI 将进度写到标准输出，错误写到标准错误；成功时输出核心服务返回的 JSON payload。退出码 0 表示
成功，2 表示参数或 JSON 错误，3 表示模型生成失败。

## 测试

- 单元测试验证 `buildModel()` 使用 GUI 设置映射并正确分发深度图和点云。
- 源码契约测试验证 GUI 与 CLI 都调用 `buildModel()`。
- CLI 进程测试验证 JSON 读取、输出 payload 和失败退出码。
- Dino 回归必须直接使用 `project_dialog.json` 的 `generate_model` 对象，先复现当前
  `Poisson 网格清理后主体过小`，再修复算法并验证输出 PLY 具有顶点和面。

## 非目标

- CLI 不打开 `.plascan` 项目，不修改项目 metadata。
- 本轮不引入新的第三方网格库。
- 不让任意 3D 模式回退为高度格网。
