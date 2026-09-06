# GUI 与 CLI 使用指南

PlaScan 的 GUI 面向交互式项目生产，CLI 面向批处理、自动化和问题复现。两者共享核心算法、
工程格式和稳定参数值；差别只在交互方式，不应维护两套算法实现。

## 选择入口

| 目标 | GUI 入口 | 对应 CLI | 说明 |
| --- | --- | --- | --- |
| 创建连接点 | 工作流程 → 创建连接点 | `match_photos_cli` | 使用同一匹配注册表、预选和连接点轨迹构建 |
| 空中三角测量 | 工作流程 → 空中三角测量 | `aerial_triangulation_cli` | 执行匹配、SfM 和 BA，并写入 Chunk 工程 |
| 完整三维重建 | 分阶段执行空三、点云和模型 | `three_d_reconstruction_cli` | 从影像/相机清单执行 SfM、MVS 和网格阶段 |
| 完整产品流水线 | 分阶段执行模型、DEM、正射 | `reconstruct_pipeline_cli` | 在三维重建后继续生成 DEM/DOM |
| 生成模型 | 工作流程 → 生成模型 | `mesh_reconstruct_cli` | 从点云或深度工件生成网格 |
| 生成纹理 | 工作流程 → 生成纹理 | `texture_map_cli` | 从网格和已定向影像生成纹理模型 |
| RPC 立体产品 | 工作流程 → 创建 DEM → RPC 立体 DEM；工作流程 → 生成正射影像 → RPC 地理正射 DOM | `rpc_stereo_products_cli` | 从带 RPC00B 的 8 位 GeoTIFF 立体像对生成并登记 DEM/DOM、质量栅格和报告 |
| 小天体全球产品 | 小天体地形工作流 | `small_body_terrain_cli` | 从体固连表面生成全球径向 DEM/DOM |
| 单阶段诊断 | 对应报告或检查窗口 | `feature_match_cli`、`bundle_adjust_cli`、`mvs_depth_reprocess_cli` 等 | 用于回归、消融和故障定位 |

GUI 的“添加照片/文件夹”是项目输入管理操作。CLI 使用 `.lis` 清单和 `--project`、
`--chunk-id` 或 `--chunk-name` 表达同一上下文，不复制文件选择界面。

## 一致的交互规则

### GUI

- 工作流对话框统一使用“一般 / 高级”层次、固定底部按钮和可滚动参数区。
- 主按钮描述实际动作，例如“创建”“生成”“开始”；“确定”只用于确认或设置类对话框。
- 自动推导的模型、设备、输出路径和有效参数必须可见。不可用选项保留原因说明，不静默降级。
- 长任务在后台执行，并持续提供阶段、进度、取消和可定位的错误信息。

### CLI

- 所有命令支持 `-h, --help`；CLI11 命令还支持 `--version`，版本输出与 GUI 一致。
- 稳定参数使用 `--kebab-case`，路径参数明确说明是文件还是目录；枚举值使用稳定英文 ID，
  例如 `auto`、`cpu`、`cuda`、`opencl`、`highest`、`high`、`medium`、`low`。
- 正常结果写入 stdout，进度和诊断写入 stderr。可机读命令的最终结果优先使用 JSON。
- 退出码统一为：`0` 成功、`1` 参数错误、`2` 输入/输出错误、`3` 算法执行错误。
- 输出目录默认拒绝覆盖；只有命令明确提供并由用户传入 `--force` 或 `--overwrite` 时才允许复用。

并非每条命令都支持 JSON 配置文件或 `--verbose`。请以该命令的 `--help` 为准，避免把某个
诊断命令的专用选项误认为全局选项。

## 参数语义

GUI 展示中文名称，CLI 和项目 JSON 保存稳定 ID。新增选项时应先确定核心配置字段，再分别做
GUI 控件和 CLI 参数适配。

| 语义 | GUI 示例 | 稳定值 / CLI 值 |
| --- | --- | --- |
| 自动设备 | 自动（CUDA → OpenCL → CPU） | `auto` |
| 计算设备 | CPU / CUDA / OpenCL | `cpu` / `cuda` / `opencl` |
| 对齐照片精度 | 最高 / 高（默认）/ 中 / 低 / 最低 | `highest` / `high` / `medium` / `low` / `lowest`（downscale `0/1/2/4/8`） |
| MVS 质量 | 超高 / 高 / 中 / 低 / 最低 | `highest` / `high` / `medium` / `low` / `lowest` |
| 深度过滤 | 自动 / 温和 / 中等 / 强 | `auto` / `mild` / `moderate` / `aggressive` |
| 匹配算法 | PlaMatch-HCT（默认）/ 自动 SIFT / SIFT + LightGlue / LoMa-R | `plamatch_hct` / `auto_sift` / `sift_lightglue` / `loma_r` |
| PlaMatch 参考预选 | 源位置 / 已估位置 / 照片序列 | `source` / `estimated` / `sequential` |

显式选择的设备或模型不可用时必须失败并说明原因；只有 `auto` 可以按项目定义的优先级选择后端。

## 快速示例

构建后的可执行文件默认位于 `build/<preset>/bin/`。先查看当前二进制实际支持的参数：

```powershell
build\windows-vcpkg-release\bin\aerial_triangulation_cli.exe --help
build\windows-vcpkg-release\bin\aerial_triangulation_cli.exe --version
```

创建连接点并写入工程：

```powershell
match_photos_cli.exe `
  --input image_camera.lis `
  --output-dir output\matches `
  --project output\survey.plascan `
  --quality high `
  --device auto
```

执行空中三角测量：

```powershell
aerial_triangulation_cli.exe `
  --input image_camera.lis `
  --output-dir output\aerial `
  --project output\survey.plascan
```

执行无界面三维重建：

```powershell
three_d_reconstruction_cli.exe image_camera.lis `
  --output-dir output\reconstruction `
  --device auto `
  --mvs-quality high `
  --mvs-backend auto `
  --point-cloud-backend auto
```

Linux/macOS 使用相同参数，只需把可执行文件路径和续行符替换为对应 shell 语法。

## 命令目录

- 相机：`camera_convert_cli`
- 标靶：`marker_detect_cli`、`marker_print_cli`
- 影像匹配：`feature_match_cli`、`match_photos_cli`
- 稠密处理：`rectify_cli`、`dense_match_cli`、`triangulate_cli`、`dense_cloud_refine_cli`
- 重建与约束：`bundle_adjust_cli`、`planetary_linescan_ba_cli`
- 工作流：`aerial_triangulation_cli`、`three_d_reconstruction_cli`、
  `reconstruct_pipeline_cli`、`mesh_reconstruct_cli`、`texture_map_cli`、
  `mvs_depth_reprocess_cli`、`small_body_terrain_cli`
- 地形：`rpc_stereo_products_cli`
- 质量检查：`model_quality_cli`、`mvs_pair_audit_cli`

某些命令只在对应核心模块或可选依赖可用时构建。完整的源码归属和扩展规则见
[`src/cli/README.md`](../src/cli/README.md)，模块边界见
[`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md)。
