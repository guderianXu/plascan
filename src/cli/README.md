# PlaScan CLI

CLI 源码按业务领域组织，并继续生成独立可执行文件。影像匹配接口不保留旧特征文件参数兼容层；
这样可以避免单个总入口把 Qt、TensorRT、MVS、网格和质量检查等依赖全部绑定在一起。

## 模块

| 目录 | 职责 | 可执行文件 |
| --- | --- | --- |
| `camera/` | 相机格式导入与转换 | `camera_convert_cli` |
| `control_points/` | 标靶检测与打印 | `marker_detect_cli`, `marker_print_cli` |
| `features/` | CUDA SIFT + TensorRT LightGlue 双影像匹配与连接点生成 | `feature_match_cli`, `match_photos_cli` |
| `dense/` | 极线校正、密集匹配、三角化与点云细化 | `rectify_cli`, `dense_match_cli`, `triangulate_cli`, `dense_cloud_refine_cli` |
| `reconstruction/` | 可独立执行的重建阶段与诊断工具 | `bundle_adjust_cli` |
| `workflows/` | GUI“工作流程”菜单对应的无界面编排入口 | `aerial_triangulation_cli`, `mesh_reconstruct_cli`, `three_d_reconstruction_cli`, `reconstruct_pipeline_cli` |
| `quality/` | 模型影像质量验收 | `model_quality_cli` |
| `common/` | CLI 共享路径、token、UTF-8 控制台、JSON、输出目录策略与摄影测量列表解析 | 内部静态库，不生成可执行文件 |
| `third_party/` | CLI11 单头文件依赖 | 不生成可执行文件 |

每个领域目录维护自己的 `CMakeLists.txt`、直接依赖和 `tests/`。顶层 `CMakeLists.txt` 只提供统一目标
与测试目标创建函数、输出目录和公共 include 路径。摄影测量列表解析由
`plascan_cli_support` 和 `plascan_cli_photogrammetry_common` 分别编译一次并复用，不再把路径、
JSON、控制台输出、覆盖保护和摄影测量列表解析复制到多个入口。

`workflows/` 的一键重建入口保持很薄：`ReconstructionCliOptions` 负责参数和默认值，
`ReconstructionPipelineRunner` 负责编排，`ReconstructionCliProgress` 与
`ReconstructionCliReport` 负责输出协议。密集点云细化、流式深度融合和点云 PLY 产物写出属于
`src/core/mvs`，CLI 只做参数/工作区适配。

`workflows/` 按用户可执行的摄影测量任务划分：空中三角测量、无 GUI 三维重建和生成模型
分别由前三个入口覆盖；`reconstruct_pipeline_cli` 覆盖包含 DEM/正射产物的完整流水线。GUI 中的
“添加照片/文件夹”属于项目输入管理，命令行通过输入清单和项目路径表达，不另建重复入口。

## Chunk 工程行为

会产生项目状态的 CLI 统一使用 `src/common/project/ProjectSession`，与 GUI 读写同一种
4.0 Chunk 工程，不再把 `headless.plascan` 当作未创建的占位路径：

- `match_photos_cli` 与 `aerial_triangulation_cli`：`--project` 不存在时创建
  `.plascan + .files/project.zip + .files/1/chunk.zip`；存在时打开根索引指定的默认 Chunk。
- `three_d_reconstruction_cli` 与 `reconstruct_pipeline_cli`：在 `--output-dir` 创建或打开
  `headless.plascan`，逐影像匹配、连接点、稀疏、稠密、模型和地形产物写入当前 Chunk 数字目录。
- `bundle_adjust_cli`：只接受已经存在的 4.0 工程，解析 `plascan:///` URI，完成后写回相机和
  `bundle_adjust_results`；默认输出统一位于当前 Chunk 的
  `bundle_adjust/<yyyyMMdd_HHmmss_zzz>/`。
- 所有工程型 CLI 支持 `--chunk-id` 或 `--chunk-name`；两者不能同时使用。显式选择后
  该 Chunk 成为工程默认 Chunk。
- 输入影像立即复制到 `.files/shared/images/<sha256>/`。相同内容只保存一次，所有 Chunk
  使用 `plascan:///shared/...` 引用，不依赖某个可删除 Chunk。
- CLI 保存会把 Chunk 内文件写成项目 URI，并增量更新 `resource_index`。其他外部文件按需
  导入；外部目录不递归复制，持久化报告只保留 `plascan-diagnostic:///` 诊断标识。
- 同一工程具有独占写锁，被 GUI 或另一个 CLI 打开时会明确报错。
- 旧版单体 ZIP、3.x 描述符及根级 `workspace/` 工程直接报错，不迁移、不改写。

## 新增或修改命令

1. 将入口放进最接近的领域目录，不在顶层继续堆放 `.cpp`。
2. 只在该目录的 `CMakeLists.txt` 声明直接依赖；共享业务能力应进入 `src/core` 或 `src/common`。
3. 稳定参数应保持兼容；已经移除的数据格式不得通过隐式转换或旧参数别名重新引入。
4. 公共参数解析和输出约定放进 `common/`，不要在多个入口复制实现。
5. 新增参数时在当前领域的 `tests/` 中补充契约或端到端测试，并由模块自己的 `CMakeLists.txt`
   注册；不要把 CLI 测试放回顶层 `tests/`。

跨模块复用的进程启动、临时文件、JSON 和测试数据构造工具集中在
`common/tests/CliTestSupport.h`，测试用例本身仍归属具体领域。
