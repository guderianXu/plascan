# PlaScan CLI

CLI 源码按业务领域组织，但继续生成独立可执行文件。这样可以保持现有脚本和自动化调用兼容，
同时避免单个总入口把 Torch、Qt、MVS、网格和质量检查等依赖全部绑定在一起。

## 模块

| 目录 | 职责 | 可执行文件 |
| --- | --- | --- |
| `camera/` | 相机格式导入与转换 | `camera_convert_cli` |
| `control_points/` | 标靶检测与打印 | `marker_detect_cli`, `marker_print_cli` |
| `features/` | 特征提取、匹配与连接点生成 | `feature_extract_cli`, `feature_match_cli`, `match_photos_cli` |
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

`workflows/` 以 GUI 菜单语义划分，而不是按底层算法划分：空中三角测量、三维重建和生成模型
分别由前三个入口覆盖；`reconstruct_pipeline_cli` 覆盖包含 DEM/正射产物的完整流水线。菜单中的
“添加照片/文件夹”属于项目输入管理，命令行通过输入清单和项目路径表达，不另建重复入口。

## 新增或修改命令

1. 将入口放进最接近的领域目录，不在顶层继续堆放 `.cpp`。
2. 只在该目录的 `CMakeLists.txt` 声明直接依赖；共享业务能力应进入 `src/core` 或 `src/common`。
3. 保留现有可执行文件名和参数兼容性；需要统一编排时由脚本或工作流 CLI 调用独立阶段。
4. 公共参数解析和输出约定放进 `common/`，不要在多个入口复制实现。
5. 新增参数时在当前领域的 `tests/` 中补充契约或端到端测试，并由模块自己的 `CMakeLists.txt`
   注册；不要把 CLI 测试放回顶层 `tests/`。

跨模块复用的进程启动、临时文件、JSON 和测试数据构造工具集中在
`common/tests/CliTestSupport.h`，测试用例本身仍归属具体领域。
