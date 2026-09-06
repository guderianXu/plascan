# PlaScan CLI

CLI 源码按业务领域组织，并继续生成独立可执行文件。影像匹配接口不保留旧特征文件参数兼容层；
这样可以避免单个总入口把 Qt、TensorRT、MVS、网格和质量检查等依赖全部绑定在一起。
面向用户的 GUI/CLI 对照、参数语义和快速示例见
[`docs/GUI_CLI_GUIDE.md`](../../docs/GUI_CLI_GUIDE.md)。

## 模块

| 目录 | 职责 | 可执行文件 |
| --- | --- | --- |
| `camera/` | 相机格式导入与转换 | `camera_convert_cli` |
| `control_points/` | 标靶检测与打印 | `marker_detect_cli`, `marker_print_cli` |
| `features/` | CUDA SIFT + TensorRT LightGlue 双影像匹配与连接点生成 | `feature_match_cli`, `match_photos_cli` |
| `dense/` | 极线校正、密集匹配、三角化与点云细化 | `rectify_cli`, `dense_match_cli`, `triangulate_cli`, `dense_cloud_refine_cli` |
| `reconstruction/` | 可独立执行的重建阶段与诊断工具 | `bundle_adjust_cli` |
| `workflows/` | GUI“工作流程”菜单对应的无界面编排入口 | `aerial_triangulation_cli`, `mesh_reconstruct_cli`, `texture_map_cli`, `three_d_reconstruction_cli`, `reconstruct_pipeline_cli` |
| `quality/` | 模型影像质量验收 | `model_quality_cli` |
| `terrain/` | RPC 立体 DEM/DOM | `rpc_stereo_products_cli` |
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
深度重放时可用 `mvs_depth_reprocess_cli --source-max-angle-deg N` 做可复现的 PatchMatch 源视图角度 A/B；
`0` 保持默认场景策略，正值只会收紧而不会放宽场景推导上限，报告和单帧产物会保留实际生效值。
启用 cap 时不会用没有实测三角化角的序列回填绕过该上限，源视图不足会显式留在诊断中。
该 cap 仅限 `patchmatch_source_plan`，不改变独立的环拍几何修复源扩展；角度是最多 2048 个共同可见且在两相机前方的稀疏点三角化角中位数，不是光轴夹角。
完整候选池是默认关闭的专家功能：两个独立 ETH3D 场景都显示它能绕过 legacy early stop、找到更好的
verified source，但 Courtyard 的发布角色仍出现退化，因此尚不能作为默认策略，也不应追加
`sourceQualityScore` 门来掩盖该问题（退化帧的该分数并不更低）。候选池/软排序采用同一二进制的三臂重放：A 不加参数；B 加
`--source-complete-visibility-pool`；C 再加 `--source-angle-soft-ranking-strength 1`。B 只关闭候选评估的
legacy early stop，仍使用原分数；C 仅在同一资格层内使用
`legacy_score*exp(-strength*t)` 重排，不改变资格、质量门、duplicate、fallback 或请求源数。强度范围为
`[0,4]`，正值要求 complete-pool 且 `--source-max-angle-deg 0`，非法组合直接报参数错误。最多 32 视图时
visibility graph 覆盖所有共视或 required pair；更多视图仍使用确定性有界 graph，并不承诺全影像对。
软角度排序只保留为内部诊断开关：Office 结果混合，Courtyard 38/38 与 B 字节级一致且没有任何计划变化，
不应在普通用户工作流、GUI 或默认配置中呈现。
默认 A 的配置 hash、source plan 与 `mvs_replay_report.json` 不新增软排序键；B/C 的逐帧
`source_angle_diagnostics.soft_ranking` 分别记录 control/treatment 候选全集、按 view/tier 的选择标志、
实际分层角度边界、数量不变量和逐候选分数/排名。
启用 cap 后只要实际源数不足，该帧最多作为 `ValidationOnly`（原本拒绝的仍拒绝），不会因一致性确认数随源数降低而成为主融合帧。

PatchMatch revision 45 的三项优化可用同一深度重放二进制做可复现消融：分别传入
`--disable-per-pixel-source-selection`、`--disable-asymmetric-propagation` 和
`--disable-geometric-guidance-pass` 可关闭每像素联合选源、近远邻/局部法线候选和冻结来源深度的第二轮
几何引导。三个开关只用于诊断 A/B，默认均启用；光度来源 bitset 与后续独立几何来源证据始终分开保存。

`mvs_depth_reprocess_cli --stage-snapshot-refs 2,6,22` 可对指定帧保存
PatchMatch 输出、跨视一致性输出、confidence 后处理输出和最终准入输出。每阶段分别保存深度、置信度和
有效掩膜，默认不启用，并受 `--stage-snapshot-max-long-edge` 与
`--stage-snapshot-budget-mib` 限制。快照写入独立 `stage_snapshots/`，manifest 明确标记
`authoritative=false`；写入或预算失败只影响诊断，不会改变生产深度、配置 hash 或准入结果。
跨视一致性及后续阶段还会保存 `depth_layer_reliability` 分类图。内部 A/B 可附加
`--depth-layer-reliability-anchor-gate`，只禁止低纹理歧义/疑似错误层的原生深度充当洞填补锚点，
不直接删除深度；该开关默认关闭且不进入普通用户 GUI。
`--depth-layer-reliability-guided-correction` 是独立的内部默认关闭开关：只有至少三个投影来源独立形成
稳定深度层，且覆盖至少两个基线方向、候选的鲁棒几何代价显著优于原生深度时，才纠正低纹理弱证据。
`AmbiguousLowTexture` 最多连续修正 1%，`RejectedLayer` 才允许严格换层；局部曲面只参与诊断，不会直接
写入产品深度。跨视阶段还会仅在连通弱区调用该帧已经记录的 PatchMatch 后端做两次有界局部假设搜索，
最终仍由同一个三源/双基线/代价优势门裁决，不会因为开关开启而提升 confidence 或准入等级。
跨视阶段快照的可选 `geometry_rerank` 九通道矩阵记录原生/候选代价、优势、有效权重、修正幅度、最弱源
confidence、源数、基线方向数和动作，便于在固定 GT 域中复核每个实际修改像素；后续阶段不重复写入
这份不变的大矩阵，仍各自保存深度、confidence 和有效掩膜。

`workflows/` 按用户可执行的摄影测量任务划分：空中三角测量、无 GUI 三维重建、生成模型和
多视图纹理分别由独立入口覆盖；`reconstruct_pipeline_cli` 覆盖包含 DEM/正射产物的完整流水线。GUI 中的
“添加照片/文件夹”属于项目输入管理，命令行通过输入清单和项目路径表达，不另建重复入口。

`texture_map_cli` 设置里的 `colorCorrection` 默认是 `false`。显式启用时，v4 只使用共同可见的同一
3D 点，在 linear-sRGB 亮度上解算 `0.90–1.10` 范围内的鲁棒标量增益；共同样本不足、高 MAD 或
可靠重叠图按连通分量独立求解，孤立视图 fail-closed 为单位增益；无可靠边时整批保持单位增益。
`texture_result.json` 及 CLI payload 会记录
`texture_exposure_correction_status`、共同观察/候选与通过 pair 数、拒绝原因计数、图连通状态和
最终增益范围，便于区分“未启用”“安全跳过”和“实际应用”。
纹理 v4 默认执行真实共享网格边上的全局 seam leveling：每条共享边采 9 个 linear-sRGB 样本，
全局解算 chart 偏移后只在默认 16px 边界带局部融合，chart 内部保持原烘焙颜色。结果 JSON 记录约束数、
调整 chart/像素数和最大实际线性修正；它不启用整图曝光校正，也不需要 8192² 全图 multiband 缓冲。
固定图集还有空闲空间时，v4 会在不增加图集分辨率和峰值图集内存的前提下，将相机 chart 最多放大 4 倍，
降低亚像素三角面退化为单个中心补色 texel 的比例；专家可用 `atlasUpscaleLimit` 将上限设为 1–4。
锐化默认关闭，避免把 chart 边界和三角面边缘放大；仍可通过 `sharpeningStrength` 显式启用。

`camera_convert_cli --pre-undistort-colmap-images` 是复杂 COLMAP 相机的推荐导入边界。它生成全有效
无畸变 PNG、valid mask、零畸变 Tsai 和逐帧 manifest，支持 `THIN_PRISM_FISHEYE`，不会把复杂参数
传播到 BA、PnP、MVS 或纹理核心。该选项仅接受 `colmap-text`（显式或自动识别）输入。

`aerial_triangulation_cli --export-camera-dir <新目录>` 只在正式 SfM/BA 模型通过质量门并成功
写出后导出最终相机。目标目录可以不存在或已存在但为空，非空目录仍拒绝覆盖；成功后包含与输入影像一一对应的
`cameras/*.tsai` 和可直接传给 `reconstruct_pipeline_cli` 的 `image_camera.lis`。缺失任一最终
相机时整批拒绝，候选搜索阶段的相机不会落盘。
工作流本身最高上报 94%，CLI 在相机导出、报告、结果登记、相机写回和工程保存后再推进到 100%。

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
6. CLI11 入口必须调用 `cli::configureApp(app)`，统一 `--help`、`--version`、帮助布局和退出码说明；
   参数说明使用中文，稳定枚举值仍使用英文 ID。

跨模块复用的进程启动、临时文件、JSON 和测试数据构造工具集中在
`common/tests/CliTestSupport.h`，测试用例本身仍归属具体领域。
