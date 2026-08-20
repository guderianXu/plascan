# PlaScan 工程格式

## 权威布局

PlaScan 只使用一种工程格式，并采用与 Metashape `.psx + .files` 相同的双实体模型：

```text
project.plascan
project.files/
├── project.zip
├── .shared-image-gc.json       # 按需创建的共享影像 tombstone 代次状态
├── shared/
│   └── images/
│       └── <sha256>/
│           └── <首次导入文件名>
├── 1/
│   ├── chunk.zip
│   ├── assets/                 # 对应流程首次写入时创建
│   ├── reconstruction/         # 对应流程首次写入时创建
│   ├── bundle_adjust/          # 对应流程首次写入时创建
│   └── reports/                # 对应流程首次写入时创建
└── 2/
    ├── chunk.zip
    └── ...                     # 可选目录按需创建
```

`.plascan` 是轻量 XML 入口，业务数据位于同名 `.files`。复制、移动、重命名、备份或
删除工程时必须成对处理两者。工程资源不以原电脑盘符作为权威位置，因此将这一对内容
复制到另一台电脑或另一种操作系统后仍可打开。

描述文件采用固定占位路径：

```xml
<?xml version="1.0"?>
<document version="4.0.0"
          type="plascan_project"
          path="{projectname}.files/project.zip"/>
```

解析器要求根节点、`type="plascan_project"` 和 `version="4.0.0"` 完全匹配，并完整
解析到 XML 结尾。路径只接受固定占位路径或与当前 basename 匹配的相对路径，拒绝
绝对路径、`..`、截断 XML 和更高或更低版本。

## 数据分层

| 层 | 存储位置 | 内容 |
|---|---|---|
| 应用设置 | `QSettings(PlaScan/plascan_gui)` | 窗口几何、最近项目、文件对话框目录 |
| 项目描述 | `project.plascan` | 格式版本、类型、主索引位置 |
| 项目索引 | `project.files/project.zip/doc.json` | 工程身份、Chunk 索引、项目 UI 状态 |
| Chunk 元数据 | `project.files/<number>/chunk.zip` | 核心、结果、工作流配置、资源索引 |
| 共享影像 | `project.files/shared/images/` | 按 SHA-256 跨 Chunk 去重的原始影像 |
| 共享影像 GC 状态 | `project.files/.shared-image-gc.json` | 原子写入的已提交代次和 tombstone |
| Chunk 数据 | `project.files/<number>/` | 特征、匹配、相机旁置文件和工作流产物 |

`project.zip` 只包含一个项目级文档：

```text
doc.json
```

其结构为：

```json
{
  "type": "plascan_project",
  "format_version": "4.0",
  "minimum_reader_version": "4.0",
  "project_id": "<uuid>",
  "created_with": "PlaScan",
  "chunk_index": {
    "schema_version": 1,
    "default_chunk_id": "<chunk-uuid>",
    "next_chunk_directory": 2,
    "chunks": [
      {
        "id": "<chunk-uuid>",
        "name": "区块 1",
        "directory": "1",
        "storage": "1/chunk.zip",
        "order": 0,
        "revision": 0
      }
    ]
  },
  "ui_state": {
    "schema_version": 1,
    "display_settings": {}
  }
}
```

每个数字目录中的 `chunk.zip` 同样只包含一个 `doc.json`：

```json
{
  "type": "plascan_chunk",
  "format_version": "1.0",
  "chunk": {},
  "project_files": {},
  "project_results": {},
  "project_config": {},
  "resource_index": {}
}
```

`project_results` 中每条新结果记录都带独立的 `schema_version`，当前为 `1`。不同结果数组
可独立升级，不再依赖整个 Chunk 格式一起变化。

原始影像位于工程级共享影像库；逐影像匹配分片、最终多视图 track、深度图、点云、模型、
纹理、DEM、DOM、参考数据和报告是对应 Chunk 数字目录中的普通文件。大文件不进入 ZIP。

每个 Chunk 使用统一目录。除 `chunk.zip` 外，这些目录都在对应流程首次写入产物时创建；
新建但尚未处理的 Chunk 不包含空的工作流目录：

```text
assets/{matches,tie_points,control_points,imported/{point_clouds,models}}
reconstruction/{sparse,mvs,model,terrain/products}
bundle_adjust/
reports/
```

### 逐影像匹配结果

匹配结果登记在 `project_results.image_match_results[]`。每条记录对应一幅影像及其唯一 `.pimatch`
分片，而不是一个影像对。`neighbors` 是用于目录树和快速筛选的冗余索引；SfM 以分片内经过校验的
owner/peer 身份和算法变体为准。

```json
{
  "schema_version": 1,
  "image": "plascan:///shared/images/<sha256>/image.tif",
  "output": "plascan:///chunk/assets/image_matches/image_<path-hash>.pimatch",
  "neighbors": ["plascan:///shared/images/<sha256>/neighbor.tif"],
  "settings": {
    "storage_format": "pimatch",
    "format_version": 1,
    "algorithm_id": "auto_sift",
    "algorithm_version": 1
  }
}
```

`.pimatch` 自身按算法变体隔离关键点观测，并包含相邻影像、置信度、残差、几何状态和缓存指纹。工程不登记独立
特征文件，也不接受旧 `.match + JSON sidecar` 作为当前结果。

`bundle_adjust/` 保存每次 BA 的可复现运行产物，按运行时间或运行模式建立子目录，例如
`ba_run_summary.json`、点/相机误差 CSV、评估图和可选的精化相机文件。CLI 与 GUI 均写入
当前 Chunk 的这个默认目录，不再自动将同类产物放入 `assets/bundle_adjust/`。最终生效
的相机参数仍写入 Chunk 文档，综合工作流报告仍写入 `reports/`。CLI 显式传入
`--output-dir` 时仍尊重用户指定的诊断输出位置。

### 正射影像结果

项目内“生成正射影像”成功后，在 `project_results.ortho_results[]` 登记一条结果。输出文件
本身位于 Chunk 普通数据目录并由资源索引管理，不写入 `chunk.zip`。相同输出路径的再次生成
会更新对应结果记录。

`ortho_results[]` 同时容纳两类来源：DEM+影像记录使用 `dem_path`、`images` 和相机统计；彩色
点云记录使用 `point_cloud_path`、`source_surface_type: "point_cloud"`、点数与覆盖统计。
点云全球模式的 `resolved_settings` 必须保存实际采用的体中心、参考半径和中央经线，避免后续
重复生成时悄然改变坐标框架。

典型记录如下；数值仅作字段示例：

```json
{
  "schema_version": 1,
  "created_at": "2026-07-30T12:00:00Z",
  "output_path": "plascan:///chunk/assets/ortho/relative_dom.tif",
  "dem_path": "plascan:///chunk/reconstruction/terrain/products/dem.tif",
  "dem_reference": "relative",
  "images": [
    "plascan:///shared/images/<sha256>/IMG_0001.tif"
  ],
  "source_image_count": 1,
  "resolution": 0.05,
  "output_resolution": 0.05,
  "pixel_size_x": 0.05,
  "pixel_size_y": 0.05,
  "min_x": 100.0,
  "min_y": 200.0,
  "max_x": 612.0,
  "max_y": 584.0,
  "width": 10240,
  "height": 7680,
  "dom_georeferenced": true,
  "projection_wkt_present": true,
  "camera_projected": true,
  "algorithm_version": "ortho_projector_v1",
  "selected_camera_count": 12,
  "loaded_camera_count": 12,
  "contributing_camera_count": 10,
  "filled_pixel_count": 73400320,
  "hole_filled_pixel_count": 128,
  "coverage_ratio": 0.933,
  "valid_pixel_count": 73400448,
  "has_coverage_alpha": true,
  "resolved_settings": {
    "projection_type": "dem_grid",
    "surface_type": "dem",
    "color_source": "images",
    "blend_mode": "mosaic",
    "sizing_mode": "pixel_size",
    "pixel_size_x": 0.05,
    "pixel_size_y": 0.05,
    "maximum_dimension": 4096,
    "bounds_enabled": false,
    "min_x": 100.0,
    "min_y": 200.0,
    "max_x": 612.0,
    "max_y": 584.0,
    "color_correction": true,
    "sharpness_weighting": false,
    "ghost_filter": false,
    "fill_holes": true,
    "hole_fill_max_area": 256,
    "hole_fill_radius": 3.0,
    "use_project_masks": false,
    "maximum_pixel_count": 100000000
  }
}
```

彩色点云全球 DOM 的关键字段示例：

```json
{
  "output_path": "plascan:///chunk/assets/ortho/point_cloud_dom.tif",
  "point_cloud_path": "plascan:///chunk/reconstruction/mvs/dense_cloud.ply",
  "source_surface_type": "point_cloud",
  "algorithm_version": "point_cloud_dom_v1",
  "projection_wkt_present": true,
  "projected_point_count": 12500000,
  "coverage_ratio": 0.81,
  "resolved_settings": {
    "projection_type": "cylindrical",
    "surface_type": "point_cloud",
    "color_source": "point_colors",
    "body_reference_auto": false,
    "body_center_x": 0.12,
    "body_center_y": -0.03,
    "body_center_z": 0.08,
    "reference_radius": 482.6,
    "central_meridian": 0.0
  }
}
```

字段约定：

- `resolved_settings` 保存核心实际采用的参数，不只保存对话框原始输入。像元未指定或使用
  `maximum_dimension` 时，`pixel_size_x/y` 是结合 DEM 网格后解析出的最终值；范围字段也是
  与 DEM 相交并按输出网格对齐后的值。
- 稳定组合为 `dem_grid + dem + images`、`planar + point_cloud + point_colors` 和
  `cylindrical + point_cloud + point_colors`；不接受跨组合参数。`blend_mode` 只对影像颜色源
  生效，支持 `mosaic`、`weighted_average`、`first_valid`。
- `sizing_mode` 支持 `pixel_size`（独立 X/Y 像元）和 `maximum_dimension`（限制最长边）；
  `bounds_enabled` 控制 `min_x/min_y/max_x/max_y` 裁剪。顶层 `resolution` 是旧消费者使用的
  X 向分辨率兼容字段，新代码应读取 `pixel_size_x/y` 和 `resolved_settings`。
- `color_correction`、`sharpness_weighting`、`ghost_filter`、`fill_holes` 和
  `use_project_masks` 分别记录颜色校正、锐度权重、鲁棒重影过滤、小孔洞填充和项目排除蒙版。
- `selected_camera_count` 是已选择影像数，`loaded_camera_count` 是成功加载影像且相机有效的
  数量，`contributing_camera_count` 是至少为一个输出像元提供颜色的相机数。
- `filled_pixel_count` 与 `coverage_ratio` 统计孔洞填充前的直接相机影像覆盖；
  `hole_filled_pixel_count` 单独统计合成填充像元。零直接覆盖时任务失败，因此不会产生只有
  黑色像元的成功记录。
- `.tif/.tiff` 输出为 R/G/B/Alpha 四波段 GeoTIFF，Alpha 的非零值表示直接覆盖或已完成
  小孔洞填充的有效颜色；文件继承最终网格地理变换及对应 DEM/点云投影 WKT。PNG 同样保存
  Alpha 但不保存地理参考。`dom_georeferenced` 和 `projection_wkt_present` 应分别判断，
  不应假设本地坐标 DEM 一定带 WKT。

根 `doc.json` 的 `chunk_index` 持久化 Chunk UUID 与数字目录的映射，以及
`next_chunk_directory`。数字目录只允许正整数并单调分配，已删除的编号永久保留为空洞；
例如已经创建 `1/2/3`，删除 `2` 后下一个目录必须是 `4`。

采用 JSON 而不是 XML 是有意选择：Metashape 值得复用的是“入口文档 → Chunk 文档 →
独立数据”的分层和稳定引用，而不是 XML 语法本身。PlaScan 已有完整的 Qt JSON
读写、校验和测试链路；在同样的 ZIP 压缩下，改成 XML 不会实质减小工程，也不会提升
大数据性能。

## 资源引用

元数据使用与设备无关的项目 URI：

```text
plascan:///chunk/assets/images/example.tif
plascan:///shared/images/<sha256>/example.tif
plascan:///resources/reference/<resource_id>/control.csv
```

`ProjectWorkspaceStore` 在 URI、当前 Chunk 数字目录和工程级共享目录之间转换。
`plascan:///chunk/...` 始终相对于拥有该 `doc.json` 的当前 Chunk 数字目录。
早期 4.0 开发版本写入的 `plascan:///workspace/...` 不再支持；当前版本只接受
`plascan:///chunk/...`。Chunk `doc.json` 的 `resource_index`
为每个文件记录稳定 ID、类型、项目相对路径、字节数和 SHA-256。打开工程时会校验索引；
资源缺失或损坏会报告具体路径，不会静默回退到其他位置。

归档条目必须使用 `/` 分隔的安全相对路径。任意层级的 `:`、`.`、`..`、控制字符、
尾随点/空格和 Windows 保留名均被拒绝；大小写折叠后指向同一目标的两个条目也视为
冲突。落盘前还会验证规范化路径及已存在父目录仍位于目标根目录内。

影像条目使用稳定 `image_uuid` 作为工程身份。特征和蒙版产物以规范化影像路径的
SHA-256 生成文件键，因此不同目录下的同名影像不会覆盖彼此。按文件名或 stem 查找
只允许唯一候选，存在多个候选时返回歧义错误。

U2Net 蒙版记录保存 `mask_method: "u2net"`，并以 `mask_inference_backend`、
`mask_inference_device`、`mask_inference_precision` 记录实际使用的 TensorRT/OpenCV CPU
后端、设备和精度；`mask_model_sha256` 是本次实际加载 ONNX 文件的 SHA-256，不是
绑定开发机的 engine 标识。TensorRT engine 只位于用户本地缓存，不写入工程资源。

Python、CUDA、模型搜索目录等机器相关配置不能写成工程资源路径。

添加影像时立即计算 SHA-256 并复制到 `shared/images/<sha256>/`。多个 Chunk 导入内容
相同的影像时引用同一个 URI，而不是引用某个可删除 Chunk 的目录。只有全部 Chunk 都
不再引用时才清理共享实体。复制完成时会立即建立进程内 active reservation，直到包含
该 URI 的 Chunk 归档代次成功提交；导入、引用发布和共享影像 GC 使用同一个项目级同步
边界。未引用实体会先记录到 `.files/.shared-image-gc.json`，只有连续两个已提交代次均未
引用且没有 active reservation 时才物理删除；该状态文件使用原子替换写入。GC 失败只
记录可重试警告，不会把已经成功提交的项目保存改判为失败。已提交代次由排序后的
`Chunk id + revision` 生成稳定 token，同一 token 的重复 GC 不会推进 tombstone。有效的
`.plascan_tmp/project_files.json` 引用也会阻止删除，覆盖“归档失败但恢复快照已落盘并释放
reservation”的窗口；如果该临时文件存在但不是可信普通文件、不可读、超过 64 MiB 或 JSON 无效，
GC 会保守中止而不是按无引用处理。无法便携化
的运行诊断路径写成
`plascan-diagnostic:///<path-token>/<label>`，不保留原电脑盘符。

## 保存与崩溃恢复

`ProjectData` 在内存中维护项目状态。GUI 修改先写入
`.files/<number>/.plascan_tmp/`，再由串行持久化线程防抖同步当前 `chunk.zip`。
关闭项目或直接析构会先等待已启动的持久化 worker，再同步提交包含最新内存状态的归档快照；
归档失败时以完整临时恢复快照作为后备。只有其中至少一种持久化成功后，才释放共享影像
reservation、运行工作区和 `.plascan.lock`。关闭失败会保留原会话和项目锁，新建/打开目标项目
必须中止切换。异步快照同时携带持久化代次和会话代次，因此旧会话迟到的 queued completion
不能给同路径重开后的新会话恢复 dirty/pending 状态。

显式保存步骤：

1. 影像复制到工程级共享库；其他外部文件复制到当前 Chunk 的
   `assets/imported/<type>/<name_uuid>/`。
2. 核心和结果元数据中的路径改写为项目 URI。
3. 从元数据引用集合增量更新资源索引；大小和修改时间未变化时复用已有 SHA-256。
4. 将核心、结果、配置和资源索引合并成完整 Chunk `doc.json`。
5. 单次原子替换 `chunk.zip` 条目并推进 Chunk `revision`。
6. 项目 UI 或 Chunk 索引变化时替换根 `doc.json`。
7. 归档提交成功后推进共享影像 GC 代次并尝试清理已连续两代无引用的 tombstone。

Chunk 内尚未创建的计划路径也会转换为项目 URI。外部目录不递归复制，报告中只保留
去盘符的诊断标识，避免工程自包含循环。

工程打开期间持有 `.files/.plascan.lock`。GUI 与 CLI、两个 CLI 或两个 GUI 不能同时写
同一个工程，这是共享影像跨进程正确性的主排他边界。共享影像 active reservation 还会持有
临时 `.files/.shared-images.lock`，为直接调用共享库的协作进程提供第二层排他；包含 URI 的归档
提交后才释放该 lease 锁，异常退出遗留锁沿用项目锁的 30 秒恢复窗口。

删除项目影像或资源时会删除 `.files` 中对应文件并更新索引。

## 外部点云与模型导入

“文件 → 导入 → 导入点云/导入模型”面向 Metashape 等软件已经导出的标准成果，不读取
Metashape 的 `.psx`、`.files`、`.oc3` 内部数据。点云支持 OBJ、PLY、XYZ；模型支持带面的
OBJ、PLY。OBJ 中的 `mtllib` 及 MTL 引用的常见纹理会在不越出源目录的前提下保持相对路径，
一并复制到当前 Chunk。

点云登记到 `project_results.dense_cloud_results[]`，主路径键为 `dense_cloud_xyz`；模型登记到
`project_results.model_results[]`，主路径键为 `final_model_path`，并按格式补充 `model_obj` 或
`model_ply`。两类记录共用以下导入字段：

- `source: "metashape_import"`、`imported: true`：标识外部标准成果导入；
- `source_file_name`、`import_format`：只保留源文件名和格式，不保存原电脑绝对路径；
- `import_directory`、`imported_dependencies[]`：工程内主文件、MTL 和纹理的完整引用集合；
- `vertex_count`、`face_count`、`has_vertex_colors`：扫描文件得到的实际统计；
- 模型另含 `has_material`、`textured`，并在存在时写入 `model_mtl`、`texture_image`。

删除工作区中的对应“稠密点云”或“3D模型”结果时，清理服务会逐一删除上述已列举且位于当前
Chunk 受管根目录内的非共享依赖。只有 `model_run_directory/model_run_id/model_runs` 或
`texture_run_directory/texture_run_id/texture_runs` 的 ID 和末两级目录严格匹配，或记录引用通过
校验的 `ownership_manifest_path` 时，才允许递归清理该记录的专属目录；普通 `output_dir`、旧版
`import_directory` 以及仅携带通用 `run_id` 的目录都只移除列举文件，目录和未列举文件保持不变。
点云记录可被现有 DEM 与正射对话框直接选择；带 RGB 的点云可生成局部平面或小天体全球投影
DOM。

`ownership_manifest_path` 若存在，必须指向候选目录内部的普通 JSON 文件，且文件不超过 1 MiB。
清理器只接受 `type: "plascan_owned_directory"`、`schema_version: 1`、`run_id` 与结果记录一致，
并且 `owned_directory` 规范化身份与候选目录完全相同的清单。清单、目录或其祖先涉及符号链接、
Windows junction、外部路径或当前 Chunk 的结构保护目录时，仍不得递归清理。

物理清理采用当前 Chunk 根目录下的
`.plascan_cleanup_trash/<transaction-id>/transaction.json` 恢复事务。清单以原子替换写入，包含
项目路径、Chunk UUID/数字目录、受管根目录，以及每个产物的 `source`、`destination`、类型和
`planned/staged` 状态。事务区还保存 `original_metadata.json`、`updated_metadata.json`，清单记录
两者的 SHA-256；恢复前会重新校验文件大小、哈希与 JSON 类型。每次移动前先登记 `planned`，移动
成功后再登记 `staged`，因此进程在任一移动边界退出后仍能恢复。事务状态为 `staging` 时，下次
清理会按逆序把产物移回原路径；若当前元数据仍匹配 WAL 中的原状态或清理后状态，还会持久化恢复
原元数据，覆盖“新元数据已提交但 committed 标记尚未落盘”的崩溃窗口。若当前元数据已经是第三种
新状态，只恢复物理产物，不覆盖后续编辑。状态为 `metadata_committed` 时仍会先与 WAL 仲裁：当前
元数据等于清理后状态才清空事务区；若回到原状态则恢复产物并重新提交原元数据；第三种状态采用保守
恢复，避免删除后来重新引用的唯一产物。恢复时会重新核对项目与 Chunk 身份、路径边界、文件类型和
链接/Windows junction，任何越界或不确定状态都会停止清理并保留事务区供人工检查。
项目打开会在解析归档、初始化运行工作区和校验资源索引之前执行只依赖项目路径的事务恢复预检，
先把因崩溃停留在事务区的受管产物恢复到原位；项目状态载入后再完成元数据仲裁和事务区清理。

事务已经完成元数据仲裁、只剩不可逆物理清除时，会先把整个事务目录原子改名到同一 Chunk 下的
`.plascan_cleanup_purging/.purging-<uuid>/`，再递归删除。该命名空间位于活动 WAL 事务区之外；即使进程
在递归删除中途退出、清单或元数据快照已被删除，下次打开也只会按严格目录名和实体目录边界继续删除，
不会把残留重新解释为待回滚事务。清除根目录或候选项若是符号链接、Windows junction 或越界路径，
恢复会停止且不会跟随链接删除外部内容。

清理元数据提交会在提交锁内等待当前文件提交完成，再独占推进项目持久化代次。后台保存快照只有在
提交锁内仍属于当前代次时才能替换 Chunk 归档或 `.plascan_tmp`；清理操作确认新代次的归档和恢复快照
均成功后，才把事务标记为
`metadata_committed` 并清空事务区。被清理代次取代的旧 worker 会恢复其未完成的 dirty 标记，
但不能在清理成功后写回旧资源引用。

GUI 发起清理后，文件哈希、事务搬移、Chunk 归档提交和共享影像 GC 在受保护的后台任务中执行；
资源树会暂时禁用重复删除入口并显示后台任务进度。任务完成前禁止新建、打开、保存、关闭项目以及
新建、重命名、删除或切换 Chunk，确保工程锁不会在事务 worker 仍访问文件时释放。应用退出销毁
项目管理器时会等待该不可中途取消的事务完成并执行最终元数据回填。

## CLI 工程会话

`ProjectSession` 是 GUI 无关的工程会话层。连接点、空三和一键重建 CLI 可创建新工程或打开
根索引指定的默认 Chunk；BA CLI 只打开已有工程。工程型 CLI 支持 `--chunk-id` 或
`--chunk-name`，显式选择后该 Chunk 成为默认 Chunk。会话负责：

1. 严格验证 4.0 描述符和两个 `doc.json`；
2. 注册当前 Chunk 数字目录为运行根；
3. 将项目 URI 解析为本地路径；
4. 合并输入影像并保持 `image_uuid`；
5. 写回相机、阶段结果、报告及资源索引。

因此 CLI 生成的工程可以直接由 GUI 打开，GUI 切换默认 Chunk 后 CLI 也会使用该 Chunk，
不会固定假设目录 `1/`。

## 模型属性记录

`model_results` 中的网格记录除模型路径、顶点数和面数外，还保存用于属性面板复现生成过程的
元数据：

- `has_vertex_colors`、`vertex_color_format`：顶点颜色是否存在及其存储格式；
- `depth_generation_parameters`：与模型关联的实际深度图质量、配置/实际邻域数量、筛选模式、
  帧数、累计处理时间和深度产物大小快照；
- `reconstruction_parameters`：表面类型、插值、严格体积掩模、颜色计算、模型质量档位、
  独立的深度质量档位、目标面数和模型处理时间；
- `software_version`：生成模型时的 PlaScan 版本；
- `model_property_schema_version`：模型属性快照结构版本，当前为 `2`；
- `model_run_id`：本次模型运行的唯一 ID；对应目录必须位于当前 Chunk 受管根内，且末两级严格为
  `model_runs/<model_run_id>/`，目录名与 ID 不一致时不得递归清理；
- `model_output_policy`：模型结果登记策略，值为 `create_versioned_result` 或
  `replace_default`；两种策略均创建独立运行目录并追加结果记录；
- `is_default_model`：当前纹理化等后续流程使用的默认模型标记；同一时刻只有一个完整且仍存在的
  模型应为 `true`；
- `model_run_directory`、`model_artifact_directory`：模型运行专属目录及其产品目录；
- `.plascan_task_run.json`：运行目录创建时原子写入的所有权标记，记录任务 ID、运行集合和结构版本；
  取消或会话过期时，只有目录层级、任务 ID 与该标记全部一致，且目录树不含链接或联接点，才允许
  递归清理未发布目录；
- `model_diagnostics_path`：模型文件校验成功后原子发布的本次运行诊断 JSON；
- `texture_run_id`、`texture_run_directory`、`texture_diagnostics_path`：独立纹理运行的 ID、
  `texture_runs/<texture_run_id>/` 专属目录和诊断 JSON；
- `tsdf_required_bytes`：TSDF 布局的内存分配估算，不等同于操作系统观测到的进程峰值内存。

旧工程中不存在的字段保持缺失，GUI 显示“不可用”；能够从仍存在的模型文件或深度记录可靠
恢复的文件大小、深度参数和软件版本会在读取时补充显示，但不会静默改写工程。首次成功登记
schema v2 模型时，如果旧 `model_results` 尚无默认标记，则把数组中最后一个仍存在的旧模型
标记为默认；新建版本不会改变该默认项。`replace_default` 也不覆盖或删除旧模型运行目录，而是等
新模型、可选纹理/MTL 和诊断文件全部验证完成后，仅把旧记录的 `is_default_model` 设为 `false`，
并把新追加记录设为默认。

发布 schema v2 记录前会解析诊断 JSON，并核对 `ok`、`diagnostics_type`、run ID、实体 run 目录以及
记录与诊断中的产物路径；所有本次生成的产物都必须存在且归属于对应 run。模型诊断类型必须为
`model`，独立纹理 run 的诊断类型必须为 `texture`。这些发布约束不影响旧项目中缺少 run 元数据的
已有记录读取。

独立纹理更新沿用模型记录中的 `model_run_id` 与 `model_diagnostics_path`，每次写入新的
`texture_runs/<texture_run_id>/`，不原地覆盖旧纹理目录。只有 OBJ、MTL、PNG 和
`texture_diagnostics_path` 均存在且非空后才更新默认模型记录；失败或取消时旧纹理字段和旧版本
文件保持可用，本次尚未登记的隔离运行目录按上述所有权规则回收。

模型质量与深度质量分别记录。`ultra` 模型默认请求 `highest` 深度，`high`、`medium`、`low`
模型分别请求同名深度档位。生成模型时，只有已有深度批次的质量不低于本次请求时才允许复用；
旧批次未记录深度质量时保持兼容，但新生成的 MVS 清单和逐帧记录会同时保存质量档位、配置源
视角数与实际源视角数，便于属性面板复核真实生效参数。

每个新深度帧同时登记 `missing_reason_path`、`missing_reason_preview_path` 和
`missing_reason_summary`。原因码图是与深度栅格同尺寸的无损单通道 PNG；彩色预览仅显示缺失
像素并保持有效深度与蒙版外区域透明。汇总对象的 `schema_version` 当前为 `1`，包含各原因像素
数、支持区域像素数、缺失像素数和支持区域内缺失率。旧工程缺少这些字段时按“诊断不可用”读取。

## 格式兼容策略

PlaScan 不打开也不自动迁移任何旧工程格式，包括旧版单体 `.plascan` ZIP，以及使用
根级 `workspace/`、`resources/` 和根 `project.zip` 业务元数据的旧版双实体工程。
描述版本或 `doc.json` 格式不匹配时立即拒绝加载，不创建数字目录、不改写归档、不移动或删除
旧资源。需要保留的旧数据必须通过独立转换工具导出后，再导入新建的 Chunk 工程。

## 文件操作规则

- 移动或复制：同时处理 `name.plascan` 和 `name.files`。
- 重命名：两者必须使用相同的新 basename。
- 删除：两者都删除才算真正删除工程。
- 单独复制描述文件无法打开，这是与 Metashape 一致的行为。
- Windows 下应用启动时会为当前用户注册 `.plascan` 文件关联；双击描述文件会启动
  PlaScan，并在主窗口显示后通过标准异步项目加载流程打开工程。

## 验证范围

自动测试覆盖中文路径、严格描述符校验、旧工程拒绝且原文件不变、
Chunk 新建/删除/切换与数字目录不复用、归档路径安全、并发写锁、
跨 Chunk 共享影像去重和引用清理、增量资源索引、
同名影像产物隔离、影像 token 歧义、成对移动、外部源删除、资源校验、工作区索引、
UI/工作流配置分离、CLI 工程会话、活动 Chunk 寻址，以及缺失资源的明确错误。
