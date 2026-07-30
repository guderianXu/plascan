# PlaScan 工程格式

## 权威布局

PlaScan 只使用一种工程格式，并采用与 Metashape `.psx + .files` 相同的双实体模型：

```text
project.plascan
project.files/
├── project.zip
├── shared/
│   └── images/
│       └── <sha256>/
│           └── <首次导入文件名>
├── 1/
│   ├── chunk.zip
│   ├── assets/
│   ├── reconstruction/
│   ├── bundle_adjust/
│   └── reports/
└── 2/
    ├── chunk.zip
    ├── assets/
    ├── reconstruction/
    ├── bundle_adjust/
    └── reports/
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

原始影像位于工程级共享影像库；特征、匹配、最终多视图 track、深度图、点云、模型、
纹理、DEM、DOM、参考数据和报告是对应 Chunk 数字目录中的普通文件。大文件不进入 ZIP。

每个 Chunk 使用统一目录：

```text
assets/{ip,matches,tie_points,control_points,imported}
reconstruction/{sparse,mvs,model,terrain/products}
bundle_adjust/
reports/
```

`bundle_adjust/` 保存每次 BA 的可复现运行产物，按运行时间或运行模式建立子目录，例如
`ba_run_summary.json`、点/相机误差 CSV、评估图和可选的精化相机文件。CLI 与 GUI 均写入
当前 Chunk 的这个默认目录，不再自动将同类产物放入 `assets/bundle_adjust/`。最终生效
的相机参数仍写入 Chunk 文档，综合工作流报告仍写入 `reports/`。CLI 显式传入
`--output-dir` 时仍尊重用户指定的诊断输出位置。

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
plascan:///workspace/assets/images/example.tif
plascan:///shared/images/<sha256>/example.tif
plascan:///resources/reference/<resource_id>/control.csv
```

`ProjectWorkspaceStore` 在 URI、当前 Chunk 数字目录和工程级共享目录之间转换。
`plascan:///workspace/...` 是首期保留的逻辑兼容 URI，不表示磁盘上仍存在根级
`workspace/`。Chunk `doc.json` 的 `resource_index`
为每个文件记录稳定 ID、类型、项目相对路径、字节数和 SHA-256。打开工程时会校验索引；
资源缺失或损坏会报告具体路径，不会静默回退到其他位置。

归档条目必须使用 `/` 分隔的安全相对路径。任意层级的 `:`、`.`、`..`、控制字符、
尾随点/空格和 Windows 保留名均被拒绝；大小写折叠后指向同一目标的两个条目也视为
冲突。落盘前还会验证规范化路径及已存在父目录仍位于目标根目录内。

影像条目使用稳定 `image_uuid` 作为工程身份。特征和蒙版产物以规范化影像路径的
SHA-256 生成文件键，因此不同目录下的同名影像不会覆盖彼此。按文件名或 stem 查找
只允许唯一候选，存在多个候选时返回歧义错误。

Python、CUDA、模型搜索目录等机器相关配置不能写成工程资源路径。

添加影像时立即计算 SHA-256 并复制到 `shared/images/<sha256>/`。多个 Chunk 导入内容
相同的影像时引用同一个 URI，而不是引用某个可删除 Chunk 的目录。只有全部 Chunk 都
不再引用时才清理共享实体。无法便携化的运行诊断路径写成
`plascan-diagnostic:///<path-token>/<label>`，不保留原电脑盘符。

## 保存与崩溃恢复

`ProjectData` 在内存中维护项目状态。GUI 修改先写入
`.files/<number>/.plascan_tmp/`，再由串行持久化线程防抖同步当前 `chunk.zip`。

显式保存步骤：

1. 影像复制到工程级共享库；其他外部文件复制到当前 Chunk 的
   `assets/imported/<stable-token>/`。
2. 核心和结果元数据中的路径改写为项目 URI。
3. 从元数据引用集合增量更新资源索引；大小和修改时间未变化时复用已有 SHA-256。
4. 将核心、结果、配置和资源索引合并成完整 Chunk `doc.json`。
5. 单次原子替换 `chunk.zip` 条目并推进 Chunk `revision`。
6. 项目 UI 或 Chunk 索引变化时替换根 `doc.json`。

Chunk 内尚未创建的计划路径也会转换为项目 URI。外部目录不递归复制，报告中只保留
去盘符的诊断标识，避免工程自包含循环。

工程打开期间持有 `.files/.plascan.lock`。GUI 与 CLI、两个 CLI 或两个 GUI 不能同时写
同一个工程。

删除项目影像或资源时会删除 `.files` 中对应文件并更新索引。

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
