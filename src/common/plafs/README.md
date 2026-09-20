# PlaScan 文件系统模块

`plascan_common_plafs` 是项目文件系统的标准 C++ 适配层，位于底层
`plascan_common_file` 和 Qt 项目门面之间。

- `PlaFile`：文件路径、存在性和原子读写；不解析具体格式。
- `PlaDir`：目录路径、创建、空目录检查和非递归成员检查；不提供递归删除。
- `PlaChunkLayout`：集中计算当前 Chunk 的 `assets`、标准产物、恢复缓存和 sidecar 路径；
  `assets/imported/<category>` 与 `assets/packed` 也由它生成。
- `PlaProjectLayout`：集中计算 `.plascan`、`.files`、共享目录和数字 Chunk 的物理路径。

本模块只计算和访问物理路径，不解析项目 XML、不打开 `project.zip`，也不登记 GUI 会话。
Qt 侧 `ProjectPackageLayout` 委托 `PlaProjectLayout`，`ProjectIO`、工作区资源暂存器、资产导入器和
项目资源删除逻辑通过 `ProjectPathBridge` 委托 `PlaChunkLayout`；运行根目录注册和默认 Chunk 选择仍属于
Qt 适配层。复制、归档、事务和递归删除策略仍由项目层负责。
具体格式继续由 `.pimatch`、连接点图、稀疏成果和项目描述各自的格式模块处理。
