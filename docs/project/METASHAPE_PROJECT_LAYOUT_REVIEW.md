# Metashape 工程结构调研与 PlaScan 采用方案

## 样本

本次检查以下 Metashape 2.3.1 工程：

- `E:/code/test/metashape/temple/test.psx`
- `E:/code/test/metashape/9small/test.psx`
- `E:/code/test/metashape/dino/dino.psx`
- `E:/code/test/metashape/hyb2/hyb2.psx`

四个样本均为解包工程，由一个很小的 `.psx` 描述文件和同名 `.files` 数据目录组成。
`.psx` 不保存业务数据，只声明格式版本及主索引位置：

```xml
<document version="1.2.0" path="{projectname}.files/project.zip"/>
```

## 逻辑层次

```text
project.psx
project.files/
├── project.zip                 # 工程级索引、活动 chunk、软件版本
├── lock                        # 可选的运行期锁文件
└── 0/                          # chunk ID
    ├── chunk.zip               # 传感器、相机、组件、坐标系、区域
    └── 0/                      # frame ID
        ├── frame.zip           # 原始影像引用及各产物索引
        ├── thumbnails/
        ├── masks/
        ├── point_cloud/
        ├── depth_maps/
        ├── dense_cloud/
        └── model/
```

大文件不会反复写入工程主索引，而是按产物类型和对象 ID 单独存放。`model.1`、
`point_cloud.10`、`depth_maps.2` 这类目录表示同类产物的多个版本或多个对象。

## 数据类型

| 类型 | 样本中的表示 | 作用 |
| --- | --- | --- |
| 工程描述 | `.psx` | 指向真实数据目录 |
| 工程索引 | `.files/project.zip` | chunk 列表、活动对象、版本信息 |
| Chunk 元数据 | `<chunk-id>/chunk.zip` | 传感器、相机、组件、区域、坐标系 |
| Frame 元数据 | `<chunk-id>/<frame-id>/frame.zip` | 照片路径、缩略图、蒙版和产物引用 |
| 原始影像 | `frame.zip` 中的路径 | 默认是外部引用，不一定复制进工程 |
| 预览缓存 | `thumbnails/` | 可重建的缩略图 |
| 编辑数据 | `masks/`、marker 元数据 | 蒙版、标记和控制信息 |
| 稀疏/密集结果 | `point_cloud/`、`dense_cloud/` | 点云及其分块数据 |
| 深度结果 | `depth_maps/` | 深度图索引和二进制数据 |
| 模型结果 | `model/`、`model.N/` | 一个或多个模型版本 |
| 运行状态 | `lock` | 防止多进程同时写入 |

`temple`、`9small` 和 `dino` 的照片主要使用相对路径；`hyb2` 仍保存了
`G:/Hayabusa2/...` 绝对路径。因此 `.psx + .files` 结构本身强调增量保存和大工程性能，
并不天然保证换电脑后原始照片仍然可用。

## PlaScan 采用方案

PlaScan 只保留一种权威工程类型：

- 入口扩展名：`.plascan`。
- 配套目录：`<name>.files/`。
- 描述文件只保存版本、类型和 `{projectname}.files/project.zip`。
- `project.zip` 只保存版本化 JSON 元数据和资源索引。
- 影像与工作流产物放在 `.files/workspace`，显式资源放在 `.files/resources`。
- 资源使用项目 URI、稳定 ID、大小和 SHA-256，不以本机绝对路径作为权威定位。

旧版单体 `.plascan` 不再作为运行时导入格式，打开时明确拒绝且不改写原文件。
项目移动、复制、重命名和删除均按 `.plascan + .files` 成对处理。

## 不应照搬的部分

- 不使用裸数字目录作为唯一可读标识；内部可用 UUID，索引中保留名称和类型。
- 不允许绝对路径成为工程唯一资源定位方式。
- 不把缩略图、缓存和不可重建业务结果混成同一生命周期。
- 不依赖一个大 XML；继续使用版本化 JSON 清单和独立资源索引。
