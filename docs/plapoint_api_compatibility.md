# PCL 1.15.1 源码兼容性基线

目标是让 PlaPoint **已有且与 PCL 对应的功能**在替换 `pcl` 为 `plapoint` 后仍可编译，并保持调用结果的语义；不以增加 PCL 全库的新算法为目标。比对基准固定为 [PCL 1.15.1 源码 tag](https://github.com/PointCloudLibrary/pcl/tree/pcl-1.15.1)；以后升级版本时须重新审查本表。这里的“兼容”指源码 API，不承诺两个库的二进制 ABI 相同。PlaMatrix 的 `Matrix`/resident API 用在算法内部；PCL 公开签名中出现的 Eigen 类型由公开接口直接提供。

**当前判定：常见调用已有对应接口，但尚未达到任意现有 PCL 程序只改命名空间即可编译。** 下表只审查 PlaPoint 现有功能所对应的 PCL 模块。任何示例编译通过，只证明该示例使用的那部分契约通过。

| 范围 | 已对齐的现有功能 | 尚未对齐的 PCL 调用 |
| --- | --- | --- |
| 点类型 | `PointXYZ`/`PointXYZRGB`/`Normal`/`PointNormal` 的对齐大小、`data`/`data_n`、RGB 打包字段与 Eigen 映射 | PCL 字段 traits、宏注册、自定义点字段反射 |
| `PointCloud<PointT>` | 自定义点容器、对齐 allocator、`PCLHeader`、传感器位姿、拼接、迭代、子集构造、组织点云访问，以及 `resize`/`assign`/`insert`/`erase`/`transient_*` 和 `getMatrixXfMap` | 元数据边界与高级浮点映射场景；矩阵属性和网格数据由独立的 `GeometryCloud<Scalar>` 持有，不占用 `PointCloud<PointT>` 模板签名 |
| 搜索 | `search::Search<PointT>` 多态入口的名称、排序选项、单点、索引、批量以及异型 XYZ 查询的 KNN/半径搜索，`search::KdTree<PointT>` 的输入子集和平方距离；根命名空间 `KdTreeFLANN<PointT>` 入口 | `KdTreeFLANN` 的自定义距离模板参数、自定义点表示和其他搜索实现 |
| 体素 | `Filter<PointT>` 基类调用，`VoxelGrid<PointT>` 的点云输入/索引子集/输出、标量及 `Eigen::Vector4f` 叶尺寸、`Eigen::Vector3f` getter | 最小点数、叶布局、下采样所有字段等其他 PCL 选项 |
| 离群点过滤 | `Filter<PointT>` 基类调用，`StatisticalOutlierRemoval<PointT>` 与 `RadiusOutlierRemoval<PointT>` 的常用参数、搜索方法注入、输入索引子集、负向选择、移除索引和组织输出；索引只限定查询点，邻域仍覆盖整云 | PCL `PCLBase`/`FilterIndices` 的完整继承接口、线程选项、极端输入时的逐分支数值语义 |
| 法线 | `Feature<PointInT, PointOutT>` 基类调用，`NormalEstimation<PointInT, PointOutT>` 的邻域、搜索表面、索引、视点和输出字段 | `PCLBase` 的完整继承接口和其他特征算法 |
| 配准 | `Registration<PointSource, PointTarget, Scalar>` 基类调用和类名查询，`IterativeClosestPoint` 的 Eigen 变换入参/返回、基本迭代参数、互反对应和两个 `getFitnessScore` 重载 | 增量变换、对应拒绝器/估计器、RANSAC 及完整 `Registration` 扩展接口 |
| PLY I/O | 点类型的 PLY ASCII/二进制读写、选中索引保存与返回码 | `PCLPointCloud2`、网格等 PCL 数据模型与重载 |

逐项来源：[PointCloud](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/common/include/pcl/point_cloud.h)、[点类型](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/common/include/pcl/impl/point_types.hpp)、[KdTree](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/search/include/pcl/search/kdtree.h)、[VoxelGrid](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/filters/include/pcl/filters/voxel_grid.h)、[NormalEstimation](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/features/include/pcl/features/normal_3d.h)、[Registration](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/registration/include/pcl/registration/registration.h)、[PLY I/O](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.15.1/io/include/pcl/io/ply_io.h)。

## 验收门槛

1. 固定要迁移的调用源码和 PCL 1.15.1；验收只覆盖 PlaPoint 已有功能。迁移时 `#include <pcl/...>` 仍需改为 `#include <plapoint/...>`，因为头文件目录并未改为 `pcl`。
2. 对每个目标调用编写同一份 C++ 契约源文件，分别以 PCL 1.15.1 与 PlaPoint 编译。PlaPoint 版只允许替换命名空间（以及在允许时替换 include/链接配置）；对返回类型、重载、模板参数和公共字段做编译期断言。
3. 对同一输入运行 PCL 与 PlaPoint，核对点数/字段、邻域顺序与距离平方、体素输出、法线方向/曲率、ICP 变换与 fitness、I/O 往返。浮点容差单独规定，不能用编译通过代替语义验证。
4. PlaPoint 内部继续使用 PlaMatrix 新的 Eigen 风格矩阵接口和 CPU/GPU resident 后端；公开 Eigen 边界显式转换，Eigen 已作为公开依赖纳入 CMake/package 配置。

未完成以上门槛之前，README 和示例只能称为“部分 PCL 风格 API”，不能称“只改命名空间即可迁移”。

当前已提供 `<plapoint/point_cloud.h>`、`<plapoint/point_types.h>`、`<plapoint/header.h>`、`<plapoint/exceptions.h>` 和 `<plapoint/features/normal_3d.h>` 等入口。PlaPoint 源码树不保留以 `pcl` 命名的文件，公开类型 `PCLHeader` 仍按 API 契约保留；原代码直接包含 `PCLHeader.h` 时需改为 `<plapoint/header.h>`。PlaPoint 的构建不依赖 PCL。
