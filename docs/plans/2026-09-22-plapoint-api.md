# PlaPoint 与 PCL 源码级 API 契约对齐及 PlaMatrix Eigen 风格接口迁移

状态：现有功能的常用公开 API 已按 PCL 1.15.1 对照，PlaPoint 数值层也已迁移到 PlaMatrix 的 `Matrix`/`ResidentMatrix`；**尚未达到任意 PCL 1.15.1 程序只改命名空间即可迁移的目标**。准确差异和验收门槛见 [PCL 1.15.1 兼容性基线](../plapoint_api_compatibility.md)。工作区已有 PlaPoint、PlaMatrix 和 PlaScan 的未提交改动；本计划只记录本任务的增量，不能据此重置其他改动。

## 目标与边界

- 让点类型、类模板、公开方法签名和结果语义与 PCL 的常用调用代码一致：`PointXYZ`、`PointXYZRGB`、`Normal`、`PointCloud<PointT>`、体素滤波、KNN/半径搜索、法线估计、点到点 ICP、PLY 输出。验收标准是 PCL 示例仅替换命名空间和头文件即可编译；逐项记录仍不兼容的点。
- 新增的矩阵计算和公开矩阵入参使用 PlaMatrix 的 `Matrix<Scalar, Rows, Cols>`、固定向量、`SelfAdjointEigenSolver` 等 Eigen 风格接口；设备工作区迁移时使用 `ExecutionContext`/`ResidentMatrix`。不在新接口里继续暴露 `DenseMatrix<Scalar, Device>`。
- PlaPoint 不依赖 PCL。以 PCL 官方接口为契约来源；对齐范围包含 `PointCloud<PointT>` 类型层面，不能只靠同名方法冒充兼容。
- 保留点云 Nx3、可选属性、确定性输出和现有 CPU/CUDA 行为；任何破坏性语义变化必须同步迁移 PlaScan 消费点和测试。

## 实施前基线与已确认事实

以下是 API 改造前发现的问题，用于解释本轮迁移的原因。

- PlaMatrix 新高层密集入口已经具备动态/固定尺寸矩阵、向量、系数访问、视图、表达式和 CPU/CUDA 自动执行；自定义 GPU 内核仍应使用 resident 层。
- PlaPoint 的 KdTree 单点查询已接受 `Matrix<Scalar, 3, 1>`，CPU/OpenCL 法线协方差已使用 `Matrix` 和自伴特征分解；`PointCloud` 公共属性及大量 CUDA 内核仍绑定旧 `DenseMatrix`。
- PCL `pcl::search::KdTree` 的单点 KNN 是 `(point, int k, indices, std::vector<float>& squared_distances)`；半径搜索是 `(point, double radius, indices, std::vector<float>& squared_distances, unsigned max_nn)`，返回值为数量。PlaPoint 当前单点查询只返回索引。
- PCL ICP 的 `getFitnessScore()` 是平均距离平方，PlaPoint 当前同名方法是内点比例。这两处不能只改签名而不处理语义。
- 当前 `PointCloud<Scalar, Device>`、`NormalEstimation<Scalar, Device>`、`IterativeClosestPoint<Scalar, Device>` 的第二模板实参是设备枚举，无法直接实例化 `PointCloud<PointXYZ>`、`NormalEstimation<PointXYZ, Normal>`、`IterativeClosestPoint<PointXYZ, PointXYZ>`。这是类型体系迁移，不能通过简单别名解决。
- 当前 `PointCloud::points()` 返回列优先矩阵，PCL `PointCloud::points` 是点结构容器；点字段访问和迭代器也不同。需要明确公开点对象与矩阵后端之间的数据同步、宽高、属性及 GPU 驻留契约。

## 目标 API 验收样例

下列代码只覆盖首轮基本调用，不能代表完整源码兼容性。用户已确定最终公开命名空间为根 `plapoint`；示例中的 `point_api` 即 `plapoint`。这些调用形态已经由对应的 `*PclApiTest` 编译并执行，但后续必须用 PCL 1.15.1 的真实公开类型、重载和语义逐项验收。

```cpp
namespace point_api = plapoint;
point_api::PointCloud<point_api::PointXYZ>::Ptr cloud(
    new point_api::PointCloud<point_api::PointXYZ>);
cloud->push_back(point_api::PointXYZ(0.0f, 0.0f, 0.0f));
point_api::search::KdTree<point_api::PointXYZ> tree;
tree.setInputCloud(cloud);
std::vector<int> indices;
std::vector<float> squared_distances;
tree.nearestKSearch(cloud->points[0], 1, indices, squared_distances);

point_api::VoxelGrid<point_api::PointXYZ> voxel;
voxel.setInputCloud(cloud);
voxel.setLeafSize(0.1f, 0.1f, 0.1f);
point_api::PointCloud<point_api::PointXYZ> filtered;
voxel.filter(filtered);

point_api::NormalEstimation<point_api::PointXYZ, point_api::Normal> normals;
normals.setInputCloud(cloud);
normals.setKSearch(10);
point_api::PointCloud<point_api::Normal> normal_cloud;
normals.compute(normal_cloud);

point_api::IterativeClosestPoint<point_api::PointXYZ, point_api::PointXYZ> icp;
icp.setInputSource(cloud);
icp.setInputTarget(cloud);
point_api::PointCloud<point_api::PointXYZ> aligned;
icp.align(aligned);
const plamatrix::Matrix4f transform = icp.getFinalTransformation();
const double mean_squared_error = icp.getFitnessScore();
```

## 迁移依赖与验收

| 阶段 | 类型和契约 | 依赖 | 验收 |
| --- | --- | --- | --- |
| A | `PointXYZ`、`PointXYZRGB`、`Normal` 和 `PointCloud<PointT>` 的点字段、`points`、`Ptr`/`ConstPtr`、`width`/`height`、`is_dense`、索引与迭代 | 根命名空间；与旧 `PointCloud<Scalar, Device>` 的冲突处理 | PCL 点云构造和访问代码编译；组织云尺寸及属性测试 |
| B | `search::KdTree<PointT>` 的 `setInputCloud(cloud, indices)`、KNN/半径、`getInputCloud`/`getIndices` | A；旧矩阵搜索迁移 | PCL 搜索示例只替换命名空间/头文件；平方距离、排序、子集和 CPU/GPU 测试 |
| C | `VoxelGrid<PointT>`、`NormalEstimation<PointInT, PointOutT>` 的输入、参数和输出点云 | A、B；曲率与法线方向定义 | PCL 滤波和法线示例编译；输出字段、点数、空/非法输入测试 |
| D | `IterativeClosestPoint<PointSource, PointTarget, Scalar>`、4x4 变换、MSE `getFitnessScore()`、终止阈值 | A、B、PlaMatrix 新固定矩阵；PlaScan 内点率调用迁移 | PCL ICP 示例编译；同一数据的变换、MSE、阈值行为对照 |
| E | PLY I/O 同名返回码接口、PlaScan 调用迁移、旧 API 收敛 | A-D | CPU 全量测试、可执行 CUDA 测试、PlaScan 消费方构建与回归 |

若以“仅修改命名空间”为验收标准，公开 API 中的向量与 4x4 变换必须是 PCL 使用的 Eigen 类型；PlaMatrix 新 `Matrix`/resident 层承接内部计算，边界进行显式转换。当前公开 API 仍是 PlaMatrix 类型，属于待迁移项。阶段验收必须区分“源码编译通过”和“数值语义一致”。

## 实施阶段

### 1. API 契约与依赖审计（已完成）

- 列出 PlaPoint 对外 API、PlaScan 调用点、PlaMatrix 新接口与 PCL 对应语义；标明可增量修改和必须协同迁移的接口。
- 给搜索结果规定排序、距离平方、空输入、非有限查询、`k > size` 与半径上限的行为。
- 给 ICP 指标规定内点比例、RMSE、均方误差和终止阈值各自的名称与单位。

### 2. 搜索 API 首轮落地（已完成 CPU 契约）

- 在 `KdTree` 上提供与 `pcl::search::KdTree` 一致的 `nearestKSearch(query, int k, indices, std::vector<float>& squared_distances)` 与 `radiusSearch(query, double radius, indices, std::vector<float>& squared_distances, unsigned max_nn)`；查询点使用 PlaMatrix 固定向量。
- 审核 `setInputCloud()` 是否需要如 PCL 一样在调用后可直接搜索，不能把额外的 `build()` 前置步骤留在“兼容”示例中。
- 沿用现有 CPU/GPU 搜索后端及确定性顺序，避免新增逐点 GPU 往返；检查输出向量为空、边界参数和异常时状态。
- 增加针对距离平方、数量、排序、并列距离和非法参数的单元测试；更新 README 调用示例。

### 3. 点类型、点云容器和法线输出直接迁移（PCL 点类型 CPU 路径已完成）

- 在 PlaMatrix resident API 与 PlaPoint CUDA 内核的内存/stream 契约稳定后，将 `PointCloud` 公开的坐标和属性类型直接迁到新 `Matrix`，设备驻留交给 resident 层；逐步迁移过滤、I/O、法线输出及 PlaScan 消费点。
- 法线估计补 `compute(output)` 形态并明确是否返回曲率；半径搜索法线只有在 CPU/GPU 邻域契约一致后再开放。
- 旧公开 `DenseMatrix` API 的删除以全部消费者迁移和 CPU/CUDA 构建通过为前提，不用永久兼容包装隐藏迁移缺口。

### 4. ICP 语义与设备工作区迁移（已完成）

- 新增明确命名的内点比例和残差指标；迁移 `getFitnessScore()` 到 PCL 意义的均方误差时同步更新全部调用点、基准和文档。
- 对齐或显式区分 `setTransformationEpsilon()`、旋转阈值的单位；最终 4x4 变换改用 PlaMatrix `Matrix<Scalar, 4, 4>`。
- CUDA KNN、ICP、法线和网格工作区迁到 `ExecutionContext`/`ResidentMatrix`，保留显式同步与错误报告。

### 5. 验证和收尾（本机验证已完成）

- 各阶段先跑受影响测试，再跑 PlaPoint CPU 全量 CTest、必要的 CUDA/OpenCL 测试、consumer smoke 和 PlaScan 相关目标。
- 对同一 XYZ 输入比较体素点数、近邻索引/距离、法线方向容差、ICP 变换、内点率和 RMSE；不把两库同名指标当作相同量。
- 更新 README、架构文档和调用示例；只清理本任务创建的临时产物，不提交或推送，除非用户另有要求。

## 当前交付范围

已在根 `plapoint` 命名空间实现 `PointXYZ`、`PointXYZRGB`、`Normal`、`PointNormal`、`PointCloud<PointT>`、`search::KdTree<PointT>`、`VoxelGrid<PointT>`、`NormalEstimation<PointInT, PointOutT>`、`IterativeClosestPoint<PointSource, PointTarget, Scalar>` 和 PLY 同名函数。`PointXYZd` 为行星大坐标提供双精度扩展。新增公开 4x4 矩阵、固定向量和法线特征分解使用 PlaMatrix 的 Eigen 风格接口。PlaScan QC 点云配准已迁到 `PointCloud<PointXYZd>` 和新 ICP。

按 PCL 1.15.1 核对后，当前已覆盖公开 Eigen 变换、主要点类型布局、常用重载与云元数据。字段 traits、部分继承接口及尚未实现的模块仍有差异，详见兼容性基线；这些不能视为完整兼容。

后续主工程迁移已将网格、渲染、过滤与 I/O 调用改为独立拥有矩阵属性的 `GeometryCloud<Scalar>`，搜索改为 `PointCloud<PointT>`/`KdTree<PointT>`。原标量/设备点云模板已移入内部设备实现，不提供公开兼容别名。新点类型 API 的 KdTree、VoxelGrid、NormalEstimation 和 ICP 目前只提供 CPU 路径；不能据此声称 GPU 点类型 API 或所有 PCL 类已兼容。

## 首轮审计记录

- 旧单点 `nearestKSearch()` 已按距离和输入索引稳定排序；旧 `radiusSearch()` 沿树遍历顺序返回。新 PCL 契约重载应独立排序半径结果，避免改变旧调用的顺序。
- KdTree 已缓存主机快照，单点查询本身走主机树。输出距离平方应从该快照和 PlaMatrix 固定向量计算，不增加 GPU 查询传输。
- 对有限但极大的坐标，内部搜索保留饱和距离。距离平方超过 PCL 输出的 `float` 可表示范围时，新输出使用正无穷，而不能丢掉该近邻。
- 新重载在合法空结果时清空两个输出；输入错误时保持调用者原有输出不变。

## 错误记录

| 问题 | 处理 |
| --- | --- |
| 根目录已有其他任务的 `task_plan.md`、`findings.md`、`progress.md` | 不覆盖；本任务使用本文件跟踪。 |
| 既有 `build-codex-cpu` 指向已不存在的 `/tmp/plamatrix-install-smoke`，构建时 `find_package(plamatrix)` 失败 | 不修改该构建树；改用 `build/tmp/` 下的新联合源码构建验证当前 PlaMatrix/PlaPoint。 |
| 定向测试首次使用了错误的临时构建目录路径 | 用 `rg --files` 找到 `build/tmp/plapoint-pcl-eigen-api/build/test/plapoint_tests`，32 项 KdTree 测试通过。 |
| PlaScan QC 迁移后 ICP 完全重合数据因浮点舍入返回旧方法名 | 对 ICP 相对种子 RMSE 的比较加入绝对容差 `1e-9`；`test_point_cloud_alignment` 五项通过。 |
| `PointXYZd` 经 PLY 适配器默认转成 float 造成大坐标失精 | 按点字段标量实例化矩阵和 PLY 读写，增加大坐标二进制回环测试。 |

## 本轮验证记录

- PlaPoint 联合源码 CPU CTest 共 427 项、零失败，其中 2 项按 CPU 配置跳过；CUDA/OpenCL CTest 共 880 项、零失败。特殊无效 OpenCL 设备用例在设定环境变量后另行通过。
- 最新配准 API 与 PlaMatrix 边界测试 5/5 通过；5 项 stream 敏感用例重复 20 轮，共 100/100 通过。
- PlaScan 14 个受影响目标构建通过；相关 CTest 选中 1231 项，1218 通过、13 项因可选依赖、Qt offscreen 或缺少 OBJ 基准路径按条件跳过，零失败。这不是 PlaScan 全量测试。
- PlaPoint 源码未增加以 PCL 命名的文件，也没有 PCL 构建依赖；PCL 对照资料移至 PlaScan 的 `docs/`。本轮未运行性能测试，未执行 commit、push、tag 或 Release。
- 详细命令、日志、跳过原因和保留的核查快照见 `build/tmp/plamatrix-module-migration/verification.md`。本轮一次性构建目录已清理，MSVC 尚未验证。
