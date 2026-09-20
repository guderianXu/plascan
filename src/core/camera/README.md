# Camera 与相机模型说明

PlaScan 的相机领域全部位于 `src/core/camera/`：`core/` 提供领域基础，`models/` 提供具体模型，
`project/` 管理项目运行态，`reference/` 管理外部导航参考；目录根部保留 ASP/Tsai、GDAL RPC、
USGSCSM ISD 等外部格式导入和相机格式转换服务。所有投影、反投影、交会和优化状态都由 typed 模型实现。

`CameraBaseline` 统一提供两相机光心距离、指定空间点的三角交会角，以及物理前方条件成立时的平均深度/基线比。SfM、MVS 等上层流程应通过该类共享基线定义，而不是重复实现光心距离与夹角计算。

旧的 `CameraModel`、`FramePinholeCamera`、`RpcCameraModel` 和 `PlanetaryLineScanCamera` 聚合类已经删除。
调用方不能通过旧类或转换包装绕过能力检查，也不会把 RPC 或推扫相机复制成静态针孔状态。

## 相机模型分层

新代码以 `camera/core`、`camera/models/*` 和 `camera/project` 为相机领域边界。定义（标定）
与实例（影像绑定的尺寸、时间、姿态或轨迹）分离，项目只持久化
`camera_definitions[]`/`camera_instances[]`；`images[*].camera` 已被明确拒绝。模型通过注册表
按字符串创建，并声明自身能力。外部参考数据走 `camera/reference` 的共享 frame/pose/time/
uncertainty 类型和显式 resolver。

- `FramePinholeDefinition` 保存共享标定，`FramePinholeInstance` 保存影像身份、尺寸和静态姿态；
  `FramePinholeNumericState` 是经过能力校验后交给 SfM、BA、MVS、网格、质检和地形模块的唯一面阵数值状态。
- `RpcDefinition`/`RpcInstance` 表示项目中的 RPC00B 相机，`RpcProjection` 提供正反投影和局部射线。
  `RpcIntersectionService` 和 `RpcBiasAdjustment` 直接使用同一个 immutable typed 实例。
- `LineScanDefinition`/`LineScanInstance` 保存推扫光学、分段逐行时间和轨迹；`LineScanProjection` 支持
  探元焦平面仿射、LRO NAC/归一化径向畸变和 6DoF/时间偏置。`PlanetaryLineScanIsdIO` 直接导入 typed 实例，
  行星激光 BA 不再经过第二套推扫数值类。项目持久化使用线阵参数 schema 2，只接受
  `direct_pose_samples` 或 `frame_composed` 轨迹、显式探元几何和分段行时；schema 1 不读取。

ISIS 控制网常见的左上像素中心 `(1, 1)` 不属于运行态公共接口。导入层必须先完成
ISIS/CSM/OpenCV 半像素换算，不能在投影函数中根据来源格式隐式猜测。

COLMAP text 内参相对影像左上角建立，其首像素中心是 `(0.5, 0.5)`。
`CameraFormatConverter` 在导入所有支持的 COLMAP 模型时执行
`cx_plascan = cx_colmap - 0.5` 和 `cy_plascan = cy_colmap - 0.5`，`fx/fy` 保持不变。
该变换只是像素原点换算，不是有损转换，因此不会额外写入 `summary.json` 的 `warnings`；
非零 COLMAP 畸变参数的现有 warning 语义保持不变。

复杂 COLMAP 模型不进入 BA、PnP、MVS 或纹理核心。导入时显式使用
`camera_convert_cli --pre-undistort-colmap-images`，转换器会在同一事务内把
`SIMPLE_RADIAL`、`RADIAL`、`OPENCV`、`FULL_OPENCV`、各鱼眼模型和
`THIN_PRISM_FISHEYE` 精确正向映射为零畸变针孔 PNG。目标焦距保守增大到每个输出像素都能映射回
源图，若最终 valid mask 仍有无效像素则整批 fail-closed；不会把复杂模型截断成 Brown 参数。
输出包含 `images/`、`valid_masks/`、更新后的 Tsai 和 `preundistort_manifest.json`，像素中心统一为
PlaScan 的首像素中心 `(0, 0)`。

项目的 `project_config.camera_model_policy` 保存当前 Chunk 的模型策略：未指定时使用新的
`frame_pinhole` 默认策略，线阵策略使用 `isis_usgscsm_linescan`。工程加载只接受规范化的
`camera_definitions`/`camera_instances` 集合，不读取旧 `.plascan` 或 `images[*].camera`；外部
Tsai/RPC/线阵文件必须先经过显式导入规范化。工作流能力由
`camera_core::CameraOperationPlan` 统一声明和校验：普通静态 SfM/BA/MVS 要求
`Projection + Ray + StaticPose + Optimization`，RPC 空三要求 `Projection + InverseProjection + Ray`，
推扫空三要求 `Projection + Ray + Trajectory + Optimization`，外部姿态参考要求 `StaticPose`。
因此 RPC 和推扫会明确拒绝进入静态针孔 SfM/BA/MVS；专用 RPC/线阵空三必须走各自模型流程，
不会把非静态模型降级成固定中心针孔。

项目相机的导入、更新和解算回写也只接受一套区分大小写的模型标识：`frame_pinhole`、`rpc00b`、
`planetary_linescan`。面阵 DTO 固定使用毫米内参、米制光心、`pixel_convention=center` 和显式
`world_frame`；RPC00B 固定使用 ECEF 世界 frame、WGS84 椭球高与 OpenCV 零基像素中心；线阵固定使用
嵌套 `optics`、`trajectory` 和 `line_timing`。`tsai`、`pinhole`、`rpc`、`line_scan` 等模型别名，
px/mm 双套字段、扁平线阵字段和缺省姿态/frame 均不会被转换或补全。

相机标定报告只保存 `initial_camera`/`adjusted_camera` 两个完整快照；GUI 不再读取
`fu_before`、`k1_after` 等扁平镜像字段，也不会把缺少规范比较记录的旧报告推断为一次已完成标定。

RPC 相机的定义和实例写入 `project_files.camera_definitions`/`camera_instances`，`model_type` 为
`rpc00b`。RPC 没有静态光心，通用阶段必须请求其实际能力，不能把它降级为固定中心针孔相机。当前针孔
SfM、BA、极线校正和 MVS 不接受 RPC；
RPC 稀疏几何应使用 `intersectRpcObservations()`，控制点偏差改正使用 `estimateRpcImageCorrection()`。
GUI 导入 TIFF 时会尝试读取内嵌 RPC，成功后自动写入对应工程影像条目；普通无 RPC TIFF 保持仅导入影像。
RPC 立体 DEM/DOM 使用独立的 `src/core/stereo_dem` 流水线，不经过针孔 MVS。

### 外部参考与求解绑定

`camera_reference_core` 是 Qt-free 的外部导航参考核心，负责坐标系/单位明确的
`CameraReferenceObservation`、`CameraReferenceResolver` 以及
`ResolvedCameraPosePrior` 工厂。Qt 目标 `camera_reference` 只负责参考集合的 JSON/sidecar
存储和比较；存储 DTO 不直接进入 SfM、BA 或 MVS 数值内核。

`ResolvedCameraPosePrior` 以稳定 `ImageId` 携带目标 solver frame 中的 `Pose`、可选协方差、杆臂应用状态和
来源信息。生产适配路径通过 factory 从已解析的 `ResolvedCameraReference` 构造；factory 对缺少姿态、杆臂语义、
目标 frame 或 provenance 的结果保持拒绝，不让 unresolved 数据进入求解。`transformProvenanceHash` 描述 frame/单位归一化和变换链，刻意不包含
影像、位置、姿态或协方差观测值，因而可在同一解析上下文的多幅影像之间比较；现有 `transformHash` 是
观测解析指纹，不能替代 provenance。

数值状态的影像身份和世界 frame 通过 `FramePinholeNumericState::bindIdentity()` 或上层
`SolverCameraBinding` 提供。工程运行器在 canonical `camera_instances` 完整覆盖所选影像时会按
`image_uuid` 传播这组绑定，并校验调用方提供的绑定；工程外部相机或 canonical 集合不完整时仍必须显式提供。
相机文件路径、文件名、列表序号和数值索引只用于定位输入，不能回退成持久化 `ImageId` 或 frame；无法证明
绑定时需要外部参考的流程会 fail-closed。

GUI 和 CLI 查询静态面阵数值状态时统一通过 `CameraProjectRuntime::planOperationForImages()` 先做工作流能力门控，
再由 `framePinholeStatesForImages()` 按 `image_uuid` 一次性、有序且全成败地解析；不在界面层从实例表再次拼装
数值相机。MVS 入口直接复用 `DenseMvs` 计划，因此 RPC/推扫相机会在数值转换前给出模型能力诊断。仍保留的引导匹配参考相机 JSON 会传递已声明的 `world_frame`；两台参考
相机声明不同 frame 时，极线几何入口会在计算相对姿态前拒绝该像对。

深度批次输入签名只包含 canonical `ImageId` 及其完整的定义/实例记录，并要求选择项按 UUID 或完整规范化路径
唯一解析。缺少 canonical 相机、重复身份、歧义路径或模型记录无效时签名为空，缓存复用会被阻断；不会用文件名、
数组下标、文件哈希或旧的 `images[*].camera` 字段生成替代身份。

## 1. 坐标系和单位约定

`FramePinholeNumericState` 使用以下运行态约定：

- `cameraToWorldRotation` 是行优先存储的 3×3 旋转矩阵 `R_cw`，表示相机坐标系到世界坐标系的旋转。
- `cameraCenter` 是世界坐标系中的相机光心 `C`，PlaScan 工程中通常使用米（m）。
- 世界点转换到相机坐标系时使用：

  ```text
  X_cam = R_cw^T * (X_world - C)
  ```

- 运行态的焦距 `focalX/focalY` 和主点 `principalX/principalY` 使用像素。
- `pixelPitch` 使用 `mm/pixel`，只负责 Tsai 文件中的毫米值与运行态像素值换算。
- `uAxisSign`、`vAxisSign` 只能是 `+1` 或 `-1`。
- `depthAxisFlipped == false` 时，物理前方为 `Z_cam > 0`；为 `true` 时，物理前方为 `Z_cam < 0`。

## 2. PlaScan 支持的 Tsai 文件格式

一个 Tsai 文件描述一台相机。当前加载器按行解析，键名不区分大小写，支持 `=` 或 `:` 分隔符。典型文件如下：

```text
VERSION_3
PINHOLE
TSAI
fu = 35.0
fv = 35.0
cu = 18.0
cv = 12.0
u_direction = 1 0 0
v_direction = 0 1 0
w_direction = 0 0 1
pitch = 0.005
k1 = -0.01
k2 = 0.001
k3 = 0.0
p1 = 0.0001
p2 = -0.0001
C = 10.0 20.0 30.0
R = 1 0 0 0 1 0 0 0 1
```

`VERSION_3`、`PINHOLE` 和 `TSAI` 是常见的 ASP 文件头。当前加载器会忽略这些标识行，真正建立相机模型的是下面的参数字段。

### 2.1 字段定义

| 字段 | 必需 | 数量 | PlaScan 解释 |
|---|---:|---:|---|
| `fu` | 是 | 1 | 水平焦距，文件值结合 `pitch` 按毫米换算为像素 |
| `fv` | 是 | 1 | 垂直焦距，文件值结合 `pitch` 按毫米换算为像素 |
| `cu` | 是 | 1 | 主点横坐标，文件值结合 `pitch` 按毫米换算为像素 |
| `cv` | 是 | 1 | 主点纵坐标，文件值结合 `pitch` 按毫米换算为像素 |
| `C` | 是 | 3 | 世界坐标系中的相机中心，通常为米 |
| `R` | 是 | 9 | 行优先 `R_cw`，即 camera-to-world 旋转矩阵 |
| `pitch` | 否 | 1 | 像元尺寸，单位 `mm/pixel`；缺省值为 `1.0` |
| `k1/k2/k3` | 否 | 各 1 | `r²/r⁴/r⁶` 径向畸变系数；缺省值为 `0` |
| `p1/p2` | 否 | 各 1 | Brown-Conrady 切向畸变系数；缺省值为 `0` |
| `u_direction` | 否 | 1 或 3 | u 轴方向；缺省为 `+1` |
| `v_direction` | 否 | 1 或 3 | v 轴方向；缺省为 `+1` |
| `w_direction` | 否 | 1 或 3 | 光轴方向；z 分量为负时启用负 Z 前向 |

方向字段既可以写成 ASP 向量格式，也可以写成 PlaScan 简化标量格式：

```text
u_direction = 1 0 0
v_direction = 0 -1 0
w_direction = 0 0 -1
```

或：

```text
u_direction = 1
v_direction = -1
w_direction = -1
```

### 2.2 内参单位换算

加载成功后，文件内参按以下方式转换成运行态像素值：

```text
focalX     = fu / pitch
focalY     = fv / pitch
principalX = cu / pitch
principalY = cv / pitch
```

例如 `fu = 35 mm`、`pitch = 0.005 mm/pixel`，运行态焦距为 `7000 pixel`。如果文件本身已经使用像素值，应将 `pitch` 写成 `1`。

### 2.3 畸变和投影公式

对归一化坐标 `x = X_cam/Z_cam`、`y = Y_cam/Z_cam`：

```text
r2 = x*x + y*y
radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y

u = uAxisSign * focalX * xd + principalX
v = vAxisSign * focalY * yd + principalY
```

`projectWorldPoint()` 会检查点是否位于物理前方；`projectWorldPointSigned()` 只排除接近零的深度，允许调用方处理任意深度符号。

### 2.4 加载和保存限制

`loadFramePinholeNumericStateFromTsaiFile()` 在以下情况下返回 `false`：

- 文件无法打开；
- 缺少 `fu/fv/cu/cv/C/R` 中任一必需字段；
- `pitch <= 0`；
- `fu <= 0` 或 `fv <= 0`。

加载结果还必须通过 `FramePinholeNumericState::validateNumericalState()`；非有限或退化姿态会被拒绝。

`saveFramePinholeNumericState()` 会把运行态像素内参乘以 `pixelPitch` 后写回文件，并将方向写成标量。
它输出参数行，不补写 `VERSION_3/PINHOLE/TSAI` 文件头。

## 3. FramePinholeNumericState 数值边界

工程内的面阵相机必须先由 `CameraProjectRuntime` 按稳定 `ImageId` 解析，并由
`CameraOperationPlan` 校验工作流要求。静态 SfM、BA 和 MVS 请求
`Projection + Ray + StaticPose + Optimization`；DOM 正射只请求
`Projection + StaticPose`。能力、世界 frame 或影像绑定不满足时，整批解析失败。

外部 Tsai 文件可以直接读入未绑定的数值状态，调用方随后必须提供明确的实例、影像和世界 frame：

```cpp
#include "FramePinholeTsaiIO.h"

xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
std::string error;
if (!xjw::camera_io::loadFramePinholeNumericStateFromTsaiFile("camera.tsai", &camera, &error))
{
    return;
}
if (!camera.bindIdentity(xjw::camera_core::CameraInstanceId("camera-1"),
                         xjw::camera_core::ImageId("image-1"),
                         xjw::coordinate_system::CoordinateFrameId("survey-world"),
                         &error))
{
    return;
}

const double world_point[3] = {100.0, 200.0, 50.0};
double pixel[2] = {0.0, 0.0};
camera.projectWorldPoint(world_point, pixel);
```

`bindIdentity()` 不从路径、文件名或数组序号推断身份和坐标系。解码得到的未绑定状态可以用于一次性数值计算，
但只有具有有效影像尺寸并完成显式绑定的状态才能通过 `toInstance()` 回到规范相机图。

重叠分析、地形、网格和质检直接调用同一状态的 `projectWorldPoint()`、`unprojectPixel()` 和
`rayForPixel()`。Brown-Conrady 畸变、像素轴和深度轴约定只维护一份。MVS 需要无畸变正深度几何时，
`prepareMvsImage()` 会同时重映射影像并返回与输出像素严格对应的工作状态；调用方不能只修改相机参数。

`scaledIntrinsics()` 用于同步缩放焦距、主点和影像尺寸；`normalizedForPositiveDepth()` 把轴方向折叠到
等价数值表示。两者都返回新的状态，不修改原对象。RPC 和推扫实例不满足静态针孔能力，必须进入各自流程。

## 4. RPC00B 使用方法

项目运行态使用 `RpcDefinition`、`RpcInstance` 和 `RpcProjection`。RPC 标准输入坐标为经度（度）、
纬度（度）和 WGS84 椭球高（米）；ECEF 接口使用 `EPSG:4978` 米。外部 GeoTIFF/NITF 通过
`RpcRasterIO` 直接导入规范的 immutable definition/instance，RPC 航三、DEM/DOM 和平面扫描共享该实例。

```cpp
#include "camera/RpcRasterIO.h"
#include "camera/models/rpc/RpcProjection.h"

std::string error;
auto camera = xjw::camera_models::rpc::importRpcRasterInstance(
    "satellite.tif",
    xjw::camera_core::CameraDefinitionId("rpc-definition-1"),
    xjw::camera_core::CameraInstanceId("rpc-instance-1"),
    xjw::camera_core::ImageId("image-1"),
    xjw::coordinate_system::CoordinateFrameId("EPSG:4978"),
    &error);
if (!camera)
{
    // GeoTIFF/NITF 没有 RPC 域、旁车 RPB 无法发现或参数非法。
    return;
}

xjw::camera_models::rpc::ImagePoint pixel;
xjw::camera_models::rpc::RpcProjection::groundToImage(*camera, {108.5, 34.2, 1250.0}, &pixel);

xjw::camera_models::rpc::RpcDefinition::GeodeticCoordinate ground;
xjw::camera_models::rpc::RpcProjection::imageToGroundAtHeight(*camera, pixel, 1250.0, &ground);
```

`importRpcRasterInstance()` 读取 GDAL 的 `RPC` metadata domain，因此支持 GDAL 驱动能够识别的影像内嵌
RPC 和与影像关联的 RPC/RPB 旁车文件。`LINE/SAMP/LAT/LONG/HEIGHT_OFF`、对应 `SCALE` 及四组
20 项系数为必需字段；`ERR_BIAS` 和 `ERR_RAND` 可选。

产品级 RPC 可叠加独立的影像空间改正，而不修改厂商提供的 80 个多项式系数。改正以原始 RPC 的
`SAMP_OFF/SAMP_SCALE` 和 `LINE_OFF/LINE_SCALE` 归一化像点为自变量，分别为 sample、line 增加三项
`常量 + sample 一次项 + line 一次项`，六个系数的输出单位均为像素。只有常量项非零时就是常用的
RPC 平移偏差模型；工程 JSON 使用 `image_correction.model = affine_normalized_v1` 保存该参数，未提供改正时
实例使用零改正。

已知控制点可通过 `estimateRpcImageCorrection()` 自动估计平移或仿射改正。默认使用 Huber 迭代重加权，
结果同时报告改正前后 RMS 和最大残差；调用方检查质量后创建带新改正值的 immutable 实例：

```cpp
#include "camera/models/rpc/RpcBiasAdjustment.h"

std::vector<xjw::camera_models::rpc::RpcControlPointObservation> control_points = loadControlPoints();
xjw::camera_models::rpc::RpcBiasAdjustmentResult adjustment;
std::string error;
if (xjw::camera_models::rpc::estimateRpcImageCorrection(*camera, control_points, &adjustment, &error))
{
    auto adjusted = camera->withImageCorrection(
        xjw::camera_core::CameraInstanceId("rpc-instance-adjusted"), adjustment.correction);
}
```

仿射模型至少需要三个且影像分布不共线的控制点；只有少量控制点或覆盖范围很小时应改用
`RpcImageCorrectionModel::Translation`。估计器使用 WGS84 经度、纬度和椭球高，正高必须先做大地水准面转换。

RPC 没有固定物理光心。`rayForPixel()` 用 `HEIGHT_OFF ± HEIGHT_SCALE` 两个高程解构造有效体积内的
局部弦线，只适合初始化和几何诊断。精确像点反算必须给出椭球高或 DEM。双像交会使用
`RpcIntersectionService::intersect()`，先以两条局部弦线初始化，再在 ECEF 中最小化四个像素残差。

## 5. 相关文件

- `camera/core/`：Qt-free 的相机 ID、坐标 frame、能力查询、Pose/不确定度和坐标变换服务。
- `camera/models/`：按能力划分的面阵针孔、RPC 和推扫实例/定义；`FramePinholeNumericState` 是静态数值入口。
- `camera/project/`：规范化项目相机记录、运行态加载和按模型独立版本校验的注册边界。
- `camera/reference/resolve/`（由 `camera_reference_core` target 提供）：外部参考 observation/resolver、
  `ResolvedCameraPosePrior` 和 provenance 校验。
- `camera/models/frame_pinhole/FramePinholeNumericState.h/.cpp`：SfM/BA/MVS 使用的 solver-owned 数值状态、
  投影、反投影、射线和正深度规范化。
- `FramePinholeTsaiIO.h/.cpp`：不依赖 Qt 的 Tsai 数值状态读取器。
- `camera/models/linescan/`：推扫定义/实例、分段行时、时变姿轨、投影和 solver-owned 偏置。
- `camera/models/LineScanModelJson.h` 与对应 decode/encode 源文件：线阵 schema 2 的唯一 JSON 编解码器。
- `PlanetaryLineScanIsdIO.h/.cpp`：USGSCSM ISD 到 `LineScanInstance` 的严格外部导入边界。
- `camera/models/rpc/`：RPC00B 定义/实例、WGS84 ECEF/大地坐标转换、正反投影、近似射线、
  控制点影像改正和双像 ECEF 迭代交会。
- `RpcRasterIO.h/.cpp`：GDAL RPC metadata domain 和关联旁车到 `RpcInstance` 的导入。
- `ProjectCameraIO.h/.cpp`、`ProjectRpcCameraIO.cpp` 与 `ProjectLineScanCameraIO.cpp`：外部针孔、RPC 和
  typed 线阵实例的严格解码与规范项目序列化。
- `../sfm/pose/CameraReferencePosePriorAdapter.h/.cpp`：按 `ImageId` 将已解析外部姿态显式对齐到
  `FramePinholeNumericState`/BA 相机顺序；不推断 frame，也不从路径补 identity。
- `CameraFormatConverter.h/.cpp`：Middlebury、EPFL、COLMAP、Metashape 等外部格式转换。
- `../mvs/MvsImagePreprocessor.h/.cpp`：将带畸变原图和 `FramePinholeNumericState` 转换为 MVS 使用的无畸变影像及工作相机。
- `test/`：线阵、RPC、Tsai 加载和格式转换测试；typed 模型测试位于各 `camera/models/*/tests/`。
