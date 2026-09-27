# PlaScan 统一坐标系模块设计

**日期：** 2026-09-20

**状态：** 设计草案，待相机目录重构完成后实施

**范围：** 相机、相机参考、控制点、SfM/BA、RPC、DEM/DOM、行星线阵与激光约束使用的空间参考、坐标变换、时间依赖和变换溯源

## 1. 结论

新模块放在 `src/core/coordinate_system/`，与 `camera`、`control_points`、`terrain`、`lidar` 平级，不放入
`src/common`，也不继续隶属于相机模块。

模块分为三个首期 target：

- `coordinate_system_types`：纯 C++20 值类型、单位、参考体、frame、CRS 描述、时间和错误类型；
- `coordinate_system_transform`：静态/动态变换图、WGS84/椭球数学、局部切平面、Jacobian、协方差和溯源；
- `coordinate_system_gdal`：基于 GDAL/OGR 的 EPSG、WKT、投影和垂直转换后端。

未来若引入 CSPICE，再增加 `coordinate_system_spice`，不能让 CSPICE、GDAL 或 Qt 成为基础类型 target 的依赖。

每个 Chunk 只拥有一个权威 `CoordinateContext`。相机、控制点、点云、DEM/DOM 和行星观测只引用其中的稳定 ID；
进入同一次 SfM/BA/MVS 求解前，数据必须显式归一化到同一个笛卡尔、在求解窗口内静态的 solver frame。
带绝对约束的工作流必须为米制；纯相对重建如果尚未恢复尺度，则必须显式标记为 unresolved project scale，不能冒充米制。

第一项业务迁移不是 DEM 或 UI，而是控制点：目前控制点能够解析源 CRS，却没有在生产数据流中调用已有的
`transformCoordinate()`，原始经纬高可能直接进入 BA。新模块必须先封住这条错误路径。

## 2. 当前实现与主要问题

| 领域 | 当前实现 | 问题 |
|---|---|---|
| 相机 frame | `src/core/camera/core/types/CameraFrames.h` | `Geodetic` 与角度单位只是标签；实际服务只做刚体变换和米/千米缩放 |
| 相机变换 | `src/core/camera/core/transform/CoordinateTransformService.*` | 只支持父子树；不能做 CRS、椭球、动态 frame、垂直基准或有锚点的 Jacobian |
| 控制点 | `src/core/control_points/reference/CoordinateReference.*` | 能用 GDAL 解析/转换，但生产调用者没有使用转换结果 |
| BA 控制点入口 | `src/core/sfm/project/MarkerBaAdapter.*`、`src/core/aerial_triangulation/reconstruction/MarkerPriorLoader.*` | 直接消费原始 `x/y/z`，缺少“已归一化”类型门禁 |
| RPC | `src/core/camera/models/rpc/RpcProjection.*` | 内置 WGS84 常量并手写经纬高与 ECEF，形成第二套权威实现 |
| RPC 空三 | `src/core/aerial_triangulation/engine/RpcCoordinates.cpp` | 单独构造局部 ENU，与通用 frame 图没有关联 |
| RPC DEM/DOM | `src/core/stereo_dem/RpcGeospatialSupport.*` | 再次使用一套 GDAL WGS84/UTM 转换与 UTM 自动选区逻辑 |
| 地形产品 | `src/core/terrain/DemDomTypes.h`、`DemDomIO.*` | WKT、原点、单位和自由字符串分散，solver frame 与产品 CRS 未分离 |
| 小天体投影 | `src/core/terrain/projection/AsteroidProjection.*` | 球面、方位等距、三轴椭球及 WKT 为局部实现，没有统一参考体注册 |
| 行星激光 | `src/core/lidar/PlanetaryLaserShot.*` | 已显式要求目标、体固系、TDB 和角度约定，但球坐标与协方差转换仍为模块私有实现 |
| 工程格式 | `docs/project/CHUNK_WORKSPACE_DESIGN.md` | 已预留 `chunk.coordinate_system`，目前没有权威 schema |

当前最危险的问题不是“少支持一种投影”，而是同一个三元数组在不同路径中可能分别表示：

- 经度、纬度、椭球高；
- ECEF 米；
- 局部 ENU 米；
- 相对重建坐标；
- 月球或小天体体固系坐标；
- 投影平面 Easting、Northing 和某种未说明的高程。

字符串相同也不能证明数值可直接混用。`EPSG:4979` 是地理三维 CRS，不是欧氏 solver frame；WKT 相同也不能证明
垂直网格、历元、轴序和变换资源相同。

## 3. 术语与不可破坏的约束

### 3.1 参考体 `ReferenceBody`

描述 Earth、Moon 或具体小天体，以及用于坐标数学的形状：球、旋转椭球、三轴椭球或显式形状模型。
它不是显示名称，必须有稳定 ID 和尺寸单位。不同参考体之间不得自动寻找变换路径。

### 3.2 参考框架 `CoordinateFrame`

描述物理空间中的原点、轴和随时间的行为，例如：

- 地球体固 ECEF；
- `MOON_ME` 等行星体固系；
- J2000 惯性系；
- 固定的局部 ENU；
- 相对 SfM 的局部笛卡尔系；
- 传感器或激光 frame。

`Geodetic` 不再是 frame kind。经纬高是数值表示/CRS，不是物理笛卡尔 frame。

### 3.3 空间参考 `SpatialReference`

描述数值坐标的解释，包括 geographic、geocentric、projected 或 engineering reference，authority/code、规范化
WKT2/PROJJSON、轴顺序、水平 datum 和垂直参考。`EPSG:4978` 与 `EPSG:4979` 属于这一层。

### 3.4 Solver frame

是数值求解专用的 `CoordinateFrame`：笛卡尔、右手、静态，并有明确的父 frame、原点、旋转和尺度状态。
绝对/地理工作流必须为米制；纯相对重建可暂用 project unit，但必须标记尺度未解，且不能进入要求物理长度的流程。
它不是输出投影 CRS，也不是相机模型的地面输入 CRS。

### 3.5 栅格地理参考 `RasterReference`

由像素约定、完整六参数仿射变换和目标 `SpatialReference` 组成。表示层允许旋转/斜切；某个算法只能处理 north-up
栅格时，由该算法显式拒绝，不能让基础类型丢失信息。

### 3.6 时间引用 `TimeReference`

动态 frame 必须使用包含时间尺度、时间原点和秒值的时间引用。仅有 `TDB + 123.5` 仍有歧义；需要说明是
J2000 ET 秒、Unix 秒、工程相对时间还是其他定义。动态变换缺时间、时间尺度不兼容或超出数据覆盖区时必须失败。

### 3.7 全局约束

1. 核心接口不接收无来源的裸 `std::array<double, 3>`。
2. 任何自动推导的 solver frame、UTM 分区、目标 CRS、历元和变换资源都必须可见、可持久化、可复现。
3. 不允许按 frame ID 相似、WKT 文本相似或单位“看起来像米”做隐式兼容。
4. 不允许在未知垂直基准时保持 `z` 不变并声称完成了三维转换。
5. 数值内核不隐式调用 CRS 服务；所有归一化发生在工作流入口，并冻结 transform provenance。
6. 相同 Chunk 的结果必须引用精确的 context revision/hash。上下文变化会使下游结果失效。

## 4. 模块和依赖方向

```text
coordinate_system_gdal ─────► coordinate_system_transform ─────► coordinate_system_types
                                      ▲                               ▲
                                      │                               │
control_points / stereo_dem / terrain ┘                camera_core ───┘
                                      ▲                    ▲
                                      │                    │
              camera_models_rpc / camera_reference        │
                                      ▲                    │
                                      └──── sfm / BA / lidar

camera_project ─────► camera_models_* ─────► camera_core
```

箭头从调用方指向依赖项。坐标模块不得依赖 camera、SfM、GUI、Qt DTO 或任何具体工作流。

| 调用 target | 允许的直接依赖 |
|---|---|
| `coordinate_system_types` | 仅 C++20 标准库 |
| `coordinate_system_transform` | `coordinate_system_types` |
| `coordinate_system_gdal` | `coordinate_system_transform`、GDAL/OGR |
| `camera_core` | `coordinate_system_types` |
| `camera_models_rpc` | `camera_core`、`coordinate_system_transform` |
| `camera_reference` | `camera_core`、`coordinate_system_transform` |
| `control_points`、`stereo_dem`、`terrain` | 按需依赖 transform 与 gdal |
| `sfm`、`bundle_adjust`、`lidar` | camera target 与 types/transform 中实际使用的最小集合 |

建议目录：

```text
src/core/coordinate_system/
├── CMakeLists.txt
├── types/
│   ├── CoordinateIds.h
│   ├── CoordinateUnits.h
│   ├── CoordinateFrame.h
│   ├── SpatialReference.h
│   ├── ReferenceBody.h
│   ├── TimeReference.h
│   ├── RasterReference.h
│   └── CoordinateErrors.h
├── context/
│   ├── CoordinateContext.h
│   ├── CoordinateContextBuilder.h
│   └── SolverFrameDefinition.h
├── transform/
│   ├── CoordinateTransformService.h
│   ├── TransformOperation.h
│   ├── TransformPlan.h
│   ├── TransformRequest.h
│   ├── TransformProvenance.h
│   ├── RigidTransformOperation.h
│   ├── EllipsoidTransformOperation.h
│   ├── LocalTangentPlaneOperation.h
│   ├── DynamicFrameOperation.h
│   └── CovarianceTransform.h
├── gdal/
│   ├── GdalSpatialReferenceParser.h
│   ├── GdalTransformOperation.h
│   └── GdalTransformFactory.h
└── tests/
```

首期不创建含糊的万能 `coordinate_system` target。调用方应明确依赖 types、transform 或 gdal，避免 GDAL 通过
PUBLIC 依赖扩散到 `camera_core`。

实施时在 `src/core/CMakeLists.txt` 中把 `coordinate_system` 排在 `camera` 之前注册，随后才允许 camera target
引用新类型；不得依靠当前目录顺序之外的偶然传递链接。

## 5. 核心数据模型

以下为接口方向，不是逐字实现稿：

```cpp
namespace xjw::coordinate_system
{
    class CoordinateFrameId;
    class SpatialReferenceId;
    class CoordinateContextId;
    class ReferenceBodyId;

    enum class CoordinateFrameKind
    {
        EarthFixed,
        BodyFixed,
        Inertial,
        LocalTangent,
        LocalCartesian,
        Sensor
    };

    enum class SpatialReferenceKind
    {
        Geographic3d,
        Geocentric3d,
        Projected3d,
        Engineering3d
    };

    enum class TimeScale
    {
        Utc,
        Tai,
        Tt,
        Tdb,
        Relative
    };

    enum class TimeOrigin
    {
        UnixEpoch,
        J2000,
        ProjectEpoch,
        Custom
    };

    struct TimeReference
    {
        TimeScale scale;
        TimeOrigin origin;
        double seconds;
        std::optional<std::string> customOriginId;
    };
}
```

`ReferenceBody` 至少记录：

- 稳定 ID 和规范名称；
- 形状类型与米制参数；
- 默认体固 frame；
- 经度方向、纬度类型等行星角度约定；
- 来源与参数 hash。

`CoordinateFrame` 由以下一种 binding 定义：

- `RootBinding`：当前 context 的根 frame；
- `StaticRigidBinding`：到父 frame 的固定刚体变换；
- `LocalTangentBinding`：参考体、经纬高锚点和 ENU/NED 轴约定，构建后冻结为确定变换；
- `DynamicBinding`：provider ID、源/目标 frame、时间覆盖区和资源版本。

`SpatialReference` 至少记录：

- `SpatialReferenceId`；
- kind 与绑定的参考体/frame；
- authority/code；
- 规范定义及 hash；
- 对外轴序与内部规范轴序；
- 水平 datum；
- `Ellipsoidal`、`Orthometric`、`PlanetaryRadius`、`Relative` 或 `Unknown` 垂直语义；
- 线性与角度单位。

内部规范数值约定固定为：

- Cartesian/projected：右手 `x/y/z`，线性量使用 metre；
- 未恢复尺度的纯相对 solver 使用专门的 project unit，且只允许 `SolverScaleStatus::Unresolved`；
- geodetic：命名字段 `longitudeRadians/latitudeRadians/heightMetres`；
- RPC 或外部格式需要 degree 时，只在导入器/模型边界转换；
- raster：明确 pixel-center 或 pixel-corner，不把两者混用。

### 5.1 `CoordinateContext`

`CoordinateContext` 是不可变快照，包含：

- context ID、schema version 和 revision；
- reference bodies；
- spatial references；
- frames 与 transform operations；
- 当前 solver frame；
- solver scale status；
- 允许的精度/垂直转换策略；
- 使用的网格、星历、姿态核或采样轨迹资源；
- canonical context hash。

编辑上下文使用 builder 产生新 revision。运行中的任务持有旧快照，不能观察到半途变更。

### 5.2 Solver frame

原 `CameraReferenceSolverFrame` 的通用字段提升为 `SolverFrameDefinition`：

```cpp
struct SolverFrameDefinition
{
    CoordinateFrameId frameId;
    CoordinateFrameId parentFrameId;
    SolverScaleStatus scaleStatus;
    LinearUnit linearUnit;
    std::array<double, 3> originInParentMetres;
    std::array<double, 9> solverToParentRotation;
    std::string normalizationHash;
};
```

`SolverScaleStatus` 首期至少包含 `Metric` 与 `Unresolved`，并参与 context、plan 和成果 provenance hash。固定首相机、
消除 gauge freedom 或把初始基线归一为 1 都不会恢复物理尺度，状态仍必须是 `Unresolved`。未来确有已知非米单位时，
可扩展为 `Resolved{unit, scaleToMetre}`，但不能用 unit 标签代替尺度求解证据。

目标 CRS/WKT 不属于 solver frame 本体；它们属于输出 `SpatialReference`。camera/reference 继续负责“如何根据外部
相机参考选择或建立 solver frame”的策略，但不再拥有通用 frame 类型。

## 6. 变换引擎

### 6.1 操作类型

变换图不再假定每条边都是刚体边，至少支持：

- `StaticRigid`：旋转、平移、线性单位；
- `EllipsoidGeodeticCartesian`：经纬高与体固 Cartesian；
- `MapProjection`：geographic/geocentric/projected CRS，由 GDAL backend 提供；
- `LocalTangentPlane`：体固 Cartesian 与固定 ENU/NED；
- `DynamicFrame`：在指定时刻求 frame 之间的旋转、平移及必要的速度信息；
- `AxisAndUnit`：只在显式 reference 定义允许时调整轴和单位；
- `RasterAffine`：pixel 与 map coordinate，保持为二维/高程相关的专用操作。

每个 operation 必须声明：支持的输入语义、是否可逆、有效域、是否需要时间/锚点、精度、资源依赖和版本。

### 6.2 请求、计划和结果

```cpp
using CoordinateReference = std::variant<CoordinateFrameId, SpatialReferenceId>;

struct TaggedPosition3
{
    CoordinateReference reference;
    std::array<double, 3> value;
    std::optional<TimeReference> time;
};

enum class CoordinateValueKind
{
    Position,
    FreeVector,
    Direction,
    Orientation,
    PositionCovariance,
    PoseCovariance
};

struct TransformRequest
{
    CoordinateReference source;
    CoordinateReference target;
    CoordinateValueKind valueKind;
    std::optional<TimeReference> time;
    std::optional<TaggedPosition3> anchor;
    TransformPolicy policy;
};
```

服务先编译不可变 `TransformPlan`，再用于单点或批量执行。计划缓存键至少包含 context hash、源/目标 reference、
value kind 和 policy；动态变换不能忽略时间覆盖或 provider revision。

结果不只返回数值，还返回：

- 计划/操作链 ID 与稳定 hash；
- backend 与算法版本；
- 使用的 grid、星历、姿态/轨迹资源 hash；
- 实际精度与是否使用近似操作；
- 轴、单位和垂直语义的规范化记录；
- 可诊断的 warning；默认策略下任何不安全情况应为 error。

### 6.3 点、向量、方向与姿态

- `Position` 可以使用平移和非线性变换；
- `FreeVector` 不使用平移，但穿过非线性 CRS 时必须提供 anchor，以该点 Jacobian 变换；
- `Direction` 同样需要局部 Jacobian，并在结果中重新归一化；
- `Orientation` 需要明确旋转方向和源/目标切空间，不能把经纬度轴直接当作笛卡尔轴；
- camera `Pose` 仍由 camera_core 拥有，camera adapter 分别调用 position/orientation API 后重建相机位姿。

原 `transformVector()` 对任意 frame 直接套旋转/比例的行为，只能保留在纯线性计划中。遇到投影、椭球或动态边而
没有 anchor/time 时必须拒绝。

### 6.4 协方差

协方差使用 `J Σ Jᵀ` 传播。类型必须携带：

- 维度与布局；
- 平移和旋转分量顺序；
- 平移单位、旋转单位；
- 表达 frame/切空间；
- 线性化点和时间；
- 是否包含位置-姿态交叉项。

RPC 的 error bias/random、像方改正协方差和 6DoF pose covariance 是不同类型，不得复用一个模糊数组。

### 6.5 动态 frame

动态 provider 接口接受完整 `TimeReference`，返回该时刻的变换与 provenance。线阵相机必须对每一行曝光时刻求值，
不能用中心时刻的一次变换代替整景。provider 还必须声明：

- 输入时间尺度与时间原点；
- 有效时间区间；
- 插值方法和阶数；
- 外推是否允许，默认不允许；
- 位置、速度、姿态与角速度的可用性；
- 核文件、姿态表或轨迹样本的内容 hash。

首期可使用“已采样轨迹 provider”，不要求立即引入 CSPICE；没有可用 provider 时返回明确的
`DynamicTransformUnavailable`，不能假设 inertial 与 body-fixed 相同。

### 6.6 失败语义

核心错误码至少包括：

- `UnknownContextOrReference`；
- `IncompatibleReferenceBody`；
- `NoTransformPath`；
- `MissingTime` / `UnsupportedTimeScale` / `TimeOutsideCoverage`；
- `MissingAnchorForDifferentialTransform`；
- `UnknownVerticalReference` / `VerticalGridUnavailable`；
- `AxisConventionMismatch`；
- `OutOfDomain`；
- `AccuracyPolicyRejected`；
- `NonFiniteInput`；
- `TransformResourceChanged`。

错误对象带 operation、源/目标 ID、输入索引和修复提示；核心不生成 Qt 文案。

## 7. Solver frame 策略

| 工作流 | 推荐 solver frame | 说明 |
|---|---|---|
| 无地理参考的普通 SfM | `relative-local` project-unit 笛卡尔系 | 不伪造 EPSG/WGS84 或 metre；显式标记 scale unresolved |
| 带 GNSS/GCP 的地球测区 | 固定 local ENU 米制系 | 锚点接近观测质心，改善 BA 条件数 |
| RPC 空三 | 局部 ENU 或明确配置的米制 solver frame | RPC 模型输入仍为 WGS84 经纬椭球高；交会可在 ECEF 中做中间计算 |
| 大范围/全球 RPC | 分区或 ECEF 中间 frame，按算法显式选择 | 不把 UTM 自动选区隐含在模型内部 |
| 月球/小天体局部测区 | 体固系上的 local tangent frame | reference body、纬度类型和经度方向必须明确 |
| 小天体全球产品 | body-fixed metre frame | 输出投影可另选球面/三轴椭球投影 |

自动建立 solver frame 时，锚点算法、输入集合、舍弃规则和最终数值必须显示在参数界面并写入工程。修改 solver frame
等同于修改全部三维数值语义，必须使稀疏、稠密、网格、DEM 和 DOM 等派生结果失效。

scale unresolved 时，只允许相对 SfM/BA/MVS、无量纲重投影误差和相对模型浏览；GCP/GNSS 绝对约束、米制 BA
阈值、DEM/DOM GSD、面积/体积/距离统计、米制点云/网格导出以及声明 EPSG/ECEF/局部米制 CRS 都必须硬失败，直至
通过可验证的尺度约束完成规范化。

## 8. Chunk 持久化

`chunk.coordinate_system` 建议使用独立 schema：

```json
{
  "coordinate_system": {
    "schema_version": 1,
    "context_id": "coordctx-<uuid>",
    "revision": 1,
    "reference_bodies": [
      {
        "id": "body-earth-wgs84",
        "shape": "biaxial_ellipsoid",
        "semi_major_m": 6378137.0,
        "inverse_flattening": 298.257223563,
        "body_fixed_frame_id": "frame-earth-fixed"
      }
    ],
    "spatial_references": [
      {
        "id": "crs-epsg-4979",
        "authority": "EPSG",
        "code": "4979",
        "kind": "geographic_3d",
        "axis_mapping": "canonical_lon_lat_height",
        "vertical_reference": "ellipsoidal",
        "canonical_definition_hash": "sha256:<hash>"
      }
    ],
    "frames": [
      {
        "id": "frame-solver-local-enu",
        "kind": "local_tangent",
        "parent_frame_id": "frame-earth-fixed",
        "linear_unit": "metre",
        "origin_in_parent_m": [1.0, 2.0, 3.0],
        "frame_to_parent_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      }
    ],
    "solver_frame_id": "frame-solver-local-enu",
    "solver_scale_status": "metric",
    "transform_resources": [],
    "context_hash": "sha256:<hash>"
  }
}
```

实际 schema 还需嵌入或通过受管资源引用 canonical WKT2/PROJJSON；仅有 hash 无法离线复现。authority/code 用于身份和
展示，canonical definition 用于冻结当时语义。

相机定义/实例只引用 `frame_id` 与 context 中的 reference；RPC 同时引用地面 `SpatialReferenceId`。外部参考、控制点和
行星激光保存 source reference 及原始数值，resolved 数据保存 solver frame ID、plan hash 和 normalization hash。

GeoTIFF 等可移植成果仍写入标准 WKT/GeoTransform；其工程记录额外保存：

- `coordinate_context_id` 与 context hash；
- 输出 spatial reference ID/hash；
- 源 solver frame ID；
- transform plan hash；
- 垂直参考和所用 grid/resource hash。

旧工程不做运行时猜测式兼容。若需要保留已有工程，提供一次性迁移工具：无法确定 CRS、垂直语义或 frame 的项目必须让
用户补充信息，不能生成“看起来能打开”的伪 context。

## 9. 各工作流接入方式

### 9.1 控制点：第一优先级

```text
导入原始坐标 + source SpatialReference
                │
                ▼
      解析/验证水平与垂直语义
                │
                ▼
       编译 source → solver plan
                │
                ▼
转换位置与协方差，记录 provenance
                │
                ▼
ResolvedControlPoint ─────► MarkerBaAdapter / MarkerPriorLoader
```

`MarkerBaAdapter` 和 `MarkerPriorLoader` 的公共入口只能接收 `ResolvedControlPoint`。原始 marker 类型不得隐式转换为已解析
类型。特别增加回归测试：EPSG:4979 的 `[longitude, latitude, height]` 不能被当作米制 `[x, y, z]`。

### 9.2 外部相机参考

- 原始 GNSS/POS/Metashape 值继续保留 source reference、time、姿态约定和文件 SHA-256；
- resolver 使用 Chunk context 转换到 solver frame；
- 杆臂是有 frame 的 vector，穿过非线性变换时需要位置/time；
- 姿态转换必须显式 camera→world 方向与传感器轴约定；
- 缺少垂直 datum、杆臂方向或姿态约定时维持 unresolved，不降级可用；
- `CameraReferenceSolverFrame` 的通用部分删除，改用 `SolverFrameDefinition`。

### 9.3 帧式针孔、SfM、BA 和 MVS

- 相机 `Pose` 保留在 camera_core，但 frame ID 和 time 类型来自坐标模块；
- 任何动态源 frame 必须在 capture time 先转换为静态 solver pose；
- `CameraInstanceSet` 的共同 frame 校验继续保留，作为归一化后的第二道门禁；
- 数值内核只接收共同 solver frame，不在循环内部查询 CRS 或动态 provider；
- 位姿、点和先验必须携带相同 context/normalization provenance。

### 9.4 RPC

- `ground_crs` 固定引用 WGS84 geographic 3D（当前 EPSG:4979）并显式 lon/lat/ellipsoidal-height 语义；
- 射线/交会使用绑定到 WGS84 参考体的 geocentric/ECEF metre frame；
- 先以测试锁定 `RpcProjection::geodeticToEcef/ecefToGeodetic` 数值，再迁入通用椭球 operation 并删除手写副本；
- `RpcCoordinates.cpp` 使用统一 local tangent builder；
- `RpcGeospatialSupport.cpp` 使用 `coordinate_system_gdal`，UTM 选择是工作流策略而不是隐藏在相机模型内；
- 批量/逐像素路径复用编译后的 plan，不能为每个点重复解析 WKT 或创建 OGR transformation。

### 9.5 行星线阵与激光

- 轨迹持久化位置 frame、惯性/体固/传感器旋转方向、time scale、time origin 和资源 provenance；
- 每行曝光时刻调用 dynamic provider；轨迹样本必须同一时间定义、严格递增并覆盖完整成像区间；
- `PlanetaryLaserReferenceSystem` 映射到 context 中的 reference body、body-fixed frame、laser frame 和时间定义；
- `planetocentricToBodyFixedMeters()` 与球坐标协方差 Jacobian 迁入通用 planetary/ellipsoid operation；
- body-fixed frame 名称不同必须失败，除非 context 中存在经验证的显式变换路径。

### 9.6 DEM/DOM 与小天体产品

- DEM/DOM 输入输出使用 `RasterReference`；
- solver frame 与输出 CRS 分开选择；
- `DemProjectionParameters` 中可解析的字符串字段逐步改为稳定 reference ID；
- GeoTIFF 写出完整 WKT/GeoTransform，工程记录写 context/provenance；
- 小天体的球、双轴/三轴形状进入 `ReferenceBody`；投影公式可继续由 terrain 提供，但注册为可审计 operation；
- 垂直语义为相对半径、椭球高或正高时必须区分，不统一叫 `elevation` 后靠调用者猜测。

## 10. 所有权边界

| 模块 | 继续拥有 | 不再拥有 |
|---|---|---|
| `coordinate_system` | frame/reference/body/time/units、solver frame、变换图、Jacobian、provenance | 相机模型、工作流策略、Qt 对话框 |
| `camera/core` | 相机 ID、definition/instance/capability、相机 Pose 语义 | frame/time 的真实定义、坐标变换服务 |
| `camera/reference` | 外部相机参考解析策略、raw/resolved 状态、杆臂/姿态约定 | 通用 solver frame 类型 |
| `control_points` | marker 业务、源字段和控制点质量 | 通用 CRS 解析/转换算法 |
| `sfm` / `bundle_adjust` | solver 数据布局和残差 | CRS 识别或隐式 frame 转换 |
| `terrain` / `stereo_dem` | 网格/产品策略、采样和写出流程 | 通用 WGS84/UTM/CRS 变换 |
| `lidar` | 激光观测、关联和 BA 约束 | 通用行星球坐标/frame/time 变换 |
| `src/common/project` | coordinate context 的工程 DTO/序列化 | 坐标数学和 GDAL 逻辑 |
| GUI/CLI | 参数展示、选择、错误呈现 | 任何变换算法 |

## 11. 实施路线

### 阶段 0：冻结设计和冲突边界

- 本文评审通过；
- 等当前相机目录归并和线阵 schema 工作完成；
- 不同时编辑相机任务正在修改的 CMake、camera、架构文档和边界基线文件。

### 阶段 1：建立唯一基础类型

- 新增 `coordinate_system_types`；
- 将 frame ID/kind、单位、刚体变换、frame 和 time 类型直接迁移；
- 一次性更新调用点并删除 camera_core 中的旧定义；
- 不保留命名空间别名、转发头、双注册或长期兼容层。

这是一次专门的机械迁移，应独立于业务行为变更，并完成全量编译门禁。

### 阶段 2：迁移静态变换并建立 context

- 把现有刚体服务迁入 `coordinate_system_transform`，先保持数值等价；
- 增加不可变 context、plan、结构化错误和稳定 provenance hash；
- 增加 WGS84/通用椭球与 local tangent operation；
- 在 Chunk 工程格式中落 `coordinate_system` schema。

### 阶段 3：封住控制点错误路径

- 引入 raw/resolved control point；
- 接入 GDAL backend、垂直策略和 covariance propagation；
- 改造 `MarkerBaAdapter`、`MarkerPriorLoader` 只接收 resolved 输入；
- 完成 EPSG:4979→local ENU/ECEF 的端到端回归。

### 阶段 4：相机参考和 RPC 去重

- 提升 solver frame；
- camera reference resolver 改用新服务；
- RPC 手写 WGS84 变换与 RPC 空三 ENU 构造迁移后删除；
- stereo DEM/DOM 改用 GDAL adapter。

### 阶段 5：地形与成果地理参考

- 引入 RasterReference；
- DEM/DOM、mosaic、GeoTIFF 和成果 manifest 记录 context/provenance；
- 清理重复 WKT/UTM/axis-order 代码。

### 阶段 6：行星动态 frame

- 接入 reference body、planetocentric/planetographic 约定；
- 实现采样轨迹 dynamic provider；
- 线阵逐行变换和激光数据接入；
- 需要时再引入可选 SPICE backend。

阶段顺序是风险顺序，不代表每阶段必须是单个提交。每个阶段都必须保持生产路径只有一个权威实现，不能长期双写。

## 12. 测试与验收

### 12.1 基础数学

- WGS84 `(lon=0, lat=0, h=0)` 到 ECEF `(a, 0, 0)`；
- 极点、反经线、负高程和高空点 round-trip；
- metre/kilometre 与外部 foot 单位边界；
- ENU 三轴方向和已知锚点；
- 球、双轴椭球、三轴椭球已知点；
- 静态刚体结果与旧服务逐项一致。

### 12.2 CRS 与垂直语义

- EPSG 轴序在导入边界规范化，但原定义可追溯；
- WGS84 geographic 3D 与 geocentric 3D；
- relative solver 的 project unit 不得被序列化或展示为 metre；
- 南北半球 UTM 与分区边界；
- orthometric→ellipsoidal 在缺 grid 时失败；
- geographic 2D + 未知 z 不得伪装成 geographic 3D；
- backend 报告 ballpark/低精度 operation 时默认拒绝约束数据。

### 12.3 Jacobian 与协方差

- 解析 Jacobian 对有限差分；
- `J Σ Jᵀ` 的对称性和半正定性；
- pose covariance 的布局/单位/frame 不完整时失败；
- 无 anchor 的非线性 vector/direction 转换失败。

### 12.4 工作流回归

- EPSG:4979 GCP 不再作为裸 XYZ 进入 BA；
- frame pinhole/SfM/BA/MVS 仍拒绝混合 solver frame；
- RPC 正反投影、交会和旧手写 WGS84 结果在容差内一致；
- DEM/DOM round-trip 保留 WKT、完整 affine 与 vertical reference；
- 月球 planetocentric/positive-east 已知点；
- 行星 body-fixed frame 不匹配时失败；
- 线阵缺 epoch、超时间覆盖或使用不兼容 time scale 时失败；
- 每行动态变换与中心时刻近似产生可检测差异。

### 12.5 工程和可复现性

- context JSON round-trip；
- canonical hash 在 Linux/GCC 与 Windows/MSVC 上一致；
- frame/resource/context 改变后旧结果被标记失效；
- gauge normalization 不得把 solver scale 从 unresolved 改成 metric；
- 缺失 WKT/grid/trajectory/kernel 资源时给出具体错误；
- plan cache 不跨 context revision 复用；
- 批量转换失败能定位具体输入索引，约束导入默认全有或全无。

## 13. 首期明确决策与暂不处理事项

首期决定：

- 复用项目现有 GDAL/OGR，不再新增一个平行 PROJ 依赖；
- 内部 canonical 线性单位为 metre，canonical 角度单位为 radian；
- 外部经纬度 degree 只在格式/模型边界出现；
- 地理约束默认禁止未知 vertical reference 和低精度 ballpark 转换；
- context/plan/normalization 使用版本化 canonical serialization + SHA-256；
- 不设进程级可变全局 frame registry；
- 不在数值内核内做隐式转换。

首期暂不处理：

- 自动联网下载 datum grid、星历或姿态核；
- 未配置 CSPICE 时的惯性系/体固系天文推算；
- 任意参考体之间的变换；
- 把所有地形投影公式立即搬入新模块；
- 为旧工程静默猜测坐标语义；
- 用同一个 `CoordinateTransformService` 替代相机模型自身的投影方程。

## 14. 开始实现前的门禁

在写生产代码前，需要满足：

1. 相机任务完成目录归并和最终构建，释放共享 CMake/camera 文件；
2. 确认本文的术语、target 划分和无兼容层迁移策略；
3. 为阶段 1 建立精确调用点清单和独立实施计划；
4. 为阶段 3 先写出“经纬高不得进入 BA”的失败测试；
5. 明确项目是否已有必须保留的地理工程；若有，单独设计显式迁移工具。

满足这些门禁后，推荐从 `coordinate_system_types` 和控制点失败测试开始，而不是先做 GUI 或扩大投影覆盖面。
