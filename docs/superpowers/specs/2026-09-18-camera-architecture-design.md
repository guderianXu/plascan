# PlaScan 相机体系重构设计

**日期：** 2026-09-18
**状态：** 核心边界已实施，面阵 SfM/BA/MVS 数值迁移已完成
**范围：** 面阵针孔、RPC、推扫线阵、外部相机参考，以及未来可插入的其他成像模型

**当前实施说明：** `camera_core`、三类内置模型、规范化工程相机图、能力检查入口和共享外部参考类型已经落地。
旧 `.plascan`/`images[*].camera`/`images[*].camera_file` 不再有兼容读取或双写路径；SfM、BA、MVS 现在在
`Projection + StaticPose + Optimization` 能力校验后一次性生成 `FramePinholeNumericState`，数值内核直接消费该状态，
不再保留 `StaticPinholeView` 或通过旧 `FramePinholeCamera` 运行。这一转换是运行时数值边界，不是工程格式兼容层。

## 1. 背景与问题

当前 `src/core/camera` 同时承担相机几何、Tsai/ASP 与 RPC 文件读写、工程 JSON、外部格式转换、RPC 改正、RPC 交会和相机基线计算。`FramePinholeCamera` 又把传感器内参、畸变和单幅影像外参放在同一个值类型中。SfM、BA、MVS 直接使用 `std::vector<FramePinholeCamera>`，因此“相机定义”和“影像实例”没有清晰边界。

`CameraModel` 当前只抽象了投影和射线查询。它可以覆盖一部分只读几何操作，但 RPC 没有单一物理光心，推扫相机的姿态随行时刻变化，三者不应被迫共享静态位姿或统一 BA 参数接口。

`camera_reference` 独立保存 GNSS/Metashape 等外部观测，拥有另一套位置、旋转、精度和坐标系字段。它目前能按 `image_uuid` 与 `images[*].camera` 做展示和误差比较，但没有共享的位姿、坐标框架和不确定度类型，也没有正式的参考解析到 BA 约束的桥接层。

本设计在开发阶段允许直接变更工程格式和公共接口。旧 `.plascan` 工程不需要继续打开，不保留旧字段的兼容层或过渡包装。

## 2. 目标与非目标

### 2.1 目标

1. 将传感器成像规律、影像实例状态、外部参考观测拆成独立对象。
2. 用能力接口表达“某种模型能做什么”，不要求所有模型实现同一套 BA 接口。
3. 让面阵针孔、RPC 和推扫线阵共享基础数据语义，同时保留各自特有的数学接口。
4. 让新增鱼眼、全景、事件相机或其他模型只需要新增模型实现和注册项，不修改核心工程数据结构。
5. 用显式坐标框架、单位、时间和协方差类型替代散落的字符串约定。
6. 让 `camera_reference` 使用与相机实例相同的位姿和坐标框架类型，并通过 resolver 进入比较或优化流程。
7. 让 SfM、BA、MVS、RPC 交会和推扫求解器在入口处声明能力要求，缺少能力时给出明确错误。
8. 将相机几何核心与格式 IO、工程持久化、外部格式转换解耦。

### 2.2 非目标

1. 本轮不把 RPC 或推扫强行改造成静态针孔模型。
2. 本轮不要求所有模型共享同一个数值优化器或参数块布局。
3. 本轮不引入运行时动态插件、共享库热加载或脚本模型注册。
4. 本轮不改变具体投影公式，除非现有实现无法满足新的类型边界。
5. 本轮不保留旧的 `images[*].camera` schema、旧 `ProjectCameraIO` 重载或兼容读取路径。

## 3. 总体架构

相机体系拆为四个领域对象和一组能力服务：

```text
CoordinateFrame / TimeReference / Uncertainty
                 │
                 ├── CameraDefinition
                 │       ├── FramePinholeDefinition
                 │       ├── RpcDefinition
                 │       ├── LineScanDefinition
                 │       └── future model definitions
                 │
                 ├── CameraInstance
                 │       ├── image id
                 │       ├── definition id
                 │       ├── static pose / trajectory / acquisition time
                 │       └── instance corrections
                 │
                 └── CameraReferenceObservation
                         ├── source pose / position / attitude
                         ├── source frame and time
                         ├── lever arm
                         └── covariance

+---------------- capability services ----------------+
| Projection | Ray | StaticPose | Trajectory | Optimize |
+------------------------------------------------------+
```

### 3.1 `CameraDefinition`

`CameraDefinition` 表示传感器和成像规律，不包含某一幅影像的优化外参。它必须具有：

- 稳定的 `CameraDefinitionId`；
- `CameraModelKind`，使用字符串注册值而不是集中式枚举分支，内置值为 `frame_pinhole`、`rpc00b`、`planetary_linescan`；
- 输入影像坐标约定；
- 成像世界坐标框架或地面参考框架；
- 模型参数 schema 版本；
- 能力描述和模型自身的校验结果。

定义对象为不可变值。修改标定参数或 RPC 系数会创建新的定义版本，不能在多个影像实例之间隐式改变共享状态。

### 3.2 `CameraInstance`

`CameraInstance` 表示一幅影像在一次采集中的相机状态。它必须具有：

- `ImageId`；
- `CameraDefinitionId`；
- 影像尺寸和像素原点信息；
- 采集时间或时间区间（没有时间的模型使用 `None`）；
- 对应的静态 `Pose`、时间轨迹或模型专用实例状态；
- 可选的实例级影像改正；
- 实例状态的 schema 版本。

面阵针孔实例保存静态位姿；RPC 实例通常只保存影像改正和有效区域，因为 RPC 定义本身已表达区域定位函数；推扫实例保存轨迹/姿态采样和行时间参数。这样 RPC 和推扫不需要伪造 `cameraCenter` 或静态 `R`。

### 3.3 `Pose`、`CoordinateFrame` 和 `TimeReference`

基础类型放在 `camera_core`，供 `camera` 和 `camera_reference` 共用：

```cpp
struct CoordinateFrameId
{
    std::string value;
};

struct CoordinateFrame
{
    CoordinateFrameId id;
    CoordinateFrameKind kind;       // ecef, geodetic, body_fixed, local_enu, local_cartesian
    LinearUnit linearUnit;          // metre, kilometre
    AngleUnit angleUnit;            // degree, radian
    std::optional<CoordinateFrameId> parent;
    RigidTransform toParent;
};

struct Pose
{
    CoordinateFrameId frame;
    std::array<double, 3> center;
    std::array<double, 9> cameraToWorldRotation;
};

struct PoseCovariance
{
    CovarianceLayout layout;        // diagonal or symmetric 6x6
    std::vector<double> values;
};

struct TimeReference
{
    TimeScale scale;                // utc, tai, tdb, relative
    double seconds;
};
```

所有几何接口接收显式 frame/unit，禁止仅用 `std::array<double, 3>` 表示没有来源的坐标。内部计算可以使用 ECEF 米或局部米制 frame，但转换必须通过明确的 `CoordinateTransformService` 完成。

### 3.4 `CameraReferenceObservation`

外部参考统一表示为可选观测，而不是复制一个针孔 `Pose`：

```cpp
struct CameraReferenceObservation
{
    ImageId image;
    ReferenceSourceId source;
    CoordinateFrameId frame;
    std::optional<TimeReference> time;
    std::optional<std::array<double, 3>> position;
    std::optional<Rotation> orientation;
    std::optional<PoseCovariance> covariance;
    std::optional<LeverArm> leverArm;
    bool enabled = true;
};
```

`Raw` 与 `Resolved` 仍然需要区分，但两者使用同一套坐标、姿态和协方差类型：

- `RawCameraReference` 只保留源文件语义和源 frame；
- `ResolvedCameraReference` 由 `CameraReferenceResolver` 生成，必须带目标 solver frame、转换链 hash 和可用性原因；
- resolver 不允许在缺少姿态约定、垂直基准或杆臂方向时静默猜测。

## 4. 能力接口

能力接口由 `camera_core/capabilities` 提供。能力是可组合的，模型只实现自己支持的部分。

### 4.1 几何能力

- `ProjectionCapability`：地面点到像点，返回像点、深度/残差、有效域和时间信息。
- `InverseProjectionCapability`：给定高度、DEM 或约束域的像点反投影。
- `RayCapability`：从像点生成射线；RPC 必须声明这是高度范围内的局部弦线，而不是物理光线。
- `StaticPoseCapability`：仅适用于有单一静态光心和姿态的实例。
- `TrajectoryCapability`：提供时间到位姿、行号到时间和时间到行号的映射，供推扫模型使用。
- `ImageCorrectionCapability`：实例级影像偏差或畸变改正。

### 4.2 优化能力

`OptimizationCapability` 不定义统一参数数组，而定义模型专用的参数块和残差工厂：

```cpp
class OptimizationCapability
{
public:
    virtual ParameterBlockSchema parameterSchema() const = 0;
    virtual std::unique_ptr<ObservationResidual> makeResidual(
        const ObservationContext &) const = 0;
    virtual void applyUpdate(std::span<const double> delta) = 0;
};
```

针孔可以提供内参、畸变和静态位姿参数；推扫可以提供轨道/姿态偏置或时间偏差；RPC 默认只提供影像改正参数或明确声明不可优化 RPC 系数。调用方通过 `CapabilityQuery` 获取能力，不能对不存在的能力做 `dynamic_cast` 后静默降级。

### 4.3 能力要求

每个处理流程声明自己的输入契约：

- MVS：`ProjectionCapability + StaticPoseCapability + ImageCorrectionCapability`；
- 针孔 SfM：`ProjectionCapability + StaticPoseCapability + OptimizationCapability`；
- RPC 空三：`ProjectionCapability + InverseProjectionCapability`，可选 `RayCapability`；
- 推扫空三：`ProjectionCapability + TrajectoryCapability + OptimizationCapability`；
- 仅做影像覆盖分析：只要求 `ProjectionCapability`。

入口统一通过 `requireCapabilities()` 检查，错误信息列出影像、模型和缺失能力。

## 5. 模型实现边界

### 5.1 面阵针孔

拆成：

- `FramePinholeDefinition`：焦距、主点、畸变、像元尺寸、轴向约定；
- `FramePinholeInstance`：影像尺寸、静态 pose、可选缩放/正深度工作状态；
- `FramePinholeProjection`：投影、反投影和射线；
- `FramePinholeOptimization`：位姿和可选内参/畸变参数块。

现有 `FramePinholeCamera` 不再同时承担 definition 和 instance；其投影算法已由上述实现和
`FramePinholeNumericState` 承载。`normalizedForPositiveDepth()` 和 `scaledIntrinsics()` 变成显式的
instance/derived-view 工厂，不修改原对象。

### 5.2 RPC00B

拆成：

- `RpcDefinition`：RPC00B 系数、归一化参数、地面 frame 和高度域；
- `RpcInstance`：影像尺寸、实例级 image correction、有效区域；
- `RpcProjection`：大地坐标/ECEF 转换、正反投影和高度约束反算；
- `RpcIntersectionService`：双 RPC 交会，作为服务而不是相机类成员。

RPC 不实现 `StaticPoseCapability`。任何需要光心的流程必须在类型层面拒绝 RPC，而不是生成一个近似光心后继续执行。

### 5.3 推扫线阵

拆成：

- `LineScanDefinition`：焦平面、畸变、探元和像素约定；
- `LineScanInstance`：轨道/姿态采样、行时间模型、body-fixed frame；
- `LineScanProjection`：逐行射线、观测行投影和地面到影像迭代；
- `LineScanTrajectory`：轨迹插值和姿态修正；
- `LineScanOptimization`：轨迹偏置、姿态偏置和时间偏差参数块。

线阵实例使用 `TimeReference` 和 `TrajectoryCapability`，不暴露静态 `cameraCenter` 作为通用属性。

### 5.4 未来模型

新增鱼眼、全景或事件相机时必须提供：

1. 一个 definition schema 和校验器；
2. 至少一个几何能力；
3. 一个注册工厂；
4. 能力契约测试；
5. 工程 JSON 的模型参数对象。

核心 `CameraDefinition`、`CameraInstance` 和 project schema 不增加新的模型专用字段。

## 6. 工程格式

新 Chunk 文档在 `project_files.camera_definitions` 和
`project_files.camera_instances` 中保存独立的相机集合，不再写入
`images[*].camera`：

```json
{
  "camera_definitions": [
    {
      "id": "camdef-001",
      "model_type": "frame_pinhole",
      "schema_version": 1,
      "frame": "local_enu",
      "parameters": {
        "fx_px": 1200.0,
        "fy_px": 1200.0,
        "cx_px": 640.0,
        "cy_px": 480.0,
        "distortion": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}
      }
    }
  ],
  "camera_instances": [
    {
      "image_uuid": "image-001",
      "definition_id": "camdef-001",
      "schema_version": 1,
      "image_size": {"samples": 1280, "lines": 960},
      "pose": {
        "frame": "local_enu",
        "center_m": [0.0, 0.0, 10.0],
        "camera_to_world_rotation": [1,0,0,0,1,0,0,0,1]
      }
    }
  ]
}
```

RPC 和推扫分别把自身参数放入 `parameters`，把实例级状态放入 `camera_instances` 的 `state` 对象。项目影像条目只保留 `image_uuid`、路径、启用状态和资源信息。

工程格式要求：

- Chunk schema 主版本提升；旧 schema 明确报错，不尝试猜测；
- `camera_definitions` 的 id 在 Chunk 内唯一；
- 每个 `camera_instance` 必须引用存在的 definition 和 image；
- 不允许同一影像同时存在多个生效实例；
- 所有 frame、unit、time 和参数 schema 都显式写出；
- 保存前执行完整交叉校验，失败时不写入部分数据。

## 7. 模块和目录边界

目标目录结构：

```text
src/core/camera_core/
  types/                  # id、frame、pose、time、uncertainty
  capabilities/           # capability interfaces and requirements
  model/                  # CameraDefinition / CameraInstance / registry

src/core/camera_models/
  frame_pinhole/
  rpc/
  linescan/

src/core/camera_io/
  tsai/
  rpc_raster/
  csm_isd/
  external_formats/       # COLMAP、EPFL、Metashape 等格式转换

src/core/camera_project/
  CameraProjectStore.*
  CameraProjectValidation.*

src/core/camera_reference/
  model/                  # shared reference observations
  resolve/                # CRS、姿态和杆臂解析
  compare/                # estimated/reference comparison
  io/                     # sidecar JSON and source import

src/core/camera_services/
  rpc_intersection/
  baseline/
  capability_validation/
```

依赖方向：

```text
camera_core
   ↑
camera_models ── camera_io
   ↑                 ↑
camera_project   camera_reference
   ↑                 ↑
sfm / mvs / BA / aerial / GUI services
```

`camera_reference` 依赖 `camera_core`、坐标转换服务和自身 IO，不依赖具体 `FramePinhole`、RPC 或线阵实现。GUI 只通过 project/reference services 访问相机数据，不直接拼 JSON。

## 8. 错误处理和状态规则

1. 未通过模型校验的 definition 不能构造成 `CameraDefinition`。
2. 缺失 frame、单位、时间尺度或必需能力时，工厂和流程入口返回结构化错误。
3. 参考 resolver 遇到未声明的姿态约定、杆臂方向或垂直基准时返回 `Unresolved`，并保存原因；禁止静默使用源值。
4. RPC 的近似射线必须携带 `approximation_kind` 和高度范围，不能伪装成普通物理射线。
5. 推扫投影失败要报告行号、时间迭代和轨迹范围。
6. 所有错误都带 `image_uuid`、definition id、model type 和 frame 信息，便于定位。

## 9. 测试策略

### 9.1 核心类型

- frame/单位转换的往返和非法单位拒绝；
- Pose 与旋转矩阵的正交性、右手性和有限值校验；
- covariance 布局和维度校验；
- capability requirement 缺失时的结构化错误。

### 9.2 模型能力契约

每种模型都必须通过统一契约测试：

- definition 序列化/反序列化往返；
- 投影结果的有限值和有效域；
- 能力集合与实际实现一致；
- 不支持的能力明确返回 unsupported；
- frame 和 pixel convention 不被隐式改变。

此外保留模型专用数学回归：针孔畸变、RPC 高度反算/交会、推扫行时刻和姿态插值。

### 9.3 工程与参考

- 新 project schema 的完整保存/加载；
- definition-instance 引用完整性；
- 同一影像重复实例和悬空 definition 拒绝；
- `CameraReferenceResolver` 的坐标、姿态、杆臂和协方差转换；
- 参考与估计结果按 image id 比较；
- RPC 参考记录显示为区域模型，不生成静态光心误差；
- 影像集合变化时 reference sidecar 明确失效。

### 9.4 流程回归

先迁移并通过面阵针孔 SfM/BA/MVS 定向测试，再接入 RPC 空三和推扫测试。任何流程只能使用自己声明的能力；全量测试前不删除旧数学回归数据。

## 10. 实施阶段

1. 建立 `camera_core` 基础类型、能力接口和 registry；删除旧模型中的隐式坐标/位姿重复类型。
2. 迁移面阵针孔到 definition/instance/能力服务，重写 SfM、BA、MVS 输入。
3. 迁移 RPC 定义、实例和交会服务，移除固定光心假设。
4. 迁移推扫定义、实例、轨迹和优化能力。
5. 重写 camera project store 和 Chunk schema，删除 `images[*].camera` 及旧 `ProjectCameraIO`。
6. 将 `camera_reference` 重构为共享类型、resolver、compare 和 sidecar IO，并接入参考约束入口。
7. 删除旧 `src/core/camera` 聚合边界，更新 CMake、文档、GUI 和测试。
8. 执行当前原生平台的受影响测试、全量测试和 `git diff --check`。

每个阶段都必须先有失败测试，再实现最小行为；阶段完成后才进入下一阶段。中间阶段不引入兼容字段或双写逻辑。

## 11. 验收标准

- 新工程不再出现 `images[*].camera`；
- 相机定义和影像实例可以独立复用、校验和版本化；
- 针孔、RPC、推扫都能通过统一 registry 创建，并返回准确能力集合；
- 任意流程在缺失能力时 fail-closed，并给出可定位错误；
- `camera_reference` 与相机实例共用 Pose/frame/covariance 语义；
- 新增模型不需要修改核心 project schema 或中心大类；
- 面阵 SfM/BA/MVS、RPC 空三和推扫定向的受影响测试全部通过；
- 代码中不保留旧 `camera` 字段的兼容读取、兼容写入或包装 API。
