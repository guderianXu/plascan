# Coordinate System

该模块是 PlaScan 中 frame、CRS、solver 尺度与坐标变换语义的唯一核心所有者，位于相机、控制点、
SfM、RPC、地形等业务模块之下。

## Targets

- `coordinate_system_types`：仅依赖 C++20 标准库；提供稳定 ID、frame、时间、`SpatialReferenceDefinition`、
  不可变 `CoordinateContext`、solver 尺度门禁与 SHA-256。
- `coordinate_system_transform`：静态刚体 frame 图、单位换算和 transform-chain hash。
- `coordinate_system_gdal`：依赖 GDAL；规范化 WKT2:2019，禁止 ballpark operation，执行显式轴序转换，
  并用数值 Jacobian 把输入对角标准差传播到目标坐标单位。
- `coordinate_system_json`：依赖 nlohmann JSON；严格编解码 `chunk.coordinate_system`，不依赖 Qt。

依赖只允许由业务模块指向上述 targets；坐标模块不能反向依赖 camera、control_points、SfM 或 GUI。

## 核心不变量

- Context 注册全部 frame 与 CRS；每个 CRS 显式引用一个已注册 frame。
- solver 只引用 frame/CRS ID，不内嵌可漂移副本。
- `Metric` solver 必须同时使用 metre frame 和匹配的 metre Cartesian CRS。
- `Unresolved` solver 必须使用 `ProjectUnit`，且不能声明空间参考。
- canonical WKT 与完整 context 分别使用 SHA-256 校验；schema 1 未知字段直接拒绝。
- 经度、纬度、椭球高等角度坐标没有显式 context 时不能进入 BA/SfM。
- 原始观测保持原值；resolved 值以 context hash、solver reference 和 normalization hash 标识来源。

首个交付切片只覆盖静态地球 CRS 到单一米制 solver reference。局部 ENU 操作、动态天体 frame、SPICE、
完整协方差、网格资源以及 camera/RPC/DEM/DOM 全链路传播在后续阶段实现；不得用默认值模拟这些语义。
后续动态 transform 查询从首版接口起必须显式接收可选 `TimeReference`，不能先发布无时间参数的接口再让
推扫线阵或惯性系调用方依赖隐含时刻。
