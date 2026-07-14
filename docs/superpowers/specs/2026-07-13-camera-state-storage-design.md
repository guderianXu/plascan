# Camera 参数唯一状态设计

## 目标

消除 `Camera` 中 `Intrinsics`、`Distortion`、`Pose` 参数结构与对应私有标量成员之间的重复表达，让三个参数结构成为类内部唯一的数据源，同时保持现有投影、Tsai 文件读写、单位换算和正深度模型转换行为不变。

## 当前问题

`Camera` 已公开定义 `Intrinsics`、`Distortion` 和 `Pose` 三个值类型，但类内部仍分别使用 `_focalX`、`_radialK1`、`_cameraCenter` 等标量保存同一组概念。当前不会产生两份持久状态，因为 getter 在调用时才创建快照；问题主要是字段需要在结构定义、私有成员和 getter 映射中重复维护。

## 方案选择

采用结构体直接存储方案：

```cpp
Intrinsics _intrinsics;
Distortion _distortion;
Pose _pose;
bool _isLoaded = false;
```

没有选择删除参数结构、只保留标量 getter，因为多个 CLI、GUI、BA 和 MVS 调用点需要按组读取参数。也不新增旧字段别名、废弃接口包装或第二套状态作为兼容层；相邻调用点在能提高可读性时直接改用结构化参数。

## 公共接口兼容性

- 保留 `intrinsics()`、`distortion()`、`pose()` 的返回类型和按值快照语义。
- 保留具有明确读取、单位转换或受控更新语义的现有公开操作；它们直接访问唯一结构化状态，不引入兼容副本。
- 保留 `setCameraCenter()` 不主动设置 `_isLoaded` 的既有行为。
- 保留 `scaledIntrinsics()` 只缩放像素焦距和主点、不改变 `pixelPitch` 的既有行为。
- 保留 `loadFromFile()` 每次重置内参、畸变、中心和旋转，但只有输入文件显式包含 `w_direction` 时才更新 `depthAxisFlipped` 的兼容行为。

## 头文件与实现文件边界

- `Camera.h` 只保留类型定义、数据成员和函数声明。
- 默认构造函数以及只有一个返回表达式的只读 getter 可以继续在类内定义。
- 包含校验、状态更新或多条语句的 setter 全部在 `Camera.cpp` 定义。
- 不改变任何公开函数签名，不为移动定义增加转发函数或兼容包装。

## 参数注释与模块文档

- `Intrinsics`、`Distortion`、`Pose` 的每个字段都注明单位、坐标系或数学含义。
- 删除没有抽象作用的 `Camera::PositiveDepthModel` 嵌套别名，直接使用 `PositiveDepthCameraModel`。
- 在 `src/core/camera/README.md` 记录当前支持的 Tsai 字段、单位、默认值、坐标约定和 Camera 常见用法。
- 文档区分“当前加载器接受的格式”和通用 ASP/Tsai 生态中的其他变体。

## 内部数据流

- 内参计算统一读取 `_intrinsics`。
- Brown-Conrady 畸变统一读取 `_distortion`。
- 坐标变换和深度方向统一读取 `_pose`。
- Tsai 解析直接写入三个结构；解析完成后仍在原位置完成 mm 到 pixel 的转换。
- Tsai 保存和 `PositiveDepthCameraModel` 转换继续通过既有公共行为获得相同数值。

## 错误处理

本次不改变任何错误条件、返回值或日志内容。缺少 Tsai 必选字段、非法 pitch、非法焦距和未初始化投影等路径保持原样。

## 验证

1. 先运行现有 `test_camera_unit`，建立重构前基线。
2. 增加或补强结构化 getter 的特征测试，覆盖内参、畸变、位姿及按值快照语义。
3. 重构后运行 `test_camera_unit`、`test_camera_tsai` 和 `test_camera_format_converter`。
4. 编译 `camera`、三个 camera 测试目标及依赖 `Camera` 公共接口的相关目标，确认没有接口回归。

## 范围限制

- 不修改相机数学模型、坐标系约定和单位约定。
- 不修改 `PositiveDepthCameraModel` 的存储设计。
- 不修改 Camera 格式转换规则。
- 不处理与本次重构无关的工作区现有改动。
