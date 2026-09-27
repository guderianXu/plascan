# PlaCoordinate 提炼与接入记录

## 目标

把 `src/core/coordinate_system` 的权威实现提炼为独立 C++20 库，使控制点、相机、SfM、DEM/DOM 和
LiDAR 能共享坐标类型与转换能力，同时保持 PlaScan 现有调用的源代码兼容。

## 已完成范围

1. `placoordinate::types`
   - frame/CRS/context 强类型 ID、frame 定义、time scale/reference；
   - `SpatialReferenceDefinition`、不可变 `CoordinateContext`、solver scale；
   - canonical definition/context SHA-256。
2. `placoordinate::transform`
   - 静态父子 frame 图；
   - 点、向量和旋转链转换；
   - metre/kilometre 换算、`ProjectUnit` fail-closed、transform-chain hash。
3. 可选组件
   - `placoordinate::gdal`：WKT/authority 规范化、显式轴序、非 ballpark CRS 转换、对角不确定度传播；
   - `placoordinate::state`：严格 JSON schema、完整 context round-trip 和 hash 防篡改。
4. 构建与兼容
   - standalone CMake、安装导出、component-aware dependency discovery 和外部 consumer；
   - `src/core/coordinate_system` forwarding headers 与旧 target facade；
   - PlaCamera 公开依赖 `placoordinate::types`，frame/time 为相同 C++ 类型。

## 依赖方向

```text
PlaCoordinate types <- transform
        ^             ^
        |             |
        +-- state     +-- gdal
        ^
        |
    PlaCamera <- PlaScan camera/control-points/SfM/DEM/DOM/LiDAR
```

PlaCamera 不依赖 transform/GDAL/state。PlaCoordinate 不依赖 PlaCamera、PlaScan、Qt 或 OpenCV。

## 后续边界

- 动态天体 frame/SPICE、网格资源管理和完整协方差传播不在 0.1 范围内。
- 新代码直接 include `<placoordinate/...>` 并链接 `placoordinate::...`；旧 facade 仅用于渐进迁移。
- 若未来把 PlaCoordinate 转为独立仓库/submodule，应保留当前安装 target 和 package config 名称。
