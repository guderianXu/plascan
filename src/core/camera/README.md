# PlaScan 相机兼容目标

`camera` 只保留为 PlaScan 内部的 CMake 兼容 target，不再拥有 C++ 源码、公开头文件或测试实现。
它通过 `INTERFACE` 依赖统一转发以下 PlaCamera 组件：

- `placamera::placamera`：相机模型、Tsai 路径/流读写和通用项目格式导入；
- `placamera::state`：canonical state 与 USGSCSM ISD 文件导入；
- `placamera::gdal`：RPC/RPB 栅格读取和 RPC 实例构造。

PlaScan 调用方直接包含 `<placamera/tsai.h>`、`<placamera/isd.h>` 或
`<placamera/rpc_raster.h>`。保留 `camera` target 是为了让现有模块逐步精简 CMake 依赖列表，
不再提供 `xjw::camera_io` API 或格式适配代码。

面阵相机统一使用 PlaCamera 的 `CentralCameraDefinition`、`CentralCameraGeometry`、
`CentralCameraModel` 与 `bindCentralCamera()`；Perspective、Fisheye、Equidistant Fisheye、
Equisolid Fisheye、Spherical、Cylindrical、完整 Metashape 标定、rolling shutter 和 rig 拓扑
均由 PlaCamera 表达。PlaScan 不在本目录增加投影公式、导入 DTO 或状态编解码。

外部相机工程的选择、工程影像匹配和 UUID 绑定位于
`src/gui/project/services/ProjectCameraImportService.*` 和 `ProjectCameraProjectImport.cpp`。
该路径直接消费 PlaCamera 的导入结果，不再执行压缩工程解包、预去畸变或 Tsai 目录输出；
Metashape 工程需提供展开后的 `doc.xml`。项目相机状态位于 `src/core/placamera_runtime/`；
项目 image UUID 与参考观测 sidecar 的绑定位于 `src/common/project/camera_reference/`。
