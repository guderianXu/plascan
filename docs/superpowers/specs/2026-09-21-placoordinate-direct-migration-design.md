# PlaCoordinate 内部直连迁移设计

## 目标

让 PlaScan 的内部源码直接依赖独立库 `PlaCoordinate`，使坐标类型、frame 图、单位换算、GDAL 变换和上下文持久化只有一个权威实现；同时保留旧 `src/core/coordinate_system` 兼容入口，避免破坏外部调用者和未完成迁移的工作区。

## 当前基线

- 权威实现位于 `3rdparty/placoordinate`，导出 `placoordinate::types`、`transform`、`gdal` 和 `state`。
- `src/core/coordinate_system` 当前只包含 forwarding headers、namespace alias 和旧 CMake target facade。
- PlaScan 内部仍有约 28 个文件直接 include `coordinate_system/...`，约 106 处使用 `xjw::coordinate_system`，5 个 CMake 模块引用 `coordinate_system_*` target。
- PlaCamera 已直接依赖 `placoordinate::types`，不属于本阶段的生产相机切换范围。

## 设计决策

### 1. 分批迁移，不做大爆炸删除

迁移分为三个批次，每批都保持可构建、可测试：

1. 公共 API 与基础模块：相机 core/models/reference、控制点、航三公共头和实现。
2. 业务消费者：SfM、LiDAR、MVS、DEM/DOM、mesh、stereo、CLI 与对应测试。
3. 清理与门禁：CMake target 直连、文档更新、旧入口扫描和兼容层收缩。

每批只做 include、命名空间和链接目标的等价替换，不改变数值算法、序列化字段、错误语义或运行时流程。

### 2. 兼容层保留到内部引用归零

`src/core/coordinate_system` 在本阶段不删除。旧 target 和头文件继续提供源代码兼容，但标记为迁移 facade；当内部 `src`/`tests` 不再使用旧入口后，再单独评估删除或拆出兼容包。

### 3. 依赖方向

```text
placoordinate::types
        ├── placoordinate::transform
        ├── placoordinate::gdal
        └── placoordinate::state
                ↑
      PlaScan core modules
                ↑
       PlaCamera / adapters
```

业务模块按实际能力链接最小 target：只使用 frame/time/context 的模块链接 `types`，frame 图模块链接 `transform`，CRS/GDAL 模块链接 `gdal`，JSON 上下文模块链接 `state`。不得通过旧 facade 传递隐藏依赖。

### 4. 公共头边界

新代码统一使用 `<placoordinate/...>` include 和 `placoordinate` namespace。公共头不得重新引入 Qt、OpenCV、PlaScan 或旧 `xjw::coordinate_system` 依赖。工程层若需要 Qt JSON，只在已有 project/IO 边界保留。

## 文件与模块范围

### 批次一：公共 API 与基础模块

- `src/core/camera/core/**`
- `src/core/camera/models/**`
- `src/core/camera/reference/**`
- `src/core/control_points/reference/**`
- `src/core/aerial_triangulation/model/**`
- `src/core/aerial_triangulation/reconstruction/**` 的坐标类型声明
- 对应 `CMakeLists.txt` 与单元测试

### 批次二：业务消费者

- `src/core/sfm/**`
- `src/core/lidar/**`
- `src/core/mvs/**`
- `src/core/mesh/**`
- `src/core/stereo_dem/**`
- `src/core/terrain/**`、`src/core/overlap/**` 中实际使用坐标 API 的文件
- `src/cli/**`、根 `tests/**` 中的坐标消费者

### 批次三：门禁与文档

- 将 `coordinate_system_*` 的业务 CMake 链接改为 `placoordinate::*`。
- 增加静态扫描测试：旧 include/namespace 只能出现在兼容层、迁移文档和明确的外部兼容测试中。
- 更新 `README.md`、`docs/PROJECT_ARCHITECTURE.md`、`src/core/coordinate_system/README.md` 和 CHANGELOG。
- 保留兼容层测试，确认旧 API 仍可编译。

## 错误处理与回滚

- 发现 ABI/API 或数值行为变化时，回退当前批次的 direct include/namespace 改动，不回滚其他并行模块的改动。
- 不通过增加全局 include path 或复制类型来修复编译错误。
- 若某模块的旧接口确实是公开兼容面，允许保留局部 facade，并在扫描白名单中记录具体路径和理由。

## 验收标准

1. 每个批次的受影响 target 能在 Linux/GCC 原生配置下编译。
2. PlaCoordinate 核心、GDAL/state、PlaCamera adapter 和每批受影响回归测试通过。
3. 最终业务源码中旧 include/namespace/target 引用归零，除兼容层、文档和显式白名单。
4. `placoordinate::types`、`transform`、`gdal`、`state` 的依赖边界不被扩大。
5. 不切换 PlaScan 生产相机默认路径，不删除兼容层，不改变现有工程 JSON 和坐标算法行为。

## 非目标

- 本阶段不实现 CSM、SPICE 或新的 PROJ 功能。
- 本阶段不重写坐标算法、不改变 frame graph 语义、不迁移工程 JSON schema。
- 本阶段不删除旧兼容层，也不进行无关的 PlaBundle、GUI 或任务运行时重构。
