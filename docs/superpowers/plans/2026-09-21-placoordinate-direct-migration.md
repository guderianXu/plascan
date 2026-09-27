# PlaCoordinate 内部直连迁移实施计划

**目标：** PlaScan 业务源码直接使用 `placoordinate` 头、命名空间与 CMake target；旧 `coordinate_system` 仅作为兼容层保留。

**设计：** [内部直连迁移设计](../specs/2026-09-21-placoordinate-direct-migration-design.md)。迁移只替换身份相同的 API 入口，不改算法和持久化语义。当前工作树还有相机、PlaBundle 等并行改动，所有替换只作用于精确的旧坐标 token。

## 任务一：建立防回退门禁

- 在 `src/core/coordinate_system/tests/` 增加只扫描 PlaScan `src`/`tests` 的 CMake 脚本；忽略旧坐标兼容目录。
- 门禁拒绝业务源码中的旧 include、旧命名空间和旧 CMake target。
- 先运行脚本确认它对当前旧引用报错，再做迁移。

## 任务二：公共头与业务代码直连

- 对当前存在的 `src`/`tests` C++ 文件做精确等价替换：`coordinate_system/...` → `placoordinate/...`，`xjw::coordinate_system` → `placoordinate`。
- 优先检查相机 core/reference、控制点和航三公共头的独立 include 完整性，再检查 SfM、MVS 等消费者。
- 不修改 `src/core/coordinate_system` forwarding headers 或 PlaCoordinate 实现。

## 任务三：CMake target 直连

- 将相机 core/reference、控制点和航三的旧 `coordinate_system_*` 链接改为对应 `placoordinate::*`。
- 不通过额外全局 include path 掩盖头依赖，逐个构建受影响 target。

## 任务四：验证和文档

- 运行静态门禁、PlaCoordinate 和相机/控制点/空三相关 CTest；再按影响范围扩展编译与回归。
- 执行本次修改路径的格式和 `git diff --check`，更新 README/架构与迁移文档。
- 保留旧兼容入口；不 commit、push 或清理别的任务产物。
