# Recovered 模型后端

此目录是 PlaScan 核心算法的一部分。`mesh.cpp` 来自用户提供的“生成模型”源码，保留面积加权 mode-3
QEM、五轮反向三角形修复、支持度/层级裁剪和定向 region 裁剪的运算顺序。为便于核对，数值实现与常量表
保持集中；超过常规 400 行提示线。参考工程的旧点云/深度三角化入口未导入。

`../RecoveredModelBuilder.cpp` 连接 MVS 内部的 OOC 后端：

1. 读取 `recovered_model_input/manifest.json` 和 d4/d8/d16 投票后的深度平面，校验尺寸、相机顺序与 SHA-256。
2. 构建深度/采样尺度金字塔、两阶段 Morton 归并、26 邻域平衡和 CUDA 多相机直方图。
3. 单 part 多层变分求解（每层 200 轮）与自适应 marching。
4. 浮点质心条件化、QEM、还原质心、反向面修复、confidence 赋值、support trim、世界变换和 region clip。
5. 原始 RGB + 完整 Brown 相机进入参考 Vulkan 七阶段取色，再按参考规则补色和 RGB 归一化。
6. 按参考字段顺序导出 `double XYZ + uchar RGB + float confidence` PLY。

## 选择与参数

GUI 与 `mesh_reconstruct_cli` 共用 `ModelWorkflowService`。`depth_tsdf` 设置在输入目录存在
`recovered_model_input` 时自动选新后端；`reconstruction_mode=recovered_ooc` 显式要求新后端，输入缺失或
损坏时返回错误。没有该输入的历史 TSDF 工作区沿用旧路径，重新生成深度图后即可使用新后端。

模型设置 JSON 接受 `compute_mode=auto/hybrid/cuda`（均使用 CUDA）、`compute_device_index`（默认 0）、`simplifyTargetFaces`、
`recovered_diagonal_scale`（默认 true，与参考生产入口一致）和 `recovered_support_levels`。
默认层计划始终到达当前平衡树的实际最大层：最大层不大于 6 时使用单层；更深的偶数树从 6 开始、
奇数树从 5 开始，每次递增 2，例如最大层 8 使用 `[6,8]`，最大层 12 使用 `[6,8,10,12]`。
显式计划同样必须每次递增 2 且结束于实际最大层。recovered 路径会把插值设为 enabled，并关闭分块和
严格体积掩模；GUI 与服务边界都会执行该约束。结果 JSON 记录实际层计划、节点数、设备、计时及 confidence 语义。

## 能力边界

- 当前生产直方图与隐式场求解需要 CUDA；不回退到其他几何算法。求解每轮和 QEM 折叠循环可取消，其他
  阶段在边界响应取消；单 part 调度仍受主存/显存容量限制。
- 输入影像宽高须能被 16 整除，三层投票深度不能由公开 d4 合成图替代。
- OOC 缓存使用 OpenEXR 3.2.2 / zlib 1.3.2 的 PXR24；这些依赖由项目标准构建入口提供。
- 导出的 XYZ 保留 double，GUI 的 `TriMesh` 展示副本仍为 float。confidence 作为 PLY 标量字段保留，
  region 新交点的 confidence 为 0。
- 顶点色标记为 `vertex_color_algorithm=recovered_vulkan_seven_stage`；使用参考 GLSL、遮挡光栅、
  几何权重、颜色累加和外推，不调用旧 MeshColorizer。通过设备 UUID 选择同一 CUDA/Vulkan 物理 GPU。
  所有取色相机须是相同偶数分辨率；相较参考固定 3072×2304 的入口，尺寸被参数化，数值 shader 未改。
  完整 RGB 当前全部驻留主存；缺失图片、相机尺寸不符或 Vulkan 不可用均明确失败，不静默换算法。
- 构建需 Vulkan 开发库和 glslangValidator；SPIR-V 嵌入 Qt 资源，安装后无需参考目录或外部 shader 文件。
  JPEG 使用项目固定版本 libjpeg-turbo 3.1.2 的准确 IDCT/上采样参数；非 JPEG 仍是 OpenCV 解码适配，
  尚未验证与参考 PNG/TIFF 解码器的逐值等价。
- 用户参考生产链尚未完成 UV chart/packing、Natural blend、seam 等纹理阶段，因此本路径只生成
  顶点色 PLY，显式请求 OBJ/纹理会报错。独立“生成纹理”和历史工作区兼容路径仍是 PlaScan 实现，
  不能称为参考链等价；也未移植参考工程的 legacy generic UV 近似。

参考数值夹具、三层数据往返、损坏拒绝和 confidence PLY 往返由 `test_recovered_model` 验证。

## 一致性与验证边界（2026-09-08）

不能认定深度或最终模型与 Metashape 等价。此前 123 相机逐值一致仅指同一受控输入下与提供的参考源码
比较；参考 README 自身仍记录深度和最终几何差异。128 相机复核还发现旧集成遗漏重建区域内 track
选邻过滤，现已改为直接使用参考 tracked selector，不提供无 track 的旧基线回退。

当前构建、回归、128 相机重新生成深度与模型的结果及命令见
`build/tmp/recovered-full-pipeline/VALIDATION.md`。保留产物用于逐阶段复核；本机仅验证 Linux/GCC/CUDA，
不代表 Windows/MSVC 通过。面数预算不是 region 裁剪后的严格上限，模型也不保证水密。

本次同版本参考重编译比较：128/128 深度逐位一致；模型的 100,938 顶点、200,998 面及 confidence
逐项一致。RGB 有 6 个顶点不同，最大差 1/255；两侧几何都有 406 条边界边和 2 条非流形边。
不要把这一场景下的参考源码一致性推广为任意场景或 Metashape 一致性。
