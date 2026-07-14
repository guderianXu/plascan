# Camera 单一模型与 MVS 去畸变设计

## 目标

删除公开的 `PositiveDepthCameraModel`，让 `xjw::Camera` 成为项目中唯一的相机模型类型。原始影像、去畸变影像、缩放影像和极线校正影像都使用 `Camera` 的不同值实例描述，不再维护第二套字段、别名或兼容包装。

## 已确认的问题

当前 MVS 直接读取原始影像，但 `PositiveDepthCameraModel` 在转换时丢弃 `Camera::Distortion`，极线校正又向 OpenCV 传入零畸变系数。因此，只要输入影像仍带 Brown-Conrady 畸变，相机投影与实际像素就会不一致。

## 方案选择

采用“单一公开模型 + 显式影像预处理 + CUDA 私有参数块”的方案：

1. `Camera` 保存完整内参、Brown-Conrady 畸变、位姿和 Tsai 轴方向，是唯一公开真源。
2. `Camera::normalizedForPositiveDepth()` 返回同类型相机，把 `u/v/w_direction` 折叠进位姿，同时保持投影结果和畸变语义一致。
3. `Camera::projectWorldPointWithDepth()` 和 `Camera::unprojectPixel()`提供 MVS 所需的正深度投影与反投影。
4. MVS 在计算前用 OpenCV 对原始影像做真实去畸变，输出影像对应的 `Camera` 使用正深度坐标、零畸变和一致的内参。
5. 缩放及极线校正继续产生新的 `Camera` 值，不增加 `RectifiedCamera`、`ProjectionCamera` 等第二类型。
6. CUDA kernel 入口可以定义私有 `GpuCameraParams`/`CamParams`，但它只在 `.cu` 内从 `Camera` 提取数值，不进入公共头文件。

没有选择给 `PositiveDepthCameraModel` 补畸变字段，因为这会继续复制相机状态；也没有选择假定所有输入已去畸变，因为当前 CLI 和 GUI 都把原始影像路径直接交给 MVS。

## Camera 数学语义

### 正深度规范化

原相机使用 `R_wc = R_cw^T`，并允许 `uAxisSign`、`vAxisSign` 和 `depthAxisFlipped`。规范化相机使用：

```text
zSign = depthAxisFlipped ? -1 : 1
xSign = zSign * uAxisSign
ySign = zSign * vAxisSign
R_wc' = diag(xSign, ySign, zSign) * R_wc
```

规范化后 `uAxisSign = vAxisSign = +1`、`depthAxisFlipped = false`，物理前方统一为 `Z_cam > 0`。径向畸变不变；为保持切向畸变投影等价，`p1' = vAxisSign * p1`、`p2' = uAxisSign * p2`。

### 投影和反投影

- `projectWorldPointWithDepth()` 使用完整 Brown-Conrady 投影，并返回物理前向正深度。
- `unprojectPixel()` 先调用现有畸变数值反演，再根据正深度恢复带符号的相机坐标，最后通过 `R_cw` 和相机中心返回世界点。
- 无效相机、非正深度、非有限参数和反畸变失败都通过 `false` 报告，不输出伪造点。

## MVS 影像数据流

```text
原始影像 + 原始 Camera
  -> Camera 正深度规范化
  -> 按规范化 Camera 的 Brown 参数 remap 去畸变
  -> 去畸变影像 + 零畸变 Camera
  -> 可选 resize（scaledIntrinsics）
  -> 可选 stereo rectification（更新内参与位姿）
  -> PatchMatch / 深度一致性 / 融合
```

深度结果保存与其栅格严格对应的工作 `Camera`。融合取色时，对原始彩色影像执行同样的正深度规范化和去畸变，再缩放到深度栅格大小，避免颜色与几何错位。

## 组件边界

- `src/core/camera/Camera.*`：相机数学、Tsai I/O、正深度规范化、投影和反投影。
- `src/core/mvs/MvsImagePreprocessor.*`：OpenCV 影像去畸变；不保存新的相机类型。
- `src/core/mvs/EpipolarRectifier.*`：输入必须是零畸变 `Camera`，输出仍是 `Camera`。
- MVS、mesh、QC、CLI/GUI 公共数据结构统一保存 `Camera`。
- `PositiveDepthCameraModel.*` 从源码和 CMake 中删除。

## 错误处理

- MVS 预处理收到无效相机、空影像或非法焦距时返回明确错误。
- 极线校正若收到非零畸变相机则失败并提示调用方先去畸变，防止再次静默忽略。
- CUDA 只接收已经规范化且零畸变的工作相机；边界转换校验这一契约。

## 测试

1. Camera 测试覆盖带轴翻转和非零切向畸变时规范化前后投影等价。
2. Camera 测试覆盖带畸变像素的投影—反投影—世界点往返。
3. MVS 预处理测试使用合成棋盘/特征点，验证非零畸变会发生重映射、输出相机畸变清零且几何对应。
4. 极线校正测试验证非零畸变输入被拒绝，去畸变输入继续通过。
5. 编译并运行 camera、MVS、mesh、QC 和相关源代码契约测试。

## 范围限制

- 不新增鱼眼、球面或柱面相机模型。
- 不改变 Tsai 文件字段和单位。
- 不保留 `PositiveDepthCameraModel` 类型别名或转发头文件。
- 不提交或回滚工作区现有的其他改动。
