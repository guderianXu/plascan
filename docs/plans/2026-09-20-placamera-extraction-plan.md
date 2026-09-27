# PlaCamera 提炼与演进计划

## 目标

把 PlaScan 中已经形成的 typed camera model 能力提炼为独立的 C++20 库
`PlaCamera`，提供稳定、可安装、可测试的模型接口，并逐步覆盖 frame、RPC00B
和 planetary line-scan。项目目标是承担与 USGS CSM 相近的“相机模型契约与
生态接入”职责，但只有在真实链接并通过 csmapi 兼容测试后，才声明 CSM
compatibility。

命名约定：

- 产品名：`PlaCamera`
- 仓库/包名：`placamera`
- C++ namespace：`placamera`
- CMake target：`placamera::placamera`

## 依赖结论

核心 target 依赖 C++20 标准库和 `placoordinate::types`。CMake 3.21+ 是构建依赖，GTest 是可关闭
的私有测试依赖。其余库按能力拆分：

| 组件 | 候选库 | 决策 |
|---|---|---|
| 模型状态 JSON | nlohmann/json | 已实现为 `placamera::state` 私有依赖；MessagePack 后续再评估 |
| 任意 CRS/椭球/epoch 转换 | PROJ | 后续 `placamera_proj` 可选依赖 |
| USGS CSM API/plugin | csmapi | 后续 `placamera_csm` 可选依赖 |
| 星历、姿态、frame kernel | CSPICE | 后续 `placamera_spice` 构建器依赖 |
| 影像与元数据 IO | GDAL/OpenCV/libtiff/libzip | 暂留 PlaScan 应用适配层 |
| 小型矩阵运算 | Eigen/PlaMatrix | v0.1 不需要 |

## 分层

```text
PlaScan GUI / CLI / project IO
          |
          v
PlaScan project IO  ---- placamera_state / proj / csm / spice (optional)
          |                         |
          +------------+------------+
                       v
              placamera::placamera
       typed frames + model contract + kernels
```

核心 API 不包含 Qt、OpenCV、GDAL、PlaScan/xjw、JSON DOM 或 PROJ 类型。
定义对象承载可复用标定，模型实例绑定影像、位姿、尺寸和采集时间；求值失败
通过结构化结果返回，构造期不变量通过 `CameraValidationError` 拒绝。

## 阶段

### PC0：独立工程门禁

- 独立配置、构建、测试、安装和 `find_package` 消费。
- 公共头逐个独立编译，并扫描禁止依赖和本机绝对路径。

### PC1：公共模型契约

- typed frame、time、pose、image coordinate、capability。
- `RasterModel::groundToImage` 和 `imageToImagingLocus`。
- 统一精度、迭代、图像范围选项和结构化失败。

### PC2：frame-pinhole

- Brown-Conrady 畸变、正/负深度轴、像素轴符号。
- 投影、反投影到给定深度、成像光线和正深度归一化。
- 与 PlaScan 现实现建立固定夹具 parity 测试。

### PC3：RPC00B

- 迁移归一化、20 项多项式、bias correction 和逆解。
- 明确 geodetic/ECEF 与 ground frame 边界。
- 以已知 RPC 元数据和往返误差建立测试。

### PC4：planetary line-scan

- 迁移时间映射、轨迹插值、光学模型和逐行反解。
- 将 CSPICE/ALE/ISD 读取留在可选 builder，不让 kernel 成为运行依赖。

### PC5：model state 与 registry

- 版本化 canonical state、确定性 JSON、MessagePack、schema migration。
- registry 只依赖字符串/字节状态，不将 JSON DOM 暴露给核心 ABI。

### PC6：生态 adapter

- PlaScan 工程 IO：直接读写 PlaCamera 定义与实例；不建立新旧相机类型的兼容层。
- PROJ adapter：CRS 到 model frame 的显式转换。
- CSM adapter：链接 csmapi，并用官方契约测试验证真实兼容性。

### PC7：生产迁移

- 以固定夹具对照验证投影结果，按完整调用边界迁移 GUI/CLI/project runtime；不保留运行时双模型或兼容 facade。
- 最后一个旧类型消费点迁移后删除旧实现，并完成全项目构建、相关回归与工程数据迁移说明。

## 当前完成范围

0.2 已完成 PC0—PC5：三类模型、definition/instance set、优化参数布局、frame
数值状态、RPC bias/intersection、两阶段 registry，以及严格 canonical JSON codec 均已
进入独立库。canonical project 原子加载与回写、固定夹具多点 parity 和两幅跟踪 RPC GeoTIFF
实数门禁已具备；PROJ/CSM/SPICE adapter 尚未实现。用户已明确要求不保留兼容层，因此 PC7
按直接调用边界迁移：MVS 已直接消费 PlaCamera，SfM 的 PnP 和外部姿态先验身份边界已切换，
SfM 重建状态、BA、部分 GUI/CLI 仍需迁移；完成跨平台与真实工程回归后才能删除旧实现。
航三针孔输入已经直接持有 PlaCamera 实例，旧数值状态不再由 `SfmAttemptRunner`
手工拼装，而只在进入仍未迁移的 `IncrementalSfm` 时生成。
旧 SfM 质量诊断所需的后方点投影现由 PlaCamera 的 `groundToImageSigned()`
明确提供；严格 `groundToImage()` 仍只接受正深度。独立 BA 报告已用原生
接口恢复该诊断残差。PlaCamera 原生双目交会现返回 SfM 筛点所需的交会角、
射线错距、逐相机像素残差和按两张影像欧氏残差定义的 RMS，并拒绝后向交会；
旧 `Intersection` 的生产调用仍需在 SfM 状态切换时直接删除。

下一处完整切换边界是 `sfm_core`：`SfmReconstruction` 的相机存储、`IncrementalSfm`
的预加载相机和已知位姿输入、`Triangulator`/`Intersection` 的射线几何、
内部 BA 的相机数值状态以及航三结果写出需共同使用 PlaCamera。不能仅在
`addImageWithCamera()` 入口转换回旧状态，否则旧求解状态和
`plabundle_adapter` 会继续留在生产链。当前 GUI 和独立 BA CLI 已移除
对该适配目标的显式链接；待 `sfm_core` 全链切换后删除适配目标和旧
`FramePinholeNumericState`，再执行源代码依赖扫描与项目级回归。
