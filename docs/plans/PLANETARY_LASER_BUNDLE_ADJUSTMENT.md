# 行星摄影测量稀疏激光测距平差

> 状态：静态 frame-camera 首版已实现
> 适用范围：行星轨道器、着陆器或巡视器的稀疏激光测高/测距 shot 与多视影像联合平差
> 当前边界：支持 PlaScan SI JSON v1 和带外部上下文的 ISIS `LidarData` JSON；不支持 line-scan/SPICE 时变轨迹

## 1. 两类激光约束是独立模式

PlaScan 明确区分以下观测模型，不能仅因为输入最终都含三维点而混用。

| 模式 | 原始观测 | 残差 | 法向 | PlaScan 入口 |
| --- | --- | --- | --- | --- |
| 扫描点云点到面 | 稠密扫描表面点、法向 | `n^T(X-Q)` | 需要 | `--laser-cloud`，带法向 PLY |
| 行星稀疏测距 shot | 发射时刻、单程斜距、标准差、落点及同期影像 | `(rho_hat-rho_obs)/sigma_rho` | 不需要 | `--laser-range-data`，JSON |

因此 `.ply`、`.las` 或 `.laz` 只是点云容器。只有 XYZ、没有斜距、历元和传感器几何时，不能恢复原始测距方程；LOLA/MOLA 一类稀疏 shot 也没有稳定局部法向，不能自动转成点到面约束。两种模式在 CLI 和服务中互斥。

ASP 的 sparse reference terrain 又是第三种模型。它把固定参考地形点 `P` 投影到左右影像并利用预先计算的视差：

```text
uL = piL(P)
r_ref = w_ref * (uL + d(uL) - piR(P))
```

它不使用法向，但也不是原始斜距约束。ASP GCP 则要求已知地面点和真实影像量测。参考：

- [ASP sparse ground truth and disparity](https://stereopipeline.readthedocs.io/en/latest/bundle_adjustment.html#sparse-ground-truth-and-using-the-disparity)
- [ASP Ground Control Points](https://stereopipeline.readthedocs.io/en/latest/tools/bundle-adjust.html#ground-control-points)
- [ASP `BaDispXyzErr`](https://github.com/NeoGeographyToolkit/StereoPipeline/blob/89801868104b3447467477e259045b047269b959/src/asp/Camera/BundleAdjustCostFuns.h#L276-L340)

## 2. 当前测距方程

对静态 frame camera，定义：

- `B`：当前 BA 与落点共同使用的天体固连坐标系；
- `C_B`：相机中心在 `B` 中的位置；
- `R_BC`：相机坐标向 `B` 的旋转；
- `ell_C`：从相机中心指向激光发射中心、在相机传感器坐标系表达的杆臂；
- `P_B`：该 shot 的天体固连落点；
- `rho_obs`、`sigma_rho`：单程观测斜距及其标准差，单位均为米。

```text
L_B     = C_B + R_BC * ell_C
rho_hat = ||P_B - L_B||
r_rho   = sqrt(w_global * w_shot) * (rho_hat - rho_obs) / sigma_rho
```

测距残差可使用按 `sigma_rho` 归一化的 Huber 损失。当前 JSON 适配器令每个 `w_shot=1`，由 CLI/服务设置全局权重；核心 BA 接口仍保留逐 shot 权重。杆臂为零时，距离不直接约束姿态；杆臂非零时，姿态通过旋转后的杆臂进入残差。

这与 ISIS 的核心 range constraint 一致，并增加了显式杆臂。ISIS 的实现把仪器位置转换到天体固连系后，与调整后的 `SurfacePoint` 求欧氏距离：

- [ISIS `BundleLidarRangeConstraint`](https://github.com/DOI-USGS/ISIS3/blob/aaa8f557bb5d1cf495d6547028835ecc5c390a9e/isis/src/control/objs/BundleUtilities/BundleLidarRangeConstraint.cpp#L124-L326)
- [ISIS `LidarControlPoint`](https://github.com/DOI-USGS/ISIS3/blob/aaa8f557bb5d1cf495d6547028835ecc5c390a9e/isis/src/control/objs/LidarControlPoint/LidarControlPoint.h#L60-L96)
- [ISIS `lrolola2isis`](https://isis.astrogeology.usgs.gov/dev/Application/presentation/Tabbed/lrolola2isis/lrolola2isis.html)

## 3. 落点三态与影像量测

每个 shot 必须显式声明 `point_mode`：

- `fixed`：落点固定在输入坐标，不创建可变落点自由度；输入可保留协方差作为元数据，但求解不使用它。
- `constrained`：落点作为独立三维参数块优化，并使用完整 `3 x 3` 天体固连 XYZ 协方差。适配器计算平方根信息矩阵 `W`，加入 `W(P-P0)`。
- `free`：落点可变且没有位置先验；必须至少有两台具有非零基线相机的真实 `measured` 像点，否则输入验证拒绝。

shot 落点不是普通 SfM `BATrack`，不进入普通 track 的重投影 RMS、过滤统计或有效 track 比例。真实 `measured` 像点作为该独立辅助落点的重投影残差；`projected` 像点永远忽略。

ISIS `LidarData` 中的 measures 是由落点反投影得到的虚拟量测。ISIS 会在平差中重新投影并将其影像残差置零，因此 PlaScan 导入时无论 JSON 内是否出现 `kind` 字段，都强制标记为 `projected`，绝不冒充真实像点。参考：

- [ISIS lidar 虚拟 image measures](https://github.com/DOI-USGS/ISIS3/blob/aaa8f557bb5d1cf495d6547028835ecc5c390a9e/isis/src/control/objs/LidarControlPoint/LidarControlPoint.cpp#L157-L213)
- [ISIS BundleAdjust 的 lidar measure 处理](https://github.com/DOI-USGS/ISIS3/blob/aaa8f557bb5d1cf495d6547028835ecc5c390a9e/isis/src/control/objs/BundleAdjust/BundleAdjust.cpp#L597-L639)

## 4. PlaScan SI JSON v1

解析器严格拒绝未知字段，不做单位、坐标系或语义猜测。下例中的字段名和值与当前实现完全一致：

```json
{
  "schema": "plascan.planetary_laser_dataset",
  "version": 1,
  "sensor_model": "frame",
  "range_type": "one_way",
  "units": {
    "length": "m",
    "angle": "deg",
    "time": "s",
    "pixel": "px"
  },
  "reference": {
    "target": "MOON",
    "body_fixed_frame": "IAU_MOON",
    "laser_frame": "LRO_LOLA",
    "time_system": "TDB_ET_SECONDS",
    "latitude_type": "planetocentric",
    "longitude_direction": "positive_east"
  },
  "shots": [
    {
      "id": "LOLA_EXAMPLE_000001",
      "point_mode": "constrained",
      "ephemeris_time_s": 812345678.125,
      "range_m": 50123.456,
      "range_sigma_m": 1.0,
      "point_body_fixed_m": [
        1737200.25,
        12000.5,
        5000.75
      ],
      "point_covariance_body_fixed_m2": [
        [100.0, 2.0, 0.0],
        [2.0, 100.0, 0.5],
        [0.0, 0.5, 4.0]
      ],
      "simultaneous_image_ids": [
        "NAC_FRAME_000042"
      ],
      "image_measures": [
        {
          "image_id": "NAC_FRAME_000042",
          "sample_px": 2531.25,
          "line_px": 1742.5,
          "kind": "measured",
          "covariance_px2": [
            [1.0, 0.0],
            [0.0, 1.0]
          ]
        },
        {
          "image_id": "NAC_FRAME_000043",
          "sample_px": 2529.0,
          "line_px": 1740.0,
          "kind": "projected"
        }
      ],
      "lever_arm_sensor_m": [0.42, -0.08, 1.15]
    },
    {
      "id": "LOLA_EXAMPLE_000002",
      "point_mode": "fixed",
      "ephemeris_time_s": 812345678.225,
      "range_m": 50124.125,
      "range_sigma_m": 1.2,
      "point_planetocentric": {
        "latitude_deg": 1.25,
        "longitude_deg": 42.5,
        "radius_m": 1737400.0
      },
      "simultaneous_image_ids": [
        "NAC_FRAME_000042"
      ],
      "lever_arm_sensor_m": [0.42, -0.08, 1.15]
    }
  ]
}
```

格式约束：

- 顶层 `sensor_model` 仅接受 `frame`、`line_scan`、`unknown`；`range_type` 仅接受 `one_way`、`round_trip`、`unknown`。
- `units` 必须恰为 `length=m`、`angle=deg`、`time=s`、`pixel=px`；时间语义由 `reference.time_system=TDB_ET_SECONDS` 声明。
- `reference` 必须显式提供 `target`、`body_fixed_frame`、`laser_frame`、`time_system`、`latitude_type=planetocentric` 和 `longitude_direction=positive_east`。
- 每个 shot 必须恰好提供 `point_body_fixed_m` 或 `point_planetocentric` 之一。球面输入立即转换成天体固连 XYZ 米制坐标。
- `constrained` 必须提供对称正定的完整 `point_covariance_body_fixed_m2`；`free` 禁止携带该软先验。
- `simultaneous_image_ids` 至少一个且不能重复；`image_measures` 可省略。若提供，`kind` 必须显式为 `measured` 或 `projected`。
- `covariance_px2` 可省略；当前核心像点观测只支持标量权重，因此适配器仅接受 `sigma^2 I` 形式的各向同性矩阵，并映射为 `1 / sigma^2`。各向异性或相关协方差会明确拒绝，不能静默压缩成标量。
- `lever_arm_sensor_m` 必须显式提供；零杆臂也写 `[0.0, 0.0, 0.0]`。

## 5. ISIS `LidarData` JSON 导入

ISIS JSON 本身不携带 PlaScan 求解所需的目标、坐标系、相机模型、range 单/往返语义和杆臂。因此解析时必须由调用方提供外部上下文；CLI 对应 `--laser-range-isis-*` 参数，并要求零杆臂也显式给出。

导入换算：

- ISIS `range` 和 `radius`：千米转米；`sigmaRange` 保持米。
- `latitude`、`longitude`、`radius`：按 planetocentric、positive-east 转天体固连 XYZ。
- 有 `aprioriMatrix` 时映射为 `constrained`，没有时映射为 `free`。
- 所有 ISIS `measures` 强制映射为 `projected`。

尤其要注意：ISIS `aprioriMatrix` 不是 XYZ 协方差。它是球面坐标 `(latitude radians, longitude radians, radius meters)` 的协方差：角度块单位为 `rad^2`，角度/半径交叉项为 `rad*m`，半径块为 `m^2`。PlaScan 在 shot 的纬度、经度和半径处计算球面到 XYZ 的雅可比 `J`，并执行：

```text
C_xyz = J * C_spherical * J^T
```

所得完整 XYZ `3 x 3` 协方差才进入 BA 白化先验。不能把 ISIS 六个上三角数值直接解释成米制 XYZ 方差。

## 6. 数据关联与严格安全边界

当前适配器只处理静态 frame camera：

1. `simultaneous_image_ids` 通过工程影像路径、文件名、stem、工程 ID 或调用方提供的稳定别名映射相机。
2. 匹配按特异性逐级进行：规范化完整 ID 唯一命中后立即采用，只有完整 ID 未命中时才回退到文件名和 stem；弱别名歧义不会覆盖已经唯一命中的完整 ISIS serial。
3. 每个 shot 必须唯一映射到一台同期 frame camera；歧义、多相机映射或未映射默认失败。只有显式启用 `allowUnmappedShots` 才跳过不属于当前影像集的 shot。
4. 当前每个 shot 只创建一个落点参数块和一条 range constraint，因此 ISIS `simultaneousImages` 若映射到多台相机会明确拒绝。ISIS 完整实现可让同一落点被多个同期 range constraint 共享；这不属于当前 single-frame/single-simultaneous-image 首版。
5. GUI 和 CLI 自动按 BA 相机顺序合并工程 `image_uuid`；ISIS `serialNumber` 等额外标识可通过 CLI 重复传入 `--laser-range-image-alias CAMERA_INDEX=IMAGE_ID`。自动 UUID 与显式别名同时保留，并写入运行选项以便复核。
6. 真实 `measured` 像点的影像 ID 未映射时默认失败；只有显式启用 `allowUnmappedMeasuredImages` 才忽略。ISIS 导入的 `projected` 像点不进入该规则和重投影残差。
7. 不按“最近时间”或“最接近影像”猜测相机。
8. 当前相机/track 坐标系字符串必须与 `reference.body_fixed_frame` 完全一致；适配器不执行隐式坐标转换。
9. 非零杆臂时，调用方声明的相机传感器 frame 必须与 `reference.laser_frame` 完全一致；当前不做传感器 frame 旋转。
10. `round_trip` 被拒绝；必须先按产品定义换算成单程几何距离。`unknown` 只有在调用方显式确认后才能按单程处理。
11. `line_scan` 被拒绝。`ephemeris_time_s` 当前仅用于保留来源和报告，不驱动相机轨迹求值。

这意味着当前实现可以验证 frame-camera 稀疏斜距联合平差链路，但不能把轨道器推扫影像的单个位姿冒充 shot 历元位姿。完整 line-scan 支持仍需 SPICE/等价轨迹、逐行曝光时间、位置和姿态插值、时钟与 frame 变换。

## 7. 求解、服务、CLI 与 GUI

- 核心 `bundle_adjust` 通过独立 `BALaserRangeConstraint` 参数块接入 Ceres CPU/CUDA BA；Legacy CPU 和 Native CUDA 不会静默忽略该能力，自动后端选择要求支持测距约束。Ceres 候选失败或被质量门控拒绝时直接报告失败，不会回退到不支持 range 的 Legacy。
- `BundleAdjustService` 加载 JSON、按求解相机顺序建立影像别名、调用适配器并写出 `planetary_laser_range_summary`，其中包含接受/跳过 shot、fixed/constrained/free 数量、忽略的 projected measures、range RMS 前后值，以及逐 shot 优化落点、相机索引和相机坐标系杆臂。服务拒绝用未建立索引对应关系的影像列表猜测相机顺序。
- 行星激光 dry-run 仍会执行 JSON、传感器模型、坐标系和影像别名预校验；它只跳过实际求解与结果写回，不能让无效输入伪装成检查成功。
- CLI 使用 `--laser-range-data`；必须显式提供 `--laser-range-camera-frame`。ISIS 输入还要提供目标、body frame、laser frame、相机模型、range 类型和三分量杆臂上下文；工程 UUID 自动合并，`--laser-range-image-alias` 用于把 ISIS `serialNumber` 等产品标识绑定到明确的相机索引。
- GUI 将合法 PlaScan SI JSON 识别为 `planetary_laser_shots` 参考数据。执行“参考地形约束重新平差”时优先选择该数据，要求用户确认求解坐标系；非零杆臂还要求确认传感器 frame。预览和质量提示单独显示 shot 数、目标/frame、数据路径与 range RMS。GUI 当前不补录 ISIS 缺失上下文，ISIS JSON 应通过 CLI 或先转换为 PlaScan SI JSON。

CLI 示例：

```powershell
bundle_adjust_cli project.plascan `
  --laser-range-data shots.json `
  --laser-range-camera-frame IAU_MOON `
  --laser-range-camera-sensor-frame LRO_LOLA `
  --laser-range-weight 1.0 `
  --laser-range-huber-delta-sigma 3.0
```

## 8. 可观性与解释限制

单条 range 只有一个标量。自由落点只由一条距离约束时仍位于球面上，不能独立恢复三维位置；这就是 `free` 必须有至少两台非零基线相机真实像点的原因。

- 固定或有可靠完整协方差的落点，可以沿激光视线方向约束相机位置。
- 零杆臂时，range 对相机姿态没有直接敏感性；非零杆臂只提供通常较弱的姿态敏感性。
- 多条近似平行、集中在小区域的 shot 仍可能病态，应与影像 tie points、GCP 或姿轨先验联合使用。
- 初始落点不能与杆臂修正后的激光发射点重合；这种零距离状态会在进入 Ceres 前明确拒绝，避免距离范数在零点产生未定义导数。
- 已知米制 range 有助于尺度，但不保证消除 BA 的全部 gauge；绝对定位仍需可靠天体固连锚点或相机先验。
- 若落点和 range 来自同一套不确定轨道解，它们可能相关；当前模型按输入协方差与 range sigma 独立处理，使用者需评估重复计权风险。

## 9. 当前测试覆盖与后续工作

现有测试覆盖：

- PlaScan SI v1 严格字段、单位、点模式、XYZ/planetocentric 转换和完整协方差；
- ISIS 千米/米换算、显式上下文、球面协方差到 XYZ 的雅可比转换及 projected measure 保护；
- shot 唯一相机映射、完整 ISIS serial 优先、工程 UUID/显式别名合并、未映射真实像点默认拒绝、坐标系/杆臂 frame、line-scan/round-trip 拒绝和 Free 可观性；
- 各向同性像点协方差到标量权重的精确映射，以及各向异性/相关协方差的显式拒绝；
- Ceres 的 Fixed、Constrained、Free、杆臂姿态耦合、独立统计、不支持后端诊断，以及质量拒绝后禁止无 range 的 Legacy 回退；
- `BundleAdjustService` 的 SI/ISIS 端到端关联、严格相机顺序、结果/杆臂写出和 line-scan 拒绝；
- GUI 平差预览和质量摘要中的行星激光 shot/range 指标，包括初始零残差退化为正残差的告警；
- 参考数据识别只把具有 ISIS `id/time/range/sigmaRange/latitude/longitude/radius` 点签名的 JSON 判为 LidarData，不会把普通 `points` 数组误分类；CLI 单独给出 range 权重或 Huber 参数而未给数据文件时会明确报错。

后续工作按以下边界推进：

1. 引入统一 SPICE 时间、天体 frame 和仪器 frame 变换层。
2. 为 line-scan 建立逐行曝光时间及 `C_B(t)`、`R_BC(t)` 轨迹参数化。
3. 在真实标定数据支持下增加 boresight、时钟偏差、光行时及相关协方差模型。
4. 增加按 shot 时间和影像曝光覆盖构建 `simultaneous_image_ids` 的预处理工具；求解器本身继续禁止隐式最近时间关联。
