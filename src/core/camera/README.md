# Camera 与 Tsai 相机文件说明

`src/core/camera` 负责 PlaScan 的统一针孔相机模型、ASP/Tsai 文本文件读写、外部相机格式转换，以及面向 MVS 的正深度工作相机生成。

## 1. 坐标系和单位约定

`Camera` 使用以下约定：

- `cameraToWorldRotation` 是行优先存储的 3×3 旋转矩阵 `R_cw`，表示相机坐标系到世界坐标系的旋转。
- `cameraCenter` 是世界坐标系中的相机光心 `C`，PlaScan 工程中通常使用米（m）。
- 世界点转换到相机坐标系时使用：

  ```text
  X_cam = R_cw^T * (X_world - C)
  ```

- 运行态的焦距 `focalX/focalY` 和主点 `principalX/principalY` 使用像素。
- `pixelPitch` 使用 `mm/pixel`，只负责 Tsai 文件中的毫米值与运行态像素值换算。
- `uAxisSign`、`vAxisSign` 只能是 `+1` 或 `-1`。
- `depthAxisFlipped == false` 时，物理前方为 `Z_cam > 0`；为 `true` 时，物理前方为 `Z_cam < 0`。

## 2. PlaScan 支持的 Tsai 文件格式

一个 Tsai 文件描述一台相机。当前加载器按行解析，键名不区分大小写，支持 `=` 或 `:` 分隔符。典型文件如下：

```text
VERSION_3
PINHOLE
TSAI
fu = 35.0
fv = 35.0
cu = 18.0
cv = 12.0
u_direction = 1 0 0
v_direction = 0 1 0
w_direction = 0 0 1
pitch = 0.005
k1 = -0.01
k2 = 0.001
k3 = 0.0
p1 = 0.0001
p2 = -0.0001
C = 10.0 20.0 30.0
R = 1 0 0 0 1 0 0 0 1
```

`VERSION_3`、`PINHOLE` 和 `TSAI` 是常见的 ASP 文件头。当前加载器会忽略这些标识行，真正建立相机模型的是下面的参数字段。

### 2.1 字段定义

| 字段 | 必需 | 数量 | PlaScan 解释 |
|---|---:|---:|---|
| `fu` | 是 | 1 | 水平焦距，文件值结合 `pitch` 按毫米换算为像素 |
| `fv` | 是 | 1 | 垂直焦距，文件值结合 `pitch` 按毫米换算为像素 |
| `cu` | 是 | 1 | 主点横坐标，文件值结合 `pitch` 按毫米换算为像素 |
| `cv` | 是 | 1 | 主点纵坐标，文件值结合 `pitch` 按毫米换算为像素 |
| `C` | 是 | 3 | 世界坐标系中的相机中心，通常为米 |
| `R` | 是 | 9 | 行优先 `R_cw`，即 camera-to-world 旋转矩阵 |
| `pitch` | 否 | 1 | 像元尺寸，单位 `mm/pixel`；缺省值为 `1.0` |
| `k1/k2/k3` | 否 | 各 1 | `r²/r⁴/r⁶` 径向畸变系数；缺省值为 `0` |
| `p1/p2` | 否 | 各 1 | Brown-Conrady 切向畸变系数；缺省值为 `0` |
| `u_direction` | 否 | 1 或 3 | u 轴方向；缺省为 `+1` |
| `v_direction` | 否 | 1 或 3 | v 轴方向；缺省为 `+1` |
| `w_direction` | 否 | 1 或 3 | 光轴方向；z 分量为负时启用负 Z 前向 |

方向字段既可以写成 ASP 向量格式，也可以写成 PlaScan 简化标量格式：

```text
u_direction = 1 0 0
v_direction = 0 -1 0
w_direction = 0 0 -1
```

或：

```text
u_direction = 1
v_direction = -1
w_direction = -1
```

### 2.2 内参单位换算

加载成功后，文件内参按以下方式转换成运行态像素值：

```text
focalX     = fu / pitch
focalY     = fv / pitch
principalX = cu / pitch
principalY = cv / pitch
```

例如 `fu = 35 mm`、`pitch = 0.005 mm/pixel`，运行态焦距为 `7000 pixel`。如果文件本身已经使用像素值，应将 `pitch` 写成 `1`。

### 2.3 畸变和投影公式

对归一化坐标 `x = X_cam/Z_cam`、`y = Y_cam/Z_cam`：

```text
r2 = x*x + y*y
radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y

u = uAxisSign * focalX * xd + principalX
v = vAxisSign * focalY * yd + principalY
```

`projectWorldPoint()` 会检查点是否位于物理前方；`projectWorldPointSigned()` 只排除接近零的深度，允许调用方处理任意深度符号。

### 2.4 加载和保存限制

`loadFromFile()` 在以下情况下返回 `false`：

- 文件无法打开；
- 缺少 `fu/fv/cu/cv/C/R` 中任一必需字段；
- `pitch <= 0`；
- `fu <= 0` 或 `fv <= 0`。

旋转矩阵行列式明显偏离 `±1` 时会输出警告，但为了兼容历史文件不会直接拒绝加载。

`saveToFile()` 会把运行态像素内参乘以 `pixelPitch` 后写回文件，并将方向写成标量。它输出参数行，不补写 `VERSION_3/PINHOLE/TSAI` 文件头。

## 3. Camera 类使用方法

### 3.1 从 Tsai 文件加载并投影

```cpp
#include "camera/Camera.h"

xjw::Camera camera;
if (!camera.loadFromFile("camera.tsai"))
{
    // 文件不存在、字段缺失或参数非法。
    return;
}

const double world_point[3] = {100.0, 200.0, 50.0};
double pixel[2] = {0.0, 0.0};
if (camera.projectWorldPoint(world_point, pixel))
{
    // pixel[0] 为 u，pixel[1] 为 v。
}
```

### 3.2 读取结构化参数

```cpp
const xjw::Camera::Intrinsics intrinsics = camera.intrinsics();
const xjw::Camera::Distortion distortion = camera.distortion();
const xjw::Camera::Pose pose = camera.pose();

const double fx_pixels = intrinsics.focalX;
const double k1 = distortion.radialK1;
const std::array<double, 3> center = pose.cameraCenter;
```

这些 getter 返回值快照。修改返回对象不会修改 `Camera`，应使用 setter 更新相机状态。

### 3.3 手动建立相机

```cpp
xjw::Camera camera;
camera.setIntrinsics(7000.0, 7000.0, 3000.0, 2000.0);
camera.setPixelPitch(0.005);
camera.setAxisDirections(1, 1);

xjw::Camera::Distortion distortion;
distortion.radialK1 = -0.01;
distortion.radialK2 = 0.001;
camera.setDistortion(distortion);

const std::array<double, 9> rotation{{1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0}};
const std::array<double, 3> center{{10.0, 20.0, 30.0}};
camera.setPose(rotation, center);
camera.setDepthAxisFlipped(false);
```

如果输入内参是毫米，可直接使用：

```cpp
camera.setIntrinsicsMillimeters(35.0, 35.0, 15.0, 10.0, 0.005);
```

### 3.4 世界坐标和相机坐标转换

```cpp
const double world_point[3] = {100.0, 200.0, 50.0};
double camera_point[3] = {0.0, 0.0, 0.0};
camera.worldToCamera(world_point, camera_point);

const std::array<double, 9> rotation_wc = camera.worldToCameraRotation();
const std::array<double, 3> translation_wc = camera.worldToCameraTranslation();
```

其中 `translation_wc` 满足：

```text
t_wc = -R_wc * C
```

### 3.5 像素反畸变

```cpp
const double distorted_pixel[2] = {1200.0, 800.0};
double normalized[2] = {0.0, 0.0};
if (camera.undistortPixel(distorted_pixel, normalized))
{
    // normalized 为反畸变后的归一化坐标 x、y。
}
```

当前返回值只表示相机是否已初始化。Newton 迭代达到上限或遇到奇异 Jacobian 时会返回当前最佳估计；严格调用方应重新投影并检查像素残差。

### 3.6 图像缩放后的内参

```cpp
xjw::Camera half_resolution = camera.scaledIntrinsics(0.5, 0.5);
```

该方法缩放 `focalX/focalY/principalX/principalY`，不改变位姿、畸变或物理 `pixelPitch`。

### 3.7 生成正深度工作相机

```cpp
xjw::Camera positive = camera.normalizedForPositiveDepth();
```

返回值仍是完整的 `Camera`，会把 u/v/w 轴方向折叠到外参中，使有效点统一满足 `Z_cam > 0`，同时保持原投影和 Brown-Conrady 畸变等价。MVS 若需要无畸变针孔几何，必须先使用 `prepareMvsImage()` 重映射影像；该函数会同时返回正深度、零畸变且与输出影像严格对应的工作相机。

### 3.8 保存 Tsai 参数

```cpp
if (!camera.saveToFile("output.tsai"))
{
    // 目标文件无法创建。
}
```

## 4. 相关文件

- `Camera.h/.cpp`：Tsai 相机状态、投影、反畸变和文件读写。
- `CameraFormatConverter.h/.cpp`：Middlebury、EPFL、COLMAP、Metashape 等外部格式转换。
- `../mvs/MvsImagePreprocessor.h/.cpp`：将带畸变原图和 `Camera` 转换为 MVS 使用的无畸变影像及工作相机。
- `test/`：Camera、Tsai 加载和格式转换测试。
