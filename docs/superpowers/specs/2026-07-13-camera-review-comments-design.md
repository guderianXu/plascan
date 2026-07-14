# Camera 模块 Review 注释设计

## 目标

为 `src/core/camera` 下的生产代码、测试代码和 CMake 构建定义补充面向代码审查的中文注释，使审查者无需反复追踪实现即可理解：

- 相机内外参的单位和存储约定；
- camera-to-world、world-to-camera 与正深度坐标之间的转换；
- Tsai/Brown-Conrady 投影、畸变和反畸变流程；
- Middlebury、EPFL、COLMAP、Metashape 输入到 PlaScan Tsai 输出的映射；
- 路径发现、输出覆盖保护、警告与失败边界；
- 每组测试验证的行为和关键预期。

本次只增加或修正注释，不改变接口、控制流、数据格式、构建目标或测试行为。

## 覆盖范围

### 生产代码

- `src/core/camera/Camera.h`
- `src/core/camera/Camera.cpp`
- `src/core/camera/PositiveDepthCameraModel.h`
- `src/core/camera/PositiveDepthCameraModel.cpp`
- `src/core/camera/CameraFormatConverter.h`
- `src/core/camera/CameraFormatConverter.cpp`

### 测试与构建代码

- `src/core/camera/test/Camera_tests.cpp`
- `src/core/camera/test/CameraFormatConverter_tests.cpp`
- `src/core/camera/test/test_tsai_loader.cpp`
- `src/core/camera/test/test.cpp`
- `src/core/camera/CMakeLists.txt`

`src/core/camera/testdata/*.tsai` 是输入数据，不修改。

## 注释层级

### 文件级注释

每个头文件和实现文件说明文件职责、输入输出、关键约定及其在模块中的位置。`CameraFormatConverter.cpp` 按内部记录、通用解析、格式解析、路径发现、输出生成和公开入口划分逻辑区段。

### 类型与接口注释

公开类型和接口使用 Doxygen 风格，说明参数、返回值、单位、失败条件和副作用。内部结构体说明字段对应的数学或文件格式语义；简单访问器不重复解释实现代码。

### 算法与分支注释

在以下位置解释“为什么”以及必要公式：

- `R_cw` 转置为 `R_wc`，以及 `t_wc = -R_wc C`；
- `w_direction`、轴符号与正深度标准化；
- 带符号透视除法、Brown-Conrady 畸变和 Newton 反演；
- 四元数归一化、外参转置和相机中心恢复；
- Metashape 调整后标定选择、主点从影像中心偏移到绝对像素坐标；
- 自动格式识别和影像目录候选顺序；
- 输出目录非空时的覆盖保护，以及不支持参数如何进入 warnings。

不为显然的赋值、循环自增、容器插入或逐个断言添加机械翻译式注释。

## 各文件重点

### Camera

统一说明内存中焦距/主点使用像素、文件中使用毫米、相机中心使用世界坐标单位。投影和位姿接口明确 camera-to-world 存储约定、前向深度判定、畸变应用顺序及数值阈值的用途。

### PositiveDepthCameraModel

说明该类型是 MVS 使用的轻量 `float` 相机快照。公开 union 保留描述性名称和历史短别名的同一内存布局；构造过程把 ASP 轴方向归一化为 `Z > 0`，反投影使用 `R_wc^T` 和相机中心恢复世界坐标。像素变换是投影后、反投影前使用的可逆 3×3 单应矩阵。

### CameraFormatConverter

说明 `CameraRecord` 是不同外部格式的统一中间表示。每种解析器记录输入矩阵的方向、相机中心推导方式、可保留的畸变项和必须告警的丢失项。路径搜索与输出部分说明候选优先级、相对路径和 shell quoting 的用途，以及异常如何转换为 `CameraConversionResult::errorMessage`。

### 测试与 CMake

测试文件使用文件级说明和场景级注释，突出测试夹具生成、坐标转换预期、排序规则、警告条件和临时目录清理。CMake 注释区分核心库、手工诊断工具以及仅在 `BUILD_TESTS` 且 GTest 可用时生成的测试目标。

## 兼容性与错误处理

- 保留所有函数签名、枚举值、目标名和输出文件名。
- 不改变当前异常文本、警告文本、阈值或格式识别优先级。
- 不改动测试临时路径和测试数据路径。
- 不在本任务中拆分超过 400 行的既有文件，避免把注释任务扩大为重构任务。
- 保留工作区中与本任务无关的未提交修改。

## 验证

1. 检查差异，确认生产和测试源码只有注释/空白变化，CMake 只有注释变化。
2. 重新配置现有测试构建：
   `cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON`
3. 构建 camera 目标：
   `cmake --build build/windows-vcpkg-cuda-release --target camera_test test_camera_unit test_camera_tsai test_camera_format_converter -j 8`
4. 运行 camera 子目录测试：
   `ctest --test-dir build/windows-vcpkg-cuda-release/src/core/camera --output-on-failure`
5. 预期 4 个目标构建成功，现有 22 个 camera 测试全部通过。

若根级 CTest 或全量构建仍被工作区现有 QC 测试的 `0xc0000135` 运行时依赖问题中止，应单独报告，不能描述为全量通过，也不在本任务中修改 QC 模块。
