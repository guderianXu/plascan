# PlaScan Metashape 式完整标记点系统设计

**日期：** 2026-07-13  
**状态：** 已确认  
**范围：** 标记点数据、照片量测、自动标靶检测、投影预测、CRS、空三、BA、比例尺和质量报告

## 1. 目标

在 PlaScan 中实现与 Metashape 工作方式一致的完整标记点系统：

- 用户可在照片任意像素处通过右键创建新标记或放置已有标记。
- 标记统一支持无坐标连接标记、控制点和检查点三种角色。
- 支持人工投影、自动检测投影、预测投影、禁用投影和阻止投影。
- 支持圆形编码标靶、AprilTag、非编码标靶的检测、编号合并和亚像素定位。
- 支持兼容标靶 PDF 生成，便于复用同一套外业标靶。
- 支持控制点坐标、水平/高程精度、项目 CRS 和垂直基准。
- 标记点可在空三前录入，作为高可信多视轨迹参与相机注册。
- 控制点用于绝对定向和约束 BA；检查点只评估精度；比例尺可作为控制或检查约束。
- 所有修改可撤销、重做并原子持久化，关闭后重新打开不丢失。

## 2. 非目标

以下能力不属于本设计，避免与标记点系统混为一体：

- 视频目标跟踪和逐帧运动跟踪。
- 相机检校使用的框标、底片 fiducial 和棋盘格检测。
- 依赖外部云服务的标靶识别。
- 用人工标记替代正常的 SIFT + LightGlue 连接点网络。

## 3. 已确认产品决策

1. 使用独立核心模块和原子 JSON sidecar，不继续扩展内联 `survey_control` metadata。
2. 采用混合式 GUI：主照片视图快速标记，双影像聚焦量测器批量复核。
3. 照片右键菜单是创建和放置标记的主入口。
4. 自动检测兼容 Metashape 支持的圆形编码标靶与 AprilTag，并支持非编码标靶。
5. 首版包含完整 CRS、水平精度和高程精度处理。
6. 标记点需要参与空三前注册、绝对定向和后续 BA。

## 4. 模块边界

新建 `src/core/control_points`，核心代码不得依赖 Qt Widgets：

```text
src/core/control_points/
├── model/          MarkerSet、Marker、MarkerProjection、ScaleBar、CRS 类型
├── io/             sidecar 读写、版本迁移、CSV/Metashape 文本导入导出
├── commands/       可撤销的标记编辑命令和变更集
├── detection/      圆形编码、AprilTag、非编码标靶检测与合并
├── geometry/       三角化、极线、投影预测、亚像素精化
├── registration/   PriorTrack、绝对定向和 SfM 适配
└── quality/        投影、控制点、检查点、比例尺质量报告
```

GUI 仅负责展示和发出命令：

```text
src/gui/markers/
├── MarkerWorkspaceController
├── MarkerOverlayItem
├── MarkerReferencePanel
├── MarkerProjectionPanel
├── MarkerFocusMeasurementDialog
├── DetectMarkersDialog
├── PrintMarkersDialog
└── MarkerTaskRunner
```

现有 `SurveyControlDialog`、`ProjectSurveyControl` 和 CSV 导入入口迁移到上述组件后删除，不保留双写兼容层。现有 BA 控制点能力由新的适配器继续调用。

## 5. 数据模型

### 5.1 MarkerSet

`MarkerSet` 是唯一运行时真源：

- `schemaVersion`
- `projectImageRevision`
- `coordinateReference`
- `markers`
- `scaleBars`
- `detectorRuns`
- `createdAt`、`updatedAt`

工程 metadata 只记录 sidecar 相对路径、对象数量、schema 版本和更新时间。

### 5.2 Marker

每个 Marker 包含：

- `id`：不可变 UUID。
- `label`：用户可修改、项目内唯一。
- `role`：`TieMarker`、`ControlPoint`、`CheckPoint`。
- `enabled`：对象级启用状态。
- `referenceCoordinate`：可选的原始坐标、源 CRS、XY/Z sigma。
- `estimatedCoordinate`：三角化或 BA 后坐标、协方差、三维残差。
- `targetIdentity`：标靶 family、编码 ID、旋转和生成来源。
- `projections`：按稳定影像 UUID 索引的二维量测。

工程影像必须补充稳定 UUID。路径只作为诊断快照和迁移依据，不能作为长期主键。

### 5.3 MarkerProjection

每幅影像对同一 Marker 最多有一条活动投影：

- 原始未旋转影像像素坐标 `x/y`。
- `state`：`ManualPinned`、`AutoDetected`、`Predicted`、`Blocked`、`Disabled`。
- `sigmaPx`、检测置信度、重投影残差。
- 检测器、候选椭圆或四边形、创建和修改时间。
- 创建投影时的影像内容签名，用于检测影像替换造成的陈旧量测。

`ManualPinned` 和已通过门控的 `AutoDetected` 参与空三与 BA；`Predicted` 只提示；`Blocked` 禁止预测和自动合并；`Disabled` 保留数据但不参与计算。

### 5.4 ScaleBar

比例尺通过两个 Marker UUID 定义，包含：

- `measuredDistance`
- `sigma`
- `enabled`
- `role`：`Control` 或 `Check`
- `estimatedDistance` 和残差

检查比例尺只报告误差，不进入 BA。

## 6. 持久化和迁移

目录结构：

```text
assets/control_points/marker_set.json
assets/control_points/detector_runs/<run-id>.json
assets/control_points/reports/marker_quality.json
assets/control_points/exports/
```

- 所有正式写入通过 `QSaveFile` 原子替换。
- GUI 命令先修改内存快照，再异步合并保存；保存失败时保留 dirty 状态并显示具体路径。
- 后台检测任务基于启动时 revision 运行，完成后以变更集合合并，不能覆盖用户在运行期间的手工编辑。
- 打开工程时验证 schema、影像 UUID、路径快照和影像内容签名。

旧 `survey_control` 迁移步骤：

1. 将 `control_points`、`check_points` 转为统一 Marker。
2. 将旧 `observations` 转为 `ManualPinned`。
3. 将比例尺端点名称解析为 Marker UUID。
4. 原子写入并重新读取验证 sidecar。
5. 验证成功后删除旧 metadata；失败时不修改原工程。

## 7. GUI 交互

### 7.1 主照片视图

`ImageViewWidget` 已提供 `viewRightClicked(scenePos)`，在其上接入标记命令。右键位置必须从显示坐标转换为原始未旋转像素坐标。

右键菜单：

- 添加新标记
- 放置已有标记 > 新标记、当前/最近标记、选择其他标记
- 移除当前投影
- 阻止此影像投影 / 解除阻止
- 打开聚焦量测

添加新标记会同时创建 `TieMarker` 和 `ManualPinned` 投影，并在照片上进入名称内联编辑。用户录入参考坐标后可切换为控制点或检查点。

同一标记在同一影像只能有一条活动投影。重新放置属于可撤销的替换操作。

### 7.2 标记面板

主窗口使用未嵌套卡片的紧凑面板：

- 标记表：启用、名称、角色、投影数、参考残差、影像 RMS、状态。
- 参考页：源坐标、估计坐标、XY/Z 精度、CRS。
- 投影页：影像、状态、像素坐标、残差、置信度。
- 比例尺表：端点、测量距离、估计距离、残差和角色。

照片条为每幅影像显示当前标记状态：人工确认、自动检测、预测、阻止或异常。

### 7.3 聚焦量测器

- 上视图固定可靠投影，下视图逐张显示候选影像。
- 自动缩放到预测区域；仅有一个投影时显示极线搜索带。
- 候选影像按预测不确定度、交会角、清晰度和残差排序。
- 支持确认、亚像素微调、禁用、阻止和跳过。
- 每次操作立即更新内存模型和质量统计，并加入统一撤销栈。

## 8. 自动标靶检测和打印

兼容范围依据 Metashape 2.2 官方手册：

- 圆形编码：12、14、16、20 bit。
- AprilTag：16h5、25h9、36h10、36h11、Circle 21h7、Standard 41h12、Standard 52h13。
- 非编码圆点和四象限标靶。

参考：

- https://www.agisoft.com/pdf/metashape-pro_2_2_en.pdf
- https://agisoft.freshdesk.com/support/solutions/articles/31000148855-coded-targets-and-scale-bars

检测流程：

1. 按图像金字塔和内存预算分块读取。
2. 在有效蒙版区域提取椭圆或四边形候选。
3. 亚像素拟合中心、尺寸和方向。
4. 解码 family、ID、旋转与奇偶校验。
5. 影像内 NMS 去重。
6. 编码 ID 跨影像合并；冲突和低置信结果进入待复核。
7. 分批发布进度和结果，任务可取消。

AprilTag 使用成熟库。圆形编码的编号兼容性通过 Metashape 生成 PDF 建立黄金测试集，不依赖其二进制实现。

非编码标靶只在相机对齐后进行跨影像合并，使用极线距离、重投影位置、尺度和局部外观门控。未对齐时仅保存单影像候选，不猜测身份。

打印模块输出兼容 PDF，包含 family、ID 范围、物理尺寸、页面尺寸、边距和标签设置。打印输出必须通过检测器回读测试。

## 9. 投影预测和精化

- 一个有效投影且相机有相对位姿：显示极线搜索带。
- 至少两个有效投影：使用稳健多视三角化估计三维点。
- 三维点通过正深度、交会角和重投影门控后，投影到其他注册影像。
- 预测必须通过影像边界、蒙版、视角和不确定度检查。
- 预测只生成 `Predicted`，不得自动提升为有效量测。
- Refine Markers 在局部窗口执行目标类型特定的中心拟合或外观相关；通过门控后生成 `AutoDetected`。

## 10. 空三和 BA

### 10.1 SfM 高可信轨迹

给 `IncrementalSfm` 增加正式 `PriorTrack` 输入，不把人工标记写入 `.sift` 或 `.match` 缓存：

```cpp
struct PriorObservation
{
    ImageId imageId;
    double x;
    double y;
    double sigmaPx;
};

struct PriorTrack
{
    std::string markerId;
    std::vector<PriorObservation> observations;
};
```

普通 SIFT/LightGlue 轨迹负责主体网络，PriorTrack 以独立来源标签和权重参与初始像对、PnP 和三角化。若 PriorTrack 与普通轨迹在同一影像竞争同一观测，必须保留用户确认轨迹并报告冲突。

### 10.2 绝对定向

至少三个非共线控制点已三角化后：

1. 将控制点参考坐标转换到项目内部笛卡尔坐标。
2. 使用带 sigma 权重的 RANSAC + Umeyama 求七参数相似变换。
3. 检查尺度、旋转、条件数和控制点空间分布。
4. 对相机、稀疏点和标记估计坐标统一应用变换。
5. 执行带控制点约束的全局 BA。

检查点不参与变换估计和 BA，只在最终模型上计算残差。

### 10.3 BA 后端

- 第一阶段：有控制点或比例尺时明确选择 Ceres CPU，禁止静默使用不支持约束的 native CUDA。
- 第二阶段：给 native CUDA 增加控制点三维残差和比例尺距离残差。
- CUDA 结果必须通过 CPU 对照测试；参数、残差或收敛状态不一致时自动回退 Ceres CPU，并在报告中记录原因。

## 11. CRS 和精度

新增通用 CRS 服务，使用项目已有 GDAL/OGR 依赖：

- 支持 EPSG、WKT、经纬度和投影坐标输入。
- 明确经纬度轴顺序和单位。
- 支持独立水平 CRS、垂直基准说明和高程单位。
- 将参考坐标转换到项目内部局部笛卡尔系后再进入 BA。
- XY sigma 与 Z sigma 分开存储和加权。
- CRS 缺失、转换失败、垂直基准未知或单位不一致时阻止控制点参与 BA，但允许作为无坐标标记保留。

## 12. 错误处理

- sidecar 损坏：只读打开并提供备份路径，不覆盖损坏文件。
- 影像 UUID 或签名不匹配：量测标记为 stale，不参与解算。
- 编码 ID 冲突：保留候选并进入人工合并，不自动覆盖。
- 几何不足：明确显示缺少影像数、交会角或已注册相机。
- CRS 不可转换：显示源 CRS、目标 CRS和 GDAL 错误。
- 后台检测与人工编辑冲突：用户编辑优先，检测结果进入待复核。
- BA 后端不支持约束：显示回退后端和原因，不静默降级。

## 13. 测试策略

### 13.1 单元测试

- MarkerSet 序列化、原子保存、损坏文件和 schema 迁移。
- UUID、标签唯一性、每影像单活动投影约束。
- 所有编辑命令的执行、撤销和重做。
- EXIF 旋转、缩放和平移下的原始像素坐标转换。
- CRS 轴顺序、单位、EPSG/WKT 和 XY/Z sigma。
- 多视三角化、极线、预测门控和绝对定向。
- PriorTrack 与普通轨迹共同注册。
- 控制点、检查点和比例尺进入 BA 的角色差异。

### 13.2 检测测试

- 对每个支持 family、有效 ID 和 0/90/180/270 度旋转生成程序化黄金样本。
- PDF 打印后回读，编号必须一致。
- 合成数据中心定位中位误差不高于 0.15 px，P95 不高于 0.35 px。
- 真实摄影数据在中心实心圆半径不小于 5 px 时，编码识别率不低于 99%，错误编号率为 0。
- 蒙版外目标不得生成候选。
- 非编码目标在未对齐工程中不得自动跨影像合并。

### 13.3 GUI 测试

- 照片右键菜单创建和放置标记。
- 拖动、删除、阻止、角色切换和撤销/重做。
- 聚焦量测器候选排序和状态更新。
- 检测任务进度、取消和关闭窗口安全性。
- 保存工程并重新打开后状态一致。

### 13.4 端到端测试

- 无相机参数：手工/编码标记与 SIFT + LightGlue 共同完成相对 SfM。
- 至少 3 个非共线控制点完成绝对定向。
- 检查点残差不影响优化结果。
- Ceres CPU 和 native CUDA 控制点 BA 数值对照。
- Metashape 兼容标靶数据集上的编号、投影数量、相机注册率和控制点残差对比。

## 14. 分阶段交付

1. **基础数据层：** MarkerSet、影像 UUID、sidecar、迁移和撤销命令。
2. **人工量测 GUI：** 右键菜单、覆盖层、标记面板和工程持久化。
3. **几何辅助：** 聚焦量测器、极线、三角化、预测和精化。
4. **控制网络：** CRS、PriorTrack、绝对定向、控制/检查点、比例尺和 Ceres BA。
5. **自动检测：** AprilTag、圆形编码、非编码目标、后台任务和兼容性黄金集。
6. **标靶打印：** 兼容 PDF 生成与检测器回读验证。
7. **CUDA BA：** 控制点/比例尺残差、CPU 对照和自动回退。
8. **产品验收：** 真实数据、Metashape 对照、文档和完整 GUI 回归。

每个阶段必须形成可独立运行和测试的交付物，不允许一次性合并后再补测试。

## 15. 验收标准

- 右键新建和放置已有标记行为与确认的界面一致。
- 所有投影写入原始未旋转像素坐标，并可撤销。
- 关闭并重新打开工程后 Marker、投影状态、CRS 和比例尺完全一致。
- 编码标靶编号兼容性、定位精度和蒙版约束达到测试阈值。
- 人工标记和编码标靶可在空三前作为高可信轨迹参与注册。
- 控制点完成绝对定向并进入 BA；检查点只报告误差。
- 非编码标靶不在未对齐时错误跨影像合并。
- GUI 长任务不阻塞主线程，支持真实进度和取消。
- CPU/CUDA BA 未通过一致性门控时明确回退并记录原因。
