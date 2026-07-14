# 标记点与测绘控制网络

`control_points` 是 PlaScan 的统一标记点模块，覆盖人工标记、自动标靶检测、
控制点/检查点、比例尺、SfM 先验轨迹、绝对定向和 BA 约束。核心模块不依赖
Qt Widgets；GUI 通过 `src/gui/markers` 中的 repository、controller 和 dialog 调用。

## 数据模型与持久化

- `MarkerSet` 是已接受标记和比例尺的唯一运行时模型。
- 每个投影保存稳定影像 UUID、原始未旋转影像像素坐标、量测状态、精度、
  置信度、来源和影像内容签名。
- 已接受结果原子写入 `assets/control_points/marker_set.json`。
- 未归并的非编码候选、重复编号、过期影像结果和人工冲突原子写入
  `assets/control_points/detection_review.json`，不会混入正式控制网络。
- 旧 `survey_control` 元数据只在打开工程时迁移一次，迁移成功后删除旧字段，
  不做双写兼容。

投影状态包括：

- `ManualPinned`：人工确认，自动检测不能覆盖。
- `AutoDetected`：已自动检测或经人工接受的候选。
- `Predicted`：由已注册相机和三维标记预测，需人工确认后参与解算。
- `Blocked`：用户明确阻止该影像上的自动投影。
- `Disabled`：保留记录但不参与当前解算。

## GUI 工作流

1. 在照片视图右键选择“添加新标记”或“放置已有标记”。
2. 在标记面板设置连接标记、控制点或检查点角色及坐标精度。
3. 使用“工具 > 标记 > 检测标靶...”执行带真实整体进度和取消的后台检测。
4. 使用“复核检测候选...”预览冲突，创建新标记、归入已有标记或丢弃。
5. 使用聚焦量测器和预测投影完成多影像人工复核。
6. 控制点参与绝对定向和 BA；检查点与检查比例尺只报告误差。

所有检测读取项目蒙版，蒙版非零像素表示排除区域。显式配置但无法读取的蒙版
会使对应影像失败，不会静默退回无蒙版检测。

## 检测与打印

当前可用族：

- AprilTag：`tag16h5`、`tag25h9`、`tag36h10`、`tag36h11`、
  `tagCircle21h7`、`tagStandard41h12`、`tagStandard52h13`。
- 非编码：`noncoded-circle`、`noncoded-four-quadrant`。

Metashape 圆形编码 `circular12/14/16/20` 的几何检测和编号必须由本地授权
Metashape 导出的 PDF/栅格黄金集验证。未安装该语料时 GUI 会禁用这些选项，
CLI 会返回明确错误。导入方法见
`testData/photogrammetry_benchmarks/marker_targets/README.md`。

打印 GUI 和 `marker_print_cli` 共用 `MarkerSheetRenderer` 与 `MarkerPdfWriter`，
因此页面尺寸、物理直径、边距和标签布局一致。实际 PDF 可这样回读：

```powershell
marker_print_cli --family tag36h11 --ids 1,2,3 --diameter-mm 30 `
  --hide-labels --output build/marker_tests/tag36h11.pdf

.\.venv\Scripts\python.exe scripts\marker_targets\verify_marker_pdf.py `
  --pdf build/marker_tests/tag36h11.pdf `
  --detector build\windows-vcpkg-cuda-release\bin\marker_detect_cli.exe `
  --family tag36h11 --expected-ids 1,2,3 --dpi 600
```

## 目录

```text
control_points/
├── model/          MarkerSet、Marker、MarkerProjection、ScaleBar
├── io/             JSON sidecar、迁移、CSV 导入导出
├── commands/       可撤销 MarkerChangeSet
├── detection/      AprilTag、非编码检测、合并和复核队列
├── geometry/       三角化、预测和亚像素几何
├── reference/      CRS 与坐标转换
├── registration/   PriorTrack 和控制网络解算
├── quality/        投影、控制点、检查点和比例尺报告
├── print/          共享页面渲染和 PDF 输出
└── tests/          模型、IO、检测、打印和控制网络测试
```
