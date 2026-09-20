# Qt adapters

GUI 和 CLI 共用的呈现及项目事件适配层。根 CMake 在 core 后注册，core 不依赖这些目标。

| target | 实现 | 依赖和消费者 |
| --- | --- | --- |
| marker_detection_qt | markers/detection/MarkerImageAdapter | QtGui、control_points；检测 GUI/CLI 和图像转换测试 |
| marker_print_qt | markers/print/MarkerSheetRenderer、MarkerPdfWriter | QtCore/QtGui、control_points、AprilTag；打印 GUI/CLI 和标靶测试 |
| terrain_report_qt | terrain/GlobalTerrainReportRenderer | QtCore/QtGui、terrain、通用 IO；地形 GUI/CLI 提供预览回调 |
| project_recovery_qt | project/ProjectResourceRecoveryBinding | QtCore、project_workflows、项目会话和日志；ProjectManager 与恢复测试 |

公共头文件由各 target 的 include 路径导出。打印仍使用 `print/MarkerPdfWriter.h`；
地形仍使用 `GlobalTerrainReportRenderer.h`。资源事件绑定使用 `ProjectResourceRecoveryBinding.h`，
入口为 `xjw::app::project::ProjectResourceRecoveryBinding::install()`；原 core 安装接口已移除。

小天体预览回调位于原成果事务内，接收临时路径；禁止自行提前发布最终成果。
关闭预览时 core 不需要呈现写入器。同步恢复操作由 core 提供，连接生命周期由 ProjectData 管理，
重复安装沿用会话属性防止重复连接。适配层不会新增任务线程或修改取消/持久化顺序。

构建入口：`cmake --build build/linux-source-release --target marker_print_cli small_body_terrain_cli plascan_gui`。
相关测试：MarkerPdfWriterTest、AprilTagDetectorTest、SmallBodyGlobalProductGeneratorTest、
ProjectResourceCleanupTest、ProjectResourceRecoveryBindingTest，以及 CoreBoundaryContractTest。
Windows/macOS 使用其实际 preset 构建目录。Qt 运行环境和平台插件沿用统一测试入口的配置。

`PLASCAN_BUILD_QT_PRESENTATION=OFF` 关闭 `marker_print_qt`、`terrain_report_qt` 和打印 CLI。
`marker_detection_qt` 与 `project_recovery_qt` 保留；它们分别服务于图像输入和项目会话，
不属于 PDF/报告呈现开关。桌面 GUI 必须启用呈现库。
core 检测调用方改用灰度 cv::Mat；QImage 调用方包含 `detection/MarkerImageAdapter.h` 并链接
`marker_detection_qt`。`MarkerDetectorInputTest` 不链接 QtGui，`MarkerImageAdapterTest`
检查 RGB、ARGB、灰度和调色板图像转换，以及空输入、取消和蒙版尺寸错误。
