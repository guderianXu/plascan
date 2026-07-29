// =============================================================================
// 文件: MainWindow.h
// 模块: main_window
// 说明:
//   PlaScan 应用的顶层主窗口类（QMainWindow 派生）。
//   负责：
//   - 初始化并持有所有顶层 UI 组件（数据树、画布、日志面板等）
//   - 创建 ProjectManager、MenuWorkflowController 等业务对象
//   - 将菜单动作信号连接到对应的业务槽
//   - 持久化/恢复窗口、面板的 UI 设置（JSON 格式）
//   - 处理应用退出时的未保存更改提示
//
//   布局结构（从左到右）:
//     中央: WorkspaceCenterWidget（影像画布 / 模型视图）
//     停靠面板: 工作区、资源属性、照片、日志
// =============================================================================

#pragma once

#include <QMainWindow>
#include <QJsonObject>
#include <QPointer>

// MainWindow: PlaScan 主窗口
// 布局：中央工作区 + 可停靠的工作区、属性、照片、日志面板

class DataTreeWidget;
class QSplitter;
class CanvasWidget;
class LogPanel;
class MainMenu;
class AppConfigManager;
class ProjectManager;
class ProjectData;
class MenuWorkflowController;
class ReconstructionWorkflowController;
class QProgressDialog;
class QDockWidget;
class QAction;
class QListWidget;
class QTabWidget;
class ProjectDashboardWidget;
class ReferencePanelWidget;
class WorkspaceCenterWidget;
class PhotoStripWidget;
class SelectionPropertiesWidget;
class DialogSettingStore;
class TaskStatusWidget;
class QDragEnterEvent;
class QDropEvent;
class QWidgetAction;
class HenuBrandWidget;
class WorkspacePanelController;
class ProjectUiHydrator;

namespace xjw::gui::markers
{
class MarkerWorkspaceController;
class MarkerReferencePanel;
}

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    // 构造函数: 按顺序执行四个 setup 方法初始化整个主窗口
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    // closeEvent: 在用户关闭窗口时触发；若有未保存更改则弹出确认对话框
    void closeEvent(QCloseEvent *event) override;
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;

private:
    // ---- 初始化（由构造函数按顺序调用）----
    void setupUi();               // 创建核心布局：中央工作区、数据树、画布、日志面板
    void setupSelectionPanels();
    void setupLogDock();          // 初始化日志 Dock 标题栏与菜单状态同步
    void setupHenanUniversityBrand();
    void setHenanUniversityBrandVisible(bool visible);
    void setupMenuConnections();  // 将菜单/工具栏 QAction 信号连接到对应的槽
    void setupProjectManager();   // 创建所有业务对象（ProjectManager 等）并完成全局信号/槽连接
    void openMarkerFocusMeasurement(const QString &markerId, const QString &preferredImagePath = {});
    void refreshDashboardTaskSnapshots(); // 将状态栏任务快照同步到只读概览页

    // ---- UI 设置持久化辅助 ----
    void selectPhoto(const QString &imagePath, bool openImage);
    void selectResource(const QString &section, const QString &resourcePath);
    bool isProjectPhotoPath(const QString &imagePath) const;
    QJsonObject currentProjectMeta() const;
    QJsonObject currentUiSettingsSnapshot() const;
    void restoreDefaultProjectDockLayout();
    bool restoreProjectDockState(const QJsonObject &settings);
    void persistCurrentUiSettings();
    void scheduleProjectMetadataRefresh(const QJsonObject &meta);
    // saveUiSetting: 将 partial JSON 片段合并写入项目 UI 持久化设置（通过 DialogSettingStore）
    // 参数: partial - 仅包含需更新键值对的 JSON 对象
    void saveUiSetting(const QJsonObject &partial);
    // currentBottomPanelKey: 返回旧版 UI 设置使用的底部面板键名，dock 化后仅用于兼容保存。
    QString currentBottomPanelKey() const;

    // ---- 导出辅助 ----
    // exportMatchedPairsToLis: 将项目中已存在的匹配影像对导出到 export/matched_pairs.lis
    bool exportMatchedPairsToLis(QString *outputPath, QString *errorMessage) const;

    // ---- 成员 ----
    Ui::MainWindow*  _ui{};                           // Qt Designer 生成的主窗口静态布局
    QSplitter*        _mainSplitter{};                 // Designer 初始占位分割器，dock 化后仅保留中央工作区
    QTabWidget*       _leftTabs{};                     // 工作区 Dock 内的选项卡容器（概览 | 工作区 | 参考）
    QDockWidget *_workspaceDock{};
    QDockWidget *_propertiesDock{};
    QDockWidget *_photosDock{};
    SelectionPropertiesWidget *_selectionProperties{};
    PhotoStripWidget *_photoStrip{};
    ProjectDashboardWidget* _dashboard{};              // 项目概览与工作流状态（只读）
    DataTreeWidget*   _dataTree{};                     // 工作区资源树（照片/匹配/点云/DEM 等分组）
    ReferencePanelWidget* _referencePanel{};           // 参考面板（相机参数外参导入）
    WorkspaceCenterWidget* _workspaceCenter{};         // 中央工作区（影像画布 + 三维模型视图）
    CanvasWidget*     _canvas{};                       // 影像画布（从 workspaceCenter 获取的直接引用）
    WorkspacePanelController *_workspacePanels{};      // Dock/工具栏可见性与菜单动作统一管理
    ProjectUiHydrator *_projectUiHydrator{};           // 分阶段刷新项目 UI，并丢弃过期请求
    Qt::WindowStates _windowStateBeforeFullScreen{Qt::WindowNoState};
public:
    CanvasWidget* canvas() const { return _canvas; }
private:
    HenuBrandWidget*  _henuBrandWidget{};              // 主工具栏中的河南大学校徽品牌区
    QWidgetAction*    _henuBrandAction{};              // 控制品牌区在工具栏中的可见性
    LogPanel*         _log{};                          // 日志面板（日志 Dock 的内容 widget）
    MainMenu*         _mainMenu{};                     // 菜单栏封装对象（管理所有 QAction）
    AppConfigManager* _config{};                       // 应用级配置管理器（窗口状态/最近项目）
    ProjectData*      _projectData{};                  // 项目数据模型（元数据 + 文件索引）
    MenuWorkflowController* _menuWorkflowController{}; // 菜单业务流程控制器（对话框调用协调）
    ReconstructionWorkflowController* _reconController{}; // 重建菜单业务控制器
    ProjectManager*   _projectManager{};               // 项目生命周期管理（新建/打开/保存/关闭）
    xjw::gui::markers::MarkerWorkspaceController *_markerWorkspaceController{};
    xjw::gui::markers::MarkerReferencePanel *_markerReferencePanel{};
    QProgressDialog*  _openProgressDialog{};           // 打开项目期间显示的模态进度对话框
    QProgressDialog*  _saveProgressDialog{};           // 保存操作期间显示的模态进度对话框
    TaskStatusWidget* _mvsTaskStatus{};                 // MVS 状态栏任务状态
    TaskStatusWidget* _meshTaskStatus{};                // 网格重建状态栏任务状态
    TaskStatusWidget* _atTaskStatus{};                  // 空三状态栏任务状态
    TaskStatusWidget* _sgTaskStatus{};                  // 特征匹配状态栏任务状态
    TaskStatusWidget* _spTaskStatus{};                  // 特征提取状态栏任务状态
    TaskStatusWidget* _dmTaskStatus{};                  // 密集匹配状态栏任务状态
    TaskStatusWidget* _overlapTaskStatus{};             // 重叠对获取状态栏任务状态
    TaskStatusWidget* _obsNetTaskStatus{};              // 观测网络状态栏任务状态
    TaskStatusWidget* _maskTaskStatus{};                // 照片蒙版生成状态栏任务状态
    QDockWidget*      _logDock{};                      // 日志 Dock 窗口容器
    DialogSettingStore*   _featureMatchingSetting{};   // 特征匹配对话框记忆化设置
    DialogSettingStore*   _uiSetting{};                // 主窗口 UI 状态记忆化设置
    bool _applyingUiSettings{};                        // 正在恢复项目 UI，阻止中间态写回
    QJsonObject _imageViewRotations;                   // 按影像路径保存的查看旋转角度
    
    QString           _lastSelectedImage;               // 最近一次被激活的影像路径（供关联操作使用）

private slots:
    // ---- 项目生命周期 ----
    // onProjectOpened: 项目打开/创建完成后刷新标题栏、数据树、画布、最近项目等 UI
    // 参数: plascanPath - .plascan 归档文件的绝对路径
    void onProjectOpened(const QString &plascanPath);
    void onProjectClosed();
    void onProjectOpenStarted(const QString &plascanPath);
    void onProjectOpenProgressChanged(const QString &message, int percent);
    void onProjectOpenFinished(bool success, const QString &message);

    // ---- UI 设置恢复 ----
    // applyUiSettings: 根据从项目文件加载的 JSON 恢复各 UI 状态（面板可见性、日志级别等）
    // 参数: ui - 完整的 UI 设置 JSON 对象
    void applyUiSettings(const QJsonObject &ui);

    // ---- 日志面板 ----
    // onLogDisplayLevelChanged: 日志级别变化时将新级别写入项目 UI 设置
    // 参数: lvl - Logger::Level 枚举的整数值
    void onLogDisplayLevelChanged(int lvl);

    // ---- 项目管理响应 ----
    // onSaveStarted: 保存操作开始时显示模态进度对话框
    void onSaveStarted();
    // onSaveFinished: 保存操作完成后隐藏进度对话框；在状态栏显示"保存完成"或"保存失败"
    // 参数: ok - true 表示保存成功
    void onSaveFinished(bool ok);
    // onMetadataDirtyChanged: 项目脏状态变化时在标题栏末尾添加或去除 "*" 标记
    // 参数: dirty - true 表示有未保存的更改
    void onMetadataDirtyChanged(bool dirty);
    // MVS 进度状态栏更新
    void onMvsProgress(const QString &stage, int percent);
    void onMvsFinished(bool success);

signals:
    void sgCancelRequested();
    void spCancelRequested();
    void dmCancelRequested();
    void overlapCancelRequested();

public slots:
    // 特征提取状态栏进度（由 MenuWorkflowController 调用）
    void showSpProgress(int total);
    void updateSpProgress(int done);
    void hideSpProgress(bool ok);
    // 特征匹配状态栏进度（内部使用，也可外部调用）
    void showSgProgress(int total);
    void updateSgProgress(int done);
    void hideSgProgress(bool ok);
    void showDmProgress(int total);
    void updateDmProgress(int done);
    void hideDmProgress(bool ok);
    void onOverlapProgress(const QString &stage, int percent);
    void onOverlapFinished(bool success);

private slots:
    // 网格重建进度状态栏更新
    void onMeshProgress(const QString &stage, int percent);
    void onMeshFinished(bool success);
    // 空三（AT）进度状态栏更新
    void onAtProgress(const QString &stage, int percent);
    void onAtFinished(bool success);
    void onCancelAt();
    // 观测网络进度状态栏更新
    void onObsNetProgress(const QString &stage, int percent);
    void onObsNetFinished(bool success);
    // 照片蒙版生成进度状态栏更新
    void onMaskGenerationProgress(const QString &stage, int done, int total);
    void onMaskGenerationFinished(bool success);
    // onClearRecentRequested: 用户请求清空最近文件列表时触发的响应函数 用户请求清空最近打开列表，弹确认框后执行
    void onClearRecentRequested();

    // onExportMatchedPairs: 导出当前项目的匹配影像对列表到 .lis 文本
    void onExportMatchedPairs();
    void onManualPointCloudPrune();
};
