// =============================================================================
// 文件: MenuWorkflowController.h
// 模块: main_window
// 说明:
//   菜单业务流程控制器。处理所有菜单/工具栏动作被触发后的业务逻辑：
//   - 弹出仍在使用的工作流程与工具参数对话框
//   - 从项目 UI 设置中记忆化加载/保存对话框参数（JSON持久化）
//   - 准备输入数据（影像列表、输出目录等）并发起后台任务
//   - 通过信号将特征点可视化选项同步到 CanvasWidget
//
//   职责划分：
//   - MainMenu: 仅负责 QAction 的定义与菜单/工具栏结构的组织
//   - MenuWorkflowController: 负责"单击菜单后做什么"的全部业务逻辑
//   这种分层可使 MainWindow 类保持简洁，同时便于独立测试菜单业务流程。
// =============================================================================

#pragma once

#include <QObject>
#include <QJsonObject>
#include <QPointer>
#include <QStringList>
#include "LayerRenderer.h"
#include "ProjectSessionContext.h"

#include <functional>

class QMainWindow;
class MainMenu;
class ProjectManager;
class DialogSettingStore;
class FeatureVisualizationController;

// MenuWorkflowController: 处理菜单触发后的业务流程（对话框、参数收集、任务发起）
// MainMenu 只负责 GUI 动作定义，本类负责业务协调
class MenuWorkflowController : public QObject
{
    Q_OBJECT

public:
    /// 构造菜单业务流程控制器。
    /// @param mainWindow 父主窗口，用于创建模态或非模态对话框。
    /// @param parent QObject 父对象。
    explicit MenuWorkflowController(QMainWindow *mainWindow, QObject *parent = nullptr);

    /// 注入项目管理器，供各菜单流程查询项目状态和发起任务。
    /// @param projectManager 当前项目管理器指针，非拥有引用。
    void setProjectManager(ProjectManager *projectManager);

    /// 绑定 MainMenu 中由本控制器负责处理的菜单动作。
    /// @param mainMenu 主菜单管理器，提供各 QAction 访问器。
    void bindActions(MainMenu *mainMenu);

public slots:
    /// 从项目 UI 设置中恢复特征点显示选项，并转发给画布层。
    /// @param ui 项目级 UI 设置 JSON，包含 feature_point_visualization 配置。
    void applySavedFeatureDisplayOptions(const QJsonObject &ui);

    /// 打开工作流程中的对齐照片参数对话框；确认后启动空中三角测量 workflow。
    void openWorkflowAerialTriangulationDialog();

    /// 打开工作流程级高级设置；当前编辑空三算法、设备和数值门限。
    void openWorkflowSettingsDialog();

    /// 打开影像重叠度分析对话框。
    void openOverlapAnalysisDialog();

    /// 打开 DEM 生成对话框。
    void openCreateDemDialog();

    /// 打开正射影像生成对话框。
    void openMapProjectDialog();

    /// 打开工作流程历史报告对话框。
    void openWorkflowReportDialog();

    /// 打开通用相机格式转换对话框。
    void openCameraConvertDialog();

    /// 打开只读相机校准对比窗口，查看空三前后的内参变化。
    void openCameraCalibrationDialog();

signals:
    /// 请求 MainWindow 将新的特征点显示选项应用到 CanvasWidget。
    /// @param opts 特征点显示选项，如颜色、尺寸、形状和透明度等。
    void requestApplyFeatureDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts);

private:
    /// 按约定优先顺序从当前项目中收集影像绝对路径列表。
    /// @return 当前项目的影像绝对路径列表；若未打开项目则返回空列表。
    QStringList getProjectImages() const;

    struct SparsePrerequisiteSummary
    {
        int imageCount = 0;
        bool hasMatches = false;
        bool blockOnMatchQuality = false;
        QJsonObject prerequisiteReport;
        QStringList missingMessages;
        QStringList warningMessages;
    };

    static SparsePrerequisiteSummary summarizeSparsePrerequisites(const QStringList &images,
                                                                  const QJsonObject &meta,
                                                                  const QString &projectPath,
                                                                  const QString &algorithmId = QString(),
                                                                  const std::function<void(int, int)> &progressCallback = {});
    QJsonObject sanitizeAerialTriangulationReferencePreselection(const QJsonObject &settings,
                                                                const QStringList &images,
                                                                const QJsonObject &projectMeta) const;
    DialogSettingStore *createDialogSettingStore(const QString &settingKey);

    /// 将工作流程高级设置与空三主对话框设置合并；主对话框字段优先。
    QJsonObject mergeAerialTriangulationSettings(const QJsonObject &dialogSettings);

    void startAerialTriangulationWorkflow(const QJsonObject &settings);
    void runUnifiedAerialTriangulation(const QJsonObject &settings,
                                       const QStringList &images,
                                       const xjw::gui::project::ProjectSessionContext &session,
                                       const QJsonObject &projectMeta,
                                       const QString &outputRoot,
                                       bool fillMissingTiePoints);
    /// 各对话框记忆化设置管理器（生命周期与控制器一致）。
    DialogSettingStore *_mapSetting = nullptr;
    DialogSettingStore *_aerialTriangulationSetting = nullptr;
    DialogSettingStore *_workflowSettingsStore = nullptr;
    FeatureVisualizationController *_featureVisualizationController = nullptr;
    QPointer<QMainWindow> _mainWindow;            // 父主窗口弱引用（不拥有）
    ProjectManager *_projectManager = nullptr;    // 注入的项目管理器（非拥有引用）
};
