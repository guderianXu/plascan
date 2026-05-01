// =============================================================================
// 文件: SimplePointCloudDialog.h
// 说明: 一键创建稠密点云的简洁对话框（傻瓜式操作，无需配置参数）
// =============================================================================
#pragma once

#include <QDialog>
#include <QJsonArray>
#include <QJsonObject>

class QLabel;
class QComboBox;
class QCheckBox;
class QPushButton;
class QProgressBar;
class ProjectManager;

/**
 * @class SimplePointCloudDialog
 * @brief 工作流程菜单入口的简洁稠密点云生成对话框。
 *
 * 界面只保留：
 *   - 当前 AT 状态简述（连接点数量、影像对）
 *   - 质量档位选择（快速/标准/精细，对应不同 processingScale）
 *   - "开始生成" 按钮
 *
 * 用户点击"开始生成"后，对话框关闭并发出 runRequested 信号，
 * 由 MenuWorkflowController 发起后台任务。
 */
class SimplePointCloudDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SimplePointCloudDialog(ProjectManager *projectManager,
                                    QWidget *parent = nullptr);

    /** 供外部预填充输出目录 */
    void setDefaultOutputDir(const QString &dir);

signals:
    /** 用户点击"开始生成"后发出，携带传给 startGenerateDenseCloudAsync 的设置 */
    void runRequested(const QJsonObject &settings);

private slots:
    void onStartClicked();

private:
    void setupUi();
    void loadAtInfo();
    void updateAtInfoLabel();
    QJsonObject buildSettings() const;

    ProjectManager *m_projectManager = nullptr;
    QString         m_outputDir;
    QJsonArray      m_atResults;
    int             m_bestAtIndex = -1; ///< 选中的 AT 结果索引

    // UI 控件
    QLabel      *m_infoLabel    = nullptr;
    QComboBox   *m_atResultCombo = nullptr;
    QComboBox   *m_qualityCombo = nullptr;
    QCheckBox   *m_colorsCheck  = nullptr;
    QCheckBox   *m_meshCheck    = nullptr;
    QPushButton *m_startBtn     = nullptr;
    QPushButton *m_cancelBtn    = nullptr;
};
