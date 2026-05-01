// =============================================================================
// 文件: CreateDemDialog.h (Enhanced Version)
// 说明: 增强版相对 DEM 参数配置对话框声明。
//       支持两种模式：
//       1. 自动模式：仅需 2 张影像 + 2 个相机文件 → 自动运行完整流水线
//       2. 手动模式：可指定中间数据（特征点、匹配、稀疏点云、密集点云）
//       智能检测已有中间结果并提供复用选项
// =============================================================================
#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

class QLineEdit;
class QDoubleSpinBox;
class QComboBox;
class QLabel;
class QPushButton;
class QRadioButton;
class QGroupBox;
class QCheckBox;
class QListWidget;
class QStackedWidget;
class QProgressBar;
class ProjectManager;

// CreateDemDialog — 增强版相对 DEM 参数配置对话框
class CreateDemDialog : public QDialog
{
    Q_OBJECT

public:
    // 工作流程模式
    enum class WorkflowMode
    {
        Automatic,  // 自动模式：从影像开始，运行完整流水线
        Manual      // 手动模式：指定中间数据
    };

    // 流水线步骤状态
    struct PipelineStepStatus
    {
        bool hasFeatures = false;      // 是否有特征点
        bool hasMatches = false;       // 是否有匹配结果
        bool hasSparseCloud = false;   // 是否有稀疏点云
        bool hasDenseCloud = false;    // 是否有密集点云

        QString featuresPath;          // 特征点路径
        QString matchesPath;           // 匹配文件路径
        QString sparseCloudPath;       // 稀疏点云路径
        QString denseCloudPath;        // 密集点云路径
    };

    explicit CreateDemDialog(ProjectManager *projectManager, QWidget *parent = nullptr);

    void setAvailableImages(const QStringList &images);
    void setDefaultOutput(const QString &outputDir);
    void applySettings(const QJsonObject &settings);
    QJsonObject currentSettings() const;

signals:
    void settingsChanged(const QJsonObject &settings);

    // 自动模式：运行完整流水线
    void requestRunFullPipeline(const QStringList &images,
                               const QString &outputDir,
                               const QJsonObject &pipelineSettings);

    // 手动模式：从指定步骤开始
    void requestRunFromStep(const QString &startStep,
                           const QJsonObject &inputData,
                           const QString &outputDir,
                           const QJsonObject &settings);

private slots:
    void onModeChanged();
    void onBrowseOutput();
    void onBrowseImages();
    void onBrowseCameras();
    void onBrowseFeatures();
    void onBrowseMatches();
    void onBrowseSparseCloud();
    void onBrowseDenseCloud();
    void onDetectExistingData();
    void onRunClicked();
    void onSettingsModified();

private:
    void setupUi();
    void setupAutomaticModeUi(QWidget *container);
    void setupManualModeUi(QWidget *container);
    void updatePipelineStatus();
    PipelineStepStatus detectPipelineStatus(const QStringList &images);
    QString formatStepStatus(bool available, const QString &path);

    ProjectManager *m_projectManager = nullptr;
    QStringList m_availableImages;

    // ── 模式选择 ──
    QRadioButton *m_autoModeRadio = nullptr;
    QRadioButton *m_manualModeRadio = nullptr;
    QStackedWidget *m_modeStack = nullptr;

    // ── 自动模式控件 ──
    QLabel *m_autoInfoLabel = nullptr;
    QListWidget *m_imageList = nullptr;
    QPushButton *m_browseImagesBtn = nullptr;
    QLineEdit *m_camera1Edit = nullptr;
    QLineEdit *m_camera2Edit = nullptr;
    QPushButton *m_browseCam1Btn = nullptr;
    QPushButton *m_browseCam2Btn = nullptr;
    QLabel *m_pipelineStatusLabel = nullptr;
    QPushButton *m_detectDataBtn = nullptr;

    // ── 手动模式控件 ──
    QLabel *m_manualInfoLabel = nullptr;
    QCheckBox *m_useFeaturesCheck = nullptr;
    QLineEdit *m_featuresEdit = nullptr;
    QPushButton *m_browseFeaturesBtn = nullptr;
    QCheckBox *m_useMatchesCheck = nullptr;
    QLineEdit *m_matchesEdit = nullptr;
    QPushButton *m_browseMatchesBtn = nullptr;
    QCheckBox *m_useSparseCheck = nullptr;
    QLineEdit *m_sparseEdit = nullptr;
    QPushButton *m_browseSparseBtn = nullptr;
    QCheckBox *m_useDenseCheck = nullptr;
    QLineEdit *m_denseEdit = nullptr;
    QPushButton *m_browseDenseBtn = nullptr;

    // ── 共用参数 ──
    QLineEdit *m_outputEdit = nullptr;
    QDoubleSpinBox *m_resSpin = nullptr;
    QComboBox *m_typeCombo = nullptr;

    // ── 按钮 ──
    QPushButton *m_runBtn = nullptr;
    QPushButton *m_closeBtn = nullptr;
};
