#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

class QLabel;
class QPushButton;
class QListWidget;
class QProgressBar;
class QStackedWidget;
class ProjectManager;

// CreateDemDialog — 傻瓜式立体 DEM 生成对话框
// 自动模式：选 2 张影像 → 点运行 → 全自动完成（特征提取→匹配→三角化→MVS CUDA→DEM）
// 手动模式：已有密集点云 → 直接生成 DEM
class CreateDemDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CreateDemDialog(ProjectManager *projectManager, QWidget *parent = nullptr);

    void setAvailableImages(const QStringList &images);
    void setDefaultOutput(const QString &outputDir);

    // 流水线进度更新（由外部调用）
    void onPipelineProgress(const QString &stage, int percent);
    void onPipelineFinished(bool success, const QString &message);

signals:
    void requestRunFullPipeline(const QStringList &images,
                                const QString &outputDir,
                                const QJsonObject &pipelineSettings);
    void requestRunFromDenseCloud(const QString &denseCloudPath,
                                  const QString &outputDir,
                                  double demResolution,
                                  const QString &demType);

private slots:
    void onBrowseImages();
    void onBrowseDenseCloud();
    void onRunClicked();
    void onModeToggled(bool autoMode);

private:
    void setupUi();
    void setRunning(bool running);
    void refreshRunButton();

    ProjectManager *m_projectManager = nullptr;
    QStringList m_availableImages;
    bool m_running = false;

    // 模式切换
    QPushButton *m_autoModeBtn  = nullptr;
    QPushButton *m_manualModeBtn = nullptr;
    QStackedWidget *m_modeStack = nullptr;

    // 自动模式
    QListWidget *m_imageList    = nullptr;
    QLabel *m_camStatusLabel    = nullptr;

    // 手动模式
    class QLineEdit *m_denseEdit = nullptr;

    // 进度区域
    QProgressBar *m_progressBar  = nullptr;
    QLabel       *m_stageLabel   = nullptr;

    // 按钮
    QPushButton *m_runBtn   = nullptr;
    QPushButton *m_closeBtn = nullptr;
};
