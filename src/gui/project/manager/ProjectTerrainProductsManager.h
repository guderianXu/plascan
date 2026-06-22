#pragma once

#include <QObject>
#include <QJsonObject>
#include <QStringList>

class QWidget;
class ProjectData;
class ProjectManager;

class ProjectTerrainProductsManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectTerrainProductsManager(ProjectManager *owner,
                                           ProjectData *projectData,
                                           QWidget *parentWidget,
                                           QObject *parent = nullptr);

    void startStereoAndPoint2DemAsync(const QStringList &images,
                                      const QString &outputDir,
                                      int threads,
                                      bool genPointCloud,
                                      double demResolution,
                                      const QString &demType,
                                      const QString &tSrs);

    // 完整流水线（自动模式）：特征提取 → 匹配 → 三角化 → MVS → DEM
    void startFullDemPipelineAsync(const QStringList &images,
                                   const QString &outputDir,
                                   const QJsonObject &pipelineSettings);

    // 从密集点云生成 DEM（手动模式）
    void startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                     const QString &outputDir,
                                     double demResolution,
                                     const QString &demType);

    void startMapProjectAsync(const QStringList &images,
                              const QString &demPath,
                              const QString &outputPath,
                              double resolution);

signals:
    void demPipelineProgressChanged(const QString &stage, int percent);
    void demPipelineFinished(bool success, const QString &message);

private:
    struct DemPipelineContext
    {
        QStringList images;
        QStringList cameraPaths;
        QString outputDir;
        QString featureAlgorithm;
        QString matchAlgorithm;
        double demResolution;
        QString demType;
    };

    bool ensureProjectOpen(const QString &message = QStringLiteral("请先打开项目"),
                           const QString &title = QStringLiteral("提示")) const;
    void runFullDemPipelineInBackground(const DemPipelineContext &ctx);

    ProjectManager *m_owner = nullptr;
    ProjectData *m_projectData = nullptr;
    QWidget *m_parentWidget = nullptr;
};
