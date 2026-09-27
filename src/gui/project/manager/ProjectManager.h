#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QString>
#include <QStringList>

#include <memory>

class ProjectData;
class ProjectLifecycleService;
class QWidget;

namespace xjw::gui::project
{
    class ProjectResourceCleanupCoordinator;
    class ProjectResourceService;
    class ProjectServiceContainer;
    class ProjectSession;
    class ProjectTaskOrchestrator;
} // namespace xjw::gui::project

class ProjectManager final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectManager(ProjectData* projectData, QWidget* parent = nullptr);
    ~ProjectManager() override;

    xjw::gui::project::ProjectServiceContainer& services() const;
    xjw::gui::project::ProjectSession& session() const;
    xjw::gui::project::ProjectTaskOrchestrator& tasks() const;
    xjw::gui::project::ProjectResourceService& resources() const;
    xjw::gui::project::ProjectResourceCleanupCoordinator& cleanup() const;
    ProjectLifecycleService& lifecycle() const;

signals:
    void projectCreated(const QString& projectPath);
    void projectOpened(const QString& projectPath);
    void projectSaved(const QString& projectPath);
    void projectClosed();
    void projectSessionChanged();
    void projectOpenStarted(const QString& projectPath);
    void projectOpenProgressChanged(const QString& message, int percent);
    void projectOpenFinished(bool success, const QString& message);
    void chunkListChanged(const QJsonArray& chunks, const QString& activeChunkId);
    void projectMetadataChanged(const QJsonObject& metadata);
    void projectMetadataUpdated(const QString& projectPath);
    void metadataDirtyChanged(bool dirty);
    void imageMatchResultAppended(const QString& imagePath);
    void maskGenerationProgressChanged(const QString& stage, int done, int total);
    void maskGenerationFinished(bool success);
    void imageImportProgressChanged(const QString& stage, int done, int total);
    void imageImportFinished(bool success, const QString& message);
    void surveyControlChanged();
    void masksGenerated(const QStringList& imagePaths);
    void interactiveMaskSaved(const QString& imagePath, quint64 revision);
    void interactiveMaskSaveFailed(const QString& imagePath, quint64 revision, const QString& message);
    void
    matchPairReady(const QString& firstImage, const QString& secondImage, const QString& matchFilePath, int matchCount);
    void bundleAdjustPreviewReady(const QJsonObject& preview);
    void saveStarted();
    void saveFinished(bool success);
    void meshProgressChanged(const QString& stage, int percent);
    void meshProgressFinished(bool success);
    void pointCloudProgressChanged(const QString& stage, int percent);
    void pointCloudProgressFinished(bool success);
    void pointCloudResultReady(const QString& path, int pointCount);
    void atProgressChanged(const QString& stage, int percent);
    void atComputeDeviceChanged(const QString& displayName);
    void atProgressFinished(bool success);
    void tiePointResultReady(const QString& sparseCloudPath, const QString& sidecarPath);
    void backgroundTaskProgressChanged(const QString& taskId, int value, int maximum);
    void backgroundTaskFinished(const QString& taskId);
    void demPipelineProgressChanged(const QString& stage, int percent);
    void demPipelineFinished(bool success, const QString& message);
    void orthoPipelineStarted();
    void orthoPipelineProgressChanged(const QString& stage, int percent);
    void orthoPipelineFinished(bool success, const QString& message, const QJsonObject& result);

private:
    std::unique_ptr<xjw::gui::project::ProjectServiceContainer> _serviceContainer;
};
