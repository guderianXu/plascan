#pragma once

#include "project/tasks/ProjectTaskContext.h"

#include <QFuture>
#include <QImage>
#include <QJsonObject>
#include <QMap>
#include <QObject>
#include <QPointer>
#include <QSet>
#include <QString>
#include <QStringList>
#include <QVector>

#include <functional>
#include <optional>

class ProjectUiMessageAdapter;

namespace xjw::gui::project
{
    class ProjectSession;
    class ProjectTaskOrchestrator;
} // namespace xjw::gui::project

using MaskSettingsProvider =
    std::function<std::optional<QJsonObject>(const QStringList& selectedImages, const QString& currentImage)>;

struct StagedMaskArtifact
{
    QString imagePath;
    QString finalPath;
    QString stagingPath;
    QJsonObject record;
};

struct PublishedMaskArtifact
{
    StagedMaskArtifact staged;
    QString backupPath;
    bool hadOriginal = false;
    bool published = false;
};

struct PublishMaskResult
{
    bool committed = false;
    bool tailAllowed = false;
    QStringList filesystemSucceededImages;
    QStringList metadataDeltaImages;
    QString errorMessage;
};

class ProjectMaskWorkflowController final : public QObject
{
    Q_OBJECT

public:
    ~ProjectMaskWorkflowController() override;

signals:
    void progressChanged(const QString& stage, int done, int total);
    void finished(bool success);
    void masksGenerated(const QStringList& imagePaths);
    void interactiveMaskSaved(const QString& imagePath, quint64 revision);
    void interactiveMaskSaveFailed(const QString& imagePath, quint64 revision, const QString& message);
    void projectMetadataUpdated(const QString& projectPath);

private:
    friend class xjw::gui::project::ProjectTaskOrchestrator;

    ProjectMaskWorkflowController(xjw::gui::project::ProjectSession* session,
                                  ProjectUiMessageAdapter* messages,
                                  MaskSettingsProvider settingsProvider,
                                  QObject* parent = nullptr);

    void setActiveImagePath(const QString& imagePath);
    void openDialog(xjw::gui::project::ProjectTaskContext context);
    void openDialogForImages(xjw::gui::project::ProjectTaskContext context, const QStringList& requestedImages);
    void clearMasksForImages(xjw::gui::project::ProjectTaskContext context, const QStringList& requestedImages);
    void saveInteractiveMask(xjw::gui::project::ProjectTaskContext context,
                             const QString& imagePath,
                             const QImage& mask,
                             const QString& method,
                             quint64 revision);
    void cancelActiveTask();
    void waitForActiveTask();
    bool hasRunningTask() const;
    bool hasPendingWork() const noexcept;
    void trackFutureForTesting(QFuture<void> future);
    void setArtifactStagedObserverForTesting(std::function<void(int)> observer);
    void setClearRemovalFailureIndexForTesting(int index);
    void setClearRecoveryCleanupFailureIndexForTesting(int index);
    void setPublicationObserverForTesting(
        std::function<void(const QString&, const StagedMaskArtifact&, const QString&)> observer);
    int ownedArtifactCountForTesting() const;
    int futureCountForTesting() const;

    PublishMaskResult publishStagedArtifacts(xjw::gui::project::ProjectTaskContext context,
                                             const QVector<StagedMaskArtifact>& artifacts,
                                             bool allowUserBatchCancel);
    bool rollbackPublishedArtifacts(xjw::gui::project::ProjectTaskContext context,
                                    QVector<PublishedMaskArtifact>* artifacts,
                                    QStringList* failures,
                                    bool allowUserBatchCancel);
    void showRollbackFailures(xjw::gui::project::ProjectTaskContext context, const QStringList& failures);

    void pruneFinishedFutures();
    void registerFuture(QFuture<void> future);

    void rememberOwnedArtifact(const QString& path);
    void forgetOwnedArtifact(const QString& path);
    void removeOwnedArtifact(const QString& path);
    void cleanupOwnedArtifacts();

    QPointer<xjw::gui::project::ProjectSession> _session;
    ProjectUiMessageAdapter* _messages = nullptr;
    const MaskSettingsProvider _settingsProvider;
    QString _activeImagePath;
    QVector<QFuture<void>> _futures;
    QSet<QString> _ownedArtifacts;
    std::function<void(int)> _artifactStagedObserverForTesting;
    std::function<void(const QString&, const StagedMaskArtifact&, const QString&)> _publicationObserverForTesting;
    int _clearRemovalFailureIndexForTesting = -1;
    int _clearRecoveryCleanupFailureIndexForTesting = -1;
    int _runningWorkers = 0;
};
