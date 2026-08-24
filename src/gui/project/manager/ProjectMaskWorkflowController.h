#pragma once

#include <QObject>
#include <QImage>

#include "GuiTaskRunner.h"

class ProjectData;
class QWidget;

class ProjectMaskWorkflowController final : public QObject
{
    Q_OBJECT

public:
    explicit ProjectMaskWorkflowController(ProjectData *projectData,
                                           QWidget *parentWidget,
                                           QObject *parent = nullptr);

    void setActiveImagePath(const QString &imagePath);
    void openDialog();
    void openDialogForImages(const QStringList &requestedImages);
    void clearMasksForImages(const QStringList &requestedImages);
    void saveInteractiveMask(const QString &imagePath,
                             const QImage &mask,
                             const QString &method,
                             quint64 revision);
    void cancelActiveTask();

signals:
    void progressChanged(const QString &stage, int done, int total);
    void finished(bool success);
    void masksGenerated(const QStringList &imagePaths);
    void interactiveMaskSaved(const QString &imagePath, quint64 revision);
    void interactiveMaskSaveFailed(const QString &imagePath,
                                   quint64 revision,
                                   const QString &message);
    void projectMetadataUpdated(const QString &projectPath);

private:
    bool matchesSession(const QString &projectPath, const QString &chunkId) const;
    void startNextInteractiveSave();

    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    QString _activeImagePath;
    xjw::gui::tasks::TaskCancellationSource _cancellation;
    bool _running = false;
    QString _pendingInteractiveImagePath;
    QImage _pendingInteractiveMask;
    QString _pendingInteractiveMethod;
    quint64 _pendingInteractiveRevision{};
    bool _hasPendingInteractiveSave{};
    bool _interactiveSaveRunning{};
};
