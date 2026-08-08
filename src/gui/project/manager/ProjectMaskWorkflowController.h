#pragma once

#include <QObject>

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
    void cancelActiveTask();

signals:
    void progressChanged(const QString &stage, int done, int total);
    void finished(bool success);
    void masksGenerated(const QStringList &imagePaths);
    void projectMetadataUpdated(const QString &projectPath);

private:
    bool matchesSession(const QString &projectPath, const QString &chunkId) const;

    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    QString _activeImagePath;
    xjw::gui::tasks::TaskCancellationSource _cancellation;
    bool _running = false;
};
