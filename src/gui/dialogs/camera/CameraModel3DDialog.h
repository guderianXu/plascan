#pragma once

#include "CameraSceneWidget.h"

#include <QDialog>

class QLabel;
class ProjectManager;
class QWidget;

// Thin project-aware dialog wrapper around the reusable 3D scene widget.
class CameraModel3DDialog : public QDialog
{
    Q_OBJECT

public:
    explicit CameraModel3DDialog(ProjectManager *projectManager,
                                 QWidget *parent = nullptr);

private slots:
    void reloadFromProject();

private:
    QVector<CameraSceneWidget::CameraPose> readCamerasFromMeta() const;

    ProjectManager *_projectManager = nullptr;
    CameraSceneWidget *_scene = nullptr;
    QLabel *_summaryLabel = nullptr;
};
