#pragma once

#include <QObject>

class ProjectData;
class QWidget;

namespace xjw::camera_reference
{
class CameraReferenceSet;
}

namespace xjw::gui::reference
{

class ProjectCameraReferenceRepository;

class CameraReferenceController final : public QObject
{
    Q_OBJECT

public:
    CameraReferenceController(ProjectData *projectData,
                              ProjectCameraReferenceRepository *repository,
                              QWidget *parentWidget,
                              QObject *parent = nullptr);

public slots:
    void importMetashapeReference();
    void exportReferences();
    void showSettingsSummary();

private:
    QString initialDirectory() const;

    ProjectData *_projectData = nullptr;
    ProjectCameraReferenceRepository *_repository = nullptr;
    QWidget *_parentWidget = nullptr;
};

} // namespace xjw::gui::reference
