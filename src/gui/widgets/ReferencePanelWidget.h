#pragma once

#include "reference/CameraReferenceTreeModel.h"

#include <QJsonObject>
#include <QWidget>

class QAction;
class QTreeView;

namespace xjw::gui::markers
{
class MarkerWorkspaceController;
}

namespace xjw::gui::reference
{
class MarkerReferenceTreeModel;
class ProjectCameraReferenceRepository;
class ScaleBarReferenceTreeModel;
}

class ReferencePanelWidget final : public QWidget
{
    Q_OBJECT

public:
    explicit ReferencePanelWidget(QWidget *parent = nullptr);
    ~ReferencePanelWidget() override;

    void setCameraReferenceRepository(
        xjw::gui::reference::ProjectCameraReferenceRepository *repository);
    void setMarkerController(xjw::gui::markers::MarkerWorkspaceController *controller);

public slots:
    void loadFromJson(const QJsonObject &metadata);
    void clearProject();

signals:
    void importCameraReferencesRequested();
    void importMarkerReferencesRequested();
    void exportCameraReferencesRequested();
    void cameraReferenceSettingsRequested();
    void imageActivated(const QString &imagePath);
    void markerActivated(const QString &markerId);
    void markerPropertiesRequested(const QString &markerId);

private:
    void buildInterface();
    void refreshAll();
    void refreshCameraReferences();
    void refreshMarkerReferences();
    void applyMode(xjw::gui::reference::ReferenceDisplayMode mode);
    void updateActionAvailability();
    QString selectedCameraUuid() const;
    QString selectedMarkerId() const;

    QJsonObject _projectMetadata;
    xjw::gui::reference::ReferenceDisplayMode _mode =
        xjw::gui::reference::ReferenceDisplayMode::Source;
    xjw::gui::reference::ProjectCameraReferenceRepository *_cameraRepository = nullptr;
    xjw::gui::markers::MarkerWorkspaceController *_markerController = nullptr;
    xjw::gui::reference::CameraReferenceTreeModel *_cameraModel = nullptr;
    xjw::gui::reference::MarkerReferenceTreeModel *_markerModel = nullptr;
    xjw::gui::reference::ScaleBarReferenceTreeModel *_scaleBarModel = nullptr;
    QTreeView *_cameraTree = nullptr;
    QTreeView *_markerTree = nullptr;
    QTreeView *_scaleBarTree = nullptr;
    QAction *_importCameraAction = nullptr;
    QAction *_importMarkerAction = nullptr;
    QAction *_exportCameraAction = nullptr;
    QAction *_toggleCameraAction = nullptr;
    QAction *_editMarkerAction = nullptr;
    QAction *_sourceModeAction = nullptr;
    QAction *_estimatedModeAction = nullptr;
    QAction *_errorModeAction = nullptr;
    QAction *_settingsAction = nullptr;
};
