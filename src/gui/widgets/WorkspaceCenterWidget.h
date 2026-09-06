#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QSize>
#include <QWidget>

#include "graph/ObservationNetworkBuilder.h"

class QStackedWidget;
class QTabBar;
class CanvasWidget;
class CameraSceneWidget;
class ObservationNetworkView;
class DualImageViewer;

namespace Ui {
class WorkspaceCenterWidget;
}

class WorkspaceCenterWidget : public QWidget
{
    Q_OBJECT
public:
    enum class ViewMode
    {
        None,
        Model,
        Image,
        Compare,
        ObservationNetwork
    };
    Q_ENUM(ViewMode)

    explicit WorkspaceCenterWidget(QWidget *parent = nullptr);
    ~WorkspaceCenterWidget() override;

    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;

    CanvasWidget *canvas() const;
    CameraSceneWidget *modelView() const;
    ViewMode currentViewMode() const;

signals:
    void viewModeChanged(ViewMode mode);

public slots:
    void showModelView();
    void showImageView(const QString &imagePath);
    void showSideBySideImages(const QString &primaryImagePath, const QString &sideImagePath);
    void showModelFile(const QString &modelPath);
    void showPointCloudFile(const QString &pointCloudPath);
    void showTiePointCloudFile(const QString &pointCloudPath,
                               const QString &sidecarPath = QString());
    void showObservationNetwork(const xjw::ObservationNetwork &net, const QString &title = QString());
    void setProjectMeta(const QJsonObject &meta);
    void highlightCameraForImage(const QString &imagePath);
    void clearHighlightedCamera();
    void resetActiveView();
    void clearProjectView();

private:
    void activateView(int index);
    void setDocumentTab(int index, const QString& text, const QString& tooltip, bool visible = true);
    void refreshModelFromMeta(const QJsonObject &meta);

    Ui::WorkspaceCenterWidget *_ui = nullptr;
    QTabBar* _documentTabs = nullptr;
    QStackedWidget *_stack = nullptr;
    CameraSceneWidget *_modelView = nullptr;
    CanvasWidget *_canvas = nullptr;
    DualImageViewer *_dualImageViewer = nullptr;
    ObservationNetworkView *_obsNetView = nullptr;
    quint64 _cameraPoseGeneration = 0;
    QJsonArray _cameraPoseMetadata;
};
