#pragma once

#include <QWidget>
#include <QJsonObject>

#include "graph/ObservationNetworkBuilder.h"

class QPushButton;
class QStackedWidget;
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
    explicit WorkspaceCenterWidget(QWidget *parent = nullptr);
    ~WorkspaceCenterWidget() override;

    CanvasWidget *canvas() const;
    CameraSceneWidget *modelView() const;

public slots:
    void showModelView();
    void showImageView(const QString &imagePath);
    void showSideBySideImages(const QString &primaryImagePath, const QString &sideImagePath);
    void showModelFile(const QString &modelPath);
    void showPointCloudFile(const QString &pointCloudPath);
    void showObservationNetwork(const xjw::ObservationNetwork &net, const QString &title = QString());
    void setProjectMeta(const QJsonObject &meta);

private:
    void refreshModelFromMeta(const QJsonObject &meta);

    Ui::WorkspaceCenterWidget *_ui = nullptr;
    QPushButton *_modelBtn = nullptr;
    QPushButton *_imageBtn = nullptr;
    QPushButton *_compareBtn = nullptr;
    QPushButton *_obsNetBtn = nullptr;
    QStackedWidget *_stack = nullptr;
    CameraSceneWidget *_modelView = nullptr;
    CanvasWidget *_canvas = nullptr;
    DualImageViewer *_dualImageViewer = nullptr;
    ObservationNetworkView *_obsNetView = nullptr;
};
