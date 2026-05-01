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

class WorkspaceCenterWidget : public QWidget
{
    Q_OBJECT
public:
    explicit WorkspaceCenterWidget(QWidget *parent = nullptr);

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

    QPushButton *m_modelBtn = nullptr;
    QPushButton *m_imageBtn = nullptr;
    QPushButton *m_compareBtn = nullptr;
    QPushButton *m_obsNetBtn = nullptr;
    QStackedWidget *m_stack = nullptr;
    CameraSceneWidget *m_modelView = nullptr;
    CanvasWidget *m_canvas = nullptr;
    DualImageViewer *m_dualImageViewer = nullptr;
    ObservationNetworkView *m_obsNetView = nullptr;
};
