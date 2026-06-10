#include "WorkspaceCenterWidget.h"

#include "CanvasWidget.h"
#include "CameraModel3DDialog.h"
#include "DualImageViewer.h"
#include "ObservationNetworkView.h"
#include "ProjectSupportUtils.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QStackedWidget>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>

WorkspaceCenterWidget::WorkspaceCenterWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    auto *topBar = new QHBoxLayout();
    topBar->setContentsMargins(6, 6, 6, 4);
    topBar->setSpacing(6);

    m_modelBtn = new QPushButton(tr("模型"), this);
    m_modelBtn->setCheckable(true);

    m_imageBtn = new QPushButton(tr("-"), this);
    m_imageBtn->setCheckable(true);
    m_imageBtn->setVisible(false);

    m_compareBtn = new QPushButton(tr("对比"), this);
    m_compareBtn->setCheckable(true);
    m_compareBtn->setVisible(false);

    m_obsNetBtn = new QPushButton(tr("观测网络"), this);
    m_obsNetBtn->setCheckable(true);
    m_obsNetBtn->setVisible(false);

    topBar->addWidget(m_modelBtn);
    topBar->addWidget(m_imageBtn);
    topBar->addWidget(m_compareBtn);
    topBar->addWidget(m_obsNetBtn);
    topBar->addStretch(1);

    m_stack = new QStackedWidget(this);
    m_modelView = new CameraSceneWidget(this);
    m_canvas = new CanvasWidget(this);
    m_dualImageViewer = new DualImageViewer(this);
    m_obsNetView = new ObservationNetworkView(this);

    m_stack->addWidget(m_modelView);
    m_stack->addWidget(m_canvas);
    m_stack->addWidget(m_dualImageViewer);
    m_stack->addWidget(m_obsNetView);

    mainLayout->addLayout(topBar);
    mainLayout->addWidget(m_stack, 1);

    connect(m_modelBtn, &QPushButton::clicked, this, &WorkspaceCenterWidget::showModelView);
    connect(m_imageBtn, &QPushButton::clicked, this, [this]()
    {
        m_stack->setCurrentWidget(m_canvas);
        m_modelBtn->setChecked(false);
        m_imageBtn->setChecked(true);
        if (m_compareBtn) m_compareBtn->setChecked(false);
        if (m_obsNetBtn) m_obsNetBtn->setChecked(false);
    });
    connect(m_compareBtn, &QPushButton::clicked, this, [this]()
    {
        if (!m_stack || !m_dualImageViewer)
        {
            return;
        }
        m_stack->setCurrentWidget(m_dualImageViewer);
        if (m_modelBtn) m_modelBtn->setChecked(false);
        if (m_imageBtn) m_imageBtn->setChecked(false);
        if (m_compareBtn) m_compareBtn->setChecked(true);
        if (m_obsNetBtn) m_obsNetBtn->setChecked(false);
    });
    connect(m_obsNetBtn, &QPushButton::clicked, this, [this]()
    {
        if (!m_stack || !m_obsNetView)
        {
            return;
        }
        m_stack->setCurrentWidget(m_obsNetView);
        if (m_modelBtn)
        {
            m_modelBtn->setChecked(false);
        }
        if (m_imageBtn)
        {
            m_imageBtn->setChecked(false);
        }
        if (m_compareBtn)
        {
            m_compareBtn->setChecked(false);
        }
        if (m_obsNetBtn)
        {
            m_obsNetBtn->setChecked(true);
        }
    });

    showModelView();
}

CanvasWidget *WorkspaceCenterWidget::canvas() const
{
    return m_canvas;
}

CameraSceneWidget *WorkspaceCenterWidget::modelView() const
{
    return m_modelView;
}

void WorkspaceCenterWidget::showModelView()
{
    if (!m_stack || !m_modelView)
    {
        return;
    }
    m_stack->setCurrentWidget(m_modelView);
    if (m_modelBtn)
    {
        m_modelBtn->setChecked(true);
    }
    if (m_imageBtn)
    {
        m_imageBtn->setChecked(false);
    }
    if (m_compareBtn)
    {
        m_compareBtn->setChecked(false);
    }
    if (m_obsNetBtn)
    {
        m_obsNetBtn->setChecked(false);
    }
}

void WorkspaceCenterWidget::showImageView(const QString &imagePath)
{
    if (!m_stack || !m_canvas)
    {
        return;
    }

    // 先切换到 canvas（确保 fitInView 使用正确的视口尺寸），再加载影像
    m_stack->setCurrentWidget(m_canvas);
    m_modelBtn->setChecked(false);
    m_imageBtn->setChecked(true);
    if (m_compareBtn) m_compareBtn->setChecked(false);
    if (m_obsNetBtn) m_obsNetBtn->setChecked(false);

    if (!imagePath.trimmed().isEmpty())
    {
        const QString base = QFileInfo(imagePath).completeBaseName();
        m_imageBtn->setText(base.isEmpty() ? QFileInfo(imagePath).fileName() : base);
        m_imageBtn->setVisible(true);
        m_canvas->showImage(imagePath);
    }
}

void WorkspaceCenterWidget::showSideBySideImages(const QString &primaryImagePath, const QString &sideImagePath)
{
    if (!m_stack || !m_dualImageViewer)
    {
        return;
    }
    if (primaryImagePath.trimmed().isEmpty() || sideImagePath.trimmed().isEmpty())
    {
        return;
    }

    m_dualImageViewer->loadMatchPair(primaryImagePath,
                                     sideImagePath,
                                     QVector<QPointF>(),
                                     QVector<QPointF>());
    m_dualImageViewer->fitBothViews();
    m_stack->setCurrentWidget(m_dualImageViewer);

    const QString primaryName = QFileInfo(primaryImagePath).completeBaseName();
    const QString sideName = QFileInfo(sideImagePath).completeBaseName();
    if (m_compareBtn)
    {
        m_compareBtn->setText(QStringLiteral("对比: %1 | %2")
                                  .arg(primaryName.isEmpty() ? QFileInfo(primaryImagePath).fileName() : primaryName)
                                  .arg(sideName.isEmpty() ? QFileInfo(sideImagePath).fileName() : sideName));
        m_compareBtn->setVisible(true);
        m_compareBtn->setChecked(true);
    }
    if (m_modelBtn) m_modelBtn->setChecked(false);
    if (m_imageBtn) m_imageBtn->setChecked(false);
    if (m_obsNetBtn) m_obsNetBtn->setChecked(false);
}

void WorkspaceCenterWidget::showModelFile(const QString &modelPath)
{
    if (!m_modelView)
    {
        return;
    }

    const QString ext = QFileInfo(modelPath).suffix().toLower();
    if (ext == QLatin1String("obj"))
    {
        m_modelView->loadModelFromObj(modelPath);
    }
    else if (ext == QLatin1String("ply"))
    {
        m_modelView->loadModelFromPly(modelPath);
    }
    else
    {
        m_modelView->loadPointCloudFromXyz(modelPath);
    }

    if (m_modelBtn)
    {
        const QString base = QFileInfo(modelPath).completeBaseName();
        m_modelBtn->setText(base.isEmpty() ? QFileInfo(modelPath).fileName() : base);
    }
    showModelView();
}

void WorkspaceCenterWidget::showPointCloudFile(const QString &pointCloudPath)
{
    if (!m_modelView)
    {
        return;
    }
    const QString ext = QFileInfo(pointCloudPath).suffix().toLower();
    if (ext == QLatin1String("ply") || ext == QLatin1String("obj"))
    {
        showModelFile(pointCloudPath);
    }
    else
    {
        m_modelView->loadPointCloudFromXyz(pointCloudPath);
        if (m_modelBtn)
        {
            const QString base = QFileInfo(pointCloudPath).completeBaseName();
            m_modelBtn->setText(base.isEmpty() ? QFileInfo(pointCloudPath).fileName() : base);
        }
        showModelView();
    }
}

void WorkspaceCenterWidget::showObservationNetwork(const xjw::ObservationNetwork &net, const QString &title)
{
    if (!m_stack || !m_obsNetView || !m_obsNetBtn)
    {
        return;
    }

    const QString buttonText = title.trimmed().isEmpty() ? tr("观测网络") : title.trimmed();
    m_obsNetBtn->setText(buttonText);
    m_obsNetBtn->setVisible(true);

    m_obsNetView->setNetwork(net);
    m_stack->setCurrentWidget(m_obsNetView);

    if (m_modelBtn)
    {
        m_modelBtn->setChecked(false);
    }
    if (m_imageBtn)
    {
        m_imageBtn->setChecked(false);
    }
    if (m_compareBtn)
    {
        m_compareBtn->setChecked(false);
    }
    m_obsNetBtn->setChecked(true);
}

void WorkspaceCenterWidget::setProjectMeta(const QJsonObject &meta)
{
    refreshModelFromMeta(meta);
}

void WorkspaceCenterWidget::refreshModelFromMeta(const QJsonObject &meta)
{
    const QJsonArray images = xjw::gui::project::projectImageEntries(meta);
    QVector<CameraSceneWidget::CameraPose> poses;
    poses.reserve(images.size());

    for (const QJsonValue &v : images)
    {
        const QJsonObject imageObject = v.toObject();
        xjw::Camera camera;
        if (!xjw::gui::project::imageCameraFromEntry(imageObject, &camera))
        {
            continue;
        }

        const std::array<double, 3> cameraCenter = camera.cameraCenter();
        const std::array<double, 9> cameraToWorldRotation = camera.cameraToWorldRotation();

        CameraSceneWidget::CameraPose pose;
        pose.name = QFileInfo(imageObject.value(QStringLiteral("path")).toString()).fileName();
        pose.center = QVector3D(
            float(cameraCenter[0]),
            float(cameraCenter[1]),
            float(cameraCenter[2]));

        QMatrix3x3 rot;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                rot(row, col) = float(cameraToWorldRotation[row * 3 + col]);
            }
        }
        pose.rotation = rot.transposed();
        poses.push_back(pose);
    }

    if (m_modelView)
    {
        m_modelView->setCameraPoses(poses);
    }
}
