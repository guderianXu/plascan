#include "WorkspaceCenterWidget.h"

#include "ui_WorkspaceCenterWidget.h"

#include "CanvasWidget.h"
#include "CameraModel3DDialog.h"
#include "DualImageViewer.h"
#include "ObservationNetworkView.h"
#include "ProjectSupportUtils.h"

#include <QPushButton>
#include <QStackedWidget>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>

namespace {

QString displayNameForPath(const QString &path)
{
    const QFileInfo info(path);
    const QString base = info.completeBaseName();
    return base.isEmpty() ? info.fileName() : base;
}

void setViewButtonText(QPushButton *button, const QString &text, const QString &tooltip)
{
    if (!button)
    {
        return;
    }

    const QString cleanText = text.trimmed().isEmpty() ? QStringLiteral("-") : text.trimmed();
    button->setText(button->fontMetrics().elidedText(cleanText, Qt::ElideMiddle, 190));
    button->setToolTip(tooltip.trimmed().isEmpty() ? cleanText : tooltip);
}

} // namespace

WorkspaceCenterWidget::WorkspaceCenterWidget(QWidget *parent)
    : QWidget(parent)
    , m_ui(new Ui::WorkspaceCenterWidget)
{
    m_ui->setupUi(this);

    m_modelBtn = m_ui->m_modelBtn;
    m_imageBtn = m_ui->m_imageBtn;
    m_compareBtn = m_ui->m_compareBtn;
    m_obsNetBtn = m_ui->m_obsNetBtn;
    m_stack = m_ui->m_stack;
    m_modelView = m_ui->m_modelView;
    m_canvas = m_ui->m_canvas;
    m_dualImageViewer = m_ui->m_dualImageViewer;
    m_obsNetView = m_ui->m_obsNetView;

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

WorkspaceCenterWidget::~WorkspaceCenterWidget()
{
    delete m_ui;
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
        setViewButtonText(m_imageBtn, displayNameForPath(imagePath), imagePath);
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

    const QString primaryName = displayNameForPath(primaryImagePath);
    const QString sideName = displayNameForPath(sideImagePath);
    if (m_compareBtn)
    {
        setViewButtonText(m_compareBtn,
                          QStringLiteral("对比: %1 | %2").arg(primaryName, sideName),
                          QStringLiteral("%1\n%2").arg(primaryImagePath, sideImagePath));
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
        setViewButtonText(m_modelBtn, displayNameForPath(modelPath), modelPath);
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
            setViewButtonText(m_modelBtn, displayNameForPath(pointCloudPath), pointCloudPath);
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
    setViewButtonText(m_obsNetBtn, buttonText, buttonText);
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
