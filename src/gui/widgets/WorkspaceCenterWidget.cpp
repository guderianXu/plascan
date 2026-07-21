#include "WorkspaceCenterWidget.h"

#include "ui_WorkspaceCenterWidget.h"

#include "CanvasWidget.h"
#include "CameraModel3DDialog.h"
#include "DualImageViewer.h"
#include "ObservationNetworkView.h"
#include "project/ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"

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
    , _ui(new Ui::WorkspaceCenterWidget)
{
    _ui->setupUi(this);

    _modelBtn = _ui->m_modelBtn;
    _imageBtn = _ui->m_imageBtn;
    _compareBtn = _ui->m_compareBtn;
    _obsNetBtn = _ui->m_obsNetBtn;
    _stack = _ui->m_stack;
    _modelView = _ui->m_modelView;
    _canvas = _ui->m_canvas;
    _dualImageViewer = _ui->m_dualImageViewer;
    _obsNetView = _ui->m_obsNetView;

    connect(_stack, &QStackedWidget::currentChanged, this, [this](int)
    {
        emit viewModeChanged(currentViewMode());
    });

    connect(_modelBtn, &QPushButton::clicked, this, &WorkspaceCenterWidget::showModelView);
    connect(_imageBtn, &QPushButton::clicked, this, [this]()
    {
        _stack->setCurrentWidget(_canvas);
        _modelBtn->setChecked(false);
        _imageBtn->setChecked(true);
        if (_compareBtn) _compareBtn->setChecked(false);
        if (_obsNetBtn) _obsNetBtn->setChecked(false);
    });
    connect(_compareBtn, &QPushButton::clicked, this, [this]()
    {
        if (!_stack || !_dualImageViewer)
        {
            return;
        }
        _stack->setCurrentWidget(_dualImageViewer);
        if (_modelBtn) _modelBtn->setChecked(false);
        if (_imageBtn) _imageBtn->setChecked(false);
        if (_compareBtn) _compareBtn->setChecked(true);
        if (_obsNetBtn) _obsNetBtn->setChecked(false);
    });
    connect(_obsNetBtn, &QPushButton::clicked, this, [this]()
    {
        if (!_stack || !_obsNetView)
        {
            return;
        }
        _stack->setCurrentWidget(_obsNetView);
        if (_modelBtn)
        {
            _modelBtn->setChecked(false);
        }
        if (_imageBtn)
        {
            _imageBtn->setChecked(false);
        }
        if (_compareBtn)
        {
            _compareBtn->setChecked(false);
        }
        if (_obsNetBtn)
        {
            _obsNetBtn->setChecked(true);
        }
    });

    showModelView();
}

WorkspaceCenterWidget::~WorkspaceCenterWidget()
{
    delete _ui;
}

QSize WorkspaceCenterWidget::minimumSizeHint() const
{
    return QSize(240, 160);
}

QSize WorkspaceCenterWidget::sizeHint() const
{
    return QSize(960, 640);
}

CanvasWidget *WorkspaceCenterWidget::canvas() const
{
    return _canvas;
}

CameraSceneWidget *WorkspaceCenterWidget::modelView() const
{
    return _modelView;
}

WorkspaceCenterWidget::ViewMode WorkspaceCenterWidget::currentViewMode() const
{
    if (!_stack)
    {
        return ViewMode::None;
    }
    if (_stack->currentWidget() == _modelView)
    {
        return ViewMode::Model;
    }
    if (_stack->currentWidget() == _canvas)
    {
        return ViewMode::Image;
    }
    if (_stack->currentWidget() == _dualImageViewer)
    {
        return ViewMode::Compare;
    }
    if (_stack->currentWidget() == _obsNetView)
    {
        return ViewMode::ObservationNetwork;
    }
    return ViewMode::None;
}

void WorkspaceCenterWidget::showModelView()
{
    if (!_stack || !_modelView)
    {
        return;
    }
    _stack->setCurrentWidget(_modelView);
    if (_modelBtn)
    {
        _modelBtn->setChecked(true);
    }
    if (_imageBtn)
    {
        _imageBtn->setChecked(false);
    }
    if (_compareBtn)
    {
        _compareBtn->setChecked(false);
    }
    if (_obsNetBtn)
    {
        _obsNetBtn->setChecked(false);
    }
}

void WorkspaceCenterWidget::showImageView(const QString &imagePath)
{
    if (!_stack || !_canvas)
    {
        return;
    }

    // 先切换到 canvas（确保 fitInView 使用正确的视口尺寸），再加载影像
    _stack->setCurrentWidget(_canvas);
    _modelBtn->setChecked(false);
    _imageBtn->setChecked(true);
    if (_compareBtn) _compareBtn->setChecked(false);
    if (_obsNetBtn) _obsNetBtn->setChecked(false);

    if (!imagePath.trimmed().isEmpty())
    {
        setViewButtonText(_imageBtn, displayNameForPath(imagePath), imagePath);
        _imageBtn->setVisible(true);
        _canvas->showImage(imagePath);
    }
}

void WorkspaceCenterWidget::showSideBySideImages(const QString &primaryImagePath, const QString &sideImagePath)
{
    if (!_stack || !_dualImageViewer)
    {
        return;
    }
    if (primaryImagePath.trimmed().isEmpty() || sideImagePath.trimmed().isEmpty())
    {
        return;
    }

    _dualImageViewer->loadMatchPair(primaryImagePath,
                                     sideImagePath,
                                     QVector<QPointF>(),
                                     QVector<QPointF>());
    _dualImageViewer->fitBothViews();
    _stack->setCurrentWidget(_dualImageViewer);

    const QString primaryName = displayNameForPath(primaryImagePath);
    const QString sideName = displayNameForPath(sideImagePath);
    if (_compareBtn)
    {
        setViewButtonText(_compareBtn,
                          QStringLiteral("对比: %1 | %2").arg(primaryName, sideName),
                          QStringLiteral("%1\n%2").arg(primaryImagePath, sideImagePath));
        _compareBtn->setVisible(true);
        _compareBtn->setChecked(true);
    }
    if (_modelBtn) _modelBtn->setChecked(false);
    if (_imageBtn) _imageBtn->setChecked(false);
    if (_obsNetBtn) _obsNetBtn->setChecked(false);
}

void WorkspaceCenterWidget::showModelFile(const QString &modelPath)
{
    if (!_modelView)
    {
        return;
    }

    const QString ext = QFileInfo(modelPath).suffix().toLower();
    if (ext == QLatin1String("obj"))
    {
        _modelView->loadModelFromObj(modelPath);
    }
    else if (ext == QLatin1String("ply"))
    {
        _modelView->loadModelFromPly(modelPath);
    }
    else
    {
        _modelView->loadPointCloudFromXyz(modelPath);
    }

    if (_modelBtn)
    {
        setViewButtonText(_modelBtn, displayNameForPath(modelPath), modelPath);
    }
    showModelView();
}

void WorkspaceCenterWidget::showPointCloudFile(const QString &pointCloudPath)
{
    if (!_modelView)
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
        _modelView->loadPointCloudFromXyz(pointCloudPath);
        if (_modelBtn)
        {
            setViewButtonText(_modelBtn, displayNameForPath(pointCloudPath), pointCloudPath);
        }
        showModelView();
    }
}

void WorkspaceCenterWidget::highlightCameraForImage(const QString &imagePath)
{
    if (_modelView)
    {
        _modelView->setHighlightedCameraPath(imagePath);
    }
}

void WorkspaceCenterWidget::clearHighlightedCamera()
{
    if (_modelView)
    {
        _modelView->clearHighlightedCamera();
    }
}

void WorkspaceCenterWidget::showObservationNetwork(const xjw::ObservationNetwork &net, const QString &title)
{
    if (!_stack || !_obsNetView || !_obsNetBtn)
    {
        return;
    }

    const QString buttonText = title.trimmed().isEmpty() ? tr("观测网络") : title.trimmed();
    setViewButtonText(_obsNetBtn, buttonText, buttonText);
    _obsNetBtn->setVisible(true);

    _obsNetView->setNetwork(net);
    _stack->setCurrentWidget(_obsNetView);

    if (_modelBtn)
    {
        _modelBtn->setChecked(false);
    }
    if (_imageBtn)
    {
        _imageBtn->setChecked(false);
    }
    if (_compareBtn)
    {
        _compareBtn->setChecked(false);
    }
    _obsNetBtn->setChecked(true);
}

void WorkspaceCenterWidget::setProjectMeta(const QJsonObject &meta)
{
    refreshModelFromMeta(meta);
}

void WorkspaceCenterWidget::refreshModelFromMeta(const QJsonObject &meta)
{
    const QJsonArray images = xjw::common::project::projectImageEntries(meta);
    QVector<CameraSceneWidget::CameraPose> poses;
    poses.reserve(images.size());

    for (const QJsonValue &v : images)
    {
        const QJsonObject imageObject = v.toObject();
        xjw::Camera camera;
        if (!xjw::common::project::imageCameraFromEntry(imageObject, &camera))
        {
            continue;
        }

        const std::array<double, 3> cameraCenter = camera.cameraCenter();
        const std::array<double, 9> cameraToWorldRotation = camera.cameraToWorldRotation();
        const xjw::Camera::Intrinsics intrinsics = camera.intrinsics();
        const QJsonObject camera_object = imageObject.value(QStringLiteral("camera")).toObject();

        CameraSceneWidget::CameraPose pose;
        pose.imagePath = imageObject.value(QStringLiteral("path")).toString();
        if (pose.imagePath.isEmpty())
        {
            pose.imagePath = imageObject.value(QStringLiteral("image_path")).toString();
        }
        const QString labelPath = pose.imagePath.isEmpty()
            ? imageObject.value(QStringLiteral("path")).toString()
            : pose.imagePath;
        pose.name = QFileInfo(labelPath).fileName();
        pose.center = QVector3D(
            float(cameraCenter[0]),
            float(cameraCenter[1]),
            float(cameraCenter[2]));
        pose.focalX = static_cast<float>(intrinsics.focalX);
        pose.focalY = static_cast<float>(intrinsics.focalY);
        pose.principalX = static_cast<float>(intrinsics.principalX);
        pose.principalY = static_cast<float>(intrinsics.principalY);
        pose.imageWidth = camera_object.value(QStringLiteral("image_width")).toInt();
        pose.imageHeight = camera_object.value(QStringLiteral("image_height")).toInt();
        pose.uAxisSign = intrinsics.uAxisSign;
        pose.vAxisSign = intrinsics.vAxisSign;
        pose.depthAxisFlipped = camera.depthAxisFlipped();

        QMatrix3x3 rot;
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                rot(row, col) = float(cameraToWorldRotation[row * 3 + col]);
            }
        }
        pose.rotation = rot;
        poses.push_back(pose);
    }

    if (_modelView)
    {
        _modelView->setCameraPoses(poses);
    }
}
