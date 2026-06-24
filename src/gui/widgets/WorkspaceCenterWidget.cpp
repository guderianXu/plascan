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

CanvasWidget *WorkspaceCenterWidget::canvas() const
{
    return _canvas;
}

CameraSceneWidget *WorkspaceCenterWidget::modelView() const
{
    return _modelView;
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
        pose.rotation = rot;
        poses.push_back(pose);
    }

    if (_modelView)
    {
        _modelView->setCameraPoses(poses);
    }
}
