#include "WorkspaceCenterWidget.h"

#include "ui_WorkspaceCenterWidget.h"

#include "CanvasWidget.h"
#include "CameraSceneWidget.h"
#include "DualImageViewer.h"
#include "ObservationNetworkView.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"

#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonObject>
#include <QPointer>
#include <QStackedWidget>
#include <QTabBar>
#include <QtConcurrent/QtConcurrentRun>

namespace {

    constexpr int ModelTabIndex = 0;
    constexpr int ImageTabIndex = 1;
    constexpr int CompareTabIndex = 2;
    constexpr int ObservationNetworkTabIndex = 3;

    QString displayNameForPath(const QString& path)
    {
        const QFileInfo info(path);
        const QString base = info.completeBaseName();
        return base.isEmpty() ? info.fileName() : base;
}

QVector<CameraSceneWidget::CameraPose> cameraPosesFromImages(const QJsonArray &images)
{
    QVector<CameraSceneWidget::CameraPose> poses;
    poses.reserve(images.size());

    for (const QJsonValue &value : images)
    {
        const QJsonObject imageObject = value.toObject();
        xjw::FramePinholeCamera camera;
        if (!xjw::common::project::imageCameraFromEntry(imageObject, &camera))
        {
            continue;
        }

        const std::array<double, 3> cameraCenter = camera.cameraCenter();
        const std::array<double, 9> cameraToWorldRotation = camera.cameraToWorldRotation();
        const xjw::FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
        const QJsonObject cameraObject = imageObject.value(QStringLiteral("camera")).toObject();

        CameraSceneWidget::CameraPose pose;
        pose.imagePath = imageObject.value(QStringLiteral("path")).toString();
        if (pose.imagePath.isEmpty())
        {
            pose.imagePath = imageObject.value(QStringLiteral("image_path")).toString();
        }
        pose.name = QFileInfo(pose.imagePath).fileName();
        pose.center = QVector3D(
            float(cameraCenter[0]),
            float(cameraCenter[1]),
            float(cameraCenter[2]));
        pose.focalX = static_cast<float>(intrinsics.focalX);
        pose.focalY = static_cast<float>(intrinsics.focalY);
        pose.principalX = static_cast<float>(intrinsics.principalX);
        pose.principalY = static_cast<float>(intrinsics.principalY);
        pose.imageWidth = cameraObject.value(QStringLiteral("image_width")).toInt();
        pose.imageHeight = cameraObject.value(QStringLiteral("image_height")).toInt();
        pose.uAxisSign = intrinsics.uAxisSign;
        pose.vAxisSign = intrinsics.vAxisSign;
        pose.depthAxisFlipped = camera.depthAxisFlipped();

        QMatrix3x3 rotation;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                rotation(row, column) = float(cameraToWorldRotation[row * 3 + column]);
            }
        }
        pose.rotation = rotation;
        poses.push_back(pose);
    }

    return poses;
}

QJsonArray cameraPoseMetadataFromImages(const QJsonArray &images)
{
    QJsonArray camera_metadata;
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        const QJsonObject camera = image.value(QStringLiteral("camera")).toObject();
        if (camera.isEmpty())
        {
            continue;
        }

        QJsonObject camera_entry;
        camera_entry[QStringLiteral("path")] = image.value(QStringLiteral("path"));
        camera_entry[QStringLiteral("image_path")] = image.value(QStringLiteral("image_path"));
        camera_entry[QStringLiteral("camera")] = camera;
        camera_metadata.append(camera_entry);
    }
    return camera_metadata;
}

} // namespace

WorkspaceCenterWidget::WorkspaceCenterWidget(QWidget *parent)
    : QWidget(parent)
    , _ui(new Ui::WorkspaceCenterWidget)
{
    _ui->setupUi(this);

    _documentTabs = _ui->m_documentTabs;
    _stack = _ui->m_stack;
    _modelView = _ui->m_modelView;
    _canvas = _ui->m_canvas;
    _dualImageViewer = _ui->m_dualImageViewer;
    _obsNetView = _ui->m_obsNetView;

    _documentTabs->addTab(tr("模型"));
    _documentTabs->addTab(tr("影像"));
    _documentTabs->addTab(tr("对比"));
    _documentTabs->addTab(tr("观测网络"));
    _documentTabs->setTabVisible(ImageTabIndex, false);
    _documentTabs->setTabVisible(CompareTabIndex, false);
    _documentTabs->setTabVisible(ObservationNetworkTabIndex, false);
    _documentTabs->setTabButton(ModelTabIndex, QTabBar::RightSide, nullptr);

    connect(_documentTabs,
            &QTabBar::currentChanged,
            this,
            [this](int index)
            {
                if (_stack && index >= 0 && index < _stack->count())
                {
                    _stack->setCurrentIndex(index);
                }
            });
    connect(_documentTabs,
            &QTabBar::tabCloseRequested,
            this,
            [this](int index)
            {
                if (!_documentTabs || index <= ModelTabIndex || index >= _documentTabs->count())
                {
                    return;
                }
                _documentTabs->setTabVisible(index, false);
                showModelView();
            });
    connect(_stack,
            &QStackedWidget::currentChanged,
            this,
            [this](int index)
            {
                if (_documentTabs && index >= 0 && index < _documentTabs->count() && _documentTabs->isTabVisible(index))
                {
                    _documentTabs->setCurrentIndex(index);
                }
                emit viewModeChanged(currentViewMode());
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

void WorkspaceCenterWidget::activateView(int index)
{
    if (!_stack || !_documentTabs || index < 0 || index >= _stack->count())
    {
        return;
    }
    _documentTabs->setTabVisible(index, true);
    _documentTabs->setCurrentIndex(index);
    _stack->setCurrentIndex(index);
}

void WorkspaceCenterWidget::setDocumentTab(int index, const QString& text, const QString& tooltip, bool visible)
{
    if (!_documentTabs || index < 0 || index >= _documentTabs->count())
    {
        return;
    }
    const QString cleanText = text.trimmed().isEmpty() ? tr("未命名") : text.trimmed();
    _documentTabs->setTabText(index, cleanText);
    _documentTabs->setTabToolTip(index, tooltip.trimmed().isEmpty() ? cleanText : tooltip);
    _documentTabs->setTabVisible(index, visible);
}

void WorkspaceCenterWidget::showModelView()
{
    if (!_stack || !_modelView)
    {
        return;
    }
    activateView(ModelTabIndex);
}

void WorkspaceCenterWidget::showImageView(const QString &imagePath)
{
    if (!_stack || !_canvas)
    {
        return;
    }

    // 先切换到 canvas（确保 fitInView 使用正确的视口尺寸），再加载影像
    activateView(ImageTabIndex);

    if (!imagePath.trimmed().isEmpty())
    {
        setDocumentTab(ImageTabIndex, displayNameForPath(imagePath), imagePath);
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
    activateView(CompareTabIndex);

    const QString primaryName = displayNameForPath(primaryImagePath);
    const QString sideName = displayNameForPath(sideImagePath);
    setDocumentTab(CompareTabIndex,
                   tr("对比：%1 | %2").arg(primaryName, sideName),
                   QStringLiteral("%1\n%2").arg(primaryImagePath, sideImagePath));
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

    setDocumentTab(ModelTabIndex, displayNameForPath(modelPath), modelPath);
    showModelView();
}

void WorkspaceCenterWidget::showPointCloudFile(const QString &pointCloudPath)
{
    if (!_modelView)
    {
        return;
    }
    const QString ext = QFileInfo(pointCloudPath).suffix().toLower();
    if (ext == QLatin1String("ply"))
    {
        _modelView->loadPointCloudFromPly(pointCloudPath);
    }
    else if (ext == QLatin1String("obj"))
    {
        _modelView->loadPointCloudFromObj(pointCloudPath);
    }
    else
    {
        _modelView->loadPointCloudFromXyz(pointCloudPath);
    }

    setDocumentTab(ModelTabIndex, displayNameForPath(pointCloudPath), pointCloudPath);
    showModelView();
}

void WorkspaceCenterWidget::showTiePointCloudFile(const QString &pointCloudPath,
                                                  const QString &sidecarPath)
{
    if (!_modelView)
    {
        return;
    }

    _modelView->loadTiePointCloudFromFile(pointCloudPath, sidecarPath);
    setDocumentTab(ModelTabIndex, displayNameForPath(pointCloudPath), pointCloudPath);
    showModelView();
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

void WorkspaceCenterWidget::resetActiveView()
{
    switch (currentViewMode())
    {
    case ViewMode::Image:
        if (_canvas)
        {
            _canvas->resetView();
        }
        break;
    case ViewMode::Model:
        if (_modelView)
        {
            _modelView->resetView();
        }
        break;
    case ViewMode::Compare:
        if (_dualImageViewer)
        {
            _dualImageViewer->resetBothViews();
        }
        break;
    case ViewMode::ObservationNetwork:
        if (_obsNetView)
        {
            _obsNetView->resetLayout();
        }
        break;
    case ViewMode::None:
        break;
    }
}

void WorkspaceCenterWidget::clearProjectView()
{
    ++_cameraPoseGeneration;
    _cameraPoseMetadata = QJsonArray();
    if (_canvas)
    {
        _canvas->showImage(QString());
    }
    if (_dualImageViewer)
    {
        _dualImageViewer->clearViewer();
    }
    if (_obsNetView)
    {
        _obsNetView->clearNetwork();
    }
    if (_modelView)
    {
        _modelView->clearProjectScene();
    }

    setDocumentTab(ModelTabIndex, tr("模型"), tr("模型"));
    setDocumentTab(ImageTabIndex, tr("影像"), tr("影像"), false);
    setDocumentTab(CompareTabIndex, tr("对比"), tr("对比"), false);
    setDocumentTab(ObservationNetworkTabIndex, tr("观测网络"), tr("观测网络"), false);
    showModelView();
}

void WorkspaceCenterWidget::showObservationNetwork(const xjw::ObservationNetwork &net, const QString &title)
{
    if (!_stack || !_obsNetView || !_documentTabs)
    {
        return;
    }

    const QString tabText = title.trimmed().isEmpty() ? tr("观测网络") : title.trimmed();
    setDocumentTab(ObservationNetworkTabIndex, tabText, tabText);

    _obsNetView->setNetwork(net);
    activateView(ObservationNetworkTabIndex);
}

void WorkspaceCenterWidget::setProjectMeta(const QJsonObject &meta)
{
    const QJsonArray images = xjw::common::project::projectImageEntries(meta);
    const QJsonArray camera_pose_metadata = cameraPoseMetadataFromImages(images);
    if (_cameraPoseMetadata == camera_pose_metadata)
    {
        return;
    }

    _cameraPoseMetadata = camera_pose_metadata;
    refreshModelFromMeta(meta);
}

void WorkspaceCenterWidget::refreshModelFromMeta(const QJsonObject &meta)
{
    const QJsonArray images = xjw::common::project::projectImageEntries(meta);
    const quint64 generation = ++_cameraPoseGeneration;
    auto *watcher = new QFutureWatcher<QVector<CameraSceneWidget::CameraPose>>(this);
    QPointer<WorkspaceCenterWidget> self(this);
    connect(watcher,
            &QFutureWatcher<QVector<CameraSceneWidget::CameraPose>>::finished,
            this,
            [self, watcher, generation]()
    {
        if (self && generation == self->_cameraPoseGeneration && self->_modelView)
        {
            self->_modelView->setCameraPoses(watcher->result());
        }
        watcher->deleteLater();
    });
    watcher->setFuture(QtConcurrent::run([images]()
    {
        return cameraPosesFromImages(images);
    }));
}
