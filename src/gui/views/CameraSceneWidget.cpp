// =============================================================================
// 文件: CameraSceneWidget.cpp
// 功能: 可复用相机三维场景控件实现
// 内容:
//   - CameraSceneWidget：Qt RHI/Vulkan 三维渲染控件
//       · 点云 / PLY 模型 / 相机平面卡片渲染（QRhiBuffer + .qsb shader）
//       · Arcball 自由旋转 + 单轴环旋转（X/Y/Z Gizmo）
//       · 中键平移、滚轮缩放
//       · 透明 QWidget 覆盖层（Gizmo 环、坐标轴、相机卡片、欧拉角）
// =============================================================================
#include "CameraSceneWidget.h"
#include "CameraSceneViewMath.h"
#include "ObjRenderPreparation.h"
#include "PointCloudEditPreparation.h"
#include "PointCloudSnapshotIO.h"
#include "SceneGeometryPreparation.h"

#include "LayerImageLoader.h"
#include "GuiTaskRunner.h"
#include "Logger.h"
#include "io/PathIO.h"

#include <QPainter>
#include <QPainterPath>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QCursor>
#include <QIODevice>
#include <QImage>
#include <QPixmap>
#include <QVector2D>
#include <QMatrix4x4>
#include <QtMath>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QElapsedTimer>
#include <QKeyEvent>
#include <QKeySequence>
#include <QLineF>
#include <QMetaObject>
#include <QPointer>
#include <QSet>
#include <rhi/qrhi.h>
#include <rhi/qshader.h>
#include <QSizePolicy>
#include <QTextStream>
#include <QThread>
#include <QTimer>
#include <QWidget>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <plapoint/io/xyz_io.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/obj_io.h>

namespace
{

constexpr int kCameraThumbnailWidth = 220;
constexpr int kCameraThumbnailHeight = 160;
constexpr int kSmallCameraAtlasSize = 2048;
constexpr int kLargeCameraAtlasSize = 4096;
constexpr int kSmallCameraAtlasCapacity =
    (kSmallCameraAtlasSize / kCameraThumbnailWidth)
    * (kSmallCameraAtlasSize / kCameraThumbnailHeight);
constexpr int kCameraInstanceStrideFloats = 20;
constexpr qint64 kFullCameraImageCacheLimitBytes = 128LL * 1024LL * 1024LL;
constexpr std::size_t kMaximumCompactSelectionPoints = 1'000'000;

using xjw::gui::point_cloud::PointCloudSnapshotStageResult;
using xjw::gui::point_cloud::stagePointCloudSnapshot;

static_assert(static_cast<int>(xjw::gui::tie_points::ColorMode::Color) == 0);
static_assert(static_cast<int>(xjw::gui::tie_points::ColorMode::Elevation) == 1);
static_assert(static_cast<int>(xjw::gui::tie_points::ColorMode::ImageCount) == 2);
static_assert(static_cast<int>(xjw::gui::model_views::ColorMode::Texture) == 0);
static_assert(static_cast<int>(xjw::gui::model_views::ColorMode::Shaded) == 1);
static_assert(static_cast<int>(xjw::gui::model_views::ColorMode::Solid) == 2);
static_assert(static_cast<int>(xjw::gui::model_views::ColorMode::Wireframe) == 3);
static_assert(static_cast<int>(xjw::gui::model_views::ColorMode::Elevation) == 4);

struct ManualPointCloudTaskResult
{
    PointCloudEditResult edit;
    PointCloudSnapshotStageResult save;
    xjw::gui::tie_points::ScalarRange imageCountRange;
};

xjw::gui::tie_points::ScalarRange imageCountRangeFor(
    const QVector<int> &imageCounts,
    std::size_t pointCount)
{
    if (imageCounts.size() != static_cast<qsizetype>(pointCount)
        || imageCounts.isEmpty())
    {
        return {};
    }
    const auto [minimum, maximum] = std::minmax_element(
        imageCounts.cbegin(),
        imageCounts.cend());
    return {double(*minimum), double(*maximum)};
}

struct SceneLoadTaskResult
{
    std::shared_ptr<RenderCloud> cloud;
    ObjRenderPreparation renderPreparation;
    PointRenderPreparation pointPreparation;
    CloudSpatialSummary spatialSummary;
    QImage textureImage;
    QString texturePath;
    QString textureWarning;
    qint64 parseElapsedMs = 0;
    qint64 prepareElapsedMs = 0;
};

struct TiePointMetadataLoadResult
{
    xjw::gui::tie_points::ImageCountMetadata metadata;
    QByteArray scalarData;
    xjw::gui::tie_points::ScalarRange range;
};

SceneLoadTaskResult prepareSceneLoad(
    std::shared_ptr<RenderCloud> cloud,
    const std::atomic_bool *cancellationFlag = nullptr)
{
    SceneLoadTaskResult result;
    result.cloud = std::move(cloud);
    if (result.cloud && result.cloud->hasFaces())
    {
        result.renderPreparation = prepareObjRenderData(
            *result.cloud, false, cancellationFlag);
        result.spatialSummary = prepareCloudSpatialSummary(
            *result.cloud, cancellationFlag);
    }
    else if (result.cloud)
    {
        result.pointPreparation = preparePointRenderData(
            *result.cloud, {}, cancellationFlag);
        result.spatialSummary = result.pointPreparation.spatialSummary;
    }
    return result;
}

QString firstDiffuseTexturePath(const QString &obj_path, const RenderCloud &cloud)
{
    QString material_name = xjw::common::io::fromUtf8Path(cloud.materialLibraryFile()).trimmed();
    if (material_name.isEmpty())
    {
        return QString();
    }

    QFileInfo material_info(material_name);
    const QString material_path = material_info.isAbsolute()
        ? material_info.absoluteFilePath()
        : QDir(QFileInfo(obj_path).absolutePath()).filePath(material_name);
    QFile material_file(material_path);
    if (!material_file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return QString();
    }

    QTextStream stream(&material_file);
    while (!stream.atEnd())
    {
        const QString line = stream.readLine().trimmed();
        if (!line.startsWith(QStringLiteral("map_Kd")))
        {
            continue;
        }
        const QString texture_reference = line.mid(6).trimmed();
        if (texture_reference.isEmpty())
        {
            return QString();
        }
        QFileInfo texture_info(texture_reference);
        return QDir::cleanPath(texture_info.isAbsolute()
            ? texture_info.absoluteFilePath()
            : QDir(QFileInfo(material_path).absolutePath()).filePath(texture_reference));
    }
    return QString();
}

SceneLoadTaskResult loadObjWithMaterialTexture(
    const QString &obj_path,
    const std::function<void(int, const QString &)> &progress,
    const std::function<bool()> &is_cancelled = {},
    const std::atomic_bool *cancellationFlag = nullptr)
{
    SceneLoadTaskResult result;
    QElapsedTimer timer;
    timer.start();
    result.cloud = plapoint::io::readObj<float>(
        xjw::common::io::toNativeNarrowPath(obj_path));
    result.parseElapsedMs = timer.elapsed();
    if ((is_cancelled && is_cancelled())
        || !result.cloud || result.cloud->size() == 0)
    {
        return result;
    }
    if (!result.cloud->hasTextureCoords() || !result.cloud->hasFaceTextureIndices())
    {
        result.textureWarning = QStringLiteral("OBJ 未包含完整的面级 UV，使用顶点颜色显示");
    }
    else
    {
        if (progress)
        {
            progress(82, QStringLiteral("正在读取 OBJ 材质纹理..."));
        }
        result.texturePath = firstDiffuseTexturePath(obj_path, *result.cloud);
        if (result.texturePath.isEmpty())
        {
            result.textureWarning = QStringLiteral("OBJ 的 MTL 未指定漫反射纹理，使用顶点颜色显示");
        }
        else if (!QFileInfo::exists(result.texturePath)
                 || !result.textureImage.load(result.texturePath))
        {
            result.textureWarning = QStringLiteral("无法读取 OBJ 纹理图像: %1")
                                        .arg(result.texturePath);
            result.textureImage = QImage();
        }
    }

    if (!result.textureImage.isNull())
    {
        result.textureImage = result.textureImage.convertToFormat(QImage::Format_RGBA8888);
        result.cloud->setTextureImageFile(
            xjw::common::io::toUtf8Path(result.texturePath));
    }
    if (is_cancelled && is_cancelled())
    {
        return result;
    }
    if (progress)
    {
        progress(88, QStringLiteral("正在准备 OBJ 渲染数据..."));
    }
    timer.restart();
    if (result.cloud->hasFaces())
    {
        result.renderPreparation = prepareObjRenderData(
            *result.cloud, !result.textureImage.isNull(), cancellationFlag);
        result.spatialSummary = prepareCloudSpatialSummary(
            *result.cloud, cancellationFlag);
    }
    else
    {
        result.pointPreparation = preparePointRenderData(
            *result.cloud, {}, cancellationFlag);
        result.spatialSummary = result.pointPreparation.spatialSummary;
    }
    result.prepareElapsedMs = timer.elapsed();
    return result;
}

QShader loadSceneShader(const QString &resourcePath, QString *errorMessage)
{
    QFile file(resourcePath);
    if (!file.open(QIODevice::ReadOnly))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("Vulkan 渲染着色器资源缺失：%1").arg(resourcePath);
        }
        return {};
    }

    QShader shader = QShader::fromSerialized(file.readAll());
    if (!shader.isValid() && errorMessage)
    {
        *errorMessage = QStringLiteral("Vulkan 渲染着色器加载失败：%1").arg(resourcePath);
    }
    return shader;
}

} // namespace

class CameraSceneOverlayWidget : public QWidget
{
public:
    explicit CameraSceneOverlayWidget(CameraSceneWidget *scene)
        : QWidget(scene)
        , _scene(scene)
    {
        setAttribute(Qt::WA_TransparentForMouseEvents, true);
        setAttribute(Qt::WA_NoSystemBackground, true);
        setAttribute(Qt::WA_TranslucentBackground, true);
        setAutoFillBackground(false);
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        if (!_scene)
        {
            return;
        }

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, true);
        _scene->paintOverlay(painter);
    }

private:
    CameraSceneWidget *_scene = nullptr;
};

// 构造函数：设置基础可用尺寸，启用鼠标追踪（悬停检测需要），
// 设置默认视角为俯仰 -25°、偏航 35°（斜上方看向场景）
CameraSceneWidget::CameraSceneWidget(QWidget *parent)
    : QRhiWidget(parent)
{
    setApi(QRhiWidget::Api::Vulkan);
    setSampleCount(1);

    setMinimumSize(240, 160);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setMouseTracking(true); // 启用鼠标追踪，以便在无按键时检测悬停轴
    _viewRot = QQuaternion::fromEulerAngles(-25.0f, 35.0f, 0.0f); // 默认斜视角
    setFocusPolicy(Qt::StrongFocus);
    updateCursor();
    const int ideal_threads = std::max(1, QThread::idealThreadCount());
    _cameraImageLoadPool.setMaxThreadCount(
        std::clamp((ideal_threads + 1) / 2, 4, 12));
    _cameraImageLoadPool.setExpiryTimeout(10000);

    _overlayWidget = new CameraSceneOverlayWidget(this);
    _overlayWidget->setGeometry(rect());
    _overlayWidget->show();
    _overlayWidget->raise();

    connect(this, &CameraSceneWidget::plyLoadProgressChanged,
            this, [this](int generation, int percent, const QString &statusText)
    {
        if (generation != _loadGen)
        {
            return;
        }
        if (!_loading && percent < 100)
        {
            return;
        }
        _loading = percent < 100;
        _plyLoadProgressPercent = qBound(0, percent, 100);
        _plyLoadProgressText = statusText;
        update();
    }, Qt::QueuedConnection);
}

CameraSceneWidget::~CameraSceneWidget()
{
    if (_sceneLoadCancellation)
    {
        _sceneLoadCancellation->store(true, std::memory_order_relaxed);
    }
    if (_manualSelectionCancellation)
    {
        _manualSelectionCancellation->store(true, std::memory_order_relaxed);
    }
    if (_manualEditCancellation)
    {
        _manualEditCancellation->store(true, std::memory_order_relaxed);
    }
    if (_tiePointMetadataCancellation)
    {
        _tiePointMetadataCancellation->store(true, std::memory_order_relaxed);
    }
    _pendingSceneLoad.reset();
    _pendingManualSelection.reset();
    _pendingTiePointMetadataLoad.reset();
    ++_cameraImageLoadGeneration;
    _cameraImageLoadQueue.clear();
    _cameraImageLoadPool.clear();
    _cameraImageLoadPool.waitForDone();
}

// 设置要渲染的相机姿态列表。
// 调用后触发重绘，场景中每个姿态点将绘制相机平面卡片和名称标注。
void CameraSceneWidget::setCameraPoses(const QVector<CameraPose> &poses)
{
    QVector<CameraPose> deduplicated_poses;
    deduplicated_poses.reserve(poses.size());
    QSet<QString> cameraKeys;
    for (const CameraPose &pose : poses)
    {
        QString cameraKey = normalizedCameraPath(pose.imagePath);
        if (cameraKey.isEmpty())
        {
            cameraKey = pose.name.trimmed();
        }
        if (!cameraKey.isEmpty())
        {
#ifdef Q_OS_WIN
            cameraKey = cameraKey.toCaseFolded();
#endif
            if (cameraKeys.contains(cameraKey))
            {
                continue;
            }
            cameraKeys.insert(cameraKey);
        }
        deduplicated_poses.push_back(pose);
    }

    auto same_pose = [this](const CameraPose &lhs, const CameraPose &rhs)
    {
        if (normalizedCameraPath(lhs.imagePath) != normalizedCameraPath(rhs.imagePath)
            || lhs.name != rhs.name
            || lhs.center != rhs.center
            || lhs.focalX != rhs.focalX
            || lhs.focalY != rhs.focalY
            || lhs.principalX != rhs.principalX
            || lhs.principalY != rhs.principalY
            || lhs.imageWidth != rhs.imageWidth
            || lhs.imageHeight != rhs.imageHeight
            || lhs.uAxisSign != rhs.uAxisSign
            || lhs.vAxisSign != rhs.vAxisSign
            || lhs.depthAxisFlipped != rhs.depthAxisFlipped)
        {
            return false;
        }
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                if (lhs.rotation(row, column) != rhs.rotation(row, column))
                {
                    return false;
                }
            }
        }
        return true;
    };

    bool poses_unchanged = _poses.size() == deduplicated_poses.size();
    bool reusable_image_sequence = poses_unchanged;
    for (qsizetype index = 0; index < _poses.size() && poses_unchanged; ++index)
    {
        poses_unchanged = same_pose(_poses.at(index), deduplicated_poses.at(index));
    }
    if (poses_unchanged)
    {
        return;
    }
    if (_manualPruneMode)
    {
        clearManualPointSelection();
    }
    for (qsizetype index = 0;
         index < _poses.size() && reusable_image_sequence;
         ++index)
    {
        reusable_image_sequence =
            normalizedCameraPath(_poses.at(index).imagePath)
            == normalizedCameraPath(deduplicated_poses.at(index).imagePath);
    }

    _poses = std::move(deduplicated_poses);
    _imagePipeline.geometryDirty = true;
    _poseIndexByNormalizedPath.clear();
    _poseIndexByNormalizedPath.reserve(_poses.size());
    for (qsizetype index = 0; index < _poses.size(); ++index)
    {
        const QString path = normalizedCameraPath(_poses.at(index).imagePath);
        if (!path.isEmpty())
        {
            _poseIndexByNormalizedPath.insert(path, static_cast<int>(index));
        }
    }
    _lockedCameraImagePoseIndex = -1;
    if (_cameraImageLocked)
    {
        _lockedCameraImagePoseIndex = _poseIndexByNormalizedPath.value(
            normalizedCameraPath(_lockedCameraImagePath), -1);
        if (_lockedCameraImagePoseIndex < 0 && !_lockedCameraImageName.isEmpty())
        {
            for (qsizetype index = 0; index < _poses.size(); ++index)
            {
                const CameraPose &pose = _poses.at(index);
                if (pose.name == _lockedCameraImageName
                    || QFileInfo(pose.imagePath).fileName() == _lockedCameraImageName)
                {
                    _lockedCameraImagePoseIndex = static_cast<int>(index);
                    break;
                }
            }
        }
    }
    _cameraViewCandidates.clear();
    _cameraViewCandidates.reserve(_poses.size());
    for (qsizetype index = 0; index < _poses.size(); ++index)
    {
        const CameraPose &pose = _poses.at(index);
        _cameraViewCandidates.push_back({
            static_cast<int>(index),
            xjw::gui::camera_scene::cameraForwardDirection(
                pose.rotation, pose.depthAxisFlipped),
            pose.center,
            !pose.imagePath.isEmpty(),
        });
    }
    _activeCameraImagePoseIndex = -1;
    _cacheDirty = true; // 相机位置变更，缓存失效
    if (!reusable_image_sequence)
    {
        ++_cameraImageLoadGeneration;
        _cameraImageCache.clear();
        _fullImageCacheLru.clear();
        _fullImageCacheBytes = 0;
        _pendingThumbnailPoseIndices.clear();
        _cameraImageLoadQueue.clear();
        _cameraImageLoadsQueued.clear();
        _cameraImageLoadFailures.clear();
        _cameraThumbnailLoadTotal = 0;
        for (const CameraPose &pose : _poses)
        {
            if (!pose.imagePath.isEmpty())
            {
                ++_cameraThumbnailLoadTotal;
            }
        }
        _cameraThumbnailLoadCompleted = 0;
        _thumbnailPipeline.resourcesDirty = true;
    }
    _thumbnailPipeline.instancesDirty = true;
    if (_showCameraImage)
    {
        updateActiveCameraForView();
    }
    update(); // 触发 Vulkan 帧重绘
}

void CameraSceneWidget::setShowGizmo(bool show)
{
    if (_showGizmo != show)
    {
        _showGizmo = show;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameras(bool show)
{
    if (_showCameras != show)
    {
        _showCameras = show;
        if (!_showCameras)
        {
            discardQueuedCameraThumbnails();
        }
        else
        {
            _thumbnailPipeline.resourcesDirty = true;
            _thumbnailPipeline.instancesDirty = true;
            _thumbnailPipeline.pipelinesDirty = true;
            _pipelinesDirty = true;
        }
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameraThumbnails(bool show)
{
    if (_showCameraThumbnails != show)
    {
        _showCameraThumbnails = show;
        if (_showCameraThumbnails)
        {
            _cameraThumbnailLoadCompleted = 0;
            _thumbnailPipeline.pipelinesDirty = true;
            _pipelinesDirty = true;
        }
        else
        {
            discardQueuedCameraThumbnails();
        }
        _thumbnailPipeline.resourcesDirty = true;
        _thumbnailPipeline.instancesDirty = true;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameraLocalAxes(bool show)
{
    if (_showCameraLocalAxes != show)
    {
        _showCameraLocalAxes = show;
        _thumbnailPipeline.instancesDirty = true;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setShowCameraImage(bool show)
{
    if (_showCameraImage != show)
    {
        _showCameraImage = show;
        if (_showCameraImage)
        {
            _imagePipeline.pipelineDirty = true;
            _pipelinesDirty = true;
            updateActiveCameraForView();
            if (_cameraImageLocked)
            {
                refreshLockedCameraImage();
            }
        }
        else
        {
            _activeCameraImagePoseIndex = -1;
        }
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setCameraImageDisplayLayer(CameraImageDisplayLayer layer)
{
    if (_cameraImageDisplayLayer != layer)
    {
        _cameraImageDisplayLayer = layer;
        updateCameraOverlay();
    }
}

void CameraSceneWidget::setCameraImageLocked(bool locked)
{
    if (_cameraImageLocked == locked)
    {
        return;
    }

    _cameraImageLocked = locked;
    _imagePipeline.geometryDirty = true;
    if (_cameraImageLocked)
    {
        refreshLockedCameraImage();
    }
    else
    {
        _lockedCameraImagePath.clear();
        _lockedCameraImageName.clear();
        _lockedCameraImagePoseIndex = -1;
        updateActiveCameraForView();
    }
    updateCameraOverlay();
}

void CameraSceneWidget::setHighlightedCameraPath(const QString &imagePath)
{
    const QString normalizedPath = normalizedCameraPath(imagePath);
    if (_highlightedCameraPath == normalizedPath)
    {
        return;
    }

    _highlightedCameraPath = normalizedPath;
    _thumbnailPipeline.instancesDirty = true;
    updateCameraOverlay();
}

void CameraSceneWidget::clearHighlightedCamera()
{
    if (_highlightedCameraPath.isEmpty())
    {
        return;
    }

    _highlightedCameraPath.clear();
    _thumbnailPipeline.instancesDirty = true;
    updateCameraOverlay();
}

void CameraSceneWidget::clearProjectScene()
{
    cancelPendingLoad();
    ++_cameraImageLoadGeneration;
    _poses.clear();
    _poseIndexByNormalizedPath.clear();
    _cameraViewCandidates.clear();
    _cloud = RenderCloud();
    updateSampleCountForGeometry();
    _isTiePointCloud = false;
    _tiePointImageCounts.clear();
    _tiePointMetadataLoading = false;
    _tiePointMetadataError.clear();
    _meshTextureImage = QImage();
    _meshTexturePath.clear();
    _meshHasTexture = false;
    clearPreparedGeometry();
    _currentCloudPath.clear();
    _cameraImageCache.clear();
    _fullImageCacheLru.clear();
    _fullImageCacheBytes = 0;
    _pendingThumbnailPoseIndices.clear();
    _cameraImageLoadQueue.clear();
    _cameraImageLoadsQueued.clear();
    _cameraImageLoadFailures.clear();
    _cameraThumbnailLoadTotal = 0;
    _cameraThumbnailLoadCompleted = 0;
    _activeCameraImagePoseIndex = -1;
    _lockedCameraImagePoseIndex = -1;
    _imagePipeline.geometryDirty = true;
    _highlightedCameraPath.clear();
    _hasFocusedGeometryBounds = false;
    _fitViewAfterLoad = false;
    _cacheDirty = true;
    _gpuDirty = true;
    _thumbnailPipeline.resourcesDirty = true;
    resetView();
}

// Reader API 本身不可中断；协作标志会跳过后续渲染准备，single-flight
// 保证同一时刻最多只有一个大文件解析任务占用 CPU 和内存。
void CameraSceneWidget::cancelPendingLoad()
{
    ++_loadGen;
    if (_sceneLoadCancellation)
    {
        _sceneLoadCancellation->store(true, std::memory_order_relaxed);
    }
    if (_manualEditCancellation)
    {
        _manualEditCancellation->store(true, std::memory_order_relaxed);
    }
    if (_tiePointMetadataCancellation)
    {
        _tiePointMetadataCancellation->store(true, std::memory_order_relaxed);
    }
    _pendingSceneLoad.reset();
    _pendingTiePointMetadataLoad.reset();
    clearManualPointSelection();
    _manualUndoStack.clear();
    _loading = false;
    _plyLoadProgressPercent = -1;
    _plyLoadProgressText.clear();
    _fitViewAfterLoad = false;
}

void CameraSceneWidget::clearPreparedGeometry()
{
    if (!_geometryUploadError.isEmpty() && _renderError == _geometryUploadError)
    {
        _renderError.clear();
    }
    _geometryUploadError.clear();
    _preparedMesh = {};
    _preparedPointBuffer = false;
    _preparedPointVertexData.clear();
    _preparedPointVertexCount = 0;
    _pointScalarBuffer.vertexData.clear();
    _pointScalarBuffer.vertexCount = 0;
    _manualHighlightPointBuffer.vertexData.clear();
    _manualHighlightPointBuffer.vertexCount = 0;
    _manualHighlightScalarBuffer.vertexData.clear();
    _manualHighlightScalarBuffer.vertexCount = 0;
    _tiePointScalarData.clear();
    _cloudSpatialSummary = {};
}

void CameraSceneWidget::applyCloudSpatialSummary(const CloudSpatialSummary &summary)
{
    _cloudSpatialSummary = summary;
    _cacheDirty = true;
    _imagePipeline.geometryDirty = true;
}

void CameraSceneWidget::applyPointPreparation(PointRenderPreparation preparation)
{
    const bool valid = preparation.isValid();
    _preparedPointVertexData = std::move(preparation.vertexData);
    _preparedPointVertexCount = preparation.pointCount;
    _preparedPointBuffer = valid;
    _pointScalarBuffer.vertexData = std::move(preparation.scalarData);
    _pointScalarBuffer.vertexCount = preparation.pointCount;
    _pointScalarBuffer.strideBytes = int(sizeof(float));
    _pointScalarBuffer.dirty = true;
    const bool has_tie_point_metadata = _isTiePointCloud
        && (!_tiePointImageCounts.isEmpty() || !_tiePointScalarData.isEmpty());
    const bool tie_point_metadata_matches = has_tie_point_metadata
        && _tiePointImageCounts.size() == static_cast<qsizetype>(preparation.pointCount)
        && _tiePointScalarData.size()
            == preparation.pointCount * static_cast<int>(sizeof(float));
    if (tie_point_metadata_matches)
    {
        _pointScalarBuffer.vertexData = _tiePointScalarData;
    }
    else if (has_tie_point_metadata)
    {
        _tiePointImageCounts.clear();
        _tiePointScalarData.clear();
        _tiePointImageCountRange = {};
        _tiePointMetadataError = tr("观测元数据点数与当前点云不一致，已忽略过期数据。");
    }
    _tiePointElevationRange = preparation.spatialSummary.elevationRange;
    applyCloudSpatialSummary(preparation.spatialSummary);
}

// 标记缓存脏 + 重算（在加载完成后或场景数据变更后调用）
void CameraSceneWidget::invalidateCache() const
{
    if (_cloudSpatialSummary.valid
        && _cloudSpatialSummary.sourcePointCount == _cloud.size())
    {
        _hasCloudBounds = true;
        _cachedCloudAABBMin = _cloudSpatialSummary.aabbMinimum;
        _cachedCloudAABBMax = _cloudSpatialSummary.aabbMaximum;
        _cachedCloudBoxVertices = _cloudSpatialSummary.boxLineVertices;

        QVector3D accumulated = _cloudSpatialSummary.center
            * static_cast<float>(_cloudSpatialSummary.validPointCount);
        std::size_t count = _cloudSpatialSummary.validPointCount;
        QVector3D minimum = _cloudSpatialSummary.aabbMinimum;
        QVector3D maximum = _cloudSpatialSummary.aabbMaximum;
        for (const CameraPose &pose : _poses)
        {
            accumulated += pose.center;
            ++count;
            minimum.setX(std::min(minimum.x(), pose.center.x()));
            minimum.setY(std::min(minimum.y(), pose.center.y()));
            minimum.setZ(std::min(minimum.z(), pose.center.z()));
            maximum.setX(std::max(maximum.x(), pose.center.x()));
            maximum.setY(std::max(maximum.y(), pose.center.y()));
            maximum.setZ(std::max(maximum.z(), pose.center.z()));
        }
        _cachedCenter = accumulated / static_cast<float>(count);
        _cachedAABBMin = minimum;
        _cachedAABBMax = maximum;
        float radius = _cloudSpatialSummary.p95Radius
            + (_cloudSpatialSummary.center - _cachedCenter).length();
        if (!_poses.isEmpty())
        {
            std::vector<float> camera_distances;
            camera_distances.reserve(static_cast<std::size_t>(_poses.size()));
            for (const CameraPose &pose : _poses)
            {
                camera_distances.push_back((pose.center - _cachedCenter).length());
            }
            const std::size_t p95 = std::min(
                camera_distances.size() - 1,
                static_cast<std::size_t>(camera_distances.size() * 0.95));
            std::nth_element(camera_distances.begin(),
                             camera_distances.begin() + p95,
                             camera_distances.end());
            radius = std::max(radius, camera_distances[p95] * 1.15f);
        }
        _cachedRadius = std::max(1.0e-4f, radius);
        _cacheDirty = false;
        return;
    }

    // Every non-empty geometry load prepares its cloud summary before it is
    // published to the widget. Do not re-scan a large cloud on the GUI thread
    // when preparation failed; keep camera-only framing and surface the load
    // or upload error instead.
    _hasCloudBounds = false;
    _cachedCloudBoxVertices.clear();
    if (_poses.isEmpty())
    {
        _cachedCenter = QVector3D(0, 0, 0);
        _cachedRadius = 10.0f;
        _cachedAABBMin = QVector3D(-10, -10, -10);
        _cachedAABBMax = QVector3D(10, 10, 10);
    }
    else
    {
        QVector3D accumulated;
        QVector3D minimum = _poses.first().center;
        QVector3D maximum = minimum;
        for (const CameraPose &pose : _poses)
        {
            accumulated += pose.center;
            minimum.setX(std::min(minimum.x(), pose.center.x()));
            minimum.setY(std::min(minimum.y(), pose.center.y()));
            minimum.setZ(std::min(minimum.z(), pose.center.z()));
            maximum.setX(std::max(maximum.x(), pose.center.x()));
            maximum.setY(std::max(maximum.y(), pose.center.y()));
            maximum.setZ(std::max(maximum.z(), pose.center.z()));
        }
        _cachedCenter = accumulated / static_cast<float>(_poses.size());
        _cachedAABBMin = minimum;
        _cachedAABBMax = maximum;
        std::vector<float> distances;
        distances.reserve(static_cast<std::size_t>(_poses.size()));
        for (const CameraPose &pose : _poses)
        {
            distances.push_back((pose.center - _cachedCenter).length());
        }
        const std::size_t p95 = std::min(
            distances.size() - 1,
            static_cast<std::size_t>(static_cast<double>(distances.size()) * 0.95));
        std::nth_element(distances.begin(), distances.begin() + p95, distances.end());
        _cachedRadius = std::max(1.0e-4f, distances[p95] * 1.15f);
    }
    _cacheDirty = false;
}

// 从 XYZ 格式文本文件异步加载点云数据，使用 plapoint IO 解析。
void CameraSceneWidget::loadPointCloudFromXyz(const QString &xyzPath)
{
    loadPointCloudFromXyzInternal(xyzPath, false, true);
}

void CameraSceneWidget::loadPointCloudFromXyzInternal(const QString &xyzPath,
                                                       bool tiePointCloud,
                                                       bool fitAfterLoad)
{
    requestSceneLoad({SceneLoadFormat::Xyz,
                      xyzPath,
                      tiePointCloud,
                      fitAfterLoad,
                      true,
                      0});
}

// 从 PLY 文件异步加载网格模型或点云。
void CameraSceneWidget::loadModelFromPly(const QString &plyPath)
{
    loadModelFromPlyInternal(plyPath, false, false, false);
}

void CameraSceneWidget::loadPointCloudFromPly(const QString &plyPath)
{
    loadModelFromPlyInternal(plyPath, false, true, true);
}

void CameraSceneWidget::loadModelFromPlyInternal(const QString &plyPath,
                                                 bool tiePointCloud,
                                                 bool fitAfterLoad,
                                                 bool pointCloudResource)
{
    requestSceneLoad({SceneLoadFormat::Ply,
                      plyPath,
                      tiePointCloud,
                      fitAfterLoad,
                      pointCloudResource,
                      0});
}

void CameraSceneWidget::loadModelFromObj(const QString &objPath)
{
    loadModelFromObjInternal(objPath, false, false, false);
}

void CameraSceneWidget::loadPointCloudFromObj(const QString &objPath)
{
    loadModelFromObjInternal(objPath, false, true, true);
}

void CameraSceneWidget::loadModelFromObjInternal(const QString &objPath,
                                                 bool tiePointCloud,
                                                 bool fitAfterLoad,
                                                 bool pointCloudResource)
{
    requestSceneLoad({SceneLoadFormat::Obj,
                      objPath,
                      tiePointCloud,
                      fitAfterLoad,
                      pointCloudResource,
                      0});
}

void CameraSceneWidget::requestSceneLoad(SceneLoadRequest request)
{
    cancelPendingLoad();
    request.generation = _loadGen;
    _currentCloudPath = request.path;
    _cloud = RenderCloud();
    _isTiePointCloud = request.tiePointCloud;
    _tiePointImageCounts.clear();
    _tiePointMetadataLoading = false;
    _tiePointMetadataError.clear();
    _meshTextureImage = QImage();
    _meshTexturePath.clear();
    _meshHasTexture = false;
    clearPreparedGeometry();
    _texturedMeshPipeline.uploadedTexturePath.clear();
    _hasFocusedGeometryBounds = false;
    _fitViewAfterLoad = request.fitAfterLoad;
    _pointCloudPointSize = 2.4f;
    _cacheDirty = true;
    _gpuDirty = true;
    _loading = true;
    _plyLoadProgressPercent = 0;
    const QString format_name = request.format == SceneLoadFormat::Xyz
        ? QStringLiteral("XYZ")
        : request.format == SceneLoadFormat::Ply
            ? QStringLiteral("PLY")
            : QStringLiteral("OBJ");
    _plyLoadProgressText = request.pointCloudResource
        ? tr("正在加载 %1 点云...").arg(format_name)
        : tr("正在加载 %1 模型...").arg(format_name);
    _pendingSceneLoad = std::move(request);
    update();
    LOG_INFO(QStringLiteral("[3D] 正在加载 %1 %2: %3")
                 .arg(format_name,
                      _pendingSceneLoad->pointCloudResource
                          ? QStringLiteral("点云")
                          : QStringLiteral("模型"),
                      _pendingSceneLoad->path));
    emit plyLoadProgressChanged(_loadGen, 0, _plyLoadProgressText);
    pumpSceneLoad();
}

void CameraSceneWidget::pumpSceneLoad()
{
    if (_sceneLoadWorkerActive || !_pendingSceneLoad)
    {
        return;
    }

    SceneLoadRequest request = std::move(*_pendingSceneLoad);
    _pendingSceneLoad.reset();
    auto cancellation = std::make_shared<std::atomic_bool>(false);
    _sceneLoadCancellation = cancellation;
    _sceneLoadWorkerActive = true;
    if (request.format == SceneLoadFormat::Ply)
    {
        emit plyLoadProgressChanged(
            request.generation,
            5,
            tr("正在完整加载 PLY 点云或模型（不抽稀）..."));
    }

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [request, cancellation]() -> SceneLoadTaskResult
        {
            auto is_cancelled = [cancellation]()
            {
                return cancellation->load(std::memory_order_relaxed);
            };
            if (is_cancelled())
            {
                return {};
            }

            if (request.format == SceneLoadFormat::Xyz)
            {
                auto cloud = plapoint::io::readXyz<float>(
                    xjw::common::io::toNativeNarrowPath(request.path));
                return is_cancelled() ? SceneLoadTaskResult()
                                      : prepareSceneLoad(std::move(cloud), cancellation.get());
            }
            if (request.format == SceneLoadFormat::Ply)
            {
                auto cloud = plapoint::io::readPly<float>(
                    xjw::common::io::toNativeNarrowPath(request.path));
                return is_cancelled() ? SceneLoadTaskResult()
                                      : prepareSceneLoad(std::move(cloud), cancellation.get());
            }
            if (!request.pointCloudResource)
            {
                return loadObjWithMaterialTexture(
                    request.path, {}, is_cancelled, cancellation.get());
            }

            SceneLoadTaskResult result;
            QElapsedTimer timer;
            timer.start();
            result.cloud = plapoint::io::readObj<float>(
                xjw::common::io::toNativeNarrowPath(request.path));
            result.parseElapsedMs = timer.elapsed();
            if (is_cancelled() || !result.cloud || result.cloud->size() == 0)
            {
                return result;
            }
            timer.restart();
            if (result.cloud->hasFaces())
            {
                result.renderPreparation = prepareObjRenderData(
                    *result.cloud, false, cancellation.get());
                result.spatialSummary = prepareCloudSpatialSummary(
                    *result.cloud, cancellation.get());
            }
            else
            {
                result.pointPreparation = preparePointRenderData(
                    *result.cloud, {}, cancellation.get());
                result.spatialSummary = result.pointPreparation.spatialSummary;
            }
            result.prepareElapsedMs = timer.elapsed();
            return result;
        },
        [request, cancellation](
            CameraSceneWidget *self,
            xjw::gui::tasks::TaskOutcome<SceneLoadTaskResult> outcome)
        {
            self->_sceneLoadWorkerActive = false;
            const bool is_current = request.generation == self->_loadGen
                && cancellation == self->_sceneLoadCancellation
                && !cancellation->load(std::memory_order_relaxed);
            if (is_current)
            {
                self->_loading = false;
                self->_plyLoadProgressPercent = -1;
                self->_plyLoadProgressText.clear();
                self->_fitViewAfterLoad = false;
            }

            if (is_current && outcome.succeeded())
            {
                SceneLoadTaskResult result = std::move(*outcome.value);
                if (result.cloud)
                {
                    self->_cloud = std::move(*result.cloud);
                    self->updateSampleCountForGeometry();
                    if (request.format == SceneLoadFormat::Obj)
                    {
                        self->_meshTextureImage = std::move(result.textureImage);
                        self->_meshTexturePath = std::move(result.texturePath);
                        self->_texturedMeshPipeline.uploadedTexturePath.clear();
                    }
                    if (self->_cloud.hasFaces())
                    {
                        self->_preparedMesh = std::move(result.renderPreparation);
                        self->applyCloudSpatialSummary(result.spatialSummary);
                    }
                    else
                    {
                        self->applyPointPreparation(std::move(result.pointPreparation));
                    }

                    if (!self->_isTiePointCloud
                        && !self->_meshTextureImage.isNull()
                        && self->_preparedMesh.hasTexturedGeometry())
                    {
                        self->setModelColorMode(ModelColorMode::Texture);
                    }
                    else if (!self->_isTiePointCloud && self->_cloud.hasColors())
                    {
                        self->setModelColorMode(ModelColorMode::Shaded);
                    }
                    else if (!self->_isTiePointCloud && self->_cloud.hasFaces())
                    {
                        self->setModelColorMode(ModelColorMode::Solid);
                    }

                    self->_pointCloudPointSize = self->_cloud.size() >= 3'000'000
                        ? 1.1f
                        : self->_cloud.size() >= 1'000'000 ? 1.4f : 2.4f;
                    self->invalidateCache();
                    if (request.fitAfterLoad)
                    {
                        self->fitViewToLoadedGeometry();
                    }
                    const QString format_name = request.format == SceneLoadFormat::Xyz
                        ? QStringLiteral("XYZ")
                        : request.format == SceneLoadFormat::Ply
                            ? QStringLiteral("PLY")
                            : QStringLiteral("OBJ");
                    LOG_INFO(QStringLiteral("[3D] %1 加载完成，共 %2 顶点 / %3 面")
                                 .arg(format_name)
                                 .arg(self->_cloud.size())
                                 .arg(self->_cloud.hasFaces()
                                          ? static_cast<int>(self->_cloud.faces()->rows())
                                          : 0));
                    if (request.format == SceneLoadFormat::Obj
                        && !request.pointCloudResource
                        && !result.textureWarning.isEmpty())
                    {
                        LOG_WARN(QStringLiteral("[3D] %1").arg(result.textureWarning));
                    }
                }
                else
                {
                    const QString error = self->tr("三维数据加载失败或文件为空：%1")
                        .arg(request.path);
                    self->_renderError = error;
                    self->setProperty("lastAsyncTaskError", error);
                    LOG_ERROR(QStringLiteral("[3D] %1").arg(error));
                }
                self->_gpuDirty = true;
                self->update();
            }
            else if (is_current)
            {
                const QString error = self->tr("三维数据加载失败：%1\n%2")
                    .arg(request.path, outcome.errorMessage);
                self->_renderError = error;
                self->setProperty("lastAsyncTaskError", error);
                LOG_ERROR(QStringLiteral("[3D] %1").arg(error));
                self->update();
            }

            outcome.value.reset();
            self->pumpSceneLoad();
        });
}

void CameraSceneWidget::loadTiePointCloudFromFile(const QString &pointCloudPath,
                                                  const QString &sidecarPath)
{
    const QString extension = QFileInfo(pointCloudPath).suffix().toLower();
    if (extension == QLatin1String("ply"))
    {
        loadModelFromPlyInternal(pointCloudPath, true, true, true);
    }
    else if (extension == QLatin1String("obj"))
    {
        loadModelFromObjInternal(pointCloudPath, true, true, true);
    }
    else
    {
        loadPointCloudFromXyzInternal(pointCloudPath, true, true);
    }

    if (_loading)
    {
        _plyLoadProgressText = tr("正在加载连接点...");
    }

    QString metadataPath = sidecarPath.trimmed();
    if (metadataPath.isEmpty())
    {
        metadataPath = xjw::gui::tie_points::inferSidecarPath(pointCloudPath);
    }
    startTiePointMetadataLoad(metadataPath, _loadGen);
    _gpuDirty = true;
    update();
}

void CameraSceneWidget::setTiePointColorMode(TiePointColorMode mode)
{
    if (_tiePointColorMode == mode)
    {
        return;
    }
    _tiePointColorMode = mode;
    update();
    requestOverlayUpdate();
}

void CameraSceneWidget::setModelColorMode(ModelColorMode mode)
{
    if (mode == ModelColorMode::Confidence ||
        mode == ModelColorMode::AssignedImage)
    {
        return;
    }
    if (_modelColorMode == mode)
    {
        return;
    }
    _modelColorMode = mode;
    emit modelColorModeChanged(mode);
    update();
    requestOverlayUpdate();
}

void CameraSceneWidget::startTiePointMetadataLoad(const QString &sidecarPath, int generation)
{
    if (_tiePointMetadataCancellation)
    {
        _tiePointMetadataCancellation->store(true, std::memory_order_relaxed);
    }
    _pendingTiePointMetadataLoad.reset();
    if (sidecarPath.trimmed().isEmpty())
    {
        _tiePointMetadataLoading = false;
        _tiePointMetadataError = tr("无观测数据");
        requestOverlayUpdate();
        return;
    }

    _tiePointMetadataLoading = true;
    _pendingTiePointMetadataLoad = TiePointMetadataRequest{
        sidecarPath,
        generation};
    pumpTiePointMetadataLoad();
}

void CameraSceneWidget::pumpTiePointMetadataLoad()
{
    if (_tiePointMetadataWorkerActive || !_pendingTiePointMetadataLoad)
    {
        return;
    }

    TiePointMetadataRequest request = std::move(*_pendingTiePointMetadataLoad);
    _pendingTiePointMetadataLoad.reset();
    auto cancellation = std::make_shared<std::atomic_bool>(false);
    _tiePointMetadataCancellation = cancellation;
    _tiePointMetadataWorkerActive = true;
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [request, cancellation]()
        {
            TiePointMetadataLoadResult result;
            result.metadata = xjw::gui::tie_points::loadImageCountMetadata(
                request.sidecarPath);
            if (cancellation->load(std::memory_order_relaxed))
            {
                return TiePointMetadataLoadResult{};
            }
            result.scalarData = preparePointScalarData(
                static_cast<std::size_t>(result.metadata.counts.size()),
                result.metadata.counts,
                &result.range,
                cancellation.get());
            return result;
        },
        [request, cancellation](
            CameraSceneWidget *self,
            xjw::gui::tasks::TaskOutcome<TiePointMetadataLoadResult> outcome)
        {
            self->_tiePointMetadataWorkerActive = false;
            const bool is_current = request.generation == self->_loadGen
                && self->_isTiePointCloud
                && cancellation == self->_tiePointMetadataCancellation
                && !cancellation->load(std::memory_order_relaxed);
            if (!is_current)
            {
                outcome.value.reset();
                self->pumpTiePointMetadataLoad();
                return;
            }
            self->_tiePointMetadataLoading = false;
            if (!outcome.succeeded())
            {
                self->_tiePointMetadataError = self->tr("读取观测元数据失败：%1")
                    .arg(outcome.errorMessage);
                self->update();
                self->requestOverlayUpdate();
                self->pumpTiePointMetadataLoad();
                return;
            }

            TiePointMetadataLoadResult result = std::move(*outcome.value);
            const int point_count = static_cast<int>(self->_cloud.size());
            const bool cloud_not_ready = point_count == 0;
            const bool metadata_matches_cloud = !cloud_not_ready
                && result.metadata.counts.size() == static_cast<qsizetype>(point_count)
                && result.scalarData.size()
                    == point_count * static_cast<int>(sizeof(float));
            if (cloud_not_ready || metadata_matches_cloud)
            {
                self->_tiePointImageCounts = std::move(result.metadata.counts);
                self->_tiePointMetadataError = result.metadata.errorMessage;
                self->_tiePointScalarData = std::move(result.scalarData);
                self->_tiePointImageCountRange = result.range;
            }
            if (metadata_matches_cloud)
            {
                self->_pointScalarBuffer.vertexData = self->_tiePointScalarData;
                self->_pointScalarBuffer.vertexCount = point_count;
                self->_pointScalarBuffer.strideBytes = int(sizeof(float));
                self->_pointScalarBuffer.dirty = true;
            }
            else if (!cloud_not_ready)
            {
                self->_tiePointImageCounts.clear();
                self->_tiePointScalarData.clear();
                self->_tiePointImageCountRange = {};
                self->_tiePointMetadataError = result.metadata.errorMessage.isEmpty()
                    ? self->tr("观测元数据点数与当前点云不一致，已忽略过期数据。")
                    : result.metadata.errorMessage;
            }
            self->update();
            self->requestOverlayUpdate();
            self->pumpTiePointMetadataLoad();
        });
}

void CameraSceneWidget::fitViewToLoadedGeometry()
{
    if (_cloud.size() == 0)
    {
        _hasFocusedGeometryBounds = false;
        return;
    }

    if (!_cloudSpatialSummary.valid
        || _cloudSpatialSummary.sourcePointCount != _cloud.size())
    {
        _hasFocusedGeometryBounds = false;
        return;
    }

    _focusedGeometryCenter = _cloudSpatialSummary.center;
    _focusedGeometryRadius = _cloudSpatialSummary.p95Radius;
    _hasFocusedGeometryBounds = true;
    _zoomScale = 1.0;
    _sceneOffsetPx = QPointF();
    _hoverAxis = HoverAxis::None;
    _dragAxis = HoverAxis::None;
    LOG_INFO(QStringLiteral("[3D] 已聚焦加载几何：中心 (%1, %2, %3)，半径 %4")
                 .arg(_focusedGeometryCenter.x(), 0, 'g', 6)
                 .arg(_focusedGeometryCenter.y(), 0, 'g', 6)
                 .arg(_focusedGeometryCenter.z(), 0, 'g', 6)
                 .arg(_focusedGeometryRadius, 0, 'g', 6));
    updateCameraOverlay();
}

// 计算场景中所有点（相机光心、点云、模型顶点）的质心作为场景中心。
// 使用缓存，仅在数据变更后重新计算。
QVector3D CameraSceneWidget::sceneCenter() const
{
    if (_hasFocusedGeometryBounds)
    {
        return _focusedGeometryCenter;
    }
    if (_cacheDirty) invalidateCache();
    return _cachedCenter;
}

// 计算场景中所有点到质心的最大距离，用于自适应相机距离、投影远裁平面等。
// 使用缓存，仅在数据变更后重新计算。
float CameraSceneWidget::sceneRadius() const
{
    if (_hasFocusedGeometryBounds)
    {
        return _focusedGeometryRadius;
    }
    if (_cacheDirty) invalidateCache();
    return _cachedRadius;
}

CameraSceneWidget::SceneMatrices CameraSceneWidget::sceneMatrices() const
{
    const QVector3D center = sceneCenter();
    const float radius = sceneRadius();
    const float distance = static_cast<float>(
        static_cast<double>(radius) * 3.2 / _zoomScale);
    const float nearPlane = distance * 0.001f;
    const float farPlane  = qMax(1000.0f, distance * 100.0f + radius * 50.0f);

    SceneMatrices matrices;
    // 从 +Z 方向看向原点：世界X→屏幕右，世界Y→屏幕上，与overlay/arcball坐标系一致
    const QVector3D eye = center + QVector3D(0.0f, 0.0f, distance);
    QMatrix4x4 view;
    view.lookAt(eye, center, QVector3D(0.0f, 1.0f, 0.0f));
    QMatrix4x4 model;
    model.setToIdentity();
    model.translate(center);
    model.rotate(_viewRot);
    model.translate(-center);
    matrices.modelView = view * model;

    const float aspect = qMax(1.0f, float(width()) / qMax(1, height()));
    matrices.projection.perspective(45.0f, aspect, nearPlane, farPlane);
    return matrices;
}

QPointF CameraSceneWidget::projectToScreen(const QVector3D &p, bool *ok) const
{
    const SceneMatrices matrices = sceneMatrices();
    QVector4D clip = matrices.projection * matrices.modelView * QVector4D(p, 1.0f);
    if (clip.w() <= 1e-6f) {
        if (ok) *ok = false;
        return QPointF();
    }
    QVector3D ndc(clip.x() / clip.w(), clip.y() / clip.w(), clip.z() / clip.w());
    if (ok) *ok = true;
    const float sx = (ndc.x() * 0.5f + 0.5f) * width() + float(_sceneOffsetPx.x());
    const float sy = (1.0f - (ndc.y() * 0.5f + 0.5f)) * height() + float(_sceneOffsetPx.y());
    return QPointF(sx, sy);
}

// 将向量从副本局部空间旋转到当前视图空间（应用 _viewRot）
QVector3D CameraSceneWidget::applyViewRotation(const QVector3D &v) const
{
    return _viewRot.rotatedVector(v);
}

// 返回当前视图四元数对应的欧拉角（度， x=pitch, y=yaw, z=roll）
QVector3D CameraSceneWidget::eulerAnglesDeg() const
{
    return _viewRot.toEulerAngles();
}

// 根据窗口尺寸自适应计算 Gizmo 操控球屏幕半径（px）
// 范围：[42, min(w,h)*0.30]，基准为 min(w,h)*0.16
qreal CameraSceneWidget::manipRadiusPx() const
{
    const qreal base = qMin(width(), height()) * 0.16;
    return qBound<qreal>(42.0, base, qMin(width(), height()) * 0.30);
}

int CameraSceneWidget::maxVisibleCameraLabels() const
{
    const int viewportBudget = qBound(8, width() / 140, 28);
    if (_poses.size() > 300)
    {
        return qMin(viewportBudget, 12);
    }
    if (_poses.size() > 120)
    {
        return qMin(viewportBudget, 18);
    }
    if (_poses.size() > 60)
    {
        return qMin(viewportBudget, 24);
    }
    return qMin(viewportBudget, 40);
}

float CameraSceneWidget::cameraImagePlaneHalfExtent(
    const CameraPose &pose,
    const QMatrix4x4 &worldToView) const
{
    if (_cacheDirty) invalidateCache();
    const float screenScaledExtent =
        xjw::gui::camera_scene::cameraPlaneHalfExtentForScreenSize(
            pose.center,
            worldToView,
            height(),
            _zoomScale,
            45.0f,
            34.0);
    if (screenScaledExtent > 0.0f)
    {
        return screenScaledExtent;
    }

    const float radius = sceneRadius();
    return qMax(1.0e-5f, radius * 0.065f);
}

bool CameraSceneWidget::isCameraHighlighted(const CameraPose &pose) const
{
    if (!_highlightedCameraPath.isEmpty())
    {
        if (normalizedCameraPath(pose.imagePath) == _highlightedCameraPath)
        {
            return true;
        }

        const QString highlightedFileName = QFileInfo(_highlightedCameraPath).fileName();
        if (!highlightedFileName.isEmpty())
        {
            return pose.name == highlightedFileName
                || QFileInfo(pose.imagePath).fileName() == highlightedFileName;
        }
    }

    return false;
}

QString CameraSceneWidget::normalizedCameraPath(const QString &imagePath) const
{
    if (imagePath.isEmpty())
    {
        return QString();
    }

    return QDir::cleanPath(QFileInfo(imagePath).absoluteFilePath());
}

QString CameraSceneWidget::cameraPlaneImageKey(const QString &imagePath, CameraImagePlaneMode mode) const
{
    const QString normalizedPath = normalizedCameraPath(imagePath);
    if (normalizedPath.isEmpty())
    {
        return QString();
    }

    QString modeKey;
    switch (mode)
    {
    case CameraImagePlaneMode::Image:
        modeKey = QStringLiteral("image");
        break;
    case CameraImagePlaneMode::Thumbnail:
        modeKey = QStringLiteral("thumb");
        break;
    case CameraImagePlaneMode::Solid:
        modeKey = QStringLiteral("solid");
        break;
    }
    return modeKey + QLatin1Char('|') + normalizedPath;
}

QImage CameraSceneWidget::cachedCameraPlaneImage(const QString &imagePath, CameraImagePlaneMode mode) const
{
    if (mode == CameraImagePlaneMode::Solid)
    {
        return QImage();
    }

    return _cameraImageCache.value(cameraPlaneImageKey(imagePath, mode));
}

void CameraSceneWidget::discardQueuedCameraThumbnails()
{
    QQueue<CameraPlaneImageRequest> retained_requests;
    while (!_cameraImageLoadQueue.isEmpty())
    {
        CameraPlaneImageRequest request = _cameraImageLoadQueue.dequeue();
        const QString key = cameraPlaneImageKey(request.path, request.mode);
        if (request.mode == CameraImagePlaneMode::Thumbnail)
        {
            _cameraImageLoadsQueued.remove(key);
        }
        else
        {
            retained_requests.enqueue(std::move(request));
        }
    }
    _cameraImageLoadQueue = std::move(retained_requests);

    const QString thumbnail_prefix = QStringLiteral("thumb|");
    for (auto it = _cameraImageCache.begin(); it != _cameraImageCache.end();)
    {
        if (it.key().startsWith(thumbnail_prefix))
        {
            it = _cameraImageCache.erase(it);
        }
        else
        {
            ++it;
        }
    }
    for (auto it = _cameraImageLoadFailures.begin();
         it != _cameraImageLoadFailures.end();)
    {
        if (it->startsWith(thumbnail_prefix))
        {
            it = _cameraImageLoadFailures.erase(it);
        }
        else
        {
            ++it;
        }
    }
    _pendingThumbnailPoseIndices.clear();
    _cameraThumbnailLoadCompleted = 0;
}

void CameraSceneWidget::requestCameraPlaneImage(const QString &imagePath, CameraImagePlaneMode mode)
{
    if (mode == CameraImagePlaneMode::Solid)
    {
        return;
    }
    if (mode == CameraImagePlaneMode::Thumbnail
        && (!_showCameras || !_showCameraThumbnails))
    {
        return;
    }

    const QString key = cameraPlaneImageKey(imagePath, mode);
    if (key.isEmpty() || _cameraImageCache.contains(key) || _cameraImageLoadsInFlight.contains(key)
        || _cameraImageLoadsQueued.contains(key)
        || _cameraImageLoadFailures.contains(key))
    {
        return;
    }

    CameraPlaneImageRequest request;
    request.path = imagePath;
    request.mode = mode;
    request.generation = _cameraImageLoadGeneration;
    if (mode == CameraImagePlaneMode::Image)
    {
        _cameraImageLoadQueue.prepend(request);
    }
    else
    {
        _cameraImageLoadQueue.enqueue(request);
    }
    _cameraImageLoadsQueued.insert(key);
    pumpCameraPlaneImageLoads();
}

void CameraSceneWidget::pumpCameraPlaneImageLoads()
{
    const int maximum_loads = _cameraImageLoadPool.maxThreadCount();
    while (_cameraImageLoadsInFlight.size() < maximum_loads
           && !_cameraImageLoadQueue.isEmpty())
    {
        const CameraPlaneImageRequest request = _cameraImageLoadQueue.dequeue();
        const QString key = cameraPlaneImageKey(request.path, request.mode);
        _cameraImageLoadsQueued.remove(key);
        if (request.generation != _cameraImageLoadGeneration || key.isEmpty()
            || (request.mode == CameraImagePlaneMode::Thumbnail
                && (!_showCameras || !_showCameraThumbnails))
            || _cameraImageCache.contains(key) || _cameraImageLoadsInFlight.contains(key)
            || _cameraImageLoadFailures.contains(key))
        {
            continue;
        }

        _cameraImageLoadsInFlight.insert(key);
        auto *watcher = new QFutureWatcher<CameraPlaneImageResult>(this);
        connect(watcher,
                &QFutureWatcher<CameraPlaneImageResult>::finished,
                this,
                [this, watcher, key]()
        {
            const CameraPlaneImageResult result = watcher->result();
            _cameraImageLoadsInFlight.remove(key);
            applyCameraPlaneImage(result);

            QString current_request_path;
            if (result.mode == CameraImagePlaneMode::Thumbnail
                && _showCameras && _showCameraThumbnails)
            {
                const int pose_index = _poseIndexByNormalizedPath.value(
                    normalizedCameraPath(result.path), -1);
                if (pose_index >= 0 && pose_index < _poses.size())
                {
                    current_request_path = _poses.at(pose_index).imagePath;
                }
            }
            else if (result.mode == CameraImagePlaneMode::Image && _showCameraImage)
            {
                const int pose_index = displayedCameraImagePoseIndex();
                if (pose_index >= 0 && pose_index < _poses.size()
                    && normalizedCameraPath(_poses.at(pose_index).imagePath)
                        == normalizedCameraPath(result.path))
                {
                    current_request_path = _poses.at(pose_index).imagePath;
                }
            }
            if (!current_request_path.isEmpty()
                && cachedCameraPlaneImage(current_request_path, result.mode).isNull())
            {
                requestCameraPlaneImage(current_request_path, result.mode);
            }
            watcher->deleteLater();
            pumpCameraPlaneImageLoads();
        });
        watcher->setFuture(QtConcurrent::run(
            &_cameraImageLoadPool,
            &CameraSceneWidget::loadCameraPlaneImage,
            request.path,
            request.mode,
            request.generation));
    }
}

void CameraSceneWidget::applyCameraPlaneImage(const CameraPlaneImageResult &result)
{
    if (result.generation != _cameraImageLoadGeneration)
    {
        return;
    }
    if (result.mode == CameraImagePlaneMode::Thumbnail
        && (!_showCameras || !_showCameraThumbnails))
    {
        return;
    }

    const QString key = cameraPlaneImageKey(result.path, result.mode);
    const QString result_path = normalizedCameraPath(result.path);
    const int pose_index = _poseIndexByNormalizedPath.value(result_path, -1);
    bool thumbnail_was_uploaded = false;
    if (result.mode == CameraImagePlaneMode::Thumbnail && pose_index >= 0)
    {
        for (const auto &page : std::as_const(_thumbnailPipeline.atlasPages))
        {
            thumbnail_was_uploaded = thumbnail_was_uploaded
                || (page && page->uploadedPoseIndices.contains(pose_index));
        }
    }
    if (!result.loaded || result.image.isNull())
    {
        if (!key.isEmpty())
        {
            const bool first_failure = !_cameraImageLoadFailures.contains(key);
            _cameraImageLoadFailures.insert(key);
            if (first_failure
                && result.mode == CameraImagePlaneMode::Thumbnail
                && !thumbnail_was_uploaded)
            {
                ++_cameraThumbnailLoadCompleted;
            }
        }
        if (!result.errorMessage.isEmpty())
        {
            LOG_WARN("%s", qUtf8Printable(result.errorMessage));
        }
        return;
    }

    if (key.isEmpty())
    {
        return;
    }
    const bool was_failed = _cameraImageLoadFailures.remove(key);

    if (result.mode == CameraImagePlaneMode::Image)
    {
        // 图片可能在 sample count 改变后的空窗期才异步到达；确保旧管线
        // 不会因纹理尺寸相同而绕过兼容性重建。
        _imagePipeline.pipelineDirty = true;
        if (_imagePipeline.uploadedImageKey == key)
        {
            _imagePipeline.uploadedImageKey.clear();
        }
        const auto existing = _cameraImageCache.constFind(key);
        if (existing != _cameraImageCache.cend())
        {
            _fullImageCacheBytes -= static_cast<qint64>(existing.value().sizeInBytes());
        }
        _fullImageCacheLru.removeAll(key);
        _fullImageCacheLru.enqueue(key);
        _fullImageCacheBytes += static_cast<qint64>(result.image.sizeInBytes());
    }
    _cameraImageCache.insert(key, result.image);
    if (result.mode == CameraImagePlaneMode::Image)
    {
        QString active_image_key;
        const int active_pose_index = displayedCameraImagePoseIndex();
        if (active_pose_index >= 0 && active_pose_index < _poses.size())
        {
            active_image_key = cameraPlaneImageKey(
                _poses.at(active_pose_index).imagePath,
                CameraImagePlaneMode::Image);
        }
        int remaining_candidates = _fullImageCacheLru.size();
        while (_fullImageCacheBytes > kFullCameraImageCacheLimitBytes
               && _fullImageCacheLru.size() > 1
               && remaining_candidates-- > 0)
        {
            const QString candidate = _fullImageCacheLru.dequeue();
            if (candidate == key || candidate == active_image_key)
            {
                _fullImageCacheLru.enqueue(candidate);
                continue;
            }
            const auto cached = _cameraImageCache.find(candidate);
            if (cached != _cameraImageCache.end())
            {
                _fullImageCacheBytes -= static_cast<qint64>(cached.value().sizeInBytes());
                _cameraImageCache.erase(cached);
            }
        }
    }
    if (result.mode == CameraImagePlaneMode::Thumbnail && pose_index >= 0)
    {
        for (const auto &page : std::as_const(_thumbnailPipeline.atlasPages))
        {
            if (page)
            {
                page->uploadedPoseIndices.remove(pose_index);
                page->imageSizes.remove(pose_index);
            }
        }
        if ((was_failed || thumbnail_was_uploaded)
            && _cameraThumbnailLoadCompleted > 0)
        {
            --_cameraThumbnailLoadCompleted;
        }
        _pendingThumbnailPoseIndices.insert(pose_index);
        _thumbnailPipeline.instancesDirty = true;
    }
    bool image_dimensions_changed = false;
    if (result.originalSize.isValid())
    {
        if (pose_index >= 0 && pose_index < _poses.size())
        {
            CameraPose &pose = _poses[pose_index];
            if (pose.imageWidth <= 0)
            {
                pose.imageWidth = result.originalSize.width();
                image_dimensions_changed = true;
            }
            if (pose.imageHeight <= 0)
            {
                pose.imageHeight = result.originalSize.height();
                image_dimensions_changed = true;
            }
        }
    }
    if (result.mode == CameraImagePlaneMode::Image
        || (image_dimensions_changed && pose_index == displayedCameraImagePoseIndex()))
    {
        _imagePipeline.geometryDirty = true;
    }
    if (result.mode == CameraImagePlaneMode::Thumbnail)
    {
        if (!_thumbnailUpdateScheduled)
        {
            _thumbnailUpdateScheduled = true;
            QTimer::singleShot(16, this, [this]()
            {
                _thumbnailUpdateScheduled = false;
                update();
            });
        }
    }
    else
    {
        update();
    }
}

CameraSceneWidget::CameraPlaneImageResult CameraSceneWidget::loadCameraPlaneImage(const QString &imagePath,
                                                                                  CameraImagePlaneMode mode,
                                                                                  int generation)
{
    CameraPlaneImageResult result;
    result.path = imagePath;
    result.mode = mode;
    result.generation = generation;

    try
    {
        const QSize targetSize = mode == CameraImagePlaneMode::Image
            ? QSize(2048, 2048)
            : QSize(kCameraThumbnailWidth, kCameraThumbnailHeight);
        QSize source_size;
        QImage image = xjw::gui::views::loadImageForDisplay(
            imagePath, QString(), targetSize, &source_size);
        if (image.isNull())
        {
            result.errorMessage = QStringLiteral("三维视图无法读取照片：%1")
                .arg(imagePath);
            return result;
        }

        result.originalSize = source_size.isValid() ? source_size : image.size();
        result.image = image.convertToFormat(QImage::Format_RGBX8888);
        result.loaded = !result.image.isNull();
    }
    catch (const std::exception &exception)
    {
        result.errorMessage = QStringLiteral("三维视图读取照片失败：%1（%2）")
            .arg(imagePath, QString::fromUtf8(exception.what()));
    }
    catch (...)
    {
        result.errorMessage = QStringLiteral("三维视图读取照片失败：%1（未知错误）")
            .arg(imagePath);
    }
    return result;
}

void CameraSceneWidget::updateActiveCameraForView()
{
    if (_cameraImageLocked || _poses.isEmpty())
    {
        return;
    }

    const int selected_pose_index = xjw::gui::camera_scene::selectCameraForView(
        _cameraViewCandidates,
        xjw::gui::camera_scene::currentWorldViewDirection(_viewRot),
        sceneCenter());
    if (_activeCameraImagePoseIndex != selected_pose_index)
    {
        _activeCameraImagePoseIndex = selected_pose_index;
        _imagePipeline.geometryDirty = true;
    }
}

int CameraSceneWidget::displayedCameraImagePoseIndex() const
{
    if (_poses.isEmpty())
    {
        return -1;
    }

    if (_cameraImageLocked)
    {
        if (_lockedCameraImagePoseIndex >= 0
            && _lockedCameraImagePoseIndex < _poses.size())
        {
            return _lockedCameraImagePoseIndex;
        }
    }

    if (_activeCameraImagePoseIndex >= 0 && _activeCameraImagePoseIndex < _poses.size())
    {
        return _activeCameraImagePoseIndex;
    }
    return -1;
}

void CameraSceneWidget::refreshLockedCameraImage()
{
    const int poseIndex = displayedCameraImagePoseIndex();
    if (poseIndex < 0 || poseIndex >= _poses.size())
    {
        _lockedCameraImagePoseIndex = -1;
        _lockedCameraImagePath.clear();
        _lockedCameraImageName.clear();
        return;
    }

    const CameraPose &pose = _poses.at(poseIndex);
    const QString previous_path = _lockedCameraImagePath;
    _lockedCameraImagePath = pose.imagePath;
    _lockedCameraImageName = pose.name;
    _lockedCameraImagePoseIndex = poseIndex;
    if (normalizedCameraPath(previous_path) != normalizedCameraPath(_lockedCameraImagePath))
    {
        _imagePipeline.geometryDirty = true;
    }
}

void CameraSceneWidget::drawFloorPivotCross(QPainter &painter) const
{
    if (_cacheDirty) invalidateCache();

    const QVector3D minimum = _hasCloudBounds ? _cachedCloudAABBMin : _cachedAABBMin;
    const QVector3D maximum = _hasCloudBounds ? _cachedCloudAABBMax : _cachedAABBMax;
    const QVector3D floorPivot((minimum.x() + maximum.x()) * 0.5f,
                               (minimum.y() + maximum.y()) * 0.5f,
                               minimum.z());

    bool okCenter = false;
    const QPointF c = projectToScreen(floorPivot, &okCenter);
    if (!okCenter)
    {
        return;
    }

    constexpr qreal half_size_pixels = 7.0;
    painter.setBrush(Qt::NoBrush);
    painter.setPen(QPen(QColor(105, 108, 112, 170), 1.2));
    painter.drawLine(c + QPointF(-half_size_pixels, 0.0),
                     c + QPointF(half_size_pixels, 0.0));
    painter.drawLine(c + QPointF(0.0, -half_size_pixels),
                     c + QPointF(0.0, half_size_pixels));
}

bool CameraSceneWidget::cameraDirectionLeaderSegment(const CameraPose &pose,
                                                     float planeHalfExtent,
                                                     QVector3D *start,
                                                     QVector3D *end) const
{
    const QVector3D forward = xjw::gui::camera_scene::cameraForwardDirection(
        pose.rotation, pose.depthAxisFlipped);
    if (!start || !end || forward.isNull() || planeHalfExtent <= 0.0f)
    {
        return false;
    }

    // 方位线与相机卡片共享同一个三维锚点和缩放尺度，并交给 Vulkan
    // 使用相同 MVP 变换。卡片随后写入深度缓冲并遮住中心部分，留下从
    // 照片平面边缘自然延伸的反向光轴，避免二维覆盖层缩放后产生脱节。
    *start = pose.center;
    *end = pose.center - forward.normalized() * planeHalfExtent * 1.35f;
    return true;
}

// Gizmo 操控球的屏幕中心点（固定为窗口中心）
QPointF CameraSceneWidget::manipCenterScreen() const
{
    return QPointF(width() * 0.5, height() * 0.5);
}

// 根据鼠标位置检测鼠标当前悬停的 Gizmo 展向轴环。
// 原理：遍历 X/Y/Z 三个轴环的散列点，计算鼠标到环上最近线段的距离，
// 距离小于阈値 12px 时判定为悬停在该轴环上。
CameraSceneWidget::HoverAxis CameraSceneWidget::pickHoverAxis(const QPoint &mousePos) const
{
    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // 计算鼠标到指定轴环的最近距离（只考虑正面可见的弧段）
    auto minDistToCircle = [&](HoverAxis axis) {
        qreal best = 1e9;
        QPointF prev;
        bool hasPrev = false;
        bool prevVisible = false;
        for (int i = 0; i <= 96; ++i) {
            const qreal t = (2.0 * M_PI * i) / 96.0;
            QVector3D pLocal;
            // 根据轴选择环面上的局部点
            if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
            else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
            else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
            QVector3D pView = applyViewRotation(pLocal);
            const bool currVisible = (pView.z() > 0.0f); // z>0 表示正面对观察者
            QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
            if (hasPrev && prevVisible && currVisible) {
                // 计算鼠标到线段 [prev, curr] 的最近距离
                const QPointF ab = curr - prev;
                const qreal ab2 = ab.x() * ab.x() + ab.y() * ab.y();
                if (ab2 > 1e-6) {
                    qreal u = ((mousePos.x() - prev.x()) * ab.x() + (mousePos.y() - prev.y()) * ab.y()) / ab2;
                    u = qBound<qreal>(0.0, u, 1.0);
                    QPointF proj = prev + ab * u;
                    best = qMin(best, QLineF(QPointF(mousePos), proj).length());
                }
            }
            prev = curr;
            prevVisible = currVisible;
            hasPrev = true;
        }
        return best;
    };

    const qreal th = 12.0; // 距离阈値（像素）
    const qreal dx = minDistToCircle(HoverAxis::X);
    const qreal dy = minDistToCircle(HoverAxis::Y);
    const qreal dz = minDistToCircle(HoverAxis::Z);
    const qreal dmin = qMin(dx, qMin(dy, dz));
    if (dmin > th) return HoverAxis::None; // 距离过远，无悬停
    // 返回距离最小的轴
    if (dx <= dy && dx <= dz) return HoverAxis::X;
    if (dy <= dx && dy <= dz) return HoverAxis::Y;
    return HoverAxis::Z;
}

// 计算鼠标附近指定轴环的切线方向（屏幕空间单位向量）。
// 用于将鼠标拖拽距离投影到切线方向以计算旋转角度。
QVector2D CameraSceneWidget::pickAxisTangent(const QPoint &mousePos, HoverAxis axis) const
{
    if (axis == HoverAxis::None) return QVector2D(1.0f, 0.0f);
    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();

    // 遍历环面散列点，找到鼠标最近的线段 [bestA, bestB]
    QPointF bestA;
    QPointF bestB;
    qreal bestDist = 1e12;
    QPointF prev;
    bool hasPrev = false;
    bool prevVisible = false;
    for (int i = 0; i <= 128; ++i)
    {
        const qreal t = (2.0 * M_PI * i) / 128.0;
        QVector3D pLocal;
        if (axis == HoverAxis::X) pLocal = QVector3D(0.0f, float(std::cos(t)), float(std::sin(t)));
        else if (axis == HoverAxis::Y) pLocal = QVector3D(float(std::cos(t)), 0.0f, float(std::sin(t)));
        else pLocal = QVector3D(float(std::cos(t)), float(std::sin(t)), 0.0f);
        const QVector3D pView = applyViewRotation(pLocal);
        const bool currVisible = (pView.z() > 0.0f);
        const QPointF curr = center2d + QPointF(pView.x() * radiusPx, -pView.y() * radiusPx);
        if (hasPrev && prevVisible && currVisible)
        {
            const QPointF ab = curr - prev;
            const qreal ab2 = ab.x() * ab.x() + ab.y() * ab.y();
            if (ab2 > 1e-9)
            {
                qreal u = ((mousePos.x() - prev.x()) * ab.x() + (mousePos.y() - prev.y()) * ab.y()) / ab2;
                u = qBound<qreal>(0.0, u, 1.0);
                const QPointF proj = prev + ab * u;
                const qreal d = QLineF(QPointF(mousePos), proj).length();
                if (d < bestDist)
                {
                    bestDist = d;
                    bestA = prev;  // 最近线段起点
                    bestB = curr;  // 最近线段终点
                }
            }
        }
        prev = curr;
        prevVisible = currVisible;
        hasPrev = true;
    }

    // 计算并归一化切线方向向量
    QVector2D tanDir(float(bestB.x() - bestA.x()), float(bestB.y() - bestA.y()));
    if (tanDir.lengthSquared() < 1e-8f) tanDir = QVector2D(1.0f, 0.0f); // 防止除以零
    return tanDir.normalized();
}
// ---------------------------------------------------------------------------
// Arcball 球面投影
//   屏幕投影规则： pView.x → 屏幕右， pView.y → 屏幕上（y已翻转）， pView.z → 朝向观察者
//   注：与 drawGreatCircle 里 cursor2d + (pView.x*r, -pView.y*r) 保持一致
// ---------------------------------------------------------------------------
QVector3D CameraSceneWidget::arcballVector(const QPoint &mousePos) const
{
    const QPointF c = manipCenterScreen();
    const float r = float(manipRadiusPx());
    if (r < 1.0f) return QVector3D(0.0f, 0.0f, 1.0f);

    const float x =  float(mousePos.x() - c.x()) / r;
    const float y = -float(mousePos.y() - c.y()) / r;  // 屏幕 Y 选转为那数学 Y
    const float len2 = x * x + y * y;
    if (len2 <= 1.0f) {
        return QVector3D(x, y, std::sqrt(1.0f - len2));   // 在球面上
    }
    // 在球外：投影到赤道圆
    const float len = std::sqrt(len2);
    return QVector3D(x / len, y / len, 0.0f);
}

// 平移偏移量无限制（允许将模型拖出视口外）
void CameraSceneWidget::clampSceneOffset()
{
    // 不做任何限制
}

// 根据当前的悬停轴/拖拽状态更新鼠标光标样式：
//   - 中键拖拽中        → 四向移动光标
//   - 左键自由旋转中    → 闭合手型光标
//   - 悬停/拖拽 X/Y/Z 环 → 带颜色和字母的自定义圆形光标
//   - 默认              → 开放手型光标
void CameraSceneWidget::updateCursor()
{
    if (_manualPruneMode && _manualSelecting)
    {
        setCursor(Qt::CrossCursor);
        return;
    }

    // 创建带颜色和轴标签的自定义光标（24x24 像素，圆心热点）
    auto axisCursor = [](const QColor &color, const QString &label) {
        QPixmap pm(24, 24);
        pm.fill(Qt::transparent);
        QPainter p(&pm);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(color, 2));
        p.drawEllipse(QPointF(12, 12), 8, 8);
        p.drawText(QRect(0, 0, 24, 24), Qt::AlignCenter, label);
        return QCursor(pm, 12, 12);
    };

    if (_middleDragging) {
        setCursor(Qt::SizeAllCursor); // 中键平移中
        return;
    }
    if (_leftDragging && _dragAxis == HoverAxis::None) {
        setCursor(Qt::ClosedHandCursor); // Arcball 自由旋转中
        return;
    }

    // 优先显示当前拖拽轴（若无则显示悬停轴）
    HoverAxis axis = (_leftDragging ? _dragAxis : _hoverAxis);
    if (axis == HoverAxis::X) {
        setCursor(axisCursor(QColor(255, 120, 120), QStringLiteral("X")));
    } else if (axis == HoverAxis::Y) {
        setCursor(axisCursor(QColor(120, 255, 120), QStringLiteral("Y")));
    } else if (axis == HoverAxis::Z) {
        setCursor(axisCursor(QColor(120, 180, 255), QStringLiteral("Z")));
    } else {
        setCursor(Qt::OpenHandCursor); // 空闲时显示开放手型
    }
}

void CameraSceneWidget::initialize(QRhiCommandBuffer *cb)
{
    Q_UNUSED(cb);

    _renderError.clear();
    _rhiReady = rhi() && api() == QRhiWidget::Api::Vulkan;
    if (!_rhiReady)
    {
        _renderError = QStringLiteral("Vulkan 渲染初始化失败，请检查显卡驱动和 Qt Vulkan 支持。");
        LOG_ERROR("%s", qPrintable(_renderError));
        return;
    }

    _colorPointPipeline.vertexShaderPath = QStringLiteral(":/shaders/camera_scene_point.vert.qsb");
    _colorPointPipeline.fragmentShaderPath = QStringLiteral(":/shaders/camera_scene_point.frag.qsb");
    _highlightPointPipeline.vertexShaderPath = _colorPointPipeline.vertexShaderPath;
    _highlightPointPipeline.fragmentShaderPath = _colorPointPipeline.fragmentShaderPath;
    _colorLinePipeline.vertexShaderPath = QStringLiteral(":/shaders/camera_scene_color.vert.qsb");
    _colorLinePipeline.fragmentShaderPath = QStringLiteral(":/shaders/camera_scene_color.frag.qsb");
    _meshTrianglePipeline.vertexShaderPath = QStringLiteral(":/shaders/camera_scene_mesh.vert.qsb");
    _meshTrianglePipeline.fragmentShaderPath = QStringLiteral(":/shaders/camera_scene_mesh.frag.qsb");
    _meshWireframePipeline.vertexShaderPath = _meshTrianglePipeline.vertexShaderPath;
    _meshWireframePipeline.fragmentShaderPath = _meshTrianglePipeline.fragmentShaderPath;
    _texturedMeshPipeline.vertexShaderPath =
        QStringLiteral(":/shaders/camera_scene_textured_mesh.vert.qsb");
    _texturedMeshPipeline.fragmentShaderPath =
        QStringLiteral(":/shaders/camera_scene_textured_mesh.frag.qsb");

    _gpuDirty = true;
    _pipelinesDirty = true;
    _imagePipeline.pipelineDirty = true;
    _thumbnailPipeline.pipelinesDirty = true;
}

void CameraSceneWidget::releaseResources()
{
    releaseGeometryBufferResources();
    releasePipelineResources(&_colorPointPipeline);
    releasePipelineResources(&_highlightPointPipeline);
    releasePipelineResources(&_colorLinePipeline);
    releasePipelineResources(&_meshTrianglePipeline);
    releasePipelineResources(&_meshWireframePipeline);
    releaseTexturedMeshPipelineResources();
    releaseImagePipelineResources();
    releaseCameraThumbnailPipelineResources();
    _thumbnailPipeline.resourcesDirty = true;
    _thumbnailPipeline.instancesDirty = true;
    _thumbnailPipeline.pipelinesDirty = true;
    _imagePipeline.pipelineDirty = true;
    _rhiReady = false;
    _gpuDirty = true;
    _pipelinesDirty = true;
}

void CameraSceneWidget::releaseGeometryBufferResources()
{
    _pointBuffer.vertexBuffer.reset();
    _pointScalarBuffer.vertexBuffer.reset();
    _manualHighlightPointBuffer.vertexBuffer.reset();
    _manualHighlightScalarBuffer.vertexBuffer.reset();
    _manualHighlightBuffersReleasePending = false;
    _meshBuffer.vertexBuffer.reset();
    _texturedMeshBuffer.vertexBuffer.reset();
    _meshTriangleIndices.indexBuffer.reset();
    _meshWireframeIndices.indexBuffer.reset();
    _lineBuffer.vertexBuffer.reset();
}

bool CameraSceneWidget::failGeometryUpload(const QString &message)
{
    _geometryUploadError = message;
    _renderError = message;
    releaseGeometryBufferResources();
    _gpuDirty = false;
    return false;
}

void CameraSceneWidget::releasePipelineResources(RhiPipelineSet *pipeline)
{
    if (!pipeline)
    {
        return;
    }
    pipeline->pipeline.reset();
    pipeline->bindings.reset();
    pipeline->uniformBuffer.reset();
}

void CameraSceneWidget::releaseTexturedMeshPipelineResources()
{
    _texturedMeshPipeline.pipeline.reset();
    _texturedMeshPipeline.bindings.reset();
    _texturedMeshPipeline.uniformBuffer.reset();
    _texturedMeshPipeline.texture.reset();
    _texturedMeshPipeline.sampler.reset();
    _texturedMeshPipeline.textureSize = QSize();
    _texturedMeshPipeline.uploadedTexturePath.clear();
}

void CameraSceneWidget::releaseImagePipelineResources()
{
    _imagePipeline.pipeline.reset();
    _imagePipeline.bindings.reset();
    _imagePipeline.vertexBuffer.reset();
    _imagePipeline.uniformBuffer.reset();
    _imagePipeline.texture.reset();
    _imagePipeline.sampler.reset();
    _imagePipeline.textureSize = QSize();
    _imagePipeline.uploadedImageKey.clear();
    _imagePipeline.uploadedGeometryKey.clear();
    _imagePipeline.planeCorners.clear();
    _imagePipeline.geometryDirty = true;
    _imagePipeline.pipelineDirty = true;
}

void CameraSceneWidget::releaseCameraThumbnailPipelineResources()
{
    _thumbnailPipeline.pipeline.reset();
    _thumbnailPipeline.leaderPipeline.reset();
    _thumbnailPipeline.leaderBindings.reset();
    _thumbnailPipeline.atlasPages.clear();
    _thumbnailPipeline.solidResource.clear();
    _thumbnailPipeline.highlightedSolidResource.clear();
    _thumbnailPipeline.leaderInstanceBuffer.reset();
    _thumbnailPipeline.uniformBuffer.reset();
    _thumbnailPipeline.sampler.reset();
    _thumbnailPipeline.atlasSize = 0;
    _thumbnailPipeline.leaderInstanceCapacity = 0;
    _thumbnailPipeline.leaderInstanceCount = 0;
    _thumbnailPipeline.segmentInstanceCount = 0;
}

void CameraSceneWidget::rollbackResourceUpdateState()
{
    auto mark_buffer_dirty = [](RhiBufferSet *buffer)
    {
        if (buffer && !buffer->vertexData.isEmpty() && buffer->vertexCount > 0)
        {
            buffer->dirty = true;
        }
    };
    mark_buffer_dirty(&_pointBuffer);
    mark_buffer_dirty(&_pointScalarBuffer);
    mark_buffer_dirty(&_manualHighlightPointBuffer);
    mark_buffer_dirty(&_manualHighlightScalarBuffer);
    mark_buffer_dirty(&_meshBuffer);
    mark_buffer_dirty(&_texturedMeshBuffer);
    mark_buffer_dirty(&_lineBuffer);
    if (!_meshTriangleIndices.indexData.isEmpty() && _meshTriangleIndices.indexCount > 0)
    {
        _meshTriangleIndices.dirty = true;
    }
    if (!_meshWireframeIndices.indexData.isEmpty() && _meshWireframeIndices.indexCount > 0)
    {
        _meshWireframeIndices.dirty = true;
    }

    _texturedMeshPipeline.uploadedTexturePath.clear();
    _imagePipeline.uploadedImageKey.clear();
    _imagePipeline.uploadedGeometryKey.clear();
    _imagePipeline.geometryDirty = true;
    _thumbnailPoseIndicesPendingCommit.clear();
    _thumbnailCacheKeysPendingCommit.clear();
    releaseCameraThumbnailPipelineResources();
    _thumbnailPipeline.resourcesDirty = true;
    _thumbnailPipeline.instancesDirty = true;
    _thumbnailPipeline.pipelinesDirty = true;
}

void CameraSceneWidget::commitResourceUpdateState()
{
    for (const int pose_index : std::as_const(_thumbnailPoseIndicesPendingCommit))
    {
        if (_pendingThumbnailPoseIndices.remove(pose_index))
        {
            ++_cameraThumbnailLoadCompleted;
        }
    }
    for (const QString &cache_key : std::as_const(_thumbnailCacheKeysPendingCommit))
    {
        _cameraImageCache.remove(cache_key);
    }
    _thumbnailPoseIndicesPendingCommit.clear();
    _thumbnailCacheKeysPendingCommit.clear();
}

void CameraSceneWidget::resizeEvent(QResizeEvent *event)
{
    QRhiWidget::resizeEvent(event);
    if (_manualPruneMode)
    {
        clearManualPointSelection();
    }
    if (_overlayWidget)
    {
        _overlayWidget->setGeometry(rect());
        _overlayWidget->raise();
    }
    _pipelinesDirty = true;
    _imagePipeline.pipelineDirty = true;
    _thumbnailPipeline.pipelinesDirty = true;
}

void CameraSceneWidget::updateSampleCountForGeometry()
{
    const int desired_sample_count = _cloud.hasFaces() ? 4 : 1;
    if (sampleCount() == desired_sample_count)
    {
        return;
    }

    setSampleCount(desired_sample_count);
    _pipelinesDirty = true;
    _imagePipeline.pipelineDirty = true;
    _thumbnailPipeline.pipelinesDirty = true;
}

bool CameraSceneWidget::uploadGpuData()
{
    const bool use_prepared_point_buffer = _preparedPointBuffer
        && !_cloud.hasFaces()
        && _preparedPointVertexCount == static_cast<int>(_cloud.size())
        && !_preparedPointVertexData.isEmpty();
    auto assignBuffer = [](RhiBufferSet &buffer,
                           const QVector<float> &data,
                           int vertexCount,
                           int strideFloats)
    {
        buffer.vertexData = QByteArray(reinterpret_cast<const char *>(data.constData()),
                                       int(data.size() * sizeof(float)));
        buffer.vertexCount = vertexCount;
        buffer.strideBytes = strideFloats * int(sizeof(float));
        buffer.dirty = true;
    };
    auto assignByteBuffer = [](RhiBufferSet &buffer,
                               const QByteArray &data,
                               int vertexCount,
                               int strideBytes)
    {
        buffer.vertexData = data;
        buffer.vertexCount = vertexCount;
        buffer.strideBytes = strideBytes;
        buffer.dirty = true;
    };
    auto assignIndexBuffer = [](RhiIndexBufferSet &buffer,
                                const QByteArray &data,
                                int indexCount)
    {
        buffer.indexData = data;
        buffer.indexCount = indexCount;
        buffer.dirty = true;
    };

    _pointBuffer.vertexCount = 0;
    _meshBuffer.vertexCount = 0;
    _texturedMeshBuffer.vertexCount = 0;
    _meshTriangleIndices.indexCount = 0;
    _meshWireframeIndices.indexCount = 0;
    _lineBuffer.vertexCount = 0;
    _pointBuffer.vertexData.clear();
    _meshBuffer.vertexData.clear();
    _texturedMeshBuffer.vertexData.clear();
    _meshTriangleIndices.indexData.clear();
    _meshWireframeIndices.indexData.clear();
    _lineBuffer.vertexData.clear();

    // ── 1. 点云：基础属性只上传一次，颜色模式在 shader 中切换。──────────
    _pointCount = 0;
    if (_cloud.size() > 0 && !_cloud.hasFaces() && use_prepared_point_buffer)
    {
        _pointBuffer.vertexData = _preparedPointVertexData;
        _pointBuffer.vertexCount = _preparedPointVertexCount;
        _pointBuffer.strideBytes = 9 * int(sizeof(float));
        _pointBuffer.dirty = true;
        _pointCount = _preparedPointVertexCount;
    }
    else if (_cloud.size() > 0 && !_cloud.hasFaces())
    {
        constexpr std::size_t maximum_single_buffer_points =
            static_cast<std::size_t>(std::numeric_limits<int>::max())
            / (9 * sizeof(float));
        const QString message = _cloud.size() > maximum_single_buffer_points
            ? tr("点云包含 %1 点，超过当前单个 GPU 顶点缓冲上限 %2 点；已停止渲染以避免内存溢出。")
                  .arg(_cloud.size())
                  .arg(maximum_single_buffer_points)
            : tr("点云 GPU 数据准备失败；已停止渲染，且不会在 GUI 线程回退重建。");
        return failGeometryUpload(message);
    }

    if (_pointCount > 0
        && (_pointScalarBuffer.vertexCount != _pointCount
            || _pointScalarBuffer.vertexData.isEmpty()))
    {
        return failGeometryUpload(
            tr("点云标量 GPU 数据未准备完成；已停止渲染，且不会在 GUI 线程回退重建。"));
    }

    // ── 2. 网格：非纹理模式共享静态 VBO，面和边使用独立 IBO。────────
    _meshVertCount = 0;
    _meshHasFaces = false;
    _meshHasTexture = false;
    if (_cloud.size() > 0 && _cloud.hasFaces())
    {
        if (!_preparedMesh.isValid())
        {
            return failGeometryUpload(
                tr("网格 GPU 数据准备失败（%1 个顶点、%2 个面）；已停止渲染，且不会在 GUI 线程回退重建。")
                    .arg(_cloud.size())
                    .arg(_cloud.faces()->rows()));
        }
        assignByteBuffer(_meshBuffer,
                         _preparedMesh.vertexData,
                         _preparedMesh.vertexCount,
                         _preparedMesh.strideBytes);
        assignIndexBuffer(_meshTriangleIndices,
                          _preparedMesh.triangleIndexData,
                          _preparedMesh.triangleIndexCount);
        assignIndexBuffer(_meshWireframeIndices,
                          _preparedMesh.wireframeIndexData,
                          _preparedMesh.wireframeIndexCount);
        if (_preparedMesh.hasTexturedGeometry())
        {
            assignByteBuffer(_texturedMeshBuffer,
                             _preparedMesh.texturedVertexData,
                             _preparedMesh.texturedVertexCount,
                             _preparedMesh.texturedStrideBytes);
        }
        _meshVertCount = _preparedMesh.vertexCount;
        _meshHasFaces = _preparedMesh.triangleIndexCount > 0;
        _meshHasTexture = _preparedMesh.hasTexturedGeometry()
            && !_meshTextureImage.isNull();
        _modelElevationRange = _preparedMesh.elevationRange;
    }
    // ── 3. 点云包围盒 ────────────────────────────────────────────────────
    // 只使用点云自身的边界，避免相机轨迹把包围盒扩展到整个场景。
    if (_cacheDirty)
    {
        invalidateCache();
    }
    if (_hasCloudBounds && !_cloud.hasFaces())
    {
        const QVector<QVector3D> &vertices = _cachedCloudBoxVertices;
        QVector<float> line_data;
        line_data.reserve(vertices.size() * 6);
        for (const QVector3D &vertex : vertices)
        {
            line_data << vertex.x() << vertex.y() << vertex.z()
                      << 0.52f << 0.58f << 0.66f;
        }
        _lineCount = static_cast<int>(vertices.size());
        assignBuffer(_lineBuffer, line_data, _lineCount, 6);
    }
    else
    {
        _lineCount = 0;
    }

    if (_cloud.size() > 0)
    {
        LOG_INFO(QStringLiteral(
                     "[3D] GPU 几何缓存已准备: 点=%1，点云缓冲=%2，网格顶点=%3，"
                     "法向量=%4，颜色=%5，面=%6")
                     .arg(_cloud.size())
                     .arg(_pointBuffer.vertexCount)
                     .arg(_meshBuffer.vertexCount)
                     .arg(_cloud.hasNormals() ? QStringLiteral("有") : QStringLiteral("无"))
                     .arg(_cloud.hasColors() ? QStringLiteral("有") : QStringLiteral("无"))
                     .arg(_cloud.hasFaces()
                              ? static_cast<int>(_cloud.faces()->rows())
                              : 0));
    }

    _gpuDirty = false;
    _geometryUploadError.clear();
    return true;
}



bool CameraSceneWidget::ensureRhiBuffer(RhiBufferSet *buffer, QRhiResourceUpdateBatch *updates)
{
    if (!buffer || !updates)
    {
        return true;
    }
    if (buffer->vertexData.isEmpty() || buffer->vertexCount <= 0)
    {
        buffer->vertexBuffer.reset();
        buffer->dirty = false;
        return true;
    }

    const quint32 byteCount = quint32(buffer->vertexData.size());
    if (!buffer->vertexBuffer || buffer->vertexBuffer->size() != byteCount)
    {
        buffer->vertexBuffer.reset(rhi()->newBuffer(QRhiBuffer::Static, QRhiBuffer::VertexBuffer, byteCount));
        if (!buffer->vertexBuffer->create())
        {
            buffer->vertexBuffer.reset();
            _renderError = QStringLiteral("Vulkan 顶点缓冲创建失败。");
            return false;
        }
        buffer->dirty = true;
    }

    if (buffer->dirty)
    {
        updates->uploadStaticBuffer(buffer->vertexBuffer.data(), buffer->vertexData);
        buffer->dirty = false;
    }
    return true;
}

bool CameraSceneWidget::ensureRhiIndexBuffer(RhiIndexBufferSet *buffer,
                                             QRhiResourceUpdateBatch *updates)
{
    if (!buffer || !updates)
    {
        return true;
    }
    if (buffer->indexData.isEmpty() || buffer->indexCount <= 0)
    {
        buffer->indexBuffer.reset();
        buffer->dirty = false;
        return true;
    }

    const quint32 byte_count = quint32(buffer->indexData.size());
    if (!buffer->indexBuffer || buffer->indexBuffer->size() != byte_count)
    {
        buffer->indexBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Static,
            QRhiBuffer::IndexBuffer,
            byte_count));
        if (!buffer->indexBuffer->create())
        {
            buffer->indexBuffer.reset();
            _renderError = QStringLiteral("Vulkan 索引缓冲创建失败。");
            return false;
        }
        buffer->dirty = true;
    }

    if (buffer->dirty)
    {
        updates->uploadStaticBuffer(buffer->indexBuffer.data(), buffer->indexData);
        buffer->dirty = false;
    }
    return true;
}

bool CameraSceneWidget::ensurePipeline(RhiPipelineSet *pipeline,
                                       int topology,
                                       int strideBytes,
                                       bool hasNormals,
                                       bool depthWrite)
{
    if (!pipeline)
    {
        return false;
    }
    if (pipeline->pipeline && !_pipelinesDirty)
    {
        return true;
    }

    QString error;
    const QShader vertexShader = loadSceneShader(pipeline->vertexShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragmentShader = loadSceneShader(pipeline->fragmentShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    releasePipelineResources(pipeline);

    pipeline->uniformBuffer.reset(rhi()->newBuffer(QRhiBuffer::Dynamic,
                                                   QRhiBuffer::UniformBuffer,
                                                   quint32(sizeof(SceneUniforms))));
    if (!pipeline->uniformBuffer->create())
    {
        releasePipelineResources(pipeline);
        _renderError = QStringLiteral("Vulkan uniform 缓冲创建失败。");
        return false;
    }

    pipeline->bindings.reset(rhi()->newShaderResourceBindings());
    pipeline->bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            pipeline->uniformBuffer.data())
    });
    if (!pipeline->bindings->create())
    {
        releasePipelineResources(pipeline);
        _renderError = QStringLiteral("Vulkan shader 资源绑定创建失败。");
        return false;
    }

    QRhiVertexInputLayout inputLayout;
    inputLayout.setBindings({ QRhiVertexInputBinding(quint32(strideBytes)) });
    if (hasNormals)
    {
        inputLayout.setAttributes({
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
            QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3, 6 * sizeof(float)),
        });
    }
    else
    {
        inputLayout.setAttributes({
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
        });
    }

    pipeline->pipeline.reset(rhi()->newGraphicsPipeline());
    pipeline->pipeline->setTopology(static_cast<QRhiGraphicsPipeline::Topology>(topology));
    pipeline->pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertexShader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragmentShader),
    });
    pipeline->pipeline->setVertexInputLayout(inputLayout);
    pipeline->pipeline->setShaderResourceBindings(pipeline->bindings.data());
    pipeline->pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    pipeline->pipeline->setSampleCount(sampleCount());
    pipeline->pipeline->setDepthTest(true);
    pipeline->pipeline->setDepthWrite(depthWrite);
    pipeline->pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    pipeline->pipeline->setCullMode(QRhiGraphicsPipeline::None);

    QRhiGraphicsPipeline::TargetBlend blend;
    blend.enable = true;
    blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
    blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    blend.srcAlpha = QRhiGraphicsPipeline::One;
    blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    pipeline->pipeline->setTargetBlends({ blend });

    if (!pipeline->pipeline->create())
    {
        releasePipelineResources(pipeline);
        _renderError = QStringLiteral("Vulkan 图形管线创建失败。");
        return false;
    }
    return true;
}

bool CameraSceneWidget::ensurePointPipeline(RhiPipelineSet *pipeline,
                                            bool highlightOnly)
{
    if (!pipeline)
    {
        return false;
    }
    if (pipeline->pipeline && !_pipelinesDirty)
    {
        return true;
    }

    QString error;
    const QShader vertex_shader = loadSceneShader(pipeline->vertexShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragment_shader = loadSceneShader(pipeline->fragmentShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    releasePipelineResources(pipeline);
    pipeline->uniformBuffer.reset(rhi()->newBuffer(
        QRhiBuffer::Dynamic,
        QRhiBuffer::UniformBuffer,
        quint32(sizeof(SceneUniforms))));
    if (!pipeline->uniformBuffer->create())
    {
        releasePipelineResources(pipeline);
        _renderError = QStringLiteral("Vulkan 点云 uniform 缓冲创建失败。");
        return false;
    }

    pipeline->bindings.reset(rhi()->newShaderResourceBindings());
    pipeline->bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            pipeline->uniformBuffer.data())
    });
    if (!pipeline->bindings->create())
    {
        releasePipelineResources(pipeline);
        _renderError = QStringLiteral("Vulkan 点云 shader 资源绑定创建失败。");
        return false;
    }

    QRhiVertexInputLayout input_layout;
    input_layout.setBindings({
        QRhiVertexInputBinding(
            9 * sizeof(float),
            QRhiVertexInputBinding::PerInstance),
        QRhiVertexInputBinding(
            sizeof(float),
            QRhiVertexInputBinding::PerInstance)
    });
    input_layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
        QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3, 6 * sizeof(float)),
        QRhiVertexInputAttribute(1, 3, QRhiVertexInputAttribute::Float, 0),
    });

    pipeline->pipeline.reset(rhi()->newGraphicsPipeline());
    pipeline->pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    pipeline->pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertex_shader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragment_shader),
    });
    pipeline->pipeline->setVertexInputLayout(input_layout);
    pipeline->pipeline->setShaderResourceBindings(pipeline->bindings.data());
    pipeline->pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    pipeline->pipeline->setSampleCount(sampleCount());
    pipeline->pipeline->setDepthTest(!highlightOnly);
    pipeline->pipeline->setDepthWrite(!highlightOnly);
    pipeline->pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    pipeline->pipeline->setCullMode(QRhiGraphicsPipeline::None);
    if (highlightOnly)
    {
        QRhiGraphicsPipeline::TargetBlend blend;
        blend.enable = true;
        blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
        blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        blend.srcAlpha = QRhiGraphicsPipeline::One;
        blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        pipeline->pipeline->setTargetBlends({blend});
    }

    if (!pipeline->pipeline->create())
    {
        releasePipelineResources(pipeline);
        _renderError = highlightOnly
            ? QStringLiteral("Vulkan 点云高亮管线创建失败。")
            : QStringLiteral("Vulkan 点云实例化图形管线创建失败。");
        return false;
    }
    return true;
}

bool CameraSceneWidget::ensureTexturedMeshPipeline(QRhiResourceUpdateBatch *updates)
{
    if (!updates)
    {
        return true;
    }
    if (!_meshHasTexture || _meshTextureImage.isNull())
    {
        releaseTexturedMeshPipelineResources();
        return true;
    }

    const bool recreateTexture = !_texturedMeshPipeline.texture
        || _texturedMeshPipeline.textureSize != _meshTextureImage.size();
    if (recreateTexture)
    {
        _texturedMeshPipeline.texture.reset(
            rhi()->newTexture(QRhiTexture::RGBA8, _meshTextureImage.size()));
        if (!_texturedMeshPipeline.texture->create())
        {
            releaseTexturedMeshPipelineResources();
            _renderError = QStringLiteral("Vulkan 模型纹理创建失败：%1").arg(_meshTexturePath);
            return false;
        }
        _texturedMeshPipeline.textureSize = _meshTextureImage.size();
        _texturedMeshPipeline.uploadedTexturePath.clear();
        _texturedMeshPipeline.pipeline.reset();
        _texturedMeshPipeline.bindings.reset();
    }
    if (!_texturedMeshPipeline.sampler)
    {
        _texturedMeshPipeline.sampler.reset(rhi()->newSampler(QRhiSampler::Linear,
                                                              QRhiSampler::Linear,
                                                              QRhiSampler::None,
                                                              QRhiSampler::ClampToEdge,
                                                              QRhiSampler::ClampToEdge));
        if (!_texturedMeshPipeline.sampler->create())
        {
            releaseTexturedMeshPipelineResources();
            _renderError = QStringLiteral("Vulkan 模型纹理采样器创建失败。");
            return false;
        }
    }
    if (_texturedMeshPipeline.pipeline && !_pipelinesDirty)
    {
        if (_texturedMeshPipeline.uploadedTexturePath != _meshTexturePath)
        {
            updates->uploadTexture(_texturedMeshPipeline.texture.data(), _meshTextureImage);
            _texturedMeshPipeline.uploadedTexturePath = _meshTexturePath;
        }
        return true;
    }

    QString error;
    const QShader vertexShader = loadSceneShader(_texturedMeshPipeline.vertexShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragmentShader = loadSceneShader(_texturedMeshPipeline.fragmentShaderPath, &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    _texturedMeshPipeline.pipeline.reset();
    _texturedMeshPipeline.bindings.reset();
    _texturedMeshPipeline.uniformBuffer.reset(
        rhi()->newBuffer(QRhiBuffer::Dynamic,
                         QRhiBuffer::UniformBuffer,
                         quint32(sizeof(SceneUniforms))));
    if (!_texturedMeshPipeline.uniformBuffer->create())
    {
        releaseTexturedMeshPipelineResources();
        _renderError = QStringLiteral("Vulkan 纹理模型 uniform 缓冲创建失败。");
        return false;
    }

    _texturedMeshPipeline.bindings.reset(rhi()->newShaderResourceBindings());
    _texturedMeshPipeline.bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            _texturedMeshPipeline.uniformBuffer.data()),
        QRhiShaderResourceBinding::sampledTexture(
            1,
            QRhiShaderResourceBinding::FragmentStage,
            _texturedMeshPipeline.texture.data(),
            _texturedMeshPipeline.sampler.data())
    });
    if (!_texturedMeshPipeline.bindings->create())
    {
        releaseTexturedMeshPipelineResources();
        _renderError = QStringLiteral("Vulkan 纹理模型 shader 资源绑定创建失败。");
        return false;
    }

    QRhiVertexInputLayout inputLayout;
    inputLayout.setBindings({
        QRhiVertexInputBinding(quint32(_texturedMeshBuffer.strideBytes))});
    inputLayout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)),
        QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3, 6 * sizeof(float)),
        QRhiVertexInputAttribute(0, 3, QRhiVertexInputAttribute::Float2, 9 * sizeof(float)),
    });

    _texturedMeshPipeline.pipeline.reset(rhi()->newGraphicsPipeline());
    _texturedMeshPipeline.pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    _texturedMeshPipeline.pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertexShader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragmentShader),
    });
    _texturedMeshPipeline.pipeline->setVertexInputLayout(inputLayout);
    _texturedMeshPipeline.pipeline->setShaderResourceBindings(_texturedMeshPipeline.bindings.data());
    _texturedMeshPipeline.pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    _texturedMeshPipeline.pipeline->setSampleCount(sampleCount());
    _texturedMeshPipeline.pipeline->setDepthTest(true);
    _texturedMeshPipeline.pipeline->setDepthWrite(true);
    _texturedMeshPipeline.pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    _texturedMeshPipeline.pipeline->setCullMode(QRhiGraphicsPipeline::None);
    if (!_texturedMeshPipeline.pipeline->create())
    {
        releaseTexturedMeshPipelineResources();
        _renderError = QStringLiteral("Vulkan 纹理模型图形管线创建失败。");
        return false;
    }
    if (_texturedMeshPipeline.uploadedTexturePath != _meshTexturePath)
    {
        updates->uploadTexture(_texturedMeshPipeline.texture.data(), _meshTextureImage);
        _texturedMeshPipeline.uploadedTexturePath = _meshTexturePath;
    }
    return true;
}

void CameraSceneWidget::drawRhiBuffer(QRhiCommandBuffer *cb,
                                      RhiBufferSet *buffer,
                                      RhiPipelineSet *pipeline,
                                      const SceneUniforms &uniforms)
{
    if (!cb || !buffer || !pipeline || !buffer->vertexBuffer || buffer->vertexCount <= 0 || !pipeline->pipeline)
    {
        return;
    }

    pipeline->uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(&uniforms, sizeof(SceneUniforms));
    cb->setGraphicsPipeline(pipeline->pipeline.data());
    cb->setShaderResources(pipeline->bindings.data());
    const QRhiCommandBuffer::VertexInput vertexInput(buffer->vertexBuffer.data(), 0);
    cb->setVertexInput(0, 1, &vertexInput);
    cb->draw(quint32(buffer->vertexCount));
}

void CameraSceneWidget::drawIndexedRhiBuffer(QRhiCommandBuffer *cb,
                                             RhiBufferSet *vertices,
                                             RhiIndexBufferSet *indices,
                                             RhiPipelineSet *pipeline,
                                             const SceneUniforms &uniforms)
{
    if (!cb || !vertices || !indices || !pipeline
        || !vertices->vertexBuffer || !indices->indexBuffer
        || vertices->vertexCount <= 0 || indices->indexCount <= 0
        || !pipeline->pipeline || !pipeline->uniformBuffer)
    {
        return;
    }

    pipeline->uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(
        &uniforms,
        sizeof(SceneUniforms));
    cb->setGraphicsPipeline(pipeline->pipeline.data());
    cb->setShaderResources(pipeline->bindings.data());
    const QRhiCommandBuffer::VertexInput vertex_input(
        vertices->vertexBuffer.data(),
        0);
    cb->setVertexInput(0,
                       1,
                       &vertex_input,
                       indices->indexBuffer.data(),
                       0,
                       QRhiCommandBuffer::IndexUInt32);
    cb->drawIndexed(quint32(indices->indexCount));
}

void CameraSceneWidget::drawPointCloud(QRhiCommandBuffer *cb,
                                       const SceneUniforms &uniforms)
{
    if (!cb || !_pointBuffer.vertexBuffer || !_pointScalarBuffer.vertexBuffer
        || _pointBuffer.vertexCount <= 0
        || _pointScalarBuffer.vertexCount != _pointBuffer.vertexCount)
    {
        return;
    }

    auto draw_points = [cb](RhiBufferSet &point_buffer,
                            RhiBufferSet &scalar_buffer,
                            RhiPipelineSet &pipeline,
                            const SceneUniforms &draw_uniforms)
    {
        if (!point_buffer.vertexBuffer || !scalar_buffer.vertexBuffer
            || point_buffer.vertexCount <= 0
            || scalar_buffer.vertexCount != point_buffer.vertexCount
            || !pipeline.pipeline || !pipeline.uniformBuffer)
        {
            return;
        }
        const QRhiCommandBuffer::VertexInput vertex_inputs[] = {
            QRhiCommandBuffer::VertexInput(point_buffer.vertexBuffer.data(), 0),
            QRhiCommandBuffer::VertexInput(scalar_buffer.vertexBuffer.data(), 0)};
        pipeline.uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(
            &draw_uniforms,
            sizeof(SceneUniforms));
        cb->setGraphicsPipeline(pipeline.pipeline.data());
        cb->setShaderResources(pipeline.bindings.data());
        cb->setVertexInput(0, 2, vertex_inputs);
        cb->draw(6, quint32(point_buffer.vertexCount));
    };
    draw_points(_pointBuffer, _pointScalarBuffer, _colorPointPipeline, uniforms);

    if (!_manualPruneMode)
    {
        return;
    }

    if (_manualPreviewValid
        && _manualHighlightPointBuffer.vertexCount > 0
        && _manualHighlightScalarBuffer.vertexCount
            == _manualHighlightPointBuffer.vertexCount)
    {
        SceneUniforms highlight_uniforms = uniforms;
        highlight_uniforms.renderModeFlags[2] = 1.0f;
        highlight_uniforms.renderModeFlags[3] = 1.0f;
        draw_points(_manualHighlightPointBuffer,
                    _manualHighlightScalarBuffer,
                    _highlightPointPipeline,
                    highlight_uniforms);
        return;
    }

    const bool use_selection_rect = _manualSelecting
        || _manualSelectionRunning
        || (_manualPreviewValid && _manualPreviewUsesScreenRect);
    if (!use_selection_rect
        || _manualSelectRect.isNull())
    {
        return;
    }

    const QRect selection_rect = _manualSelectRect.normalized().intersected(rect());
    if (selection_rect.width() <= 0 || selection_rect.height() <= 0)
    {
        return;
    }
    const float viewport_width = float(qMax(1, width()));
    const float viewport_height = float(qMax(1, height()));
    const float minimum_x = 2.0f * float(selection_rect.x()) / viewport_width - 1.0f;
    const float maximum_x = 2.0f
        * float(selection_rect.x() + selection_rect.width()) / viewport_width - 1.0f;
    float minimum_y = 0.0f;
    float maximum_y = 0.0f;
    if (rhi()->isYUpInNDC())
    {
        minimum_y = 1.0f - 2.0f
            * float(selection_rect.y() + selection_rect.height()) / viewport_height;
        maximum_y = 1.0f - 2.0f * float(selection_rect.y()) / viewport_height;
    }
    else
    {
        minimum_y = 2.0f * float(selection_rect.y()) / viewport_height - 1.0f;
        maximum_y = 2.0f
            * float(selection_rect.y() + selection_rect.height()) / viewport_height - 1.0f;
    }

    SceneUniforms highlight_uniforms = uniforms;
    highlight_uniforms.renderModeFlags[2] = 1.0f;
    highlight_uniforms.scalarRange = {
        minimum_x,
        minimum_y,
        maximum_x,
        maximum_y};
    draw_points(_pointBuffer,
                _pointScalarBuffer,
                _highlightPointPipeline,
                highlight_uniforms);
}

void CameraSceneWidget::drawTexturedMesh(QRhiCommandBuffer *cb, const SceneUniforms &uniforms)
{
    if (!cb || !_texturedMeshBuffer.vertexBuffer
        || _texturedMeshBuffer.vertexCount <= 0
        || !_texturedMeshPipeline.pipeline || !_texturedMeshPipeline.uniformBuffer)
    {
        return;
    }

    _texturedMeshPipeline.uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(
        &uniforms, sizeof(SceneUniforms));
    cb->setGraphicsPipeline(_texturedMeshPipeline.pipeline.data());
    cb->setShaderResources(_texturedMeshPipeline.bindings.data());
    const QRhiCommandBuffer::VertexInput vertexInput(
        _texturedMeshBuffer.vertexBuffer.data(),
        0);
    cb->setVertexInput(0, 1, &vertexInput);
    cb->draw(quint32(_texturedMeshBuffer.vertexCount));
}

bool CameraSceneWidget::ensureImagePipeline(QRhiResourceUpdateBatch *updates)
{
    if (!updates)
    {
        return true;
    }
    if (!_showCameraImage)
    {
        releaseImagePipelineResources();
        return true;
    }

    const int pose_index = displayedCameraImagePoseIndex();
    if (pose_index < 0 || pose_index >= _poses.size())
    {
        releaseImagePipelineResources();
        return true;
    }

    const CameraPose &pose = _poses.at(pose_index);
    const QString image_key = cameraPlaneImageKey(
        pose.imagePath,
        CameraImagePlaneMode::Image);
    requestCameraPlaneImage(pose.imagePath, CameraImagePlaneMode::Image);
    const QImage image = cachedCameraPlaneImage(pose.imagePath, CameraImagePlaneMode::Image);
    if (image.isNull())
    {
        if (_imagePipeline.uploadedImageKey != image_key)
        {
            releaseImagePipelineResources();
        }
        return true;
    }

    if (!_imagePipeline.vertexBuffer)
    {
        _imagePipeline.vertexBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic, QRhiBuffer::VertexBuffer, 6 * 5 * sizeof(float)));
        if (!_imagePipeline.vertexBuffer->create())
        {
            releaseImagePipelineResources();
            _renderError = QStringLiteral("Vulkan 照片顶点缓冲创建失败。");
            return false;
        }
    }
    if (!_imagePipeline.uniformBuffer)
    {
        _imagePipeline.uniformBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(ImagePlaneUniforms)));
        if (!_imagePipeline.uniformBuffer->create())
        {
            releaseImagePipelineResources();
            _renderError = QStringLiteral("Vulkan 照片矩阵缓冲创建失败。");
            return false;
        }
    }

    const bool recreate_texture = !_imagePipeline.texture || _imagePipeline.textureSize != image.size();
    if (recreate_texture)
    {
        _imagePipeline.texture.reset(rhi()->newTexture(QRhiTexture::RGBA8, image.size()));
        if (!_imagePipeline.texture->create())
        {
            releaseImagePipelineResources();
            _renderError = QStringLiteral("Vulkan 照片纹理创建失败：%1").arg(pose.imagePath);
            return false;
        }
        _imagePipeline.textureSize = image.size();
        _imagePipeline.uploadedImageKey.clear();
        _imagePipeline.pipeline.reset();
        _imagePipeline.bindings.reset();
    }
    const bool texture_upload_pending = _imagePipeline.uploadedImageKey != image_key;
    QImage texture_upload_image;
    if (texture_upload_pending)
    {
        // 三维相机卡片是实体遮挡面；忽略源图可能携带的 alpha，
        // 否则连接点会从照片透明通道中穿出。
        texture_upload_image = image;
        if (rhi()->isYUpInNDC())
        {
            texture_upload_image = texture_upload_image.mirrored();
        }
    }

    const bool geometry_upload_pending = _imagePipeline.geometryDirty
        || _imagePipeline.uploadedGeometryKey != image_key;
    QVector<QVector3D> geometry_corners;
    std::array<float, 30> geometry_vertices{};
    if (geometry_upload_pending)
    {
        geometry_corners = displayedCameraImagePlaneCorners();
        if (geometry_corners.size() != 4)
        {
            releaseImagePipelineResources();
            return true;
        }

        const QVector3D &p1 = geometry_corners.at(0);
        const QVector3D &p2 = geometry_corners.at(1);
        const QVector3D &p3 = geometry_corners.at(2);
        const QVector3D &p4 = geometry_corners.at(3);
        geometry_vertices = {
            p1.x(), p1.y(), p1.z(), 1.0f, 0.0f,
            p2.x(), p2.y(), p2.z(), 0.0f, 0.0f,
            p3.x(), p3.y(), p3.z(), 0.0f, 1.0f,
            p1.x(), p1.y(), p1.z(), 1.0f, 0.0f,
            p3.x(), p3.y(), p3.z(), 0.0f, 1.0f,
            p4.x(), p4.y(), p4.z(), 1.0f, 1.0f,
        };
    }

    if (!_imagePipeline.sampler)
    {
        _imagePipeline.sampler.reset(rhi()->newSampler(QRhiSampler::Linear,
                                                       QRhiSampler::Linear,
                                                       QRhiSampler::None,
                                                       QRhiSampler::ClampToEdge,
                                                       QRhiSampler::ClampToEdge));
        if (!_imagePipeline.sampler->create())
        {
            releaseImagePipelineResources();
            _renderError = QStringLiteral("Vulkan 照片采样器创建失败。");
            return false;
        }
    }

    auto queue_pending_uploads = [this,
                                  updates,
                                  image_key,
                                  texture_upload_pending,
                                  texture_upload_image,
                                  geometry_upload_pending,
                                  geometry_corners,
                                  geometry_vertices]()
    {
        if (texture_upload_pending)
        {
            updates->uploadTexture(_imagePipeline.texture.data(), texture_upload_image);
            _imagePipeline.uploadedImageKey = image_key;
        }
        if (geometry_upload_pending)
        {
            updates->updateDynamicBuffer(
                _imagePipeline.vertexBuffer.data(),
                0,
                sizeof(geometry_vertices),
                geometry_vertices.data());
            _imagePipeline.planeCorners = geometry_corners;
            _imagePipeline.uploadedGeometryKey = image_key;
            _imagePipeline.geometryDirty = false;
        }
    };

    if (_imagePipeline.pipeline
        && !_imagePipeline.pipelineDirty
        && !_pipelinesDirty)
    {
        queue_pending_uploads();
        return true;
    }

    QString error;
    const QShader vertex_shader = loadSceneShader(
        QStringLiteral(":/shaders/camera_scene_image.vert.qsb"), &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }
    const QShader fragment_shader = loadSceneShader(
        QStringLiteral(":/shaders/camera_scene_image.frag.qsb"), &error);
    if (!error.isEmpty())
    {
        _renderError = error;
        return false;
    }

    _imagePipeline.pipeline.reset();
    _imagePipeline.bindings.reset(rhi()->newShaderResourceBindings());
    _imagePipeline.bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(0,
                                                  QRhiShaderResourceBinding::VertexStage,
                                                  _imagePipeline.uniformBuffer.data()),
        QRhiShaderResourceBinding::sampledTexture(1,
                                                  QRhiShaderResourceBinding::FragmentStage,
                                                  _imagePipeline.texture.data(),
                                                  _imagePipeline.sampler.data())
    });
    if (!_imagePipeline.bindings->create())
    {
        releaseImagePipelineResources();
        _renderError = QStringLiteral("Vulkan 照片 shader 资源绑定创建失败。");
        return false;
    }

    QRhiVertexInputLayout input_layout;
    input_layout.setBindings({QRhiVertexInputBinding(5 * sizeof(float))});
    input_layout.setAttributes({
        QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float2, 3 * sizeof(float)),
    });

    _imagePipeline.pipeline.reset(rhi()->newGraphicsPipeline());
    _imagePipeline.pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
    _imagePipeline.pipeline->setShaderStages({
        QRhiShaderStage(QRhiShaderStage::Vertex, vertex_shader),
        QRhiShaderStage(QRhiShaderStage::Fragment, fragment_shader),
    });
    _imagePipeline.pipeline->setVertexInputLayout(input_layout);
    _imagePipeline.pipeline->setShaderResourceBindings(_imagePipeline.bindings.data());
    _imagePipeline.pipeline->setRenderPassDescriptor(renderTarget()->renderPassDescriptor());
    _imagePipeline.pipeline->setSampleCount(sampleCount());
    _imagePipeline.pipeline->setDepthTest(false);
    _imagePipeline.pipeline->setDepthWrite(false);
    _imagePipeline.pipeline->setCullMode(QRhiGraphicsPipeline::None);
    if (!_imagePipeline.pipeline->create())
    {
        releaseImagePipelineResources();
        _renderError = QStringLiteral("Vulkan 照片合成管线创建失败。");
        return false;
    }
    _imagePipeline.pipelineDirty = false;
    queue_pending_uploads();
    return true;
}

bool CameraSceneWidget::ensureSolidCameraBatchResource(
    QSharedPointer<RhiCameraThumbnailResource> *resource,
    const QColor &color,
    QRhiResourceUpdateBatch *updates)
{
    if (!resource || !updates || !_thumbnailPipeline.uniformBuffer
        || !_thumbnailPipeline.sampler)
    {
        return false;
    }

    const int required_capacity = qMax(1, static_cast<int>(_poses.size()));
    if (*resource && (*resource)->instanceBuffer
        && (*resource)->instanceCapacity >= required_capacity)
    {
        return true;
    }

    auto batch_resource = QSharedPointer<RhiCameraThumbnailResource>::create();
    batch_resource->instanceCapacity = required_capacity;
    batch_resource->instanceBuffer.reset(rhi()->newBuffer(
        QRhiBuffer::Dynamic,
        QRhiBuffer::VertexBuffer,
        required_capacity * kCameraInstanceStrideFloats * int(sizeof(float))));
    if (!batch_resource->instanceBuffer->create())
    {
        _renderError = QStringLiteral("Vulkan 相机实例缓冲创建失败。");
        return false;
    }

    const QImage color_image(QSize(1, 1), QImage::Format_RGBX8888);
    QImage upload_image = color_image;
    upload_image.fill(color);
    batch_resource->texture.reset(rhi()->newTexture(QRhiTexture::RGBA8, upload_image.size()));
    if (!batch_resource->texture->create())
    {
        _renderError = QStringLiteral("Vulkan 相机批量颜色纹理创建失败。");
        return false;
    }
    batch_resource->bindings.reset(rhi()->newShaderResourceBindings());
    batch_resource->bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(0,
                                                  QRhiShaderResourceBinding::VertexStage,
                                                  _thumbnailPipeline.uniformBuffer.data()),
        QRhiShaderResourceBinding::sampledTexture(1,
                                                  QRhiShaderResourceBinding::FragmentStage,
                                                  batch_resource->texture.data(),
                                                  _thumbnailPipeline.sampler.data())
    });
    if (!batch_resource->bindings->create())
    {
        _renderError = QStringLiteral("Vulkan 相机批量 shader 资源创建失败。");
        return false;
    }

    *resource = batch_resource;
    updates->uploadTexture(batch_resource->texture.data(), upload_image);
    return true;
}

bool CameraSceneWidget::ensureCameraThumbnailAtlasPage(
    int page_index,
    QRhiResourceUpdateBatch *updates)
{
    if (page_index < 0 || !updates || _thumbnailPipeline.atlasSize <= 0
        || !_thumbnailPipeline.uniformBuffer || !_thumbnailPipeline.sampler)
    {
        return false;
    }

    if (_thumbnailPipeline.atlasPages.size() <= page_index)
    {
        _thumbnailPipeline.atlasPages.resize(page_index + 1);
    }
    if (_thumbnailPipeline.atlasPages.at(page_index))
    {
        return true;
    }

    const int columns = _thumbnailPipeline.atlasSize / kCameraThumbnailWidth;
    const int rows = _thumbnailPipeline.atlasSize / kCameraThumbnailHeight;
    const int instance_capacity = qMax(1, columns * rows);
    auto page = QSharedPointer<RhiCameraThumbnailAtlasPage>::create();
    page->instanceCapacity = instance_capacity;
    page->instanceBuffer.reset(rhi()->newBuffer(
        QRhiBuffer::Dynamic,
        QRhiBuffer::VertexBuffer,
        instance_capacity * kCameraInstanceStrideFloats * int(sizeof(float))));
    if (!page->instanceBuffer->create())
    {
        _renderError = QStringLiteral("Vulkan 相机纹理图集实例缓冲创建失败。");
        return false;
    }

    const QSize atlas_size(_thumbnailPipeline.atlasSize, _thumbnailPipeline.atlasSize);
    page->texture.reset(rhi()->newTexture(QRhiTexture::RGBA8, atlas_size));
    if (!page->texture->create())
    {
        _renderError = QStringLiteral("Vulkan 相机纹理图集创建失败：%1×%1。")
            .arg(_thumbnailPipeline.atlasSize);
        return false;
    }

    page->bindings.reset(rhi()->newShaderResourceBindings());
    page->bindings->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(0,
                                                  QRhiShaderResourceBinding::VertexStage,
                                                  _thumbnailPipeline.uniformBuffer.data()),
        QRhiShaderResourceBinding::sampledTexture(1,
                                                  QRhiShaderResourceBinding::FragmentStage,
                                                  page->texture.data(),
                                                  _thumbnailPipeline.sampler.data())
    });
    if (!page->bindings->create())
    {
        _renderError = QStringLiteral("Vulkan 相机纹理图集 shader 资源创建失败。");
        return false;
    }

    _thumbnailPipeline.atlasPages[page_index] = page;
    return true;
}

bool CameraSceneWidget::ensureCameraThumbnailPipeline(QRhiResourceUpdateBatch *updates)
{
    if (!updates)
    {
        return true;
    }
    if (!_showCameras)
    {
        releaseCameraThumbnailPipelineResources();
        _thumbnailPipeline.resourcesDirty = true;
        _thumbnailPipeline.instancesDirty = true;
        _thumbnailPipeline.pipelinesDirty = true;
        return true;
    }

    if (_thumbnailPipeline.resourcesDirty)
    {
        if (_showCameraThumbnails)
        {
            _cameraThumbnailLoadCompleted = 0;
        }
        _thumbnailPipeline.atlasPages.clear();
        _thumbnailPipeline.solidResource.clear();
        _thumbnailPipeline.highlightedSolidResource.clear();
        _thumbnailPipeline.leaderInstanceBuffer.reset();
        _thumbnailPipeline.leaderBindings.reset();
        _thumbnailPipeline.leaderPipeline.reset();
        _thumbnailPipeline.leaderInstanceCapacity = 0;
        _thumbnailPipeline.leaderInstanceCount = 0;
        _thumbnailPipeline.segmentInstanceCount = 0;
        _thumbnailPipeline.pipeline.reset();
        _thumbnailPipeline.atlasSize = 0;
        _thumbnailPipeline.resourcesDirty = false;
        _thumbnailPipeline.instancesDirty = true;
        _thumbnailPipeline.pipelinesDirty = true;
        _pendingThumbnailPoseIndices.clear();
        if (_showCameraThumbnails)
        {
            for (qsizetype pose_index = 0; pose_index < _poses.size(); ++pose_index)
            {
                const CameraPose &pose = _poses.at(pose_index);
                if (pose.imagePath.isEmpty())
                {
                    continue;
                }
                const QString thumbnail_key = cameraPlaneImageKey(
                    pose.imagePath, CameraImagePlaneMode::Thumbnail);
                if (_cameraImageLoadFailures.contains(thumbnail_key))
                {
                    ++_cameraThumbnailLoadCompleted;
                    continue;
                }
                if (!cachedCameraPlaneImage(
                        pose.imagePath, CameraImagePlaneMode::Thumbnail).isNull())
                {
                    _pendingThumbnailPoseIndices.insert(static_cast<int>(pose_index));
                }
                else
                {
                    requestCameraPlaneImage(
                        pose.imagePath, CameraImagePlaneMode::Thumbnail);
                }
            }
        }
    }

    if (!_thumbnailPipeline.uniformBuffer)
    {
        _thumbnailPipeline.uniformBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(CameraPlaneUniforms)));
        if (!_thumbnailPipeline.uniformBuffer->create())
        {
            releaseCameraThumbnailPipelineResources();
            _renderError = QStringLiteral("Vulkan 相机缩略图矩阵缓冲创建失败。");
            return false;
        }
    }
    if (!_thumbnailPipeline.sampler)
    {
        _thumbnailPipeline.sampler.reset(rhi()->newSampler(QRhiSampler::Linear,
                                                           QRhiSampler::Linear,
                                                           QRhiSampler::None,
                                                           QRhiSampler::ClampToEdge,
                                                           QRhiSampler::ClampToEdge));
        if (!_thumbnailPipeline.sampler->create())
        {
            releaseCameraThumbnailPipelineResources();
            _renderError = QStringLiteral("Vulkan 相机缩略图采样器创建失败。");
            return false;
        }
    }

    if (_showCameraThumbnails && _thumbnailPipeline.atlasSize <= 0)
    {
        const int maximum_texture_size = rhi()->resourceLimit(QRhi::TextureSizeMax);
        const int desired_atlas_size = _poses.size() <= kSmallCameraAtlasCapacity
            ? kSmallCameraAtlasSize
            : kLargeCameraAtlasSize;
        _thumbnailPipeline.atlasSize = qMin(maximum_texture_size, desired_atlas_size);
        if (_thumbnailPipeline.atlasSize < kCameraThumbnailWidth
            || _thumbnailPipeline.atlasSize < kCameraThumbnailHeight)
        {
            _renderError = QStringLiteral("Vulkan 设备不支持相机缩略图纹理图集。");
            return false;
        }
    }

    if (!ensureSolidCameraBatchResource(
            &_thumbnailPipeline.solidResource,
            QColor(57, 112, 173),
            updates)
        || !ensureSolidCameraBatchResource(
            &_thumbnailPipeline.highlightedSolidResource,
            QColor(205, 60, 70),
            updates))
    {
        return false;
    }

    const int atlas_columns = _thumbnailPipeline.atlasSize > 0
        ? _thumbnailPipeline.atlasSize / kCameraThumbnailWidth
        : 0;
    const int atlas_rows = _thumbnailPipeline.atlasSize > 0
        ? _thumbnailPipeline.atlasSize / kCameraThumbnailHeight
        : 0;
    const int atlas_slots_per_page = atlas_columns * atlas_rows;
    const QList<int> pending_thumbnail_indices =
        _pendingThumbnailPoseIndices.values();
    for (const int pose_index : pending_thumbnail_indices)
    {
        if (!_showCameraThumbnails || pose_index < 0 || pose_index >= _poses.size())
        {
            _pendingThumbnailPoseIndices.remove(pose_index);
            continue;
        }
        const CameraPose &pose = _poses.at(pose_index);
        const CameraImagePlaneMode planeMode = CameraImagePlaneMode::Thumbnail;
        if (pose.imagePath.isEmpty())
        {
            continue;
        }
        const int page_index = static_cast<int>(pose_index) / atlas_slots_per_page;
        const auto existing_page = page_index < _thumbnailPipeline.atlasPages.size()
            ? _thumbnailPipeline.atlasPages.at(page_index)
            : QSharedPointer<RhiCameraThumbnailAtlasPage>();
        if (existing_page
            && existing_page->uploadedPoseIndices.contains(pose_index))
        {
            _pendingThumbnailPoseIndices.remove(pose_index);
            continue;
        }

        const QImage image = cachedCameraPlaneImage(pose.imagePath, planeMode);
        if (image.isNull())
        {
            continue;
        }
        if (!ensureCameraThumbnailAtlasPage(page_index, updates))
        {
            return false;
        }

        const auto page = _thumbnailPipeline.atlasPages.at(page_index);
        const int slot_index = static_cast<int>(pose_index) % atlas_slots_per_page;
        const QPoint destination(
            (slot_index % atlas_columns) * kCameraThumbnailWidth,
            (slot_index / atlas_columns) * kCameraThumbnailHeight);
        QImage upload_image = image;
        if (rhi()->isYUpInNDC())
        {
            upload_image = upload_image.mirrored();
        }
        QRhiTextureSubresourceUploadDescription subresource(upload_image);
        subresource.setDestinationTopLeft(destination);
        updates->uploadTexture(
            page->texture.data(),
            QRhiTextureUploadDescription({QRhiTextureUploadEntry(0, 0, subresource)}));
        page->uploadedPoseIndices.insert(pose_index);
        page->imageSizes.insert(pose_index, upload_image.size());
        _thumbnailPoseIndicesPendingCommit.insert(pose_index);
        _thumbnailCacheKeysPendingCommit.insert(
            cameraPlaneImageKey(pose.imagePath, planeMode));
        _thumbnailPipeline.instancesDirty = true;
    }

    if (!_thumbnailPipeline.solidResource
        || !_thumbnailPipeline.solidResource->bindings)
    {
        return true;
    }
    if (!_thumbnailPipeline.pipeline
        || !_thumbnailPipeline.leaderPipeline
        || _thumbnailPipeline.pipelinesDirty
        || _pipelinesDirty)
    {
        QString error;
        const QShader camera_vertex_shader = loadSceneShader(
            QStringLiteral(":/shaders/camera_scene_camera.vert.qsb"), &error);
        if (!error.isEmpty())
        {
            _renderError = error;
            return false;
        }
        const QShader leader_vertex_shader = loadSceneShader(
            QStringLiteral(":/shaders/camera_scene_camera_leader.vert.qsb"), &error);
        if (!error.isEmpty())
        {
            _renderError = error;
            return false;
        }
        const QShader image_fragment_shader = loadSceneShader(
            QStringLiteral(":/shaders/camera_scene_image.frag.qsb"), &error);
        if (!error.isEmpty())
        {
            _renderError = error;
            return false;
        }
        const QShader color_fragment_shader = loadSceneShader(
            QStringLiteral(":/shaders/camera_scene_color.frag.qsb"), &error);
        if (!error.isEmpty())
        {
            _renderError = error;
            return false;
        }

        QRhiVertexInputLayout instance_layout;
        instance_layout.setBindings({QRhiVertexInputBinding(
            kCameraInstanceStrideFloats * sizeof(float),
            QRhiVertexInputBinding::PerInstance)});
        instance_layout.setAttributes({
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3, 4 * sizeof(float)),
            QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3, 8 * sizeof(float)),
            QRhiVertexInputAttribute(0, 3, QRhiVertexInputAttribute::Float3, 12 * sizeof(float)),
            QRhiVertexInputAttribute(0, 4, QRhiVertexInputAttribute::Float4, 16 * sizeof(float)),
        });

        _thumbnailPipeline.leaderBindings.reset(rhi()->newShaderResourceBindings());
        _thumbnailPipeline.leaderBindings->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(
                0,
                QRhiShaderResourceBinding::VertexStage,
                _thumbnailPipeline.uniformBuffer.data())
        });
        if (!_thumbnailPipeline.leaderBindings->create())
        {
            _renderError = QStringLiteral("Vulkan 相机方位线资源绑定创建失败。");
            return false;
        }

        _thumbnailPipeline.pipeline.reset(rhi()->newGraphicsPipeline());
        _thumbnailPipeline.pipeline->setTopology(QRhiGraphicsPipeline::Triangles);
        _thumbnailPipeline.pipeline->setShaderStages({
            QRhiShaderStage(QRhiShaderStage::Vertex, camera_vertex_shader),
            QRhiShaderStage(QRhiShaderStage::Fragment, image_fragment_shader),
        });
        _thumbnailPipeline.pipeline->setVertexInputLayout(instance_layout);
        _thumbnailPipeline.pipeline->setShaderResourceBindings(
            _thumbnailPipeline.solidResource->bindings.data());
        _thumbnailPipeline.pipeline->setRenderPassDescriptor(
            renderTarget()->renderPassDescriptor());
        _thumbnailPipeline.pipeline->setSampleCount(sampleCount());
        _thumbnailPipeline.pipeline->setDepthTest(true);
        _thumbnailPipeline.pipeline->setDepthWrite(true);
        _thumbnailPipeline.pipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
        _thumbnailPipeline.pipeline->setCullMode(QRhiGraphicsPipeline::None);
        if (!_thumbnailPipeline.pipeline->create())
        {
            _renderError = QStringLiteral("Vulkan 相机实例化图形管线创建失败。");
            return false;
        }

        _thumbnailPipeline.leaderPipeline.reset(rhi()->newGraphicsPipeline());
        _thumbnailPipeline.leaderPipeline->setTopology(QRhiGraphicsPipeline::Lines);
        _thumbnailPipeline.leaderPipeline->setShaderStages({
            QRhiShaderStage(QRhiShaderStage::Vertex, leader_vertex_shader),
            QRhiShaderStage(QRhiShaderStage::Fragment, color_fragment_shader),
        });
        _thumbnailPipeline.leaderPipeline->setVertexInputLayout(instance_layout);
        _thumbnailPipeline.leaderPipeline->setShaderResourceBindings(
            _thumbnailPipeline.leaderBindings.data());
        _thumbnailPipeline.leaderPipeline->setRenderPassDescriptor(
            renderTarget()->renderPassDescriptor());
        _thumbnailPipeline.leaderPipeline->setSampleCount(sampleCount());
        _thumbnailPipeline.leaderPipeline->setDepthTest(true);
        _thumbnailPipeline.leaderPipeline->setDepthWrite(false);
        _thumbnailPipeline.leaderPipeline->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
        _thumbnailPipeline.leaderPipeline->setCullMode(QRhiGraphicsPipeline::None);
        if (!_thumbnailPipeline.leaderPipeline->create())
        {
            _renderError = QStringLiteral("Vulkan 相机方位线实例化管线创建失败。");
            return false;
        }
        _thumbnailPipeline.pipelinesDirty = false;
    }

    if (!_thumbnailPipeline.instancesDirty)
    {
        return true;
    }

    QVector<float> solid_instances;
    QVector<float> highlighted_instances;
    QVector<float> segment_instances;
    QVector<QVector<float>> atlas_instances(_thumbnailPipeline.atlasPages.size());
    solid_instances.reserve(_poses.size() * kCameraInstanceStrideFloats);
    highlighted_instances.reserve(kCameraInstanceStrideFloats);
    const int segment_multiplier = _showCameraLocalAxes ? 4 : 1;
    segment_instances.reserve(
        _poses.size() * segment_multiplier * kCameraInstanceStrideFloats);
    int leader_count = 0;

    auto append_instance = [](QVector<float> *target,
                              const CameraPose &pose,
                              const QVector4D &uv_rect)
    {
        const xjw::gui::camera_scene::CameraImagePlaneAxes axes =
            xjw::gui::camera_scene::cameraImagePlaneAxes(
                pose.rotation, pose.uAxisSign, pose.vAxisSign);
        const QVector3D forward = xjw::gui::camera_scene::cameraForwardDirection(
            pose.rotation, pose.depthAxisFlipped);
        target->append({
            pose.center.x(), pose.center.y(), pose.center.z(), 0.0f,
            axes.right.x(), axes.right.y(), axes.right.z(), 0.0f,
            axes.up.x(), axes.up.y(), axes.up.z(), 0.0f,
            forward.x(), forward.y(), forward.z(), 0.0f,
            uv_rect.x(), uv_rect.y(), uv_rect.z(), uv_rect.w(),
        });
    };

    auto append_segment = [](QVector<float> *target,
                             const CameraPose &pose,
                             const QVector3D &direction,
                             const QVector3D &color,
                             float length_scale)
    {
        target->append({
            pose.center.x(), pose.center.y(), pose.center.z(), 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            direction.x(), direction.y(), direction.z(), 0.0f,
            color.x(), color.y(), color.z(), length_scale,
        });
    };

    for (qsizetype pose_index = 0; pose_index < _poses.size(); ++pose_index)
    {
        const CameraPose &pose = _poses.at(pose_index);
        const QVector3D forward = xjw::gui::camera_scene::cameraForwardDirection(
            pose.rotation, pose.depthAxisFlipped);
        if (!forward.isNull())
        {
            append_segment(&segment_instances,
                           pose,
                           forward,
                           QVector3D(0.10f, 0.10f, 0.10f),
                           -1.35f);
            ++leader_count;
        }
        int atlas_page_index = -1;
        QVector4D uv_rect(0.0f, 0.0f, 1.0f, 1.0f);
        if (_showCameraThumbnails && atlas_slots_per_page > 0)
        {
            atlas_page_index = static_cast<int>(pose_index) / atlas_slots_per_page;
            const auto page = atlas_page_index < _thumbnailPipeline.atlasPages.size()
                ? _thumbnailPipeline.atlasPages.at(atlas_page_index)
                : QSharedPointer<RhiCameraThumbnailAtlasPage>();
            if (page && page->uploadedPoseIndices.contains(static_cast<int>(pose_index)))
            {
                const int slot_index = static_cast<int>(pose_index) % atlas_slots_per_page;
                const QSize image_size = page->imageSizes.value(static_cast<int>(pose_index));
                const int atlas_x = (slot_index % atlas_columns) * kCameraThumbnailWidth;
                const int atlas_y = (slot_index / atlas_columns) * kCameraThumbnailHeight;
                const float atlas_size = static_cast<float>(_thumbnailPipeline.atlasSize);
                uv_rect = QVector4D(
                    (static_cast<float>(atlas_x) + 0.5f) / atlas_size,
                    (static_cast<float>(atlas_y) + 0.5f) / atlas_size,
                    (static_cast<float>(atlas_x + image_size.width()) - 0.5f) / atlas_size,
                    (static_cast<float>(atlas_y + image_size.height()) - 0.5f) / atlas_size);
            }
            else
            {
                atlas_page_index = -1;
            }
        }

        if (atlas_page_index >= 0)
        {
            append_instance(&atlas_instances[atlas_page_index], pose, uv_rect);
        }
        else if (isCameraHighlighted(pose))
        {
            append_instance(&highlighted_instances, pose, uv_rect);
        }
        else
        {
            append_instance(&solid_instances, pose, uv_rect);
        }
    }

    if (_showCameraLocalAxes)
    {
        for (const CameraPose &pose : _poses)
        {
            const auto axes = xjw::gui::camera_scene::cameraLocalAxes(
                pose.rotation, pose.depthAxisFlipped);
            append_segment(&segment_instances,
                           pose,
                           axes.x,
                           QVector3D(220.0f, 55.0f, 55.0f) / 255.0f,
                           0.75f);
            append_segment(&segment_instances,
                           pose,
                           axes.y,
                           QVector3D(35.0f, 165.0f, 70.0f) / 255.0f,
                           0.75f);
            append_segment(&segment_instances,
                           pose,
                           axes.z,
                           QVector3D(45.0f, 105.0f, 225.0f) / 255.0f,
                           0.75f);
        }
    }

    auto upload_instances = [this, updates](auto &resource,
                                             const QVector<float> &instances) -> bool
    {
        if (!resource)
        {
            return instances.isEmpty();
        }
        const int instance_count = instances.size() / kCameraInstanceStrideFloats;
        if (instance_count > resource->instanceCapacity)
        {
            resource->instanceCapacity = qMax(1, instance_count);
            resource->instanceBuffer.reset(rhi()->newBuffer(
                QRhiBuffer::Dynamic,
                QRhiBuffer::VertexBuffer,
                resource->instanceCapacity * kCameraInstanceStrideFloats * int(sizeof(float))));
            if (!resource->instanceBuffer->create())
            {
                resource->instanceBuffer.reset();
                resource->instanceCapacity = 0;
                _renderError = QStringLiteral("Vulkan 相机实例缓冲扩容失败。");
                return false;
            }
        }
        resource->instanceCount = instance_count;
        if (!instances.isEmpty())
        {
            updates->updateDynamicBuffer(
                resource->instanceBuffer.data(),
                0,
                instances.size() * int(sizeof(float)),
                instances.constData());
        }
        return true;
    };

    if (!upload_instances(_thumbnailPipeline.solidResource, solid_instances)
        || !upload_instances(_thumbnailPipeline.highlightedSolidResource, highlighted_instances))
    {
        return false;
    }
    for (qsizetype page_index = 0; page_index < _thumbnailPipeline.atlasPages.size(); ++page_index)
    {
        auto page = _thumbnailPipeline.atlasPages[page_index];
        if (!upload_instances(page, atlas_instances.at(page_index)))
        {
            return false;
        }
    }

    const int segment_count = segment_instances.size() / kCameraInstanceStrideFloats;
    if (!_thumbnailPipeline.leaderInstanceBuffer
        || _thumbnailPipeline.leaderInstanceCapacity < segment_count)
    {
        _thumbnailPipeline.leaderInstanceCapacity = qMax(1, segment_count);
        _thumbnailPipeline.leaderInstanceBuffer.reset(rhi()->newBuffer(
            QRhiBuffer::Dynamic,
            QRhiBuffer::VertexBuffer,
            _thumbnailPipeline.leaderInstanceCapacity
                * kCameraInstanceStrideFloats * int(sizeof(float))));
        if (!_thumbnailPipeline.leaderInstanceBuffer->create())
        {
            _thumbnailPipeline.leaderInstanceBuffer.reset();
            _thumbnailPipeline.leaderInstanceCapacity = 0;
            _renderError = QStringLiteral("Vulkan 相机方位线实例缓冲创建失败。");
            return false;
        }
    }
    _thumbnailPipeline.leaderInstanceCount = leader_count;
    _thumbnailPipeline.segmentInstanceCount = segment_count;
    if (!segment_instances.isEmpty())
    {
        updates->updateDynamicBuffer(
            _thumbnailPipeline.leaderInstanceBuffer.data(),
            0,
            segment_instances.size() * int(sizeof(float)),
            segment_instances.constData());
    }
    _thumbnailPipeline.instancesDirty = false;
    return true;
}

void CameraSceneWidget::drawCameraThumbnails(QRhiCommandBuffer *cb,
                                             const QMatrix4x4 &mvp,
                                             const QMatrix4x4 &model_view)
{
    if (!cb || !_showCameras || !_thumbnailPipeline.pipeline
        || !_thumbnailPipeline.uniformBuffer || !_thumbnailPipeline.solidResource)
    {
        return;
    }

    CameraPlaneUniforms uniforms;
    std::copy_n(mvp.constData(), 16, uniforms.mvp.begin());
    std::copy_n(model_view.constData(), 16, uniforms.modelView.begin());
    uniforms.viewportZoom = {
        static_cast<float>(qMax(1, height())),
        static_cast<float>(_zoomScale),
        static_cast<float>(xjw::gui::camera_scene::cameraPlaneScreenHalfExtentPixels(
            _zoomScale, 34.0)),
        qMax(1.0e-5f, sceneRadius() * 0.065f),
    };
    _thumbnailPipeline.uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(
        &uniforms, sizeof(uniforms));

    const int segment_draw_count = _showCameraLocalAxes
            && !_leftDragging && !_middleDragging
        ? _thumbnailPipeline.segmentInstanceCount
        : _thumbnailPipeline.leaderInstanceCount;
    if (_thumbnailPipeline.leaderPipeline
        && _thumbnailPipeline.leaderBindings
        && _thumbnailPipeline.leaderInstanceBuffer
        && segment_draw_count > 0)
    {
        cb->setGraphicsPipeline(_thumbnailPipeline.leaderPipeline.data());
        cb->setShaderResources(_thumbnailPipeline.leaderBindings.data());
        const QRhiCommandBuffer::VertexInput instance_input(
            _thumbnailPipeline.leaderInstanceBuffer.data(), 0);
        cb->setVertexInput(0, 1, &instance_input);
        cb->draw(2, quint32(segment_draw_count));
    }

    cb->setGraphicsPipeline(_thumbnailPipeline.pipeline.data());
    auto draw_instances = [cb](const auto &resource)
    {
        if (!resource || !resource->instanceBuffer || !resource->bindings
            || resource->instanceCount <= 0)
        {
            return;
        }
        cb->setShaderResources(resource->bindings.data());
        const QRhiCommandBuffer::VertexInput instance_input(
            resource->instanceBuffer.data(), 0);
        cb->setVertexInput(0, 1, &instance_input);
        cb->draw(6, quint32(resource->instanceCount));
    };
    draw_instances(_thumbnailPipeline.solidResource);
    draw_instances(_thumbnailPipeline.highlightedSolidResource);
    for (const auto &page : std::as_const(_thumbnailPipeline.atlasPages))
    {
        draw_instances(page);
    }
}

void CameraSceneWidget::drawActiveCameraImage(QRhiCommandBuffer *cb, const QMatrix4x4 &mvp)
{
    if (!cb || !_showCameraImage || !_imagePipeline.pipeline || !_imagePipeline.vertexBuffer
        || !_imagePipeline.uniformBuffer)
    {
        return;
    }

    const int pose_index = displayedCameraImagePoseIndex();
    if (pose_index < 0 || pose_index >= _poses.size())
    {
        return;
    }
    const QString active_key = cameraPlaneImageKey(
        _poses.at(pose_index).imagePath, CameraImagePlaneMode::Image);
    if (_imagePipeline.uploadedImageKey != active_key)
    {
        return;
    }
    if (_imagePipeline.geometryDirty
        || _imagePipeline.uploadedGeometryKey != active_key
        || _imagePipeline.planeCorners.size() != 4)
    {
        return;
    }
    ImagePlaneUniforms uniforms;
    std::copy_n(mvp.constData(), 16, uniforms.mvp.begin());
    _imagePipeline.uniformBuffer->fullDynamicBufferUpdateForCurrentFrame(
        &uniforms, sizeof(uniforms));
    cb->setGraphicsPipeline(_imagePipeline.pipeline.data());
    cb->setShaderResources(_imagePipeline.bindings.data());
    const QRhiCommandBuffer::VertexInput vertex_input(_imagePipeline.vertexBuffer.data(), 0);
    cb->setVertexInput(0, 1, &vertex_input);
    cb->draw(6);
}

QVector<QVector3D> CameraSceneWidget::displayedCameraImagePlaneCorners() const
{
    const int poseIndex = displayedCameraImagePoseIndex();
    if (poseIndex < 0 || poseIndex >= _poses.size())
    {
        return {};
    }

    const CameraPose &pose = _poses.at(poseIndex);
    const QString image_key = cameraPlaneImageKey(
        pose.imagePath, CameraImagePlaneMode::Image);
    if (!_imagePipeline.geometryDirty
        && _imagePipeline.uploadedGeometryKey == image_key
        && _imagePipeline.planeCorners.size() == 4)
    {
        return _imagePipeline.planeCorners;
    }
    const QImage image = cachedCameraPlaneImage(pose.imagePath, CameraImagePlaneMode::Image);
    if (image.isNull())
    {
        return {};
    }

    const int imageWidth = pose.imageWidth > 0 ? pose.imageWidth : image.width();
    const int imageHeight = pose.imageHeight > 0 ? pose.imageHeight : image.height();
    const QVector3D forward = xjw::gui::camera_scene::cameraForwardDirection(
        pose.rotation, pose.depthAxisFlipped);
    const xjw::gui::camera_scene::CameraImagePlaneAxes axes =
        xjw::gui::camera_scene::cameraImagePlaneAxes(
            pose.rotation, pose.uAxisSign, pose.vAxisSign);
    return xjw::gui::camera_scene::calibratedImagePlaneCorners(
        pose.center,
        forward,
        axes.right,
        axes.up,
        sceneCenter(),
        pose.focalX,
        pose.focalY,
        pose.principalX,
        pose.principalY,
        imageWidth,
        imageHeight);
}

QPainterPath CameraSceneWidget::foregroundCameraImageOcclusionPath() const
{
    QPainterPath occlusionPath;
    if (!_showCameraImage
        || _cameraImageDisplayLayer != CameraImageDisplayLayer::Foreground)
    {
        return occlusionPath;
    }

    const QVector<QVector3D> corners = displayedCameraImagePlaneCorners();
    if (corners.size() != 4)
    {
        return occlusionPath;
    }

    QPolygonF projectedPlane;
    projectedPlane.reserve(corners.size());
    for (const QVector3D &corner : corners)
    {
        bool projected = false;
        const QPointF screenPoint = projectToScreen(corner, &projected);
        if (!projected)
        {
            return {};
        }
        projectedPlane.push_back(screenPoint);
    }

    occlusionPath.addPolygon(projectedPlane);
    occlusionPath.closeSubpath();
    return occlusionPath;
}

void CameraSceneWidget::drawSceneGeometry(QRhiCommandBuffer *cb,
                                          SceneUniforms &uniforms)
{
    auto apply_scalar_range = [](SceneUniforms *target,
                                 const xjw::gui::tie_points::ScalarRange &range)
    {
        if (target && range.isValid())
        {
            target->scalarRange = {
                float(range.minimum),
                float(range.maximum),
                0.0f,
                0.0f};
        }
        else if (target)
        {
            target->scalarRange = {};
        }
    };

    const float point_diameter = _isTiePointCloud
        ? qMax(2.4f, xjw::gui::tie_points::pointSizeForMode(_tiePointColorMode))
        : _pointCloudPointSize;
    uniforms.lightDirPointSize = {
        -0.45f,
        0.70f,
        0.70f,
        point_diameter * float(devicePixelRatioF())};
    const float pixel_ratio = float(devicePixelRatioF());
    const QSize viewport_size(
        qMax(1, qRound(float(width()) * pixel_ratio)),
        qMax(1, qRound(float(height()) * pixel_ratio)));
    uniforms.viewportSize = {
        float(viewport_size.width()),
        float(viewport_size.height()),
        0.0f,
        0.0f};

    const TiePointColorMode point_mode = _isTiePointCloud
        ? _tiePointColorMode
        : TiePointColorMode::Color;
    const bool has_image_counts = _isTiePointCloud
        && _tiePointImageCounts.size() == static_cast<qsizetype>(_pointBuffer.vertexCount);
    uniforms.renderModeFlags = {
        float(static_cast<int>(point_mode)),
        has_image_counts ? 1.0f : 0.0f,
        0.0f,
        0.0f};
    apply_scalar_range(
        &uniforms,
        point_mode == TiePointColorMode::ImageCount
            ? _tiePointImageCountRange
            : _tiePointElevationRange);
    drawPointCloud(cb, uniforms);

    uniforms.lightDirPointSize[3] = 1.0f;
    uniforms.renderModeFlags = {
        float(static_cast<int>(_modelColorMode)),
        _preparedMesh.hasVertexColors ? 1.0f : 0.0f,
        0.0f,
        0.0f};
    apply_scalar_range(&uniforms, _modelElevationRange);
    if (_modelColorMode == ModelColorMode::Wireframe && _meshHasFaces)
    {
        drawIndexedRhiBuffer(cb,
                             &_meshBuffer,
                             &_meshWireframeIndices,
                             &_meshWireframePipeline,
                             uniforms);
    }
    else if (_modelColorMode == ModelColorMode::Texture && _meshHasTexture)
    {
        drawTexturedMesh(cb, uniforms);
    }
    else if (_meshHasFaces)
    {
        drawIndexedRhiBuffer(cb,
                             &_meshBuffer,
                             &_meshTriangleIndices,
                             &_meshTrianglePipeline,
                             uniforms);
    }

    uniforms.renderModeFlags = {};
    uniforms.scalarRange = {};
    drawRhiBuffer(cb, &_lineBuffer, &_colorLinePipeline, uniforms);
}

void CameraSceneWidget::render(QRhiCommandBuffer *cb)
{
    if (!_rhiReady || !rhi() || !renderTarget())
    {
        if (_renderError.isEmpty())
        {
            _renderError = QStringLiteral("Vulkan 渲染初始化失败，请检查显卡驱动和 Qt Vulkan 支持。");
        }
        requestOverlayUpdate();
        return;
    }

    if (_gpuDirty)
    {
        if (!uploadGpuData())
        {
            requestOverlayUpdate();
            return;
        }
    }
    if (!_geometryUploadError.isEmpty())
    {
        _renderError = _geometryUploadError;
        requestOverlayUpdate();
        return;
    }
    if (_manualHighlightBuffersReleasePending)
    {
        _manualHighlightPointBuffer.vertexBuffer.reset();
        _manualHighlightScalarBuffer.vertexBuffer.reset();
        _manualHighlightBuffersReleasePending = false;
    }

    if (!ensurePointPipeline(&_colorPointPipeline, false) ||
        !ensurePointPipeline(&_highlightPointPipeline, true) ||
        !ensurePipeline(&_colorLinePipeline,
                        int(QRhiGraphicsPipeline::Lines),
                        6 * int(sizeof(float)),
                        false) ||
        !ensurePipeline(&_meshTrianglePipeline,
                        int(QRhiGraphicsPipeline::Triangles),
                        _meshBuffer.strideBytes > 0
                            ? _meshBuffer.strideBytes
                            : 9 * int(sizeof(float)),
                        true) ||
        !ensurePipeline(&_meshWireframePipeline,
                        int(QRhiGraphicsPipeline::Lines),
                        _meshBuffer.strideBytes > 0
                            ? _meshBuffer.strideBytes
                            : 9 * int(sizeof(float)),
                        true))
    {
        requestOverlayUpdate();
        return;
    }
    QRhiResourceUpdateBatch *updates = rhi()->nextResourceUpdateBatch();
    _thumbnailPoseIndicesPendingCommit.clear();
    _thumbnailCacheKeysPendingCommit.clear();
    auto abort_update_batch = [this, updates]()
    {
        // The batch owns references to every queued upload. Release it before
        // resetting any QRhi resource that may be referenced by those uploads.
        if (updates)
        {
            updates->release();
        }
        rollbackResourceUpdateState();
        requestOverlayUpdate();
    };
    if (!ensureRhiBuffer(&_pointBuffer, updates) ||
        !ensureRhiBuffer(&_pointScalarBuffer, updates) ||
        !ensureRhiBuffer(&_manualHighlightPointBuffer, updates) ||
        !ensureRhiBuffer(&_manualHighlightScalarBuffer, updates) ||
        !ensureRhiBuffer(&_meshBuffer, updates) ||
        !ensureRhiBuffer(&_texturedMeshBuffer, updates) ||
        !ensureRhiBuffer(&_lineBuffer, updates) ||
        !ensureRhiIndexBuffer(&_meshTriangleIndices, updates) ||
        !ensureRhiIndexBuffer(&_meshWireframeIndices, updates))
    {
        abort_update_batch();
        return;
    }
    if (!ensureTexturedMeshPipeline(updates)
        || !ensureCameraThumbnailPipeline(updates)
        || !ensureImagePipeline(updates))
    {
        abort_update_batch();
        return;
    }
    _pipelinesDirty = false;
    _renderError.clear();

    const SceneMatrices matrices = sceneMatrices();
    QMatrix4x4 shift;
    shift.setToIdentity();
    shift.translate(float(2.0 * _sceneOffsetPx.x() / qMax(1, width())),
                    float(-2.0 * _sceneOffsetPx.y() / qMax(1, height())),
                    0.0f);
    const QMatrix4x4 mv = matrices.modelView;
    const QMatrix4x4 mvp = rhi()->clipSpaceCorrMatrix() * shift * matrices.projection * mv;

    cb->beginPass(renderTarget(),
                  QColor::fromRgbF(1.0f, 1.0f, 1.0f, 1.0f),
                  QRhiDepthStencilClearValue(1.0f, 0),
                  updates);
    commitResourceUpdateState();
    const QSize pixelSize = renderTarget()->pixelSize();
    cb->setViewport(QRhiViewport(0.0f, 0.0f, float(pixelSize.width()), float(pixelSize.height())));

    SceneUniforms uniforms;
    std::copy_n(mvp.constData(), 16, uniforms.mvp.begin());
    std::copy_n(mv.constData(), 16, uniforms.modelView.begin());
    QMatrix4x4 normal_matrix;
    normal_matrix.setToIdentity();
    const QMatrix3x3 normal3x3 = mv.normalMatrix();
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            normal_matrix(row, col) = normal3x3(row, col);
        }
    }
    std::copy_n(normal_matrix.constData(), 16, uniforms.normalMatrix.begin());
    if (_cameraImageDisplayLayer == CameraImageDisplayLayer::Background)
    {
        drawActiveCameraImage(cb, mvp);
    }
    drawSceneGeometry(cb, uniforms);
    // 相机平面最后参与同一个深度缓冲。它只覆盖实际位于其后的点，
    // 并在共面时优先保留照片，避免连接点从照片表面穿透出来。
    drawCameraThumbnails(cb, mvp, mv);
    if (_cameraImageDisplayLayer == CameraImageDisplayLayer::Foreground)
    {
        drawActiveCameraImage(cb, mvp);
    }

    cb->endPass();

    requestOverlayUpdate();
}

void CameraSceneWidget::updateCameraOverlay()
{
    update();
    requestOverlayUpdate();
}

void CameraSceneWidget::requestOverlayUpdate()
{
    if (!_overlayWidget)
    {
        return;
    }
    _overlayWidget->setGeometry(rect());
    _overlayWidget->raise();
    _overlayWidget->update();
}

void CameraSceneWidget::drawRotationGizmo(QPainter &painter) const
{
    if (!_showGizmo)
    {
        return;
    }

    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();
    QRadialGradient gradient(
        center2d - QPointF(radiusPx * 0.18, radiusPx * 0.18),
        radiusPx * 1.25);
    gradient.setColorAt(0.0, QColor(245, 245, 248, 40));
    gradient.setColorAt(1.0, QColor(175, 178, 186, 28));
    painter.setPen(QPen(QColor(210, 210, 216, 44), 1.0));
    painter.setBrush(gradient);
    painter.drawEllipse(center2d, radiusPx, radiusPx);

    const auto axisPen = [this](HoverAxis axis, const QColor &base)
    {
        const bool highlighted =
            (_hoverAxis == axis) || (_dragAxis == axis && _leftDragging);
        QColor color = base;
        if (highlighted)
        {
            color = color.lighter(150);
        }
        return QPen(color, highlighted ? 4.0 : 2.0);
    };
    const auto drawGreatCircle = [&](HoverAxis axis, const QColor &color)
    {
        painter.setPen(axisPen(axis, color));
        QPointF previous;
        QPointF first;
        bool hasPrevious = false;
        bool previousVisible = false;
        bool firstVisible = false;
        for (int index = 0; index <= 128; ++index)
        {
            const qreal angle = (2.0 * M_PI * index) / 128.0;
            QVector3D localPoint;
            if (axis == HoverAxis::X)
            {
                localPoint = QVector3D(0.0f, float(std::cos(angle)), float(std::sin(angle)));
            }
            else if (axis == HoverAxis::Y)
            {
                localPoint = QVector3D(float(std::cos(angle)), 0.0f, float(std::sin(angle)));
            }
            else
            {
                localPoint = QVector3D(float(std::cos(angle)), float(std::sin(angle)), 0.0f);
            }
            const QVector3D viewPoint = applyViewRotation(localPoint);
            const bool visible = viewPoint.z() > 0.0f;
            const QPointF current = center2d + QPointF(
                viewPoint.x() * radiusPx,
                -viewPoint.y() * radiusPx);
            if (!hasPrevious)
            {
                first = current;
                firstVisible = visible;
            }
            else if (previousVisible && visible)
            {
                painter.drawLine(previous, current);
            }
            previous = current;
            previousVisible = visible;
            hasPrevious = true;
        }
        if (hasPrevious && firstVisible && previousVisible)
        {
            painter.drawLine(previous, first);
        }
    };
    drawGreatCircle(HoverAxis::X, QColor(255, 110, 110, 150));
    drawGreatCircle(HoverAxis::Y, QColor(110, 255, 150, 150));
    drawGreatCircle(HoverAxis::Z, QColor(110, 170, 255, 150));
}

void CameraSceneWidget::paintOverlay(QPainter &painter)
{
    if (!painter.isActive())
    {
        return;
    }

    if (!_renderError.isEmpty())
    {
        painter.fillRect(rect(), Qt::white);
        painter.setPen(QColor(180, 42, 42));
        painter.drawText(rect().adjusted(12, 12, -12, -12),
                         Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap,
                         _renderError);
        return;
    }

    // 前景照片是最终遮挡层。Vulkan 场景先被不透明照片覆盖，随后绘制的
    // QWidget 叠加内容也必须使用同一照片投影区域裁剪，不能再次穿透照片。
    painter.save();
    const QPainterPath foregroundImageOcclusion = foregroundCameraImageOcclusionPath();
    if (!foregroundImageOcclusion.isEmpty())
    {
        QPainterPath visibleScene;
        visibleScene.addRect(QRectF(rect()));
        visibleScene = visibleScene.subtracted(foregroundImageOcclusion);
        painter.setClipPath(visibleScene, Qt::IntersectClip);
    }
    const bool interactive_camera_motion = _leftDragging || _middleDragging;

    // 点云完全由 Vulkan 点图元管线绘制；覆盖层只保留交互控件和文字，
    // 避免在每帧中通过 QPainter 再次遍历全部点。

    if (_showCameras)
    {
        const int labelBudget = maxVisibleCameraLabels();
        const int cameraCount = static_cast<int>(_poses.size());
        const bool drawAllCameraLabels = _poses.size() <= maxVisibleCameraLabels();
        const int cameraLabelStride = drawAllCameraLabels
            ? 1
            : qMax(1, static_cast<int>(std::ceil(double(cameraCount) / double(qMax(1, labelBudget)))));

        if (_poses.isEmpty())
        {
            painter.setPen(QColor(120, 120, 120));
            painter.drawText(rect(), Qt::AlignCenter, tr("暂无相机参数，显示默认模型球"));
        }

        // 相机卡片和方位线已在 Vulkan 三维场景中批量绘制。覆盖层只保留
        // 少量文件名；拖动时跳过全部文字布局，避免影响轨迹球帧率。
        if (!interactive_camera_motion)
        {
            const QMatrix4x4 camera_model_view = sceneMatrices().modelView;
            for (qsizetype poseIndex = 0; poseIndex < _poses.size(); ++poseIndex)
            {
                const CameraPose &pose = _poses.at(poseIndex);
                const bool highlighted = isCameraHighlighted(pose);
                const bool drawCameraLabel = highlighted
                    || drawAllCameraLabels
                    || poseIndex == 0
                    || poseIndex == _poses.size() - 1
                    || poseIndex % cameraLabelStride == 0;
                if (!drawCameraLabel)
                {
                    continue;
                }

                bool centerOk = false;
                const QPointF center = projectToScreen(pose.center, &centerOk);
                if (!centerOk)
                {
                    continue;
                }
                QVector3D leaderStart;
                QVector3D leaderEnd;
                const float halfExtent = cameraImagePlaneHalfExtent(
                    pose, camera_model_view);
                const bool hasLeader = cameraDirectionLeaderSegment(
                    pose, halfExtent, &leaderStart, &leaderEnd);
                bool leaderEndOk = false;
                const QPointF leaderEndScreen = hasLeader
                    ? projectToScreen(leaderEnd, &leaderEndOk)
                    : QPointF();
                const QPointF labelAnchor = leaderEndOk ? leaderEndScreen : center;
                const bool placeLabelLeft = leaderEndOk && leaderEndScreen.x() < center.x();
                const QString labelSource = pose.imagePath.isEmpty() ? pose.name : pose.imagePath;
                const QString label = QFileInfo(labelSource).fileName().isEmpty()
                    ? pose.name
                    : QFileInfo(labelSource).fileName();
                const qreal labelWidth = painter.fontMetrics().horizontalAdvance(label);
                const QPointF textOffset = placeLabelLeft
                    ? QPointF(-labelWidth - 5.0, -2.0)
                    : QPointF(5.0, -2.0);
                painter.setPen(highlighted
                    ? QColor(210, 45, 65, 230)
                    : (drawAllCameraLabels ? QColor(60, 60, 60) : QColor(45, 45, 45, 170)));
                painter.drawText(labelAnchor + textOffset, label);
            }
        }

    }

    // 操控球是交互前景层，必须在点云和相机标注之后绘制，避免缩小时
    // 被高密度点云遮挡而无法识别或拖动。
    drawRotationGizmo(painter);
    drawFloorPivotCross(painter);
    painter.restore();

    if (!interactive_camera_motion)
    {
        drawTiePointLegend(painter);
        drawModelLegend(painter);
    }

    const QPoint origin(width() - 64, height() - 64);
    const QVector3D ex = applyViewRotation(QVector3D(1, 0, 0)).normalized();
    const QVector3D ey = applyViewRotation(QVector3D(0, 1, 0)).normalized();
    const QVector3D ez = applyViewRotation(QVector3D(0, 0, 1)).normalized();
    auto drawMiniAxis = [&](const QVector3D &dir, const QColor &color, const QString &label) {
        const QPoint end(origin.x() + int(dir.x() * 28.0f), origin.y() - int(dir.y() * 28.0f));
        painter.setPen(QPen(color, 2));
        painter.drawLine(origin, end);
        painter.setPen(color);
        painter.drawText(end + QPoint(4, -2), label);
    };
    painter.setPen(QPen(QColor(80, 80, 80), 1));
    painter.setBrush(QColor(80, 80, 80));
    painter.drawEllipse(QPointF(origin), 2.5, 2.5);
    drawMiniAxis(ex, QColor(210, 50, 50), QStringLiteral("X"));
    drawMiniAxis(ey, QColor(30, 160, 60), QStringLiteral("Y"));
    drawMiniAxis(ez, QColor(40, 100, 220), QStringLiteral("Z"));
    painter.setPen(QColor(100, 100, 110));
    const QVector3D euler = eulerAnglesDeg();
    painter.drawText(origin + QPoint(-84, 26),
                     QStringLiteral("Yaw %1°  Pitch %2°  Roll %3°")
                         .arg(QString::number(euler.y(), 'f', 1))
                         .arg(QString::number(euler.x(), 'f', 1))
                         .arg(QString::number(euler.z(), 'f', 1)));

    if (_manualPruneMode)
    {
        painter.setPen(QPen(QColor(255, 90, 90, 220), 1.5, Qt::DashLine));
        painter.setBrush(QColor(255, 90, 90, 40));
        if (!_manualSelectRect.isNull())
        {
            painter.drawRect(_manualSelectRect.normalized());
        }

        painter.setPen(QColor(235, 80, 80));
        QString manual_status = tr(
            "手动剔除模式：右键框选高亮，前进侧键删除，Ctrl+Z 撤销（已选 %1）")
            .arg(static_cast<int>(_manualPreviewIndices.size()));
        if (_manualSelectionRunning)
        {
            manual_status = tr("正在后台计算框选点...");
        }
        else if (_manualEditRunning)
        {
            manual_status = tr("正在后台更新并保存点云...");
        }
        painter.drawText(QPointF(14.0, 24.0), manual_status);
    }

    drawPlyLoadProgressOverlay(painter);
    drawCameraThumbnailProgressOverlay(painter);
}

void CameraSceneWidget::drawTiePointLegend(QPainter &painter) const
{
    if (!_isTiePointCloud || _tiePointColorMode == TiePointColorMode::Color ||
        _cloud.size() == 0)
    {
        return;
    }

    const bool elevationMode = _tiePointColorMode == TiePointColorMode::Elevation;
    const bool imageCountReady =
        _tiePointImageCounts.size() == static_cast<qsizetype>(_cloud.size());
    if (!elevationMode && !imageCountReady)
    {
        const QString status = _tiePointMetadataLoading
            ? tr("影像数：正在读取观测数据...")
            : tr("影像数：%1").arg(
                  _tiePointMetadataError.isEmpty() ? tr("无观测数据")
                                                   : _tiePointMetadataError);
        const QRectF statusRect(24.0, height() - 58.0, 310.0, 30.0);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 255, 255, 220));
        painter.drawRoundedRect(statusRect, 4.0, 4.0);
        painter.setPen(QColor(85, 85, 90));
        painter.drawText(statusRect.adjusted(10.0, 0.0, -8.0, 0.0),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         status);
        return;
    }

    const xjw::gui::tie_points::ScalarRange range =
        elevationMode ? _tiePointElevationRange : _tiePointImageCountRange;
    if (!range.isValid())
    {
        return;
    }

    const qreal legendHeight = qBound<qreal>(112.0, height() * 0.22, 188.0);
    const QRectF panel(22.0, height() - legendHeight - 70.0, 174.0, legendHeight + 48.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.top() + 28.0,
                     20.0,
                     legendHeight);

    painter.setPen(QPen(QColor(205, 205, 210, 190), 1.0));
    painter.setBrush(QColor(255, 255, 255, 220));
    painter.drawRoundedRect(panel, 5.0, 5.0);

    QLinearGradient gradient(bar.topLeft(), bar.bottomLeft());
    constexpr int colorStopCount = 5;
    for (int stopIndex = 0; stopIndex < colorStopCount; ++stopIndex)
    {
        const double position =
            static_cast<double>(stopIndex) / static_cast<double>(colorStopCount - 1);
        const double rampValue = elevationMode ? 1.0 - position : position;
        gradient.setColorAt(position,
                            xjw::gui::tie_points::scalarRampColor(rampValue));
    }
    painter.fillRect(bar, gradient);
    painter.setPen(QColor(100, 100, 105));
    painter.drawRect(bar);

    painter.setPen(QColor(50, 50, 55));
    painter.drawText(QRectF(panel.left() + 12.0,
                            panel.top() + 4.0,
                            panel.width() - 24.0,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     elevationMode ? tr("连接点 — 高程 (Z)")
                                   : tr("连接点 — 影像数"));

    auto formatValue = [elevationMode](double value)
    {
        if (!elevationMode)
        {
            return QString::number(qRound(value)) + QStringLiteral(" 张");
        }
        return QString::number(value, 'g', 7);
    };
    const double middle = (range.minimum + range.maximum) * 0.5;
    const qreal labelLeft = bar.right() + 10.0;
    const qreal labelWidth = panel.right() - labelLeft - 6.0;
    painter.drawText(QRectF(labelLeft, bar.top() - 9.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.maximum));
    painter.drawText(QRectF(labelLeft,
                            bar.center().y() - 10.0,
                            labelWidth,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(middle));
    painter.drawText(QRectF(labelLeft, bar.bottom() - 11.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.minimum));
}

void CameraSceneWidget::drawModelLegend(QPainter &painter) const
{
    const bool elevationMode = _modelColorMode == ModelColorMode::Elevation;
    const bool confidenceMode = _modelColorMode == ModelColorMode::Confidence;
    if (_isTiePointCloud || !_cloud.hasFaces() || (!elevationMode && !confidenceMode))
    {
        return;
    }

    const xjw::gui::tie_points::ScalarRange range =
        elevationMode
        ? _modelElevationRange
        : xjw::gui::tie_points::ScalarRange{1.0, 100.0};
    if (!range.isValid())
    {
        return;
    }

    const qreal legendHeight = qBound<qreal>(112.0, height() * 0.22, 188.0);
    const QRectF panel(22.0, height() - legendHeight - 70.0, 174.0, legendHeight + 48.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.top() + 28.0,
                     20.0,
                     legendHeight);

    painter.setPen(QPen(QColor(205, 205, 210, 190), 1.0));
    painter.setBrush(QColor(255, 255, 255, 220));
    painter.drawRoundedRect(panel, 5.0, 5.0);

    QLinearGradient gradient(bar.topLeft(), bar.bottomLeft());
    constexpr int colorStopCount = 5;
    for (int stopIndex = 0; stopIndex < colorStopCount; ++stopIndex)
    {
        const double position =
            static_cast<double>(stopIndex) / static_cast<double>(colorStopCount - 1);
        const QColor color = elevationMode
            ? xjw::gui::tie_points::scalarRampColor(1.0 - position)
            : xjw::gui::tie_points::imageCountColor(
                  qRound(100.0 - position * 99.0), range);
        gradient.setColorAt(position, color);
    }
    painter.fillRect(bar, gradient);
    painter.setPen(QColor(100, 100, 105));
    painter.drawRect(bar);

    painter.setPen(QColor(50, 50, 55));
    painter.drawText(QRectF(panel.left() + 12.0,
                            panel.top() + 4.0,
                            panel.width() - 24.0,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     elevationMode ? tr("模型 — 高程 (Z)")
                                   : tr("模型 — 可信度"));

    const auto formatValue = [elevationMode](double value)
    {
        return elevationMode
            ? QString::number(value, 'g', 7)
            : QString::number(qRound(value));
    };
    const double middle = (range.minimum + range.maximum) * 0.5;
    const qreal labelLeft = bar.right() + 10.0;
    const qreal labelWidth = panel.right() - labelLeft - 6.0;
    painter.drawText(QRectF(labelLeft, bar.top() - 9.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.maximum));
    painter.drawText(QRectF(labelLeft,
                            bar.center().y() - 10.0,
                            labelWidth,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(middle));
    painter.drawText(QRectF(labelLeft, bar.bottom() - 11.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.minimum));
}

void CameraSceneWidget::drawPlyLoadProgressOverlay(QPainter &painter)
{
    if (!_loading || _plyLoadProgressPercent < 0)
    {
        return;
    }

    const int panelWidth = qMin(width() - 48, 520);
    if (panelWidth <= 160 || height() <= 100)
    {
        return;
    }

    const QRectF panel(24.0, height() - 72.0, panelWidth, 48.0);
    const QRectF bar(panel.left() + 16.0, panel.bottom() - 16.0, panel.width() - 32.0, 6.0);
    const qreal fillWidth = bar.width() * qBound(0, _plyLoadProgressPercent, 100) / 100.0;

    painter.save();
    painter.setPen(QPen(QColor(70, 82, 96, 160), 1.0));
    painter.setBrush(QColor(250, 252, 255, 235));
    painter.drawRoundedRect(panel, 6.0, 6.0);

    painter.setPen(QColor(34, 48, 68));
    const QString title = _plyLoadProgressText.isEmpty()
        ? tr("正在加载密集点云...")
        : _plyLoadProgressText;
    painter.drawText(QRectF(panel.left() + 16.0,
                            panel.top() + 8.0,
                            panel.width() - 96.0,
                            20.0),
                     Qt::AlignVCenter | Qt::AlignLeft,
                     title);
    painter.drawText(QRectF(panel.right() - 70.0,
                            panel.top() + 8.0,
                            54.0,
                            20.0),
                     Qt::AlignVCenter | Qt::AlignRight,
                     QStringLiteral("%1%").arg(qBound(0, _plyLoadProgressPercent, 100)));

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(218, 226, 238));
    painter.drawRoundedRect(bar, 3.0, 3.0);
    painter.setBrush(QColor(36, 115, 218));
    painter.drawRoundedRect(QRectF(bar.left(), bar.top(), fillWidth, bar.height()), 3.0, 3.0);
    painter.restore();
}

void CameraSceneWidget::drawCameraThumbnailProgressOverlay(QPainter &painter) const
{
    if (!_showCameras || !_showCameraThumbnails || _cameraThumbnailLoadTotal <= 0
        || _cameraThumbnailLoadCompleted >= _cameraThumbnailLoadTotal)
    {
        return;
    }

    const int completed = qBound(
        0, _cameraThumbnailLoadCompleted, _cameraThumbnailLoadTotal);
    const int panel_width = qMin(width() - 48, 360);
    if (panel_width <= 160 || height() <= 100)
    {
        return;
    }

    const QRectF panel(width() - panel_width - 24.0, 24.0, panel_width, 46.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.bottom() - 14.0,
                     panel.width() - 28.0,
                     6.0);
    const qreal fill_width = bar.width()
        * static_cast<qreal>(completed)
        / static_cast<qreal>(_cameraThumbnailLoadTotal);

    painter.save();
    painter.setPen(QPen(QColor(70, 82, 96, 150), 1.0));
    painter.setBrush(QColor(250, 252, 255, 230));
    painter.drawRoundedRect(panel, 6.0, 6.0);
    painter.setPen(QColor(34, 48, 68));
    painter.drawText(panel.adjusted(14.0, 4.0, -14.0, -16.0),
                     Qt::AlignVCenter | Qt::AlignLeft,
                     tr("正在加载相机影像 %1/%2")
                         .arg(completed)
                         .arg(_cameraThumbnailLoadTotal));
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(218, 226, 238));
    painter.drawRoundedRect(bar, 3.0, 3.0);
    painter.setBrush(QColor(36, 115, 218));
    painter.drawRoundedRect(
        QRectF(bar.left(), bar.top(), fill_width, bar.height()), 3.0, 3.0);
    painter.restore();
}

void CameraSceneWidget::clearManualPointSelection()
{
    ++_manualSelectionGeneration;
    if (_manualSelectionCancellation)
    {
        _manualSelectionCancellation->store(true, std::memory_order_relaxed);
    }
    _pendingManualSelection.reset();
    _manualSelecting = false;
    _manualSelectionRunning = false;
    _manualDeletePending = false;
    _manualSelectRect = QRect();
    _manualPreviewIndices.clear();
    _manualPreviewValid = false;
    _manualPreviewUsesScreenRect = false;
    _manualHighlightPointBuffer.vertexData.clear();
    _manualHighlightPointBuffer.vertexCount = 0;
    _manualHighlightPointBuffer.dirty = true;
    _manualHighlightScalarBuffer.vertexData.clear();
    _manualHighlightScalarBuffer.vertexCount = 0;
    _manualHighlightScalarBuffer.dirty = true;
    _manualHighlightBuffersReleasePending = true;
}

bool CameraSceneWidget::setManualPruneModeEnabled(bool enabled, QString *errorMessage)
{
    if (_manualEditRunning)
    {
        if (errorMessage)
        {
            *errorMessage = tr("点云正在后台更新，请等待当前操作完成。");
        }
        return false;
    }
    if (enabled)
    {
        if ((_cloud.size() == 0))
        {
            if (errorMessage)
            {
                *errorMessage = tr("当前未加载点云数据。");
            }
            return false;
        }
        if (_cloud.hasFaces())
        {
            if (errorMessage)
            {
                *errorMessage = tr("当前为网格模型，手动剔除仅支持点云。");
            }
            return false;
        }
        if (_isTiePointCloud && _tiePointMetadataLoading)
        {
            if (errorMessage)
            {
                *errorMessage = tr("连接点观测元数据仍在加载，请稍候。");
            }
            return false;
        }
    }

    _manualPruneMode = enabled;
    clearManualPointSelection();
    if (!enabled)
    {
        _manualUndoStack.clear();
    }
    updateCursor();
    update();
    return true;
}

void CameraSceneWidget::startManualPointSelection(const QRect &screenRect)
{
    const QRect selection_rect = screenRect.normalized().intersected(rect());
    clearManualPointSelection();
    _manualSelectRect = selection_rect;
    const int generation = _manualSelectionGeneration;
    const int load_generation = _loadGen;
    if (!_manualPruneMode || _manualEditRunning
        || selection_rect.width() < 3 || selection_rect.height() < 3)
    {
        _manualSelectionRunning = false;
        _manualPreviewValid = true;
        update();
        return;
    }
    if (_isTiePointCloud && _tiePointMetadataLoading)
    {
        _manualSelectionRunning = false;
        setProperty("lastAsyncTaskError",
                    tr("连接点观测元数据仍在加载，请稍候。"));
        update();
        return;
    }

    QByteArray vertex_data;
    QByteArray scalar_data;
    int stride_bytes = 9 * int(sizeof(float));
    if (_preparedPointBuffer
        && _preparedPointVertexCount == static_cast<int>(_cloud.size())
        && !_preparedPointVertexData.isEmpty())
    {
        vertex_data = _preparedPointVertexData;
    }
    else if (_pointBuffer.vertexCount == static_cast<int>(_cloud.size())
             && !_pointBuffer.vertexData.isEmpty())
    {
        vertex_data = _pointBuffer.vertexData;
        stride_bytes = _pointBuffer.strideBytes;
    }
    if (vertex_data.isEmpty() || stride_bytes < 3 * int(sizeof(float)))
    {
        _manualSelectionRunning = false;
        _manualPreviewValid = true;
        update();
        return;
    }
    if (_pointScalarBuffer.vertexCount == static_cast<int>(_cloud.size())
        && _pointScalarBuffer.vertexData.size()
            == _pointScalarBuffer.vertexCount * int(sizeof(float)))
    {
        scalar_data = _pointScalarBuffer.vertexData;
    }

    const SceneMatrices matrices = sceneMatrices();
    const QMatrix4x4 clip_matrix = matrices.projection * matrices.modelView;
    const QSize viewport_size(qMax(1, width()), qMax(1, height()));
    const QPointF scene_offset = _sceneOffsetPx;
    ManualSelectionRequest request;
    request.vertexData = std::move(vertex_data);
    request.scalarData = std::move(scalar_data);
    request.screenRect = selection_rect;
    request.clipMatrix = clip_matrix;
    request.viewportSize = viewport_size;
    request.sceneOffset = scene_offset;
    request.strideBytes = stride_bytes;
    request.generation = generation;
    request.loadGeneration = load_generation;
    _pendingManualSelection = std::move(request);
    _manualSelectionRunning = true;
    update();
    pumpManualPointSelection();
}

void CameraSceneWidget::pumpManualPointSelection()
{
    if (_manualSelectionWorkerActive || !_pendingManualSelection)
    {
        return;
    }

    ManualSelectionRequest request = std::move(*_pendingManualSelection);
    _pendingManualSelection.reset();
    auto cancellation = std::make_shared<std::atomic_bool>(false);
    _manualSelectionCancellation = cancellation;
    _manualSelectionWorkerActive = true;
    const int generation = request.generation;
    const int load_generation = request.loadGeneration;
    const int stride_bytes = request.strideBytes;
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [request = std::move(request), cancellation]()
        {
            return preparePointSelection(request.vertexData,
                                         request.strideBytes,
                                         request.scalarData,
                                         request.screenRect,
                                         request.clipMatrix,
                                         request.viewportSize,
                                         request.sceneOffset,
                                         kMaximumCompactSelectionPoints,
                                         cancellation.get());
        },
        [generation, load_generation, stride_bytes, cancellation](
            CameraSceneWidget *self,
            xjw::gui::tasks::TaskOutcome<PointSelectionPreparation> outcome)
        {
            self->_manualSelectionWorkerActive = false;
            const bool is_current = generation == self->_manualSelectionGeneration
                && load_generation == self->_loadGen
                && cancellation == self->_manualSelectionCancellation
                && !cancellation->load(std::memory_order_relaxed)
                && self->_manualPruneMode;
            if (!is_current)
            {
                outcome.value.reset();
                self->pumpManualPointSelection();
                return;
            }
            self->_manualSelectionRunning = false;
            if (outcome.succeeded())
            {
                PointSelectionPreparation selection = std::move(*outcome.value);
                const bool has_selected_points = !selection.indices.empty();
                const bool has_compact_highlight = selection.pointCount > 0;
                self->_manualPreviewIndices = std::move(selection.indices);
                self->_manualPreviewUsesScreenRect =
                    has_selected_points && !has_compact_highlight;
                self->_manualHighlightPointBuffer.vertexData =
                    std::move(selection.vertexData);
                self->_manualHighlightPointBuffer.vertexCount = selection.pointCount;
                self->_manualHighlightPointBuffer.strideBytes = stride_bytes;
                self->_manualHighlightPointBuffer.dirty = true;
                self->_manualHighlightScalarBuffer.vertexData =
                    std::move(selection.scalarData);
                self->_manualHighlightScalarBuffer.vertexCount = selection.pointCount;
                self->_manualHighlightScalarBuffer.strideBytes = int(sizeof(float));
                self->_manualHighlightScalarBuffer.dirty = true;
                self->_manualPreviewValid = true;
            }
            else
            {
                self->clearManualPointSelection();
                self->setProperty("lastAsyncTaskError", outcome.errorMessage);
            }

            const bool apply_pending_delete = self->_manualDeletePending;
            self->_manualDeletePending = false;
            if (apply_pending_delete && self->_manualPreviewValid)
            {
                self->startManualPruneApply();
            }
            self->updateCursor();
            self->update();
            outcome.value.reset();
            self->pumpManualPointSelection();
        });
}

void CameraSceneWidget::pushManualUndoDelta(PointCloudEditDelta delta)
{
    if (!_manualPruneMode || !delta.isValid())
    {
        return;
    }
    _manualUndoStack.push_back(std::move(delta));
    if (static_cast<int>(_manualUndoStack.size()) > _manualUndoLimit)
    {
        _manualUndoStack.erase(_manualUndoStack.begin());
    }
}

void CameraSceneWidget::applyManualPointCloudResult(
    PointCloudEditResult result,
    const xjw::gui::tie_points::ScalarRange &imageCountRange)
{
    if (!result.isValid())
    {
        return;
    }

    _cloud = std::move(*result.cloud);
    _tiePointImageCounts = std::move(result.imageCounts);
    if (_isTiePointCloud)
    {
        _tiePointImageCountRange = imageCountRange;
        _tiePointScalarData = result.renderPreparation.scalarData;
    }
    else
    {
        _tiePointScalarData.clear();
        _tiePointImageCountRange = {};
    }
    applyPointPreparation(std::move(result.renderPreparation));
    clearManualPointSelection();
    _gpuDirty = true;
    update();
}

void CameraSceneWidget::startManualPruneApply()
{
    if (!_manualPruneMode || _manualEditRunning || !_manualPreviewValid
        || _manualPreviewIndices.empty() || _cloud.size() == 0)
    {
        return;
    }
    if (_isTiePointCloud && _tiePointMetadataLoading)
    {
        setProperty("lastAsyncTaskError",
                    tr("连接点观测元数据仍在加载，请稍候。"));
        return;
    }

    std::vector<PointVertexIndex> selected_indices = std::move(_manualPreviewIndices);
    const int load_generation = _loadGen;
    const QString path = _currentCloudPath;
    const QVector<int> image_counts = _tiePointImageCounts;
    auto source = std::make_shared<RenderCloud>(std::move(_cloud));
    auto cancellation = std::make_shared<std::atomic_bool>(false);
    _manualEditCancellation = cancellation;
    clearManualPointSelection();
    _manualEditRunning = true;
    update();

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [source,
         selected_indices = std::move(selected_indices),
         image_counts,
         path,
         cancellation]() mutable
        {
            ManualPointCloudTaskResult task_result;
            task_result.edit = filterPointCloudWithDelta(
                source,
                std::move(selected_indices),
                image_counts,
                cancellation.get());
            if (task_result.edit.isValid())
            {
                task_result.imageCountRange = imageCountRangeFor(
                    task_result.edit.imageCounts,
                    task_result.edit.cloud->size());
                task_result.save = stagePointCloudSnapshot(
                    path, *task_result.edit.cloud, cancellation.get());
            }
            return task_result;
        },
        [source, load_generation, cancellation](
            CameraSceneWidget *self,
            xjw::gui::tasks::TaskOutcome<ManualPointCloudTaskResult> outcome) mutable
        {
            self->_manualEditRunning = false;
            if (self->_manualEditCancellation == cancellation)
            {
                self->_manualEditCancellation.reset();
            }
            if (load_generation != self->_loadGen)
            {
                if (outcome.succeeded())
                {
                    outcome.value->save.discard();
                }
                return;
            }
            if (!outcome.succeeded() || !outcome.value->edit.isValid())
            {
                self->_cloud = std::move(*source);
                const QString error_message = outcome.succeeded()
                    ? self->tr("点云删除未生成有效结果。")
                    : outcome.errorMessage;
                self->setProperty("lastAsyncTaskError", error_message);
                emit self->manualPruneSaveFailed(error_message);
                self->update();
                return;
            }

            ManualPointCloudTaskResult task_result = std::move(*outcome.value);
            if (task_result.save.success)
            {
                QString save_error;
                if (!task_result.save.commit(&save_error))
                {
                    task_result.save.success = false;
                    task_result.save.errorMessage = save_error;
                }
            }
            const int removed_count = static_cast<int>(
                task_result.edit.undo.removedIndices.size());
            PointCloudEditDelta undo = std::move(task_result.edit.undo);
            self->applyManualPointCloudResult(
                std::move(task_result.edit),
                task_result.imageCountRange);
            self->pushManualUndoDelta(std::move(undo));
            emit self->manualPruneApplied(
                removed_count,
                static_cast<int>(self->_cloud.size()));
            if (task_result.save.success)
            {
                emit self->manualPruneSaved(
                    task_result.save.path,
                    task_result.save.pointCount);
            }
            else
            {
                emit self->manualPruneSaveFailed(task_result.save.errorMessage);
            }
        });
}

bool CameraSceneWidget::undoLastManualPrune(QString *errorMessage)
{
    if (!_manualPruneMode)
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前未处于手动剔除模式。");
        }
        return false;
    }
    if (_manualEditRunning)
    {
        if (errorMessage)
        {
            *errorMessage = tr("点云正在后台更新，请等待当前操作完成。");
        }
        return false;
    }
    if (_isTiePointCloud && _tiePointMetadataLoading)
    {
        if (errorMessage)
        {
            *errorMessage = tr("连接点观测元数据仍在加载，请稍候。");
        }
        return false;
    }
    if (_manualUndoStack.empty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("没有可撤销的删除操作。");
        }
        return false;
    }

    auto delta = std::make_shared<PointCloudEditDelta>(
        std::move(_manualUndoStack.back()));
    _manualUndoStack.pop_back();
    const int load_generation = _loadGen;
    const QString path = _currentCloudPath;
    const QVector<int> image_counts = _tiePointImageCounts;
    auto source = std::make_shared<RenderCloud>(std::move(_cloud));
    auto cancellation = std::make_shared<std::atomic_bool>(false);
    _manualEditCancellation = cancellation;
    clearManualPointSelection();
    _manualEditRunning = true;
    update();

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [source, delta, image_counts, path, cancellation]
        {
            ManualPointCloudTaskResult task_result;
            task_result.edit = restorePointCloudFromDelta(
                source,
                *delta,
                image_counts,
                cancellation.get());
            if (task_result.edit.isValid())
            {
                task_result.imageCountRange = imageCountRangeFor(
                    task_result.edit.imageCounts,
                    task_result.edit.cloud->size());
                task_result.save = stagePointCloudSnapshot(
                    path, *task_result.edit.cloud, cancellation.get());
            }
            return task_result;
        },
        [source, delta, load_generation, cancellation](
            CameraSceneWidget *self,
            xjw::gui::tasks::TaskOutcome<ManualPointCloudTaskResult> outcome) mutable
        {
            self->_manualEditRunning = false;
            if (self->_manualEditCancellation == cancellation)
            {
                self->_manualEditCancellation.reset();
            }
            if (load_generation != self->_loadGen)
            {
                if (outcome.succeeded())
                {
                    outcome.value->save.discard();
                }
                return;
            }
            if (!outcome.succeeded() || !outcome.value->edit.isValid())
            {
                self->_cloud = std::move(*source);
                self->_manualUndoStack.push_back(std::move(*delta));
                const QString error_message = outcome.succeeded()
                    ? self->tr("点云撤销未生成有效结果。")
                    : outcome.errorMessage;
                self->setProperty("lastAsyncTaskError", error_message);
                emit self->manualPruneSaveFailed(error_message);
                self->update();
                return;
            }

            ManualPointCloudTaskResult task_result = std::move(*outcome.value);
            if (task_result.save.success)
            {
                QString save_error;
                if (!task_result.save.commit(&save_error))
                {
                    task_result.save.success = false;
                    task_result.save.errorMessage = save_error;
                }
            }
            self->applyManualPointCloudResult(
                std::move(task_result.edit),
                task_result.imageCountRange);
            emit self->manualPruneUndone(static_cast<int>(self->_cloud.size()));
            if (task_result.save.success)
            {
                emit self->manualPruneSaved(
                    task_result.save.path,
                    task_result.save.pointCount);
            }
            else
            {
                emit self->manualPruneSaveFailed(task_result.save.errorMessage);
            }
        });
    return true;
}

void CameraSceneWidget::mousePressEvent(QMouseEvent *event)
{
    setFocus();

    if (_manualPruneMode && event->button() == Qt::RightButton)
    {
        if (_manualEditRunning)
        {
            event->accept();
            return;
        }
        clearManualPointSelection();
        _manualSelecting = true;
        _manualSelectStart = event->pos();
        _manualSelectRect = QRect(_manualSelectStart, _manualSelectStart);
        updateCursor();
        event->accept();
        update();
        return;
    }

    if (_manualPruneMode && event->button() == Qt::ForwardButton)
    {
        if (_manualSelectionRunning)
        {
            _manualDeletePending = true;
        }
        else
        {
            startManualPruneApply();
        }
        event->accept();
        update();
        return;
    }

    if (_manualPruneMode
        && (event->button() == Qt::LeftButton
            || event->button() == Qt::MiddleButton))
    {
        clearManualPointSelection();
    }

    _lastMousePos = event->pos();
    if (event->button() == Qt::LeftButton) {
        _leftDragging = true;
        _dragAxis = _hoverAxis;
        // 无论单轴还是自由旋转，均记录按下时的旋转状态
        _viewRotAtPress = _viewRot;
        if (_dragAxis != HoverAxis::None) {
            _dragAxisDir = pickAxisTangent(event->pos(), _dragAxis);
            // 注意：Y 轴环的屏幕切线方向在参数化时与鼠标拖拽方向有符号差，
            // 在多数情况下需要翻转切线方向以使鼠标向右/上时视图按直觉旋转。
            // 仅对 Y 轴做翻转修正，避免对 X/Z 轴产生副作用。
            if (_dragAxis == HoverAxis::Y) _dragAxisDir = -_dragAxisDir;
        } else {
            // Arcball 自由旋转：记录按下那一刻球面坐标
            _arcballPressVector = arcballVector(event->pos());
        }
        updateCursor();
        event->accept();
        return;
    }
    if (event->button() == Qt::MiddleButton) {
        _middleDragging = true;
        updateCursor();
        event->accept();
        return;
    }
    QRhiWidget::mousePressEvent(event);
}

void CameraSceneWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (_manualPruneMode && _manualSelecting && (event->buttons() & Qt::RightButton))
    {
        _manualSelectRect = QRect(_manualSelectStart, event->pos()).normalized();
        _manualPreviewIndices.clear();
        _manualPreviewValid = false;
        event->accept();
        update();
        return;
    }

    const QPoint delta = event->pos() - _lastMousePos;

    if (_leftDragging && (event->buttons() & Qt::LeftButton)) {
        if (_dragAxis == HoverAxis::None) {
            // ── Arcball 自由旋转 ──────────────────────────────────────────────
            // 将当前鼠标投影到球面，计算从按下点到当前点的旋转，
            // 再䈛到按下时保存的初始视图旋转上——
            // 这样球面上最始点击处就会一直跟随鼠标移动。
            const QVector3D v2 = arcballVector(event->pos());
            const QVector3D axis = QVector3D::crossProduct(_arcballPressVector, v2);
            if (axis.lengthSquared() > 1e-10f) {
                const float dot = qBound(-1.0f,
                    QVector3D::dotProduct(_arcballPressVector, v2), 1.0f);
                const float angleDeg = qRadiansToDegrees(std::acos(dot));
                const QQuaternion delta_q =
                    QQuaternion::fromAxisAndAngle(axis.normalized(), angleDeg);
                // 应用到按下时的初始旋转（非增量式，避免浮点漂移）
                _viewRot = (delta_q * _viewRotAtPress).normalized();
            }
        } else {
            // ── 单轴环旋转 ─────────────────────────────────────────────────────
            // 目标：环面的法向方向（即 X/Y/Z 轴在当前世界空间中的指向）固定不动，
            //       环只在自身平面内"自旋"，看起来像环面始终保持水平/垂直。
            // 实现：将本地轴转换到世界空间 axisView，绕 axisView 前乘旋转。
            //       前乘（世界空间旋转）效果：环法向不变，环平面姿态不变，
            //       场景内容（相机等）绕该轴旋转。
            const QVector2D d(float(delta.x()), float(delta.y()));
            const float scalar = QVector2D::dotProduct(d, _dragAxisDir);
            const float ang = scalar * 0.35f;
            // 取该环的本地法向轴，转换为当前视图下的世界方向
            QVector3D localAxis;
            if (_dragAxis == HoverAxis::X)      localAxis = QVector3D(1.0f, 0.0f, 0.0f);
            else if (_dragAxis == HoverAxis::Y) localAxis = QVector3D(0.0f, 1.0f, 0.0f);
            else                                  localAxis = QVector3D(0.0f, 0.0f, 1.0f);
            const QVector3D axisWorld = applyViewRotation(localAxis).normalized();
            // 绕世界空间轴前乘：new_rot = qa_world * old_rot
            // 这样环的法向量方向(axisWorld)在此次旋转后保持恒定
            const QQuaternion qa = QQuaternion::fromAxisAndAngle(axisWorld, ang);
            _viewRot = (qa * _viewRot).normalized();
        }
        if (_showCameraImage && !_cameraImageLocked)
        {
            updateActiveCameraForView();
        }
        update();
    } else if (_middleDragging && (event->buttons() & Qt::MiddleButton)) {
        // 中键平移：1:1 映射鼠标像素，无论缩放倍率如何，拖拽同量始终移动同距离
        _sceneOffsetPx += QPointF(delta.x(), delta.y());
        clampSceneOffset();
        update();
    } else {
        const HoverAxis newHover = pickHoverAxis(event->pos());
        if (newHover != _hoverAxis) {
            _hoverAxis = newHover;
            updateCursor();
            update();
        }
    }

    _lastMousePos = event->pos();
    QRhiWidget::mouseMoveEvent(event);
}

void CameraSceneWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (_manualPruneMode && event->button() == Qt::RightButton)
    {
        _manualSelecting = false;
        _manualSelectRect = _manualSelectRect.normalized();
        startManualPointSelection(_manualSelectRect);
        updateCursor();
        update();
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        _leftDragging = false;
        _dragAxis = HoverAxis::None;
    }
    if (event->button() == Qt::MiddleButton) {
        _middleDragging = false;
    }
    updateCursor();
    update();
    QRhiWidget::mouseReleaseEvent(event);
}

void CameraSceneWidget::wheelEvent(QWheelEvent *event)
{
    if (_leftDragging || _middleDragging) {
        event->ignore();
        return;
    }

    const QPoint angle = event->angleDelta();
    if (!angle.isNull())
    {
        const double wheel_steps = static_cast<double>(angle.y()) / 120.0;
        const double factor = std::pow(1.10, wheel_steps);
        applyZoomFactor(factor);
    }
    event->accept();
}

void CameraSceneWidget::zoomIn()
{
    applyZoomFactor(1.10);
}

void CameraSceneWidget::zoomOut()
{
    applyZoomFactor(1.0 / 1.10);
}

void CameraSceneWidget::resetView()
{
    if (_manualPruneMode)
    {
        clearManualPointSelection();
    }
    _viewRot = QQuaternion();
    _zoomScale = 1.0;
    _sceneOffsetPx = QPointF();
    _hoverAxis = HoverAxis::None;
    _dragAxis = HoverAxis::None;
    updateCameraOverlay();
}

void CameraSceneWidget::applyZoomFactor(double factor)
{
    if (!std::isfinite(factor) || factor <= 0.0)
    {
        return;
    }

    const double next_zoom_scale = _zoomScale * factor;
    if (!std::isfinite(next_zoom_scale) || next_zoom_scale <= 0.0)
    {
        return;
    }

    if (_manualPruneMode)
    {
        clearManualPointSelection();
    }
    _zoomScale = next_zoom_scale;
    clampSceneOffset();
    update();
}

void CameraSceneWidget::keyPressEvent(QKeyEvent *event)
{
    if (_manualPruneMode && event->matches(QKeySequence::Undo))
    {
        QString errorMessage;
        undoLastManualPrune(&errorMessage);
        event->accept();
        return;
    }
    QRhiWidget::keyPressEvent(event);
}
