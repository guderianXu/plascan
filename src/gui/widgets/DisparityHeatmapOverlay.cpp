// =============================================================================
// 文件: DisparityHeatmapOverlay.cpp
// 功能: 视差热力图叠加实现
// =============================================================================
#include "DisparityHeatmapOverlay.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

#include <QEvent>
#include <QFutureWatcher>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QPainter>
#include <QPointer>
#include <QtConcurrent/QtConcurrent>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ImageViewWidget.h"
#include "io/PathIO.h"

namespace
{

struct HeatmapSettings
{
    float minimum = 0.0f;
    float maximum = 256.0f;
    bool autoRange = true;
    int colormap = cv::COLORMAP_JET;
    bool showInvalid = false;
};

struct HeatmapBuildResult
{
    QImage image;
    QString error;
    bool cancelled = false;
};

struct DisparityLoadResult
{
    cv::Mat disparity;
    QString error;
    bool cancelled = false;
};

bool isCancelled(const std::shared_ptr<std::atomic_bool> &cancellation)
{
    return cancellation && cancellation->load(std::memory_order_relaxed);
}

HeatmapBuildResult buildHeatmapImage(
    const cv::Mat &disparity,
    const HeatmapSettings &settings,
    const std::shared_ptr<std::atomic_bool> &cancellation)
{
    HeatmapBuildResult result;
    if (isCancelled(cancellation))
    {
        result.cancelled = true;
        return result;
    }
    if (disparity.empty() || disparity.channels() != 1)
    {
        result.error = QStringLiteral("视差图必须是非空的单通道图像");
        return result;
    }

    try
    {
        cv::Mat disparity_float;
        disparity.convertTo(disparity_float, CV_32FC1);
        cv::Mat valid_mask(disparity_float.size(), CV_8UC1, cv::Scalar(0));
        for (int row = 0; row < disparity_float.rows; ++row)
        {
            if ((row & 63) == 0 && isCancelled(cancellation))
            {
                result.cancelled = true;
                return result;
            }
            const float *source_row = disparity_float.ptr<float>(row);
            uchar *mask_row = valid_mask.ptr<uchar>(row);
            for (int col = 0; col < disparity_float.cols; ++col)
            {
                mask_row[col] = std::isfinite(source_row[col]) && source_row[col] > 0.0f
                    ? 255
                    : 0;
            }
        }

        if (isCancelled(cancellation))
        {
            result.cancelled = true;
            return result;
        }

        float minimum = settings.minimum;
        float maximum = settings.maximum;
        if (settings.autoRange)
        {
            double minimum_value = 0.0;
            double maximum_value = 1.0;
            if (cv::countNonZero(valid_mask) > 0)
            {
                cv::minMaxLoc(
                    disparity_float,
                    &minimum_value,
                    &maximum_value,
                    nullptr,
                    nullptr,
                    valid_mask);
            }
            minimum = static_cast<float>(minimum_value);
            maximum = static_cast<float>(maximum_value);
        }
        if (!std::isfinite(minimum))
        {
            minimum = 0.0f;
        }
        if (!std::isfinite(maximum) || maximum <= minimum)
        {
            maximum = minimum + 1.0f;
        }

        cv::Mat normalized_source = disparity_float.clone();
        normalized_source.setTo(minimum, ~valid_mask);
        cv::max(normalized_source, minimum, normalized_source);
        cv::min(normalized_source, maximum, normalized_source);
        normalized_source = (normalized_source - minimum) / (maximum - minimum) * 255.0f;

        if (isCancelled(cancellation))
        {
            result.cancelled = true;
            return result;
        }

        cv::Mat normalized;
        normalized_source.convertTo(normalized, CV_8UC1);
        cv::Mat colored;
        cv::applyColorMap(normalized, colored, settings.colormap);
        cv::Mat rgb;
        cv::cvtColor(colored, rgb, cv::COLOR_BGR2RGB);

        QImage image(rgb.cols, rgb.rows, QImage::Format_RGBA8888);
        if (image.isNull())
        {
            result.error = QStringLiteral("无法为视差热图分配图像内存");
            return result;
        }

        for (int row = 0; row < rgb.rows; ++row)
        {
            if ((row & 63) == 0 && isCancelled(cancellation))
            {
                result.cancelled = true;
                return result;
            }
            const uchar *rgb_row = rgb.ptr<uchar>(row);
            const uchar *validRow = valid_mask.ptr<uchar>(row);
            uchar *output_row = image.scanLine(row);
            for (int col = 0; col < rgb.cols; ++col)
            {
                uchar *pixel = output_row + col * 4;
                if (validRow[col])
                {
                    pixel[0] = rgb_row[col * 3 + 0];
                    pixel[1] = rgb_row[col * 3 + 1];
                    pixel[2] = rgb_row[col * 3 + 2];
                }
                else
                {
                    pixel[0] = 0;
                    pixel[1] = 0;
                    pixel[2] = 0;
                }

                pixel[3] = settings.showInvalid || validRow[col] ? 255 : 0;
            }
        }
        result.image = std::move(image);
    }
    catch (const cv::Exception &exception)
    {
        result.error = QStringLiteral("生成视差热图失败：%1")
                           .arg(QString::fromUtf8(exception.what()));
    }
    catch (const std::exception &exception)
    {
        result.error = QStringLiteral("生成视差热图失败：%1")
                           .arg(QString::fromUtf8(exception.what()));
    }
    catch (...)
    {
        result.error = QStringLiteral("生成视差热图失败：未知后台异常");
    }
    return result;
}

} // namespace

DisparityHeatmapOverlay::DisparityHeatmapOverlay(QWidget *parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_NoSystemBackground);
    setAttribute(Qt::WA_TranslucentBackground);
}

DisparityHeatmapOverlay::~DisparityHeatmapOverlay()
{
    cancelLoad();
    cancelBuild();
}

bool DisparityHeatmapOverlay::loadDisparity(const QString &filepath)
{
    if (filepath.trimmed().isEmpty())
    {
        return false;
    }

    cancelLoad();
    cancelBuild();
    const quint64 generation = ++_loadGeneration;
    ++_buildGeneration;
    _disparity.release();
    _heatmapImage = QImage();
    _heatmap = QPixmap();
    clearCache();
    update();

    auto cancellation = std::make_shared<std::atomic_bool>(false);
    _loadCancellation = cancellation;
    QFuture<DisparityLoadResult> future = QtConcurrent::run(
        [filepath, cancellation]() -> DisparityLoadResult
        {
            DisparityLoadResult result;
            if (isCancelled(cancellation))
            {
                result.cancelled = true;
                return result;
            }
            try
            {
                result.disparity = xjw::common::io::readImage(filepath, cv::IMREAD_UNCHANGED);
            }
            catch (const cv::Exception &exception)
            {
                result.error = QStringLiteral("读取视差图失败：%1")
                                   .arg(QString::fromUtf8(exception.what()));
                return result;
            }
            catch (const std::exception &exception)
            {
                result.error = QStringLiteral("读取视差图失败：%1")
                                   .arg(QString::fromUtf8(exception.what()));
                return result;
            }
            if (isCancelled(cancellation))
            {
                result.disparity.release();
                result.cancelled = true;
                return result;
            }
            if (result.disparity.empty())
            {
                result.error = QStringLiteral("无法读取视差图：%1").arg(filepath);
            }
            return result;
        });

    auto *watcher = new QFutureWatcher<DisparityLoadResult>(this);
    connect(watcher, &QFutureWatcher<DisparityLoadResult>::finished, this,
            [this, watcher, cancellation, generation]()
            {
                DisparityLoadResult result;
                try
                {
                    result = watcher->result();
                }
                catch (const std::exception &exception)
                {
                    result.error = QStringLiteral("读取视差图失败：%1")
                        .arg(QString::fromUtf8(exception.what()));
                }
                catch (...)
                {
                    result.error = QStringLiteral("读取视差图失败：未知后台异常");
                }
                watcher->deleteLater();
                if (generation != _loadGeneration || isCancelled(cancellation) || result.cancelled)
                {
                    return;
                }
                _loadCancellation.reset();
                if (!result.error.isEmpty())
                {
                    emit loadFailed(result.error);
                    return;
                }
                acceptDisparity(result.disparity);
            });
    watcher->setFuture(future);
    return true;
}

bool DisparityHeatmapOverlay::loadDisparity(const cv::Mat &disparity)
{
    if (disparity.empty())
    {
        return false;
    }
    cancelLoad();
    ++_loadGeneration;
    acceptDisparity(disparity.clone());
    return true;
}

void DisparityHeatmapOverlay::clear()
{
    cancelLoad();
    cancelBuild();
    ++_loadGeneration;
    ++_buildGeneration;
    ++_sourceRevision;
    _disparity.release();
    _heatmapImage = QImage();
    _heatmap = QPixmap();
    clearCache();
    update();
}

void DisparityHeatmapOverlay::setTargetView(ImageViewWidget *view)
{
    if (_targetView == view)
    {
        syncToTargetViewport();
        return;
    }

    const bool was_visible = isVisible();
    if (_targetView && _targetView->view() && _targetView->view()->viewport())
    {
        _targetView->view()->viewport()->removeEventFilter(this);
        disconnect(_targetView, nullptr, this, nullptr);
    }
    _targetView = view;
    if (!_targetView || !_targetView->view() || !_targetView->view()->viewport())
    {
        hide();
        return;
    }

    QWidget *viewport = _targetView->view()->viewport();
    setParent(viewport);
    viewport->installEventFilter(this);
    connect(_targetView, &ImageViewWidget::viewTransformChanged,
            this, [this](const QTransform &)
            {
                update();
            }, Qt::QueuedConnection);
    syncToTargetViewport();
    if (was_visible)
    {
        show();
    }
}

ImageViewWidget *DisparityHeatmapOverlay::targetView() const
{
    return _targetView.data();
}

QTransform DisparityHeatmapOverlay::sceneToOverlayTransform() const
{
    if (!_targetView || !_targetView->view())
    {
        return QTransform();
    }
    return _targetView->view()->viewportTransform();
}

QPointF DisparityHeatmapOverlay::mapSceneToOverlay(const QPointF &scenePoint) const
{
    return sceneToOverlayTransform().map(scenePoint);
}

void DisparityHeatmapOverlay::syncToTargetViewport()
{
    if (!_targetView || !_targetView->view() || !_targetView->view()->viewport())
    {
        return;
    }
    QWidget *viewport = _targetView->view()->viewport();
    if (parentWidget() != viewport)
    {
        setParent(viewport);
    }
    setGeometry(viewport->rect());
    raise();
    update();
}

void DisparityHeatmapOverlay::setOpacity(float opacity)
{
    const float bounded_opacity = std::clamp(opacity, 0.0f, 1.0f);
    if (_opacity == bounded_opacity)
    {
        return;
    }
    _opacity = bounded_opacity;
    update();
}

void DisparityHeatmapOverlay::setDisparityRange(float min, float max)
{
    if (_dispMin == min && _dispMax == max && !_autoRange)
    {
        return;
    }
    _dispMin = min;
    _dispMax = max;
    _autoRange = false;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setAutoRange(bool enabled)
{
    if (_autoRange == enabled)
    {
        return;
    }
    _autoRange = enabled;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setColormap(int cvColormap)
{
    if (_colormap == cvColormap)
    {
        return;
    }
    _colormap = cvColormap;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setShowInvalid(bool show)
{
    if (_showInvalid == show)
    {
        return;
    }
    _showInvalid = show;
    rebuildHeatmap();
}

bool DisparityHeatmapOverlay::eventFilter(QObject *watched, QEvent *event)
{
    if (_targetView && _targetView->view() &&
        watched == _targetView->view()->viewport() &&
        (event->type() == QEvent::Resize || event->type() == QEvent::Show))
    {
        syncToTargetViewport();
    }
    return QWidget::eventFilter(watched, event);
}

void DisparityHeatmapOverlay::paintEvent(QPaintEvent *)
{
    if (_heatmap.isNull() || !_targetView || !_targetView->view())
    {
        return;
    }

    QRectF scene_rect;
    if (_targetView->view()->scene())
    {
        scene_rect = _targetView->view()->scene()->sceneRect();
    }
    if (scene_rect.isEmpty())
    {
        scene_rect = QRectF(QPointF(0.0, 0.0), QSizeF(_heatmap.size()));
    }

    QPainter painter(this);
    painter.setOpacity(_opacity);
    painter.setWorldTransform(sceneToOverlayTransform());
    painter.drawPixmap(scene_rect, _heatmap, QRectF(_heatmap.rect()));
}

void DisparityHeatmapOverlay::rebuildHeatmap()
{
    cancelBuild();
    const quint64 generation = ++_buildGeneration;
    if (_disparity.empty())
    {
        _heatmapImage = QImage();
        _heatmap = QPixmap();
        update();
        return;
    }

    const QString cache_key = currentCacheKey();
    const auto cached = _heatmapCache.constFind(cache_key);
    if (cached != _heatmapCache.cend())
    {
        _heatmapCacheOrder.removeAll(cache_key);
        _heatmapCacheOrder.append(cache_key);
        applyHeatmap(cache_key, cached.value());
        return;
    }

    HeatmapSettings settings;
    settings.minimum = _dispMin;
    settings.maximum = _dispMax;
    settings.autoRange = _autoRange;
    settings.colormap = _colormap;
    settings.showInvalid = _showInvalid;
    const cv::Mat disparity = _disparity;
    auto cancellation = std::make_shared<std::atomic_bool>(false);
    _buildCancellation = cancellation;

    QFuture<HeatmapBuildResult> future = QtConcurrent::run(
        [disparity, settings, cancellation]()
        {
            return buildHeatmapImage(disparity, settings, cancellation);
        });
    auto *watcher = new QFutureWatcher<HeatmapBuildResult>(this);
    connect(watcher, &QFutureWatcher<HeatmapBuildResult>::finished, this,
            [this, watcher, cancellation, generation, cache_key]()
            {
                HeatmapBuildResult result;
                try
                {
                    result = watcher->result();
                }
                catch (const std::exception &exception)
                {
                    result.error = QStringLiteral("生成视差热图失败：%1")
                        .arg(QString::fromUtf8(exception.what()));
                }
                catch (...)
                {
                    result.error = QStringLiteral("生成视差热图失败：未知后台异常");
                }
                watcher->deleteLater();
                if (generation != _buildGeneration || isCancelled(cancellation) || result.cancelled)
                {
                    return;
                }
                _buildCancellation.reset();
                if (!result.error.isEmpty())
                {
                    emit loadFailed(result.error);
                    return;
                }
                cacheHeatmap(cache_key, result.image);
                applyHeatmap(cache_key, result.image);
            });
    watcher->setFuture(future);
}

void DisparityHeatmapOverlay::acceptDisparity(cv::Mat disparity)
{
    cancelBuild();
    _disparity = std::move(disparity);
    ++_sourceRevision;
    _heatmapImage = QImage();
    _heatmap = QPixmap();
    clearCache();
    update();
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::applyHeatmap(const QString &, const QImage &image)
{
    _heatmapImage = image;
    _heatmap = QPixmap::fromImage(_heatmapImage);
    update();
    emit heatmapReady(_heatmapImage.size());
}

QString DisparityHeatmapOverlay::currentCacheKey() const
{
    QString invalid_key;
    if (_showInvalid)
    {
        invalid_key = QStringLiteral("1");
    }
    else
    {
        invalid_key = QStringLiteral("0");
    }
    return QStringLiteral("%1|%2|%3|%4|%5|%6")
        .arg(_sourceRevision)
        .arg(_autoRange ? 1 : 0)
        .arg(QString::number(_dispMin, 'g', 9))
        .arg(QString::number(_dispMax, 'g', 9))
        .arg(_colormap)
        .arg(invalid_key);
}

void DisparityHeatmapOverlay::clearCache()
{
    _heatmapCache.clear();
    _heatmapCacheOrder.clear();
    _heatmapCacheBytes = 0;
}

void DisparityHeatmapOverlay::cacheHeatmap(const QString &cacheKey, const QImage &image)
{
    const qsizetype image_bytes = image.sizeInBytes();
    if (image_bytes <= 0 || image_bytes > MaximumCacheBytes)
    {
        return;
    }
    while (!_heatmapCacheOrder.isEmpty() &&
           _heatmapCacheBytes + image_bytes > MaximumCacheBytes)
    {
        const QString oldest_key = _heatmapCacheOrder.takeFirst();
        const auto oldest = _heatmapCache.find(oldest_key);
        if (oldest != _heatmapCache.end())
        {
            _heatmapCacheBytes -= oldest.value().sizeInBytes();
            _heatmapCache.erase(oldest);
        }
    }
    _heatmapCache.insert(cacheKey, image);
    _heatmapCacheOrder.removeAll(cacheKey);
    _heatmapCacheOrder.append(cacheKey);
    _heatmapCacheBytes += image_bytes;
}

void DisparityHeatmapOverlay::cancelLoad()
{
    if (_loadCancellation)
    {
        _loadCancellation->store(true, std::memory_order_relaxed);
        _loadCancellation.reset();
    }
}

void DisparityHeatmapOverlay::cancelBuild()
{
    if (_buildCancellation)
    {
        _buildCancellation->store(true, std::memory_order_relaxed);
        _buildCancellation.reset();
    }
}
