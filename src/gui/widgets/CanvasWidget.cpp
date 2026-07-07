#include "CanvasWidget.h"

#include "LayerFeatureLoader.h"
#include "LayerRenderer.h"
#include "ProjectIO.h"
#include "GuiTaskRunner.h"
#include "io/PathIO.h"

#include <QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <QPointer>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <QGraphicsScene>
#include <QDir>
#include <QFileInfo>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QScrollBar>
#include <QVariant>
#include <QVariantMap>
#include <QFile>
#include <QDataStream>
#include <QPointF>
#include <QVector>
#include <QGraphicsPixmapItem>
#include "Logger.h"

// CanvasWidget 实现：创建并持有一个 QGraphicsScene，作为渲染的起点

CanvasWidget::CanvasWidget(QWidget *parent)
    : QGraphicsView(parent)
{
    auto *scene = new QGraphicsScene(this);
    setScene(scene);

    // 渲染器：负责把影像层加入到 scene
    _layerRenderer = new LayerRenderer(scene, this);

    // 默认设置：平滑缩放，开启抗锯齿（若需要可调整）
    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::SmoothPixmapTransform, true);

    // 缩放体验：以鼠标位置为锚点缩放；拖拽时以鼠标为锚点
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorUnderMouse);

    // 鼠标/交互将在后续扩展中实现
    // 特征点加载 watcher 在每次请求时创建，并用 generation guard 丢弃过期结果。
}

void CanvasWidget::applyFeatureDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts)
{
    if (!_layerRenderer) return;
    // 保存当前选项并立即注入渲染器
    _currentFeatureOpts = opts;
    _layerRenderer->setFeatureDisplayOptions(opts);
    // Use the options' showPoints to control visibility: 当 opts.showPoints 为 true 时加载特征点，否则清除
    if (opts.showPoints && !_currentImagePath.trimmed().isEmpty()) {
        _layerRenderer->clearFeatureLayers();
        startSpLoadForImage(_currentImagePath);
    } else {
        _layerRenderer->clearFeatureLayers();
    }
}

// 当控件从隐藏变为可见时触发（如 QStackedWidget 切换到本页面）。
// 若上次 fitInView 时控件不可见（视口为 0），此处重新适配以确保图像充满视图。
void CanvasWidget::showEvent(QShowEvent *event)
{
    QGraphicsView::showEvent(event);
    if (scene() && !scene()->sceneRect().isEmpty() && _zoomFactor == 1.0)
    {
        fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
    }
}

void CanvasWidget::showImage(const QString &path)
{
    if (!scene() || !_layerRenderer)
    {
        return;
    }

    // 清理旧覆盖层；影像图层在新影像加载完成后替换，避免磁盘解码期间界面空白。
    // NOTE: 不调用 scene()->clear() —— 那会使 QGraphicsScene 删除所有 items，
    // 导致 LayerRenderer 持有的指针变为悬空并在后续删除时造成双重释放。
    _layerRenderer->clearFeatureLayers();
    // 确保清除上一次的匹配连线层，避免其干扰新的场景布局
    _layerRenderer->clearMatchLayers();
    _layerRenderer->clearMaskLayers();

    if (path.trimmed().isEmpty())
    {
        _layerRenderer->clear();
        return;
    }

    // 将当前项目路径注入渲染器，用于把非 8-bit 影像转换缓存写入项目 .plascan_tmp。
    // 说明：CanvasWidget 不直接依赖 ProjectManager 头文件，这里通过 QObject 动态属性读取。
    // MainWindow/ProjectManager 会在运行时设置该属性。
    if (_layerRenderer)
    {
        const QVariant v = property("currentProjectPath");
        if (v.isValid())
        {
            _layerRenderer->setCurrentProjectPath(v.toString());
        }
    }

    // 记录当前影像路径
    _currentImagePath = path;

    const QString pathCopy = path;
    const QString projectPath = property("currentProjectPath").toString();
    auto *watcher = new QFutureWatcher<QImage>(this);
    _imageWatcher = watcher;
    QPointer<CanvasWidget> self(this);
    connect(watcher, &QFutureWatcher<QImage>::finished, watcher, [self, watcher, loadedPath = pathCopy]()
    {
        const QImage image = watcher->result();
        watcher->deleteLater();
        if (!self)
        {
            return;
        }
        if (watcher != self->_imageWatcher)
        {
            return;
        }
        self->_imageWatcher = nullptr;
        if (QDir::cleanPath(loadedPath) != QDir::cleanPath(self->_currentImagePath))
        {
            return;
        }
        if (image.isNull() || !self->_layerRenderer)
        {
            LOG_WARN(QStringLiteral("showImage: failed to load image %1").arg(loadedPath));
            return;
        }

        self->_layerRenderer->clear();
        if (!self->_layerRenderer->addImageLayer(image, 0))
        {
            return;
        }
        self->reloadMaskOverlay();

        // 通知外部当前活跃影像已变更（MainWindow 据此持久化状态）
        emit self->activeImageChanged(loadedPath);

        // 让视图自动适配内容
        self->scene()->setSceneRect(self->scene()->itemsBoundingRect());
        self->fitInView(self->scene()->sceneRect(), Qt::KeepAspectRatio);

        // 重新适配后，重置缩放因子
        self->_zoomFactor = 1.0;

        // 自动加载特征点（默认启用）
        // 跳过非项目影像（如深度图 depth_*.png）的特征点加载，避免无意义的 .sp 查找
        const QString fileName = QFileInfo(loadedPath).fileName();
        const bool isDepthMap = fileName.startsWith(QLatin1String("depth_"), Qt::CaseInsensitive)
                             && fileName.endsWith(QLatin1String(".png"), Qt::CaseInsensitive);
        if (!isDepthMap) {
            self->startSpLoadForImage(loadedPath);
        }
    });
    QFuture<QImage> future = QtConcurrent::run([pathCopy, projectPath]() {
        return LayerRenderer::loadImageForDisplay(pathCopy, projectPath);
    });
    watcher->setFuture(future);
}

void CanvasWidget::showMatchedPair(const QString &imgA, const QString &imgB, const QString &matchFile)
{
    if (!scene() || !_layerRenderer) return;

    LOG_DEBUG(QStringLiteral("showMatchedPair: imgA=%1 imgB=%2 match=%3").arg(imgA, imgB, matchFile));

    // clear existing layers
    _layerRenderer->clearFeatureLayers();
    _layerRenderer->clearMatchLayers();
    _layerRenderer->clearMaskLayers();
    _layerRenderer->clear();

    // parse match file (robust raw parser based on ASP parse_match_file.py)
    QVector<QPointF> ptsA, ptsB;
    QFile f(matchFile);
    if (!QFileInfo(matchFile).exists()) {
        LOG_WARN(QStringLiteral("showMatchedPair: match file not found: %1").arg(matchFile));
    }

    if (f.open(QIODevice::ReadOnly)) {
        LOG_DEBUG(QStringLiteral("showMatchedPair: parsing raw match file %1").arg(matchFile));
        using namespace Qt;
        auto readBytes = [&](qint64 n) -> QByteArray {
            QByteArray b = f.read(n);
            if (b.size() != n) {
                LOG_WARN(QStringLiteral("showMatchedPair: unexpected EOF when reading %1 bytes").arg(n));
            }
            return b;
        };

        auto readUint64LE = [&](quint64 &out) -> bool {
            QByteArray b = readBytes(8);
            if (b.size() != 8) return false;
            quint64 v = 0;
            memcpy(&v, b.constData(), 8);
            out = qFromLittleEndian<quint64>(v);
            return true;
        };
        auto readUint32LE = [&](quint32 &out) -> bool {
            QByteArray b = readBytes(4);
            if (b.size() != 4) return false;
            quint32 v = 0; memcpy(&v, b.constData(), 4); out = qFromLittleEndian<quint32>(v); return true;
        };
        auto readInt32LE = [&](qint32 &out) -> bool {
            QByteArray b = readBytes(4);
            if (b.size() != 4) return false;
            qint32 v = 0; memcpy(&v, b.constData(), 4); out = qFromLittleEndian<qint32>(v); return true;
        };
        auto readFloatLE = [&](float &out) -> bool {
            QByteArray b = readBytes(4);
            if (b.size() != 4) return false;
            quint32 v = 0; memcpy(&v, b.constData(), 4); v = qFromLittleEndian<quint32>(v);
            float fval; memcpy(&fval, &v, sizeof(float)); out = fval; return true;
        };
        auto readInt8 = [&](qint8 &out) -> bool {
            QByteArray b = readBytes(1);
            if (b.size() != 1) return false;
            out = static_cast<qint8>(b.at(0)); return true;
        };

        quint64 size1 = 0, size2 = 0;
        if (!readUint64LE(size1) || !readUint64LE(size2)) {
            LOG_WARN(QStringLiteral("showMatchedPair: failed to read header sizes"));
            f.close();
        } else {
            LOG_DEBUG(QStringLiteral("showMatchedPair: parsed header size1=%1 size2=%2").arg(size1).arg(size2));

            auto read_ip_raw = [&]() -> QPointF {
                float x=0.0f, y=0.0f;
                qint32 xi=0, yi=0;
                float orientation=0.0f, scale=0.0f, interest=0.0f;
                qint8 pol=0; quint32 octave=0, scale_lvl=0; quint64 ndesc=0;
                if (!readFloatLE(x)) return QPointF();
                if (!readFloatLE(y)) return QPointF();
                if (!readInt32LE(xi)) return QPointF();
                if (!readInt32LE(yi)) return QPointF();
                if (!readFloatLE(orientation)) return QPointF();
                if (!readFloatLE(scale)) return QPointF();
                if (!readFloatLE(interest)) return QPointF();
                if (!readInt8(pol)) return QPointF();
                if (!readUint32LE(octave)) return QPointF();
                if (!readUint32LE(scale_lvl)) return QPointF();
                if (!readUint64LE(ndesc)) return QPointF();

                qint64 toSkip = static_cast<qint64>(ndesc) * static_cast<qint64>(sizeof(float));
                if (toSkip > 0)
                {
                    // guard the seek length to avoid absurd values
                    const qint64 maxSkip = 1LL << 30; // ~1GB
                    if (toSkip > maxSkip)
                    {
                        LOG_WARN(QStringLiteral("showMatchedPair: ndesc too large (%1), skipping descriptors aborted")
                                     .arg(ndesc));
                    }
                    else
                    {
                        const qint64 cur = f.pos();
                        if (!f.seek(cur + toSkip))
                        {
                            LOG_WARN(QStringLiteral("showMatchedPair: failed to skip descriptor bytes %1").arg(toSkip));
                        }
                    }
                }
                return QPointF(static_cast<qreal>(x), static_cast<qreal>(y));
            };

            for (quint64 i = 0; i < size1; ++i) ptsA.append(read_ip_raw());
            for (quint64 i = 0; i < size2; ++i) ptsB.append(read_ip_raw());
            LOG_DEBUG(QStringLiteral("showMatchedPair: read ptsA=%1 ptsB=%2").arg(ptsA.size()).arg(ptsB.size()));
            f.close();
        }
    }

    // Ensure renderer knows project path (so caching/convert works same as showImage)
    if (_layerRenderer)
    {
        const QVariant v = property("currentProjectPath");
        if (v.isValid())
        {
            _layerRenderer->setCurrentProjectPath(v.toString());
        }
    }

    // Add stitched images
    QGraphicsPixmapItem *itemA = nullptr;
    QGraphicsPixmapItem *itemB = nullptr;
    if (!_layerRenderer->addStitchedImagePair(imgA, imgB, &itemA, &itemB, 20))
    {
        qWarning() << "addStitchedImagePair failed for" << imgA << imgB;
        LOG_WARN(QStringLiteral("showMatchedPair: addStitchedImagePair failed for %1 <-> %2").arg(imgA, imgB));
        // fallback: show single image
        showImage(imgA);
        return;
    }

    // Validate that pixmaps are non-empty
    if (!itemA || itemA->pixmap().isNull()) {
        qWarning() << "Failed to load image A for stitched view:" << imgA;
        // cleanup any added B
        if (itemA) { _layerRenderer->clear(); }
        showImage(imgA);
        return;
    }
    if (!itemB || itemB->pixmap().isNull()) {
        qWarning() << "Failed to load image B for stitched view:" << imgB;
        // treat as single image
        _layerRenderer->clear();
        showImage(imgA);
        return;
    }

    qreal bOffsetX = itemB ? itemB->pos().x() : 0.0;

    // Debug: report positions and sizes of stitched items
    if (itemA)
    {
        LOG_DEBUG(QStringLiteral("showMatchedPair: itemA pos=(%1,%2) size=%3x%4")
                      .arg(itemA->pos().x())
                      .arg(itemA->pos().y())
                      .arg(itemA->pixmap().width())
                      .arg(itemA->pixmap().height()));
    }
    if (itemB)
    {
        LOG_DEBUG(QStringLiteral("showMatchedPair: itemB pos=(%1,%2) size=%3x%4")
                      .arg(itemB->pos().x())
                      .arg(itemB->pos().y())
                      .arg(itemB->pixmap().width())
                      .arg(itemB->pixmap().height()));
    }

    // draw match lines if points parsed
    if (!ptsA.isEmpty() && !ptsB.isEmpty())
    {
        LOG_DEBUG(QStringLiteral("showMatchedPair: drawing match lines with offset %1").arg(bOffsetX));

        // Log a few sample points for diagnostics
        for (int si = 0; si < qMin(5, ptsA.size()); ++si)
        {
            const QPointF &pa = ptsA.at(si);
            const QPointF &pb = (si < ptsB.size()) ? ptsB.at(si) : QPointF();
            LOG_DEBUG(QStringLiteral("showMatchedPair: sample[%1] A=(%2,%3) B=(%4,%5)")
                          .arg(si)
                          .arg(pa.x())
                          .arg(pa.y())
                          .arg(pb.x())
                          .arg(pb.y()));
        }

        // Filter out obviously invalid points (non-finite or extreme) to avoid corrupting scene bounds
        QVector<QPointF> fA;
        fA.reserve(ptsA.size());
        QVector<QPointF> fB;
        fB.reserve(ptsB.size());
        const double MAX_COORD = 1e7; // arbitrary large threshold
        const int n = qMin(ptsA.size(), ptsB.size());
        for (int i = 0; i < n; ++i)
        {
            const QPointF &a = ptsA.at(i);
            const QPointF &b = ptsB.at(i);
            const bool aok = std::isfinite(a.x()) &&
                             std::isfinite(a.y()) &&
                             std::fabs(a.x()) <= MAX_COORD &&
                             std::fabs(a.y()) <= MAX_COORD;
            const bool bok = std::isfinite(b.x()) &&
                             std::isfinite(b.y()) &&
                             std::fabs(b.x()) <= MAX_COORD &&
                             std::fabs(b.y()) <= MAX_COORD;
            if (aok && bok)
            {
                fA.append(a);
                fB.append(b);
            }
            else
            {
                LOG_DEBUG(QStringLiteral("showMatchedPair: skipping invalid pair index %1 A=(%2,%3) B=(%4,%5)")
                              .arg(i)
                              .arg(a.x())
                              .arg(a.y())
                              .arg(b.x())
                              .arg(b.y()));
            }
        }

        LOG_DEBUG(QStringLiteral("showMatchedPair: filtered ptsA=%1 ptsB=%2").arg(fA.size()).arg(fB.size()));

        if (!fA.isEmpty() && !fB.isEmpty())
        {
            _layerRenderer->addMatchLines(fA, fB, bOffsetX);
        }
        else
        {
            LOG_DEBUG(QStringLiteral("showMatchedPair: no valid match points after filtering"));
        }
    }
    else
    {
        LOG_DEBUG(QStringLiteral("showMatchedPair: no match points to draw"));
    }
    // 强制刷新并输出场景信息，帮助诊断“白屏”
    if (scene())
    {
        // log item count and bounding rect
        const auto itemsCount = scene()->items().size();
        const QRectF boundsBefore = scene()->itemsBoundingRect();
        LOG_DEBUG(QStringLiteral("showMatchedPair: scene items=%1 boundsBefore=(%2,%3,%4,%5)")
                      .arg(itemsCount)
                      .arg(boundsBefore.x())
                      .arg(boundsBefore.y())
                      .arg(boundsBefore.width())
                      .arg(boundsBefore.height()));
        // Ask scene and viewport to update immediately
        scene()->update();
        viewport()->update();
    }

    // fit view to items bounding rect (robust if sceneRect was large/empty)
    QRectF rect = scene()->itemsBoundingRect();
    LOG_DEBUG(QStringLiteral("showMatchedPair: itemsBoundingRect after draw=(%1,%2,%3,%4)")
                  .arg(rect.x())
                  .arg(rect.y())
                  .arg(rect.width())
                  .arg(rect.height()));
    if (!rect.isEmpty())
    {
        scene()->setSceneRect(rect);
        fitInView(rect, Qt::KeepAspectRatio);
    }
    else
    {
        // fallback to show single image
        qWarning() << "itemsBoundingRect empty after stitching; falling back";
        LOG_WARN(QStringLiteral("showMatchedPair: itemsBoundingRect empty after stitching, "
                                "falling back to single image"));
        _layerRenderer->clear();
        showImage(imgA);
        return;
    }
    _zoomFactor = 1.0;
}

void CanvasWidget::setActiveFeatureSuffix(const QString &suffix)
{
    if (suffix.isEmpty() || suffix == _activeFeatureSuffix) return;
    _activeFeatureSuffix = suffix;
    // 清除当前影像的缓存, 强制重新加载
    if (!_currentImagePath.isEmpty())
        setShowInterestPoints(_showInterestPoints);
}

QStringList CanvasWidget::availableFeatureSuffixes() const
{
    if (_currentImagePath.isEmpty()) return {};
    const QString projectPath = property("currentProjectPath").toString();
    return ProjectIO::availableFeatureSuffixes(projectPath, _currentImagePath);
}

void CanvasWidget::setShowInterestPoints(bool show)
{
    _showInterestPoints = show;
    if (!_layerRenderer) return;

    if (_showInterestPoints && !_currentImagePath.trimmed().isEmpty())
    {
        // 异步加载当前影像的特征文件
        _layerRenderer->clearFeatureLayers();
        startSpLoadForImage(_currentImagePath);
    }
    else
    {
        _layerRenderer->clearFeatureLayers();
    }
}

void CanvasWidget::startSpLoadForImage(const QString &imagePath)
{
    if (imagePath.trimmed().isEmpty()) return;

    const QString imagePathCopy = imagePath;
    const QString activeSuffix = _activeFeatureSuffix;
    const QString projectPath = property("currentProjectPath").toString();
    const bool shouldEstimateOrientation = _currentFeatureOpts.showOrientation;
    const int generation = ++_featureLoadGeneration;
    // 检查缓存 (key 含 suffix)
    QFileInfo fiCheck(imagePathCopy);
    const QString cacheKey = imagePathCopy + activeSuffix;
    auto it = _spCache.find(cacheKey);
    if (it != _spCache.end()) {
        if (it->second.first == fiCheck.lastModified()) {
            const bool isCurrentImage = QDir::cleanPath(imagePathCopy) == QDir::cleanPath(_currentImagePath);
            if (isCurrentImage && _layerRenderer) {
                _layerRenderer->setFeatureDisplayOptions(_currentFeatureOpts);
                _layerRenderer->clearFeatureLayers();
                if (!it->second.second.empty()) _layerRenderer->addFeatureItems(it->second.second);
            }
            emit featuresLoaded(imagePathCopy, static_cast<int>(it->second.second.size()));
            return;
        }
    }
    
    QPointer<CanvasWidget> self(this);
    xjw::gui::tasks::runGuarded(
        this,
        [imagePathCopy, activeSuffix, projectPath, shouldEstimateOrientation]() -> std::vector<cv::KeyPoint>
        {
            std::vector<cv::KeyPoint> empty;
            // 使用当前选中的后缀查找特征文件。
            const QString spFile = ProjectIO::featureOutputPathForImage(
                projectPath, imagePathCopy, activeSuffix);
            LOG_DEBUG(QStringLiteral("startSpLoadForImage: suffix=%1 file=%2")
                .arg(activeSuffix, spFile));

            if (spFile.isEmpty() || !QFile::exists(spFile))
            {
                LOG_DEBUG(QStringLiteral("startSpLoadForImage: no %1 found for %2")
                    .arg(activeSuffix, imagePathCopy));
                return empty;
            }

            // 读取特征文件 (支持所有提取器类型)。
            std::vector<cv::KeyPoint> keypoints = xjw::gui::views::loadFeatureKeypointsFromFile(spFile);
            if (keypoints.empty())
            {
                LOG_WARN(QStringLiteral("startSpLoadForImage: failed to read feature file %1").arg(spFile));
                return empty;
            }

            if (shouldEstimateOrientation)
            {
                // 尝试从影像中估计每个 keypoint 的方向（梯度方向），以便显示方向箭头。
                // 默认不开方向显示时跳过整图 imread/Sobel，避免切换影像时抢占磁盘与 CPU。
                try
                {
                    cv::Mat img = xjw::common::io::readImage(imagePathCopy, cv::IMREAD_GRAYSCALE);
                    if (!img.empty())
                    {
                        cv::Mat gx, gy;
                        cv::Sobel(img, gx, CV_32F, 1, 0, 3);
                        cv::Sobel(img, gy, CV_32F, 0, 1, 3);
                        for (auto &kp : keypoints)
                        {
                            int x = static_cast<int>(std::round(kp.pt.x));
                            int y = static_cast<int>(std::round(kp.pt.y));
                            if (x >= 0 && x < gx.cols && y >= 0 && y < gx.rows)
                            {
                                float vx = gx.at<float>(y, x);
                                float vy = gy.at<float>(y, x);
                                if (std::isfinite(vx) &&
                                    std::isfinite(vy) &&
                                    (std::abs(vx) > 1e-6f || std::abs(vy) > 1e-6f))
                                {
                                    double ang =
                                        std::atan2(static_cast<double>(vy), static_cast<double>(vx)) * 180.0 / M_PI;
                                    if (ang < 0)
                                    {
                                        ang += 360.0;
                                    }
                                    kp.angle = static_cast<float>(ang);
                                }
                                else
                                {
                                    kp.angle = 0.0f;
                                }
                            }
                            else
                            {
                                kp.angle = 0.0f;
                            }
                        }
                    }
                }
                catch (...)
                {
                    // 估计失败则忽略，保持原有角度值。
                }
            }

            LOG_DEBUG(QStringLiteral("startSpLoadForImage: loaded %1 keypoints from %2")
                          .arg(static_cast<int>(keypoints.size()))
                          .arg(spFile));
            return keypoints;
        },
        [self, imagePathCopy, activeSuffix, generation](CanvasWidget *widget,
                                                        std::vector<cv::KeyPoint> kps) mutable
        {
            if (!self || widget != self.data())
            {
                return;
            }
            if (generation != self->_featureLoadGeneration)
            {
                return;
            }

            const bool isCurrentImage = QDir::cleanPath(imagePathCopy) == QDir::cleanPath(self->_currentImagePath);
            if (!imagePathCopy.trimmed().isEmpty())
            {
                QFileInfo fi(imagePathCopy);
                const QString key = imagePathCopy + activeSuffix;
                self->_spCache[key] = std::make_pair(fi.lastModified(), kps);
            }
            if (isCurrentImage && self->_layerRenderer)
            {
                // 重新应用当前显示设置，确保使用 UI 中的参数。
                self->_layerRenderer->setFeatureDisplayOptions(self->_currentFeatureOpts);
                self->_layerRenderer->clearFeatureLayers();
                if (!kps.empty())
                {
                    self->_layerRenderer->addFeatureItems(kps);
                }
            }
            // 发出信号以便主窗体更新状态栏 / 面板。
            emit self->featuresLoaded(imagePathCopy, static_cast<int>(kps.size()));
        });
}

void CanvasWidget::reloadMaskOverlay()
{
    if (!_layerRenderer)
    {
        return;
    }

    _layerRenderer->clearMaskLayers();
    if (!_showMaskOverlay || _currentImagePath.trimmed().isEmpty())
    {
        return;
    }

    const QString projectPath = property("currentProjectPath").toString();
    const QString maskPath = ProjectIO::findMaskForImage(projectPath, _currentImagePath);
    if (maskPath.isEmpty())
    {
        return;
    }
    _layerRenderer->addMaskContourLayer(maskPath);
}

void CanvasWidget::setShowMaskOverlay(bool show)
{
    _showMaskOverlay = show;
    reloadMaskOverlay();
}

void CanvasWidget::reloadInterestPoints(const QString &imagePath)
{
    if (imagePath.trimmed().isEmpty()) return;

    // 删除缓存条目以确保后续读取不会直接走缓存分支
    auto it = _spCache.find(imagePath);
    if (it != _spCache.end()) {
        _spCache.erase(it);
    }

    // 仅在用户开启叠加兴趣点或当前图像为目标时刷新渲染
    if (_showInterestPoints && !_currentImagePath.trimmed().isEmpty()) {
        // 如果请求的路径不是当前显示的影像，仍尝试加载以更新缓存（但不叠加到当前视图）
        if (QDir::cleanPath(imagePath) == QDir::cleanPath(_currentImagePath)) {
            // 对当前显示影像，直接触发加载并叠加
            _layerRenderer->clearFeatureLayers();
            startSpLoadForImage(_currentImagePath);
        } else {
            // 对非当前显示影像，只更新缓存：启动后台加载但不要修改当前 scene
            startSpLoadForImage(imagePath);
        }
    } else {
        // 即使未开启叠加，也更新缓存以便后续打开影像时能立即得到最新数据
        startSpLoadForImage(imagePath);
    }
}

void CanvasWidget::immediateReloadInterestPoints(const QString &imagePath)
{
    if (imagePath.trimmed().isEmpty()) return;

    const bool isCurrentImage = (QDir::cleanPath(imagePath) == QDir::cleanPath(_currentImagePath));

    // 删除缓存条目以确保后续读取不会直接走缓存分支
    auto it = _spCache.find(imagePath);
    if (it != _spCache.end()) {
        _spCache.erase(it);
    }

    // 直接在主线程同步读取特征文件并更新显示，使用当前活动的后缀
    const QString projectPath = property("currentProjectPath").toString();
    const QString spFile = ProjectIO::featureFileForSuffix(projectPath, imagePath, _activeFeatureSuffix);
    if (spFile.isEmpty())
    {
        LOG_DEBUG(QStringLiteral("immediateReloadInterestPoints: no %1 found for %2")
                      .arg(_activeFeatureSuffix, imagePath));
        // 仅在当前显示影像时清除场景中的兴趣点，避免覆盖用户正在看的其它影像
        if (isCurrentImage && _layerRenderer)
        {
            _layerRenderer->clearFeatureLayers();
        }
        emit featuresLoaded(imagePath, 0);
        return;
    }

    std::vector<cv::KeyPoint> keypoints = xjw::gui::views::loadFeatureKeypointsFromFile(spFile);
    if (keypoints.empty())
    {
        LOG_WARN(QStringLiteral("immediateReloadInterestPoints: failed to read .sp file %1").arg(spFile));
        if (isCurrentImage && _layerRenderer)
        {
            _layerRenderer->clearFeatureLayers();
        }
        emit featuresLoaded(imagePath, 0);
        return;
    }

    if (_currentFeatureOpts.showOrientation)
    {
        // 估计每个 keypoint 的方向（用于显示方向箭头）。
        // 注意：startSpLoadForImage 的异步路径有做这个；这里同步刷新也需要同样处理，否则 showOrientation 不会生效。
        try
        {
            cv::Mat img = xjw::common::io::readImage(imagePath, cv::IMREAD_GRAYSCALE);
            if (!img.empty())
            {
                cv::Mat gx, gy;
                cv::Sobel(img, gx, CV_32F, 1, 0, 3);
                cv::Sobel(img, gy, CV_32F, 0, 1, 3);
                for (auto &kp : keypoints)
                {
                    int x = static_cast<int>(std::round(kp.pt.x));
                    int y = static_cast<int>(std::round(kp.pt.y));
                    if (x >= 0 && x < gx.cols && y >= 0 && y < gx.rows)
                    {
                        float vx = gx.at<float>(y, x);
                        float vy = gy.at<float>(y, x);
                        if (std::isfinite(vx) &&
                            std::isfinite(vy) &&
                            (std::abs(vx) > 1e-6f || std::abs(vy) > 1e-6f))
                        {
                            double ang =
                                std::atan2(static_cast<double>(vy), static_cast<double>(vx)) * 180.0 / M_PI;
                            if (ang < 0)
                            {
                                ang += 360.0;
                            }
                            kp.angle = static_cast<float>(ang);
                        }
                        else
                        {
                            kp.angle = 0.0f;
                        }
                    }
                    else
                    {
                        kp.angle = 0.0f;
                    }
                }
            }
        }
        catch (...)
        {
            // 忽略方向估计失败
        }
    }

    // 更新 cache (key 含 suffix, 支持多提取器)
    QFileInfo fi(imagePath);
    _spCache[imagePath + _activeFeatureSuffix] = std::make_pair(fi.lastModified(), keypoints);

    // 仅当刷新的是“当前显示的影像”时才更新场景，避免处理批量图像时最后一张覆盖当前视图。
    if (isCurrentImage && _layerRenderer)
    {
        _layerRenderer->setFeatureDisplayOptions(_currentFeatureOpts);
        if (!_showInterestPoints || !_currentFeatureOpts.showPoints)
        {
            _layerRenderer->clearFeatureLayers();
        }
        else
        {
            _layerRenderer->clearFeatureLayers();
            if (!keypoints.empty())
            {
                _layerRenderer->addFeatureItems(keypoints);
            }
        }
    }

    LOG_DEBUG(QStringLiteral("immediateReloadInterestPoints: loaded %1 keypoints from %2")
                  .arg(static_cast<int>(keypoints.size()))
                  .arg(spFile));
    emit featuresLoaded(imagePath, static_cast<int>(keypoints.size()));
}

QList<QVariantMap> CanvasWidget::getCachedInterestPointsAsVariant(const QString &imagePath) const
{
    QList<QVariantMap> out;
    if (imagePath.trimmed().isEmpty()) return out;
    auto it = _spCache.find(imagePath);
    if (it == _spCache.end()) return out;
    for (const auto &kp : it->second.second) {
        QVariantMap m;
        m.insert(QStringLiteral("x"), kp.pt.x);
        m.insert(QStringLiteral("y"), kp.pt.y);
        m.insert(QStringLiteral("size"), kp.size);
        m.insert(QStringLiteral("angle"), kp.angle);
        m.insert(QStringLiteral("response"), kp.response); // score
        out.append(m);
    }
    return out;
}

QString CanvasWidget::currentImagePath() const
{
    return _currentImagePath;
}

void CanvasWidget::zoomIn()
{
    // 放大 1.2 倍
    const double next = _zoomFactor * _zoomStep;
    if (next > _zoomMax) return;
    _zoomFactor = next;
    scale(_zoomStep, _zoomStep);
}

void CanvasWidget::zoomOut()
{
    // 缩小 1/1.2
    const double next = _zoomFactor / _zoomStep;
    if (next < _zoomMin) return;
    _zoomFactor = next;
    scale(1.0 / _zoomStep, 1.0 / _zoomStep);
}

void CanvasWidget::resetView()
{
    // 恢复为默认变换并居中场景
    resetTransform();
    _zoomFactor = 1.0;
    if (scene()) {
        fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
    }
}

void CanvasWidget::wheelEvent(QWheelEvent *event)
{
    if (!event)
    {
        return;
    }

    // angleDelta().y() > 0 表示向上滚动（放大）
    const int delta = event->angleDelta().y();
    if (delta == 0)
    {
        event->accept();
        return;
    }

    if (delta > 0)
    {
        zoomIn();
    }
    else
    {
        zoomOut();
    }
    event->accept();
}

void CanvasWidget::mousePressEvent(QMouseEvent *event)
{
    if (!event) return;

    if (event->button() == Qt::LeftButton)
    {
        _isPanning = false;
        _lastPanPoint = event->pos();
        setCursor(Qt::OpenHandCursor);
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void CanvasWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (!event) return;

    if (event->buttons() & Qt::LeftButton)
    {
        const QPoint delta = event->pos() - _lastPanPoint;
        if (!_isPanning)
        {
            if (std::abs(delta.x()) >= _panThreshold || std::abs(delta.y()) >= _panThreshold)
            {
                _isPanning = true;
                setCursor(Qt::ClosedHandCursor);
            }
        }

        if (_isPanning)
        {
            // 反向滚动以产生拖动手感
            horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
            verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
            _lastPanPoint = event->pos();
            event->accept();
            return;
        }
    }

    QGraphicsView::mouseMoveEvent(event);
}

void CanvasWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (!event) return;

    if (event->button() == Qt::LeftButton)
    {
        if (_isPanning)
        {
            // 结束平移
            _isPanning = false;
            setCursor(Qt::ArrowCursor);
            event->accept();
            return;
        }

        // 没有发生拖动：仅将光标恢复为箭头，不进行单击缩放（用户已请求禁用点击放大行为）
        setCursor(Qt::ArrowCursor);
        event->accept();
        return;
    }

    QGraphicsView::mouseReleaseEvent(event);
}
