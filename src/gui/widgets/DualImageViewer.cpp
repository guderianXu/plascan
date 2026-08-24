#include "DualImageViewer.h"

#include "ui_DualImageViewer.h"

#include "ImageViewWidget.h"
#include "MatchLineOverlay.h"
#include "ImageMatchFile.h"
#include "project/ProjectMetadata.h"
#include <QFileInfo>

#include <QTimer>
#include <QMessageBox>
#include <QDebug>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>

namespace {

bool identityMatchesImage(const xjw::image_matching::ImageIdentity &identity,
                          const QString &imagePath)
{
    return xjw::common::project::imageReferenceMatchesToken(
        identity.path, identity.displayName, imagePath);
}

bool betterBlock(const xjw::image_matching::NeighborMatchBlock &left,
                 const xjw::image_matching::NeighborMatchBlock &right)
{
    if (left.geometryPassed != right.geometryPassed)
    {
        return left.geometryPassed;
    }
    if (left.geometryInlierCount != right.geometryInlierCount)
    {
        return left.geometryInlierCount > right.geometryInlierCount;
    }
    if (left.rawMatchCount != right.rawMatchCount)
    {
        return left.rawMatchCount > right.rawMatchCount;
    }
    return left.createdTimeMs > right.createdTimeMs;
}

}

DualImageViewer::DualImageViewer(QWidget *parent)
    : QWidget(parent)
    , _leftView(nullptr)
    , _rightView(nullptr)
    , _overlay(nullptr)
    , _syncEnabled(false)
    , _syncing(false)
{
    setupLayout();
    connectSignals();

    // 创建延迟更新定时器
    _overlayUpdateTimer = new QTimer(this);
    _overlayUpdateTimer->setSingleShot(true);
    // 同一事件循环内合并连续滚动/缩放信号，避免额外一帧的固定延迟。
    _overlayUpdateTimer->setInterval(0);
    connect(_overlayUpdateTimer, &QTimer::timeout,
            this, &DualImageViewer::updateOverlayNow);

}

ImageViewWidget* DualImageViewer::leftView() const
{
    return _leftView.data();
}

ImageViewWidget* DualImageViewer::rightView() const
{
    return _rightView.data();
}

MatchLineOverlay* DualImageViewer::overlay() const
{
    return _overlay.data();
}

void DualImageViewer::highlightMatchIndex(int index)
{
    if (!_overlay) return;
    QVector<int> idxs;
    if (index >= 0) idxs.append(index);
    _overlay->setHighlightedIndices(idxs);
    _overlay->setShowOnlyHighlighted(true);
    scheduleOverlayUpdate();
}

void DualImageViewer::clearMatchHighlights()
{
    if (!_overlay) return;
    _overlay->clearHighlightedIndices();
    _overlay->setShowOnlyHighlighted(false);
    scheduleOverlayUpdate();
}

void DualImageViewer::setShowAllMatches(bool showAll)
{
    if (!_overlay) return;
    _overlay->setShowOnlyHighlighted(!showAll);
    if (showAll) _overlay->clearHighlightedIndices();
    scheduleOverlayUpdate();
}

DualImageViewer::~DualImageViewer()
{
    const QSet<QFutureWatcher<MatchParseResult> *> watchers = _matchLoadWatchers;
    for (QFutureWatcher<MatchParseResult> *watcher : watchers)
    {
        disconnect(watcher, nullptr, nullptr, nullptr);
        watcher->cancel();
    }
    for (QFutureWatcher<MatchParseResult> *watcher : watchers)
    {
        watcher->waitForFinished();
    }
    _matchLoadWatchers.clear();

    // 停止并断开延迟更新定时器，防止在widget销毁后触发回调
    if (_overlayUpdateTimer) {
        _overlayUpdateTimer->stop();
        disconnect(_overlayUpdateTimer, nullptr, this, nullptr);
    }

    // 断开与视图的信号连接，清理覆盖层内容
    if (_leftView) {
        disconnect(_leftView, nullptr, this, nullptr);
        _leftView->clearMatchPoints();
    }
    if (_rightView) {
        disconnect(_rightView, nullptr, this, nullptr);
        _rightView->clearMatchPoints();
    }
    if (_overlay) {
        disconnect(_overlay, nullptr, this, nullptr);
        // 清空覆盖层数据，避免后续paint访问已释放的数据
        _overlay->setMatches(QVector<QPointF>(), QVector<QPointF>());
        _overlay->hide();
    }
}

void DualImageViewer::setupLayout()
{
    Ui::DualImageViewer ui;
    ui.setupUi(this);

    _leftView = ui.m_leftView;
    _rightView = ui.m_rightView;
    ui.m_splitter->setStretchFactor(0, 1);
    ui.m_splitter->setStretchFactor(1, 1);
    
    // 创建覆盖层（在所有控件之上）
    _overlay = new MatchLineOverlay(this);
    _overlay->setViewWidgets(_leftView, _rightView);
    _overlay->setAttribute(Qt::WA_TransparentForMouseEvents); // 鼠标事件穿透
    _overlay->raise(); // 确保在最上层
    _overlay->show(); // 显式显示
}

void DualImageViewer::connectSignals()
{
    // 连接视图变化信号
    connect(_leftView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::onLeftViewChanged);
    connect(_rightView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::onRightViewChanged);
    
    // 覆盖层更新
    connect(_leftView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::scheduleOverlayUpdate);
    connect(_rightView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::scheduleOverlayUpdate);
    connect(_overlay, &MatchLineOverlay::visibleMatchesChanged,
            this, &DualImageViewer::scheduleOverlayUpdate, Qt::QueuedConnection);

    auto forwardImageLoadFailure = [this](const QString &, const QString &message)
    {
        emit loadFailed(message);
    };
    connect(_leftView, &ImageViewWidget::imageLoadFailed,
            this, forwardImageLoadFailure, Qt::QueuedConnection);
    connect(_rightView, &ImageViewWidget::imageLoadFailed,
            this, forwardImageLoadFailure, Qt::QueuedConnection);
    connect(_rightView, &ImageViewWidget::viewRightClicked,
            this, &DualImageViewer::markerCandidatePicked);
}

void DualImageViewer::setMarkerMeasurement(const QString &anchorImage,
                                           const QString &candidateImage,
                                           const QPointF &anchorPixel,
                                           const std::optional<QPointF> &candidatePixel)
{
    ++_matchLoadGeneration;
    const QVector<QPointF> anchor_points{anchorPixel};
    const QVector<QPointF> candidate_points = candidatePixel.has_value()
        ? QVector<QPointF>{candidatePixel.value()}
        : QVector<QPointF>{};
    _leftView->loadImage(anchorImage);
    _rightView->loadImage(candidateImage);
    _leftView->setMatchPoints(anchor_points);
    _rightView->setMatchPoints(candidate_points);
    _matchPtsA = candidatePixel.has_value() ? anchor_points : QVector<QPointF>{};
    _matchPtsB = candidate_points;
    _overlay->setMatches(_matchPtsA, _matchPtsB);
    _overlay->setInlierMask(QVector<bool>());
    _overlay->setVisible(candidatePixel.has_value());
    updateOverlayGeometry();
    updateOverlayNow();
}

bool DualImageViewer::loadMatchPair(const QString &imgA, const QString &imgB,
                                    const QString &matchFile,
                                    const QString &algorithmId,
                                    std::uint32_t algorithmVersion,
                                    const QByteArray &configFingerprint)
{
    const quint64 generation = ++_matchLoadGeneration;
    if (matchFile.trimmed().isEmpty())
    {
        applyMatchPairData(imgA, imgB, QVector<QPointF>{}, QVector<QPointF>{});
        return true;
    }

    QFuture<MatchParseResult> future = QtConcurrent::run(
        [matchFile, imgA, imgB, algorithmId, algorithmVersion, configFingerprint]()
        {
            return parseMatchFile(matchFile,
                                  imgA,
                                  imgB,
                                  algorithmId,
                                  algorithmVersion,
                                  configFingerprint);
        });
    auto *watcher = new QFutureWatcher<MatchParseResult>(this);
    _matchLoadWatchers.insert(watcher);
    QPointer<DualImageViewer> self(this);
    connect(watcher, &QFutureWatcher<MatchParseResult>::finished,
            watcher, [self, watcher, generation, imgA, imgB, matchFile]()
    {
        if (self)
        {
            self->_matchLoadWatchers.remove(watcher);
        }
        const MatchParseResult result = watcher->result();
        watcher->deleteLater();
        if (!self || generation != self->_matchLoadGeneration)
        {
            return;
        }
        if (!result.success)
        {
            emit self->loadFailed(result.error.isEmpty()
                ? self->tr("无法解析匹配文件：%1").arg(matchFile)
                : result.error);
            return;
        }

        self->applyMatchPairData(imgA, imgB, result.pointsA, result.pointsB);
        if (result.inlierMask.size() == result.pointsA.size())
        {
            self->_overlay->setInlierMask(result.inlierMask);
            const int valid_count = static_cast<int>(std::count(
                result.inlierMask.cbegin(), result.inlierMask.cend(), true));
            emit self->matchValidityLoaded(
                valid_count, result.inlierMask.size() - valid_count);
        }
        else
        {
            self->_overlay->setInlierMask(QVector<bool>());
            emit self->matchValidityLoaded(-1, -1);
        }
    });
    watcher->setFuture(future);
    return true;
}

void DualImageViewer::loadMatchPair(const QString &imgA, const QString &imgB,
                                    const QVector<QPointF> &ptsA,
                                    const QVector<QPointF> &ptsB)
{
    ++_matchLoadGeneration;
    applyMatchPairData(imgA, imgB, ptsA, ptsB);
}

void DualImageViewer::applyMatchPairData(const QString &imgA,
                                         const QString &imgB,
                                         const QVector<QPointF> &ptsA,
                                         const QVector<QPointF> &ptsB)
{
    // 加载图像
    if (!_leftView->loadImage(imgA)) {
        emit loadFailed(tr("无法加载图像：%1").arg(imgA));
        return;
    }
    
    if (!_rightView->loadImage(imgB)) {
        emit loadFailed(tr("无法加载图像：%1").arg(imgB));
        return;
    }
    
    // 保存匹配数据
    _matchPtsA = ptsA;
    _matchPtsB = ptsB;
    
    // 设置匹配点到视图
    _leftView->setMatchPoints(ptsA);
    _rightView->setMatchPoints(ptsB);
    
    // 设置匹配数据到覆盖层
    _overlay->setMatches(ptsA, ptsB);
    _overlay->setInlierMask(QVector<bool>());
    
    // 更新覆盖层几何
    updateOverlayGeometry();
    // 立即计算并同步点的可见性
    updateOverlayNow();
    
    emit matchDataLoaded(ptsA.size());
    emit matchValidityLoaded(-1, -1);
}

void DualImageViewer::setSyncMode(bool enabled)
{
    if (_syncEnabled == enabled) return;
    
    _syncEnabled = enabled;
    
    // 如果启用同步，立即同步一次
    if (enabled && _leftView && _rightView) {
        _syncing = true;
        _rightView->setTransform(_leftView->currentTransform());
        _syncing = false;
    }
}

void DualImageViewer::fitBothViews()
{
    _leftView->fitToView();
    _rightView->fitToView();
}

void DualImageViewer::resetBothViews()
{
    _leftView->resetZoom();
    _rightView->resetZoom();
}

void DualImageViewer::clearViewer()
{
    ++_matchLoadGeneration;
    _matchPtsA.clear();
    _matchPtsB.clear();
    if (_leftView)
    {
        _leftView->clearImage();
    }
    if (_rightView)
    {
        _rightView->clearImage();
    }
    if (_overlay)
    {
        _overlay->setMatches({}, {});
        _overlay->clearHighlightedIndices();
    }
    updateOverlayGeometry();
}

QString DualImageViewer::leftImagePath() const
{
    return _leftView ? _leftView->imagePath() : QString();
}

QString DualImageViewer::rightImagePath() const
{
    return _rightView ? _rightView->imagePath() : QString();
}

void DualImageViewer::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    updateOverlayGeometry();
    scheduleOverlayUpdate();
}

void DualImageViewer::onLeftViewChanged(const QTransform &transform)
{
    if (_syncEnabled && !_syncing) {
        _syncing = true;
        _rightView->setTransform(transform);
        _syncing = false;
    }
}

void DualImageViewer::onRightViewChanged(const QTransform &transform)
{
    if (_syncEnabled && !_syncing) {
        _syncing = true;
        _leftView->setTransform(transform);
        _syncing = false;
    }
}

void DualImageViewer::scheduleOverlayUpdate()
{
    // 以 0 ms 单次定时器合并同一事件循环内的连续变换，同时保持拖动跟手。
    if (!_overlayUpdateTimer) return;
    if (!_overlayUpdateTimer->isActive()) {
        _overlayUpdateTimer->start();
    }
}

void DualImageViewer::updateOverlayNow()
{
    if (_overlayUpdateTimer)
    {
        _overlayUpdateTimer->stop();
    }
    updateOverlayGeometry();
    // 防护：若组件已经被删除或指针失效则不进行更新
    if (!_overlay || !_leftView || !_rightView) return;
    _overlay->updateOverlay();
    
    const QVector<int> visible_matches = _overlay->visibleMatches();
    _leftView->setVisibleMatchIndices(visible_matches);
    _rightView->setVisibleMatchIndices(visible_matches);
}

void DualImageViewer::updateOverlayGeometry()
{
    if (_overlay) {
        _overlay->setGeometry(rect());
        _overlay->raise();
        _overlay->show();
    }
}

DualImageViewer::MatchParseResult DualImageViewer::parseMatchFile(
    const QString &matchFile,
    const QString &imgA,
    const QString &imgB,
    const QString &algorithmId,
    std::uint32_t algorithmVersion,
    const QByteArray &configFingerprint)
{
    MatchParseResult result;

    xjw::image_matching::ImageMatchShard shard;
    QString read_error;
    if (!xjw::image_matching::ImageMatchFile::read(matchFile, &shard, &read_error))
    {
        qWarning() << "无法读取匹配分片:" << matchFile << read_error;
        result.error = tr("无法读取匹配文件 %1：%2").arg(matchFile, read_error);
        return result;
    }

    const bool owner_is_a = identityMatchesImage(shard.owner, imgA);
    const bool owner_is_b = identityMatchesImage(shard.owner, imgB);
    if (!owner_is_a && !owner_is_b)
    {
        qWarning() << "匹配分片 owner 与查看影像不一致:" << shard.owner.path;
        result.error = tr("匹配文件与当前影像不一致：%1").arg(matchFile);
        return result;
    }
    const QString peer_image = owner_is_a ? imgB : imgA;

    const xjw::image_matching::NeighborMatchBlock *selected_block = nullptr;
    for (const xjw::image_matching::NeighborMatchBlock &block : shard.neighbors)
    {
        if (!identityMatchesImage(block.peer, peer_image))
        {
            continue;
        }
        if (!algorithmId.trimmed().isEmpty() &&
            !block.isCompatible(algorithmId, algorithmVersion, configFingerprint))
        {
            continue;
        }
        if (!selected_block || betterBlock(block, *selected_block))
        {
            selected_block = &block;
        }
    }
    if (!selected_block)
    {
        result.error = tr("匹配文件中没有当前像对的数据：%1").arg(matchFile);
        return result;
    }

    result.pointsA.reserve(static_cast<int>(selected_block->matches.size()));
    result.pointsB.reserve(static_cast<int>(selected_block->matches.size()));
    result.inlierMask.reserve(static_cast<int>(selected_block->matches.size()));
    for (const xjw::image_matching::MatchRecord &match : selected_block->matches)
    {
        const xjw::image_matching::KeypointObservation *owner_observation =
            selected_block->findOwnerObservation(match.ownerFeatureId);
        if (!owner_observation)
        {
            continue;
        }

        const QPointF owner_point(owner_observation->x, owner_observation->y);
        const QPointF peer_point(match.peerX, match.peerY);
        result.pointsA.append(owner_is_a ? owner_point : peer_point);
        result.pointsB.append(owner_is_a ? peer_point : owner_point);
        result.inlierMask.append(xjw::image_matching::hasFlag(
            match.flags, xjw::image_matching::MatchRecordFlag::GeometryInlier));
    }
    result.success = !result.pointsA.isEmpty()
        && result.pointsA.size() == result.pointsB.size();
    if (!result.success)
    {
        result.error = tr("匹配文件中没有可显示的匹配点：%1").arg(matchFile);
    }
    return result;
}
