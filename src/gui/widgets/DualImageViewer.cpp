#include "DualImageViewer.h"

#include "ui_DualImageViewer.h"

#include "ImageViewWidget.h"
#include "MatchLineOverlay.h"
#include "DisparityHeatmapOverlay.h"
#include "ImageMatchFile.h"
#include "project/ProjectMetadata.h"
#include <QFileInfo>

#include <QTimer>
#include <QMessageBox>
#include <QDebug>

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
    , _disparityOverlay(nullptr)
    , _syncEnabled(false)
    , _syncing(false)
{
    setupLayout();
    connectSignals();

    // 创建延迟更新定时器
    _overlayUpdateTimer = new QTimer(this);
    _overlayUpdateTimer->setSingleShot(true);
    _overlayUpdateTimer->setInterval(16); // ~60 FPS
    connect(_overlayUpdateTimer, &QTimer::timeout,
            this, &DualImageViewer::updateOverlayNow);

    // 创建视差热力图覆盖层
    _disparityOverlay = new DisparityHeatmapOverlay(this);
    _disparityOverlay->hide();
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

DisparityHeatmapOverlay* DualImageViewer::disparityOverlay() const
{
    return _disparityOverlay;
}

void DualImageViewer::setOverlayMode(int mode)
{
    _overlayMode = mode;
    _overlay->setVisible(mode == 0);
    _disparityOverlay->setVisible(mode == 1);
    scheduleOverlayUpdate();
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
            this, &DualImageViewer::onLeftViewChanged, Qt::QueuedConnection);
    connect(_rightView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::onRightViewChanged, Qt::QueuedConnection);
    
    // 覆盖层更新
    connect(_leftView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::scheduleOverlayUpdate, Qt::QueuedConnection);
    connect(_rightView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::scheduleOverlayUpdate, Qt::QueuedConnection);
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
    if (matchFile.trimmed().isEmpty()) {
        loadMatchPair(imgA, imgB, QVector<QPointF>{}, QVector<QPointF>{});
        return true;
    }

    // 解析匹配文件
    QVector<QPointF> ptsA, ptsB;
    QVector<bool> inlier_mask;
    if (!parseMatchFile(matchFile,
                        imgA,
                        imgB,
                        algorithmId,
                        algorithmVersion,
                        configFingerprint,
                        ptsA,
                        ptsB,
                        inlier_mask)) {
        emit loadFailed(tr("无法解析匹配文件：%1").arg(matchFile));
        return false;
    }
    
    // 加载图像和匹配点
    loadMatchPair(imgA, imgB, ptsA, ptsB);
    if (inlier_mask.size() == ptsA.size())
    {
        _overlay->setInlierMask(inlier_mask);
        const int valid_count = static_cast<int>(std::count(
            inlier_mask.cbegin(), inlier_mask.cend(), true));
        emit matchValidityLoaded(valid_count, inlier_mask.size() - valid_count);
    }
    else
    {
        _overlay->setInlierMask(QVector<bool>());
        emit matchValidityLoaded(-1, -1);
    }
    return true;
}

void DualImageViewer::loadMatchPair(const QString &imgA, const QString &imgB,
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
    if (_disparityOverlay)
    {
        _disparityOverlay->hide();
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
    // 延迟更新以提高性能（合并多次连续的更新请求）
    if (!_overlayUpdateTimer) return;
    if (!_overlayUpdateTimer->isActive()) {
        _overlayUpdateTimer->start();
    }
}

void DualImageViewer::updateOverlayNow()
{
    updateOverlayGeometry();
    // 防护：若组件已经被删除或指针失效则不进行更新
    if (!_overlay || !_leftView || !_rightView) return;
    _overlay->updateOverlay();
    
    // 同步点的可见性：仅显示在两边都可见的匹配点
    QVector<int> vis = _overlay->visibleMatches();

    // 为避免使用不正确的大小（可能导致大内存分配），使用视图中实际的点图元数量
    int leftCount = _leftView ? _leftView->matchItemCount() : 0;
    int rightCount = _rightView ? _rightView->matchItemCount() : 0;
    int useCount = qMax(leftCount, rightCount);
    if (useCount <= 0) return;

    // 端点必须与实际可见连线使用同一索引集合。空集合表示当前没有可显示
    // 的连线，而不是“未设置筛选”，因此此时保持全部端点隐藏。
    QVector<bool> mask(useCount, false);
    for (int idx : vis)
    {
        if (idx >= 0 && idx < useCount)
        {
            mask[idx] = true;
        }
    }

    if (_leftView) _leftView->setMatchVisibilityMask(mask);
    if (_rightView) _rightView->setMatchVisibilityMask(mask);
}

void DualImageViewer::updateOverlayGeometry()
{
    if (_overlay) {
        _overlay->setGeometry(rect());
        _overlay->raise();
        _overlay->show();
    }
    if (_disparityOverlay) {
        _disparityOverlay->setGeometry(rect());
        _disparityOverlay->raise();
    }
}

bool DualImageViewer::parseMatchFile(const QString &matchFile,
                                     const QString &imgA,
                                     const QString &imgB,
                                     const QString &algorithmId,
                                     std::uint32_t algorithmVersion,
                                     const QByteArray &configFingerprint,
                                     QVector<QPointF> &ptsA,
                                     QVector<QPointF> &ptsB,
                                     QVector<bool> &inlierMask)
{
    ptsA.clear();
    ptsB.clear();
    inlierMask.clear();

    xjw::image_matching::ImageMatchShard shard;
    QString read_error;
    if (!xjw::image_matching::ImageMatchFile::read(matchFile, &shard, &read_error))
    {
        qWarning() << "无法读取匹配分片:" << matchFile << read_error;
        return false;
    }

    const bool owner_is_a = identityMatchesImage(shard.owner, imgA);
    const bool owner_is_b = identityMatchesImage(shard.owner, imgB);
    if (!owner_is_a && !owner_is_b)
    {
        qWarning() << "匹配分片 owner 与查看影像不一致:" << shard.owner.path;
        return false;
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
        return false;
    }

    ptsA.reserve(static_cast<int>(selected_block->matches.size()));
    ptsB.reserve(static_cast<int>(selected_block->matches.size()));
    inlierMask.reserve(static_cast<int>(selected_block->matches.size()));
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
        ptsA.append(owner_is_a ? owner_point : peer_point);
        ptsB.append(owner_is_a ? peer_point : owner_point);
        inlierMask.append(xjw::image_matching::hasFlag(
            match.flags, xjw::image_matching::MatchRecordFlag::GeometryInlier));
    }
    return !ptsA.isEmpty() && ptsA.size() == ptsB.size();
}
