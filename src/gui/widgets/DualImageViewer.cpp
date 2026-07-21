#include "DualImageViewer.h"

#include "ui_DualImageViewer.h"

#include "ImageViewWidget.h"
#include "MatchLineOverlay.h"
#include "DisparityHeatmapOverlay.h"
#include "MatchValidityAnalyzer.h"
#include "project/ProjectMetadata.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QTextStream>

#include <QTimer>
#include <QFile>
#include <QDataStream>
#include <QMessageBox>
#include <QDebug>
#include <QRegularExpression>

#include <algorithm>

namespace {

bool readSgmtMatchFile(const QString &matchFile,
                       QString &image0Name,
                       QString &image1Name,
                       QVector<int> &matches0)
{
    QFile file(matchFile);
    if (!file.open(QIODevice::ReadOnly)) return false;

    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);

    char magic[4];
    if (in.readRawData(magic, 4) != 4 || strncmp(magic, "SGMT", 4) != 0) {
        return false;
    }

    quint32 version = 0;
    in >> version;
    if (version != 1) return false;

    quint32 img0Len = 0;
    quint32 img1Len = 0;
    in >> img0Len;
    QByteArray img0Bytes(img0Len, 0);
    if (in.readRawData(img0Bytes.data(), img0Len) != static_cast<int>(img0Len)) return false;
    image0Name = QString::fromUtf8(img0Bytes);

    in >> img1Len;
    QByteArray img1Bytes(img1Len, 0);
    if (in.readRawData(img1Bytes.data(), img1Len) != static_cast<int>(img1Len)) return false;
    image1Name = QString::fromUtf8(img1Bytes);

    qint32 numMatches = 0;
    qint32 numKp0 = 0;
    qint32 numKp1 = 0;
    in >> numMatches >> numKp0 >> numKp1;
    Q_UNUSED(numMatches);
    Q_UNUSED(numKp1);

    if (numKp0 < 0) return false;
    matches0.resize(numKp0);

    for (int i = 0; i < numKp0; ++i) {
        qint32 matchIdx = -1;
        float score = 0.0f;
        in >> matchIdx >> score;
        Q_UNUSED(score);
        matches0[i] = static_cast<int>(matchIdx);
    }

    return true;
}

bool readSpPoints(const QString &spPath, QVector<QPointF> &points)
{
    points.clear();
    QFile file(spPath);
    if (!file.open(QIODevice::ReadOnly)) return false;

    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);
    in.setByteOrder(QDataStream::LittleEndian);

    char magic[4];
    if (in.readRawData(magic, 4) != 4 || strncmp(magic, "SPBT", 4) != 0) {
        return false;
    }

    quint32 version = 0;
    in >> version;
    if (version != 1) return false;

    quint32 imageNameLen = 0;
    in >> imageNameLen;
    QByteArray imageNameBytes(imageNameLen, 0);
    if (in.readRawData(imageNameBytes.data(), imageNameLen) != static_cast<int>(imageNameLen)) return false;

    quint32 numKeypoints = 0;
    in >> numKeypoints;
    points.reserve(static_cast<int>(numKeypoints));

    for (quint32 i = 0; i < numKeypoints; ++i) {
        float x = 0.f;
        float y = 0.f;
        float score = 0.f;
        in >> x >> y >> score;
        Q_UNUSED(score);
        points.append(QPointF(x, y));
    }

    return true;
}

QString findExistingPath(const QStringList &candidates)
{
    for (const QString &cand : candidates) {
        QString clean = QDir::cleanPath(cand);
        if (QFile::exists(clean)) return clean;
    }
    return QString();
}

using xjw::common::project::imageReferenceMatchesToken;

enum class MatchFileDisplayOrder
{
    Direct,
    Reversed,
    Unknown
};

MatchFileDisplayOrder displayOrderForMatchFile(const QString &fileImage0Path,
                                               const QString &fileImage0Name,
                                               const QString &fileImage1Path,
                                               const QString &fileImage1Name,
                                               const QString &displayImageA,
                                               const QString &displayImageB)
{
    const bool direct =
        imageReferenceMatchesToken(fileImage0Path, fileImage0Name, displayImageA) &&
        imageReferenceMatchesToken(fileImage1Path, fileImage1Name, displayImageB);
    const bool reversed =
        imageReferenceMatchesToken(fileImage0Path, fileImage0Name, displayImageB) &&
        imageReferenceMatchesToken(fileImage1Path, fileImage1Name, displayImageA);

    if (direct)
    {
        return MatchFileDisplayOrder::Direct;
    }
    if (reversed)
    {
        return MatchFileDisplayOrder::Reversed;
    }
    return MatchFileDisplayOrder::Unknown;
}

void appendSidecarMatchedPoints(const QJsonArray &points0,
                                const QJsonArray &points1,
                                bool reversed,
                                QVector<QPointF> &ptsA,
                                QVector<QPointF> &ptsB)
{
    for (int i = 0; i < points0.size(); ++i)
    {
        const QJsonArray p0 = points0.at(i).toArray();
        const QJsonArray p1 = points1.at(i).toArray();
        if (p0.size() < 2 || p1.size() < 2)
        {
            continue;
        }

        const QJsonArray &leftPoint = reversed ? p1 : p0;
        const QJsonArray &rightPoint = reversed ? p0 : p1;
        ptsA.append(QPointF(leftPoint.at(0).toDouble(), leftPoint.at(1).toDouble()));
        ptsB.append(QPointF(rightPoint.at(0).toDouble(), rightPoint.at(1).toDouble()));
    }
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

void DualImageViewer::highlightMatchIndices(const QVector<int> &indices)
{
    if (!_overlay) return;
    _overlay->setHighlightedIndices(indices);
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
                                    const QString &matchFile)
{
    if (matchFile.trimmed().isEmpty()) {
        loadMatchPair(imgA, imgB, QVector<QPointF>{}, QVector<QPointF>{});
        return true;
    }

    // 解析匹配文件
    QVector<QPointF> ptsA, ptsB;
    if (!parseMatchFile(matchFile, imgA, imgB, ptsA, ptsB)) {
        emit loadFailed(tr("无法解析匹配文件：%1").arg(matchFile));
        return false;
    }
    
    // 加载图像和匹配点
    loadMatchPair(imgA, imgB, ptsA, ptsB);
    const MatchValidityResult validity = analyzeMatchTrackValidity(matchFile, imgA, imgB);
    if (validity.hasTrackValidity && validity.inlierMask.size() == ptsA.size())
    {
        _overlay->setInlierMask(validity.inlierMask);
        emit matchValidityLoaded(validity.validCount, validity.invalidCount);
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
    emit syncModeChanged(enabled);
    
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

int DualImageViewer::totalMatchCount() const
{
    return _matchPtsA.size();
}

int DualImageViewer::visibleMatchCount() const
{
    if (!_overlay) return -1;
    return _overlay->visibleMatches().size();
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

    QVector<bool> mask(useCount, false);
    if (_overlay->showOnlyHighlighted()) {
        // 高亮模式下不隐藏点，避免用户看不到已选点
        std::fill(mask.begin(), mask.end(), true);
    } else {
        if (vis.isEmpty()) {
            // 无可见连线时不应隐藏所有点（例如手动模式仅有待配对点）
            std::fill(mask.begin(), mask.end(), true);
        } else {
            for (int idx : vis) {
                if (idx >= 0 && idx < useCount) mask[idx] = true;
            }
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
                                     QVector<QPointF> &ptsA,
                                     QVector<QPointF> &ptsB)
{
    ptsA.clear();
    ptsB.clear();

    QString image0Name, image1Name;
    QVector<int> matches0;
    if (readSgmtMatchFile(matchFile, image0Name, image1Name, matches0)) {
        QFileInfo matchFi(matchFile);
        QString matchDir = matchFi.absolutePath();
        QString assetsDir = QDir(matchDir).filePath("..");
        QString projectRoot = QDir(assetsDir).filePath("..");

        // 优先读取与 .match 同名的 sidecar json（由统一特征匹配流程写入）
        QString sp0Path;
        QString sp1Path;
        const QString sidecarPath = matchFile + ".json";
        QFile sidecarFile(sidecarPath);
        if (sidecarFile.open(QIODevice::ReadOnly)) {
            QJsonParseError perr;
            QJsonDocument sdoc = QJsonDocument::fromJson(sidecarFile.readAll(), &perr);
            sidecarFile.close();
            if (perr.error == QJsonParseError::NoError && sdoc.isObject()) {
                QJsonObject sobj = sdoc.object();

                // 优先直接使用 sidecar 中缓存的匹配点坐标（无需读取 .sp）
                QJsonArray m0 = sobj.value("matched_points0").toArray();
                QJsonArray m1 = sobj.value("matched_points1").toArray();
                if (!m0.isEmpty() && m0.size() == m1.size()) {
                    const MatchFileDisplayOrder order = displayOrderForMatchFile(
                        sobj.value("image0_path").toString(),
                        sobj.value("image0_name").toString(image0Name),
                        sobj.value("image1_path").toString(),
                        sobj.value("image1_name").toString(image1Name),
                        imgA,
                        imgB);
                    appendSidecarMatchedPoints(
                        m0,
                        m1,
                        order == MatchFileDisplayOrder::Reversed,
                        ptsA,
                        ptsB);
                    if (!ptsA.isEmpty() && ptsA.size() == ptsB.size()) {
                        return true;
                    }
                    ptsA.clear();
                    ptsB.clear();
                }

                sp0Path = sobj.value("sp0_path").toString();
                sp1Path = sobj.value("sp1_path").toString();
            }
        }

        if (sp0Path.isEmpty() || sp1Path.isEmpty()) {
            // 回退：尝试所有特征文件后缀 (.sp/.dsk/.alk/.sift 等)
            static const char *suffixes[] = {".sp",".dsk",".alk",".sift",".orb",".akz",".dedode"};
            for (const char *suf : suffixes) {
                if (sp0Path.isEmpty()) {
                    QStringList c0 = {
                        QDir(assetsDir).filePath("ip/" + image0Name + suf),
                        QDir(projectRoot).filePath("assets/ip/" + image0Name + suf),
                        QDir(projectRoot).filePath("ip/" + image0Name + suf),
                    };
                    sp0Path = findExistingPath(c0);
                }
                if (sp1Path.isEmpty()) {
                    QStringList c1 = {
                        QDir(assetsDir).filePath("ip/" + image1Name + suf),
                        QDir(projectRoot).filePath("assets/ip/" + image1Name + suf),
                        QDir(projectRoot).filePath("ip/" + image1Name + suf),
                    };
                    sp1Path = findExistingPath(c1);
                }
                if (!sp0Path.isEmpty() && !sp1Path.isEmpty()) break;
            }
        }

        if (sp0Path.isEmpty() || sp1Path.isEmpty()) {
            qWarning() << "无法找到特征文件:" << image0Name << image1Name;
            return false;
        }

        QVector<QPointF> kpts0;
        QVector<QPointF> kpts1;
        if (!readSpPoints(sp0Path, kpts0) || !readSpPoints(sp1Path, kpts1)) {
            qWarning() << "无法读取 .sp 文件:" << sp0Path << sp1Path;
            return false;
        }

        QVector<QPointF> filePts0;
        QVector<QPointF> filePts1;
        for (int idx0 = 0; idx0 < matches0.size(); ++idx0) {
            int idx1 = matches0[idx0];
            if (idx1 >= 0 && idx0 < kpts0.size() && idx1 < kpts1.size()) {
                filePts0.append(kpts0[idx0]);
                filePts1.append(kpts1[idx1]);
            }
        }

        const MatchFileDisplayOrder order = displayOrderForMatchFile(
            QString(),
            image0Name,
            QString(),
            image1Name,
            imgA,
            imgB);
        const bool reversed = order == MatchFileDisplayOrder::Reversed;
        ptsA = reversed ? filePts1 : filePts0;
        ptsB = reversed ? filePts0 : filePts1;

        return !ptsA.isEmpty();
    }
    
    // 回退：尝试 JSON 或文本格式（旧格式）
    QFile f(matchFile);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QByteArray raw = f.readAll();
    f.close();

    // Try JSON first
    QJsonParseError perr;
    QJsonDocument doc = QJsonDocument::fromJson(raw, &perr);
    if (doc.isArray()) {
        QJsonArray arr = doc.array();
        for (const QJsonValue &val : arr) {
            if (val.isArray()) {
                QJsonArray a = val.toArray();
                if (a.size() >= 4) {
                    double x1 = a.at(0).toDouble();
                    double y1 = a.at(1).toDouble();
                    double x2 = a.at(2).toDouble();
                    double y2 = a.at(3).toDouble();
                    ptsA.append(QPointF(x1, y1));
                    ptsB.append(QPointF(x2, y2));
                }
            } else if (val.isObject()) {
                QJsonObject o = val.toObject();
                double x1 = o.value(QStringLiteral("x1")).toDouble(o.value(QStringLiteral("xa")).toDouble());
                double y1 = o.value(QStringLiteral("y1")).toDouble(o.value(QStringLiteral("ya")).toDouble());
                double x2 = o.value(QStringLiteral("x2")).toDouble(o.value(QStringLiteral("xb")).toDouble());
                double y2 = o.value(QStringLiteral("y2")).toDouble(o.value(QStringLiteral("yb")).toDouble());
                ptsA.append(QPointF(x1, y1));
                ptsB.append(QPointF(x2, y2));
            }
        }
        return !ptsA.isEmpty();
    }

    // Fallback to text lines
    QTextStream ts(raw);
    while (!ts.atEnd()) {
        QString line = ts.readLine().trimmed();
        if (line.isEmpty()) continue;
        QStringList parts = line.split(QRegularExpression(QStringLiteral("[\\s,]+")), Qt::SkipEmptyParts);
        if (parts.size() < 4) continue;
        bool ok1,ok2,ok3,ok4;
        double x1 = parts.at(0).toDouble(&ok1);
        double y1 = parts.at(1).toDouble(&ok2);
        double x2 = parts.at(2).toDouble(&ok3);
        double y2 = parts.at(3).toDouble(&ok4);
        if (ok1 && ok2 && ok3 && ok4) {
            ptsA.append(QPointF(x1,y1));
            ptsB.append(QPointF(x2,y2));
        }
    }
    return !ptsA.isEmpty();
}
