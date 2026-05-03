#include "DualImageViewer.h"
#include "ImageViewWidget.h"
#include "MatchLineOverlay.h"
#include "DisparityHeatmapOverlay.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QTextStream>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSplitter>
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

}

DualImageViewer::DualImageViewer(QWidget *parent)
    : QWidget(parent)
    , m_leftView(nullptr)
    , m_rightView(nullptr)
    , m_overlay(nullptr)
    , m_disparityOverlay(nullptr)
    , m_syncEnabled(false)
    , m_syncing(false)
{
    setupLayout();
    connectSignals();

    // 创建延迟更新定时器
    m_overlayUpdateTimer = new QTimer(this);
    m_overlayUpdateTimer->setSingleShot(true);
    m_overlayUpdateTimer->setInterval(16); // ~60 FPS
    connect(m_overlayUpdateTimer, &QTimer::timeout,
            this, &DualImageViewer::updateOverlayNow);

    // 创建视差热力图覆盖层
    m_disparityOverlay = new DisparityHeatmapOverlay(this);
    m_disparityOverlay->hide();
}

ImageViewWidget* DualImageViewer::leftView() const
{
    return m_leftView.data();
}

ImageViewWidget* DualImageViewer::rightView() const
{
    return m_rightView.data();
}

MatchLineOverlay* DualImageViewer::overlay() const
{
    return m_overlay.data();
}

DisparityHeatmapOverlay* DualImageViewer::disparityOverlay() const
{
    return m_disparityOverlay;
}

void DualImageViewer::setOverlayMode(int mode)
{
    m_overlayMode = mode;
    m_overlay->setVisible(mode == 0);
    m_disparityOverlay->setVisible(mode == 1);
    scheduleOverlayUpdate();
}

void DualImageViewer::highlightMatchIndex(int index)
{
    if (!m_overlay) return;
    QVector<int> idxs;
    if (index >= 0) idxs.append(index);
    m_overlay->setHighlightedIndices(idxs);
    m_overlay->setShowOnlyHighlighted(true);
    scheduleOverlayUpdate();
}

void DualImageViewer::highlightMatchIndices(const QVector<int> &indices)
{
    if (!m_overlay) return;
    m_overlay->setHighlightedIndices(indices);
    m_overlay->setShowOnlyHighlighted(true);
    scheduleOverlayUpdate();
}

void DualImageViewer::clearMatchHighlights()
{
    if (!m_overlay) return;
    m_overlay->clearHighlightedIndices();
    m_overlay->setShowOnlyHighlighted(false);
    scheduleOverlayUpdate();
}

void DualImageViewer::setShowAllMatches(bool showAll)
{
    if (!m_overlay) return;
    m_overlay->setShowOnlyHighlighted(!showAll);
    if (showAll) m_overlay->clearHighlightedIndices();
    scheduleOverlayUpdate();
}

DualImageViewer::~DualImageViewer()
{
    // 停止并断开延迟更新定时器，防止在widget销毁后触发回调
    if (m_overlayUpdateTimer) {
        m_overlayUpdateTimer->stop();
        disconnect(m_overlayUpdateTimer, nullptr, this, nullptr);
    }

    // 断开与视图的信号连接，清理覆盖层内容
    if (m_leftView) {
        disconnect(m_leftView, nullptr, this, nullptr);
        m_leftView->clearMatchPoints();
    }
    if (m_rightView) {
        disconnect(m_rightView, nullptr, this, nullptr);
        m_rightView->clearMatchPoints();
    }
    if (m_overlay) {
        disconnect(m_overlay, nullptr, this, nullptr);
        // 清空覆盖层数据，避免后续paint访问已释放的数据
        m_overlay->setMatches(QVector<QPointF>(), QVector<QPointF>());
        m_overlay->hide();
    }
}

void DualImageViewer::setupLayout()
{
    // 创建左右视图
    m_leftView = new ImageViewWidget(this);
    m_rightView = new ImageViewWidget(this);
    
    // 使用 QSplitter 允许用户调整左右比例
    QSplitter *splitter = new QSplitter(Qt::Horizontal, this);
    splitter->addWidget(m_leftView);
    splitter->addWidget(m_rightView);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    
    // 主布局
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(splitter);
    setLayout(mainLayout);
    
    // 创建覆盖层（在所有控件之上）
    m_overlay = new MatchLineOverlay(this);
    m_overlay->setViewWidgets(m_leftView, m_rightView);
    m_overlay->setAttribute(Qt::WA_TransparentForMouseEvents); // 鼠标事件穿透
    m_overlay->raise(); // 确保在最上层
    m_overlay->show(); // 显式显示
}

void DualImageViewer::connectSignals()
{
    // 连接视图变化信号
    connect(m_leftView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::onLeftViewChanged, Qt::QueuedConnection);
    connect(m_rightView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::onRightViewChanged, Qt::QueuedConnection);
    
    // 覆盖层更新
    connect(m_leftView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::scheduleOverlayUpdate, Qt::QueuedConnection);
    connect(m_rightView, &ImageViewWidget::viewTransformChanged,
            this, &DualImageViewer::scheduleOverlayUpdate, Qt::QueuedConnection);
}

bool DualImageViewer::loadMatchPair(const QString &imgA, const QString &imgB,
                                    const QString &matchFile)
{
    // 解析匹配文件
    QVector<QPointF> ptsA, ptsB;
    if (!parseMatchFile(matchFile, ptsA, ptsB)) {
        emit loadFailed(tr("无法解析匹配文件：%1").arg(matchFile));
        return false;
    }
    
    // 加载图像和匹配点
    loadMatchPair(imgA, imgB, ptsA, ptsB);
    return true;
}

void DualImageViewer::loadMatchPair(const QString &imgA, const QString &imgB,
                                    const QVector<QPointF> &ptsA,
                                    const QVector<QPointF> &ptsB)
{
    // 加载图像
    if (!m_leftView->loadImage(imgA)) {
        emit loadFailed(tr("无法加载图像：%1").arg(imgA));
        return;
    }
    
    if (!m_rightView->loadImage(imgB)) {
        emit loadFailed(tr("无法加载图像：%1").arg(imgB));
        return;
    }
    
    // 保存匹配数据
    m_matchPtsA = ptsA;
    m_matchPtsB = ptsB;
    
    // 设置匹配点到视图
    m_leftView->setMatchPoints(ptsA);
    m_rightView->setMatchPoints(ptsB);
    
    // 设置匹配数据到覆盖层
    m_overlay->setMatches(ptsA, ptsB);
    
    // 更新覆盖层几何
    updateOverlayGeometry();
    // 立即计算并同步点的可见性
    updateOverlayNow();
    
    emit matchDataLoaded(ptsA.size());
}

void DualImageViewer::setSyncMode(bool enabled)
{
    if (m_syncEnabled == enabled) return;
    
    m_syncEnabled = enabled;
    emit syncModeChanged(enabled);
    
    // 如果启用同步，立即同步一次
    if (enabled && m_leftView && m_rightView) {
        m_syncing = true;
        m_rightView->setTransform(m_leftView->currentTransform());
        m_syncing = false;
    }
}

void DualImageViewer::fitBothViews()
{
    m_leftView->fitToView();
    m_rightView->fitToView();
}

void DualImageViewer::resetBothViews()
{
    m_leftView->resetZoom();
    m_rightView->resetZoom();
}

int DualImageViewer::totalMatchCount() const
{
    return m_matchPtsA.size();
}

int DualImageViewer::visibleMatchCount() const
{
    if (!m_overlay) return -1;
    return m_overlay->visibleMatches().size();
}

QString DualImageViewer::leftImagePath() const
{
    return m_leftView ? m_leftView->imagePath() : QString();
}

QString DualImageViewer::rightImagePath() const
{
    return m_rightView ? m_rightView->imagePath() : QString();
}

void DualImageViewer::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    updateOverlayGeometry();
}

void DualImageViewer::onLeftViewChanged(const QTransform &transform)
{
    if (m_syncEnabled && !m_syncing) {
        m_syncing = true;
        m_rightView->setTransform(transform);
        m_syncing = false;
    }
}

void DualImageViewer::onRightViewChanged(const QTransform &transform)
{
    if (m_syncEnabled && !m_syncing) {
        m_syncing = true;
        m_leftView->setTransform(transform);
        m_syncing = false;
    }
}

void DualImageViewer::scheduleOverlayUpdate()
{
    // 延迟更新以提高性能（合并多次连续的更新请求）
    if (!m_overlayUpdateTimer) return;
    if (!m_overlayUpdateTimer->isActive()) {
        m_overlayUpdateTimer->start();
    }
}

void DualImageViewer::updateOverlayNow()
{
    updateOverlayGeometry();
    // 防护：若组件已经被删除或指针失效则不进行更新
    if (!m_overlay || !m_leftView || !m_rightView) return;
    m_overlay->updateOverlay();
    
    // 同步点的可见性：仅显示在两边都可见的匹配点
    QVector<int> vis = m_overlay->visibleMatches();

    // 为避免使用不正确的大小（可能导致大内存分配），使用视图中实际的点图元数量
    int leftCount = m_leftView ? m_leftView->matchItemCount() : 0;
    int rightCount = m_rightView ? m_rightView->matchItemCount() : 0;
    int useCount = qMax(leftCount, rightCount);
    if (useCount <= 0) return;

    QVector<bool> mask(useCount, false);
    if (m_overlay->showOnlyHighlighted()) {
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

    if (m_leftView) m_leftView->setMatchVisibilityMask(mask);
    if (m_rightView) m_rightView->setMatchVisibilityMask(mask);
}

void DualImageViewer::updateOverlayGeometry()
{
    if (m_overlay) {
        m_overlay->setGeometry(rect());
        m_overlay->raise();
        m_overlay->show();
    }
    if (m_disparityOverlay) {
        m_disparityOverlay->setGeometry(rect());
        m_disparityOverlay->raise();
    }
}

bool DualImageViewer::parseMatchFile(const QString &matchFile,
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

        // 优先读取与 .match 同名的 sidecar json（由 SuperGlueRunner 写入）
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
                    for (int i = 0; i < m0.size(); ++i) {
                        const QJsonArray p0 = m0.at(i).toArray();
                        const QJsonArray p1 = m1.at(i).toArray();
                        if (p0.size() >= 2 && p1.size() >= 2) {
                            ptsA.append(QPointF(p0.at(0).toDouble(), p0.at(1).toDouble()));
                            ptsB.append(QPointF(p1.at(0).toDouble(), p1.at(1).toDouble()));
                        }
                    }
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

        for (int idx0 = 0; idx0 < matches0.size(); ++idx0) {
            int idx1 = matches0[idx0];
            if (idx1 >= 0 && idx0 < kpts0.size() && idx1 < kpts1.size()) {
                ptsA.append(kpts0[idx0]);
                ptsB.append(kpts1[idx1]);
            }
        }

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
