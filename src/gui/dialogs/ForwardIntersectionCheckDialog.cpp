#include "ForwardIntersectionCheckDialog.h"
#include "ui_ForwardIntersectionCheckDialog.h"

#include "ProjectManager.h"
#include "ProjectIO.h"
#include "ProjectSupportUtils.h"
#include "Camera.h"
#include "ImageViewWidget.h"
#include "MatchLineOverlay.h"
#include "DualImageViewer.h"
#include "Logger.h"

#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHeaderView>
#include <QTabWidget>
#include <QHBoxLayout>
#include <QShortcut>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

QString normalizePath(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath());
}

bool pathTokenMatchesImage(const QString &token, const QString &imagePath)
{
    if (token.isEmpty() || imagePath.isEmpty()) return false;
    const QString a = normalizePath(token);
    const QString b = normalizePath(imagePath);
    if (a == b) return true;
    if (QFileInfo(token).fileName() == QFileInfo(imagePath).fileName()) return true;
    return QFileInfo(token).completeBaseName() == QFileInfo(imagePath).completeBaseName();
}

struct IntersectionBatchCandidate
{
    QVector<xjw::Intersection::Result> results;
    int validCount = 0;
    int finiteRmsCount = 0;
    double meanRms = std::numeric_limits<double>::infinity();
    bool camera1DepthFlipped = false;
    bool camera2DepthFlipped = false;
};

IntersectionBatchCandidate evaluateIntersectionBatch(const xjw::Camera &camera1,
                                                     const xjw::Camera &camera2,
                                                     const QVector<QPointF> &points1,
                                                     const QVector<QPointF> &points2)
{
    IntersectionBatchCandidate candidate;
    candidate.camera1DepthFlipped = camera1.depthAxisFlipped();
    candidate.camera2DepthFlipped = camera2.depthAxisFlipped();

    const int count = std::min(static_cast<int>(points1.size()), static_cast<int>(points2.size()));
    candidate.results.reserve(count);

    double rmsSum = 0.0;
    for (int index = 0; index < count; ++index)
    {
        const xjw::Intersection::Result result = xjw::Intersection::intersectPair(
            camera1,
            points1.at(index).x(),
            points1.at(index).y(),
            camera2,
            points2.at(index).x(),
            points2.at(index).y());
        candidate.results.push_back(result);

        if (result.valid)
        {
            ++candidate.validCount;
        }
        if (std::isfinite(result.reproj_error_rms))
        {
            rmsSum += result.reproj_error_rms;
            ++candidate.finiteRmsCount;
        }
    }

    if (candidate.finiteRmsCount > 0)
    {
        candidate.meanRms = rmsSum / static_cast<double>(candidate.finiteRmsCount);
    }

    return candidate;
}

IntersectionBatchCandidate selectBestIntersectionBatch(const xjw::Camera &baseCamera1,
                                                       const xjw::Camera &baseCamera2,
                                                       const QVector<QPointF> &points1,
                                                       const QVector<QPointF> &points2)
{
    IntersectionBatchCandidate bestCandidate;
    bool hasBestCandidate = false;

    for (int flipMask = 0; flipMask < 4; ++flipMask)
    {
        xjw::Camera camera1 = baseCamera1;
        xjw::Camera camera2 = baseCamera2;
        if ((flipMask & 0x1) != 0)
        {
            camera1.setDepthAxisFlipped(!camera1.depthAxisFlipped());
        }
        if ((flipMask & 0x2) != 0)
        {
            camera2.setDepthAxisFlipped(!camera2.depthAxisFlipped());
        }

        IntersectionBatchCandidate candidate = evaluateIntersectionBatch(camera1, camera2, points1, points2);
        LOG_INFO(
            QStringLiteral("[前方交汇] 深度组合评估: cam1Flip=%1 cam2Flip=%2 valid=%3/%4 finiteRms=%5 meanRms=%6")
                .arg(candidate.camera1DepthFlipped ? QStringLiteral("true") : QStringLiteral("false"))
                .arg(candidate.camera2DepthFlipped ? QStringLiteral("true") : QStringLiteral("false"))
                .arg(candidate.validCount)
                .arg(candidate.results.size())
                .arg(candidate.finiteRmsCount)
                .arg(candidate.meanRms, 0, 'f', 6));

        if (!hasBestCandidate
            || candidate.validCount > bestCandidate.validCount
            || (candidate.validCount == bestCandidate.validCount
                && candidate.finiteRmsCount > bestCandidate.finiteRmsCount)
            || (candidate.validCount == bestCandidate.validCount
                && candidate.finiteRmsCount == bestCandidate.finiteRmsCount
                && candidate.meanRms < bestCandidate.meanRms))
        {
            bestCandidate = std::move(candidate);
            hasBestCandidate = true;
        }
    }

    return bestCandidate;
}

} // namespace

ForwardIntersectionCheckDialog::ForwardIntersectionCheckDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
{
    setWindowTitle(tr("前方交汇检测"));
    resize(1200, 820);
    setupUi();
    loadImagesWithCamera();
}

ForwardIntersectionCheckDialog::~ForwardIntersectionCheckDialog() = default;

void ForwardIntersectionCheckDialog::setupUi()
{
    Ui::ForwardIntersectionCheckDialog form;
    form.setupUi(this);

    m_image1Combo = form.m_image1Combo;
    m_image2Combo = form.m_image2Combo;
    m_pickModeCombo = form.m_pickModeCombo;
    m_viewer = form.m_viewer;
    m_hintLabel = form.m_hintLabel;
    m_deleteSelectedBtn = form.m_deleteSelectedBtn;
    m_clearManualBtn = form.m_clearManualBtn;
    m_runBtn = form.m_runBtn;
    m_pairTable = form.m_pairTable;
    m_resultTable = form.m_resultTable;
    m_tabWidget = form.m_tabWidget;

    m_pickModeCombo->clear();
    m_pickModeCombo->addItem(tr("连接点自动选点（全点）"), QStringLiteral("auto"));
    m_pickModeCombo->addItem(tr("手动选点（多点）"), QStringLiteral("manual"));

    m_pairTable->setColumnCount(5);
    m_pairTable->setHorizontalHeaderLabels({
        tr("序号"), tr("u1"), tr("v1"), tr("u2"), tr("v2")
    });
    m_pairTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_pairTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_pairTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pairTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_pairTable->horizontalHeader()->setStretchLastSection(true);

    m_resultTable->setColumnCount(10);
    m_resultTable->setHorizontalHeaderLabels({
        tr("序号"), tr("有效"), tr("X"), tr("Y"), tr("Z"), tr("交汇角(deg)"), tr("射线距离(m)"),
        tr("误差1(px)"), tr("误差2(px)"), tr("RMS(px)")
    });
    m_resultTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_resultTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_resultTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_resultTable->horizontalHeader()->setStretchLastSection(true);
    m_resultTable->horizontalHeader()->setSectionsClickable(true);
    m_resultTable->horizontalHeader()->setSortIndicatorShown(true);
    m_resultTable->setSortingEnabled(false); // 手动排序，避免 Qt 自动排序破坏 UserRole 映射

    connect(m_image1Combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ForwardIntersectionCheckDialog::onImageSelectionChanged);
    connect(m_image2Combo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ForwardIntersectionCheckDialog::onImageSelectionChanged);
    connect(m_pickModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int)
    {
        const bool manual = (m_pickModeCombo->currentData().toString() == QStringLiteral("manual"));
        m_deleteSelectedBtn->setEnabled(manual);
        m_clearManualBtn->setEnabled(manual);
        m_hintLabel->setText(manual
            ? tr("手动模式：右键依次在左右图像选点完成配对。")
            : tr("自动模式：将读取匹配结果中的全部连接点进行批量交汇检验。"));
        if (!manual)
        {
            m_pendingFirstSide = -1;
        }
        applyPendingPointHint();
        refreshViewer(false);
    });
    connect(m_deleteSelectedBtn, &QPushButton::clicked, this, &ForwardIntersectionCheckDialog::onDeleteSelectedPairs);
    connect(m_clearManualBtn, &QPushButton::clicked, this, &ForwardIntersectionCheckDialog::onClearManualPoints);
    connect(m_runBtn, &QPushButton::clicked, this, &ForwardIntersectionCheckDialog::onRunCheck);
    // 右键配对：监听左右视图的右键点击
    if (m_viewer->leftView()) {
        connect(m_viewer->leftView(), &ImageViewWidget::viewRightClicked,
                this, &ForwardIntersectionCheckDialog::onViewerLeftRightClicked);
        connect(m_viewer->leftView(), &ImageViewWidget::matchPointClicked,
            this, [this](int index, const QPointF &)
            {
                onViewerPointClicked(index);
            });
    }
    if (m_viewer->rightView()) {
        connect(m_viewer->rightView(), &ImageViewWidget::viewRightClicked,
                this, &ForwardIntersectionCheckDialog::onViewerRightRightClicked);
        connect(m_viewer->rightView(), &ImageViewWidget::matchPointClicked,
                this, [this](int index, const QPointF &)
                {
                    onViewerPointClicked(index);
                });
    }
    connect(m_pairTable, &QTableWidget::cellClicked,
        this, &ForwardIntersectionCheckDialog::onPairTableClicked);
    connect(m_resultTable, &QTableWidget::cellClicked,
        this, &ForwardIntersectionCheckDialog::onResultTableClicked);
    connect(m_resultTable->horizontalHeader(), &QHeaderView::sectionClicked,
        this, &ForwardIntersectionCheckDialog::onResultTableHeaderClicked);

    // 绑定 Delete 键用于删除所选配对（当表格有焦点时生效）
    QShortcut *delShortcut = new QShortcut(QKeySequence::Delete, m_pairTable);
    connect(delShortcut, &QShortcut::activated, this, &ForwardIntersectionCheckDialog::onDeleteSelectedPairs);
}

void ForwardIntersectionCheckDialog::loadImagesWithCamera()
{
    m_image1Combo->clear();
    m_image2Combo->clear();

    if (!m_projectManager) return;
    QJsonObject meta = m_projectManager->currentMeta();
    if (meta.value(QStringLiteral("project_files")).isObject()) {
        meta = meta.value(QStringLiteral("project_files")).toObject();
    }

    const QJsonArray images = meta.value(QStringLiteral("images")).toArray();
    for (const QJsonValue &v : images) {
        const QJsonObject obj = v.toObject();
        const QString path = obj.value(QStringLiteral("path")).toString();
        const QJsonObject cam = obj.value(QStringLiteral("camera")).toObject();
        if (path.isEmpty() || cam.isEmpty()) continue;
        const QString name = QFileInfo(path).fileName().isEmpty() ? path : QFileInfo(path).fileName();
        m_image1Combo->addItem(name, path);
        m_image2Combo->addItem(name, path);
    }

    if (m_image1Combo->count() > 1) m_image2Combo->setCurrentIndex(1);
    onImageSelectionChanged();
}

// ── 辅助：从 sidecar JSON 文件中提取两张影像的匹配点 ─────────────────────────
static bool loadSidecarMatchPoints(const QString &sidecarPath,
                                    const QString &img1, const QString &img2,
                                    QVector<QPointF> *pts1, QVector<QPointF> *pts2)
{
    if (!QFile::exists(sidecarPath)) return false;
    QFile f(sidecarPath);
    if (!f.open(QIODevice::ReadOnly)) return false;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    f.close();
    if (!doc.isObject()) return false;

    const QJsonObject sidecar = doc.object();
    const QJsonArray p0 = sidecar.value(QStringLiteral("matched_points0")).toArray();
    const QJsonArray p1 = sidecar.value(QStringLiteral("matched_points1")).toArray();
    if (p0.isEmpty() || p0.size() != p1.size()) return false;

    const QString im0 = sidecar.value(QStringLiteral("image0_path")).toString();
    const QString im1 = sidecar.value(QStringLiteral("image1_path")).toString();
    const bool direct  = pathTokenMatchesImage(im0, img1) && pathTokenMatchesImage(im1, img2);
    const bool reverse = pathTokenMatchesImage(im0, img2) && pathTokenMatchesImage(im1, img1);
    if (!direct && !reverse) return false;

    pts1->clear();
    pts2->clear();
    for (int i = 0; i < p0.size(); ++i) {
        const QJsonArray a0 = p0.at(i).toArray();
        const QJsonArray a1 = p1.at(i).toArray();
        if (a0.size() < 2 || a1.size() < 2) continue;
        if (direct) {
            pts1->append(QPointF(a0.at(0).toDouble(), a0.at(1).toDouble()));
            pts2->append(QPointF(a1.at(0).toDouble(), a1.at(1).toDouble()));
        } else {
            pts1->append(QPointF(a1.at(0).toDouble(), a1.at(1).toDouble()));
            pts2->append(QPointF(a0.at(0).toDouble(), a0.at(1).toDouble()));
        }
    }
    return !pts1->isEmpty();
}

bool ForwardIntersectionCheckDialog::collectAutoPointPairs(QVector<QPointF> *pts1, QVector<QPointF> *pts2, QString *sourceInfo)
{
    if (!pts1 || !pts2) return false;
    pts1->clear();
    pts2->clear();
    if (!m_projectManager) return false;

    const QString img1 = selectedImage1();
    const QString img2 = selectedImage2();
    if (img1.isEmpty() || img2.isEmpty()) return false;

    // ── 方式一（优先）：直接扫描 assets/matches/*.match.json 文件系统 ────────
    // 不依赖惰性加载的 ipmatch_results，与 MatchPairSelectorDialog 保持一致
    const QString projectPath = m_projectManager->currentProjectPath();
    if (!projectPath.isEmpty()) {
        const QString assetsDir = ProjectIO::projectAssetsDir(projectPath);
        const QString matchDirPath = QDir(assetsDir).filePath(QStringLiteral("matches"));
        QDir matchDir(matchDirPath);
        if (matchDir.exists()) {
            const QStringList jsonFiles = matchDir.entryList(
                QStringList{QStringLiteral("*.match.json")}, QDir::Files, QDir::Time);
            for (const QString &jf : jsonFiles) {
                const QString fullPath = matchDir.filePath(jf);
                if (loadSidecarMatchPoints(fullPath, img1, img2, pts1, pts2)) {
                    if (sourceInfo) *sourceInfo = jf;
                    return true;
                }
            }
        }
    }

    // ── 方式二（回退）：读取 project_results 中的 ipmatch_results ─────────────
    QJsonObject meta = m_projectManager->currentMeta();
    if (meta.value(QStringLiteral("project_files")).isObject()) {
        meta = meta.value(QStringLiteral("project_files")).toObject();
    }

    const QJsonArray matchResults = meta.value(QStringLiteral("ipmatch_results")).toArray();
    for (const QJsonValue &v : matchResults) {
        if (!v.isObject()) continue;
        const QJsonObject rec = v.toObject();
        const QJsonObject settings = rec.value(QStringLiteral("settings")).toObject();
        const QJsonArray imageFiles = settings.value(QStringLiteral("image_files")).toArray();

        bool has1 = false;
        bool has2 = false;
        for (const QJsonValue &it : imageFiles) {
            const QString token = it.toString();
            has1 = has1 || pathTokenMatchesImage(token, img1);
            has2 = has2 || pathTokenMatchesImage(token, img2);
        }
        if (!(has1 && has2)) continue;

        QString sidecarPath = settings.value(QStringLiteral("sidecar_json")).toString();
        if (sidecarPath.isEmpty()) {
            sidecarPath = rec.value(QStringLiteral("output")).toString() + QStringLiteral(".json");
        }
        if (loadSidecarMatchPoints(sidecarPath, img1, img2, pts1, pts2)) {
            if (sourceInfo) *sourceInfo = QFileInfo(sidecarPath).fileName();
            return true;
        }
    }

    return false;
}

bool ForwardIntersectionCheckDialog::buildCameraFromImageMeta(const QJsonObject &imgObj, xjw::Camera *cam, QString *errorMsg) const
{
    if (!cam) return false;
    const QJsonObject camObj = imgObj.value(QStringLiteral("camera")).toObject();
    if (camObj.isEmpty()) {
        if (errorMsg) *errorMsg = tr("影像缺少相机参数");
        return false;
    }

    if (!xjw::gui::project::cameraFromJson(camObj, cam)) {
        if (errorMsg) {
            *errorMsg = tr("相机参数解析失败（请检查单位字段、C/R 或 pitch）");
        }
        return false;
    }

    const auto intrinsics = cam->intrinsics();
    const auto center = cam->cameraCenter();
    LOG_INFO(
        QStringLiteral("[前方交汇] 相机解析: image=%1 fu_px=%2 fv_px=%3 cu_px=%4 cv_px=%5 pitch=%6 depthFlip=%7 depthFieldPresent=%8 C=(%9,%10,%11)")
            .arg(QFileInfo(imgObj.value(QStringLiteral("path")).toString()).fileName())
            .arg(intrinsics.focalX, 0, 'f', 6)
            .arg(intrinsics.focalY, 0, 'f', 6)
            .arg(intrinsics.principalX, 0, 'f', 6)
            .arg(intrinsics.principalY, 0, 'f', 6)
            .arg(intrinsics.pixelPitch, 0, 'f', 9)
            .arg(cam->depthAxisFlipped() ? QStringLiteral("true") : QStringLiteral("false"))
            .arg(camObj.contains(QStringLiteral("depth_axis_flipped")) ? QStringLiteral("true") : QStringLiteral("false"))
            .arg(center[0], 0, 'f', 6)
            .arg(center[1], 0, 'f', 6)
            .arg(center[2], 0, 'f', 6));
    return true;
}

QJsonObject ForwardIntersectionCheckDialog::findImageMetaByPath(const QString &imagePath) const
{
    if (!m_projectManager) return QJsonObject();
    QJsonObject meta = m_projectManager->currentMeta();
    if (meta.value(QStringLiteral("project_files")).isObject()) {
        meta = meta.value(QStringLiteral("project_files")).toObject();
    }
    const QString target = normalizePath(imagePath);
    for (const QJsonValue &v : meta.value(QStringLiteral("images")).toArray()) {
        const QJsonObject obj = v.toObject();
        if (normalizePath(obj.value(QStringLiteral("path")).toString()) == target) return obj;
    }
    return QJsonObject();
}

void ForwardIntersectionCheckDialog::refreshViewer(bool reloadImages)
{
    if (!m_viewer) return;
    const QString img1 = selectedImage1();
    const QString img2 = selectedImage2();
    if (img1.isEmpty() || img2.isEmpty()) return;

    if (reloadImages || normalizePath(m_viewer->leftImagePath()) != normalizePath(img1)
        || normalizePath(m_viewer->rightImagePath()) != normalizePath(img2)) {
        m_viewer->loadMatchPair(img1, img2, QVector<QPointF>{}, QVector<QPointF>{});
    }

    QVector<QPointF> display1 = m_currentPts1;
    QVector<QPointF> display2 = m_currentPts2;
    const bool manual = (m_pickModeCombo->currentData().toString() == QStringLiteral("manual"));
    if (manual && m_pendingFirstSide == 0) {
        display1.append(m_pendingFirstPoint);
    } else if (manual && m_pendingFirstSide == 1) {
        display2.append(m_pendingFirstPoint);
    }

    if (m_viewer->leftView()) m_viewer->leftView()->setMatchPoints(display1);
    if (m_viewer->rightView()) m_viewer->rightView()->setMatchPoints(display2);
    if (m_viewer->overlay()) m_viewer->overlay()->setMatches(m_currentPts1, m_currentPts2);

    // 默认不显示全部连线，仅在表格点击后高亮显示
    m_viewer->setShowAllMatches(false);
}

void ForwardIntersectionCheckDialog::refreshPairTable()
{
    m_pairTable->setRowCount(0);
    const int n = std::min(static_cast<int>(m_currentPts1.size()), static_cast<int>(m_currentPts2.size()));
    m_pairTable->setRowCount(n);

    for (int i = 0; i < n; ++i) {
        m_pairTable->setItem(i, 0, new QTableWidgetItem(QString::number(i + 1)));
        m_pairTable->setItem(i, 1, new QTableWidgetItem(QString::number(m_currentPts1.at(i).x(), 'f', 6)));
        m_pairTable->setItem(i, 2, new QTableWidgetItem(QString::number(m_currentPts1.at(i).y(), 'f', 6)));
        m_pairTable->setItem(i, 3, new QTableWidgetItem(QString::number(m_currentPts2.at(i).x(), 'f', 6)));
        m_pairTable->setItem(i, 4, new QTableWidgetItem(QString::number(m_currentPts2.at(i).y(), 'f', 6)));
    }
}

void ForwardIntersectionCheckDialog::fillResultTable(const QVector<xjw::Intersection::Result> &results)
{
    // 重新填充时重置排序状态
    m_resultSortCol = -1;
    m_resultTable->horizontalHeader()->setSortIndicator(-1, Qt::DescendingOrder);

    QVector<int> order(results.size());
    for (int i = 0; i < order.size(); ++i) order[i] = i;
    fillResultTableOrdered(order);
}

void ForwardIntersectionCheckDialog::fillResultTableOrdered(const QVector<int> &order)
{
    m_resultTable->setRowCount(0);
    m_resultTable->setRowCount(order.size());

    for (int row = 0; row < order.size(); ++row) {
        const int origIdx = order[row];
        const auto &r = m_currentResults.at(origIdx);

        auto *item0 = new QTableWidgetItem(QString::number(origIdx + 1));
        item0->setData(Qt::UserRole, origIdx); // 存储原始索引，排序后仍能找到对应点
        m_resultTable->setItem(row, 0, item0);
        m_resultTable->setItem(row, 1, new QTableWidgetItem(r.valid ? tr("是") : tr("否")));
        m_resultTable->setItem(row, 2, new QTableWidgetItem(QString::number(r.point[0], 'f', 6)));
        m_resultTable->setItem(row, 3, new QTableWidgetItem(QString::number(r.point[1], 'f', 6)));
        m_resultTable->setItem(row, 4, new QTableWidgetItem(QString::number(r.point[2], 'f', 6)));
        m_resultTable->setItem(row, 5, new QTableWidgetItem(QString::number(r.angle_deg, 'f', 6)));
        m_resultTable->setItem(row, 6, new QTableWidgetItem(QString::number(r.ray_miss_distance, 'f', 6)));
        m_resultTable->setItem(row, 7, new QTableWidgetItem(QString::number(r.reproj_error_cam1, 'f', 6)));
        m_resultTable->setItem(row, 8, new QTableWidgetItem(QString::number(r.reproj_error_cam2, 'f', 6)));
        m_resultTable->setItem(row, 9, new QTableWidgetItem(QString::number(r.reproj_error_rms, 'f', 6)));
    }
}

void ForwardIntersectionCheckDialog::applyPendingPointHint()
{
    const bool manual = (m_pickModeCombo->currentData().toString() == QStringLiteral("manual"));
    if (!manual) {
        m_hintLabel->setText(tr("自动模式：将读取匹配结果中的全部连接点进行批量交汇检验。"));
        return;
    }

    if (m_pendingFirstSide == 0) {
        m_hintLabel->setText(tr("已在左侧选择点；请在右侧右键选择配对点"));
    } else if (m_pendingFirstSide == 1) {
        m_hintLabel->setText(tr("已在右侧选择点；请在左侧右键选择配对点"));
    } else {
        m_hintLabel->setText(tr("手动模式：右键依次在左右图像选点完成配对。"));
    }
}

void ForwardIntersectionCheckDialog::clearAllSelections()
{
    m_currentHighlighted = -1;
    if (m_pairTable) m_pairTable->clearSelection();
    if (m_resultTable) m_resultTable->clearSelection();
    if (m_viewer) m_viewer->clearMatchHighlights();
    if (m_viewer && m_viewer->leftView()) m_viewer->leftView()->clearHighlight();
    if (m_viewer && m_viewer->rightView()) m_viewer->rightView()->clearHighlight();
}

QJsonObject ForwardIntersectionCheckDialog::buildBatchResultJson(const QVector<QPointF> &pts1,
                                                                 const QVector<QPointF> &pts2,
                                                                 const QVector<xjw::Intersection::Result> &results,
                                                                 const QString &mode,
                                                                 const QString &autoSource) const
{
    QJsonObject obj;
    obj[QStringLiteral("created_at")] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
    obj[QStringLiteral("image0_path")] = normalizePath(selectedImage1());
    obj[QStringLiteral("image1_path")] = normalizePath(selectedImage2());
    obj[QStringLiteral("image0_name")] = QFileInfo(selectedImage1()).fileName();
    obj[QStringLiteral("image1_name")] = QFileInfo(selectedImage2()).fileName();
    obj[QStringLiteral("pick_mode")] = mode;
    if (!autoSource.isEmpty()) obj[QStringLiteral("auto_source")] = autoSource;

    QJsonArray points;
    int validCount = 0;
    double rmsSum = 0.0;
    int rmsCount = 0;

    const int n = std::min({static_cast<int>(pts1.size()),
                            static_cast<int>(pts2.size()),
                            static_cast<int>(results.size())});
    for (int i = 0; i < n; ++i) {
        const auto &r = results.at(i);
        QJsonObject one;
        one[QStringLiteral("index")] = i;
        QJsonArray p0; p0.append(pts1.at(i).x()); p0.append(pts1.at(i).y());
        QJsonArray p1; p1.append(pts2.at(i).x()); p1.append(pts2.at(i).y());
        one[QStringLiteral("point0_uv")] = p0;
        one[QStringLiteral("point1_uv")] = p1;

        QJsonObject m;
        m[QStringLiteral("valid")] = r.valid;
        m[QStringLiteral("X")] = r.point[0];
        m[QStringLiteral("Y")] = r.point[1];
        m[QStringLiteral("Z")] = r.point[2];
        m[QStringLiteral("angle_deg")] = r.angle_deg;
        m[QStringLiteral("ray_miss_distance")] = r.ray_miss_distance;
        m[QStringLiteral("reproj_error_cam1")] = r.reproj_error_cam1;
        m[QStringLiteral("reproj_error_cam2")] = r.reproj_error_cam2;
        m[QStringLiteral("reproj_error_rms")] = r.reproj_error_rms;
        one[QStringLiteral("metrics")] = m;

        if (r.valid) ++validCount;
        if (std::isfinite(r.reproj_error_rms)) {
            rmsSum += r.reproj_error_rms;
            ++rmsCount;
        }
        points.append(one);
    }

    QJsonObject summary;
    summary[QStringLiteral("total_points")] = n;
    summary[QStringLiteral("valid_points")] = validCount;
    summary[QStringLiteral("valid_ratio")] = (n > 0 ? static_cast<double>(validCount) / n : 0.0);
    summary[QStringLiteral("mean_rms")] = (rmsCount > 0 ? rmsSum / rmsCount : 0.0);

    obj[QStringLiteral("points")] = points;
    obj[QStringLiteral("summary")] = summary;

    if (n > 0) {
        const auto &r0 = results.first();
        QJsonObject metrics;
        metrics[QStringLiteral("valid")] = r0.valid;
        metrics[QStringLiteral("X")] = r0.point[0];
        metrics[QStringLiteral("Y")] = r0.point[1];
        metrics[QStringLiteral("Z")] = r0.point[2];
        metrics[QStringLiteral("angle_deg")] = r0.angle_deg;
        metrics[QStringLiteral("ray_miss_distance")] = r0.ray_miss_distance;
        metrics[QStringLiteral("reproj_error_cam1")] = r0.reproj_error_cam1;
        metrics[QStringLiteral("reproj_error_cam2")] = r0.reproj_error_cam2;
        metrics[QStringLiteral("reproj_error_rms")] = r0.reproj_error_rms;
        obj[QStringLiteral("metrics")] = metrics;
    }

    return obj;
}

QString ForwardIntersectionCheckDialog::selectedImage1() const
{
    return m_image1Combo->currentData().toString();
}

QString ForwardIntersectionCheckDialog::selectedImage2() const
{
    return m_image2Combo->currentData().toString();
}

void ForwardIntersectionCheckDialog::onImageSelectionChanged()
{
    m_manualPts1.clear();
    m_manualPts2.clear();
    m_currentPts1.clear();
    m_currentPts2.clear();
    m_currentResults.clear();
    m_pendingFirstSide = -1;
    m_currentPairsEditable = false;
    clearAllSelections();
    refreshPairTable();
    m_resultTable->setRowCount(0);
    refreshViewer(true);
    applyPendingPointHint();
}

void ForwardIntersectionCheckDialog::onClearManualPoints()
{
    m_manualPts1.clear();
    m_manualPts2.clear();
    m_currentPts1 = m_manualPts1;
    m_currentPts2 = m_manualPts2;
    m_currentResults.clear();
    m_pendingFirstSide = -1;
    m_currentPairsEditable = true;
    clearAllSelections();
    refreshPairTable();
    refreshViewer(false);
    applyPendingPointHint();
    m_resultTable->setRowCount(0);
}

void ForwardIntersectionCheckDialog::onViewerLeftRightClicked(const QPointF &scenePos)
{
    // Right-click on left view
    if (m_pickModeCombo->currentData().toString() != QStringLiteral("manual")) return;

    if (m_pendingFirstSide == -1) {
        // start pairing from left
        m_pendingFirstSide = 0;
        m_pendingFirstPoint = scenePos;
        applyPendingPointHint();
        refreshViewer(false);
        return;
    }

    if (m_pendingFirstSide == 0) {
        // replace pending left point
        m_pendingFirstPoint = scenePos;
        applyPendingPointHint();
        refreshViewer(false);
        return;
    }

    if (m_pendingFirstSide == 1) {
        // previously had right first, now left completes pair
        m_manualPts1.append(scenePos);
        m_manualPts2.append(m_pendingFirstPoint);
        m_pendingFirstSide = -1;
        m_currentPts1 = m_manualPts1;
        m_currentPts2 = m_manualPts2;
        m_currentResults.clear();
        m_currentPairsEditable = true;
        clearAllSelections();
        refreshPairTable();
        refreshViewer(false);
        // 选中并高亮刚添加的配对
        const int newRow = m_currentPts1.size() - 1;
        if (newRow >= 0) {
            m_pairTable->selectRow(newRow);
            onPairTableClicked(newRow, 0);
        }
        applyPendingPointHint();
    }
}

void ForwardIntersectionCheckDialog::onViewerRightRightClicked(const QPointF &scenePos)
{
    // Right-click on right view
    if (m_pickModeCombo->currentData().toString() != QStringLiteral("manual")) return;

    if (m_pendingFirstSide == -1) {
        // start pairing from right
        m_pendingFirstSide = 1;
        m_pendingFirstPoint = scenePos;
        applyPendingPointHint();
        refreshViewer(false);
        return;
    }

    if (m_pendingFirstSide == 1) {
        // replace pending right point
        m_pendingFirstPoint = scenePos;
        applyPendingPointHint();
        refreshViewer(false);
        return;
    }

    if (m_pendingFirstSide == 0) {
        // previously had left first, now right completes pair
        m_manualPts1.append(m_pendingFirstPoint);
        m_manualPts2.append(scenePos);
        m_pendingFirstSide = -1;
        m_currentPts1 = m_manualPts1;
        m_currentPts2 = m_manualPts2;
        m_currentResults.clear();
        m_currentPairsEditable = true;
        clearAllSelections();
        refreshPairTable();
        refreshViewer(false);
        // 选中并高亮刚添加的配对
        const int newRow2 = m_currentPts1.size() - 1;
        if (newRow2 >= 0) {
            m_pairTable->selectRow(newRow2);
            onPairTableClicked(newRow2, 0);
        }
        applyPendingPointHint();
    }
}

void ForwardIntersectionCheckDialog::onPairTableClicked(int row, int column)
{
    Q_UNUSED(column);
    if (row < 0) {
        clearAllSelections();
        return;
    }

    if (row == m_currentHighlighted) {
        clearAllSelections();
        return;
    }

    // 清除结果表选中，避免双表同时高亮造成混乱
    if (m_resultTable) m_resultTable->clearSelection();
    m_viewer->highlightMatchIndex(row);
    // 同时高亮左右视图中的匹配点
    if (m_viewer->leftView()) m_viewer->leftView()->highlightPoint(row);
    if (m_viewer->rightView()) m_viewer->rightView()->highlightPoint(row);
    m_currentHighlighted = row;
}

void ForwardIntersectionCheckDialog::onResultTableClicked(int row, int column)
{
    Q_UNUSED(column);
    if (row < 0) {
        clearAllSelections();
        return;
    }

    // 获取该行对应的原始索引（排序后行号不等于原始索引）
    const auto *item0 = m_resultTable->item(row, 0);
    const int origIdx = item0 ? item0->data(Qt::UserRole).toInt() : row;

    if (origIdx == m_currentHighlighted) {
        clearAllSelections();
        return;
    }

    // 同步选中配对表对应行
    if (m_pairTable) {
        m_pairTable->clearSelection();
        if (origIdx < m_pairTable->rowCount()) {
            m_pairTable->selectRow(origIdx);
            m_pairTable->scrollTo(m_pairTable->model()->index(origIdx, 0),
                                  QAbstractItemView::PositionAtCenter);
        }
    }
    m_viewer->highlightMatchIndex(origIdx);
    if (m_viewer->leftView()) m_viewer->leftView()->highlightPoint(origIdx);
    if (m_viewer->rightView()) m_viewer->rightView()->highlightPoint(origIdx);
    m_currentHighlighted = origIdx;
}

void ForwardIntersectionCheckDialog::onViewerPointClicked(int index)
{
    if (index < 0) {
        clearAllSelections();
        return;
    }

    // 再次点击同一点则取消高亮
    if (index == m_currentHighlighted) {
        clearAllSelections();
        return;
    }

    // 高亮对应连线和左右视图中的匹配点
    m_viewer->highlightMatchIndex(index);
    if (m_viewer->leftView()) m_viewer->leftView()->highlightPoint(index);
    if (m_viewer->rightView()) m_viewer->rightView()->highlightPoint(index);
    m_currentHighlighted = index;

    // 同步选中并滚动配对点表
    if (m_pairTable && index < m_pairTable->rowCount()) {
        m_pairTable->clearSelection();
        m_pairTable->selectRow(index);
        m_pairTable->scrollTo(m_pairTable->model()->index(index, 0),
                              QAbstractItemView::PositionAtCenter);
    }

    // 若前方交汇结果表有数据，按 UserRole 查找对应行并跳转
    if (m_resultTable && m_resultTable->rowCount() > 0) {
        m_resultTable->clearSelection();
        for (int row = 0; row < m_resultTable->rowCount(); ++row) {
            const auto *item0 = m_resultTable->item(row, 0);
            if (item0 && item0->data(Qt::UserRole).toInt() == index) {
                m_resultTable->selectRow(row);
                m_resultTable->scrollTo(m_resultTable->model()->index(row, 0),
                                        QAbstractItemView::PositionAtCenter);
                break;
            }
        }
        if (m_tabWidget) m_tabWidget->setCurrentIndex(1);
    } else {
        if (m_tabWidget) m_tabWidget->setCurrentIndex(0);
    }
}

void ForwardIntersectionCheckDialog::onResultTableHeaderClicked(int col)
{
    // 仅允许对指定列排序：1=有效, 5=交汇角, 9=RMS
    if (col != 1 && col != 5 && col != 9) return;
    if (m_currentResults.isEmpty())
    {
        return;
    }

    if (m_resultSortCol == col)
    {
        m_resultSortOrder = (m_resultSortOrder == Qt::DescendingOrder)
                          ? Qt::AscendingOrder : Qt::DescendingOrder;
    }
    else
    {
        m_resultSortCol = col;
        m_resultSortOrder = Qt::DescendingOrder; // 默认大到小
    }

    QVector<int> order(m_currentResults.size());
    for (int i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }

    std::stable_sort(order.begin(), order.end(), [&](int a, int b)
    {
        const auto &ra = m_currentResults.at(a);
        const auto &rb = m_currentResults.at(b);
        bool aGreater;
        if (col == 1)
        {
            if (ra.valid == rb.valid)
            {
                return false;
            }
            aGreater = ra.valid && !rb.valid; // 是 > 否
        }
        else if (col == 5)
        {
            if (ra.angle_deg == rb.angle_deg)
            {
                return false;
            }
            aGreater = ra.angle_deg > rb.angle_deg;
        }
        else
        { // col == 9, RMS
            if (ra.reproj_error_rms == rb.reproj_error_rms)
            {
                return false;
            }
            aGreater = ra.reproj_error_rms > rb.reproj_error_rms;
        }
        return (m_resultSortOrder == Qt::DescendingOrder) ? aGreater : !aGreater;
    });

    clearAllSelections();
    fillResultTableOrdered(order);
    m_resultTable->horizontalHeader()->setSortIndicator(col, m_resultSortOrder);
}

void ForwardIntersectionCheckDialog::onDeleteSelectedPairs()
{
    if (!m_currentPairsEditable || !m_pairTable)
    {
        return;
    }
    const QModelIndexList rows = m_pairTable->selectionModel()->selectedRows();
    if (rows.isEmpty())
    {
        return;
    }

    QVector<int> indices;
    indices.reserve(rows.size());
    for (const QModelIndex &idx : rows)
    {
        indices.append(idx.row());
    }
    std::sort(indices.begin(), indices.end(), std::greater<int>());

    for (int row : indices)
    {
        if (row >= 0 && row < m_manualPts1.size() && row < m_manualPts2.size())
        {
            m_manualPts1.removeAt(row);
            m_manualPts2.removeAt(row);
        }
    }

    m_currentPts1 = m_manualPts1;
    m_currentPts2 = m_manualPts2;
    m_currentResults.clear();
    m_resultTable->setRowCount(0);
    clearAllSelections();
    refreshPairTable();
    refreshViewer(false);
}

void ForwardIntersectionCheckDialog::onRunCheck()
{
    if (!m_projectManager) return;

    const QString img1 = selectedImage1();
    const QString img2 = selectedImage2();
    if (img1.isEmpty() || img2.isEmpty() || normalizePath(img1) == normalizePath(img2)) {
        QMessageBox::warning(this, tr("提示"), tr("请选择两张不同的影像"));
        return;
    }

    const QJsonObject imgObj1 = findImageMetaByPath(img1);
    const QJsonObject imgObj2 = findImageMetaByPath(img2);
    if (imgObj1.isEmpty() || imgObj2.isEmpty()) {
        QMessageBox::warning(this, tr("提示"), tr("无法读取影像元数据"));
        return;
    }

    xjw::Camera cam1;
    xjw::Camera cam2;
    QString err;
    if (!buildCameraFromImageMeta(imgObj1, &cam1, &err) || !buildCameraFromImageMeta(imgObj2, &cam2, &err)) {
        QMessageBox::critical(this, tr("错误"), err);
        return;
    }

    const QString mode = m_pickModeCombo->currentData().toString();
    QVector<QPointF> pts1;
    QVector<QPointF> pts2;
    QString autoSource;

    if (mode == QStringLiteral("manual")) {
        pts1 = m_manualPts1;
        pts2 = m_manualPts2;
        if (pts1.isEmpty() || pts2.isEmpty() || pts1.size() != pts2.size()) {
            QMessageBox::warning(this, tr("提示"), tr("请先添加至少一组手动点对"));
            return;
        }
    } else {
        if (!collectAutoPointPairs(&pts1, &pts2, &autoSource)) {
            QMessageBox::warning(this, tr("提示"), tr("未找到可用连接点结果，无法自动选点"));
            return;
        }
    }

    m_currentPts1 = pts1;
    m_currentPts2 = pts2;
    m_currentPairsEditable = (mode == QStringLiteral("manual"));
    clearAllSelections();
    refreshPairTable();

    const IntersectionBatchCandidate bestCandidate = selectBestIntersectionBatch(cam1, cam2, pts1, pts2);
    LOG_INFO(
        QStringLiteral("[前方交汇] 采用深度组合: cam1Flip=%1 cam2Flip=%2 valid=%3/%4 meanRms=%5")
            .arg(bestCandidate.camera1DepthFlipped ? QStringLiteral("true") : QStringLiteral("false"))
            .arg(bestCandidate.camera2DepthFlipped ? QStringLiteral("true") : QStringLiteral("false"))
            .arg(bestCandidate.validCount)
            .arg(bestCandidate.results.size())
            .arg(bestCandidate.meanRms, 0, 'f', 6));

    QVector<xjw::Intersection::Result> results = bestCandidate.results;

    m_currentResults = results;
    refreshViewer(false);
    fillResultTable(results);

    QJsonObject saveObj = buildBatchResultJson(pts1, pts2, results, mode, autoSource);
    QString saveErr;
    if (!m_projectManager->appendIntersectionResult(saveObj, &saveErr)) {
        QMessageBox::warning(this, tr("提示"), tr("结果计算完成，但保存失败: %1").arg(saveErr));
    }
}
