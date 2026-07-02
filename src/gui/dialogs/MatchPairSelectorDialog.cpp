// =============================================================================
// 文件: MatchPairSelectorDialog.cpp
// 说明: MatchPairSelectorDialog 的实现。
//       直接扫描 assets/matches/*.match 文件系统，无需加载大 JSON 元数据，
//       支持在匹配处理过程中实时刷新（通过 projectMetadataChanged 信号触发）。
// =============================================================================
#include "MatchPairSelectorDialog.h"
#include "MatchViewerDialog.h"
#include "MatchResultCatalog.h"
#include "ProjectManager.h"
#include "ProjectIO.h"
#include "Logger.h"
#include "ui_MatchPairSelectorDialog.h"

#include <QComboBox>
#include <QTableWidget>
#include <QPushButton>
#include <QLabel>
#include <QHeaderView>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonValue>
#include <QMessageBox>
#include <QFile>
#include <QDataStream>
#include <QTimer>
#include <QSet>
#include <QRegularExpression>

#include <algorithm>

namespace {

QString normalizedImagePathKey(const QString &path)
{
    QString key = QDir::cleanPath(path.trimmed());
    key.replace(QLatin1Char('\\'), QLatin1Char('/'));
    return key.toLower();
}

QString imageBaseKey(const QString &path)
{
    const QString base = QFileInfo(path.trimmed()).completeBaseName();
    return (base.isEmpty() ? path.trimmed() : base).toLower();
}

bool imageTokenMatches(const QString &candidate, const QString &imagePath)
{
    if (candidate.trimmed().isEmpty() || imagePath.trimmed().isEmpty())
    {
        return false;
    }

    return normalizedImagePathKey(candidate) == normalizedImagePathKey(imagePath) ||
           imageBaseKey(candidate) == imageBaseKey(imagePath);
}

QString canonicalPairKeyForImages(const QString &imageA, const QString &imageB)
{
    QString keyA = normalizedImagePathKey(imageA);
    QString keyB = normalizedImagePathKey(imageB);
    if (keyA > keyB)
    {
        std::swap(keyA, keyB);
    }
    return keyA + QStringLiteral("|") + keyB;
}

QString resolveProjectImagePath(const QString &token,
                                const QStringList &projectImages,
                                const QMap<QString, QString> &baseToPath)
{
    const QString normalizedToken = normalizedImagePathKey(token);
    for (const QString &imagePath : projectImages)
    {
        if (normalizedImagePathKey(imagePath) == normalizedToken)
        {
            return imagePath;
        }
    }

    const QString base = imageBaseKey(token);
    if (baseToPath.contains(base))
    {
        return baseToPath.value(base);
    }

    return token;
}

bool pairContainsImage(const QString &imageA,
                       const QString &imageB,
                       const QString &imagePath,
                       QString *otherImageToken)
{
    if (imageTokenMatches(imageA, imagePath))
    {
        if (otherImageToken)
        {
            *otherImageToken = imageB;
        }
        return true;
    }
    if (imageTokenMatches(imageB, imagePath))
    {
        if (otherImageToken)
        {
            *otherImageToken = imageA;
        }
        return true;
    }
    return false;
}

bool groupContainsImage(const xjw::pipeline::MatchPairGroup &group,
                        const QString &imagePath,
                        QString *otherImageToken)
{
    if (pairContainsImage(group.imageA, group.imageB, imagePath, otherImageToken))
    {
        return true;
    }

    for (const xjw::pipeline::MatchVariant &variant : group.variants)
    {
        if (pairContainsImage(variant.imageA, variant.imageB, imagePath, otherImageToken))
        {
            return true;
        }
    }
    return false;
}

QString matchVariantAlgorithmLabel(const xjw::pipeline::MatchVariant &variant)
{
    const QString label = xjw::pipeline::MatchResultCatalog::algorithmDisplayLabel(variant);
    return label == QStringLiteral("unknown") ? QStringLiteral("(未知算法)") : label;
}

QString matchVariantReasonLabel(const xjw::pipeline::MatchVariant &variant)
{
    if (variant.compatible)
    {
        return QStringLiteral("可查看");
    }
    if (variant.status == QStringLiteral("missing_sidecar"))
    {
        return QStringLiteral("缺少 sidecar");
    }
    if (variant.status == QStringLiteral("invalid_sidecar"))
    {
        return QStringLiteral("sidecar 无效");
    }
    if (variant.status == QStringLiteral("invalid_match_file"))
    {
        return QStringLiteral("匹配文件无效");
    }
    if (variant.status == QStringLiteral("mismatched_image_names"))
    {
        return QStringLiteral("影像名不一致");
    }
    if (variant.status == QStringLiteral("missing_image_names"))
    {
        return QStringLiteral("缺少影像名");
    }
    return variant.reason.isEmpty() ? variant.status : variant.reason;
}

int bestDisplayVariantIndex(const xjw::pipeline::MatchPairGroup &group)
{
    if (group.bestVariantIndex >= 0 && group.bestVariantIndex < group.variants.size())
    {
        return group.bestVariantIndex;
    }
    for (int i = 0; i < group.variants.size(); ++i)
    {
        if (group.variants.at(i).compatible)
        {
            return i;
        }
    }
    return group.variants.isEmpty() ? -1 : 0;
}

int compatibleVariantCount(const QVector<xjw::pipeline::MatchVariant> &variants)
{
    int count = 0;
    for (const xjw::pipeline::MatchVariant &variant : variants)
    {
        if (variant.compatible && !variant.matchFilePath.trimmed().isEmpty())
        {
            ++count;
        }
    }
    return count;
}

QString availableAlgorithmText(const QVector<xjw::pipeline::MatchVariant> &variants)
{
    QStringList labels;
    for (const xjw::pipeline::MatchVariant &variant : variants)
    {
        if (!variant.compatible || variant.matchFilePath.trimmed().isEmpty())
        {
            continue;
        }

        const QString label = matchVariantAlgorithmLabel(variant);
        if (!labels.contains(label))
        {
            labels.append(label);
        }
    }

    return labels.isEmpty() ? QStringLiteral("无") : labels.join(QStringLiteral(", "));
}

QString catalogRowStatusText(const QVector<xjw::pipeline::MatchVariant> &variants,
                             int selectedVariantIndex,
                             int compatibleCount)
{
    if (compatibleCount > 1)
    {
        return QStringLiteral("可查看（%1 个算法）").arg(compatibleCount);
    }
    if (compatibleCount == 1)
    {
        return QStringLiteral("可查看");
    }
    if (selectedVariantIndex >= 0 && selectedVariantIndex < variants.size())
    {
        return QStringLiteral("不可用：%1").arg(matchVariantReasonLabel(variants.at(selectedVariantIndex)));
    }
    return QStringLiteral("不可用");
}

QString variantsTooltip(const QVector<xjw::pipeline::MatchVariant> &variants)
{
    QStringList lines;
    for (const xjw::pipeline::MatchVariant &variant : variants)
    {
        const QString counts = variant.hasInlierStats
            ? QStringLiteral("内点 %1 / 总 %2")
                  .arg(variant.geometricVerifiedInliers)
                  .arg(variant.totalMatches)
            : QStringLiteral("总 %1").arg(variant.totalMatches);
        lines.append(QStringLiteral("%1：%2，%3")
                         .arg(matchVariantAlgorithmLabel(variant),
                              matchVariantReasonLabel(variant),
                              counts));
    }
    return lines.join(QLatin1Char('\n'));
}

} // namespace

// 构造函数：初始化对话框，构建界面，加载项目影像列表
MatchPairSelectorDialog::MatchPairSelectorDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , _projectManager(projectManager)
    , _selectedMatchIndex(-1)   // -1 表示初始无选中行
{
    setWindowTitle(tr("匹配查看器"));
    resize(800, 600);

    // 获取 matches 目录（直接扫描，不依赖元数据）
    if (_projectManager && !_projectManager->currentProjectPath().isEmpty()) {
        const QString assetsDir = ProjectIO::projectAssetsDir(_projectManager->currentProjectPath());
        _matchDir = QDir(assetsDir).filePath(QStringLiteral("matches"));
    }

    setupUI();
    loadProjectImages();

    // ── 实时刷新：projectMetadataChanged / matchPairReady 时自动更新视图 ──
    // 使用防抖 QTimer，避免高频更新导致 UI 闪烁（300ms 内不再触发才真正刷新）
    _refreshTimer = new QTimer(this);
    _refreshTimer->setSingleShot(true);
    _refreshTimer->setInterval(300);
    connect(_refreshTimer, &QTimer::timeout, this, &MatchPairSelectorDialog::onRefresh);

    if (_projectManager) {
        connect(_projectManager, &ProjectManager::projectMetadataChanged,
                this, &MatchPairSelectorDialog::scheduleRefresh);
        connect(_projectManager, &ProjectManager::matchPairReady,
                this, [this](const QString &, const QString &, const QString &, int) {
                    scheduleRefresh();
                });
    }
}

MatchPairSelectorDialog::~MatchPairSelectorDialog()
{
}

// setupUI: 构建对话框的整体界面布局
// 包含：顶部图像选择区、中间匹配表格、底部状态栏和操作按钮
void MatchPairSelectorDialog::setupUI()
{
    Ui::MatchPairSelectorDialog ui;
    ui.setupUi(this);

    _imageComboBox = ui.m_imageComboBox;
    _matchTable = ui.m_matchTable;
    _viewDetailBtn = ui.m_viewDetailBtn;
    _refreshBtn = ui.m_refreshBtn;
    _statusLabel = ui.m_statusLabel;

    connect(_imageComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchPairSelectorDialog::onCurrentImageChanged);
    connect(_refreshBtn, &QPushButton::clicked, this, &MatchPairSelectorDialog::onRefresh);

    setupTable();

    connect(_viewDetailBtn, &QPushButton::clicked,
            this, &MatchPairSelectorDialog::onViewDetailedMatch);

    connect(ui.closeBtn, &QPushButton::clicked, this, &QDialog::accept);
}

// setupTable: 初始化匹配对信息表格
// 设置列数（4列）、列头、列宽、选择行为、文字对齐、交替行颜色等属性
void MatchPairSelectorDialog::setupTable()
{
    _matchTable->setColumnCount(6);

    QStringList headers;
    headers << tr("图像") << tr("最佳算法") << tr("有效内点")
            << tr("总匹配") << tr("可用算法") << tr("状态");
    _matchTable->setHorizontalHeaderLabels(headers);

    // 设置列宽
    _matchTable->setColumnWidth(0, 320);
    _matchTable->setColumnWidth(1, 150);
    _matchTable->setColumnWidth(2, 90);
    _matchTable->setColumnWidth(3, 90);
    _matchTable->setColumnWidth(4, 190);
    _matchTable->setColumnWidth(5, 150);
    
    // 设置表格属性
    _matchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _matchTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _matchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _matchTable->horizontalHeader()->setStretchLastSection(true);
    _matchTable->verticalHeader()->setVisible(false);
    _matchTable->setAlternatingRowColors(true);
    
    // 连接信号
    connect(_matchTable, &QTableWidget::cellClicked,
            this, &MatchPairSelectorDialog::onMatchPairSelected);
    connect(_matchTable, &QTableWidget::cellDoubleClicked,
            this, &MatchPairSelectorDialog::onMatchPairDoubleClicked);
}

// loadProjectImages: 从项目管理器读取所有影像，填充顶部下拉框并默认选中第一项
void MatchPairSelectorDialog::loadProjectImages()
{
    if (!_projectManager) {
        _statusLabel->setText(tr("错误：未找到项目管理器"));
        return;
    }
    
    // 获取项目中的所有图像
    _allImages = _projectManager->getAllImages();
    
    if (_allImages.isEmpty()) {
        _statusLabel->setText(tr("项目中没有图像"));
        return;
    }
    
    // 填充下拉框
    _imageComboBox->clear();
    for (const QString &img : _allImages) {
        QString displayName = QFileInfo(img).fileName();
        _imageComboBox->addItem(displayName, img);
    }
    
    // 选择第一个图像
    if (_imageComboBox->count() > 0) {
        _imageComboBox->setCurrentIndex(0);
    }
}

void MatchPairSelectorDialog::onCurrentImageChanged(int index)
{
    if (index < 0 || index >= _imageComboBox->count()) {
        return;
    }
    
    _currentImage = _imageComboBox->itemData(index).toString();
    loadMatchPairsForImage(_currentImage);
}

void MatchPairSelectorDialog::loadMatchPairsForImage(const QString &imagePath)
{
    _matchTable->setRowCount(0);
    _currentMatches.clear();
    _selectedMatchIndex = -1;
    _viewDetailBtn->setEnabled(false);
    
    // 解析匹配数据
    _currentMatches = parseMatchDataForImage(imagePath);
    
    if (_currentMatches.isEmpty()) {
        _statusLabel->setText(tr("该图像没有匹配数据"));
        return;
    }
    
    // 填充表格
    _matchTable->setRowCount(_currentMatches.size());
    
    for (int i = 0; i < _currentMatches.size(); ++i) {
        const MatchInfo &info = _currentMatches[i];
        const QString tooltip = variantsTooltip(info.variants);

        // 图像名称
        QTableWidgetItem *nameItem = new QTableWidgetItem(info.imageName);
        nameItem->setToolTip(info.imagePath);
        _matchTable->setItem(i, 0, nameItem);

        // 算法名
        QString algoDisplay = info.algorithm;
        if (algoDisplay.isEmpty()) algoDisplay = tr("(旧格式)");
        QTableWidgetItem *algoItem = new QTableWidgetItem(algoDisplay);
        algoItem->setTextAlignment(Qt::AlignCenter);
        if (!tooltip.isEmpty())
        {
            algoItem->setToolTip(tooltip);
        }
        _matchTable->setItem(i, 1, algoItem);

        // 有效内点
        QTableWidgetItem *validItem = new QTableWidgetItem(
            info.matchFilePath.isEmpty() || !info.hasInlierStats
                ? QStringLiteral("-")
                : QString::number(info.validPoints));
        validItem->setTextAlignment(Qt::AlignCenter);
        _matchTable->setItem(i, 2, validItem);

        // 总匹配
        QTableWidgetItem *totalItem = new QTableWidgetItem(
            info.matchFilePath.isEmpty()
                ? tr("未匹配")
                : QString::number(info.totalPoints));
        totalItem->setTextAlignment(Qt::AlignCenter);
        _matchTable->setItem(i, 3, totalItem);

        // 可用算法
        QTableWidgetItem *availableItem = new QTableWidgetItem(
            info.availableAlgorithms.isEmpty() ? QStringLiteral("无") : info.availableAlgorithms);
        availableItem->setTextAlignment(Qt::AlignCenter);
        if (!tooltip.isEmpty())
        {
            availableItem->setToolTip(tooltip);
        }
        _matchTable->setItem(i, 4, availableItem);

        // 状态
        QTableWidgetItem *statusItem = new QTableWidgetItem(
            info.status.isEmpty()
                ? (info.matchFilePath.isEmpty() ? tr("候选 / 未匹配") : tr("可查看"))
                : info.status);
        statusItem->setTextAlignment(Qt::AlignCenter);
        if (!tooltip.isEmpty())
        {
            statusItem->setToolTip(tooltip);
        }
        _matchTable->setItem(i, 5, statusItem);
    }
    
    int overlapCandidateCount = 0;
    for (const MatchInfo &info : _currentMatches)
    {
        if (info.overlapCandidate && info.matchFilePath.isEmpty())
        {
            ++overlapCandidateCount;
        }
    }

    if (overlapCandidateCount > 0)
    {
        _statusLabel->setText(tr("找到 %1 个影像对（含 %2 个重叠候选）")
                                   .arg(_currentMatches.size())
                                   .arg(overlapCandidateCount));
    }
    else
    {
        _statusLabel->setText(tr("找到 %1 个匹配对").arg(_currentMatches.size()));
    }
}

QList<MatchPairSelectorDialog::MatchInfo> MatchPairSelectorDialog::parseMatchDataForImage(const QString &imagePath)
{
    QList<MatchInfo> matches;

    if (!_projectManager) return matches;

    const QString baseName = QFileInfo(imagePath).completeBaseName();

    // ── 构建 baseName/fileName → 完整路径 映射（供查找配对影像使用）──────────────
    QMap<QString, QString> baseToPath;    // normalized completeBaseName → fullPath
    for (const QString &imgPath : _allImages) {
        const QString base = imageBaseKey(imgPath);
        if (!baseToPath.contains(base)) baseToPath.insert(base, imgPath);
    }

    // ── 方式一：通过 Catalog 扫描 matchDir/*.match 文件并按影像对聚合 ────────
    QSet<QString> seenMatchFiles;
    QSet<QString> seenPairKeys;

    if (!_matchDir.isEmpty())
    {
        xjw::pipeline::MatchResultCatalogConfig config;
        config.matchDirectory = _matchDir;
        const xjw::pipeline::MatchResultCatalogSummary summary =
            xjw::pipeline::MatchResultCatalog(config).scan();

        for (const xjw::pipeline::MatchPairGroup &group : summary.pairGroups)
        {
            QString otherToken;
            if (!groupContainsImage(group, imagePath, &otherToken))
            {
                continue;
            }

            const QString otherImagePath = resolveProjectImagePath(otherToken, _allImages, baseToPath);
            if (otherImagePath.trimmed().isEmpty() || imageTokenMatches(otherImagePath, imagePath))
            {
                continue;
            }

            const QString pairKey = canonicalPairKeyForImages(imagePath, otherImagePath);
            if (seenPairKeys.contains(pairKey))
            {
                continue;
            }

            const int selectedVariantIndex = bestDisplayVariantIndex(group);
            MatchInfo info;
            info.imagePath = otherImagePath;
            info.imageName = QFileInfo(otherImagePath).fileName();
            if (info.imageName.isEmpty())
            {
                info.imageName = otherToken;
            }
            info.variants = group.variants;
            info.compatibleVariantCount = compatibleVariantCount(info.variants);
            info.availableAlgorithms = availableAlgorithmText(info.variants);
            info.status = catalogRowStatusText(info.variants,
                                               selectedVariantIndex,
                                               info.compatibleVariantCount);

            if (selectedVariantIndex >= 0 && selectedVariantIndex < group.variants.size())
            {
                const xjw::pipeline::MatchVariant &variant = group.variants.at(selectedVariantIndex);
                info.algorithm = matchVariantAlgorithmLabel(variant);
                info.totalPoints = variant.totalMatches;
                info.hasInlierStats = variant.hasInlierStats;
                info.validPoints = variant.hasInlierStats ? variant.geometricVerifiedInliers : 0;
                info.invalidPoints = variant.hasInlierStats
                    ? std::max(0, variant.totalMatches - variant.geometricVerifiedInliers)
                    : 0;
                if (variant.compatible)
                {
                    info.matchFilePath = variant.matchFilePath;
                }
            }

            for (const xjw::pipeline::MatchVariant &variant : group.variants)
            {
                if (!variant.matchFilePath.trimmed().isEmpty())
                {
                    seenMatchFiles.insert(QDir::cleanPath(variant.matchFilePath));
                }
            }
            seenPairKeys.insert(pairKey);
            matches.append(info);
        }
    }

    // ── 方式二：兜底 — 扫描项目元数据（针对仅有元数据无文件的历史记录）──────────
    {
        QJsonObject meta = _projectManager->currentMeta();
        const QString baseFileName = QFileInfo(imagePath).fileName();
        QMap<QString, QString> imageNameToPath;
        for (const QString &img : _allImages) {
            imageNameToPath[QFileInfo(img).fileName()] = img;
        }

        QJsonArray ipmatchResults = meta.value(QStringLiteral("ipmatch_results")).toArray();
        for (const QJsonValue &val : ipmatchResults) {
            if (!val.isObject()) continue;
            const QJsonObject rec = val.toObject();

            // 新格式：顶层 image0/image1
            QString img0 = rec.value(QStringLiteral("image0")).toString();
            QString img1 = rec.value(QStringLiteral("image1")).toString();

            // 兼容旧格式：settings.image_files
            if (img0.isEmpty() || img1.isEmpty()) {
                const QJsonArray imgArr = rec.value(QStringLiteral("settings"))
                    .toObject().value(QStringLiteral("image_files")).toArray();
                if (imgArr.size() >= 2) { img0 = imgArr[0].toString(); img1 = imgArr[1].toString(); }
            }
            if (img0.isEmpty() || img1.isEmpty()) continue;

            QString matchedPath;
            bool containsCurrent = false;
            for (const QString &p : {img0, img1}) {
                if (QFileInfo(p).fileName() == baseFileName ||
                    QFileInfo(p).completeBaseName() == baseName) {
                    containsCurrent = true;
                } else {
                    const QString fn = QFileInfo(p).fileName();
                    matchedPath = imageNameToPath.contains(fn) ? imageNameToPath[fn] : p;
                }
            }
            if (!containsCurrent || matchedPath.isEmpty()) continue;

            QString matchFile = rec.value(QStringLiteral("output")).toString();
            if (matchFile.isEmpty() || !QFile::exists(matchFile))
                matchFile = findMatchFile(imagePath, matchedPath);
            if (matchFile.isEmpty()) continue;

            const QString pairKey = canonicalPairKeyForImages(imagePath, matchedPath);
            if (seenPairKeys.contains(pairKey)) continue;

            const QString cleanPath = QDir::cleanPath(matchFile);
            if (seenMatchFiles.contains(cleanPath)) continue;
            seenMatchFiles.insert(cleanPath);
            seenPairKeys.insert(pairKey);

            matches.append(getMatchStatistics(imagePath, matchedPath, matchFile));
        }
    }

    matches.append(loadOverlapCandidatesForImage(imagePath, seenPairKeys, baseToPath));

    return matches;
}

QList<MatchPairSelectorDialog::MatchInfo> MatchPairSelectorDialog::loadOverlapCandidatesForImage(
    const QString &imagePath,
    const QSet<QString> &seenPairKeys,
    const QMap<QString, QString> &baseToPath) const
{
    QList<MatchInfo> candidates;
    if (!_projectManager || _projectManager->currentProjectPath().isEmpty())
    {
        return candidates;
    }

    const QString overlapDir = QDir(ProjectIO::projectAssetsDir(_projectManager->currentProjectPath()))
                                   .filePath(QStringLiteral("overlap"));
    const QString jsonPath = QDir(overlapDir).filePath(QStringLiteral("vocabulary_overlap_pairs.json"));
    const QString lisPath = QDir(overlapDir).filePath(QStringLiteral("vocabulary_overlap_pairs.lis"));
    QSet<QString> seenOverlapPairs;

    auto appendCandidate = [&](const QString &imageA,
                               const QString &imageB,
                               double overlapScore,
                               const QString &sourcePath)
    {
        if (imageA.trimmed().isEmpty() || imageB.trimmed().isEmpty())
        {
            return;
        }

        if (!imageTokenMatches(imageA, imagePath) && !imageTokenMatches(imageB, imagePath))
        {
            return;
        }

        const QString otherToken = imageTokenMatches(imageA, imagePath) ? imageB : imageA;
        const QString otherImagePath = resolveProjectImagePath(otherToken, _allImages, baseToPath);
        if (otherImagePath.trimmed().isEmpty() || imageTokenMatches(otherImagePath, imagePath))
        {
            return;
        }

        const QString pairKey = canonicalPairKeyForImages(imagePath, otherImagePath);
        if (seenPairKeys.contains(pairKey) || seenOverlapPairs.contains(pairKey))
        {
            return;
        }
        seenOverlapPairs.insert(pairKey);

        MatchInfo info;
        info.imagePath = otherImagePath;
        info.imageName = QFileInfo(otherImagePath).fileName();
        if (info.imageName.isEmpty())
        {
            info.imageName = otherToken;
        }
        info.algorithm = tr("重叠候选");
        info.totalPoints = 0;
        info.validPoints = 0;
        info.invalidPoints = 0;
        info.matchFilePath.clear();
        info.availableAlgorithms = tr("未匹配");
        info.status = tr("候选 / 未匹配");
        info.overlapCandidate = true;
        info.overlapScore = overlapScore;
        info.overlapSource = sourcePath;
        candidates.append(info);
    };

    if (QFile::exists(jsonPath))
    {
        QFile jsonFile(jsonPath);
        if (jsonFile.open(QIODevice::ReadOnly))
        {
            const QJsonDocument doc = QJsonDocument::fromJson(jsonFile.readAll());
            jsonFile.close();
            const QJsonArray candidateArray = doc.object().value(QStringLiteral("candidates")).toArray();
            for (const QJsonValue &value : candidateArray)
            {
                const QJsonObject object = value.toObject();
                if (object.contains(QStringLiteral("accepted")) &&
                    !object.value(QStringLiteral("accepted")).toBool(false))
                {
                    continue;
                }
                appendCandidate(object.value(QStringLiteral("image_a")).toString(),
                                object.value(QStringLiteral("image_b")).toString(),
                                object.value(QStringLiteral("overlap_score")).toDouble(
                                    object.value(QStringLiteral("bow_score")).toDouble(0.0)),
                                jsonPath);
            }
        }
    }

    if (!candidates.isEmpty())
    {
        std::sort(candidates.begin(), candidates.end(), [](const MatchInfo &a, const MatchInfo &b)
        {
            return a.imageName < b.imageName;
        });
        return candidates;
    }

    QFile lisFile(lisPath);
    if (lisFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream stream(&lisFile);
        while (!stream.atEnd())
        {
            const QString line = stream.readLine().trimmed();
            if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
            {
                continue;
            }
            const QStringList parts = line.split(QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts);
            if (parts.size() >= 2)
            {
                appendCandidate(parts.at(0), parts.at(1), 0.0, lisPath);
            }
        }
        lisFile.close();
    }

    std::sort(candidates.begin(), candidates.end(), [](const MatchInfo &a, const MatchInfo &b)
    {
        return a.imageName < b.imageName;
    });
    return candidates;
}

QString MatchPairSelectorDialog::findMatchFile(const QString &imgA, const QString &imgB)
{
    if (!_projectManager) return QString();
    
    QString baseNameA = QFileInfo(imgA).completeBaseName();
    QString baseNameB = QFileInfo(imgB).completeBaseName();
    
    // 第一优先：从 ipmatch_results 元数据中查找
    QJsonObject meta = _projectManager->currentMeta();
    QJsonArray ipmatchResults = meta.value("ipmatch_results").toArray();
    
    for (const QJsonValue &val : ipmatchResults) {
        if (!val.isObject()) continue;
        
        QJsonObject result = val.toObject();
        QJsonObject settings = result.value("settings").toObject();
        QJsonArray imageFiles = settings.value("image_files").toArray();
        
        // 提取影像基础名（支持完整路径或仅基础名）
        QSet<QString> imageBaseNames;
        for (const QJsonValue &imgVal : imageFiles) {
            QString imgPath = imgVal.toString();
            imageBaseNames.insert(QFileInfo(imgPath).completeBaseName());
        }
        
        // 检查是否匹配当前两张影像
        if (imageBaseNames.contains(baseNameA) && imageBaseNames.contains(baseNameB)) {
            QString outputPath = result.value("output").toString();
            if (!outputPath.isEmpty() && QFile::exists(outputPath)) {
                return outputPath;
            }
            
            // 路径不存在时，尝试在项目的 assets/matches 中查找同名文件
            if (!outputPath.isEmpty()) {
                QString fileName = QFileInfo(outputPath).fileName();
                QString projectRoot = QFileInfo(_projectManager->currentProjectPath()).absolutePath();
                QString matchesDir = QDir(projectRoot).filePath("assets/matches");
                QString candidatePath = QDir(matchesDir).filePath(fileName);
                
                if (QFile::exists(candidatePath)) {
                    return candidatePath;
                }
            }
        }
    }
    
    // 第二优先：在 assets/matches 目录中按文件名模式直接搜索
    QString plascanPath = _projectManager->currentProjectPath();
    if (plascanPath.isEmpty()) return QString();
    
    QString projectRoot = QFileInfo(plascanPath).absolutePath();
    QString matchesDir = QDir(projectRoot).filePath("assets/matches");
    
    // 尝试常见命名模式（双向）
    QStringList patterns = {
        QString("%1__%2.match").arg(baseNameA, baseNameB),
        QString("%1__%2.match").arg(baseNameB, baseNameA),
        QString("%1-%2.match").arg(baseNameA, baseNameB),
        QString("%1-%2.match").arg(baseNameB, baseNameA)
    };
    
    for (const QString &pattern : patterns) {
        QString fullPath = QDir(matchesDir).filePath(pattern);
        if (QFile::exists(fullPath)) {
            return fullPath;
        }
    }
    
    return QString();
}

MatchPairSelectorDialog::MatchInfo MatchPairSelectorDialog::getMatchStatistics(
    const QString &imgA, const QString &imgB, const QString &matchFile)
{
    Q_UNUSED(imgA);
    MatchInfo info;
    info.imagePath = imgB;
    info.imageName = QFileInfo(imgB).fileName();
    info.matchFilePath = matchFile;
    info.totalPoints = 0;
    info.validPoints = 0;
    info.invalidPoints = 0;
    info.availableAlgorithms = tr("(旧格式)");
    info.status = tr("可查看");

    QFile file(matchFile);
    if (!file.open(QIODevice::ReadOnly)) {
        info.status = tr("不可用：无法读取匹配文件");
        return info;
    }

    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_5_15);

    char magic[4];
    if (in.readRawData(magic, 4) == 4 && strncmp(magic, "SGMT", 4) == 0) {
        quint32 version = 0;
        in >> version;
        if (version == 1) {
            quint32 img0Len = 0;
            quint32 img1Len = 0;
            in >> img0Len;
            file.seek(file.pos() + static_cast<qint64>(img0Len));
            in >> img1Len;
            file.seek(file.pos() + static_cast<qint64>(img1Len));

            qint32 numMatches = 0;
            qint32 numKp0 = 0;
            qint32 numKp1 = 0;
            in >> numMatches >> numKp0 >> numKp1;
            Q_UNUSED(numKp0);
            Q_UNUSED(numKp1);

            if (numMatches > 0) {
                info.totalPoints = numMatches;
                info.validPoints = numMatches;
                info.invalidPoints = 0;
            }
        }
    }
    
    return info;
}

void MatchPairSelectorDialog::onMatchPairSelected(int row, int column)
{
    Q_UNUSED(column);
    
    if (row < 0 || row >= _currentMatches.size()) {
        _selectedMatchIndex = -1;
        _viewDetailBtn->setEnabled(false);
        return;
    }
    
    _selectedMatchIndex = row;
    _viewDetailBtn->setEnabled(true);
    
    const MatchInfo &info = _currentMatches[row];
    if (info.matchFilePath.isEmpty())
    {
        const QString status = info.status.isEmpty() ? tr("尚未匹配") : info.status;
        _statusLabel->setText(tr("已选择：%1（%2）").arg(info.imageName, status));
    }
    else
    {
        _statusLabel->setText(tr("已选择：%1（%2，总匹配 %3）")
            .arg(info.imageName)
            .arg(info.algorithm.isEmpty() ? tr("(旧格式)") : info.algorithm)
            .arg(info.totalPoints));
    }
}

void MatchPairSelectorDialog::onMatchPairDoubleClicked(int row, int column)
{
    Q_UNUSED(column);
    
    _selectedMatchIndex = row;
    onViewDetailedMatch();
}

void MatchPairSelectorDialog::onViewDetailedMatch()
{
    if (_selectedMatchIndex < 0 || _selectedMatchIndex >= _currentMatches.size()) {
        QMessageBox::warning(this, tr("未选择匹配对"), 
            tr("请先选择要查看的匹配对"));
        return;
    }
    
    const MatchInfo &info = _currentMatches[_selectedMatchIndex];
    
    // 打开详细匹配查看器
    auto *viewer = new MatchViewerDialog(_currentImage, info.imagePath,
                                         info.matchFilePath, this);
    viewer->setAttribute(Qt::WA_DeleteOnClose);

    // 传递项目路径以启用项目级记忆化
    if (_projectManager) {
        viewer->setProjectPath(_projectManager->currentProjectPath());
    }
    viewer->setMatchVariants(info.variants, info.matchFilePath);

    viewer->exec();
}

void MatchPairSelectorDialog::onRefresh()
{
    // 更新 matchDir（防止项目切换后路径变化）
    if (_projectManager && !_projectManager->currentProjectPath().isEmpty()) {
        const QString assetsDir = ProjectIO::projectAssetsDir(_projectManager->currentProjectPath());
        _matchDir = QDir(assetsDir).filePath(QStringLiteral("matches"));
    }

    loadProjectImages();

    if (!_currentImage.isEmpty()) {
        loadMatchPairsForImage(_currentImage);
    }
}

void MatchPairSelectorDialog::scheduleRefresh()
{
    if (_refreshTimer) _refreshTimer->start();
}
