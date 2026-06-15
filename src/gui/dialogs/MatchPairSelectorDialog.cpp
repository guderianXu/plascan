// =============================================================================
// 文件: MatchPairSelectorDialog.cpp
// 说明: MatchPairSelectorDialog 的实现。
//       直接扫描 assets/matches/*.match 文件系统，无需加载大 JSON 元数据，
//       支持在匹配处理过程中实时刷新（通过 projectMetadataChanged 信号触发）。
// =============================================================================
#include "MatchPairSelectorDialog.h"
#include "MatchViewerDialog.h"
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

} // namespace

// 构造函数：初始化对话框，构建界面，加载项目影像列表
MatchPairSelectorDialog::MatchPairSelectorDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
    , m_selectedMatchIndex(-1)   // -1 表示初始无选中行
{
    setWindowTitle(tr("匹配查看器"));
    resize(800, 600);

    // 获取 matches 目录（直接扫描，不依赖元数据）
    if (m_projectManager && !m_projectManager->currentProjectPath().isEmpty()) {
        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        m_matchDir = QDir(assetsDir).filePath(QStringLiteral("matches"));
    }

    setupUI();
    loadProjectImages();

    // ── 实时刷新：projectMetadataChanged / matchPairReady 时自动更新视图 ──
    // 使用防抖 QTimer，避免高频更新导致 UI 闪烁（300ms 内不再触发才真正刷新）
    m_refreshTimer = new QTimer(this);
    m_refreshTimer->setSingleShot(true);
    m_refreshTimer->setInterval(300);
    connect(m_refreshTimer, &QTimer::timeout, this, &MatchPairSelectorDialog::onRefresh);

    if (m_projectManager) {
        connect(m_projectManager, &ProjectManager::projectMetadataChanged,
                this, &MatchPairSelectorDialog::scheduleRefresh);
        connect(m_projectManager, &ProjectManager::matchPairReady,
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

    m_imageComboBox = ui.m_imageComboBox;
    m_matchTable = ui.m_matchTable;
    m_viewDetailBtn = ui.m_viewDetailBtn;
    m_refreshBtn = ui.m_refreshBtn;
    m_statusLabel = ui.m_statusLabel;

    connect(m_imageComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchPairSelectorDialog::onCurrentImageChanged);
    connect(m_refreshBtn, &QPushButton::clicked, this, &MatchPairSelectorDialog::onRefresh);

    setupTable();

    connect(m_viewDetailBtn, &QPushButton::clicked, 
            this, &MatchPairSelectorDialog::onViewDetailedMatch);

    connect(ui.closeBtn, &QPushButton::clicked, this, &QDialog::accept);
}

// setupTable: 初始化匹配对信息表格
// 设置列数（4列）、列头、列宽、选择行为、文字对齐、交替行颜色等属性
void MatchPairSelectorDialog::setupTable()
{
    m_matchTable->setColumnCount(5);

    QStringList headers;
    headers << tr("图像") << tr("算法") << tr("总计") << tr("有效") << tr("无效");
    m_matchTable->setHorizontalHeaderLabels(headers);

    // 设置列宽
    m_matchTable->setColumnWidth(0, 320);
    m_matchTable->setColumnWidth(1, 100);
    m_matchTable->setColumnWidth(2, 80);
    m_matchTable->setColumnWidth(3, 80);
    m_matchTable->setColumnWidth(4, 80);
    
    // 设置表格属性
    m_matchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_matchTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_matchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_matchTable->horizontalHeader()->setStretchLastSection(true);
    m_matchTable->verticalHeader()->setVisible(false);
    m_matchTable->setAlternatingRowColors(true);
    
    // 连接信号
    connect(m_matchTable, &QTableWidget::cellClicked,
            this, &MatchPairSelectorDialog::onMatchPairSelected);
    connect(m_matchTable, &QTableWidget::cellDoubleClicked,
            this, &MatchPairSelectorDialog::onMatchPairDoubleClicked);
}

// loadProjectImages: 从项目管理器读取所有影像，填充顶部下拉框并默认选中第一项
void MatchPairSelectorDialog::loadProjectImages()
{
    if (!m_projectManager) {
        m_statusLabel->setText(tr("错误：未找到项目管理器"));
        return;
    }
    
    // 获取项目中的所有图像
    m_allImages = m_projectManager->getAllImages();
    
    if (m_allImages.isEmpty()) {
        m_statusLabel->setText(tr("项目中没有图像"));
        return;
    }
    
    // 填充下拉框
    m_imageComboBox->clear();
    for (const QString &img : m_allImages) {
        QString displayName = QFileInfo(img).fileName();
        m_imageComboBox->addItem(displayName, img);
    }
    
    // 选择第一个图像
    if (m_imageComboBox->count() > 0) {
        m_imageComboBox->setCurrentIndex(0);
    }
}

void MatchPairSelectorDialog::onCurrentImageChanged(int index)
{
    if (index < 0 || index >= m_imageComboBox->count()) {
        return;
    }
    
    m_currentImage = m_imageComboBox->itemData(index).toString();
    loadMatchPairsForImage(m_currentImage);
}

void MatchPairSelectorDialog::loadMatchPairsForImage(const QString &imagePath)
{
    m_matchTable->setRowCount(0);
    m_currentMatches.clear();
    m_selectedMatchIndex = -1;
    m_viewDetailBtn->setEnabled(false);
    
    // 解析匹配数据
    m_currentMatches = parseMatchDataForImage(imagePath);
    
    if (m_currentMatches.isEmpty()) {
        m_statusLabel->setText(tr("该图像没有匹配数据"));
        return;
    }
    
    // 填充表格
    m_matchTable->setRowCount(m_currentMatches.size());
    
    for (int i = 0; i < m_currentMatches.size(); ++i) {
        const MatchInfo &info = m_currentMatches[i];

        // 图像名称
        QTableWidgetItem *nameItem = new QTableWidgetItem(info.imageName);
        nameItem->setToolTip(info.imagePath);
        m_matchTable->setItem(i, 0, nameItem);

        // 算法名
        QString algoDisplay = info.algorithm;
        if (algoDisplay.isEmpty()) algoDisplay = tr("(旧格式)");
        QTableWidgetItem *algoItem = new QTableWidgetItem(algoDisplay);
        algoItem->setTextAlignment(Qt::AlignCenter);
        m_matchTable->setItem(i, 1, algoItem);

        // 总计
        QTableWidgetItem *totalItem = new QTableWidgetItem(
            info.overlapCandidate && info.matchFilePath.isEmpty()
                ? tr("未匹配")
                : QString::number(info.totalPoints));
        totalItem->setTextAlignment(Qt::AlignCenter);
        m_matchTable->setItem(i, 2, totalItem);

        // 有效
        QTableWidgetItem *validItem = new QTableWidgetItem(
            info.overlapCandidate && info.matchFilePath.isEmpty()
                ? QStringLiteral("-")
                : QString::number(info.validPoints));
        validItem->setTextAlignment(Qt::AlignCenter);
        m_matchTable->setItem(i, 3, validItem);

        // 无效
        QTableWidgetItem *invalidItem = new QTableWidgetItem(
            info.overlapCandidate && info.matchFilePath.isEmpty()
                ? QStringLiteral("-")
                : QString::number(info.invalidPoints));
        invalidItem->setTextAlignment(Qt::AlignCenter);
        m_matchTable->setItem(i, 4, invalidItem);
    }
    
    int overlapCandidateCount = 0;
    for (const MatchInfo &info : m_currentMatches)
    {
        if (info.overlapCandidate && info.matchFilePath.isEmpty())
        {
            ++overlapCandidateCount;
        }
    }

    if (overlapCandidateCount > 0)
    {
        m_statusLabel->setText(tr("找到 %1 个影像对（含 %2 个重叠候选）")
                                   .arg(m_currentMatches.size())
                                   .arg(overlapCandidateCount));
    }
    else
    {
        m_statusLabel->setText(tr("找到 %1 个匹配对").arg(m_currentMatches.size()));
    }
}

QList<MatchPairSelectorDialog::MatchInfo> MatchPairSelectorDialog::parseMatchDataForImage(const QString &imagePath)
{
    QList<MatchInfo> matches;

    if (!m_projectManager) return matches;

    const QString baseName = QFileInfo(imagePath).completeBaseName();

    // ── 构建 baseName/fileName → 完整路径 映射（供查找配对影像使用）──────────────
    QMap<QString, QString> baseToPath;    // completeBaseName → fullPath
    for (const QString &imgPath : m_allImages) {
        const QString base = QFileInfo(imgPath).completeBaseName();
        if (!baseToPath.contains(base)) baseToPath.insert(base, imgPath);
    }

    // ── 方式一：扫描 matchDir/*.match 文件（实时，无需等待元数据写入）───────────
    QSet<QString> seenMatchFiles;
    QSet<QString> seenPairKeys;

    // 已知算法后缀名列表(匹配文件名中可能出现)
    static const QStringList knownAlgos = {"superglue","lightglue","loftr","roma",
                                            "orb_bf_hamming","sift_bf_l2","sift_flann"};

    if (!m_matchDir.isEmpty()) {
        QDir matchDirObj(m_matchDir);
        const QStringList matchFiles = matchDirObj.entryList(
            QStringList{QStringLiteral("*.match")}, QDir::Files);

        for (const QString &mf : matchFiles) {
            const QString stem = QFileInfo(mf).completeBaseName();  // e.g. "A__B_superglue"
            const QStringList parts = stem.split(QStringLiteral("__"));
            if (parts.size() != 2) continue;

            const QString &partA = parts[0];
            QString partB = parts[1];
            // 解析算法后缀: "B_algo" -> base="B", algo="algo"
            QString algoName;
            for (const auto &algo : knownAlgos) {
                const QString suffix = "_" + algo;
                if (partB.endsWith(suffix)) {
                    partB.chop(suffix.size());
                    algoName = algo;
                    break;
                }
            }

            // 检查是否包含当前影像
            bool containsCurrent = false;
            QString otherBase;
            if (partA == baseName)      { containsCurrent = true; otherBase = partB; }
            else if (partB == baseName) { containsCurrent = true; otherBase = partA; }

            if (!containsCurrent) continue;

            const QString matchFilePath = matchDirObj.filePath(mf);
            const QString cleanPath = QDir::cleanPath(matchFilePath);
            if (seenMatchFiles.contains(cleanPath)) continue;
            seenMatchFiles.insert(cleanPath);

            const QString otherImagePath = baseToPath.value(otherBase, otherBase);
            seenPairKeys.insert(canonicalPairKeyForImages(imagePath, otherImagePath));
            MatchInfo info = getMatchStatistics(imagePath, otherImagePath, matchFilePath);
            if (info.imageName.isEmpty())
                info.imageName = otherBase;
            if (info.imagePath.isEmpty())
                info.imagePath = otherImagePath;
            if (!algoName.isEmpty())
                info.algorithm = algoName;
            matches.append(info);
        }
    }

    // ── 方式二：兜底 — 扫描项目元数据（针对仅有元数据无文件的历史记录）──────────
    // 仅当 matchDir 不存在或为空时才回退到元数据读取
    if (matches.isEmpty() && m_matchDir.isEmpty()) {
        QJsonObject meta = m_projectManager->currentMeta();
        const QString baseFileName = QFileInfo(imagePath).fileName();
        QMap<QString, QString> imageNameToPath;
        for (const QString &img : m_allImages) {
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

            const QString cleanPath = QDir::cleanPath(matchFile);
            if (seenMatchFiles.contains(cleanPath)) continue;
            seenMatchFiles.insert(cleanPath);
            seenPairKeys.insert(canonicalPairKeyForImages(imagePath, matchedPath));

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
    if (!m_projectManager || m_projectManager->currentProjectPath().isEmpty())
    {
        return candidates;
    }

    const QString overlapDir = QDir(ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath()))
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
        const QString otherImagePath = resolveProjectImagePath(otherToken, m_allImages, baseToPath);
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
    if (!m_projectManager) return QString();
    
    QString baseNameA = QFileInfo(imgA).completeBaseName();
    QString baseNameB = QFileInfo(imgB).completeBaseName();
    
    // 第一优先：从 ipmatch_results 元数据中查找
    QJsonObject meta = m_projectManager->currentMeta();
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
                QString projectRoot = QFileInfo(m_projectManager->currentProjectPath()).absolutePath();
                QString matchesDir = QDir(projectRoot).filePath("assets/matches");
                QString candidatePath = QDir(matchesDir).filePath(fileName);
                
                if (QFile::exists(candidatePath)) {
                    return candidatePath;
                }
            }
        }
    }
    
    // 第二优先：在 assets/matches 目录中按文件名模式直接搜索
    QString plascanPath = m_projectManager->currentProjectPath();
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

    QFile file(matchFile);
    if (!file.open(QIODevice::ReadOnly)) {
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
    
    if (row < 0 || row >= m_currentMatches.size()) {
        m_selectedMatchIndex = -1;
        m_viewDetailBtn->setEnabled(false);
        return;
    }
    
    m_selectedMatchIndex = row;
    m_viewDetailBtn->setEnabled(true);
    
    const MatchInfo &info = m_currentMatches[row];
    if (info.overlapCandidate && info.matchFilePath.isEmpty())
    {
        m_statusLabel->setText(tr("已选择：%1（重叠候选，尚未匹配）").arg(info.imageName));
    }
    else
    {
        m_statusLabel->setText(tr("已选择：%1 (%2 个匹配点)")
            .arg(info.imageName)
            .arg(info.totalPoints));
    }
}

void MatchPairSelectorDialog::onMatchPairDoubleClicked(int row, int column)
{
    Q_UNUSED(column);
    
    m_selectedMatchIndex = row;
    onViewDetailedMatch();
}

void MatchPairSelectorDialog::onViewDetailedMatch()
{
    if (m_selectedMatchIndex < 0 || m_selectedMatchIndex >= m_currentMatches.size()) {
        QMessageBox::warning(this, tr("未选择匹配对"), 
            tr("请先选择要查看的匹配对"));
        return;
    }
    
    const MatchInfo &info = m_currentMatches[m_selectedMatchIndex];
    
    // 打开详细匹配查看器
    auto *viewer = new MatchViewerDialog(m_currentImage, info.imagePath, 
                                         info.matchFilePath, this);
    viewer->setAttribute(Qt::WA_DeleteOnClose);

    // 传递项目路径以启用项目级记忆化
    if (m_projectManager) {
        viewer->setProjectPath(m_projectManager->currentProjectPath());
    }

    viewer->exec();
}

void MatchPairSelectorDialog::onRefresh()
{
    // 更新 matchDir（防止项目切换后路径变化）
    if (m_projectManager && !m_projectManager->currentProjectPath().isEmpty()) {
        const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
        m_matchDir = QDir(assetsDir).filePath(QStringLiteral("matches"));
    }

    loadProjectImages();

    if (!m_currentImage.isEmpty()) {
        loadMatchPairsForImage(m_currentImage);
    }
}

void MatchPairSelectorDialog::scheduleRefresh()
{
    if (m_refreshTimer) m_refreshTimer->start();
}
