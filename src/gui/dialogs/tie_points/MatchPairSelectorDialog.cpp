// =============================================================================
// 文件: MatchPairSelectorDialog.cpp
// 说明: MatchPairSelectorDialog 的实现。
//       后台扫描 assets/image_matches/*.pimatch 逐影像分片，无需加载成对 sidecar，
//       支持在匹配处理过程中实时刷新（通过 projectMetadataChanged 信号触发）。
// =============================================================================
#include "tie_points/MatchPairSelectorDialog.h"
#include "tie_points/MatchViewerDialog.h"
#include "preparation/MatchResultCatalog.h"
#include "ImageMatchFile.h"
#include "ProjectManager.h"
#include "project/ProjectIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "Logger.h"
#include "ui_MatchPairSelectorDialog.h"

#include <QComboBox>
#include <QCollator>
#include <QTableWidget>
#include <QPushButton>
#include <QProgressBar>
#include <QLabel>
#include <QHeaderView>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonValue>
#include <QMessageBox>
#include <QFile>
#include <QTimer>
#include <QSet>
#include <QRegularExpression>
#include <QFutureWatcher>
#include <QPointer>
#include <QSignalBlocker>
#include <QMetaObject>
#include <QtConcurrent/QtConcurrent>

#include <algorithm>

namespace {

using xjw::common::project::imageBaseToken;
using xjw::common::project::imageTokensReferToSameImage;
using xjw::common::project::normalizedImageToken;

QString canonicalPairKeyForImages(const QString &imageA, const QString &imageB)
{
    return xjw::common::project::canonicalImagePairKey(
        normalizedImageToken(imageA),
        normalizedImageToken(imageB),
        QStringLiteral("|"));
}

bool pairContainsImage(const QString &imageA,
                       const QString &imageB,
                       const QString &imagePath,
                       QString *otherImageToken)
{
    if (imageTokensReferToSameImage(imageA, imagePath))
    {
        if (otherImageToken)
        {
            *otherImageToken = imageB;
        }
        return true;
    }
    if (imageTokensReferToSameImage(imageB, imagePath))
    {
        if (otherImageToken)
        {
            *otherImageToken = imageA;
        }
        return true;
    }
    return false;
}

bool groupContainsImage(const xjw::aerial_triangulation::MatchPairGroup &group,
                        const QString &imagePath,
                        QString *otherImageToken)
{
    if (pairContainsImage(group.imageA, group.imageB, imagePath, otherImageToken))
    {
        return true;
    }

    for (const xjw::aerial_triangulation::MatchVariant &variant : group.variants)
    {
        if (pairContainsImage(variant.imageA, variant.imageB, imagePath, otherImageToken))
        {
            return true;
        }
    }
    return false;
}

QString matchVariantAlgorithmLabel(const xjw::aerial_triangulation::MatchVariant &variant)
{
    const QString label = xjw::aerial_triangulation::MatchResultCatalog::algorithmDisplayLabel(variant);
    return label == QStringLiteral("unknown") ? QStringLiteral("(未知算法)") : label;
}

QString matchVariantReasonLabel(const xjw::aerial_triangulation::MatchVariant &variant)
{
    if (variant.compatible)
    {
        return QStringLiteral("可查看");
    }
    if (variant.status == QStringLiteral("invalid_match_file"))
    {
        return QStringLiteral("匹配分片无效");
    }
    return variant.reason.isEmpty() ? variant.status : variant.reason;
}

int bestDisplayVariantIndex(const xjw::aerial_triangulation::MatchPairGroup &group)
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

int compatibleVariantCount(const QVector<xjw::aerial_triangulation::MatchVariant> &variants)
{
    int count = 0;
    for (const xjw::aerial_triangulation::MatchVariant &variant : variants)
    {
        if (variant.compatible && !variant.matchFilePath.trimmed().isEmpty())
        {
            ++count;
        }
    }
    return count;
}

QString availableAlgorithmText(const QVector<xjw::aerial_triangulation::MatchVariant> &variants)
{
    QStringList labels;
    for (const xjw::aerial_triangulation::MatchVariant &variant : variants)
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

QString catalogRowStatusText(const QVector<xjw::aerial_triangulation::MatchVariant> &variants,
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

QString variantsTooltip(const QVector<xjw::aerial_triangulation::MatchVariant> &variants)
{
    QStringList lines;
    for (const xjw::aerial_triangulation::MatchVariant &variant : variants)
    {
        const QString counts = variant.hasInlierStats
            ? QStringLiteral("几何内点 %1 / 原始匹配 %2")
                  .arg(variant.geometricVerifiedInliers)
                  .arg(variant.totalMatches)
            : QStringLiteral("原始匹配 %1").arg(variant.totalMatches);
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

    setupUI();

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

    QTimer::singleShot(0, this, &MatchPairSelectorDialog::onRefresh);
}

MatchPairSelectorDialog::~MatchPairSelectorDialog()
{
    if (_priorityMatchLoadWatcher)
    {
        disconnect(_priorityMatchLoadWatcher, nullptr, this, nullptr);
        _priorityMatchLoadWatcher->deleteLater();
        _priorityMatchLoadWatcher = nullptr;
    }
    if (_matchLoadWatcher)
    {
        disconnect(_matchLoadWatcher, nullptr, this, nullptr);
        _matchLoadWatcher->deleteLater();
        _matchLoadWatcher = nullptr;
    }
}

void MatchPairSelectorDialog::setInitialImagePath(const QString &imagePath)
{
    const QString requested_image = imagePath.trimmed();
    if (requested_image.isEmpty())
    {
        return;
    }

    // 构造后首次刷新尚未填充下拉框时，loadProjectImages() 会按该路径恢复选择。
    _currentImage = requested_image;
    if (!_imageComboBox)
    {
        return;
    }

    for (int index = 0; index < _imageComboBox->count(); ++index)
    {
        const QString candidate = _imageComboBox->itemData(index).toString();
        if (normalizedImageToken(candidate) != normalizedImageToken(requested_image))
        {
            continue;
        }

        _currentImage = candidate;
        if (_imageComboBox->currentIndex() != index)
        {
            _imageComboBox->setCurrentIndex(index);
        }
        else
        {
            loadMatchPairsForImage(_currentImage);
        }
        return;
    }
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
    _scanProgressBar = ui.m_scanProgressBar;
    _scanProgressBar->setRange(0, 100);
    _scanProgressBar->setValue(0);
    _scanProgressBar->setVisible(false);

    connect(_imageComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchPairSelectorDialog::onCurrentImageChanged);
    connect(_refreshBtn, &QPushButton::clicked, this, &MatchPairSelectorDialog::onRefresh);

    setupTable();

    connect(_viewDetailBtn, &QPushButton::clicked,
            this, &MatchPairSelectorDialog::onViewDetailedMatch);

    connect(ui.closeBtn, &QPushButton::clicked, this, &QDialog::accept);
}

// setupTable: 初始化匹配对信息表格
// 设置列头、列宽、选择行为、文字对齐、交替行颜色等属性
void MatchPairSelectorDialog::setupTable()
{
    _matchTable->setColumnCount(6);

    QStringList headers;
    headers << tr("图像") << tr("原始匹配")
            << tr("有效连接点") << tr("无效匹配")
            << tr("最佳算法") << tr("状态");
    _matchTable->setHorizontalHeaderLabels(headers);

    // 设置列宽
    _matchTable->setColumnWidth(0, 320);
    _matchTable->setColumnWidth(1, 110);
    _matchTable->setColumnWidth(2, 110);
    _matchTable->setColumnWidth(3, 90);
    _matchTable->setColumnWidth(4, 170);
    _matchTable->setColumnWidth(5, 160);
    
    // 设置表格属性
    _matchTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _matchTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _matchTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _matchTable->horizontalHeader()->setStretchLastSection(true);
    _matchTable->horizontalHeader()->setSectionsClickable(true);
    _matchTable->horizontalHeader()->setSortIndicatorShown(true);
    _matchTable->horizontalHeader()->setSortIndicator(_sortColumn, _sortOrder);
    _matchTable->verticalHeader()->setVisible(false);
    _matchTable->setAlternatingRowColors(true);
    
    // 连接信号
    connect(_matchTable, &QTableWidget::cellClicked,
            this, &MatchPairSelectorDialog::onMatchPairSelected);
    connect(_matchTable, &QTableWidget::cellDoubleClicked,
            this, &MatchPairSelectorDialog::onMatchPairDoubleClicked);
    connect(_matchTable->horizontalHeader(), &QHeaderView::sectionClicked,
            this, &MatchPairSelectorDialog::onSortSectionClicked);
}

// loadProjectImages: 从项目管理器读取所有影像，填充顶部下拉框并默认选中第一项
void MatchPairSelectorDialog::loadProjectImages()
{
    const QString previousImage = _currentImage;
    const QSignalBlocker blocker(_imageComboBox);
    _imageComboBox->clear();
    _currentImage.clear();

    if (!_projectManager) {
        _scanProgressBar->setVisible(false);
        _statusLabel->setText(tr("错误：未找到项目管理器"));
        return;
    }
    
    // 获取项目中的所有图像
    _allImages = _projectManager->getAllImages();
    
    if (_allImages.isEmpty()) {
        _matchTable->setRowCount(0);
        _currentMatches.clear();
        _scanProgressBar->setVisible(false);
        _statusLabel->setText(tr("项目中没有图像"));
        return;
    }
    
    // 填充下拉框
    int selectedIndex = 0;
    for (const QString &img : _allImages) {
        QString displayName = QFileInfo(img).fileName();
        _imageComboBox->addItem(displayName, img);
        if (!previousImage.isEmpty() && normalizedImageToken(img) == normalizedImageToken(previousImage))
        {
            selectedIndex = _imageComboBox->count() - 1;
        }
    }
    
    // 选择上次图像或第一个图像；信号被阻塞，避免在填充下拉框时同步触发扫描。
    if (_imageComboBox->count() > 0) {
        _imageComboBox->setCurrentIndex(selectedIndex);
        _currentImage = _imageComboBox->itemData(selectedIndex).toString();
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

    if (imagePath.trimmed().isEmpty())
    {
        _scanProgressBar->setVisible(false);
        _statusLabel->setText(tr("请选择图像"));
        return;
    }

    _scanProgressBar->setRange(0, 100);
    _scanProgressBar->setValue(0);
    _scanProgressBar->setVisible(true);
    _statusLabel->setText(tr("正在优先加载当前影像匹配..."));
    setMatchControlsBusy(true);
    startAsyncMatchPairLoad(imagePath);
}

void MatchPairSelectorDialog::populateMatchTable()
{
    sortCurrentMatches();
    _matchTable->setRowCount(0);
    _selectedMatchIndex = -1;
    _viewDetailBtn->setEnabled(false);

    if (_currentMatches.isEmpty()) {
        _statusLabel->setText(tr("该图像没有匹配数据"));
        return;
    }

    _matchTable->setRowCount(_currentMatches.size());
    
    for (int i = 0; i < _currentMatches.size(); ++i) {
        const MatchInfo &info = _currentMatches[i];
        const QString tooltip = variantsTooltip(info.variants);

        // 图像名称
        QTableWidgetItem *nameItem = new QTableWidgetItem(info.imageName);
        nameItem->setToolTip(info.imagePath);
        _matchTable->setItem(i, 0, nameItem);

        // 原始匹配：LightGlue/传统匹配直接输出的两两匹配数量。
        QTableWidgetItem *totalItem = new QTableWidgetItem(
            info.matchFilePath.isEmpty()
                ? tr("未匹配")
                : QString::number(info.totalPoints));
        totalItem->setTextAlignment(Qt::AlignCenter);
        _matchTable->setItem(i, 1, totalItem);

        // 有效连接点：按 Metashape View Matches 语义，只使用空三后保留下来的 tie point。
        // 两视图几何验证内点仍保留在算法变体摘要中，不混入有效连接点统计。
        const bool hasValidityStats = info.hasTrackValidity;
        QTableWidgetItem *validItem = new QTableWidgetItem(
            info.matchFilePath.isEmpty() || !hasValidityStats
                ? QStringLiteral("-")
                : QString::number(info.validPoints));
        validItem->setTextAlignment(Qt::AlignCenter);
        _matchTable->setItem(i, 2, validItem);

        // 无效匹配：原始匹配中没有进入最终空三轨迹的部分。
        QTableWidgetItem *invalidItem = new QTableWidgetItem(
            info.matchFilePath.isEmpty() || !hasValidityStats
                ? QStringLiteral("-")
                : QString::number(info.invalidPoints));
        invalidItem->setTextAlignment(Qt::AlignCenter);
        _matchTable->setItem(i, 3, invalidItem);

        // 算法名
        QString algoDisplay = info.algorithm;
        if (algoDisplay.isEmpty()) algoDisplay = tr("(旧格式)");
        QTableWidgetItem *algoItem = new QTableWidgetItem(algoDisplay);
        algoItem->setTextAlignment(Qt::AlignCenter);
        if (!tooltip.isEmpty())
        {
            algoItem->setToolTip(tooltip);
        }
        _matchTable->setItem(i, 4, algoItem);

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

void MatchPairSelectorDialog::onSortSectionClicked(int column)
{
    if (column < 0 || column > 3)
    {
        return;
    }

    if (_sortColumn == column)
    {
        _sortOrder = _sortOrder == Qt::DescendingOrder
            ? Qt::AscendingOrder
            : Qt::DescendingOrder;
    }
    else
    {
        _sortColumn = column;
        _sortOrder = Qt::DescendingOrder;
    }
    _matchTable->horizontalHeader()->setSortIndicator(_sortColumn, _sortOrder);
    populateMatchTable();
}

void MatchPairSelectorDialog::sortCurrentMatches()
{
    QCollator filename_collator;
    filename_collator.setCaseSensitivity(Qt::CaseInsensitive);
    filename_collator.setNumericMode(true);

    const int sort_column = _sortColumn;
    const Qt::SortOrder sort_order = _sortOrder;
    std::stable_sort(_currentMatches.begin(), _currentMatches.end(),
                     [&filename_collator, sort_column, sort_order](const MatchInfo &left,
                                                                  const MatchInfo &right)
    {
        const auto compare_filenames = [&filename_collator](const MatchInfo &lhs, const MatchInfo &rhs)
        {
            return filename_collator.compare(lhs.imageName, rhs.imageName);
        };
        if (sort_column == 0)
        {
            const int comparison = compare_filenames(left, right);
            return sort_order == Qt::AscendingOrder ? comparison < 0 : comparison > 0;
        }

        const auto has_metric = [sort_column](const MatchInfo &info)
        {
            return sort_column == 1 ? !info.matchFilePath.isEmpty() : info.hasTrackValidity;
        };
        const bool left_has_metric = has_metric(left);
        const bool right_has_metric = has_metric(right);
        if (left_has_metric != right_has_metric)
        {
            return left_has_metric;
        }
        if (!left_has_metric)
        {
            return compare_filenames(left, right) < 0;
        }

        const auto metric = [sort_column](const MatchInfo &info)
        {
            if (sort_column == 1)
            {
                return info.totalPoints;
            }
            if (sort_column == 2)
            {
                return info.validPoints;
            }
            return info.invalidPoints;
        };
        const int left_metric = metric(left);
        const int right_metric = metric(right);
        if (left_metric == right_metric)
        {
            return compare_filenames(left, right) < 0;
        }
        return sort_order == Qt::AscendingOrder
            ? left_metric < right_metric
            : left_metric > right_metric;
    });
}

void MatchPairSelectorDialog::setMatchControlsBusy(bool busy)
{
    if (_refreshBtn)
    {
        _refreshBtn->setEnabled(!busy);
    }
    if (_viewDetailBtn)
    {
        _viewDetailBtn->setEnabled(!busy && _selectedMatchIndex >= 0);
    }
}

MatchPairSelectorDialog::MatchDataSnapshot MatchPairSelectorDialog::makeSnapshot() const
{
    MatchDataSnapshot snapshot;
    if (!_projectManager)
    {
        return snapshot;
    }

    snapshot.projectPath = _projectManager->currentProjectPath();
    if (!snapshot.projectPath.isEmpty())
    {
        snapshot.matchDir = xjw::common::project::ProjectIO::imageMatchOutputDir(snapshot.projectPath);
    }
    snapshot.allImages = _allImages.isEmpty() ? _projectManager->getAllImages() : _allImages;
    snapshot.meta = _projectManager->currentMeta();
    return snapshot;
}

void MatchPairSelectorDialog::startAsyncMatchPairLoad(const QString &imagePath)
{
    const MatchDataSnapshot snapshot = makeSnapshot();
    const int generation = ++_matchLoadGeneration;

    auto releaseWatcher = [this](QFutureWatcher<MatchInfoList> *&watcher)
    {
        if (!watcher)
        {
            return;
        }
        disconnect(watcher, nullptr, this, nullptr);
        watcher->deleteLater();
        watcher = nullptr;
    };
    releaseWatcher(_priorityMatchLoadWatcher);
    releaseWatcher(_matchLoadWatcher);

    _priorityMatchLoadWatcher = new QFutureWatcher<MatchInfoList>(this);
    _priorityMatchLoadWatcher->setProperty("generation", generation);
    _priorityMatchLoadWatcher->setProperty("imagePath", imagePath);
    _priorityMatchLoadWatcher->setProperty("priorityLoad", true);
    connect(_priorityMatchLoadWatcher,
            &QFutureWatcher<MatchInfoList>::finished,
            this,
            &MatchPairSelectorDialog::onMatchPairsLoaded);
    _priorityMatchLoadWatcher->setFuture(QtConcurrent::run([snapshot, imagePath]()
    {
        return MatchPairSelectorDialog::parsePriorityMatchDataForImageFromSnapshot(snapshot, imagePath);
    }));
}

void MatchPairSelectorDialog::startFullMatchPairLoad(const MatchDataSnapshot &snapshot,
                                                    const QString &imagePath,
                                                    int generation)
{
    if (_matchLoadWatcher)
    {
        disconnect(_matchLoadWatcher, nullptr, this, nullptr);
        _matchLoadWatcher->deleteLater();
        _matchLoadWatcher = nullptr;
    }

    _scanProgressBar->setRange(0, 100);
    _scanProgressBar->setValue(0);
    _scanProgressBar->setVisible(true);
    _statusLabel->setText(tr("正在后台访问全部匹配数据：0%"));

    _matchLoadWatcher = new QFutureWatcher<MatchInfoList>(this);
    _matchLoadWatcher->setProperty("generation", generation);
    _matchLoadWatcher->setProperty("imagePath", imagePath);
    _matchLoadWatcher->setProperty("priorityLoad", false);
    connect(_matchLoadWatcher,
            &QFutureWatcher<MatchInfoList>::finished,
            this,
            &MatchPairSelectorDialog::onMatchPairsLoaded);
    QPointer<MatchPairSelectorDialog> self(this);
    auto progressCallback = [self, generation, imagePath](int processed, int total)
    {
        if (!self)
        {
            return;
        }

        QMetaObject::invokeMethod(self.data(), [self, generation, imagePath, processed, total]()
        {
            if (!self)
            {
                return;
            }
            self->setFullScanProgress(processed, total, imagePath, generation);
        }, Qt::QueuedConnection);
    };

    _matchLoadWatcher->setFuture(QtConcurrent::run([snapshot, imagePath, progressCallback]()
    {
        return MatchPairSelectorDialog::parseMatchDataForImageFromSnapshot(snapshot, imagePath, progressCallback);
    }));
}

void MatchPairSelectorDialog::setFullScanProgress(int processed,
                                                  int total,
                                                  const QString &imagePath,
                                                  int generation)
{
    if (!_matchLoadWatcher)
    {
        return;
    }
    if (generation != _matchLoadGeneration ||
        normalizedImageToken(imagePath) != normalizedImageToken(_currentImage))
    {
        return;
    }

    const int percent = total <= 0
        ? 0
        : qBound(0, static_cast<int>((static_cast<qint64>(processed) * 100) / total), 100);
    _scanProgressBar->setRange(0, 100);
    _scanProgressBar->setValue(percent);
    _scanProgressBar->setVisible(true);
    _statusLabel->setText(tr("正在后台访问全部匹配数据：%1%").arg(percent));
}

void MatchPairSelectorDialog::onMatchPairsLoaded()
{
    QObject *senderObject = sender();
    auto *watcher = static_cast<QFutureWatcher<MatchInfoList> *>(senderObject);
    if (!watcher)
    {
        return;
    }

    const int generation = watcher->property("generation").toInt();
    const QString imagePath = watcher->property("imagePath").toString();
    const bool priorityLoad = watcher->property("priorityLoad").toBool();
    const bool isCurrent =
        (priorityLoad ? watcher == _priorityMatchLoadWatcher : watcher == _matchLoadWatcher) &&
        generation == _matchLoadGeneration &&
        normalizedImageToken(imagePath) == normalizedImageToken(_currentImage);

    MatchInfoList matches = watcher->result();
    watcher->deleteLater();
    if (watcher == _priorityMatchLoadWatcher)
    {
        _priorityMatchLoadWatcher = nullptr;
    }
    if (watcher == _matchLoadWatcher)
    {
        _matchLoadWatcher = nullptr;
    }

    if (!isCurrent)
    {
        return;
    }

    const QStringList current_project_images =
        _projectManager ? _projectManager->getAllImages() : QStringList();
    if (xjw::common::project::resolveProjectImagePathFromToken(imagePath,
                                                               current_project_images)
            .isEmpty())
    {
        scheduleRefresh();
        return;
    }
    matches.erase(
        std::remove_if(matches.begin(),
                       matches.end(),
                       [&current_project_images](const MatchInfo &match)
                       {
                           return xjw::common::project::resolveProjectImagePathFromToken(
                                      match.imagePath,
                                      current_project_images)
                               .isEmpty();
                       }),
        matches.end());

    if (priorityLoad)
    {
        if (!matches.isEmpty())
        {
            _currentMatches = matches;
            populateMatchTable();
            _scanProgressBar->setVisible(true);
            _statusLabel->setText(tr("已优先加载 %1 个匹配对，正在后台访问全部匹配数据...")
                                      .arg(_currentMatches.size()));
        }
        else
        {
            _scanProgressBar->setVisible(true);
            _statusLabel->setText(tr("正在后台访问全部匹配数据..."));
        }
        if (_refreshBtn)
        {
            _refreshBtn->setEnabled(false);
        }
        startFullMatchPairLoad(makeSnapshot(), imagePath, generation);
        return;
    }

    _currentMatches = matches;
    setMatchControlsBusy(false);
    populateMatchTable();
    _scanProgressBar->setVisible(false);
}

MatchPairSelectorDialog::MatchInfoList MatchPairSelectorDialog::parsePriorityMatchDataForImageFromSnapshot(
    const MatchDataSnapshot &snapshot,
    const QString &imagePath)
{
    MatchInfoList matches;
    if (snapshot.allImages.isEmpty() || snapshot.matchDir.trimmed().isEmpty())
    {
        return matches;
    }

    const QString shardPath = xjw::image_matching::ImageMatchFile::filePathForImage(
        snapshot.matchDir, imagePath);
    xjw::image_matching::ImageMatchShard shard;
    QString readError;
    if (!xjw::image_matching::ImageMatchFile::read(shardPath, &shard, &readError))
    {
        return matches;
    }

    QMap<QString, MatchInfo> byPeer;
    for (const xjw::image_matching::NeighborMatchBlock &block : shard.neighbors)
    {
        const QString otherImagePath =
            xjw::common::project::resolveProjectImagePathFromToken(block.peer.path,
                                                                   snapshot.allImages);
        if (otherImagePath.trimmed().isEmpty() || imageTokensReferToSameImage(otherImagePath, imagePath))
        {
            continue;
        }

        xjw::aerial_triangulation::MatchVariant variant;
        variant.imageA = imagePath;
        variant.imageB = otherImagePath;
        variant.algorithmId = block.algorithmId;
        variant.algorithmVersion = block.algorithmVersion;
        variant.configFingerprint = block.configFingerprint;
        variant.matchFilePath = shardPath;
        variant.peerMatchFilePath = xjw::image_matching::ImageMatchFile::filePathForImage(
            snapshot.matchDir, otherImagePath);
        variant.totalMatches = static_cast<int>(block.rawMatchCount);
        variant.geometricVerifiedInliers = static_cast<int>(block.geometryInlierCount);
        variant.tiePointMatches = static_cast<int>(block.tiePointMatchCount);
        variant.geometryPassed = block.geometryPassed;
        variant.compatible = true;
        variant.status = QStringLiteral("priority_loaded");

        MatchInfo &info = byPeer[otherImagePath];
        if (info.imagePath.isEmpty())
        {
            info.imagePath = otherImagePath;
            info.imageName = QFileInfo(otherImagePath).fileName();
            info.matchFilePath = shardPath;
        }
        info.variants.push_back(variant);
        const bool useVariant = info.algorithm.isEmpty() ||
            variant.geometricVerifiedInliers > info.validPoints;
        if (useVariant)
        {
            info.algorithm = matchVariantAlgorithmLabel(variant);
            info.totalPoints = variant.totalMatches;
            info.hasInlierStats = true;
            info.validPoints = variant.tiePointMatches > 0
                ? variant.tiePointMatches
                : variant.geometricVerifiedInliers;
            info.invalidPoints = std::max(0, info.totalPoints - info.validPoints);
            info.hasTrackValidity = variant.tiePointMatches > 0;
        }
    }

    for (auto it = byPeer.begin(); it != byPeer.end(); ++it)
    {
        MatchInfo info = it.value();
        info.compatibleVariantCount = compatibleVariantCount(info.variants);
        info.availableAlgorithms = availableAlgorithmText(info.variants);
        info.status = info.hasTrackValidity ? tr("已对齐（快速）") : tr("可查看（快速）");
        matches.append(std::move(info));
    }

    std::sort(matches.begin(), matches.end(), [](const MatchInfo &a, const MatchInfo &b)
    {
        return a.imageName < b.imageName;
    });
    return matches;
}

MatchPairSelectorDialog::MatchInfoList MatchPairSelectorDialog::parseMatchDataForImageFromSnapshot(
    const MatchDataSnapshot &snapshot,
    const QString &imagePath,
    const MatchScanProgressCallback &progressCallback)
{
    MatchInfoList matches;

    if (snapshot.allImages.isEmpty()) return matches;

    const QString baseName = QFileInfo(imagePath).completeBaseName();

    // ── 构建 baseName/fileName → 完整路径 映射（供查找配对影像使用）──────────────
    QMap<QString, QString> baseToPath;    // normalized completeBaseName → fullPath
    for (const QString &imgPath : snapshot.allImages) {
        const QString base = imageBaseToken(imgPath);
        if (!baseToPath.contains(base)) baseToPath.insert(base, imgPath);
    }

    // 后台完整扫描用于补齐所有分片的算法变体与统计；首屏已由当前影像的单个
    // `.pimatch` 快速加载，不会等待整个目录。
    QSet<QString> seenMatchFiles;
    QSet<QString> seenPairKeys;
    if (!snapshot.matchDir.isEmpty())
    {
        xjw::aerial_triangulation::MatchResultCatalogConfig config;
        config.matchDirectory = snapshot.matchDir;
        config.targetImagePath = imagePath;
        config.targetImagePaths = snapshot.allImages;
        config.progressCallback = progressCallback;
        const xjw::aerial_triangulation::MatchResultCatalogSummary summary =
            xjw::aerial_triangulation::MatchResultCatalog(config).scan();

        for (const xjw::aerial_triangulation::MatchPairGroup &group : summary.pairGroups)
        {
            QString otherToken;
            if (!groupContainsImage(group, imagePath, &otherToken))
            {
                continue;
            }

            const QString otherImagePath =
                xjw::common::project::resolveProjectImagePathFromToken(otherToken,
                                                                       snapshot.allImages);
            if (otherImagePath.trimmed().isEmpty() ||
                imageTokensReferToSameImage(otherImagePath, imagePath))
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
                const xjw::aerial_triangulation::MatchVariant &variant = group.variants.at(selectedVariantIndex);
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
                    if (variant.tiePointMatches > 0)
                    {
                        info.hasTrackValidity = true;
                        info.validPoints = variant.tiePointMatches;
                        info.invalidPoints = std::max(0, variant.totalMatches - variant.tiePointMatches);
                        info.status = info.compatibleVariantCount > 1
                            ? tr("已对齐（%1 个算法）").arg(info.compatibleVariantCount)
                            : tr("已对齐");
                    }
                }
            }

            for (const xjw::aerial_triangulation::MatchVariant &variant : group.variants)
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

    matches.append(loadOverlapCandidatesForImageFromSnapshot(snapshot, imagePath, seenPairKeys, baseToPath));

    return matches;
}

MatchPairSelectorDialog::MatchInfoList MatchPairSelectorDialog::loadOverlapCandidatesForImageFromSnapshot(
    const MatchDataSnapshot &snapshot,
    const QString &imagePath,
    const QSet<QString> &seenPairKeys,
    const QMap<QString, QString> &baseToPath)
{
    MatchInfoList candidates;
    if (snapshot.projectPath.isEmpty())
    {
        return candidates;
    }

    const QString overlapDir = QDir(xjw::common::project::ProjectIO::projectAssetsDir(snapshot.projectPath))
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

        if (!imageTokensReferToSameImage(imageA, imagePath) &&
            !imageTokensReferToSameImage(imageB, imagePath))
        {
            return;
        }

        const QString otherToken = imageTokensReferToSameImage(imageA, imagePath) ? imageB : imageA;
        const QString otherImagePath =
            xjw::common::project::resolveProjectImagePathFromToken(otherToken,
                                                                   snapshot.allImages);
        if (otherImagePath.trimmed().isEmpty() ||
            imageTokensReferToSameImage(otherImagePath, imagePath))
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
        const QString algorithm = info.algorithm.isEmpty() ? tr("(旧格式)") : info.algorithm;
        const bool hasValidityStats = info.hasTrackValidity;
        if (hasValidityStats)
        {
            _statusLabel->setText(tr("已选择：%1（%2，原始 %3，有效连接点 %4，无效匹配 %5）")
                .arg(info.imageName)
                .arg(algorithm)
                .arg(info.totalPoints)
                .arg(info.validPoints)
                .arg(info.invalidPoints));
        }
        else
        {
            _statusLabel->setText(tr("已选择：%1（%2，总计 %3）")
                .arg(info.imageName)
                .arg(algorithm)
                .arg(info.totalPoints));
        }
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
    QVector<MatchViewerDialog::MatchPairOption> pair_options;
    pair_options.reserve(_currentMatches.size());
    for (const MatchInfo &match : _currentMatches)
    {
        MatchViewerDialog::MatchPairOption option;
        option.imageA = _currentImage;
        option.imageB = match.imagePath;
        option.matchFile = match.matchFilePath;
        option.variants = match.variants;
        pair_options.append(option);
    }
    viewer->setAvailablePairs(pair_options, _currentImage, info.imagePath);

    viewer->exec();
}

void MatchPairSelectorDialog::onRefresh()
{
    loadProjectImages();

    if (!_currentImage.isEmpty()) {
        loadMatchPairsForImage(_currentImage);
    }
}

void MatchPairSelectorDialog::scheduleRefresh()
{
    if (_refreshTimer) _refreshTimer->start();
}
