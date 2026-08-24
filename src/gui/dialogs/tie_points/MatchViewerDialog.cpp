// =============================================================================
// 文件: MatchViewerDialog.cpp
// 功能: MatchViewerDialog 的实现
// 职责:
//   - 绑定 .ui 中的工具栏、显示选项控件组、状态栏
//   - 通过 DualImageViewer 加载并展示两张影像及匹配连线
//   - 将用户操作（同步缩放、线宽、透明度等）实时转发给 MatchLineOverlay
//   - 通过 project_dialog.json 持久化显示配置（项目级）
// =============================================================================
#include "tie_points/MatchViewerDialog.h"
#include "DualImageViewer.h"        // 双图并列查看器
#include "ImageViewWidget.h"        // 单张影像可缩放/平移控件
#include "MatchLineOverlay.h"       // 匹配连线覆盖层（负责绘制连接线）
#include "settings/DialogSettingStore.h" // 项目级记忆化
#include "settings/DialogSettingKeys.h"
#include "ui_MatchViewerDialog.h"

#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QDir>
#include <QComboBox>
#include <QHBoxLayout>
#include <QSizePolicy>
#include <QSet>
#include <QSignalBlocker>

namespace
{

QString normalizedMatchPath(const QString &path)
{
    return QDir::cleanPath(path.trimmed()).toLower();
}

bool sameMatchPath(const QString &lhs, const QString &rhs)
{
    const QString left = normalizedMatchPath(lhs);
    const QString right = normalizedMatchPath(rhs);
    return !left.isEmpty() && left == right;
}

QString variantAlgorithmLabel(const xjw::aerial_triangulation::MatchVariant &variant)
{
    return xjw::aerial_triangulation::MatchResultCatalog::algorithmDisplayLabel(variant);
}

QString variantComboLabel(const xjw::aerial_triangulation::MatchVariant &variant)
{
    const QString counts = variant.hasInlierStats
        ? QStringLiteral("几何内点 %1 / 原始 %2")
              .arg(variant.geometricVerifiedInliers)
              .arg(variant.totalMatches)
        : QStringLiteral("原始 %1").arg(variant.totalMatches);
    return QStringLiteral("%1 · %2").arg(variantAlgorithmLabel(variant), counts);
}

} // namespace

// 构造函数
// imgA      — 左侧影像路径
// imgB      — 右侧影像路径
// matchFile — 当前像对任意一侧的 `.pimatch` 分片路径
// parent    — 父窗口
MatchViewerDialog::MatchViewerDialog(const QString &imgA, const QString &imgB,
                                     const QString &matchFile, QWidget *parent)
    : QDialog(parent)
    , _imageA(imgA)
    , _imageB(imgB)
    , _matchFile(matchFile)  // 保存匹配文件路径
    , _totalMatches(0)       // 初始化匹配点计数
{
    _sparseMatchFileMissing = _matchFile.trimmed().isEmpty();

    // 窗口标题显示两张影像的文件名（去除目录部分）
    setWindowTitle(tr("匹配查看：%1 <-> %2")
                   .arg(QFileInfo(imgA).fileName())
                   .arg(QFileInfo(imgB).fileName()));
    resize(1400, 800);  // 初始窗口尺寸：宽 1400px，高 800px

    Ui::MatchViewerDialog form;
    form.setupUi(this);

    _statusLabel = form.m_statusLabel;

    _syncModeChk = form.m_syncModeChk;
    _fitBtn = form.m_fitBtn;
    _resetBtn = form.m_resetBtn;
    _zoomInBtn = form.m_zoomInBtn;
    _zoomOutBtn = form.m_zoomOutBtn;

    auto *image_selection_widget = new QWidget(this);
    image_selection_widget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    auto *image_selection_layout = new QHBoxLayout(image_selection_widget);
    image_selection_layout->setContentsMargins(0, 0, 0, 0);
    image_selection_layout->setSpacing(4);
    _leftImageCombo = new QComboBox(image_selection_widget);
    _rightImageCombo = new QComboBox(image_selection_widget);
    _leftImageCombo->setObjectName(QStringLiteral("leftImageCombo"));
    _rightImageCombo->setObjectName(QStringLiteral("rightImageCombo"));
    _leftImageCombo->setMinimumWidth(120);
    _leftImageCombo->setMaximumWidth(180);
    _rightImageCombo->setMinimumWidth(120);
    _rightImageCombo->setMaximumWidth(180);
    _leftImageCombo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    _rightImageCombo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    image_selection_layout->addWidget(new QLabel(tr("左影像:"), image_selection_widget));
    image_selection_layout->addWidget(_leftImageCombo);
    image_selection_layout->addWidget(new QLabel(tr("右影像:"), image_selection_widget));
    image_selection_layout->addWidget(_rightImageCombo);
    form.toolbarLayout->insertWidget(0, image_selection_widget);

    _variantCombo = new QComboBox(this);
    _variantCombo->setToolTip(tr("选择稀疏匹配算法结果"));
    _variantCombo->setMinimumContentsLength(24);
    _variantCombo->setMinimumWidth(240);
    _variantCombo->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    _variantCombo->setVisible(false);
    int spacer_index = form.toolbarLayout->count();
    for (int index = 0; index < form.toolbarLayout->count(); ++index)
    {
        if (form.toolbarLayout->itemAt(index)->spacerItem())
        {
            spacer_index = index;
            break;
        }
    }
    form.toolbarLayout->insertWidget(spacer_index, _variantCombo);

    _lineWidthSpin = form.m_lineWidthSpin;
    _opacitySlider = form.m_opacitySlider;
    _maxCountSpin = form.m_maxCountSpin;
    _showEndPointsChk = form.m_showEndPointsChk;
    _showOnlyInliersChk = form.m_showOnlyInliersChk;

    _viewer = new DualImageViewer(this);
    form.viewerLayout->addWidget(_viewer);

    connect(_syncModeChk, &QCheckBox::toggled, this, &MatchViewerDialog::onSyncModeToggled);
    connect(_fitBtn, &QPushButton::clicked, this, &MatchViewerDialog::onFitToView);
    connect(_resetBtn, &QPushButton::clicked, this, &MatchViewerDialog::onResetView);
    connect(_zoomInBtn, &QPushButton::clicked, this, &MatchViewerDialog::onZoomIn);
    connect(_zoomOutBtn, &QPushButton::clicked, this, &MatchViewerDialog::onZoomOut);
    connect(_variantCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchViewerDialog::onVariantChanged);
    connect(_leftImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchViewerDialog::onImageSelectionChanged);
    connect(_rightImageCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchViewerDialog::onImageSelectionChanged);

    connect(_lineWidthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MatchViewerDialog::onLineWidthChanged);
    connect(_opacitySlider, &QSlider::valueChanged, this, &MatchViewerDialog::onOpacityChanged);
    connect(_maxCountSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MatchViewerDialog::onMaxCountChanged);
    connect(_showEndPointsChk, &QCheckBox::toggled, this, &MatchViewerDialog::onShowEndPointsToggled);
    connect(_showOnlyInliersChk, &QCheckBox::toggled, this, &MatchViewerDialog::onShowOnlyInliersToggled);

    // 连接 DualImageViewer 的数据加载信号到本对话框的回调槽
    connect(_viewer, &DualImageViewer::matchDataLoaded,
            this, &MatchViewerDialog::onMatchDataLoaded);
    connect(_viewer, &DualImageViewer::matchValidityLoaded,
            this, &MatchViewerDialog::onMatchValidityLoaded);
    connect(_viewer, &DualImageViewer::loadFailed,
            this, &MatchViewerDialog::onLoadFailed);
    // 从 project_dialog.json 恢复上次保存的显示参数（若已设置项目路径）
    loadSettings();
    
    MatchPairOption initial_pair;
    initial_pair.imageA = imgA;
    initial_pair.imageB = imgB;
    initial_pair.matchFile = matchFile;
    setAvailablePairs({initial_pair}, imgA, imgB);
}

// 析构函数：在对话框关闭前将当前显示参数持久化到 project_dialog.json
MatchViewerDialog::~MatchViewerDialog()
{
    saveSettings();
}

void MatchViewerDialog::setMatchVariants(const QVector<xjw::aerial_triangulation::MatchVariant> &variants,
                                         const QString &selectedMatchFile)
{
    _matchVariants = variants;
    if (!_variantCombo)
    {
        return;
    }

    int selectedComboIndex = -1;
    int selectedVariantIndex = -1;
    {
        const QSignalBlocker blocker(_variantCombo);
        _variantCombo->clear();
        for (int i = 0; i < _matchVariants.size(); ++i)
        {
            const xjw::aerial_triangulation::MatchVariant &variant = _matchVariants.at(i);
            if (!variant.compatible || variant.matchFilePath.trimmed().isEmpty())
            {
                continue;
            }

            _variantCombo->addItem(variantComboLabel(variant), i);
            const int comboIndex = _variantCombo->count() - 1;
            _variantCombo->setItemData(comboIndex, variant.matchFilePath, Qt::ToolTipRole);
            if (sameMatchPath(variant.matchFilePath, selectedMatchFile) ||
                sameMatchPath(variant.matchFilePath, _matchFile))
            {
                selectedComboIndex = comboIndex;
            }
        }

        if (selectedComboIndex < 0 && _variantCombo->count() > 0)
        {
            selectedComboIndex = 0;
        }
        if (selectedComboIndex >= 0)
        {
            _variantCombo->setCurrentIndex(selectedComboIndex);
            selectedVariantIndex = _variantCombo->itemData(selectedComboIndex).toInt();
        }
    }

    _variantCombo->setVisible(_variantCombo->count() > 1);

    if (selectedVariantIndex >= 0 && selectedVariantIndex < _matchVariants.size())
    {
        // 一个分片可以保存多个配置变体；即使文件路径相同，也必须按变体键重载。
        applyMatchVariant(_matchVariants.at(selectedVariantIndex), true);
    }
    else
    {
        _currentVariantSummary.clear();
        updateStatusBar();
    }
}

void MatchViewerDialog::setAvailablePairs(const QVector<MatchPairOption> &pairs,
                                          const QString &selectedImageA,
                                          const QString &selectedImageB)
{
    _pairOptions = pairs;
    if (findPairOption(selectedImageA, selectedImageB) < 0)
    {
        MatchPairOption selected_pair;
        selected_pair.imageA = selectedImageA;
        selected_pair.imageB = selectedImageB;
        selected_pair.matchFile = _matchFile;
        _pairOptions.prepend(selected_pair);
    }

    QStringList images;
    QSet<QString> seen_images;
    auto append_image = [&images, &seen_images](const QString &path)
    {
        const QString normalized = normalizedMatchPath(path);
        if (normalized.isEmpty() || seen_images.contains(normalized))
        {
            return;
        }
        seen_images.insert(normalized);
        images.append(path);
    };
    for (const MatchPairOption &pair : _pairOptions)
    {
        append_image(pair.imageA);
        append_image(pair.imageB);
    }

    _updatingImageSelectors = true;
    {
        const QSignalBlocker left_blocker(_leftImageCombo);
        const QSignalBlocker right_blocker(_rightImageCombo);
        _leftImageCombo->clear();
        _rightImageCombo->clear();
        for (const QString &image : images)
        {
            const QString label = QFileInfo(image).fileName();
            _leftImageCombo->addItem(label, image);
            _rightImageCombo->addItem(label, image);
            const int index = _leftImageCombo->count() - 1;
            _leftImageCombo->setItemData(index, image, Qt::ToolTipRole);
            _rightImageCombo->setItemData(index, image, Qt::ToolTipRole);
        }

        auto select_image = [](QComboBox *combo, const QString &image)
        {
            for (int index = 0; index < combo->count(); ++index)
            {
                if (sameMatchPath(combo->itemData(index).toString(), image))
                {
                    combo->setCurrentIndex(index);
                    return;
                }
            }
        };
        select_image(_leftImageCombo, selectedImageA);
        select_image(_rightImageCombo, selectedImageB);
    }
    _updatingImageSelectors = false;
    applySelectedPair();
}

int MatchViewerDialog::findPairOption(const QString &imageA, const QString &imageB) const
{
    for (int index = 0; index < _pairOptions.size(); ++index)
    {
        const MatchPairOption &pair = _pairOptions.at(index);
        if ((sameMatchPath(pair.imageA, imageA) && sameMatchPath(pair.imageB, imageB)) ||
            (sameMatchPath(pair.imageA, imageB) && sameMatchPath(pair.imageB, imageA)))
        {
            return index;
        }
    }
    return -1;
}

QString MatchViewerDialog::counterpartForImage(const QString &imagePath) const
{
    for (const MatchPairOption &pair : _pairOptions)
    {
        if (sameMatchPath(pair.imageA, imagePath))
        {
            return pair.imageB;
        }
        if (sameMatchPath(pair.imageB, imagePath))
        {
            return pair.imageA;
        }
    }
    return {};
}

void MatchViewerDialog::onImageSelectionChanged()
{
    if (_updatingImageSelectors || !_leftImageCombo || !_rightImageCombo)
    {
        return;
    }

    QString left_image = _leftImageCombo->currentData().toString();
    QString right_image = _rightImageCombo->currentData().toString();
    if (findPairOption(left_image, right_image) < 0)
    {
        QComboBox *changed_combo = qobject_cast<QComboBox *>(sender());
        _updatingImageSelectors = true;
        if (changed_combo == _leftImageCombo)
        {
            right_image = counterpartForImage(left_image);
            for (int index = 0; index < _rightImageCombo->count(); ++index)
            {
                if (sameMatchPath(_rightImageCombo->itemData(index).toString(), right_image))
                {
                    _rightImageCombo->setCurrentIndex(index);
                    break;
                }
            }
        }
        else
        {
            left_image = counterpartForImage(right_image);
            for (int index = 0; index < _leftImageCombo->count(); ++index)
            {
                if (sameMatchPath(_leftImageCombo->itemData(index).toString(), left_image))
                {
                    _leftImageCombo->setCurrentIndex(index);
                    break;
                }
            }
        }
        _updatingImageSelectors = false;
    }
    applySelectedPair();
}

void MatchViewerDialog::applySelectedPair()
{
    if (!_leftImageCombo || !_rightImageCombo)
    {
        return;
    }

    const QString image_a = _leftImageCombo->currentData().toString();
    const QString image_b = _rightImageCombo->currentData().toString();
    const int pair_index = findPairOption(image_a, image_b);
    if (pair_index < 0)
    {
        return;
    }

    const MatchPairOption &pair = _pairOptions.at(pair_index);
    _imageA = image_a;
    _imageB = image_b;
    _matchFile = pair.matchFile;
    _sparseMatchFileMissing = _matchFile.trimmed().isEmpty();
    _totalMatches = 0;
    _validMatches = -1;
    _invalidMatches = -1;
    _currentVariantSummary.clear();
    _showOnlyInliersChk->setEnabled(false);
    _showOnlyInliersChk->setChecked(false);
    setWindowTitle(tr("匹配查看：%1 <-> %2")
                       .arg(QFileInfo(_imageA).fileName(), QFileInfo(_imageB).fileName()));

    if (!pair.variants.isEmpty())
    {
        setMatchVariants(pair.variants, pair.matchFile);
        if (_variantCombo->count() == 0)
        {
            _viewer->loadMatchPair(_imageA, _imageB, _matchFile);
            updateStatusBar();
        }
    }
    else
    {
        _matchVariants.clear();
        {
            const QSignalBlocker blocker(_variantCombo);
            _variantCombo->clear();
        }
        _variantCombo->hide();
        _viewer->loadMatchPair(_imageA, _imageB, _matchFile);
        updateStatusBar();
    }
}

// setProjectPath: 设置项目路径以启用记忆化，并立即加载已保存的设置
void MatchViewerDialog::setProjectPath(const QString &plascanPath)
{
    if (plascanPath.isEmpty()) return;
    if (!_setting) _setting = new DialogSettingStore(DialogSettingKeys::MatchViewer, this);
    _setting->setProjectPath(plascanPath);
    loadSettings();
}

// onSyncModeToggled: 同步缩放/平移开关切换，转发给 DualImageViewer
void MatchViewerDialog::onSyncModeToggled(bool checked)
{
    _viewer->setSyncMode(checked);
}

// onFitToView: 将左右两张图像都缩放到适合当前视口大小
void MatchViewerDialog::onFitToView()
{
    _viewer->fitBothViews();
}

// onResetView: 将左右两张图像缩放重置为 100%（原始像素大小）
void MatchViewerDialog::onResetView()
{
    _viewer->resetBothViews();
}

// onZoomIn: 同时对左右视图执行放大操作
void MatchViewerDialog::onZoomIn()
{
    if (_viewer->leftView()) _viewer->leftView()->zoomIn();
    if (_viewer->rightView()) _viewer->rightView()->zoomIn();
}

// onZoomOut: 同时对左右视图执行缩小操作
void MatchViewerDialog::onZoomOut()
{
    if (_viewer->leftView()) _viewer->leftView()->zoomOut();
    if (_viewer->rightView()) _viewer->rightView()->zoomOut();
}

// onLineWidthChanged: 更新 MatchLineOverlay 中连线的绘制宽度
void MatchViewerDialog::onLineWidthChanged(double value)
{
    _viewer->overlay()->setLineWidth(value);
}

// onOpacityChanged: 将滑块的 0–100 整数值转换为 0.0–1.0 浮点透明度并更新覆盖层
void MatchViewerDialog::onOpacityChanged(int value)
{
    _viewer->overlay()->setOpacity(value / 100.0);
}

// onMaxCountChanged: 限制 MatchLineOverlay 最多显示的连线条数（0 = 无限制）
void MatchViewerDialog::onMaxCountChanged(int value)
{
    _viewer->overlay()->setMaxDisplayCount(value);
}

// onShowEndPointsToggled: 控制是否在连线两端绘制关键点圆圈
void MatchViewerDialog::onShowEndPointsToggled(bool checked)
{
    _viewer->overlay()->setShowEndPoints(checked);
}

// onShowOnlyInliersToggled: 控制是否仅显示内点连线（需要先加载内点标记）
void MatchViewerDialog::onShowOnlyInliersToggled(bool checked)
{
    _viewer->overlay()->setShowOnlyInliers(checked);
}

void MatchViewerDialog::onVariantChanged(int index)
{
    if (!_variantCombo || index < 0)
    {
        return;
    }

    const int variantIndex = _variantCombo->itemData(index).toInt();
    if (variantIndex < 0 || variantIndex >= _matchVariants.size())
    {
        return;
    }

    applyMatchVariant(_matchVariants.at(variantIndex), true);
}

void MatchViewerDialog::applyMatchVariant(const xjw::aerial_triangulation::MatchVariant &variant, bool forceReload)
{
    if (!variant.compatible || variant.matchFilePath.trimmed().isEmpty())
    {
        return;
    }

    const QString previousMatchFile = _matchFile;
    _matchFile = variant.matchFilePath;
    const bool willReload = _viewer && (forceReload || !sameMatchPath(previousMatchFile, _matchFile));
    _sparseMatchFileMissing = false;
    _totalMatches = variant.totalMatches;
    if (willReload)
    {
        _validMatches = -1;
        _invalidMatches = -1;
        if (_showOnlyInliersChk)
        {
            _showOnlyInliersChk->setEnabled(false);
            _showOnlyInliersChk->setChecked(false);
        }
    }
    _currentVariantSummary = variantComboLabel(variant);
    updateStatusBar();

    if (willReload)
    {
        _viewer->loadMatchPair(_imageA,
                               _imageB,
                               _matchFile,
                               variant.algorithmId,
                               variant.algorithmVersion,
                               variant.configFingerprint);
    }
}

// onMatchDataLoaded: 匹配数据加载成功的回调，更新总匹配数并刷新状态栏
void MatchViewerDialog::onMatchDataLoaded(int count)
{
    _totalMatches = count;  // 保存总匹配点数

    // 兼容旧版本保存值：历史配置可能保存 0（全部）或 500。
    // 打开大匹配文件时保持有限渲染预算，避免一次性绘制过多连线。
    if (_maxCountSpin &&
        (_maxCountSpin->value() <= 0 || _maxCountSpin->value() == 500) &&
        count > 5000)
    {
        _maxCountSpin->setValue(5000);
        if (_viewer && _viewer->overlay())
        {
            _viewer->overlay()->setMaxDisplayCount(5000);
        }
    }

    updateStatusBar();       // 立即刷新状态栏文字
}

// onLoadFailed: 匹配数据加载失败的回调，直接将错误信息显示在状态栏
void MatchViewerDialog::onLoadFailed(const QString &error)
{
    _statusLabel->setText(tr("加载失败：%1").arg(error));
}

void MatchViewerDialog::onMatchValidityLoaded(int validCount, int invalidCount)
{
    _validMatches = validCount;
    _invalidMatches = invalidCount;

    const bool hasTrackValidity = validCount >= 0 && invalidCount >= 0;
    if (_showOnlyInliersChk)
    {
        _showOnlyInliersChk->setEnabled(hasTrackValidity);
        if (!hasTrackValidity)
        {
            _showOnlyInliersChk->setChecked(false);
        }
        _showOnlyInliersChk->setToolTip(
            hasTrackValidity
                ? tr("只显示空三后保留下来的有效连接点")
                : tr("完成空中三角测量后可按有效连接点过滤"));
    }

    updateStatusBar();
}

// updateStatusBar: 刷新底部状态栏文字，显示当前总匹配点数
// 若后续需要显示可见点数、内点数等，可在此处扩展
void MatchViewerDialog::updateStatusBar()
{
    if (_sparseMatchFileMissing)
    {
        _statusLabel->setText(tr("尚未生成匹配：当前仅显示重叠候选影像对"));
        return;
    }

    QString status = tr("总匹配点数：%1").arg(_totalMatches);
    if (_validMatches >= 0 && _invalidMatches >= 0)
    {
        status += tr(" | 有效连接点：%1 | 无效匹配：%2").arg(_validMatches).arg(_invalidMatches);
    }
    if (!_currentVariantSummary.isEmpty())
    {
        status += tr(" | 算法：%1").arg(_currentVariantSummary);
    }
    
    // 可以添加更多统计信息（如可见点数）：
    // int visible = _viewer->visibleMatchCount();
    // if (visible >= 0) {
    //     status += tr(" | 可见：%1").arg(visible);
    // }
    
    _statusLabel->setText(status);
}

// loadSettings: 从 project_dialog.json 恢复上次保存的显示参数（项目级记忆化）
// 若记忆化管理器未初始化或无已保存数据，则使用控件默认值
void MatchViewerDialog::loadSettings()
{
    if (!_setting) return;
    const QJsonObject cfg = _setting->load();
    if (cfg.isEmpty()) return;

    // 逐项恢复显示选项
    bool syncMode = cfg.value(QStringLiteral("syncMode")).toBool(false);
    _syncModeChk->setChecked(syncMode);
    
    double lineWidth = cfg.value(QStringLiteral("lineWidth")).toDouble(1.5);
    _lineWidthSpin->setValue(lineWidth);
    
    int opacity = cfg.value(QStringLiteral("opacity")).toInt(70);
    _opacitySlider->setValue(opacity);
    
    int maxCount = cfg.value(QStringLiteral("maxCount")).toInt(5000);
    // 兼容旧版本默认值：历史默认 500 或全部(0) 都会导致大匹配文件打开时卡顿。
    if (maxCount <= 0 || maxCount == 500)
    {
        maxCount = 5000;
    }
    _maxCountSpin->setValue(maxCount);
    
    bool showEndPoints = cfg.value(QStringLiteral("showEndPoints")).toBool(true);
    _showEndPointsChk->setChecked(showEndPoints);
}

// saveSettings: 将当前界面上的显示参数持久化到 project_dialog.json（析构时调用）
void MatchViewerDialog::saveSettings()
{
    if (!_setting) return;

    QJsonObject cfg;
    cfg[QStringLiteral("syncMode")]      = _syncModeChk->isChecked();
    cfg[QStringLiteral("lineWidth")]     = _lineWidthSpin->value();
    cfg[QStringLiteral("opacity")]       = _opacitySlider->value();
    cfg[QStringLiteral("maxCount")]      = _maxCountSpin->value();
    cfg[QStringLiteral("showEndPoints")] = _showEndPointsChk->isChecked();

    _setting->save(cfg);
}
