#include "FeatureMatchingDialog.h"
#include "ui_FeatureMatchingDialog.h"
#include "FeaturePairPlanner.h"
#include "MatchViewerDialog.h"

#include <QAbstractItemView>
#include <QListWidget>
#include <QPushButton>
#include <QJsonObject>
#include <QJsonArray>
#include <QLineEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QTextEdit>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QLabel>
#include <QStackedWidget>
#include <QToolButton>
#include <QMessageBox>
#include <QFileInfo>
#include <QRegularExpression>

#include "AlgorithmCompat.h"

namespace
{

void setComboDataOrFirst(QComboBox *combo, const QString &data)
{
    if (!combo)
    {
        return;
    }

    const int index = combo->findData(data);
    combo->setCurrentIndex(index >= 0 ? index : 0);
}

QString normalizeFeatureSuffix(QString suffix)
{
    suffix = suffix.trimmed().toLower();
    if (suffix.isEmpty())
    {
        return QString();
    }
    if (!suffix.startsWith(QLatin1Char('.')))
    {
        suffix.prepend(QLatin1Char('.'));
    }
    return suffix;
}

QStringList normalizeFeatureSuffixes(const QStringList &suffixes)
{
    QStringList normalizedSuffixes;
    for (const QString &suffix : suffixes)
    {
        const QString normalized = normalizeFeatureSuffix(suffix);
        if (!normalized.isEmpty() && !normalizedSuffixes.contains(normalized))
        {
            normalizedSuffixes.append(normalized);
        }
    }
    return normalizedSuffixes;
}

} // namespace

FeatureMatchingDialog::FeatureMatchingDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    setupConnections();
}

FeatureMatchingDialog::~FeatureMatchingDialog() = default;

void FeatureMatchingDialog::setupUi() 
{
    setWindowTitle(tr("特征匹配"));
    resize(1000, 700);

    Ui::FeatureMatchingDialog ui;
    ui.setupUi(this);

    m_imageInputWidget = ui.m_imageInputWidget;
    m_selectAllBtn = ui.m_selectAllBtn;
    m_deselectAllBtn = ui.m_deselectAllBtn;
    m_imageList = ui.m_imageList;
    m_pairPreview = ui.m_pairPreview;
    m_lisFileLine = ui.m_lisFileLine;
    m_addLisBtn = ui.m_addLisBtn;
    m_clearLisBtn = ui.m_clearLisBtn;
    m_generatePairsBtn = ui.m_generatePairsBtn;
    m_outputLine = ui.m_outputLine;
    m_browseOutBtn = ui.m_browseOutBtn;

    m_matchAlgorithmCombo = ui.m_matchAlgorithmCombo;
    m_featureSuffixLabel = ui.m_featureSuffixLabel;
    m_featureSuffixCombo = ui.m_featureSuffixCombo;
    m_maxKeypointsSpin = ui.m_maxKeypointsSpin;
    m_outlierMethodCombo = ui.m_outlierMethodCombo;
    m_paramStack = ui.m_paramStack;

    m_modelTypeCombo = ui.m_modelTypeCombo;
    m_matchThresholdSpin = ui.m_matchThresholdSpin;
    m_sinkhornIterSpin = ui.m_sinkhornIterSpin;
    m_batchSizeSpin = ui.m_batchSizeSpin;
    m_inputWidthSpin = ui.m_inputWidthSpin;
    m_inputHeightSpin = ui.m_inputHeightSpin;

    m_lgMatchThresholdSpin = ui.m_lgMatchThresholdSpin;
    m_lgBatchSizeSpin = ui.m_lgBatchSizeSpin;
    m_lgInputWidthSpin = ui.m_lgInputWidthSpin;
    m_lgInputHeightSpin = ui.m_lgInputHeightSpin;

    m_loftrModelTypeCombo = ui.m_loftrModelTypeCombo;
    m_loftrMatchThresholdSpin = ui.m_loftrMatchThresholdSpin;
    m_romaModelTypeCombo = ui.m_romaModelTypeCombo;
    m_romaMatchThresholdSpin = ui.m_romaMatchThresholdSpin;
    m_romaMaxKeypointsSpin = ui.m_romaMaxKeypointsSpin;

    m_advancedGroup = ui.m_advancedGroup;
    m_outlierReprojSpin = ui.m_outlierReprojSpin;
    m_outlierConfidenceSpin = ui.m_outlierConfidenceSpin;
    m_outlierMaxItersSpin = ui.m_outlierMaxItersSpin;
    m_outlierMinInliersSpin = ui.m_outlierMinInliersSpin;

    m_systemGroup = ui.m_systemGroup;
    m_deviceCombo = ui.m_deviceCombo;
    m_numThreadsSpin = ui.m_numThreadsSpin;
    m_cudaParallelSpin = ui.m_cudaParallelSpin;

    m_debugGroup = ui.m_debugGroup;
    m_saveCsvChk = ui.m_saveCsvChk;
    m_saveVisChk = ui.m_saveVisChk;
    m_verboseChk = ui.m_verboseChk;

    m_resetBtn = ui.m_resetBtn;
    m_viewMatchesBtn = ui.m_viewMatchesBtn;
    m_runBtn = ui.m_runBtn;
    m_cancelBtn = ui.m_cancelBtn;

    if (ui.topSplit)
    {
        ui.topSplit->setStretchFactor(0, 3);
        ui.topSplit->setStretchFactor(1, 2);
    }

    m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    m_matchAlgorithmCombo->clear();
    m_matchAlgorithmCombo->addItem(tr("SuperGlue"), "superglue");
    m_matchAlgorithmCombo->addItem(tr("LightGlue"), "lightglue");
    m_matchAlgorithmCombo->addItem(tr("LoFTR"), "loftr");
    m_matchAlgorithmCombo->addItem(tr("RoMa"), "roma");
    m_matchAlgorithmCombo->addItem(tr("BF-Hamming (ORB)"), "orb_bf_hamming");
    m_matchAlgorithmCombo->addItem(tr("BF-L2 (SIFT)"), "sift_bf_l2");
    m_matchAlgorithmCombo->addItem(tr("FLANN (SIFT)"), "sift_flann");
    setComboDataOrFirst(m_matchAlgorithmCombo, QStringLiteral("lightglue"));

    m_outlierMethodCombo->clear();
    m_outlierMethodCombo->addItem(tr("不剔除"), "none");
    m_outlierMethodCombo->addItem(tr("Fundamental RANSAC"), "fundamental");
    m_outlierMethodCombo->addItem(tr("Fundamental USAC_MAGSAC （推荐）"), "fundamental_usac_magsac");
    m_outlierMethodCombo->addItem(tr("Homography RANSAC"), "homography");
    m_outlierMethodCombo->addItem(tr("Affine RANSAC"), "affine");
    m_outlierMethodCombo->setCurrentIndex(2);

    m_modelTypeCombo->clear();
    m_modelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
    m_modelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");

    m_loftrModelTypeCombo->clear();
    m_loftrModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
    m_loftrModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");

    m_romaModelTypeCombo->clear();
    m_romaModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
    m_romaModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");

    m_deviceCombo->clear();
    m_deviceCombo->addItems({tr("CUDA（如可用）"), tr("CPU")});

    connect(ui.advancedToggle, &QToolButton::toggled, this, [this, advancedToggle = ui.advancedToggle](bool checked)
    {
        m_advancedGroup->setVisible(checked);
        advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.systemToggle, &QToolButton::toggled, this, [this, systemToggle = ui.systemToggle](bool checked)
    {
        m_systemGroup->setVisible(checked);
        systemToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.debugToggle, &QToolButton::toggled, this, [this, debugToggle = ui.debugToggle](bool checked)
    {
        m_debugGroup->setVisible(checked);
        debugToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });
}

void FeatureMatchingDialog::setupConnections() 
{
    // 全选/清除按钮
    connect(m_selectAllBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onSelectAll);
    connect(m_deselectAllBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onDeselectAll);
    // 文件操作
    connect(m_addLisBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onAddLisFile);
    connect(m_clearLisBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onClearLis);
    connect(m_generatePairsBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onGeneratePairs);
    connect(m_browseOutBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onBrowseOutput);
    
    // 底部按钮
    connect(m_runBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onRun);
    connect(m_viewMatchesBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onViewMatches);
    connect(m_cancelBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onCancel);
    connect(m_resetBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onResetDefaults);
    
    // 算法切换：更新参数控件启用状态 + 持久化
    connect(m_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int){ onAlgorithmOrFeatureChanged(); });
    connect(m_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 特征类型切换
    connect(m_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int){ updatePreview(); });
    connect(m_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 初始状态
    onAlgorithmOrFeatureChanged();
    connect(m_modelTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_matchThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierReprojSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierConfidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierMaxItersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_outlierMinInliersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_inputWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_inputHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_cudaParallelSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_sinkhornIterSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_batchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // LightGlue 面板控件
    connect(m_lgMatchThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lgBatchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lgInputWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lgInputHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 影像列表 checkbox 变化时保存
    connect(m_imageList, &QListWidget::itemChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 输出路径、lis 文件改变时保存
    connect(m_outputLine, &QLineEdit::textChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(m_lisFileLine, &QLineEdit::textChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
}

void FeatureMatchingDialog::onSelectAll()
{
    for (int i = 0; i < m_imageList->count(); ++i) {
        m_imageList->item(i)->setCheckState(Qt::Checked);
    }
    emitSettingsNow();
}

void FeatureMatchingDialog::onDeselectAll()
{
    for (int i = 0; i < m_imageList->count(); ++i) {
        m_imageList->item(i)->setCheckState(Qt::Unchecked);
    }
    emitSettingsNow();
}

void FeatureMatchingDialog::onAddLisFile()
{
    QString path = QFileDialog::getOpenFileName(this, tr("选择lis文件"),
                                                QString(),
                                                tr("lis文件 (*.lis *.txt);;所有文件 (*.*)"));
    if (!path.isEmpty()) {
        m_lisFileLine->setText(path);
        updatePreview();
    }
}

void FeatureMatchingDialog::onClearLis()
{
    m_lisFileLine->clear();
    updatePreview();
}

void FeatureMatchingDialog::onGeneratePairs()
{
    QString lisPath = m_lisFileLine->text().trimmed();

    if (!lisPath.isEmpty()) {
        m_currentPairs = parseLisFile(lisPath);
    } else {
        m_currentPairs = generateAllPairs();
    }

    updatePreview();
    emitSettingsNow();
}

void FeatureMatchingDialog::onBrowseOutput()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("选择输出目录"));
    if (!dir.isEmpty()) {
        m_outputLine->setText(dir);
    }
}

void FeatureMatchingDialog::onRun()
{
    if (m_currentPairs.isEmpty()) {
        QMessageBox::warning(this, tr("警告"), 
                           tr("请先生成匹配对列表"));
        return;
    }
    
    QJsonObject config = collectSettings();
    emit runRequested(config, m_currentPairs);
    accept();
}

void FeatureMatchingDialog::onCancel()
{
    reject();
}

void FeatureMatchingDialog::onResetDefaults()
{
    setComboDataOrFirst(m_matchAlgorithmCombo, QStringLiteral("lightglue"));
    m_modelTypeCombo->setCurrentIndex(0);  // outdoor
    m_outlierMethodCombo->setCurrentIndex(2);  // Fundamental USAC_MAGSAC（推荐默认，最优粗差剔除）
    m_matchThresholdSpin->setValue(0.15);
    m_maxKeypointsSpin->setValue(-1);
    m_sinkhornIterSpin->setValue(150);
    m_batchSizeSpin->setValue(1);
    m_outlierReprojSpin->setValue(1.5);
    m_outlierConfidenceSpin->setValue(0.9999);
    m_outlierMaxItersSpin->setValue(10000);
    m_outlierMinInliersSpin->setValue(25);
    m_inputWidthSpin->setValue(-1);
    m_inputHeightSpin->setValue(-1);
    m_lgMatchThresholdSpin->setValue(0.15);
    m_lgBatchSizeSpin->setValue(1);
    m_lgInputWidthSpin->setValue(-1);
    m_lgInputHeightSpin->setValue(-1);
    m_deviceCombo->setCurrentIndex(0);
    m_numThreadsSpin->setValue(-1);
    m_cudaParallelSpin->setValue(1);
    m_saveCsvChk->setChecked(false);
    m_saveVisChk->setChecked(false);
    m_verboseChk->setChecked(false);

    emitSettingsNow();
}

void FeatureMatchingDialog::onAlgorithmChanged(int)
{
    onAlgorithmOrFeatureChanged();
}

void FeatureMatchingDialog::setAvailableFeatureSuffixes(const QStringList &suffixes)
{
    m_projectFeatureSuffixes = normalizeFeatureSuffixes(suffixes);
    refreshFeatureSuffixChoices();
}

void FeatureMatchingDialog::refreshFeatureSuffixChoices()
{
    const QString previousSuffix = selectedFeatureSuffix();
    const QString algo = m_matchAlgorithmCombo->currentData().toString();
    QStringList suffixes;
    if (!xjw::feature_match::isEndToEndAlgorithm(algo))
    {
        const QStringList compatibleSuffixes = xjw::feature_match::compatibleFeatureSuffixes(algo);
        if (m_projectFeatureSuffixes.isEmpty())
        {
            suffixes = compatibleSuffixes;
        }
        else
        {
            for (const QString &suffix : compatibleSuffixes)
            {
                if (m_projectFeatureSuffixes.contains(suffix))
                {
                    suffixes.append(suffix);
                }
            }
        }
    }

    m_featureSuffixCombo->blockSignals(true);
    m_featureSuffixCombo->clear();
    if (suffixes.size() > 1)
    {
        m_featureSuffixCombo->addItem(tr("所有特征类型"), QStringLiteral("__all__"));
    }
    for (const auto &s : suffixes)
    {
        m_featureSuffixCombo->addItem(s, s);
    }

    int restoreIndex = m_featureSuffixCombo->findData(previousSuffix);
    if (restoreIndex < 0 && previousSuffix == QStringLiteral("__all__") && suffixes.size() > 1)
    {
        restoreIndex = m_featureSuffixCombo->findData(QStringLiteral("__all__"));
    }
    if (restoreIndex < 0 && m_featureSuffixCombo->count() > 0)
    {
        restoreIndex = 0;
    }
    if (restoreIndex >= 0)
    {
        m_featureSuffixCombo->setCurrentIndex(restoreIndex);
    }
    m_featureSuffixCombo->blockSignals(false);

    const bool visible = !suffixes.isEmpty();
    m_featureSuffixLabel->setVisible(visible);
    m_featureSuffixCombo->setVisible(visible);
}

QString FeatureMatchingDialog::selectedFeatureSuffix() const
{
    QVariant data = m_featureSuffixCombo->currentData();
    return data.isValid() ? data.toString() : m_featureSuffixCombo->currentText();
}

void FeatureMatchingDialog::onAlgorithmOrFeatureChanged()
{
    const QString algo = m_matchAlgorithmCombo->currentData().toString();

    // 算法参数面板切换
    if (algo == "superglue")
        m_paramStack->setCurrentIndex(0);
    else if (algo == "lightglue")
        m_paramStack->setCurrentIndex(1);
    else if (algo == "loftr")
        m_paramStack->setCurrentIndex(3);
    else if (algo == "roma")
        m_paramStack->setCurrentIndex(4);
    else
        m_paramStack->setCurrentIndex(2);

    // 更新特征后缀选择器
    refreshFeatureSuffixChoices();

    updatePreview();
}

void FeatureMatchingDialog::onViewMatches()
{
    emit viewMatchesRequested();
}

void FeatureMatchingDialog::applySettings(const QJsonObject &settings)
{
    bool block = blockSignals(true);

    const QString matchAlgorithm = settings.value("match_algorithm").toString("lightglue");
    setComboDataOrFirst(m_matchAlgorithmCombo, matchAlgorithm);

    const QString outlierMethod = settings.value("outlier_method").toString("fundamental_usac_magsac");
    const int outlierIdx = m_outlierMethodCombo->findData(outlierMethod);
    if (outlierIdx >= 0) m_outlierMethodCombo->setCurrentIndex(outlierIdx);

    m_maxKeypointsSpin->setValue(settings.value("max_keypoints").toInt(-1));

    // SuperGlue 面板
    const QString modelType = settings.value("model_type").toString("outdoor");
    const int modelIdx = m_modelTypeCombo->findData(modelType);
    if (modelIdx >= 0) m_modelTypeCombo->setCurrentIndex(modelIdx);
    m_matchThresholdSpin->setValue(settings.value("match_threshold").toDouble(0.15));
    m_sinkhornIterSpin->setValue(settings.value("sinkhorn_iterations").toInt(150));
    m_batchSizeSpin->setValue(settings.value("batch_size").toInt(1));
    m_inputWidthSpin->setValue(settings.value("input_width").toInt(-1));
    m_inputHeightSpin->setValue(settings.value("input_height").toInt(-1));

    // LightGlue 面板（独立 key，避免与 SuperGlue 冲突）
    m_lgMatchThresholdSpin->setValue(settings.value("lg_match_threshold").toDouble(
        settings.value("match_threshold").toDouble(0.15)));  // 兼容旧格式
    m_lgBatchSizeSpin->setValue(settings.value("lg_batch_size").toInt(
        settings.value("batch_size").toInt(1)));
    m_lgInputWidthSpin->setValue(settings.value("lg_input_width").toInt(
        settings.value("input_width").toInt(-1)));
    m_lgInputHeightSpin->setValue(settings.value("lg_input_height").toInt(
        settings.value("input_height").toInt(-1)));

    // LoFTR 面板
    if (m_loftrModelTypeCombo) {
        int idx = m_loftrModelTypeCombo->findData(settings.value("loftr_model_type").toString("outdoor"));
        m_loftrModelTypeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    if (m_loftrMatchThresholdSpin)
        m_loftrMatchThresholdSpin->setValue(settings.value("loftr_match_threshold").toDouble(0.2));

    // RoMa 面板
    if (m_romaModelTypeCombo) {
        int idx = m_romaModelTypeCombo->findData(settings.value("roma_model_type").toString("outdoor"));
        m_romaModelTypeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    if (m_romaMatchThresholdSpin)
        m_romaMatchThresholdSpin->setValue(settings.value("roma_match_threshold").toDouble(0.05));
    if (m_romaMaxKeypointsSpin)
        m_romaMaxKeypointsSpin->setValue(settings.value("roma_max_keypoints").toInt(10000));

    // 高级参数
    m_outlierReprojSpin->setValue(settings.value("outlier_reproj_threshold").toDouble(1.5));
    m_outlierConfidenceSpin->setValue(settings.value("outlier_confidence").toDouble(0.9999));
    m_outlierMaxItersSpin->setValue(settings.value("outlier_max_iters").toInt(10000));
    m_outlierMinInliersSpin->setValue(settings.value("outlier_min_inliers").toInt(25));

    // 系统参数
    m_deviceCombo->setCurrentIndex(settings.value("use_cuda").toBool(true) ? 0 : 1);
    m_numThreadsSpin->setValue(settings.value("num_threads").toInt(-1));
    m_cudaParallelSpin->setValue(settings.value("cuda_parallel_pairs").toInt(1));

    // 调试参数
    m_saveCsvChk->setChecked(settings.value("save_csv").toBool(false));
    m_saveVisChk->setChecked(settings.value("save_visualization").toBool(false));
    m_verboseChk->setChecked(settings.value("verbose").toBool(false));

    m_outputLine->setText(settings.value("output_dir").toString());
    m_lisFileLine->setText(settings.value("lis_file").toString());

    // 恢复已生成的匹配对
    const QJsonArray savedPairs = settings.value(QStringLiteral("generated_pairs")).toArray();
    if (!savedPairs.isEmpty()) {
        m_currentPairs.clear();
        m_currentPairs.reserve(savedPairs.size());
        for (const QJsonValue &v : savedPairs)
            m_currentPairs.append(v.toString());
        updatePreview();
    }

    blockSignals(block);
    onAlgorithmOrFeatureChanged();

    // 恢复特征后缀选择（必须在 onAlgorithmOrFeatureChanged 之后，因为该函数会填充后缀列表）
    const QString featureSuffix = settings.value("feature_suffix").toString();
    if (!featureSuffix.isEmpty() && m_featureSuffixCombo->count() > 0) {
        int idx = m_featureSuffixCombo->findData(featureSuffix);
        if (idx >= 0)
            m_featureSuffixCombo->setCurrentIndex(idx);
    }
}

void FeatureMatchingDialog::setProjectImages(const QStringList &imagePaths)
{
    m_imageList->clear();
    for (const QString &path : imagePaths) {
        QFileInfo fi(path);
        QListWidgetItem *item = new QListWidgetItem(fi.fileName());
        item->setData(Qt::UserRole, path);
        item->setCheckState(Qt::Checked);
        m_imageList->addItem(item);
    }
}

void FeatureMatchingDialog::updatePreview()
{
    QString preview;
    preview += tr("匹配对数量: %1\n\n").arg(m_currentPairs.size());
    
    int maxShow = 20;
    for (int i = 0; i < qMin(m_currentPairs.size(), maxShow); ++i) {
        preview += m_currentPairs[i] + "\n";
    }
    
    if (m_currentPairs.size() > maxShow) {
        preview += tr("... （还有 %1 对）\n").arg(m_currentPairs.size() - maxShow);
    }
    
    m_pairPreview->setPlainText(preview);
}

void FeatureMatchingDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

QJsonObject FeatureMatchingDialog::collectSettings() const
{
    QJsonObject obj;

    // 基础参数
    const QString algo = m_matchAlgorithmCombo->currentData().toString();
    obj["match_algorithm"] = algo;
    obj["feature_suffix"] = selectedFeatureSuffix();
    obj["outlier_method"] = m_outlierMethodCombo->currentData().toString();
    obj["max_keypoints"] = m_maxKeypointsSpin->value();

    // 算法专属参数
    if (algo == "superglue") {
        obj["model_type"] = m_modelTypeCombo->currentData().toString();
        obj["match_threshold"] = m_matchThresholdSpin->value();
        obj["sinkhorn_iterations"] = m_sinkhornIterSpin->value();
        obj["batch_size"] = m_batchSizeSpin->value();
        obj["input_width"] = m_inputWidthSpin->value();
        obj["input_height"] = m_inputHeightSpin->value();
    } else if (algo == "lightglue") {
        obj["model_type"] = "outdoor";
        obj["lg_match_threshold"] = m_lgMatchThresholdSpin->value();
        obj["lg_batch_size"] = m_lgBatchSizeSpin->value();
        obj["lg_input_width"] = m_lgInputWidthSpin->value();
        obj["lg_input_height"] = m_lgInputHeightSpin->value();
    } else if (algo == "loftr") {
        obj["loftr_model_type"] = m_loftrModelTypeCombo->currentData().toString();
        obj["loftr_match_threshold"] = m_loftrMatchThresholdSpin->value();
    } else if (algo == "roma") {
        obj["roma_model_type"] = m_romaModelTypeCombo->currentData().toString();
        obj["roma_match_threshold"] = m_romaMatchThresholdSpin->value();
        obj["roma_max_keypoints"] = m_romaMaxKeypointsSpin->value();
    } else {
        // 传统算法无这些参数，填充默认值
        obj["model_type"] = "outdoor";
        obj["match_threshold"] = 0.15;
        obj["sinkhorn_iterations"] = 0;
        obj["batch_size"] = 1;
        obj["input_width"] = -1;
        obj["input_height"] = -1;
    }

    // 高级参数（RANSAC）
    obj["outlier_reproj_threshold"] = m_outlierReprojSpin->value();
    obj["outlier_confidence"] = m_outlierConfidenceSpin->value();
    obj["outlier_max_iters"] = m_outlierMaxItersSpin->value();
    obj["outlier_min_inliers"] = m_outlierMinInliersSpin->value();

    // 系统参数
    obj["use_cuda"] = (m_deviceCombo->currentIndex() == 0);
    obj["num_threads"] = m_numThreadsSpin->value();
    obj["cuda_parallel_pairs"] = m_cudaParallelSpin->value();

    // 调试参数
    obj["save_csv"] = m_saveCsvChk->isChecked();
    obj["save_visualization"] = m_saveVisChk->isChecked();
    obj["verbose"] = m_verboseChk->isChecked();

    // 输出设置
    obj["output_dir"] = m_outputLine->text();
    obj["lis_file"] = m_lisFileLine->text();

    // 保存已生成的匹配对列表
    QJsonArray pairsArr;
    for (const QString &p : m_currentPairs)
        pairsArr.append(p);
    obj[QStringLiteral("generated_pairs")] = pairsArr;

    return obj;
}

QStringList FeatureMatchingDialog::parseLisFile(const QString &lisPath) const
{
    QStringList pairs;
    QFile file(lisPath);
    
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(const_cast<FeatureMatchingDialog*>(this), tr("错误"),
                           tr("无法打开lis文件: %1").arg(lisPath));
        return pairs;
    }
    
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#")) continue;
        
        QStringList parts = line.split(QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts);
        if (parts.size() >= 2) {
            QString img1 = QFileInfo(parts[0]).fileName();
            QString img2 = QFileInfo(parts[1]).fileName();
            pairs.append(QString("%1__%2").arg(img1, img2));
        }
    }
    
    file.close();
    return pairs;
}

QStringList FeatureMatchingDialog::generateAllPairs() const
{
    QStringList pairs;
    QStringList selected;

    // 收集选中的影像
    for (int i = 0; i < m_imageList->count(); ++i) {
        QListWidgetItem *item = m_imageList->item(i);
        if (item->checkState() == Qt::Checked) {
            QString baseName = QFileInfo(item->data(Qt::UserRole).toString()).completeBaseName();
            selected.append(baseName);
        }
    }

    return xjw::gui::planFeatureMatchPairs(selected);
}
