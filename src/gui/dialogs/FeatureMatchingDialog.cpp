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

    _imageInputWidget = ui.m_imageInputWidget;
    _selectAllBtn = ui.m_selectAllBtn;
    _deselectAllBtn = ui.m_deselectAllBtn;
    _imageList = ui.m_imageList;
    _pairPreview = ui.m_pairPreview;
    _lisFileLine = ui.m_lisFileLine;
    _addLisBtn = ui.m_addLisBtn;
    _clearLisBtn = ui.m_clearLisBtn;
    _generatePairsBtn = ui.m_generatePairsBtn;
    _outputLine = ui.m_outputLine;
    _browseOutBtn = ui.m_browseOutBtn;

    _matchAlgorithmCombo = ui.m_matchAlgorithmCombo;
    _featureSuffixLabel = ui.m_featureSuffixLabel;
    _featureSuffixCombo = ui.m_featureSuffixCombo;
    _maxKeypointsSpin = ui.m_maxKeypointsSpin;
    _outlierMethodCombo = ui.m_outlierMethodCombo;
    _paramStack = ui.m_paramStack;

    _modelTypeCombo = ui.m_modelTypeCombo;
    _matchThresholdSpin = ui.m_matchThresholdSpin;
    _sinkhornIterSpin = ui.m_sinkhornIterSpin;
    _batchSizeSpin = ui.m_batchSizeSpin;
    _inputWidthSpin = ui.m_inputWidthSpin;
    _inputHeightSpin = ui.m_inputHeightSpin;

    _lgMatchThresholdSpin = ui.m_lgMatchThresholdSpin;
    _lgBatchSizeSpin = ui.m_lgBatchSizeSpin;
    _lgInputWidthSpin = ui.m_lgInputWidthSpin;
    _lgInputHeightSpin = ui.m_lgInputHeightSpin;

    _loftrModelTypeCombo = ui.m_loftrModelTypeCombo;
    _loftrMatchThresholdSpin = ui.m_loftrMatchThresholdSpin;
    _romaModelTypeCombo = ui.m_romaModelTypeCombo;
    _romaMatchThresholdSpin = ui.m_romaMatchThresholdSpin;
    _romaMaxKeypointsSpin = ui.m_romaMaxKeypointsSpin;

    _advancedGroup = ui.m_advancedGroup;
    _outlierReprojSpin = ui.m_outlierReprojSpin;
    _outlierConfidenceSpin = ui.m_outlierConfidenceSpin;
    _outlierMaxItersSpin = ui.m_outlierMaxItersSpin;
    _outlierMinInliersSpin = ui.m_outlierMinInliersSpin;

    _systemGroup = ui.m_systemGroup;
    _deviceCombo = ui.m_deviceCombo;
    _numThreadsSpin = ui.m_numThreadsSpin;
    _cudaParallelSpin = ui.m_cudaParallelSpin;

    _debugGroup = ui.m_debugGroup;
    _saveCsvChk = ui.m_saveCsvChk;
    _saveVisChk = ui.m_saveVisChk;
    _verboseChk = ui.m_verboseChk;

    _resetBtn = ui.m_resetBtn;
    _viewMatchesBtn = ui.m_viewMatchesBtn;
    _runBtn = ui.m_runBtn;
    _cancelBtn = ui.m_cancelBtn;

    if (ui.topSplit)
    {
        ui.topSplit->setStretchFactor(0, 3);
        ui.topSplit->setStretchFactor(1, 2);
    }

    _imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);

    _matchAlgorithmCombo->clear();
    _matchAlgorithmCombo->addItem(tr("SuperGlue"), "superglue");
    _matchAlgorithmCombo->addItem(tr("LightGlue"), "lightglue");
    _matchAlgorithmCombo->addItem(tr("LoFTR"), "loftr");
    _matchAlgorithmCombo->addItem(tr("RoMa"), "roma");
    _matchAlgorithmCombo->addItem(tr("BF-Hamming (ORB)"), "orb_bf_hamming");
    _matchAlgorithmCombo->addItem(tr("BF-L2 (SIFT)"), "sift_bf_l2");
    _matchAlgorithmCombo->addItem(tr("FLANN (SIFT)"), "sift_flann");
    setComboDataOrFirst(_matchAlgorithmCombo, QStringLiteral("lightglue"));

    _outlierMethodCombo->clear();
    _outlierMethodCombo->addItem(tr("不剔除"), "none");
    _outlierMethodCombo->addItem(tr("Fundamental RANSAC"), "fundamental");
    _outlierMethodCombo->addItem(tr("Fundamental USAC_MAGSAC （推荐）"), "fundamental_usac_magsac");
    _outlierMethodCombo->addItem(tr("Homography RANSAC"), "homography");
    _outlierMethodCombo->addItem(tr("Affine RANSAC"), "affine");
    _outlierMethodCombo->setCurrentIndex(2);

    _modelTypeCombo->clear();
    _modelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
    _modelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");

    _loftrModelTypeCombo->clear();
    _loftrModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
    _loftrModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");

    _romaModelTypeCombo->clear();
    _romaModelTypeCombo->addItem(tr("Outdoor （室外/航空）"), "outdoor");
    _romaModelTypeCombo->addItem(tr("Indoor （室内/近景）"), "indoor");

    _deviceCombo->clear();
    _deviceCombo->addItems({tr("CUDA（如可用）"), tr("CPU")});

    connect(ui.advancedToggle, &QToolButton::toggled, this, [this, advancedToggle = ui.advancedToggle](bool checked)
    {
        _advancedGroup->setVisible(checked);
        advancedToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.systemToggle, &QToolButton::toggled, this, [this, systemToggle = ui.systemToggle](bool checked)
    {
        _systemGroup->setVisible(checked);
        systemToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });

    connect(ui.debugToggle, &QToolButton::toggled, this, [this, debugToggle = ui.debugToggle](bool checked)
    {
        _debugGroup->setVisible(checked);
        debugToggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });
}

void FeatureMatchingDialog::setupConnections() 
{
    // 全选/清除按钮
    connect(_selectAllBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onSelectAll);
    connect(_deselectAllBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onDeselectAll);
    // 文件操作
    connect(_addLisBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onAddLisFile);
    connect(_clearLisBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onClearLis);
    connect(_generatePairsBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onGeneratePairs);
    connect(_browseOutBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onBrowseOutput);
    
    // 底部按钮
    connect(_runBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onRun);
    connect(_viewMatchesBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onViewMatches);
    connect(_cancelBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onCancel);
    connect(_resetBtn, &QPushButton::clicked, this, &FeatureMatchingDialog::onResetDefaults);
    
    // 算法切换：更新参数控件启用状态 + 持久化
    connect(_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int){ onAlgorithmOrFeatureChanged(); });
    connect(_matchAlgorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 特征类型切换
    connect(_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int){ updatePreview(); });
    connect(_featureSuffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 初始状态
    onAlgorithmOrFeatureChanged();
    connect(_modelTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_outlierMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_matchThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_maxKeypointsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_deviceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_outlierReprojSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_outlierConfidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_outlierMaxItersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_outlierMinInliersSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_inputWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_inputHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
        this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_cudaParallelSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_sinkhornIterSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_batchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // LightGlue 面板控件
    connect(_lgMatchThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_lgBatchSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_lgInputWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_lgInputHeightSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 影像列表 checkbox 变化时保存
    connect(_imageList, &QListWidget::itemChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
    // 输出路径、lis 文件改变时保存
    connect(_outputLine, &QLineEdit::textChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
    connect(_lisFileLine, &QLineEdit::textChanged,
            this, &FeatureMatchingDialog::emitSettingsNow);
}

void FeatureMatchingDialog::onSelectAll()
{
    for (int i = 0; i < _imageList->count(); ++i) {
        _imageList->item(i)->setCheckState(Qt::Checked);
    }
    emitSettingsNow();
}

void FeatureMatchingDialog::onDeselectAll()
{
    for (int i = 0; i < _imageList->count(); ++i) {
        _imageList->item(i)->setCheckState(Qt::Unchecked);
    }
    emitSettingsNow();
}

void FeatureMatchingDialog::onAddLisFile()
{
    QString path = QFileDialog::getOpenFileName(this, tr("选择lis文件"),
                                                QString(),
                                                tr("lis文件 (*.lis *.txt);;所有文件 (*.*)"));
    if (!path.isEmpty()) {
        _lisFileLine->setText(path);
        updatePreview();
    }
}

void FeatureMatchingDialog::onClearLis()
{
    _lisFileLine->clear();
    updatePreview();
}

void FeatureMatchingDialog::onGeneratePairs()
{
    QString lisPath = _lisFileLine->text().trimmed();

    if (!lisPath.isEmpty()) {
        _currentPairs = parseLisFile(lisPath);
    } else {
        _currentPairs = generateAllPairs();
    }

    updatePreview();
    emitSettingsNow();
}

void FeatureMatchingDialog::onBrowseOutput()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("选择输出目录"));
    if (!dir.isEmpty()) {
        _outputLine->setText(dir);
    }
}

void FeatureMatchingDialog::onRun()
{
    if (_currentPairs.isEmpty()) {
        QMessageBox::warning(this, tr("警告"), 
                           tr("请先生成匹配对列表"));
        return;
    }
    
    QJsonObject config = collectSettings();
    emit runRequested(config, _currentPairs);
    accept();
}

void FeatureMatchingDialog::onCancel()
{
    reject();
}

void FeatureMatchingDialog::onResetDefaults()
{
    setComboDataOrFirst(_matchAlgorithmCombo, QStringLiteral("lightglue"));
    _modelTypeCombo->setCurrentIndex(0);  // outdoor
    _outlierMethodCombo->setCurrentIndex(2);  // Fundamental USAC_MAGSAC（推荐默认，最优粗差剔除）
    _matchThresholdSpin->setValue(0.15);
    _maxKeypointsSpin->setValue(-1);
    _sinkhornIterSpin->setValue(150);
    _batchSizeSpin->setValue(1);
    _outlierReprojSpin->setValue(1.5);
    _outlierConfidenceSpin->setValue(0.9999);
    _outlierMaxItersSpin->setValue(10000);
    _outlierMinInliersSpin->setValue(25);
    _inputWidthSpin->setValue(-1);
    _inputHeightSpin->setValue(-1);
    _lgMatchThresholdSpin->setValue(0.15);
    _lgBatchSizeSpin->setValue(1);
    _lgInputWidthSpin->setValue(-1);
    _lgInputHeightSpin->setValue(-1);
    _deviceCombo->setCurrentIndex(0);
    _numThreadsSpin->setValue(-1);
    _cudaParallelSpin->setValue(1);
    _saveCsvChk->setChecked(false);
    _saveVisChk->setChecked(false);
    _verboseChk->setChecked(false);

    emitSettingsNow();
}

void FeatureMatchingDialog::onAlgorithmChanged(int)
{
    onAlgorithmOrFeatureChanged();
}

void FeatureMatchingDialog::setAvailableFeatureSuffixes(const QStringList &suffixes)
{
    _projectFeatureSuffixes = xjw::feature_match::normalizedFeatureSuffixes(suffixes);
    refreshFeatureSuffixChoices();
}

void FeatureMatchingDialog::refreshFeatureSuffixChoices()
{
    const QString previousSuffix = selectedFeatureSuffix();
    const QString algo = _matchAlgorithmCombo->currentData().toString();
    QStringList suffixes;
    if (!xjw::feature_match::isEndToEndAlgorithm(algo))
    {
        const QStringList compatibleSuffixes = xjw::feature_match::compatibleFeatureSuffixes(algo);
        if (_projectFeatureSuffixes.isEmpty())
        {
            suffixes = compatibleSuffixes;
        }
        else
        {
            for (const QString &suffix : compatibleSuffixes)
            {
                if (_projectFeatureSuffixes.contains(suffix))
                {
                    suffixes.append(suffix);
                }
            }
        }
    }

    _featureSuffixCombo->blockSignals(true);
    _featureSuffixCombo->clear();
    if (suffixes.size() > 1)
    {
        _featureSuffixCombo->addItem(tr("所有特征类型"), QStringLiteral("__all__"));
    }
    for (const auto &s : suffixes)
    {
        _featureSuffixCombo->addItem(s, s);
    }

    int restoreIndex = _featureSuffixCombo->findData(previousSuffix);
    if (restoreIndex < 0 && previousSuffix == QStringLiteral("__all__") && suffixes.size() > 1)
    {
        restoreIndex = _featureSuffixCombo->findData(QStringLiteral("__all__"));
    }
    if (restoreIndex < 0 && _featureSuffixCombo->count() > 0)
    {
        restoreIndex = 0;
    }
    if (restoreIndex >= 0)
    {
        _featureSuffixCombo->setCurrentIndex(restoreIndex);
    }
    _featureSuffixCombo->blockSignals(false);

    const bool visible = !suffixes.isEmpty();
    _featureSuffixLabel->setVisible(visible);
    _featureSuffixCombo->setVisible(visible);
}

QString FeatureMatchingDialog::selectedFeatureSuffix() const
{
    QVariant data = _featureSuffixCombo->currentData();
    return data.isValid() ? data.toString() : _featureSuffixCombo->currentText();
}

void FeatureMatchingDialog::onAlgorithmOrFeatureChanged()
{
    const QString algo = _matchAlgorithmCombo->currentData().toString();

    // 算法参数面板切换
    if (algo == "superglue")
        _paramStack->setCurrentIndex(0);
    else if (algo == "lightglue")
        _paramStack->setCurrentIndex(1);
    else if (algo == "loftr")
        _paramStack->setCurrentIndex(3);
    else if (algo == "roma")
        _paramStack->setCurrentIndex(4);
    else
        _paramStack->setCurrentIndex(2);

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
    setComboDataOrFirst(_matchAlgorithmCombo, matchAlgorithm);

    const QString outlierMethod = settings.value("outlier_method").toString("fundamental_usac_magsac");
    const int outlierIdx = _outlierMethodCombo->findData(outlierMethod);
    if (outlierIdx >= 0) _outlierMethodCombo->setCurrentIndex(outlierIdx);

    _maxKeypointsSpin->setValue(settings.value("max_keypoints").toInt(-1));

    // SuperGlue 面板
    const QString modelType = settings.value("model_type").toString("outdoor");
    const int modelIdx = _modelTypeCombo->findData(modelType);
    if (modelIdx >= 0) _modelTypeCombo->setCurrentIndex(modelIdx);
    _matchThresholdSpin->setValue(settings.value("match_threshold").toDouble(0.15));
    _sinkhornIterSpin->setValue(settings.value("sinkhorn_iterations").toInt(150));
    _batchSizeSpin->setValue(settings.value("batch_size").toInt(1));
    _inputWidthSpin->setValue(settings.value("input_width").toInt(-1));
    _inputHeightSpin->setValue(settings.value("input_height").toInt(-1));

    // LightGlue 面板（独立 key，避免与 SuperGlue 冲突）
    _lgMatchThresholdSpin->setValue(settings.value("lg_match_threshold").toDouble(
        settings.value("match_threshold").toDouble(0.15)));  // 兼容旧格式
    _lgBatchSizeSpin->setValue(settings.value("lg_batch_size").toInt(
        settings.value("batch_size").toInt(1)));
    _lgInputWidthSpin->setValue(settings.value("lg_input_width").toInt(
        settings.value("input_width").toInt(-1)));
    _lgInputHeightSpin->setValue(settings.value("lg_input_height").toInt(
        settings.value("input_height").toInt(-1)));

    // LoFTR 面板
    if (_loftrModelTypeCombo) {
        int idx = _loftrModelTypeCombo->findData(settings.value("loftr_model_type").toString("outdoor"));
        _loftrModelTypeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    if (_loftrMatchThresholdSpin)
        _loftrMatchThresholdSpin->setValue(settings.value("loftr_match_threshold").toDouble(0.2));

    // RoMa 面板
    if (_romaModelTypeCombo) {
        int idx = _romaModelTypeCombo->findData(settings.value("roma_model_type").toString("outdoor"));
        _romaModelTypeCombo->setCurrentIndex(idx >= 0 ? idx : 0);
    }
    if (_romaMatchThresholdSpin)
        _romaMatchThresholdSpin->setValue(settings.value("roma_match_threshold").toDouble(0.05));
    if (_romaMaxKeypointsSpin)
        _romaMaxKeypointsSpin->setValue(settings.value("roma_max_keypoints").toInt(10000));

    // 高级参数
    _outlierReprojSpin->setValue(settings.value("outlier_reproj_threshold").toDouble(1.5));
    _outlierConfidenceSpin->setValue(settings.value("outlier_confidence").toDouble(0.9999));
    _outlierMaxItersSpin->setValue(settings.value("outlier_max_iters").toInt(10000));
    _outlierMinInliersSpin->setValue(settings.value("outlier_min_inliers").toInt(25));

    // 系统参数
    _deviceCombo->setCurrentIndex(settings.value("use_cuda").toBool(true) ? 0 : 1);
    _numThreadsSpin->setValue(settings.value("num_threads").toInt(-1));
    _cudaParallelSpin->setValue(settings.value("cuda_parallel_pairs").toInt(1));

    // 调试参数
    _saveCsvChk->setChecked(settings.value("save_csv").toBool(false));
    _saveVisChk->setChecked(settings.value("save_visualization").toBool(false));
    _verboseChk->setChecked(settings.value("verbose").toBool(false));

    _outputLine->setText(settings.value("output_dir").toString());
    _lisFileLine->setText(settings.value("lis_file").toString());

    // 恢复已生成的匹配对
    const QJsonArray savedPairs = settings.value(QStringLiteral("generated_pairs")).toArray();
    if (!savedPairs.isEmpty()) {
        _currentPairs.clear();
        _currentPairs.reserve(savedPairs.size());
        for (const QJsonValue &v : savedPairs)
            _currentPairs.append(v.toString());
        updatePreview();
    }

    blockSignals(block);
    onAlgorithmOrFeatureChanged();

    // 恢复特征后缀选择（必须在 onAlgorithmOrFeatureChanged 之后，因为该函数会填充后缀列表）
    const QString featureSuffix = settings.value("feature_suffix").toString();
    if (!featureSuffix.isEmpty() && _featureSuffixCombo->count() > 0) {
        int idx = _featureSuffixCombo->findData(featureSuffix);
        if (idx >= 0)
            _featureSuffixCombo->setCurrentIndex(idx);
    }
}

void FeatureMatchingDialog::setProjectImages(const QStringList &imagePaths)
{
    _imageList->clear();
    for (const QString &path : imagePaths) {
        QFileInfo fi(path);
        QListWidgetItem *item = new QListWidgetItem(fi.fileName());
        item->setData(Qt::UserRole, path);
        item->setCheckState(Qt::Checked);
        _imageList->addItem(item);
    }
}

void FeatureMatchingDialog::updatePreview()
{
    QString preview;
    preview += tr("匹配对数量: %1\n\n").arg(_currentPairs.size());
    
    int maxShow = 20;
    for (int i = 0; i < qMin(_currentPairs.size(), maxShow); ++i) {
        preview += _currentPairs[i] + "\n";
    }
    
    if (_currentPairs.size() > maxShow) {
        preview += tr("... （还有 %1 对）\n").arg(_currentPairs.size() - maxShow);
    }
    
    _pairPreview->setPlainText(preview);
}

void FeatureMatchingDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

QJsonObject FeatureMatchingDialog::collectSettings() const
{
    QJsonObject obj;

    // 基础参数
    const QString algo = _matchAlgorithmCombo->currentData().toString();
    obj["match_algorithm"] = algo;
    obj["feature_suffix"] = selectedFeatureSuffix();
    obj["outlier_method"] = _outlierMethodCombo->currentData().toString();
    obj["max_keypoints"] = _maxKeypointsSpin->value();

    // 算法专属参数
    if (algo == "superglue") {
        obj["model_type"] = _modelTypeCombo->currentData().toString();
        obj["match_threshold"] = _matchThresholdSpin->value();
        obj["sinkhorn_iterations"] = _sinkhornIterSpin->value();
        obj["batch_size"] = _batchSizeSpin->value();
        obj["input_width"] = _inputWidthSpin->value();
        obj["input_height"] = _inputHeightSpin->value();
    } else if (algo == "lightglue") {
        obj["model_type"] = "outdoor";
        obj["lg_match_threshold"] = _lgMatchThresholdSpin->value();
        obj["lg_batch_size"] = _lgBatchSizeSpin->value();
        obj["lg_input_width"] = _lgInputWidthSpin->value();
        obj["lg_input_height"] = _lgInputHeightSpin->value();
    } else if (algo == "loftr") {
        obj["loftr_model_type"] = _loftrModelTypeCombo->currentData().toString();
        obj["loftr_match_threshold"] = _loftrMatchThresholdSpin->value();
    } else if (algo == "roma") {
        obj["roma_model_type"] = _romaModelTypeCombo->currentData().toString();
        obj["roma_match_threshold"] = _romaMatchThresholdSpin->value();
        obj["roma_max_keypoints"] = _romaMaxKeypointsSpin->value();
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
    obj["outlier_reproj_threshold"] = _outlierReprojSpin->value();
    obj["outlier_confidence"] = _outlierConfidenceSpin->value();
    obj["outlier_max_iters"] = _outlierMaxItersSpin->value();
    obj["outlier_min_inliers"] = _outlierMinInliersSpin->value();

    // 系统参数
    obj["use_cuda"] = (_deviceCombo->currentIndex() == 0);
    obj["num_threads"] = _numThreadsSpin->value();
    obj["cuda_parallel_pairs"] = _cudaParallelSpin->value();

    // 调试参数
    obj["save_csv"] = _saveCsvChk->isChecked();
    obj["save_visualization"] = _saveVisChk->isChecked();
    obj["verbose"] = _verboseChk->isChecked();

    // 输出设置
    obj["output_dir"] = _outputLine->text();
    obj["lis_file"] = _lisFileLine->text();

    // 保存已生成的匹配对列表
    QJsonArray pairsArr;
    for (const QString &p : _currentPairs)
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
    for (int i = 0; i < _imageList->count(); ++i) {
        QListWidgetItem *item = _imageList->item(i);
        if (item->checkState() == Qt::Checked) {
            QString baseName = QFileInfo(item->data(Qt::UserRole).toString()).completeBaseName();
            selected.append(baseName);
        }
    }

    return xjw::gui::planFeatureMatchPairs(selected);
}
