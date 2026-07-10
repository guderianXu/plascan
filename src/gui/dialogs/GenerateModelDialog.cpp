#include "GenerateModelDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QToolButton>
#include <QVBoxLayout>

namespace
{

constexpr const char *kSourceData = "source_data";
constexpr const char *kSourceLabel = "source_label";
constexpr const char *kSourcePath = "source_path";
constexpr const char *kDisplay = "display";
constexpr const char *kSupported = "supported";
constexpr const char *kNote = "note";

QString defaultSourceLabel(const QString &sourceData)
{
    if (sourceData == QStringLiteral("tie_points"))
    {
        return QStringLiteral("连接点");
    }
    if (sourceData == QStringLiteral("depth_maps"))
    {
        return QStringLiteral("深度图");
    }
    if (sourceData == QStringLiteral("point_cloud"))
    {
        return QStringLiteral("点云");
    }
    if (sourceData == QStringLiteral("model"))
    {
        return QStringLiteral("模型");
    }
    return sourceData;
}

int qualityOctreeDepth(const QString &quality)
{
    if (quality == QStringLiteral("ultra"))
    {
        return 11;
    }
    if (quality == QStringLiteral("high"))
    {
        return 10;
    }
    if (quality == QStringLiteral("low"))
    {
        return 8;
    }
    return 9;
}

int qualityMeshResolution(const QString &quality)
{
    if (quality == QStringLiteral("ultra"))
    {
        return 384;
    }
    if (quality == QStringLiteral("high"))
    {
        return 320;
    }
    if (quality == QStringLiteral("low"))
    {
        return 192;
    }
    return 256;
}

QString qualityProfile(const QString &quality)
{
    if (quality == QStringLiteral("low"))
    {
        return QStringLiteral("lite");
    }
    if (quality == QStringLiteral("high") || quality == QStringLiteral("ultra"))
    {
        return QStringLiteral("detail");
    }
    return QStringLiteral("balanced");
}

} // namespace

GenerateModelDialog::GenerateModelDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("生成模型"));
    setMinimumWidth(520);

    auto *mainLayout = new QVBoxLayout(this);

    auto *generalGroup = new QGroupBox(tr("一般"), this);
    auto *generalForm = new QFormLayout(generalGroup);
    generalForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    _sourceCombo = new QComboBox(generalGroup);
    _sourceItemCombo = new QComboBox(generalGroup);
    _surfaceTypeCombo = new QComboBox(generalGroup);
    _qualityCombo = new QComboBox(generalGroup);
    _faceCountCombo = new QComboBox(generalGroup);
    _saveEachStepCheck = new QCheckBox(tr("在每个步骤完成后保存项目"), generalGroup);

    _surfaceTypeCombo->addItem(tr("任意 (3D)"), QStringLiteral("arbitrary_3d"));
    _surfaceTypeCombo->addItem(tr("高度场 (2.5D)"), QStringLiteral("height_field"));

    _qualityCombo->addItem(tr("低"), QStringLiteral("low"));
    _qualityCombo->addItem(tr("中"), QStringLiteral("medium"));
    _qualityCombo->addItem(tr("高"), QStringLiteral("high"));
    _qualityCombo->addItem(tr("超高"), QStringLiteral("ultra"));
    _qualityCombo->setCurrentIndex(2);

    _faceCountCombo->addItem(tr("低 (60,000)"), 60000);
    _faceCountCombo->addItem(tr("中 (120,000)"), 120000);
    _faceCountCombo->addItem(tr("高 (240,000)"), 240000);
    _faceCountCombo->addItem(tr("自适应"), 0);
    _faceCountCombo->setCurrentIndex(2);

    generalForm->addRow(tr("源数据:"), _sourceCombo);
    generalForm->addRow(tr("数据项:"), _sourceItemCombo);
    generalForm->addRow(tr("表面类型:"), _surfaceTypeCombo);
    generalForm->addRow(tr("质量:"), _qualityCombo);
    generalForm->addRow(tr("面数:"), _faceCountCombo);
    generalForm->addRow(QString(), _saveEachStepCheck);
    mainLayout->addWidget(generalGroup);

    auto *regionGroup = new QGroupBox(tr("区域"), this);
    auto *regionForm = new QFormLayout(regionGroup);
    _splitRegionCheck = new QCheckBox(tr("分割成区块"), regionGroup);
    auto *coordinateLabel = new QLabel(tr("Local Coordinates (m)"), regionGroup);
    _blockSizeSpin = new QDoubleSpinBox(regionGroup);
    _blockSizeSpin->setRange(1.0, 100000.0);
    _blockSizeSpin->setDecimals(1);
    _blockSizeSpin->setValue(250.0);
    _blockSizeSpin->setSuffix(tr(" m"));
    auto *originLabel = new QLabel(tr("X: -5    Y: -5"), regionGroup);
    _skipBoundaryBlocksCheck = new QCheckBox(tr("跳过边界外的块"), regionGroup);
    coordinateLabel->setEnabled(false);
    originLabel->setEnabled(false);
    regionForm->addRow(QString(), _splitRegionCheck);
    regionForm->addRow(tr("坐标系统:"), coordinateLabel);
    regionForm->addRow(tr("区块大小 (米):"), _blockSizeSpin);
    regionForm->addRow(tr("网格原点:"), originLabel);
    regionForm->addRow(QString(), _skipBoundaryBlocksCheck);
    mainLayout->addWidget(regionGroup);

    _advancedToggle = new QToolButton(this);
    _advancedToggle->setText(tr("高级"));
    _advancedToggle->setCheckable(true);
    _advancedToggle->setChecked(false);
    _advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    _advancedToggle->setArrowType(Qt::RightArrow);
    mainLayout->addWidget(_advancedToggle);

    _advancedContent = new QWidget(this);
    auto *advancedForm = new QFormLayout(_advancedContent);
    advancedForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    advancedForm->setContentsMargins(0, 0, 0, 0);
    advancedForm->setVerticalSpacing(8);

    _interpolationCombo = new QComboBox(_advancedContent);
    _interpolationCombo->addItem(tr("已启用 (默认)"), QStringLiteral("enabled"));
    _interpolationCombo->addItem(tr("已禁用"), QStringLiteral("disabled"));
    _interpolationCombo->addItem(tr("外推"), QStringLiteral("extrapolated"));

    _depthFilterCombo = new QComboBox(_advancedContent);
    _depthFilterCombo->addItem(tr("温和"), QStringLiteral("mild"));
    _depthFilterCombo->addItem(tr("中等"), QStringLiteral("moderate"));
    _depthFilterCombo->addItem(tr("强"), QStringLiteral("aggressive"));
    _depthFilterCombo->addItem(tr("禁用"), QStringLiteral("disabled"));

    _calculateColorsCheck = new QCheckBox(tr("计算顶点颜色"), _advancedContent);
    _strictMasksCheck = new QCheckBox(tr("使用严格的体积掩模"), _advancedContent);
    _reuseDepthMapsCheck = new QCheckBox(tr("重用深度图"), _advancedContent);
    _replaceDefaultCheck = new QCheckBox(tr("要替换默认模型吗"), _advancedContent);
    _calculateColorsCheck->setChecked(true);
    _reuseDepthMapsCheck->setChecked(true);

    advancedForm->addRow(tr("插值:"), _interpolationCombo);
    advancedForm->addRow(tr("深度过滤:"), _depthFilterCombo);
    advancedForm->addRow(QString(), _calculateColorsCheck);
    advancedForm->addRow(QString(), _strictMasksCheck);
    advancedForm->addRow(QString(), _reuseDepthMapsCheck);
    advancedForm->addRow(QString(), _replaceDefaultCheck);
    mainLayout->addWidget(_advancedContent);

    _statusLabel = new QLabel(this);
    _statusLabel->setWordWrap(true);
    mainLayout->addWidget(_statusLabel);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    _okButton = buttonBox->button(QDialogButtonBox::Ok);
    _okButton->setText(tr("OK"));
    buttonBox->button(QDialogButtonBox::Cancel)->setText(tr("Cancel"));
    mainLayout->addWidget(buttonBox);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &GenerateModelDialog::onRun);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(_advancedToggle, &QToolButton::toggled, this, [this](bool expanded)
    {
        setAdvancedExpanded(expanded);
    });

    connect(_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenerateModelDialog::onSourceTypeChanged);
    connect(_sourceItemCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenerateModelDialog::emitSettingsNow);
    connect(_surfaceTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenerateModelDialog::emitSettingsNow);
    connect(_qualityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenerateModelDialog::emitSettingsNow);
    connect(_faceCountCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenerateModelDialog::emitSettingsNow);
    connect(_saveEachStepCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);
    connect(_splitRegionCheck, &QCheckBox::toggled, this, [this]()
    {
        updateBlockControlsAvailability();
        emitSettingsNow();
    });
    connect(_blockSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &GenerateModelDialog::emitSettingsNow);
    connect(_skipBoundaryBlocksCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);
    connect(_interpolationCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenerateModelDialog::emitSettingsNow);
    connect(_depthFilterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &GenerateModelDialog::emitSettingsNow);
    connect(_calculateColorsCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);
    connect(_strictMasksCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);
    connect(_reuseDepthMapsCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);
    connect(_replaceDefaultCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);

    setAdvancedExpanded(false);
    refreshSourceTypes();
}

void GenerateModelDialog::applySettings(const QJsonObject &settings)
{
    _pendingSourceData = settings.value(QLatin1String(kSourceData)).toString();
    _pendingSourcePath = settings.value(QLatin1String(kSourcePath)).toString(
        settings.value(QStringLiteral("denseCloudPath")).toString());

    const QString surfaceType = settings.value(QStringLiteral("surface_type")).toString();
    if (!surfaceType.isEmpty())
    {
        const int index = _surfaceTypeCombo->findData(surfaceType);
        if (index >= 0)
        {
            _surfaceTypeCombo->setCurrentIndex(index);
        }
    }

    const QString quality = settings.value(QStringLiteral("quality")).toString();
    if (!quality.isEmpty())
    {
        const int index = _qualityCombo->findData(quality);
        if (index >= 0)
        {
            _qualityCombo->setCurrentIndex(index);
        }
    }

    if (settings.contains(QStringLiteral("targetFaces")))
    {
        const int index = _faceCountCombo->findData(settings.value(QStringLiteral("targetFaces")).toInt());
        if (index >= 0)
        {
            _faceCountCombo->setCurrentIndex(index);
        }
    }

    _saveEachStepCheck->setChecked(settings.value(QStringLiteral("saveAfterEachStep")).toBool(false));
    _calculateColorsCheck->setChecked(settings.value(QStringLiteral("calculateVertexColors")).toBool(true));
    _strictMasksCheck->setChecked(settings.value(QStringLiteral("strictVolumetricMasks")).toBool(false));
    _reuseDepthMapsCheck->setChecked(settings.value(QStringLiteral("reuseDepthMaps")).toBool(false));
    _replaceDefaultCheck->setChecked(settings.value(QStringLiteral("replaceDefaultModel")).toBool(false));
}

void GenerateModelDialog::setSourceCandidates(const QJsonArray &candidates)
{
    _candidates = candidates;
    refreshSourceTypes();
}

QJsonObject GenerateModelDialog::collectSettings() const
{
    const QJsonObject candidate = currentCandidate();
    const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
    const QString sourcePath = candidate.value(QLatin1String(kSourcePath)).toString();
    QString quality = _qualityCombo->currentData().toString();
    if (quality.isEmpty())
    {
        quality = QStringLiteral("high");
    }
    QString surfaceType = _surfaceTypeCombo->currentData().toString();
    if (surfaceType.isEmpty())
    {
        surfaceType = QStringLiteral("arbitrary_3d");
    }
    int targetFaces = _faceCountCombo->currentData().toInt();
    if (targetFaces < 0)
    {
        targetFaces = 240000;
    }

    QJsonObject settings;
    settings[QLatin1String(kSourceData)] = sourceData;
    settings[QLatin1String(kSourceLabel)] =
        candidate.value(QLatin1String(kSourceLabel)).toString(defaultSourceLabel(sourceData));
    settings[QLatin1String(kSourcePath)] = sourcePath;
    settings[QStringLiteral("source_display")] = candidate.value(QLatin1String(kDisplay)).toString();
    settings[QStringLiteral("source_supported")] = candidate.value(QLatin1String(kSupported)).toBool(false);
    settings[QStringLiteral("surface_type")] = surfaceType;
    settings[QStringLiteral("quality")] = quality;
    settings[QStringLiteral("qualityProfile")] = qualityProfile(quality);
    settings[QStringLiteral("octreeDepth")] = qualityOctreeDepth(quality);
    settings[QStringLiteral("meshResolution")] = qualityMeshResolution(quality);
    settings[QStringLiteral("targetFaces")] = targetFaces;
    settings[QStringLiteral("simplifyTargetFaces")] = targetFaces;
    settings[QStringLiteral("method")] = surfaceType == QStringLiteral("height_field")
        ? QStringLiteral("Height Field")
        : QStringLiteral("Poisson Surface");
    settings[QStringLiteral("export_format")] = QStringLiteral("PLY");
    settings[QStringLiteral("saveAfterEachStep")] = _saveEachStepCheck->isChecked();
    settings[QStringLiteral("splitIntoBlocks")] = _splitRegionCheck->isChecked();
    settings[QStringLiteral("blockSizeMeters")] = _blockSizeSpin->value();
    settings[QStringLiteral("skipBoundaryBlocks")] = _skipBoundaryBlocksCheck->isChecked();
    settings[QStringLiteral("interpolation")] = _interpolationCombo->currentData().toString();
    settings[QStringLiteral("depthFiltering")] = _depthFilterCombo->currentData().toString();
    settings[QStringLiteral("calculateVertexColors")] = _calculateColorsCheck->isChecked();
    settings[QStringLiteral("strictVolumetricMasks")] = _strictMasksCheck->isChecked();
    settings[QStringLiteral("reuseDepthMaps")] = _reuseDepthMapsCheck->isChecked();
    settings[QStringLiteral("replaceDefaultModel")] = _replaceDefaultCheck->isChecked();

    if (sourceData == QStringLiteral("point_cloud")
        || sourceData == QStringLiteral("tie_points")
        || sourceData == QStringLiteral("model"))
    {
        settings[QStringLiteral("denseCloudPath")] = sourcePath;
    }
    if (sourceData == QStringLiteral("depth_maps"))
    {
        settings[QStringLiteral("depthMapSourcePath")] = sourcePath;
    }

    return settings;
}

QJsonObject GenerateModelDialog::currentCandidate() const
{
    return _sourceItemCombo->currentData().toJsonObject();
}

void GenerateModelDialog::refreshSourceTypes()
{
    const QString currentSource = !_pendingSourceData.isEmpty()
        ? _pendingSourceData
        : _sourceCombo->currentData().toString();

    _sourceCombo->blockSignals(true);
    _sourceCombo->clear();

    QStringList addedTypes;
    for (const QJsonValue &value : _candidates)
    {
        const QJsonObject candidate = value.toObject();
        const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
        if (sourceData.isEmpty() || addedTypes.contains(sourceData))
        {
            continue;
        }

        addedTypes.push_back(sourceData);
        _sourceCombo->addItem(candidate.value(QLatin1String(kSourceLabel)).toString(defaultSourceLabel(sourceData)),
                              sourceData);
    }

    if (_sourceCombo->count() == 0)
    {
        _sourceCombo->addItem(tr("无可用源数据"), QString());
    }

    const int index = _sourceCombo->findData(currentSource);
    if (index >= 0)
    {
        _sourceCombo->setCurrentIndex(index);
    }
    _sourceCombo->blockSignals(false);

    refreshSourceItems();
}

void GenerateModelDialog::refreshSourceItems()
{
    const QString sourceData = _sourceCombo->currentData().toString();
    const QString currentPath = !_pendingSourcePath.isEmpty()
        ? _pendingSourcePath
        : currentCandidate().value(QLatin1String(kSourcePath)).toString();

    _sourceItemCombo->blockSignals(true);
    _sourceItemCombo->clear();

    for (const QJsonValue &value : _candidates)
    {
        const QJsonObject candidate = value.toObject();
        if (candidate.value(QLatin1String(kSourceData)).toString() != sourceData)
        {
            continue;
        }

        const QString display = candidate.value(QLatin1String(kDisplay)).toString(
            candidate.value(QLatin1String(kSourcePath)).toString());
        _sourceItemCombo->addItem(display, candidate);
    }

    int index = -1;
    for (int i = 0; i < _sourceItemCombo->count(); ++i)
    {
        const QJsonObject candidate = _sourceItemCombo->itemData(i).toJsonObject();
        if (candidate.value(QLatin1String(kSourcePath)).toString() == currentPath)
        {
            index = i;
            break;
        }
    }
    if (index >= 0)
    {
        _sourceItemCombo->setCurrentIndex(index);
    }
    else if (_sourceItemCombo->count() > 0)
    {
        _sourceItemCombo->setCurrentIndex(0);
    }

    _sourceItemCombo->blockSignals(false);
    _pendingSourceData.clear();
    _pendingSourcePath.clear();
    updateAvailability();
    emitSettingsNow();
}

void GenerateModelDialog::setAdvancedExpanded(bool expanded)
{
    _advancedToggle->setChecked(expanded);
    _advancedToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    _advancedContent->setVisible(expanded);
    setMinimumHeight(expanded ? 620 : 430);
    if (layout())
    {
        layout()->invalidate();
    }
    adjustSize();
}

void GenerateModelDialog::updateAvailability()
{
    const QJsonObject candidate = currentCandidate();
    const bool supported = candidate.value(QLatin1String(kSupported)).toBool(false);
    const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
    const QString note = candidate.value(QLatin1String(kNote)).toString();
    const bool hasCandidate = !sourceData.isEmpty();

    updateBlockControlsAvailability();
    _okButton->setEnabled(hasCandidate && supported);
    if (!hasCandidate)
    {
        _statusLabel->setText(tr("当前项目还没有可用于生成模型的源数据。"));
        return;
    }
    if (!supported)
    {
        _statusLabel->setText(note.isEmpty()
            ? tr("该源数据当前不能直接用于生成模型。")
            : note);
        return;
    }
    _statusLabel->setText(note.isEmpty()
        ? tr("输出: 项目目录下的 model/products/model_from_mesh.ply")
        : note);
}

void GenerateModelDialog::updateBlockControlsAvailability()
{
    const QString sourceData = _sourceCombo->currentData().toString();
    const bool blockCapable =
        sourceData == QStringLiteral("depth_maps") || sourceData == QStringLiteral("point_cloud");

    _splitRegionCheck->setEnabled(blockCapable);
    _blockSizeSpin->setEnabled(blockCapable && _splitRegionCheck->isChecked());
    _skipBoundaryBlocksCheck->setEnabled(blockCapable && _splitRegionCheck->isChecked());
}

void GenerateModelDialog::emitSettingsNow()
{
    updateAvailability();
    emit settingsChanged(collectSettings());
}

void GenerateModelDialog::onRun()
{
    const QJsonObject settings = collectSettings();
    if (!settings.value(QStringLiteral("source_supported")).toBool(false))
    {
        return;
    }

    emit runRequested(settings);
    accept();
}

void GenerateModelDialog::onSourceTypeChanged()
{
    refreshSourceItems();
}
