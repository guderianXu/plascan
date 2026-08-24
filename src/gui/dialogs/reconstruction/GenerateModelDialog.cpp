#include "reconstruction/GenerateModelDialog.h"
#include "shared/WorkflowParameterDialogStyle.h"
#include "PointCloudWorkflowConfig.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWindow>

namespace
{

constexpr const char *kSourceData = "source_data";
constexpr const char *kSourceLabel = "source_label";
constexpr const char *kSourcePath = "source_path";
constexpr const char *kDisplay = "display";
constexpr const char *kSupported = "supported";
constexpr const char *kNote = "note";
constexpr const char *kAutomaticDepthMaps = "automatic_depth_maps";
constexpr const char *kDepthQualityProfile = "depth_quality_profile";
constexpr const char *kDepthBatchCompatible = "depth_batch_compatible";
constexpr const char *kDepthBatchCompatibilityReason =
    "depth_batch_compatibility_reason";
constexpr const char *kComputeMode = "compute_mode";

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

QString depthQualityDisplayName(const QString &profile)
{
    if (profile == QStringLiteral("highest")) return QStringLiteral("超高");
    if (profile == QStringLiteral("high")) return QStringLiteral("高");
    if (profile == QStringLiteral("low")) return QStringLiteral("低");
    if (profile == QStringLiteral("lowest")) return QStringLiteral("最低");
    return QStringLiteral("中");
}

} // namespace

GenerateModelDialog::GenerateModelDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("生成模型"));
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);

    auto *mainLayout = new QVBoxLayout(this);
    xjw::gui::dialogs::configureWorkflowDialogLayout(mainLayout);

    _contentScrollArea = new QScrollArea(this);
    _contentScrollArea->setObjectName(QStringLiteral("workflowParameterScrollArea"));
    xjw::gui::dialogs::configureWorkflowScrollArea(_contentScrollArea);
    auto *contentWidget = new QWidget(_contentScrollArea);
    contentWidget->setObjectName(QStringLiteral("workflowParameterContent"));
    auto *contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(9);

    auto *generalGroup = new QGroupBox(tr("一般"), contentWidget);
    generalGroup->setObjectName(QStringLiteral("workflowGeneralGroup"));
    auto *generalForm = new QFormLayout(generalGroup);
    xjw::gui::dialogs::configureWorkflowForm(generalForm);

    _sourceCombo = new QComboBox(generalGroup);
    _sourceItemCombo = new QComboBox(generalGroup);
    _surfaceTypeCombo = new QComboBox(generalGroup);
    _qualityCombo = new QComboBox(generalGroup);
    _faceCountCombo = new QComboBox(generalGroup);
    _saveEachStepCheck = new QCheckBox(tr("在每个步骤完成后保存项目"), generalGroup);

    _sourceCombo->setObjectName(QStringLiteral("modelSourceCombo"));
    _sourceItemCombo->setObjectName(QStringLiteral("modelSourceItemCombo"));
    _sourceItemCombo->hide();
    _surfaceTypeCombo->setObjectName(QStringLiteral("modelSurfaceTypeCombo"));
    _qualityCombo->setObjectName(QStringLiteral("modelQualityCombo"));
    _faceCountCombo->setObjectName(QStringLiteral("modelFaceCountCombo"));
    for (QComboBox *comboBox : {
             _sourceCombo, _sourceItemCombo, _surfaceTypeCombo, _qualityCombo, _faceCountCombo})
    {
        xjw::gui::dialogs::configureWorkflowComboBox(comboBox);
    }

    _surfaceTypeCombo->addItem(tr("任意 (3D)"), QStringLiteral("arbitrary_3d"));
    _surfaceTypeCombo->addItem(tr("高度场 (2.5D)"), QStringLiteral("height_field"));

    _qualityCombo->addItem(tr("低"), QStringLiteral("low"));
    _qualityCombo->addItem(tr("中"), QStringLiteral("medium"));
    _qualityCombo->addItem(tr("高"), QStringLiteral("high"));
    _qualityCombo->addItem(tr("超高"), QStringLiteral("ultra"));
    _qualityCombo->setCurrentIndex(2);

    _effectiveDepthQualityLabel = new QLabel(generalGroup);
    _effectiveDepthQualityLabel->setObjectName(
        QStringLiteral("effectiveDepthQualityLabel"));
    _effectiveDepthQualityLabel->setWordWrap(true);

    _faceCountCombo->addItem(tr("低 (60,000)"), 60000);
    _faceCountCombo->addItem(tr("中 (120,000)"), 120000);
    _faceCountCombo->addItem(tr("高 (240,000)"), 240000);
    _faceCountCombo->addItem(tr("自适应"), 0);
    _faceCountCombo->setCurrentIndex(2);

    generalForm->addRow(tr("源数据:"), _sourceCombo);
    generalForm->addRow(tr("表面类型:"), _surfaceTypeCombo);
    generalForm->addRow(tr("质量:"), _qualityCombo);
    generalForm->addRow(tr("深度质量:"), _effectiveDepthQualityLabel);
    generalForm->addRow(tr("面数:"), _faceCountCombo);
    generalForm->addRow(QString(), _saveEachStepCheck);
    contentLayout->addWidget(generalGroup);

    auto *regionGroup = new QGroupBox(tr("区域"), contentWidget);
    regionGroup->setObjectName(QStringLiteral("workflowRegionGroup"));
    auto *regionForm = new QFormLayout(regionGroup);
    xjw::gui::dialogs::configureWorkflowForm(regionForm);
    _splitRegionCheck = new QCheckBox(tr("分割成区块"), regionGroup);
    _splitRegionCheck->setObjectName(QStringLiteral("splitRegionCheck"));
    _coordinateLabel = new QLabel(tr("Local Coordinates (m)"), regionGroup);
    _coordinateLabel->setObjectName(QStringLiteral("coordinateSystemLabel"));
    _blockSizeSpin = new QDoubleSpinBox(regionGroup);
    _blockSizeSpin->setObjectName(QStringLiteral("blockSizeSpin"));
    _blockSizeSpin->setRange(1.0, 100000.0);
    _blockSizeSpin->setDecimals(1);
    _blockSizeSpin->setValue(250.0);
    _blockSizeSpin->setSuffix(tr(" m"));
    _originLabel = new QLabel(tr("X: -5    Y: -5"), regionGroup);
    _originLabel->setObjectName(QStringLiteral("gridOriginLabel"));
    _skipBoundaryBlocksCheck = new QCheckBox(tr("跳过边界外的块"), regionGroup);
    _coordinateLabel->setEnabled(false);
    _originLabel->setEnabled(false);
    regionForm->addRow(QString(), _splitRegionCheck);
    regionForm->addRow(tr("坐标系统:"), _coordinateLabel);
    regionForm->addRow(tr("区块大小 (米):"), _blockSizeSpin);
    regionForm->addRow(tr("网格原点:"), _originLabel);
    regionForm->addRow(QString(), _skipBoundaryBlocksCheck);
    contentLayout->addWidget(regionGroup);

    _advancedToggle = new QToolButton(contentWidget);
    _advancedToggle->setObjectName(QStringLiteral("workflowAdvancedToggle"));
    _advancedToggle->setText(tr("高级"));
    _advancedToggle->setCheckable(true);
    _advancedToggle->setChecked(false);
    _advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    _advancedToggle->setArrowType(Qt::RightArrow);
    _advancedToggle->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    contentLayout->addWidget(_advancedToggle);

    _advancedContent = new QGroupBox(tr("高级参数"), contentWidget);
    _advancedContent->setObjectName(QStringLiteral("workflowAdvancedGroup"));
    auto *advancedForm = new QFormLayout(_advancedContent);
    xjw::gui::dialogs::configureWorkflowForm(advancedForm);

    _interpolationCombo = new QComboBox(_advancedContent);
    _interpolationCombo->setObjectName(QStringLiteral("modelInterpolationCombo"));
    _interpolationCombo->addItem(tr("已启用 (默认)"), QStringLiteral("enabled"));
    _interpolationCombo->addItem(tr("已禁用"), QStringLiteral("disabled"));
    _interpolationCombo->addItem(tr("外推"), QStringLiteral("extrapolated"));

    _depthFilterCombo = new QComboBox(_advancedContent);
    _depthFilterCombo->setObjectName(QStringLiteral("modelDepthFilterCombo"));
    _depthFilterCombo->addItem(tr("温和"), QStringLiteral("mild"));
    _depthFilterCombo->addItem(tr("中等"), QStringLiteral("moderate"));
    _depthFilterCombo->addItem(tr("强"), QStringLiteral("aggressive"));
    _depthFilterCombo->addItem(tr("禁用"), QStringLiteral("disabled"));
    xjw::gui::dialogs::configureWorkflowComboBox(_interpolationCombo);
    xjw::gui::dialogs::configureWorkflowComboBox(_depthFilterCombo);

    _calculateColorsCheck = new QCheckBox(tr("计算顶点颜色"), _advancedContent);
    _strictMasksCheck = new QCheckBox(tr("使用严格的体积掩模"), _advancedContent);
    _reuseDepthMapsCheck = new QCheckBox(tr("重用深度图"), _advancedContent);
    _reuseDepthMapsCheck->setObjectName(QStringLiteral("reuseDepthMapsCheck"));
    _replaceDefaultCheck = new QCheckBox(tr("替换默认模型"), _advancedContent);
    _calculateColorsCheck->setChecked(true);
    _reuseDepthMapsCheck->setChecked(true);

    advancedForm->addRow(tr("插值:"), _interpolationCombo);
    advancedForm->addRow(tr("深度过滤:"), _depthFilterCombo);
    advancedForm->addRow(QString(), _calculateColorsCheck);
    advancedForm->addRow(QString(), _strictMasksCheck);
    advancedForm->addRow(QString(), _reuseDepthMapsCheck);
    advancedForm->addRow(QString(), _replaceDefaultCheck);
    contentLayout->addWidget(_advancedContent);

    _contentScrollArea->setWidget(contentWidget);
    mainLayout->addWidget(_contentScrollArea);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttonBox->setObjectName(QStringLiteral("workflowButtonBox"));
    xjw::gui::dialogs::configureWorkflowButtonBox(buttonBox, tr("生成"));
    _okButton = buttonBox->button(QDialogButtonBox::Ok);
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
    connect(_reuseDepthMapsCheck, &QCheckBox::toggled, this, [this](bool checked)
    {
        if (_hasReusableDepthMaps)
        {
            _reuseDepthMapsRequested = checked;
        }
        emitSettingsNow();
    });
    connect(_replaceDefaultCheck, &QCheckBox::toggled, this, &GenerateModelDialog::emitSettingsNow);

    setAdvancedExpanded(false);
    refreshSourceTypes();
}

void GenerateModelDialog::showEvent(QShowEvent *event)
{
    QDialog::showEvent(event);
    if (!_screenChangeConnected && windowHandle())
    {
        connect(windowHandle(), &QWindow::screenChanged, this, [this](QScreen *targetScreen)
        {
            bindScreenGeometryUpdates(targetScreen);
            refreshScrollableDialogSize();
        });
        _screenChangeConnected = true;
    }

    bindScreenGeometryUpdates(screen());
    refreshScrollableDialogSize();
}

void GenerateModelDialog::applySettings(const QJsonObject &settings)
{
    _pendingSourceData = settings.value(QLatin1String(kSourceData)).toString();
    _pendingSourcePath = settings.value(QLatin1String(kSourcePath)).toString(
        settings.value(QStringLiteral("denseCloudPath")).toString());
    QString compute_mode = settings.value(QLatin1String(kComputeMode))
        .toString()
        .trimmed()
        .toLower();
    if (compute_mode.isEmpty())
    {
        const QString backend = settings.value(
            QStringLiteral("patch_match_backend")).toString(
                settings.value(QStringLiteral("processingDevice")).toString())
            .trimmed()
            .toLower();
        compute_mode = backend == QStringLiteral("cuda") ||
                backend == QStringLiteral("opencl")
            ? backend
            : QStringLiteral("hybrid");
    }
    if (compute_mode == QStringLiteral("cuda") ||
        compute_mode == QStringLiteral("opencl") ||
        compute_mode == QStringLiteral("hybrid"))
    {
        _computeMode = compute_mode;
    }

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

    const QString interpolation = settings.value(QStringLiteral("interpolation")).toString();
    if (!interpolation.isEmpty())
    {
        const int index = _interpolationCombo->findData(interpolation);
        if (index >= 0)
        {
            _interpolationCombo->setCurrentIndex(index);
        }
    }

    const QString depth_filtering = settings.value(QStringLiteral("depthFiltering")).toString();
    if (!depth_filtering.isEmpty())
    {
        const int index = _depthFilterCombo->findData(depth_filtering);
        if (index >= 0)
        {
            _depthFilterCombo->setCurrentIndex(index);
        }
    }

    _saveEachStepCheck->setChecked(settings.value(QStringLiteral("saveAfterEachStep")).toBool(false));
    _calculateColorsCheck->setChecked(settings.value(QStringLiteral("calculateVertexColors")).toBool(true));
    _strictMasksCheck->setChecked(settings.value(QStringLiteral("strictVolumetricMasks")).toBool(false));
    _reuseDepthMapsRequested = settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    _reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested);
    _replaceDefaultCheck->setChecked(settings.value(QStringLiteral("replaceDefaultModel")).toBool(false));
}

void GenerateModelDialog::setSourceCandidates(const QJsonArray &candidates)
{
    _candidates = candidates;
    _hasReusableDepthMaps = false;
    bool has_depth_candidate = false;
    bool can_generate_depth_maps = false;
    for (const QJsonValue &value : _candidates)
    {
        const QJsonObject candidate = value.toObject();
        const QString source_data =
            candidate.value(QLatin1String(kSourceData)).toString();
        if (source_data == QStringLiteral("tie_points") &&
            candidate.value(QLatin1String(kSupported)).toBool(false))
        {
            can_generate_depth_maps = true;
        }
        if (source_data == QStringLiteral("depth_maps"))
        {
            has_depth_candidate = true;
            if (candidate.value(QLatin1String(kSupported)).toBool(false) &&
                candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true) &&
                !candidate.value(QLatin1String(kSourcePath)).toString().trimmed().isEmpty())
            {
                _hasReusableDepthMaps = true;
            }
        }
    }

    // “深度图”是正式模型工作流，不是只有磁盘产物存在时才出现的
    // 资源快捷方式。没有兼容批次时仍提供该入口，由工作流先估计深度图，
    // 然后直接执行 TSDF 表面重建。
    if (!has_depth_candidate)
    {
        QJsonObject automatic_depth_maps;
        automatic_depth_maps[QLatin1String(kSourceData)] =
            QStringLiteral("depth_maps");
        automatic_depth_maps[QLatin1String(kSourceLabel)] =
            QStringLiteral("深度图");
        automatic_depth_maps[QLatin1String(kSourcePath)] = QString();
        automatic_depth_maps[QLatin1String(kDisplay)] =
            QStringLiteral("自动生成深度图");
        automatic_depth_maps[QLatin1String(kSupported)] =
            can_generate_depth_maps;
        automatic_depth_maps[QLatin1String(kAutomaticDepthMaps)] = true;
        automatic_depth_maps[QLatin1String(kNote)] = can_generate_depth_maps
            ? QStringLiteral("缺少深度图时将自动估计深度图，再直接进行 TSDF 表面重建。")
            : QStringLiteral("需要先完成通过质量门控的照片对齐，才能自动估计深度图。");
        _candidates.prepend(automatic_depth_maps);

        if (_pendingSourceData.isEmpty() ||
            _pendingSourceData == QStringLiteral("tie_points"))
        {
            _pendingSourceData = QStringLiteral("depth_maps");
            _pendingSourcePath.clear();
        }
    }
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
    settings[QStringLiteral("modelQualityProfile")] = qualityProfile(quality);
    settings[QStringLiteral("qualityProfile")] = qualityProfile(quality);
    settings[QStringLiteral("depthQualityProfile")] =
        xjw::core::project::depthQualityProfileForModelQuality(quality);
    settings[QStringLiteral("octreeDepth")] = qualityOctreeDepth(quality);
    settings[QStringLiteral("meshResolution")] = qualityMeshResolution(quality);
    settings[QStringLiteral("targetFaces")] = targetFaces;
    settings[QStringLiteral("simplifyTargetFaces")] = targetFaces;
    if (sourceData == QStringLiteral("depth_maps") &&
        surfaceType == QStringLiteral("arbitrary_3d"))
    {
        settings[QStringLiteral("method")] = QStringLiteral("Depth TSDF");
        settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
    }
    else
    {
        settings[QStringLiteral("method")] = surfaceType == QStringLiteral("height_field")
            ? QStringLiteral("Height Field")
            : QStringLiteral("Poisson Surface");
    }
    // 模型生成只负责网格；OBJ/MTL/纹理图由独立的“生成纹理”流程创建。
    settings[QStringLiteral("export_format")] = QStringLiteral("PLY");
    settings[QStringLiteral("saveAfterEachStep")] = _saveEachStepCheck->isChecked();
    settings[QStringLiteral("splitIntoBlocks")] = _splitRegionCheck->isChecked();
    settings[QStringLiteral("blockSizeMeters")] = _blockSizeSpin->value();
    settings[QStringLiteral("skipBoundaryBlocks")] = _skipBoundaryBlocksCheck->isChecked();
    settings[QStringLiteral("interpolation")] = _interpolationCombo->currentData().toString();
    settings[QStringLiteral("depthFiltering")] = _depthFilterCombo->currentData().toString();
    settings[QStringLiteral("calculateVertexColors")] = _calculateColorsCheck->isChecked();
    settings[QStringLiteral("strictVolumetricMasks")] = _strictMasksCheck->isChecked();
    const bool selected_depth_batch_compatible =
        candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true);
    const bool reuse_depth_maps =
        _hasReusableDepthMaps && selected_depth_batch_compatible &&
        _reuseDepthMapsRequested;
    settings[QStringLiteral("reuseDepthMaps")] = reuse_depth_maps;
    const bool automatic_depth_maps =
        candidate.value(QLatin1String(kAutomaticDepthMaps)).toBool(false);
    settings[QStringLiteral("automatic_depth_maps")] = automatic_depth_maps;
    settings[QStringLiteral("force_depth_recompute")] =
        automatic_depth_maps ||
        (sourceData == QStringLiteral("depth_maps") &&
         !reuse_depth_maps);
    settings[QStringLiteral("replaceDefaultModel")] = _replaceDefaultCheck->isChecked();
    settings[QLatin1String(kComputeMode)] = _computeMode;
    settings[QStringLiteral("patch_match_backend")] =
        _computeMode == QStringLiteral("hybrid")
            ? QStringLiteral("auto")
            : _computeMode;
    settings[QStringLiteral("processingDevice")] =
        _computeMode == QStringLiteral("hybrid")
            ? QStringLiteral("auto")
            : _computeMode;

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
    if (_contentScrollArea && _contentScrollArea->widget() && _contentScrollArea->widget()->layout())
    {
        _contentScrollArea->widget()->layout()->invalidate();
        _contentScrollArea->widget()->layout()->activate();
        _contentScrollArea->widget()->adjustSize();
    }
    refreshScrollableDialogSize();
}

void GenerateModelDialog::bindScreenGeometryUpdates(QScreen *targetScreen)
{
    if (_screenGeometryConnection)
    {
        QObject::disconnect(_screenGeometryConnection);
        _screenGeometryConnection = {};
    }
    if (!targetScreen)
    {
        return;
    }

    _screenGeometryConnection = connect(targetScreen, &QScreen::availableGeometryChanged, this, [this]()
    {
        refreshScrollableDialogSize();
    });
}

void GenerateModelDialog::refreshScrollableDialogSize()
{
    const int visibleWidth = isVisible() ? width() : 0;
    updateScrollableContentHeight();
    if (layout())
    {
        layout()->invalidate();
        layout()->activate();
    }
    adjustSize();
    if (visibleWidth > 0)
    {
        resize(visibleWidth, height());
    }
}

void GenerateModelDialog::updateScrollableContentHeight()
{
    if (!_contentScrollArea || !_contentScrollArea->widget())
    {
        return;
    }

    const int preferredHeight = _contentScrollArea->widget()->sizeHint().height();
    const QScreen *targetScreen = screen();
    const int availableHeight = targetScreen ? targetScreen->availableGeometry().height() : preferredHeight;
    const int maximumContentHeight = qMax(240, availableHeight - 70);
    const int contentHeight = qMin(preferredHeight + 24, maximumContentHeight);
    _contentScrollArea->setMinimumHeight(contentHeight);
    _contentScrollArea->setMaximumHeight(contentHeight);
    _contentScrollArea->updateGeometry();
}

void GenerateModelDialog::updateAvailability()
{
    const QJsonObject candidate = currentCandidate();
    const bool supported = candidate.value(QLatin1String(kSupported)).toBool(false);
    const QString sourceData = candidate.value(QLatin1String(kSourceData)).toString();
    const bool hasCandidate = !sourceData.isEmpty();

    const QString requested_depth_quality =
        xjw::core::project::depthQualityProfileForModelQuality(
            _qualityCombo->currentData().toString());
    const auto depth_parameters = xjw::core::project::depthQualityParameters(
        xjw::core::project::depthQualityProfileFromId(requested_depth_quality));
    const QString stored_depth_quality = candidate.value(
        QLatin1String(kDepthQualityProfile)).toString();
    const bool stored_quality_known = !stored_depth_quality.isEmpty();
    const bool stored_quality_sufficient = !stored_quality_known ||
        xjw::core::project::depthQualityRank(stored_depth_quality) >=
            xjw::core::project::depthQualityRank(requested_depth_quality);
    const bool selected_depth_batch_compatible =
        candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true);
    const bool can_reuse_depth_maps =
        _hasReusableDepthMaps && selected_depth_batch_compatible &&
        stored_quality_sufficient;

    _effectiveDepthQualityLabel->setText(
        tr("%1（比例 %2，%3 轮，%4×%4 邻域，配置源视角 %5）")
            .arg(depthQualityDisplayName(requested_depth_quality))
            .arg(depth_parameters.resScale, 0, 'g', 3)
            .arg(depth_parameters.iterations)
            .arg(depth_parameters.patchSize)
            .arg(depth_parameters.minViews));
    _effectiveDepthQualityLabel->setToolTip(
        tr("场景分类和影像对质量仍可能限制实际源视角数；运行记录会同时保存配置值与实际值。"));

    _reuseDepthMapsCheck->setEnabled(can_reuse_depth_maps);
    if (sourceData == QStringLiteral("depth_maps") &&
        !selected_depth_batch_compatible)
    {
        _reuseDepthMapsRequested = false;
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(false);
        const QString incompatibility_reason = candidate.value(
            QLatin1String(kDepthBatchCompatibilityReason)).toString();
        _reuseDepthMapsCheck->setToolTip(incompatibility_reason.isEmpty()
            ? tr("现有深度图与当前工程不兼容；将自动重新估计深度图。")
            : tr("现有深度图与当前工程不兼容；将自动重新估计：%1")
                  .arg(incompatibility_reason));
    }
    else if (_hasReusableDepthMaps && !stored_quality_sufficient)
    {
        _reuseDepthMapsRequested = false;
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(false);
        _reuseDepthMapsCheck->setToolTip(
            tr("现有深度图质量为“%1”，低于当前请求的“%2”；将按当前质量重新计算。")
                .arg(depthQualityDisplayName(stored_depth_quality),
                     depthQualityDisplayName(requested_depth_quality)));
    }
    else if (_hasReusableDepthMaps)
    {
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested);
        _reuseDepthMapsCheck->setToolTip(stored_quality_known
            ? tr("复用质量为“%1”的兼容深度图。")
                  .arg(depthQualityDisplayName(stored_depth_quality))
            : tr("复用兼容深度图；旧批次未记录质量档位。"));
    }
    else
    {
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(false);
        _reuseDepthMapsCheck->setToolTip(
            tr("当前项目没有可复用的深度图；生成模型时将先自动估计深度图。"));
    }

    updateBlockControlsAvailability();
    _okButton->setEnabled(hasCandidate && supported);
}

void GenerateModelDialog::updateBlockControlsAvailability()
{
    const QString sourceData = _sourceCombo->currentData().toString();
    const bool blockCapable =
        sourceData == QStringLiteral("depth_maps") || sourceData == QStringLiteral("point_cloud");
    const bool splitEnabled = blockCapable && _splitRegionCheck->isChecked();

    _splitRegionCheck->setEnabled(blockCapable);
    _coordinateLabel->setEnabled(splitEnabled);
    _blockSizeSpin->setEnabled(splitEnabled);
    _originLabel->setEnabled(splitEnabled);
    _skipBoundaryBlocksCheck->setEnabled(splitEnabled);
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

    accept();
    emit runRequested(settings);
}

void GenerateModelDialog::onSourceTypeChanged()
{
    refreshSourceItems();
}
