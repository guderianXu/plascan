#include "reconstruction/GenerateModelDialog.h"
#include "shared/WorkflowParameterDialogStyle.h"
#include "PointCloudWorkflowConfig.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

#include <cmath>
#include <QWindow>

namespace
{

    constexpr const char* kSourceData = "source_data";
    constexpr const char* kSourceLabel = "source_label";
    constexpr const char* kSourcePath = "source_path";
    constexpr const char* kDisplay = "display";
    constexpr const char* kSupported = "supported";
    constexpr const char* kNote = "note";
    constexpr const char* kAutomaticDepthMaps = "automatic_depth_maps";
    constexpr const char* kDepthQualityProfile = "depth_quality_profile";
    constexpr const char* kDepthBatchCompatible = "depth_batch_compatible";
    constexpr const char* kDepthBatchCompatibilityReason = "depth_batch_compatibility_reason";
    constexpr const char* kComputeMode = "compute_mode";

    QString defaultSourceLabel(const QString& sourceData)
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
        if (sourceData == QStringLiteral("rpc_height_plane_sweep"))
        {
            return QStringLiteral("RPC 高程平面扫描");
        }
        return sourceData;
    }

    bool hasValidRpcImagePair(const QJsonObject& candidate)
    {
        const QJsonArray paths = candidate.value(QStringLiteral("rpcImagePaths")).toArray();
        return paths.size() == 2 && paths.at(0).isString() && paths.at(1).isString() &&
               !paths.at(0).toString().trimmed().isEmpty() && !paths.at(1).toString().trimmed().isEmpty();
    }

} // namespace

GenerateModelDialog::GenerateModelDialog(QWidget* parent) : QDialog(parent)
{
    setWindowTitle(tr("生成模型"));
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);

    auto* mainLayout = new QVBoxLayout(this);
    xjw::gui::dialogs::configureWorkflowDialogLayout(mainLayout);

    _contentScrollArea = new QScrollArea(this);
    _contentScrollArea->setObjectName(QStringLiteral("workflowParameterScrollArea"));
    xjw::gui::dialogs::configureWorkflowScrollArea(_contentScrollArea);
    auto* contentWidget = new QWidget(_contentScrollArea);
    contentWidget->setObjectName(QStringLiteral("workflowParameterContent"));
    auto* contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setContentsMargins(0, 0, 0, 0);
    contentLayout->setSpacing(9);

    auto* generalGroup = new QGroupBox(tr("一般"), contentWidget);
    generalGroup->setObjectName(QStringLiteral("workflowGeneralGroup"));
    auto* generalForm = new QFormLayout(generalGroup);
    xjw::gui::dialogs::configureWorkflowForm(generalForm);

    _sourceCombo = new QComboBox(generalGroup);
    _sourceItemCombo = new QComboBox(generalGroup);
    _faceCountModeCombo = new QComboBox(generalGroup);
    _customFaceCountSpin = new QSpinBox(generalGroup);
    _rpcHeightMinSpin = new QDoubleSpinBox(generalGroup);
    _rpcHeightMaxSpin = new QDoubleSpinBox(generalGroup);

    _sourceCombo->setObjectName(QStringLiteral("modelSourceCombo"));
    _sourceItemCombo->setObjectName(QStringLiteral("modelSourceItemCombo"));
    _sourceItemCombo->hide();
    _faceCountModeCombo->setObjectName(QStringLiteral("modelFaceCountModeCombo"));
    _customFaceCountSpin->setObjectName(QStringLiteral("modelCustomFaceCountSpin"));
    _rpcHeightMinSpin->setObjectName(QStringLiteral("rpcHeightMinMetersSpin"));
    _rpcHeightMaxSpin->setObjectName(QStringLiteral("rpcHeightMaxMetersSpin"));
    for (QComboBox* comboBox : {_sourceCombo, _sourceItemCombo, _faceCountModeCombo})
    {
        xjw::gui::dialogs::configureWorkflowComboBox(comboBox);
    }

    _effectiveDepthQualityLabel = new QLabel(generalGroup);
    _effectiveDepthQualityLabel->setObjectName(QStringLiteral("effectiveDepthQualityLabel"));
    _effectiveDepthQualityLabel->setWordWrap(true);

    _effectiveSurfaceQualityLabel = new QLabel(generalGroup);
    _effectiveSurfaceQualityLabel->setObjectName(QStringLiteral("effectiveSurfaceQualityLabel"));
    _effectiveSurfaceQualityLabel->setWordWrap(true);
    _faceCountModeCombo->addItem(tr("低 (20,000)"), QStringLiteral("low"));
    _faceCountModeCombo->addItem(tr("中 (100,000)"), QStringLiteral("medium"));
    _faceCountModeCombo->addItem(tr("高 (200,000)"), QStringLiteral("high"));
    _faceCountModeCombo->addItem(tr("自定义"), QStringLiteral("custom"));
    _faceCountModeCombo->setCurrentIndex(2);
    _customFaceCountSpin->setRange(1, 2000000);
    _customFaceCountSpin->setValue(200000);
    _customFaceCountSpin->setSuffix(tr(" 面"));
    for (QDoubleSpinBox* spinBox : {_rpcHeightMinSpin, _rpcHeightMaxSpin})
    {
        spinBox->setRange(-1000000.0, 1000000.0);
        spinBox->setDecimals(3);
        spinBox->setSuffix(tr(" m"));
        spinBox->setVisible(false);
    }

    generalForm->addRow(tr("源数据:"), _sourceCombo);
    generalForm->addRow(tr("深度质量:"), _effectiveDepthQualityLabel);
    generalForm->addRow(tr("表面算法:"), _effectiveSurfaceQualityLabel);
    generalForm->addRow(tr("面数:"), _faceCountModeCombo);
    generalForm->addRow(tr("自定义面数:"), _customFaceCountSpin);
    generalForm->addRow(tr("RPC 最低高程:"), _rpcHeightMinSpin);
    generalForm->addRow(tr("RPC 最高高程:"), _rpcHeightMaxSpin);
    contentLayout->addWidget(generalGroup);

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
    auto* advancedForm = new QFormLayout(_advancedContent);
    xjw::gui::dialogs::configureWorkflowForm(advancedForm);

    _reuseDepthMapsCheck = new QCheckBox(tr("重用深度图"), _advancedContent);
    _reuseDepthMapsCheck->setObjectName(QStringLiteral("reuseDepthMapsCheck"));
    _replaceDefaultCheck = new QCheckBox(tr("替换默认模型"), _advancedContent);
    _reuseDepthMapsCheck->setChecked(true);

    advancedForm->addRow(QString(), _reuseDepthMapsCheck);
    advancedForm->addRow(QString(), _replaceDefaultCheck);
    contentLayout->addWidget(_advancedContent);

    _contentScrollArea->setWidget(contentWidget);
    mainLayout->addWidget(_contentScrollArea);

    auto* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttonBox->setObjectName(QStringLiteral("workflowButtonBox"));
    xjw::gui::dialogs::configureWorkflowButtonBox(buttonBox, tr("生成"));
    _okButton = buttonBox->button(QDialogButtonBox::Ok);
    mainLayout->addWidget(buttonBox);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &GenerateModelDialog::onRun);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(_advancedToggle, &QToolButton::toggled, this, [this](bool expanded) { setAdvancedExpanded(expanded); });

    connect(_sourceCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::onSourceTypeChanged);
    connect(_sourceItemCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_faceCountModeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(
        _customFaceCountSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &GenerateModelDialog::emitSettingsNow);
    connect(_rpcHeightMinSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_rpcHeightMaxSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this,
            &GenerateModelDialog::emitSettingsNow);
    connect(_reuseDepthMapsCheck,
            &QCheckBox::toggled,
            this,
            [this](bool checked)
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

void GenerateModelDialog::showEvent(QShowEvent* event)
{
    QDialog::showEvent(event);
    if (!_screenChangeConnected && windowHandle())
    {
        connect(windowHandle(),
                &QWindow::screenChanged,
                this,
                [this](QScreen* targetScreen)
                {
                    bindScreenGeometryUpdates(targetScreen);
                    refreshScrollableDialogSize();
                });
        _screenChangeConnected = true;
    }

    bindScreenGeometryUpdates(screen());
    refreshScrollableDialogSize();
}

void GenerateModelDialog::applySettings(const QJsonObject& settings)
{
    _pendingSourceData = settings.value(QLatin1String(kSourceData)).toString();
    _pendingSourcePath = settings.value(QLatin1String(kSourcePath))
                             .toString(settings.value(QStringLiteral("denseCloudPath")).toString());
    QString compute_mode = settings.value(QLatin1String(kComputeMode)).toString().trimmed().toLower();
    if (compute_mode.isEmpty())
    {
        const QString backend = settings.value(QStringLiteral("patch_match_backend"))
                                    .toString(settings.value(QStringLiteral("processingDevice")).toString())
                                    .trimmed()
                                    .toLower();
        compute_mode = backend == QStringLiteral("cuda") || backend == QStringLiteral("opencl")
                           ? backend
                           : QStringLiteral("hybrid");
    }
    if (compute_mode == QStringLiteral("cuda") || compute_mode == QStringLiteral("opencl") ||
        compute_mode == QStringLiteral("hybrid"))
    {
        _computeMode = compute_mode;
    }

    const QString face_count_mode = settings.value(QStringLiteral("faceCountMode")).toString().trimmed();
    const int mode_index = _faceCountModeCombo->findData(face_count_mode);
    if (mode_index >= 0)
    {
        _faceCountModeCombo->setCurrentIndex(mode_index);
    }
    else if (settings.contains(QStringLiteral("targetFaces")))
    {
        const int legacy_faces = settings.value(QStringLiteral("targetFaces")).toInt();
        const QString legacy_mode = legacy_faces <= 20000    ? QStringLiteral("low")
                                    : legacy_faces <= 100000 ? QStringLiteral("medium")
                                    : legacy_faces <= 200000 ? QStringLiteral("high")
                                                             : QStringLiteral("custom");
        _faceCountModeCombo->setCurrentIndex(_faceCountModeCombo->findData(legacy_mode));
        _customFaceCountSpin->setValue(qBound(1, legacy_faces, 2000000));
    }
    _customFaceCountSpin->setValue(
        qBound(1, settings.value(QStringLiteral("faceCountCustom")).toInt(_customFaceCountSpin->value()), 2000000));
    if (settings.contains(QStringLiteral("rpcHeightMinMeters")))
    {
        _rpcHeightMinSpin->setValue(settings.value(QStringLiteral("rpcHeightMinMeters")).toDouble());
    }
    if (settings.contains(QStringLiteral("rpcHeightMaxMeters")))
    {
        _rpcHeightMaxSpin->setValue(settings.value(QStringLiteral("rpcHeightMaxMeters")).toDouble());
    }
    _reuseDepthMapsRequested = settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    _reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested);
    _replaceDefaultCheck->setChecked(settings.value(QStringLiteral("replaceDefaultModel")).toBool(false));
}

void GenerateModelDialog::setSourceCandidates(const QJsonArray& candidates)
{
    _candidates = QJsonArray();
    _hasReusableDepthMaps = false;
    bool has_depth_candidate = false;
    bool has_rpc_candidate = false;
    bool can_generate_depth_maps = false;
    for (const QJsonValue& value : candidates)
    {
        const QJsonObject candidate = value.toObject();
        const QString source_data = candidate.value(QLatin1String(kSourceData)).toString();
        if (source_data == QStringLiteral("tie_points") && candidate.value(QLatin1String(kSupported)).toBool(false))
        {
            can_generate_depth_maps = true;
        }
        if (source_data == QStringLiteral("depth_maps"))
        {
            _candidates.append(candidate);
            has_depth_candidate = true;
            if (candidate.value(QLatin1String(kSupported)).toBool(false) &&
                candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true) &&
                !candidate.value(QLatin1String(kSourcePath)).toString().trimmed().isEmpty())
            {
                _hasReusableDepthMaps = true;
            }
            if (candidate.contains(QStringLiteral("rpcHeightMinMeters")))
            {
                _rpcHeightMinSpin->setValue(candidate.value(QStringLiteral("rpcHeightMinMeters")).toDouble());
            }
            if (candidate.contains(QStringLiteral("rpcHeightMaxMeters")))
            {
                _rpcHeightMaxSpin->setValue(candidate.value(QStringLiteral("rpcHeightMaxMeters")).toDouble());
            }
        }
        else if (source_data == QStringLiteral("rpc_height_plane_sweep") && hasValidRpcImagePair(candidate))
        {
            _candidates.append(candidate);
            has_rpc_candidate = true;
            if (candidate.contains(QStringLiteral("rpcHeightMinMeters")))
            {
                _rpcHeightMinSpin->setValue(candidate.value(QStringLiteral("rpcHeightMinMeters")).toDouble());
            }
            if (candidate.contains(QStringLiteral("rpcHeightMaxMeters")))
            {
                _rpcHeightMaxSpin->setValue(candidate.value(QStringLiteral("rpcHeightMaxMeters")).toDouble());
            }
        }
    }

    // 仅针孔候选可自动估计深度图后进入 recovered OOC；不允许回落到旧表面算法。
    if (!has_depth_candidate && !has_rpc_candidate)
    {
        QJsonObject automatic_depth_maps;
        automatic_depth_maps[QLatin1String(kSourceData)] = QStringLiteral("depth_maps");
        automatic_depth_maps[QLatin1String(kSourceLabel)] = QStringLiteral("深度图");
        automatic_depth_maps[QLatin1String(kSourcePath)] = QString();
        automatic_depth_maps[QLatin1String(kDisplay)] = QStringLiteral("自动生成深度图");
        automatic_depth_maps[QLatin1String(kSupported)] = can_generate_depth_maps;
        automatic_depth_maps[QLatin1String(kAutomaticDepthMaps)] = true;
        automatic_depth_maps[QLatin1String(kNote)] =
            can_generate_depth_maps ? QStringLiteral("缺少深度图时将自动估计深度图，再执行参考已验证的 recovered OOC。")
                                    : QStringLiteral("需要先完成通过质量门控的照片对齐，才能自动估计深度图。");
        _candidates.prepend(automatic_depth_maps);

        if (_pendingSourceData.isEmpty() || _pendingSourceData == QStringLiteral("tie_points"))
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
    const QString face_count_mode = _faceCountModeCombo->currentData().toString();
    const int custom_faces = _customFaceCountSpin->value();
    const int target_faces = face_count_mode == QStringLiteral("low")      ? 20000
                             : face_count_mode == QStringLiteral("medium") ? 100000
                             : face_count_mode == QStringLiteral("high")   ? 200000
                                                                           : custom_faces;
    const bool rpc_mode = usesRpcHeightPlaneSweep();

    QJsonObject settings;
    settings[QStringLiteral("modelGenerationContractRevision")] = 1;
    settings[QLatin1String(kSourceData)] = sourceData;
    settings[QLatin1String(kSourceLabel)] =
        candidate.value(QLatin1String(kSourceLabel)).toString(defaultSourceLabel(sourceData));
    settings[QLatin1String(kSourcePath)] = sourcePath;
    settings[QStringLiteral("source_display")] = candidate.value(QLatin1String(kDisplay)).toString();
    settings[QStringLiteral("source_supported")] = candidate.value(QLatin1String(kSupported)).toBool(false);
    settings[QStringLiteral("depthQualityProfile")] = QStringLiteral("medium");
    settings[QStringLiteral("surfaceQualityProfile")] =
        rpc_mode ? QStringLiteral("rpc_height_plane_sweep") : QStringLiteral("recovered_ooc");
    settings[QStringLiteral("reconstruction_mode")] =
        rpc_mode ? QStringLiteral("rpc_height_plane_sweep") : QStringLiteral("recovered_ooc");
    settings[QStringLiteral("faceCountMode")] = face_count_mode;
    settings[QStringLiteral("faceCountCustom")] = custom_faces;
    settings[QStringLiteral("simplifyTargetFaces")] = target_faces;
    settings[QStringLiteral("requestedTargetFaces")] = target_faces;
    settings[QStringLiteral("export_format")] = QStringLiteral("PLY");
    settings[QStringLiteral("depthFiltering")] = QStringLiteral("mild");
    if (rpc_mode)
    {
        settings[QStringLiteral("rpcImagePaths")] = candidate.value(QStringLiteral("rpcImagePaths"));
        settings[QStringLiteral("rpcHeightMinMeters")] = _rpcHeightMinSpin->value();
        settings[QStringLiteral("rpcHeightMaxMeters")] = _rpcHeightMaxSpin->value();
    }
    const bool selected_depth_batch_compatible = candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true);
    const bool reuse_depth_maps = _hasReusableDepthMaps && selected_depth_batch_compatible && _reuseDepthMapsRequested;
    settings[QStringLiteral("reuseDepthMaps")] = reuse_depth_maps;
    const bool automatic_depth_maps = candidate.value(QLatin1String(kAutomaticDepthMaps)).toBool(false);
    settings[QStringLiteral("automatic_depth_maps")] = automatic_depth_maps;
    settings[QStringLiteral("force_depth_recompute")] =
        automatic_depth_maps || (sourceData == QStringLiteral("depth_maps") && !reuse_depth_maps);
    settings[QStringLiteral("replaceDefaultModel")] = _replaceDefaultCheck->isChecked();
    settings[QLatin1String(kComputeMode)] = _computeMode;
    settings[QStringLiteral("patch_match_backend")] =
        _computeMode == QStringLiteral("hybrid") ? QStringLiteral("auto") : _computeMode;
    settings[QStringLiteral("processingDevice")] =
        _computeMode == QStringLiteral("hybrid") ? QStringLiteral("auto") : _computeMode;

    if (sourceData == QStringLiteral("point_cloud") || sourceData == QStringLiteral("tie_points") ||
        sourceData == QStringLiteral("model"))
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

bool GenerateModelDialog::usesRecoveredModelPipeline() const
{
    return currentCandidate().value(QLatin1String(kSourceData)).toString() == QStringLiteral("depth_maps") &&
           !usesRpcHeightPlaneSweep();
}

bool GenerateModelDialog::usesRpcHeightPlaneSweep() const
{
    const QJsonObject candidate = currentCandidate();
    return candidate.value(QLatin1String(kSourceData)).toString() == QStringLiteral("rpc_height_plane_sweep") &&
           hasValidRpcImagePair(candidate);
}

void GenerateModelDialog::refreshSourceTypes()
{
    const QString currentSource =
        !_pendingSourceData.isEmpty() ? _pendingSourceData : _sourceCombo->currentData().toString();

    _sourceCombo->blockSignals(true);
    _sourceCombo->clear();

    QStringList addedTypes;
    for (const QJsonValue& value : _candidates)
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

    for (const QJsonValue& value : _candidates)
    {
        const QJsonObject candidate = value.toObject();
        if (candidate.value(QLatin1String(kSourceData)).toString() != sourceData)
        {
            continue;
        }

        const QString display =
            candidate.value(QLatin1String(kDisplay)).toString(candidate.value(QLatin1String(kSourcePath)).toString());
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

void GenerateModelDialog::bindScreenGeometryUpdates(QScreen* targetScreen)
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

    _screenGeometryConnection =
        connect(targetScreen, &QScreen::availableGeometryChanged, this, [this]() { refreshScrollableDialogSize(); });
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
    const QScreen* targetScreen = screen();
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
    const bool selected_depth_batch_compatible = candidate.value(QLatin1String(kDepthBatchCompatible)).toBool(true);
    const bool can_reuse_depth_maps = _hasReusableDepthMaps && selected_depth_batch_compatible;
    const bool rpc_mode = usesRpcHeightPlaneSweep();
    _effectiveDepthQualityLabel->setText(rpc_mode ? tr("RPC 高程平面扫描（需物理高程范围）")
                                                  : tr("中（d4，参考已验证）"));
    _effectiveSurfaceQualityLabel->setText(rpc_mode ? tr("RPC 高程平面扫描") : tr("recovered_ooc（参考已验证链）"));
    _customFaceCountSpin->setVisible(_faceCountModeCombo->currentData().toString() == QStringLiteral("custom"));
    _rpcHeightMinSpin->setVisible(rpc_mode);
    _rpcHeightMaxSpin->setVisible(rpc_mode);
    _reuseDepthMapsCheck->setEnabled(can_reuse_depth_maps && !rpc_mode);
    if (sourceData == QStringLiteral("depth_maps") && !selected_depth_batch_compatible)
    {
        _reuseDepthMapsRequested = false;
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(false);
        const QString incompatibility_reason =
            candidate.value(QLatin1String(kDepthBatchCompatibilityReason)).toString();
        _reuseDepthMapsCheck->setToolTip(
            incompatibility_reason.isEmpty()
                ? tr("现有深度图与当前工程不兼容；将自动重新估计深度图。")
                : tr("现有深度图与当前工程不兼容；将自动重新估计：%1").arg(incompatibility_reason));
    }
    else if (_hasReusableDepthMaps)
    {
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested);
        _reuseDepthMapsCheck->setToolTip(tr("复用兼容深度图；实际执行参数将写入模型记录。"));
    }
    else
    {
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(false);
        _reuseDepthMapsCheck->setToolTip(tr("当前项目没有可复用的深度图；生成模型时将先自动估计深度图。"));
    }

    const bool rpc_source = sourceData == QStringLiteral("rpc_height_plane_sweep");
    const bool valid_rpc_configuration = !rpc_source ||
                                         (rpc_mode && std::isfinite(_rpcHeightMinSpin->value()) &&
                                          std::isfinite(_rpcHeightMaxSpin->value()) &&
                                          _rpcHeightMinSpin->value() < _rpcHeightMaxSpin->value());
    _okButton->setEnabled(hasCandidate && supported && valid_rpc_configuration);
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
    if (currentCandidate().value(QLatin1String(kSourceData)).toString() == QStringLiteral("rpc_height_plane_sweep") &&
        (settings.value(QStringLiteral("reconstruction_mode")).toString() != QStringLiteral("rpc_height_plane_sweep") ||
         !hasValidRpcImagePair(currentCandidate()) ||
         !std::isfinite(settings.value(QStringLiteral("rpcHeightMinMeters")).toDouble()) ||
         !std::isfinite(settings.value(QStringLiteral("rpcHeightMaxMeters")).toDouble()) ||
         settings.value(QStringLiteral("rpcHeightMinMeters")).toDouble() >=
             settings.value(QStringLiteral("rpcHeightMaxMeters")).toDouble()))
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
