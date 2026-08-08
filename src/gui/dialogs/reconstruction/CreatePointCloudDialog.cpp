#include "reconstruction/CreatePointCloudDialog.h"

#include "shared/WorkflowParameterDialogStyle.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QVBoxLayout>

CreatePointCloudDialog::CreatePointCloudDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("创建点云"));
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);

    auto *main_layout = new QVBoxLayout(this);
    xjw::gui::dialogs::configureWorkflowDialogLayout(main_layout);

    auto *general_group = new QGroupBox(tr("一般"), this);
    general_group->setObjectName(QStringLiteral("pointCloudGeneralGroup"));
    auto *general_form = new QFormLayout(general_group);
    xjw::gui::dialogs::configureWorkflowForm(general_form);

    auto *source_label = new QLabel(tr("深度图（由正式空三结果生成）"), general_group);
    source_label->setObjectName(QStringLiteral("pointCloudSourceValueLabel"));
    source_label->setWordWrap(true);

    _qualityCombo = new QComboBox(general_group);
    _qualityCombo->setObjectName(QStringLiteral("pointCloudQualityCombo"));
    _qualityCombo->addItem(tr("超高"), QStringLiteral("highest"));
    _qualityCombo->addItem(tr("高"), QStringLiteral("high"));
    _qualityCombo->addItem(tr("中"), QStringLiteral("medium"));
    _qualityCombo->addItem(tr("低"), QStringLiteral("low"));
    _qualityCombo->addItem(tr("最低"), QStringLiteral("lowest"));
    xjw::gui::dialogs::configureWorkflowComboBox(_qualityCombo);

    _reuseDepthMapsCheck = new QCheckBox(tr("重用深度图"), general_group);
    _reuseDepthMapsCheck->setObjectName(QStringLiteral("reuseDepthMapsCheck"));
    _reuseDepthMapsCheck->setChecked(true);
    _saveEachStepCheck = new QCheckBox(tr("在每个步骤完成后保存项目"), general_group);
    _saveEachStepCheck->setObjectName(QStringLiteral("savePointCloudEachStepCheck"));

    general_form->addRow(tr("源数据:"), source_label);
    general_form->addRow(tr("质量:"), _qualityCombo);
    general_form->addRow(QString(), _reuseDepthMapsCheck);
    general_form->addRow(QString(), _saveEachStepCheck);
    main_layout->addWidget(general_group);

    auto *advanced_group = new QGroupBox(tr("高级"), this);
    advanced_group->setObjectName(QStringLiteral("pointCloudAdvancedGroup"));
    auto *advanced_form = new QFormLayout(advanced_group);
    xjw::gui::dialogs::configureWorkflowForm(advanced_form);

    _depthFilterCombo = new QComboBox(advanced_group);
    _depthFilterCombo->setObjectName(QStringLiteral("pointCloudDepthFilterCombo"));
    _depthFilterCombo->addItem(tr("温和"), QStringLiteral("mild"));
    _depthFilterCombo->addItem(tr("中等"), QStringLiteral("moderate"));
    _depthFilterCombo->addItem(tr("强"), QStringLiteral("aggressive"));
    xjw::gui::dialogs::configureWorkflowComboBox(_depthFilterCombo);

    _calculateColorsCheck = new QCheckBox(tr("计算点颜色"), advanced_group);
    _calculateColorsCheck->setObjectName(QStringLiteral("calculatePointColorsCheck"));
    _calculateColorsCheck->setChecked(true);
    auto *confidence_note = new QLabel(
        tr("当前点云格式暂不保存逐点置信度。"), advanced_group);
    confidence_note->setObjectName(QStringLiteral("pointCloudConfidenceNote"));
    confidence_note->setWordWrap(true);
    _replaceDefaultCheck = new QCheckBox(tr("替换默认点云"), advanced_group);
    _replaceDefaultCheck->setObjectName(QStringLiteral("replaceDefaultPointCloudCheck"));

    for (QCheckBox *check_box : {
             _reuseDepthMapsCheck,
             _saveEachStepCheck,
             _calculateColorsCheck,
             _replaceDefaultCheck})
    {
        xjw::gui::dialogs::configureWorkflowCheckBox(check_box);
    }

    advanced_form->addRow(tr("深度过滤:"), _depthFilterCombo);
    advanced_form->addRow(QString(), _calculateColorsCheck);
    advanced_form->addRow(tr("置信度:"), confidence_note);
    advanced_form->addRow(QString(), _replaceDefaultCheck);
    main_layout->addWidget(advanced_group);

    _statusLabel = new QLabel(this);
    _statusLabel->setObjectName(QStringLiteral("workflowStatusLabel"));
    _statusLabel->setWordWrap(true);
    main_layout->addWidget(_statusLabel);

    auto *button_box = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    button_box->setObjectName(QStringLiteral("workflowButtonBox"));
    xjw::gui::dialogs::configureWorkflowButtonBox(button_box);
    _okButton = button_box->button(QDialogButtonBox::Ok);
    main_layout->addWidget(button_box);

    connect(button_box, &QDialogButtonBox::accepted,
            this, &CreatePointCloudDialog::onRun);
    connect(button_box, &QDialogButtonBox::rejected,
            this, &QDialog::reject);

    for (QComboBox *combo_box : {_qualityCombo, _depthFilterCombo})
    {
        connect(combo_box, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &CreatePointCloudDialog::emitSettingsNow);
    }
    connect(_reuseDepthMapsCheck, &QCheckBox::toggled, this, [this](bool checked)
    {
        if (_hasReusableDepthMaps)
        {
            _reuseDepthMapsRequested = checked;
        }
        emitSettingsNow();
    });
    connect(_replaceDefaultCheck, &QCheckBox::toggled, this, [this](bool checked)
    {
        if (_hasExistingPointCloud)
        {
            _replaceDefaultRequested = checked;
        }
        emitSettingsNow();
    });
    for (QCheckBox *check_box : {
             _saveEachStepCheck,
             _calculateColorsCheck})
    {
        connect(check_box, &QCheckBox::toggled,
                this, &CreatePointCloudDialog::emitSettingsNow);
    }

    updateAvailability();
    adjustSize();
}

void CreatePointCloudDialog::applySettings(const QJsonObject &settings)
{
    const QSignalBlocker quality_blocker(_qualityCombo);
    const QSignalBlocker reuse_blocker(_reuseDepthMapsCheck);
    const QSignalBlocker save_blocker(_saveEachStepCheck);
    const QSignalBlocker filter_blocker(_depthFilterCombo);
    const QSignalBlocker colors_blocker(_calculateColorsCheck);
    const QSignalBlocker replace_blocker(_replaceDefaultCheck);

    const int quality_index = _qualityCombo->findData(
        settings.value(QStringLiteral("qualityProfile")).toString());
    if (quality_index >= 0)
    {
        _qualityCombo->setCurrentIndex(quality_index);
    }

    const int filter_index = _depthFilterCombo->findData(
        settings.value(QStringLiteral("depthFilterMode")).toString());
    if (filter_index >= 0)
    {
        _depthFilterCombo->setCurrentIndex(filter_index);
    }

    _reuseDepthMapsRequested =
        settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    _reuseDepthMapsCheck->setChecked(_reuseDepthMapsRequested);
    _saveEachStepCheck->setChecked(
        settings.value(QStringLiteral("saveAfterEachStep")).toBool(false));
    _calculateColorsCheck->setChecked(
        settings.value(QStringLiteral("calculatePointColors")).toBool(true));
    _replaceDefaultRequested =
        settings.value(QStringLiteral("replaceDefaultPointCloud")).toBool(false);
    _replaceDefaultCheck->setChecked(_replaceDefaultRequested);
    updateAvailability();
}

void CreatePointCloudDialog::setProjectState(bool hasProductionSparseResult,
                                             bool hasReusableDepthMaps,
                                             bool hasExistingPointCloud,
                                             const QString &blockingReason)
{
    _hasProductionSparseResult = hasProductionSparseResult;
    _hasReusableDepthMaps = hasReusableDepthMaps;
    _hasExistingPointCloud = hasExistingPointCloud;
    _blockingReason = blockingReason.trimmed();
    updateAvailability();
}

QJsonObject CreatePointCloudDialog::collectSettings() const
{
    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("qualityProfile")] = _qualityCombo->currentData().toString();
    settings[QStringLiteral("depthFilterMode")] = _depthFilterCombo->currentData().toString();
    settings[QStringLiteral("reuseDepthMaps")] = _reuseDepthMapsCheck->isChecked();
    settings[QStringLiteral("force_depth_recompute")] = !_reuseDepthMapsCheck->isChecked();
    settings[QStringLiteral("saveAfterEachStep")] = _saveEachStepCheck->isChecked();
    settings[QStringLiteral("calculatePointColors")] = _calculateColorsCheck->isChecked();
    settings[QStringLiteral("keepColor")] = _calculateColorsCheck->isChecked();
    settings[QStringLiteral("calculatePointConfidence")] = false;
    settings[QStringLiteral("replaceDefaultPointCloud")] =
        _replaceDefaultCheck->isChecked();
    return settings;
}

void CreatePointCloudDialog::emitSettingsNow()
{
    updateAvailability();
    emit settingsChanged(collectSettings());
}

void CreatePointCloudDialog::updateAvailability()
{
    _reuseDepthMapsCheck->setEnabled(_hasReusableDepthMaps);
    {
        const QSignalBlocker blocker(_reuseDepthMapsCheck);
        _reuseDepthMapsCheck->setChecked(
            _hasReusableDepthMaps && _reuseDepthMapsRequested);
    }
    if (!_hasReusableDepthMaps)
    {
        _reuseDepthMapsCheck->setToolTip(
            tr("当前项目没有可复用深度图，将按所选质量重新计算。"));
    }
    else
    {
        _reuseDepthMapsCheck->setToolTip(
            tr("复用项目中与当前空三结果兼容的深度图。"));
    }

    _replaceDefaultCheck->setEnabled(_hasExistingPointCloud);
    {
        const QSignalBlocker blocker(_replaceDefaultCheck);
        _replaceDefaultCheck->setChecked(
            _hasExistingPointCloud && _replaceDefaultRequested);
    }

    _okButton->setEnabled(_hasProductionSparseResult);
    if (!_hasProductionSparseResult)
    {
        _statusLabel->setText(_blockingReason.isEmpty()
            ? tr("当前项目没有可用的正式空三结果，请先运行空中三角测量。")
            : _blockingReason);
        return;
    }

    _statusLabel->setText(_hasReusableDepthMaps
        ? tr("已检测到兼容深度图，可选择复用后创建点云。")
        : tr("将从正式空三结果估计深度图并创建点云。"));
}

void CreatePointCloudDialog::onRun()
{
    if (!_hasProductionSparseResult)
    {
        return;
    }

    const QJsonObject settings = collectSettings();
    accept();
    emit runRequested(settings);
}
