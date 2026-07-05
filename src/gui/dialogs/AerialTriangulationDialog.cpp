#include "AerialTriangulationDialog.h"
#include "ui_AerialTriangulationDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QJsonObject>
#include <QLayout>
#include <QSizePolicy>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QToolButton>

namespace
{

void setComboByData(QComboBox *combo, const QString &data)
{
    if (!combo)
    {
        return;
    }

    const int index = combo->findData(data);
    if (index >= 0)
    {
        combo->setCurrentIndex(index);
    }
}

QString comboDataOr(QComboBox *combo, const QString &fallback)
{
    if (!combo)
    {
        return fallback;
    }

    const QString value = combo->currentData().toString();
    return value.isEmpty() ? fallback : value;
}

QString normalizeReferenceSource(QString value)
{
    if (value == QStringLiteral("camera_pose"))
    {
        return QStringLiteral("estimated");
    }
    if (value == QStringLiteral("image_coordinates"))
    {
        return QStringLiteral("sequence");
    }
    return value;
}

void stabilizeInputControl(QWidget *widget)
{
    if (!widget)
    {
        return;
    }
    widget->setMinimumHeight(28);
    widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void stabilizeCheckBox(QCheckBox *checkBox)
{
    if (!checkBox)
    {
        return;
    }
    checkBox->setMinimumHeight(24);
    checkBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

} // namespace

AerialTriangulationDialog::AerialTriangulationDialog(QWidget *parent)
    : QDialog(parent),
      _ui(std::make_unique<Ui::AerialTriangulationDialog>())
{
    _ui->setupUi(this);
    setupUi();
}

AerialTriangulationDialog::~AerialTriangulationDialog() = default;

void AerialTriangulationDialog::setupUi()
{
    setWindowTitle(QStringLiteral("空中三角测量"));
    _ui->m_statusLabel->hide();

    _ui->m_qualityCombo->clear();
    _ui->m_qualityCombo->addItem(QStringLiteral("最高"), QStringLiteral("highest"));
    _ui->m_qualityCombo->addItem(QStringLiteral("高"), QStringLiteral("high"));
    _ui->m_qualityCombo->addItem(QStringLiteral("中"), QStringLiteral("medium"));
    _ui->m_qualityCombo->addItem(QStringLiteral("低"), QStringLiteral("low"));
    _ui->m_qualityCombo->addItem(QStringLiteral("最低"), QStringLiteral("lowest"));
    setComboByData(_ui->m_qualityCombo, QStringLiteral("high"));

    _ui->m_referenceSourceCombo->clear();
    _ui->m_referenceSourceCombo->addItem(QStringLiteral("源代码"), QStringLiteral("source_code"));
    _ui->m_referenceSourceCombo->addItem(QStringLiteral("估计"), QStringLiteral("estimated"));
    _ui->m_referenceSourceCombo->addItem(QStringLiteral("序列"), QStringLiteral("sequence"));
    setComboByData(_ui->m_referenceSourceCombo, QStringLiteral("source_code"));

    _ui->m_maskApplyCombo->clear();
    _ui->m_maskApplyCombo->addItem(QStringLiteral("无"), QStringLiteral("none"));
    _ui->m_maskApplyCombo->addItem(QStringLiteral("关键点"), QStringLiteral("keypoints"));
    _ui->m_maskApplyCombo->addItem(QStringLiteral("连接点"), QStringLiteral("tiepoints"));
    setComboByData(_ui->m_maskApplyCombo, QStringLiteral("none"));

    _ui->m_genericPreselectionCheck->setChecked(true);
    _ui->m_referencePreselectionCheck->setChecked(false);
    _ui->m_referenceSourceCombo->setEnabled(false);
    _ui->m_resetAlignmentCheck->setChecked(true);
    _ui->m_saveAfterEachStepCheck->setChecked(false);
    _ui->m_keypointLimitSpin->setRange(0, 1000000);
    _ui->m_keypointLimitSpin->setValue(40000);
    _ui->m_tiepointLimitSpin->setRange(0, 1000000);
    _ui->m_tiepointLimitSpin->setValue(4000);
    _ui->m_excludeFixedTiePointsCheck->setChecked(true);
    _ui->m_guidedImageMatchingCheck->setChecked(false);
    _ui->m_adaptiveCameraModelCheck->setChecked(false);
    stabilizeInputControl(_ui->m_qualityCombo);
    stabilizeInputControl(_ui->m_referenceSourceCombo);
    stabilizeInputControl(_ui->m_keypointLimitSpin);
    stabilizeInputControl(_ui->m_tiepointLimitSpin);
    stabilizeInputControl(_ui->m_maskApplyCombo);
    stabilizeCheckBox(_ui->m_genericPreselectionCheck);
    stabilizeCheckBox(_ui->m_referencePreselectionCheck);
    stabilizeCheckBox(_ui->m_resetAlignmentCheck);
    stabilizeCheckBox(_ui->m_saveAfterEachStepCheck);
    stabilizeCheckBox(_ui->m_excludeFixedTiePointsCheck);
    stabilizeCheckBox(_ui->m_guidedImageMatchingCheck);
    stabilizeCheckBox(_ui->m_adaptiveCameraModelCheck);
    setAdvancedExpanded(false);

    connect(_ui->m_buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(_ui->m_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(_ui->m_advancedToggle, &QToolButton::toggled, this, [this](bool expanded)
    {
        setAdvancedExpanded(expanded);
    });

    connect(_ui->m_referencePreselectionCheck, &QCheckBox::toggled, this, [this](bool enabled)
    {
        _ui->m_referenceSourceCombo->setEnabled(enabled);
        emitSettingsChanged();
    });

    connect(_ui->m_qualityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AerialTriangulationDialog::emitSettingsChanged);
    connect(_ui->m_referenceSourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AerialTriangulationDialog::emitSettingsChanged);
    connect(_ui->m_maskApplyCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &AerialTriangulationDialog::emitSettingsChanged);

    const auto connectCheckBox = [this](QCheckBox *checkBox)
    {
        connect(checkBox, &QCheckBox::toggled, this, &AerialTriangulationDialog::emitSettingsChanged);
    };
    connectCheckBox(_ui->m_genericPreselectionCheck);
    connectCheckBox(_ui->m_resetAlignmentCheck);
    connectCheckBox(_ui->m_saveAfterEachStepCheck);
    connectCheckBox(_ui->m_excludeFixedTiePointsCheck);
    connectCheckBox(_ui->m_guidedImageMatchingCheck);
    connectCheckBox(_ui->m_adaptiveCameraModelCheck);

    connect(_ui->m_keypointLimitSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &AerialTriangulationDialog::emitSettingsChanged);
    connect(_ui->m_tiepointLimitSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &AerialTriangulationDialog::emitSettingsChanged);
}

void AerialTriangulationDialog::setAdvancedExpanded(bool expanded)
{
    _ui->m_advancedToggle->setChecked(expanded);
    _ui->m_advancedToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    _ui->m_advancedContent->setVisible(expanded);
    setMinimumHeight(expanded ? 560 : 360);
    if (layout())
    {
        layout()->invalidate();
    }
    adjustSize();
}

void AerialTriangulationDialog::setImageCount(int count)
{
    _ui->m_statusLabel->setText(QStringLiteral("当前项目影像：%1 张").arg(qMax(0, count)));
}

void AerialTriangulationDialog::applySettings(const QJsonObject &settings)
{
    _applyingSettings = true;
    const QSignalBlocker blockQuality(_ui->m_qualityCombo);
    const QSignalBlocker blockGeneric(_ui->m_genericPreselectionCheck);
    const QSignalBlocker blockReference(_ui->m_referencePreselectionCheck);
    const QSignalBlocker blockReferenceSource(_ui->m_referenceSourceCombo);
    const QSignalBlocker blockReset(_ui->m_resetAlignmentCheck);
    const QSignalBlocker blockSave(_ui->m_saveAfterEachStepCheck);
    const QSignalBlocker blockKeypoint(_ui->m_keypointLimitSpin);
    const QSignalBlocker blockTiepoint(_ui->m_tiepointLimitSpin);
    const QSignalBlocker blockMask(_ui->m_maskApplyCombo);
    const QSignalBlocker blockExclude(_ui->m_excludeFixedTiePointsCheck);
    const QSignalBlocker blockGuided(_ui->m_guidedImageMatchingCheck);
    const QSignalBlocker blockAdaptive(_ui->m_adaptiveCameraModelCheck);

    setComboByData(_ui->m_qualityCombo, settings.value(QStringLiteral("quality")).toString(QStringLiteral("high")));
    _ui->m_genericPreselectionCheck->setChecked(
        settings.value(QStringLiteral("generic_preselection")).toBool(true));
    _ui->m_referencePreselectionCheck->setChecked(
        settings.value(QStringLiteral("reference_preselection")).toBool(false));
    const QString referenceSource = normalizeReferenceSource(
        settings.value(QStringLiteral("reference_preselection_source")).toString(QStringLiteral("source_code")));
    setComboByData(_ui->m_referenceSourceCombo, referenceSource);
    _ui->m_resetAlignmentCheck->setChecked(
        settings.value(QStringLiteral("reset_current_alignment")).toBool(true));
    _ui->m_saveAfterEachStepCheck->setChecked(
        settings.value(QStringLiteral("save_project_after_each_step")).toBool(false));
    _ui->m_keypointLimitSpin->setValue(settings.value(QStringLiteral("keypoint_limit")).toInt(40000));
    _ui->m_tiepointLimitSpin->setValue(settings.value(QStringLiteral("tiepoint_limit")).toInt(4000));
    setComboByData(_ui->m_maskApplyCombo,
                   settings.value(QStringLiteral("mask_apply_mode")).toString(QStringLiteral("none")));
    _ui->m_excludeFixedTiePointsCheck->setChecked(
        settings.value(QStringLiteral("exclude_fixed_tie_points")).toBool(true));
    _ui->m_guidedImageMatchingCheck->setChecked(
        settings.value(QStringLiteral("guided_image_matching")).toBool(false));
    _ui->m_adaptiveCameraModelCheck->setChecked(
        settings.value(QStringLiteral("adaptive_camera_model_fitting")).toBool(false));

    _ui->m_referenceSourceCombo->setEnabled(_ui->m_referencePreselectionCheck->isChecked());
    _applyingSettings = false;
}

QJsonObject AerialTriangulationDialog::collectSettings() const
{
    QJsonObject settings;
    settings[QStringLiteral("workflow_kind")] = QStringLiteral("aerial_triangulation_dialog_only");
    settings[QStringLiteral("quality")] = comboDataOr(_ui->m_qualityCombo, QStringLiteral("high"));
    settings[QStringLiteral("generic_preselection")] = _ui->m_genericPreselectionCheck->isChecked();
    settings[QStringLiteral("reference_preselection")] = _ui->m_referencePreselectionCheck->isChecked();
    settings[QStringLiteral("reference_preselection_source")] =
        comboDataOr(_ui->m_referenceSourceCombo, QStringLiteral("source_code"));
    settings[QStringLiteral("reset_current_alignment")] = _ui->m_resetAlignmentCheck->isChecked();
    settings[QStringLiteral("save_project_after_each_step")] = _ui->m_saveAfterEachStepCheck->isChecked();
    settings[QStringLiteral("keypoint_limit")] = _ui->m_keypointLimitSpin->value();
    settings[QStringLiteral("tiepoint_limit")] = _ui->m_tiepointLimitSpin->value();
    settings[QStringLiteral("mask_apply_mode")] = comboDataOr(_ui->m_maskApplyCombo, QStringLiteral("none"));
    settings[QStringLiteral("exclude_fixed_tie_points")] = _ui->m_excludeFixedTiePointsCheck->isChecked();
    settings[QStringLiteral("guided_image_matching")] = _ui->m_guidedImageMatchingCheck->isChecked();
    settings[QStringLiteral("adaptive_camera_model_fitting")] = _ui->m_adaptiveCameraModelCheck->isChecked();
    return settings;
}

void AerialTriangulationDialog::emitSettingsChanged()
{
    if (_applyingSettings)
    {
        return;
    }
    emit settingsChanged(collectSettings());
}
