#include "MaskEditorSettingsDialog.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QSettings>
#include <QSpinBox>
#include <QVBoxLayout>

namespace
{

    constexpr auto SettingsGroup = "mask_editor";

    QSpinBox* makeSpinBox(int minimum, int maximum, const QString& suffix, QWidget* parent)
    {
        auto* spinBox = new QSpinBox(parent);
        spinBox->setRange(minimum, maximum);
        spinBox->setSuffix(suffix);
        return spinBox;
    }

} // namespace

MaskEditorSettings loadMaskEditorSettings()
{
    QSettings settings;
    settings.beginGroup(QLatin1String(SettingsGroup));
    MaskEditorSettings result;
    result.excludePixels = settings.value(QStringLiteral("exclude_pixels"), true).toBool();
    result.brushRadius = settings.value(QStringLiteral("brush_radius"), 28).toInt();
    result.colorTolerance = settings.value(QStringLiteral("color_tolerance"), 24).toInt();
    result.scissorsSearchRadius = settings.value(QStringLiteral("scissors_search_radius"), 12).toInt();
    result.overlayOpacity = settings.value(QStringLiteral("overlay_opacity"), 38).toInt();
    settings.endGroup();
    return result;
}

void saveMaskEditorSettings(const MaskEditorSettings& settingsValue)
{
    QSettings settings;
    settings.beginGroup(QLatin1String(SettingsGroup));
    settings.setValue(QStringLiteral("exclude_pixels"), settingsValue.excludePixels);
    settings.setValue(QStringLiteral("brush_radius"), settingsValue.brushRadius);
    settings.setValue(QStringLiteral("color_tolerance"), settingsValue.colorTolerance);
    settings.setValue(QStringLiteral("scissors_search_radius"), settingsValue.scissorsSearchRadius);
    settings.setValue(QStringLiteral("overlay_opacity"), settingsValue.overlayOpacity);
    settings.endGroup();
}

MaskEditorSettingsDialog::MaskEditorSettingsDialog(QWidget* parent) : QDialog(parent)
{
    setWindowTitle(tr("蒙版设置"));
    setModal(true);
    setMinimumWidth(390);

    const MaskEditorSettings current = loadMaskEditorSettings();
    auto* layout = new QVBoxLayout(this);
    auto* description = new QLabel(tr("蒙版中的红色区域会从特征提取、连接点筛选和重建中排除。\n"
                                      "按住 Alt 可临时反转当前的添加/擦除方式。"),
                                   this);
    description->setWordWrap(true);
    layout->addWidget(description);

    auto* form = new QFormLayout;
    _excludePixelsCheck = new QCheckBox(tr("绘制时添加排除区域"), this);
    _excludePixelsCheck->setChecked(current.excludePixels);
    form->addRow(tr("编辑方式"), _excludePixelsCheck);

    _brushRadiusSpin = makeSpinBox(1, 512, tr(" 像素"), this);
    _brushRadiusSpin->setValue(current.brushRadius);
    form->addRow(tr("智能绘画半径"), _brushRadiusSpin);

    _colorToleranceSpin = makeSpinBox(0, 255, QString(), this);
    _colorToleranceSpin->setValue(current.colorTolerance);
    _colorToleranceSpin->setToolTip(tr("数值越大，魔棒和智能绘画会接受越宽的颜色范围。"));
    form->addRow(tr("颜色容差"), _colorToleranceSpin);

    _scissorsRadiusSpin = makeSpinBox(1, 48, tr(" 像素"), this);
    _scissorsRadiusSpin->setValue(current.scissorsSearchRadius);
    form->addRow(tr("智能剪刀吸附范围"), _scissorsRadiusSpin);

    _overlayOpacitySpin = makeSpinBox(5, 90, QStringLiteral(" %"), this);
    _overlayOpacitySpin->setValue(current.overlayOpacity);
    form->addRow(tr("蒙版叠加透明度"), _overlayOpacitySpin);
    layout->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layout->addWidget(buttons);
}

MaskEditorSettings MaskEditorSettingsDialog::settings() const
{
    MaskEditorSettings result;
    result.excludePixels = _excludePixelsCheck->isChecked();
    result.brushRadius = _brushRadiusSpin->value();
    result.colorTolerance = _colorToleranceSpin->value();
    result.scissorsSearchRadius = _scissorsRadiusSpin->value();
    result.overlayOpacity = _overlayOpacitySpin->value();
    return result;
}
