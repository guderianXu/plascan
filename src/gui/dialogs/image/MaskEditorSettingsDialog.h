#pragma once

#include <QDialog>

class QCheckBox;
class QSpinBox;

struct MaskEditorSettings
{
    bool excludePixels = true;
    int brushRadius = 28;
    int colorTolerance = 24;
    int scissorsSearchRadius = 12;
    int overlayOpacity = 38;
};

MaskEditorSettings loadMaskEditorSettings();
void saveMaskEditorSettings(const MaskEditorSettings& settings);

class MaskEditorSettingsDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit MaskEditorSettingsDialog(QWidget* parent = nullptr);

    MaskEditorSettings settings() const;

private:
    QCheckBox* _excludePixelsCheck{};
    QSpinBox* _brushRadiusSpin{};
    QSpinBox* _colorToleranceSpin{};
    QSpinBox* _scissorsRadiusSpin{};
    QSpinBox* _overlayOpacitySpin{};
};
