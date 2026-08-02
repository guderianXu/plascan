#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QString>
#include <QStringList>

class QButtonGroup;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QSpinBox;
class QWidget;

class GenerateMaskDialog : public QDialog
{
    Q_OBJECT
public:
    explicit GenerateMaskDialog(const QStringList &selectedImages = {},
                                const QString &currentImage = QString(),
                                QWidget *parent = nullptr);

    QJsonObject collectSettings() const;

private slots:
    void updateThresholdState();
    void updateMethodState();
    void updateU2NetStatusText();

private:
    QStringList _selectedImages;
    QString _currentImage;
    QComboBox *_methodCombo{};
    QComboBox *_operationCombo{};
    QButtonGroup *_scopeGroup{};
    QWidget *_thresholdParameterPanel{};
    QWidget *_u2netParameterPanel{};
    QCheckBox *_autoThresholdCheck{};
    QDoubleSpinBox *_thresholdSpin{};
    QSpinBox *_morphologyRadiusSpin{};
    QSpinBox *_minComponentAreaSpin{};
    QComboBox *_u2netDeviceCombo{};
    QCheckBox *_u2netAllowFallbackCheck{};
    QLabel *_u2netModelStatusLabel{};
    QSpinBox *_u2netInputSizeSpin{};
    QDoubleSpinBox *_u2netMaskThresholdSpin{};
    QLabel *_selectionLabel{};
};
