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
class QPushButton;
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

signals:
    void sam21InstallRequested(const QString &variantToken);

public slots:
    void refreshSam21ModelStatus();

private slots:
    void updateThresholdState();
    void updateMethodState();
    void updateSam21StatusText();
    void updateU2NetStatusText();

private:
    QStringList _selectedImages;
    QString _currentImage;
    QComboBox *_methodCombo{};
    QComboBox *_operationCombo{};
    QButtonGroup *_scopeGroup{};
    QWidget *_thresholdParameterPanel{};
    QWidget *_sam21ParameterPanel{};
    QWidget *_u2netParameterPanel{};
    QCheckBox *_autoThresholdCheck{};
    QDoubleSpinBox *_thresholdSpin{};
    QSpinBox *_morphologyRadiusSpin{};
    QSpinBox *_minComponentAreaSpin{};
    QComboBox *_sam21VariantCombo{};
    QComboBox *_sam21DeviceCombo{};
    QComboBox *_sam21PromptModeCombo{};
    QCheckBox *_sam21AllowFallbackCheck{};
    QPushButton *_sam21InstallButton{};
    QLabel *_sam21ModelStatusLabel{};
    QSpinBox *_sam21CudaDeviceSpin{};
    QSpinBox *_sam21InputSizeSpin{};
    QDoubleSpinBox *_sam21MaskThresholdSpin{};
    QComboBox *_u2netDeviceCombo{};
    QCheckBox *_u2netAllowFallbackCheck{};
    QLabel *_u2netModelStatusLabel{};
    QSpinBox *_u2netInputSizeSpin{};
    QDoubleSpinBox *_u2netMaskThresholdSpin{};
    QLabel *_selectionLabel{};
};
