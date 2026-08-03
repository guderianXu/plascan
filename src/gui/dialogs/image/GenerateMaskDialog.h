#pragma once

#include "model/ModelFileResolver.h"

#include <QDialog>
#include <QJsonObject>
#include <QString>
#include <QStringList>

class QButtonGroup;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QDialogButtonBox;
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
                                QWidget *parent = nullptr,
                                xjw::common::model::ModelFileSearchOptions modelSearchOptions = {});

    QJsonObject collectSettings() const;

private slots:
    void updateThresholdState();
    void updateMethodState();
    void updateU2NetStatusText();
    void downloadU2NetModel();

private:
    QStringList _selectedImages;
    QString _currentImage;
    xjw::common::model::ModelFileSearchOptions _modelSearchOptions;
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
    QPushButton *_u2netDownloadButton{};
    QSpinBox *_u2netInputSizeSpin{};
    QDoubleSpinBox *_u2netMaskThresholdSpin{};
    QLabel *_selectionLabel{};
    QDialogButtonBox *_buttons{};
};
