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

private:
    QStringList _selectedImages;
    QString _currentImage;
    QComboBox *_methodCombo{};
    QComboBox *_operationCombo{};
    QButtonGroup *_scopeGroup{};
    QCheckBox *_autoThresholdCheck{};
    QDoubleSpinBox *_thresholdSpin{};
    QSpinBox *_morphologyRadiusSpin{};
    QSpinBox *_minComponentAreaSpin{};
    QLabel *_selectionLabel{};
};
