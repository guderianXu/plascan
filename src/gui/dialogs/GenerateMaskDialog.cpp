#include "GenerateMaskDialog.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QVBoxLayout>

GenerateMaskDialog::GenerateMaskDialog(const QStringList &selectedImages,
                                       const QString &currentImage,
                                       QWidget *parent)
    : QDialog(parent)
    , _selectedImages(selectedImages)
    , _currentImage(currentImage)
{
    setWindowTitle(tr("生成蒙版"));
    setModal(true);

    _methodCombo = new QComboBox(this);
    _methodCombo->setObjectName(QStringLiteral("methodCombo"));
    _methodCombo->addItem(tr("黑色背景 / 无效区"), QStringLiteral("black_background"));
    _methodCombo->addItem(tr("亮度阈值"), QStringLiteral("threshold"));

    _operationCombo = new QComboBox(this);
    _operationCombo->setObjectName(QStringLiteral("operationCombo"));
    _operationCombo->addItem(tr("替换"), QStringLiteral("replace"));
    _operationCombo->addItem(tr("结合"), QStringLiteral("union"));
    _operationCombo->addItem(tr("插值"), QStringLiteral("intersection"));
    _operationCombo->addItem(tr("差异"), QStringLiteral("difference"));

    _autoThresholdCheck = new QCheckBox(tr("自动阈值"), this);
    _autoThresholdCheck->setObjectName(QStringLiteral("autoThresholdCheck"));
    _autoThresholdCheck->setChecked(true);

    _thresholdSpin = new QDoubleSpinBox(this);
    _thresholdSpin->setObjectName(QStringLiteral("thresholdSpin"));
    _thresholdSpin->setRange(0.0, 255.0);
    _thresholdSpin->setDecimals(2);
    _thresholdSpin->setSingleStep(1.0);
    _thresholdSpin->setValue(3.0);

    _morphologyRadiusSpin = new QSpinBox(this);
    _morphologyRadiusSpin->setObjectName(QStringLiteral("morphologyRadiusSpin"));
    _morphologyRadiusSpin->setRange(0, 32);
    _morphologyRadiusSpin->setValue(2);

    _minComponentAreaSpin = new QSpinBox(this);
    _minComponentAreaSpin->setObjectName(QStringLiteral("minComponentAreaSpin"));
    _minComponentAreaSpin->setRange(1, 100000000);
    _minComponentAreaSpin->setValue(64);

    auto *paramLayout = new QFormLayout;
    paramLayout->addRow(tr("方法:"), _methodCombo);
    paramLayout->addRow(tr("操作:"), _operationCombo);
    paramLayout->addRow(QString(), _autoThresholdCheck);
    paramLayout->addRow(tr("阈值:"), _thresholdSpin);
    paramLayout->addRow(tr("边界平滑半径:"), _morphologyRadiusSpin);
    paramLayout->addRow(tr("最小区域面积:"), _minComponentAreaSpin);

    auto *paramGroup = new QGroupBox(tr("参数"), this);
    paramGroup->setLayout(paramLayout);

    _scopeGroup = new QButtonGroup(this);
    _scopeGroup->setObjectName(QStringLiteral("scopeGroup"));

    auto *allImagesRadio = new QRadioButton(tr("所有图像"), this);
    auto *selectedImagesRadio = new QRadioButton(tr("选定图像"), this);
    auto *currentImageRadio = new QRadioButton(tr("当前照片"), this);
    _scopeGroup->addButton(allImagesRadio, 0);
    _scopeGroup->addButton(selectedImagesRadio, 1);
    _scopeGroup->addButton(currentImageRadio, 2);

    selectedImagesRadio->setChecked(true);
    currentImageRadio->setEnabled(!_currentImage.trimmed().isEmpty());

    _selectionLabel = new QLabel(
        tr("选定图像: %1").arg(_selectedImages.size()), this);
    _selectionLabel->setObjectName(QStringLiteral("selectionLabel"));

    auto *scopeLayout = new QVBoxLayout;
    auto *scopeRowLayout = new QHBoxLayout;
    scopeRowLayout->addWidget(allImagesRadio);
    scopeRowLayout->addWidget(selectedImagesRadio);
    scopeRowLayout->addWidget(currentImageRadio);
    scopeRowLayout->addStretch();
    scopeLayout->addLayout(scopeRowLayout);
    scopeLayout->addWidget(_selectionLabel);

    auto *scopeGroup = new QGroupBox(tr("应用于"), this);
    scopeGroup->setLayout(scopeLayout);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)->setText(tr("OK"));
    buttons->button(QDialogButtonBox::Cancel)->setText(tr("Cancel"));
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(_autoThresholdCheck, &QCheckBox::toggled, this, &GenerateMaskDialog::updateThresholdState);

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(paramGroup);
    mainLayout->addWidget(scopeGroup);
    mainLayout->addWidget(buttons);

    updateThresholdState();
}

QJsonObject GenerateMaskDialog::collectSettings() const
{
    QJsonObject settings;
    settings.insert(QStringLiteral("method"), _methodCombo->currentData().toString());
    settings.insert(QStringLiteral("operation"), _operationCombo->currentData().toString());
    settings.insert(QStringLiteral("auto_threshold"), _autoThresholdCheck->isChecked());
    settings.insert(QStringLiteral("threshold"), _thresholdSpin->value());
    settings.insert(QStringLiteral("morphology_radius"), _morphologyRadiusSpin->value());
    settings.insert(QStringLiteral("min_component_area"), _minComponentAreaSpin->value());

    const int checkedScope = _scopeGroup->checkedId();
    QString scope = QStringLiteral("selected_images");
    if (checkedScope == 0)
    {
        scope = QStringLiteral("all_images");
    }
    else if (checkedScope == 2)
    {
        scope = QStringLiteral("current_image");
    }
    settings.insert(QStringLiteral("scope"), scope);
    settings.insert(QStringLiteral("current_image"), _currentImage);

    QJsonArray selectedImages;
    for (const QString &path : _selectedImages)
    {
        selectedImages.append(path);
    }
    settings.insert(QStringLiteral("selected_images"), selectedImages);
    return settings;
}

void GenerateMaskDialog::updateThresholdState()
{
    if (_thresholdSpin)
    {
        _thresholdSpin->setEnabled(!_autoThresholdCheck->isChecked());
    }
}
