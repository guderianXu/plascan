#include "image/GenerateMaskDialog.h"

#include "model/Sam21ModelCatalog.h"
#include "model/TorchScriptModelResolver.h"
#include "model/U2NetModelCatalog.h"

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
#include <QSignalBlocker>
#include <QSpinBox>
#include <QVBoxLayout>

namespace
{

constexpr int kSam21StatusLabelRole = Qt::UserRole + 1;
constexpr int kSam21StatusDetailRole = Qt::UserRole + 2;
constexpr int kSam21InstalledRole = Qt::UserRole + 3;

} // namespace

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
    _methodCombo->addItem(tr("AI: U2Net ONNX"), QStringLiteral("u2net"));
    _methodCombo->addItem(tr("AI: SAM2.1 TorchScript"), QStringLiteral("sam21"));

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

    _sam21VariantCombo = new QComboBox(this);
    _sam21VariantCombo->setObjectName(QStringLiteral("sam21VariantCombo"));

    _sam21DeviceCombo = new QComboBox(this);
    _sam21DeviceCombo->setObjectName(QStringLiteral("sam21DeviceCombo"));
    _sam21DeviceCombo->addItem(tr("CUDA"), QStringLiteral("cuda"));
    _sam21DeviceCombo->addItem(tr("CPU"), QStringLiteral("cpu"));

    _sam21AllowFallbackCheck = new QCheckBox(tr("CUDA 不可用时回退 CPU"), this);
    _sam21AllowFallbackCheck->setObjectName(QStringLiteral("sam21AllowFallbackCheck"));
    _sam21AllowFallbackCheck->setChecked(true);

    _sam21CudaDeviceSpin = new QSpinBox(this);
    _sam21CudaDeviceSpin->setObjectName(QStringLiteral("sam21CudaDeviceSpin"));
    _sam21CudaDeviceSpin->setRange(0, 16);
    _sam21CudaDeviceSpin->setValue(0);

    _sam21PromptModeCombo = new QComboBox(this);
    _sam21PromptModeCombo->setObjectName(QStringLiteral("sam21PromptModeCombo"));
    _sam21PromptModeCombo->addItem(tr("整张照片 box prompt"), QStringLiteral("full_image_box"));

    _sam21InputSizeSpin = new QSpinBox(this);
    _sam21InputSizeSpin->setObjectName(QStringLiteral("sam21InputSizeSpin"));
    _sam21InputSizeSpin->setRange(256, 2048);
    _sam21InputSizeSpin->setSingleStep(64);
    _sam21InputSizeSpin->setValue(1024);

    _sam21MaskThresholdSpin = new QDoubleSpinBox(this);
    _sam21MaskThresholdSpin->setObjectName(QStringLiteral("sam21MaskThresholdSpin"));
    _sam21MaskThresholdSpin->setRange(-10.0, 10.0);
    _sam21MaskThresholdSpin->setDecimals(3);
    _sam21MaskThresholdSpin->setSingleStep(0.1);
    _sam21MaskThresholdSpin->setValue(0.0);

    _u2netDeviceCombo = new QComboBox(this);
    _u2netDeviceCombo->setObjectName(QStringLiteral("u2netDeviceCombo"));
    _u2netDeviceCombo->addItem(tr("CUDA"), QStringLiteral("cuda"));
    _u2netDeviceCombo->addItem(tr("CPU"), QStringLiteral("cpu"));

    _u2netAllowFallbackCheck = new QCheckBox(tr("CUDA 不可用时回退 CPU"), this);
    _u2netAllowFallbackCheck->setObjectName(QStringLiteral("u2netAllowFallbackCheck"));
    _u2netAllowFallbackCheck->setChecked(true);

    _u2netModelStatusLabel = new QLabel(this);
    _u2netModelStatusLabel->setObjectName(QStringLiteral("u2netModelStatusLabel"));
    _u2netModelStatusLabel->setWordWrap(true);

    _u2netInputSizeSpin = new QSpinBox(this);
    _u2netInputSizeSpin->setObjectName(QStringLiteral("u2netInputSizeSpin"));
    _u2netInputSizeSpin->setRange(128, 1024);
    _u2netInputSizeSpin->setSingleStep(32);
    _u2netInputSizeSpin->setValue(320);

    _u2netMaskThresholdSpin = new QDoubleSpinBox(this);
    _u2netMaskThresholdSpin->setObjectName(QStringLiteral("u2netMaskThresholdSpin"));
    _u2netMaskThresholdSpin->setRange(0.01, 0.99);
    _u2netMaskThresholdSpin->setDecimals(3);
    _u2netMaskThresholdSpin->setSingleStep(0.05);
    _u2netMaskThresholdSpin->setValue(0.5);

    _sam21ModelStatusLabel = new QLabel(this);
    _sam21ModelStatusLabel->setObjectName(QStringLiteral("sam21ModelStatusLabel"));
    _sam21ModelStatusLabel->setWordWrap(true);

    _sam21InstallButton = new QPushButton(tr("安装模型..."), this);
    _sam21InstallButton->setObjectName(QStringLiteral("sam21InstallButton"));

    auto *sam21StatusLayout = new QHBoxLayout;
    sam21StatusLayout->setContentsMargins(0, 0, 0, 0);
    sam21StatusLayout->addWidget(_sam21ModelStatusLabel, 1);
    sam21StatusLayout->addWidget(_sam21InstallButton);
    auto *sam21StatusWidget = new QWidget(this);
    sam21StatusWidget->setLayout(sam21StatusLayout);

    _thresholdParameterPanel = new QWidget(this);
    _thresholdParameterPanel->setObjectName(QStringLiteral("thresholdParameterPanel"));
    auto *thresholdLayout = new QFormLayout(_thresholdParameterPanel);
    thresholdLayout->setContentsMargins(0, 0, 0, 0);
    thresholdLayout->addRow(QString(), _autoThresholdCheck);
    thresholdLayout->addRow(tr("阈值:"), _thresholdSpin);
    thresholdLayout->addRow(tr("边界平滑半径:"), _morphologyRadiusSpin);
    thresholdLayout->addRow(tr("最小区域面积:"), _minComponentAreaSpin);

    _sam21ParameterPanel = new QWidget(this);
    _sam21ParameterPanel->setObjectName(QStringLiteral("sam21ParameterPanel"));
    auto *sam21Layout = new QFormLayout(_sam21ParameterPanel);
    sam21Layout->setContentsMargins(0, 0, 0, 0);
    sam21Layout->addRow(tr("SAM2.1 模型:"), _sam21VariantCombo);
    sam21Layout->addRow(tr("SAM2.1 状态:"), sam21StatusWidget);
    sam21Layout->addRow(tr("SAM2.1 设备:"), _sam21DeviceCombo);
    sam21Layout->addRow(QString(), _sam21AllowFallbackCheck);
    sam21Layout->addRow(tr("CUDA 设备:"), _sam21CudaDeviceSpin);
    sam21Layout->addRow(tr("SAM2.1 提示:"), _sam21PromptModeCombo);
    sam21Layout->addRow(tr("SAM2.1 输入尺寸:"), _sam21InputSizeSpin);
    sam21Layout->addRow(tr("SAM2.1 mask 阈值:"), _sam21MaskThresholdSpin);

    _u2netParameterPanel = new QWidget(this);
    _u2netParameterPanel->setObjectName(QStringLiteral("u2netParameterPanel"));
    auto *u2netLayout = new QFormLayout(_u2netParameterPanel);
    u2netLayout->setContentsMargins(0, 0, 0, 0);
    u2netLayout->addRow(tr("U2Net 状态:"), _u2netModelStatusLabel);
    u2netLayout->addRow(tr("U2Net 设备:"), _u2netDeviceCombo);
    u2netLayout->addRow(QString(), _u2netAllowFallbackCheck);
    u2netLayout->addRow(tr("U2Net 输入尺寸:"), _u2netInputSizeSpin);
    u2netLayout->addRow(tr("U2Net 前景阈值:"), _u2netMaskThresholdSpin);

    auto *paramLayout = new QFormLayout;
    paramLayout->addRow(tr("方法:"), _methodCombo);
    paramLayout->addRow(tr("操作:"), _operationCombo);
    paramLayout->addRow(_thresholdParameterPanel);
    paramLayout->addRow(_sam21ParameterPanel);
    paramLayout->addRow(_u2netParameterPanel);

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
    connect(_methodCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &GenerateMaskDialog::updateMethodState);
    connect(_sam21VariantCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &GenerateMaskDialog::updateSam21StatusText);
    connect(_sam21DeviceCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &GenerateMaskDialog::updateMethodState);
    connect(_u2netDeviceCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &GenerateMaskDialog::updateMethodState);
    connect(_sam21InstallButton, &QPushButton::clicked,
            this,
            [this]()
            {
                if (!_sam21VariantCombo)
                {
                    return;
                }
                QString token = _sam21VariantCombo->currentData().toString();
                if (token.trimmed().isEmpty())
                {
                    token = QStringLiteral("tiny");
                }
                emit sam21InstallRequested(token);
            });

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(paramGroup);
    mainLayout->addWidget(scopeGroup);
    mainLayout->addWidget(buttons);

    refreshSam21ModelStatus();
    updateU2NetStatusText();
    updateMethodState();
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
    settings.insert(QStringLiteral("sam21_variant"), _sam21VariantCombo->currentData().toString());
    settings.insert(QStringLiteral("sam21_device"), _sam21DeviceCombo->currentData().toString());
    settings.insert(QStringLiteral("sam21_allow_fallback"), _sam21AllowFallbackCheck->isChecked());
    settings.insert(QStringLiteral("sam21_cuda_device"), _sam21CudaDeviceSpin->value());
    settings.insert(QStringLiteral("sam21_prompt_mode"), _sam21PromptModeCombo->currentData().toString());
    settings.insert(QStringLiteral("sam21_input_size"), _sam21InputSizeSpin->value());
    settings.insert(QStringLiteral("sam21_mask_threshold"), _sam21MaskThresholdSpin->value());
    settings.insert(QStringLiteral("u2net_device"), _u2netDeviceCombo->currentData().toString());
    settings.insert(QStringLiteral("u2net_allow_fallback"), _u2netAllowFallbackCheck->isChecked());
    settings.insert(QStringLiteral("u2net_input_size"), _u2netInputSizeSpin->value());
    settings.insert(QStringLiteral("u2net_mask_threshold"), _u2netMaskThresholdSpin->value());

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
        const bool isSam21 = _methodCombo
            && _methodCombo->currentData().toString() == QLatin1String("sam21");
        const bool isU2Net = _methodCombo
            && _methodCombo->currentData().toString() == QLatin1String("u2net");
        _thresholdSpin->setEnabled(!isSam21 && !isU2Net && !_autoThresholdCheck->isChecked());
    }
}

void GenerateMaskDialog::updateMethodState()
{
    const bool isSam21 = _methodCombo
        && _methodCombo->currentData().toString() == QLatin1String("sam21");
    const bool isU2Net = _methodCombo
        && _methodCombo->currentData().toString() == QLatin1String("u2net");
    const bool isAiMethod = isSam21 || isU2Net;

    if (_thresholdParameterPanel)
    {
        _thresholdParameterPanel->setVisible(!isAiMethod);
    }
    if (_sam21ParameterPanel)
    {
        _sam21ParameterPanel->setVisible(isSam21);
    }
    if (_u2netParameterPanel)
    {
        _u2netParameterPanel->setVisible(isU2Net);
    }

    if (_autoThresholdCheck)
    {
        _autoThresholdCheck->setEnabled(!isAiMethod);
    }
    if (_morphologyRadiusSpin)
    {
        _morphologyRadiusSpin->setEnabled(!isAiMethod);
    }
    if (_minComponentAreaSpin)
    {
        _minComponentAreaSpin->setEnabled(!isAiMethod);
    }

    if (_sam21VariantCombo)
    {
        _sam21VariantCombo->setEnabled(isSam21);
    }
    if (_sam21DeviceCombo)
    {
        _sam21DeviceCombo->setEnabled(isSam21);
    }
    if (_sam21AllowFallbackCheck)
    {
        _sam21AllowFallbackCheck->setEnabled(isSam21);
    }
    if (_sam21CudaDeviceSpin)
    {
        const bool cudaSelected = _sam21DeviceCombo
            && _sam21DeviceCombo->currentData().toString() == QLatin1String("cuda");
        _sam21CudaDeviceSpin->setEnabled(isSam21 && cudaSelected);
    }
    if (_sam21PromptModeCombo)
    {
        _sam21PromptModeCombo->setEnabled(isSam21);
    }
    if (_sam21InputSizeSpin)
    {
        _sam21InputSizeSpin->setEnabled(isSam21);
    }
    if (_sam21MaskThresholdSpin)
    {
        _sam21MaskThresholdSpin->setEnabled(isSam21);
    }
    if (_u2netModelStatusLabel)
    {
        _u2netModelStatusLabel->setEnabled(isU2Net);
    }
    if (_u2netDeviceCombo)
    {
        _u2netDeviceCombo->setEnabled(isU2Net);
    }
    if (_u2netAllowFallbackCheck)
    {
        _u2netAllowFallbackCheck->setEnabled(isU2Net);
    }
    if (_u2netInputSizeSpin)
    {
        _u2netInputSizeSpin->setEnabled(isU2Net);
    }
    if (_u2netMaskThresholdSpin)
    {
        _u2netMaskThresholdSpin->setEnabled(isU2Net);
    }

    updateSam21StatusText();
    updateU2NetStatusText();
    updateThresholdState();
}

void GenerateMaskDialog::refreshSam21ModelStatus()
{
    if (!_sam21VariantCombo)
    {
        return;
    }

    QString currentToken = _sam21VariantCombo->currentData().toString();
    if (currentToken.trimmed().isEmpty())
    {
        currentToken = QStringLiteral("tiny");
    }
    const QSignalBlocker block(_sam21VariantCombo);
    _sam21VariantCombo->clear();

    xjw::common::model::TorchScriptModelResolver resolver;
    for (const auto &spec : xjw::common::model::sam21ModelSpecs())
    {
        const auto status = xjw::common::model::sam21ModelStatus(resolver, spec.token);
        const QString itemText = QStringLiteral("%1 (%2)").arg(status.spec.displayName, status.label);
        _sam21VariantCombo->addItem(itemText, status.spec.token);
        const int index = _sam21VariantCombo->count() - 1;
        _sam21VariantCombo->setItemData(index, status.label, kSam21StatusLabelRole);
        _sam21VariantCombo->setItemData(index, status.detail, kSam21StatusDetailRole);
        _sam21VariantCombo->setItemData(index, status.isFullyInstalled, kSam21InstalledRole);
        _sam21VariantCombo->setItemData(index, status.detail, Qt::ToolTipRole);
    }

    const int restoreIndex = _sam21VariantCombo->findData(currentToken);
    if (restoreIndex >= 0)
    {
        _sam21VariantCombo->setCurrentIndex(restoreIndex);
    }
    else if (_sam21VariantCombo->count() > 0)
    {
        _sam21VariantCombo->setCurrentIndex(0);
    }

    updateSam21StatusText();
}

void GenerateMaskDialog::updateSam21StatusText()
{
    const bool isSam21 = _methodCombo
        && _methodCombo->currentData().toString() == QLatin1String("sam21");

    const int index = _sam21VariantCombo ? _sam21VariantCombo->currentIndex() : -1;
    QString label = index >= 0
        ? _sam21VariantCombo->itemData(index, kSam21StatusLabelRole).toString()
        : QString();
    if (label.trimmed().isEmpty())
    {
        label = QStringLiteral("未安装");
    }
    const QString detail = index >= 0
        ? _sam21VariantCombo->itemData(index, kSam21StatusDetailRole).toString()
        : QString();
    const bool installed = index >= 0
        && _sam21VariantCombo->itemData(index, kSam21InstalledRole).toBool();

    if (_sam21ModelStatusLabel)
    {
        _sam21ModelStatusLabel->setEnabled(isSam21);
        _sam21ModelStatusLabel->setText(detail.isEmpty()
            ? tr("状态：%1").arg(label)
            : tr("状态：%1；%2").arg(label, detail));
    }
    if (_sam21InstallButton)
    {
        _sam21InstallButton->setEnabled(isSam21 && !installed);
    }
}

void GenerateMaskDialog::updateU2NetStatusText()
{
    const bool isU2Net = _methodCombo
        && _methodCombo->currentData().toString() == QLatin1String("u2net");
    const xjw::common::model::TorchScriptModelResolver resolver;
    const auto status = xjw::common::model::u2netModelStatus(resolver);

    if (_u2netModelStatusLabel)
    {
        _u2netModelStatusLabel->setEnabled(isU2Net);
        _u2netModelStatusLabel->setText(
            tr("状态：%1；%2").arg(status.label, status.detail));
    }
}
