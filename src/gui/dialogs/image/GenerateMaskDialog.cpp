#include "image/GenerateMaskDialog.h"

#include "application/ModelPackageDownloadDialog.h"
#include "birefnet/BiRefNetMaskGenerator.h"
#include "model/BiRefNetModelCatalog.h"
#include "model/ModelAssetCatalog.h"
#include "model/ModelFileResolver.h"
#include "model/U2NetModelCatalog.h"
#include "u2net/U2NetMaskGenerator.h"

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

#include <utility>

GenerateMaskDialog::GenerateMaskDialog(const QStringList &selectedImages,
                                       const QString &currentImage,
                                       QWidget *parent,
                                       xjw::common::model::ModelFileSearchOptions modelSearchOptions)
    : QDialog(parent)
    , _selectedImages(selectedImages)
    , _currentImage(currentImage)
    , _modelSearchOptions(std::move(modelSearchOptions))
{
    setWindowTitle(tr("生成蒙版"));
    setModal(true);

    _methodCombo = new QComboBox(this);
    _methodCombo->setObjectName(QStringLiteral("methodCombo"));
    _methodCombo->addItem(tr("黑色背景 / 无效区"), QStringLiteral("black_background"));
    _methodCombo->addItem(tr("亮度阈值"), QStringLiteral("threshold"));
    _methodCombo->addItem(tr("AI: U2Net ONNX"), QStringLiteral("u2net"));
    _methodCombo->addItem(tr("AI: BiRefNet Dynamic（推荐）"), QStringLiteral("birefnet_dynamic"));

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

    _u2netDeviceCombo = new QComboBox(this);
    _u2netDeviceCombo->setObjectName(QStringLiteral("u2netDeviceCombo"));
    _u2netDeviceCombo->addItem(tr("自动（推荐）"), QStringLiteral("auto"));
    _u2netDeviceCombo->addItem(tr("TensorRT GPU"), QStringLiteral("tensorrt"));
    _u2netDeviceCombo->addItem(tr("OpenCV CPU"), QStringLiteral("cpu"));

    _u2netAllowFallbackCheck = new QCheckBox(tr("TensorRT 失败时回退 OpenCV CPU"), this);
    _u2netAllowFallbackCheck->setObjectName(QStringLiteral("u2netAllowFallbackCheck"));
    _u2netAllowFallbackCheck->setChecked(false);

    _u2netModelStatusLabel = new QLabel(this);
    _u2netModelStatusLabel->setObjectName(QStringLiteral("u2netModelStatusLabel"));
    _u2netModelStatusLabel->setWordWrap(true);
    _u2netModelStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);

    _u2netDownloadButton = new QPushButton(tr("下载 U2Net 模型（167.8 MiB）"), this);
    _u2netDownloadButton->setObjectName(QStringLiteral("u2netDownloadButton"));

    _u2netInputSizeSpin = new QSpinBox(this);
    _u2netInputSizeSpin->setObjectName(QStringLiteral("u2netInputSizeSpin"));
    _u2netInputSizeSpin->setRange(xjw::mask::kU2NetModelInputSize,
                                  xjw::mask::kU2NetModelInputSize);
    _u2netInputSizeSpin->setValue(xjw::mask::kU2NetModelInputSize);
    _u2netInputSizeSpin->setToolTip(tr("随附 U2Net ONNX 的固定输入尺寸"));

    _u2netMaskThresholdSpin = new QDoubleSpinBox(this);
    _u2netMaskThresholdSpin->setObjectName(QStringLiteral("u2netMaskThresholdSpin"));
    _u2netMaskThresholdSpin->setRange(0.01, 0.99);
    _u2netMaskThresholdSpin->setDecimals(3);
    _u2netMaskThresholdSpin->setSingleStep(0.05);
    _u2netMaskThresholdSpin->setValue(0.5);

    _thresholdParameterPanel = new QWidget(this);
    _thresholdParameterPanel->setObjectName(QStringLiteral("thresholdParameterPanel"));
    auto *threshold_layout = new QFormLayout(_thresholdParameterPanel);
    threshold_layout->setContentsMargins(0, 0, 0, 0);
    threshold_layout->addRow(QString(), _autoThresholdCheck);
    threshold_layout->addRow(tr("阈值:"), _thresholdSpin);
    threshold_layout->addRow(tr("边界平滑半径:"), _morphologyRadiusSpin);
    threshold_layout->addRow(tr("最小区域面积:"), _minComponentAreaSpin);

    _u2netParameterPanel = new QWidget(this);
    _u2netParameterPanel->setObjectName(QStringLiteral("u2netParameterPanel"));
    auto *u2net_layout = new QFormLayout(_u2netParameterPanel);
    u2net_layout->setContentsMargins(0, 0, 0, 0);
    u2net_layout->addRow(tr("模型状态:"), _u2netModelStatusLabel);
    u2net_layout->addRow(QString(), _u2netDownloadButton);
    u2net_layout->addRow(tr("推理设备:"), _u2netDeviceCombo);
    u2net_layout->addRow(QString(), _u2netAllowFallbackCheck);
    u2net_layout->addRow(tr("模型输入尺寸:"), _u2netInputSizeSpin);
    u2net_layout->addRow(tr("前景阈值:"), _u2netMaskThresholdSpin);

    auto *parameter_layout = new QFormLayout;
    parameter_layout->addRow(tr("方法:"), _methodCombo);
    parameter_layout->addRow(tr("操作:"), _operationCombo);
    parameter_layout->addRow(_thresholdParameterPanel);
    parameter_layout->addRow(_u2netParameterPanel);
    auto *parameter_group = new QGroupBox(tr("参数"), this);
    parameter_group->setLayout(parameter_layout);

    _scopeGroup = new QButtonGroup(this);
    _scopeGroup->setObjectName(QStringLiteral("scopeGroup"));
    auto *all_images_radio = new QRadioButton(tr("所有图像"), this);
    auto *selected_images_radio = new QRadioButton(tr("选定图像"), this);
    auto *current_image_radio = new QRadioButton(tr("当前照片"), this);
    _scopeGroup->addButton(all_images_radio, 0);
    _scopeGroup->addButton(selected_images_radio, 1);
    _scopeGroup->addButton(current_image_radio, 2);
    selected_images_radio->setChecked(true);
    current_image_radio->setEnabled(!_currentImage.trimmed().isEmpty());

    _selectionLabel = new QLabel(tr("选定图像: %1").arg(_selectedImages.size()), this);
    _selectionLabel->setObjectName(QStringLiteral("selectionLabel"));
    auto *scope_row = new QHBoxLayout;
    scope_row->addWidget(all_images_radio);
    scope_row->addWidget(selected_images_radio);
    scope_row->addWidget(current_image_radio);
    scope_row->addStretch();
    auto *scope_layout = new QVBoxLayout;
    scope_layout->addLayout(scope_row);
    scope_layout->addWidget(_selectionLabel);
    auto *scope_group = new QGroupBox(tr("应用于"), this);
    scope_group->setLayout(scope_layout);

    _buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    _buttons->setObjectName(QStringLiteral("maskDialogButtons"));
    _buttons->button(QDialogButtonBox::Ok)->setText(tr("OK"));
    _buttons->button(QDialogButtonBox::Cancel)->setText(tr("Cancel"));
    connect(_buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(_buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(_autoThresholdCheck, &QCheckBox::toggled, this, &GenerateMaskDialog::updateThresholdState);
    connect(_methodCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &GenerateMaskDialog::updateMethodState);
    connect(_u2netDeviceCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &GenerateMaskDialog::updateMethodState);
    connect(_u2netDownloadButton, &QPushButton::clicked,
            this, &GenerateMaskDialog::downloadAiModel);

    auto *main_layout = new QVBoxLayout(this);
    main_layout->addWidget(parameter_group);
    main_layout->addWidget(scope_group);
    main_layout->addWidget(_buttons);

    updateAiModelStatusText();
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
    const QString ai_backend = _u2netDeviceCombo->currentData().toString();
    settings.insert(QStringLiteral("ai_backend"), ai_backend);
    settings.insert(QStringLiteral("ai_allow_fallback"), _u2netAllowFallbackCheck->isChecked());
    settings.insert(QStringLiteral("ai_input_size"), _u2netInputSizeSpin->value());
    settings.insert(QStringLiteral("ai_mask_threshold"), _u2netMaskThresholdSpin->value());
    settings.insert(QStringLiteral("u2net_backend"), ai_backend);
    // Keep the legacy key readable by older project builds. New values deliberately
    // describe the inference backend rather than an OpenCV CUDA device.
    settings.insert(QStringLiteral("u2net_device"), ai_backend);
    settings.insert(QStringLiteral("u2net_allow_fallback"), _u2netAllowFallbackCheck->isChecked());
    settings.insert(QStringLiteral("u2net_input_size"), _u2netInputSizeSpin->value());
    settings.insert(QStringLiteral("u2net_mask_threshold"), _u2netMaskThresholdSpin->value());

    QString scope = QStringLiteral("selected_images");
    if (_scopeGroup->checkedId() == 0)
    {
        scope = QStringLiteral("all_images");
    }
    else if (_scopeGroup->checkedId() == 2)
    {
        scope = QStringLiteral("current_image");
    }
    settings.insert(QStringLiteral("scope"), scope);
    settings.insert(QStringLiteral("current_image"), _currentImage);

    QJsonArray selected_images;
    for (const QString &path : _selectedImages)
    {
        selected_images.append(path);
    }
    settings.insert(QStringLiteral("selected_images"), selected_images);
    return settings;
}

void GenerateMaskDialog::updateThresholdState()
{
    const QString method = _methodCombo ? _methodCombo->currentData().toString() : QString();
    const bool is_ai = method == QLatin1String("u2net") ||
                       method == QLatin1String("birefnet_dynamic");
    _thresholdSpin->setEnabled(!is_ai && !_autoThresholdCheck->isChecked());
}

void GenerateMaskDialog::updateMethodState()
{
    const QString method = _methodCombo ? _methodCombo->currentData().toString() : QString();
    const bool is_u2net = method == QLatin1String("u2net");
    const bool is_birefnet = method == QLatin1String("birefnet_dynamic");
    const bool is_ai = is_u2net || is_birefnet;

    if (is_ai && _configuredAiMethod != method)
    {
        const QSignalBlocker blocker(_u2netDeviceCombo);
        _u2netDeviceCombo->clear();
        if (is_u2net)
        {
            _u2netDeviceCombo->addItem(tr("自动（推荐）"), QStringLiteral("auto"));
            _u2netDeviceCombo->addItem(tr("TensorRT GPU"), QStringLiteral("tensorrt"));
            _u2netDeviceCombo->addItem(tr("OpenCV CPU"), QStringLiteral("cpu"));
            _u2netInputSizeSpin->setRange(xjw::mask::kU2NetModelInputSize,
                                          xjw::mask::kU2NetModelInputSize);
            _u2netInputSizeSpin->setValue(xjw::mask::kU2NetModelInputSize);
            _u2netInputSizeSpin->setToolTip(tr("随附 U2Net ONNX 的固定输入尺寸"));
            _u2netDownloadButton->setText(tr("下载 U2Net 模型（167.8 MiB）"));
        }
        else
        {
            _u2netDeviceCombo->addItem(tr("TensorRT GPU（必需）"), QStringLiteral("tensorrt"));
            _u2netInputSizeSpin->setRange(xjw::mask::kBiRefNetDynamicInputSize,
                                          xjw::mask::kBiRefNetDynamicInputSize);
            _u2netInputSizeSpin->setValue(xjw::mask::kBiRefNetDynamicInputSize);
            _u2netInputSizeSpin->setToolTip(
                tr("BiRefNet Dynamic 权重的部署模型固定使用 1024×1024 输入"));
            _u2netDownloadButton->setText(tr("下载 BiRefNet Dynamic 模型"));
            _u2netAllowFallbackCheck->setChecked(false);
        }
        _configuredAiMethod = method;
    }

    _thresholdParameterPanel->setVisible(!is_ai);
    _u2netParameterPanel->setVisible(is_ai);
    _autoThresholdCheck->setEnabled(!is_ai);
    _morphologyRadiusSpin->setEnabled(!is_ai);
    _minComponentAreaSpin->setEnabled(!is_ai);
    _u2netModelStatusLabel->setEnabled(is_ai);
    _u2netDeviceCombo->setEnabled(is_ai);
    const bool force_tensorrt = _u2netDeviceCombo
        && _u2netDeviceCombo->currentData().toString() == QLatin1String("tensorrt");
    _u2netAllowFallbackCheck->setEnabled(is_u2net && force_tensorrt);
    _u2netAllowFallbackCheck->setVisible(!is_birefnet);
    _u2netInputSizeSpin->setEnabled(false);
    _u2netMaskThresholdSpin->setEnabled(is_ai);
    updateAiModelStatusText();
    updateThresholdState();
}

void GenerateMaskDialog::updateAiModelStatusText()
{
    const QString method = _methodCombo ? _methodCombo->currentData().toString() : QString();
    const bool is_u2net = method == QLatin1String("u2net");
    const bool is_birefnet = method == QLatin1String("birefnet_dynamic");
    if (!is_u2net && !is_birefnet)
    {
        _u2netModelStatusLabel->setEnabled(false);
        _u2netDownloadButton->setVisible(false);
        if (_buttons && _buttons->button(QDialogButtonBox::Ok))
        {
            _buttons->button(QDialogButtonBox::Ok)->setEnabled(true);
        }
        return;
    }

    const xjw::common::model::ModelFileResolver resolver(_modelSearchOptions);
    QString status_label;
    QString status_detail;
    bool is_installed = false;
    QString backend_status;
    QString capability_summary;
    bool backend_available = true;
    if (is_u2net)
    {
        const auto status = xjw::common::model::u2netModelStatus(resolver);
        static const xjw::mask::U2NetInferenceCapabilities capabilities =
            xjw::mask::detectU2NetInferenceCapabilities();
        status_label = status.label;
        status_detail = status.detail;
        is_installed = status.isInstalled;
        capability_summary = QString::fromStdString(capabilities.summary);
        if (capabilities.tensorRtAvailable)
        {
            backend_status = tr("TensorRT %1 可用；GPU：%2；%3")
                .arg(QString::fromStdString(capabilities.tensorRtVersion),
                     QString::fromStdString(capabilities.gpuName),
                     capabilities.supportsFp16 ? tr("优先 FP16") : tr("使用 FP32"));
        }
        else if (capabilities.tensorRtCompiled)
        {
            backend_status = tr("TensorRT 已编译但当前不可用：%1；自动模式将使用 OpenCV CPU")
                .arg(QString::fromStdString(capabilities.errorMessage));
        }
        else
        {
            backend_status = tr("当前构建未启用 TensorRT；自动模式将使用 OpenCV CPU");
        }
    }
    else
    {
        const auto status = xjw::common::model::biRefNetDynamicModelStatus(resolver);
        static const xjw::mask::BiRefNetInferenceCapabilities capabilities =
            xjw::mask::detectBiRefNetInferenceCapabilities();
        status_label = status.label;
        status_detail = status.detail;
        is_installed = status.isInstalled;
        backend_available = capabilities.tensorRtAvailable;
        capability_summary = QString::fromStdString(capabilities.summary);
        if (capabilities.tensorRtAvailable)
        {
            backend_status = tr("TensorRT %1 可用；GPU：%2；%3；不支持 CPU 回退")
                .arg(QString::fromStdString(capabilities.tensorRtVersion),
                     QString::fromStdString(capabilities.gpuName),
                     capabilities.supportsFp16 ? tr("优先 FP16") : tr("使用 FP32"));
        }
        else
        {
            const QString reason = QString::fromStdString(capabilities.errorMessage);
            backend_status = tr("BiRefNet Dynamic 必须使用 TensorRT GPU，当前不可用：%1")
                                 .arg(reason.isEmpty() ? tr("当前构建或设备不支持 TensorRT") : reason);
        }
    }
    _u2netModelStatusLabel->setEnabled(true);
    _u2netModelStatusLabel->setText(
        tr("状态：%1；%2\n计算后端：%3").arg(status_label, status_detail, backend_status));
    _u2netModelStatusLabel->setToolTip(capability_summary);
    _u2netDownloadButton->setVisible(!is_installed);
    _u2netDownloadButton->setEnabled(!is_installed);
    if (_buttons && _buttons->button(QDialogButtonBox::Ok))
    {
        _buttons->button(QDialogButtonBox::Ok)->setEnabled(is_installed && backend_available);
    }
}

void GenerateMaskDialog::downloadAiModel()
{
    const QString method = _methodCombo ? _methodCombo->currentData().toString() : QString();
    const xjw::common::model::ModelFileResolver resolver(_modelSearchOptions);
    const auto package = method == QLatin1String("birefnet_dynamic")
                             ? xjw::common::model::biRefNetDynamicOnnxPackage()
                             : xjw::common::model::u2NetOnnxPackage();
    const QString target_directory =
        xjw::common::model::modelPackageInstallDirectory(package, resolver);

    QString error;
    if (ModelPackageDownloadDialog::downloadPackage(
            package, target_directory, this, &error))
    {
        updateAiModelStatusText();
    }
}
