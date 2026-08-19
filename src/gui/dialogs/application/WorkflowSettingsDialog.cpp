/**
 * @file WorkflowSettingsDialog.cpp
 * @brief 多工作流程项目设置对话框实现。
 */

#include "application/WorkflowSettingsDialog.h"

#include "application/ModelPackageDownloadDialog.h"
#include "shared/WorkflowParameterDialogStyle.h"

#include "ImageMatchingRegistry.h"
#include "GpuDeviceLease.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosRuntime.h"
#include "PatchMatchCUDA.h"
#include "model/ModelAssetCatalog.h"
#include "model/ModelFileResolver.h"

#include <QColor>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPalette>
#include <QPushButton>
#include <QSet>
#include <QSignalBlocker>
#include <QStandardItemModel>
#include <QStackedWidget>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

namespace
{

constexpr auto kAerialWorkflowId = "aerial_triangulation";
constexpr auto kModelGenerationWorkflowId = "generate_model";
constexpr auto kDemWorkflowId = "dem";
constexpr auto kOrthomosaicWorkflowId = "orthomosaic";
constexpr auto kRetiredReconstructionWorkflowId = "reconstruction";
constexpr auto kSiftLightGlueAlgorithmId = "sift_lightglue";
constexpr auto kCudaSiftAlgorithmId = "cuda_sift";
constexpr auto kLoMaRAlgorithmId = "loma_r";
constexpr auto kCudaComputeMode = "cuda";
constexpr auto kOpenClComputeMode = "opencl";
constexpr auto kHybridComputeMode = "hybrid";
constexpr int kWorkflowSettingsVersion = 7;

struct WorkflowEntry
{
    const char *id;
    const char *displayName;
};

constexpr WorkflowEntry kWorkflowEntries[] = {
    {kAerialWorkflowId, "空中三角测量"},
    {kModelGenerationWorkflowId, "生成模型"},
    {kDemWorkflowId, "创建 DEM"},
    {kOrthomosaicWorkflowId, "生成正射影像"}};

QJsonObject defaultAerialSettings()
{
    QJsonObject settings;
    settings[QStringLiteral("algorithm_id")] = QString::fromLatin1(kSiftLightGlueAlgorithmId);
    settings[QStringLiteral("lightglue_tensorrt_engine")] = QString();
    settings[QStringLiteral("loma_r_tensorrt_package")] = QString();
    settings[QStringLiteral("loma_r_keypoint_budget")] = 0;
    return settings;
}

QJsonObject defaultModelGenerationSettings()
{
    QJsonObject settings;
    settings[QStringLiteral("compute_mode")] =
        QString::fromLatin1(kHybridComputeMode);
    return settings;
}

bool isKnownWorkflow(const QString &workflowId)
{
    for (const WorkflowEntry &entry : kWorkflowEntries)
    {
        if (workflowId == QString::fromLatin1(entry.id))
        {
            return true;
        }
    }
    return false;
}

QJsonObject normalizedSettings(const QJsonObject &settings)
{
    QJsonObject normalized = WorkflowSettingsDialog::defaultSettings();
    const QString selected = settings.value(QStringLiteral("selected_workflow"))
        .toString()
        .trimmed()
        .toLower();
    if (isKnownWorkflow(selected))
    {
        normalized[QStringLiteral("selected_workflow")] = selected;
    }

    // 保留未知工作流程对象，允许更新后的程序在旧版本中打开并保存项目时不丢数据。
    QJsonObject workflows = settings.value(QStringLiteral("workflows")).toObject();
    QJsonObject normalizedWorkflows = normalized.value(QStringLiteral("workflows")).toObject();
    for (auto it = workflows.constBegin(); it != workflows.constEnd(); ++it)
    {
        if (it.key() != QLatin1String(kRetiredReconstructionWorkflowId) &&
            it.value().isObject())
        {
            normalizedWorkflows.insert(it.key(), it.value());
        }
    }
    normalizedWorkflows[QString::fromLatin1(kAerialWorkflowId)] =
        WorkflowSettingsDialog::aerialTriangulationSettings(settings);
    normalizedWorkflows[QString::fromLatin1(kModelGenerationWorkflowId)] =
        WorkflowSettingsDialog::modelGenerationSettings(settings);
    normalized[QStringLiteral("workflows")] = normalizedWorkflows;
    return normalized;
}

QWidget *makeUnavailableWorkflowPage(const QString &workflowId,
                                     const QString &title,
                                     QWidget *parent)
{
    auto *page = new QWidget(parent);
    page->setObjectName(QStringLiteral("workflowPage_%1").arg(workflowId));
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);

    auto *message = new QLabel(
        QStringLiteral("%1 的项目级设置尚未开放。\n"
                       "运行时请在对应工作流对话框中设置已支持的参数。")
            .arg(title),
        page);
    message->setObjectName(QStringLiteral("workflowUnavailableMessage_%1").arg(workflowId));
    message->setWordWrap(true);
    layout->addWidget(message);
    layout->addStretch(1);
    return page;
}

} // namespace

WorkflowSettingsDialog::WorkflowSettingsDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    applySettings(defaultSettings());
}

QJsonObject WorkflowSettingsDialog::defaultSettings()
{
    QJsonObject workflows;
    workflows[QString::fromLatin1(kAerialWorkflowId)] = defaultAerialSettings();
    workflows[QString::fromLatin1(kModelGenerationWorkflowId)] =
        defaultModelGenerationSettings();
    workflows[QString::fromLatin1(kDemWorkflowId)] = QJsonObject();
    workflows[QString::fromLatin1(kOrthomosaicWorkflowId)] = QJsonObject();

    QJsonObject settings;
    settings[QStringLiteral("workflow_settings_version")] = kWorkflowSettingsVersion;
    settings[QStringLiteral("selected_workflow")] = QString::fromLatin1(kAerialWorkflowId);
    settings[QStringLiteral("workflows")] = workflows;
    return settings;
}

QJsonObject WorkflowSettingsDialog::aerialTriangulationSettings(const QJsonObject &settings)
{
    QJsonObject source;
    const QJsonObject workflows = settings.value(QStringLiteral("workflows")).toObject();
    if (workflows.value(QString::fromLatin1(kAerialWorkflowId)).isObject())
    {
        source = workflows.value(QString::fromLatin1(kAerialWorkflowId)).toObject();
    }
    else
    {
        // v2 将空三字段直接放在根对象中；只迁移仍属于工作流程决策的字段。
        source = settings;
    }

    QJsonObject aerial = defaultAerialSettings();
    const QString algorithmId = source.value(QStringLiteral("algorithm_id"))
        .toString()
        .trimmed()
        .toLower();
    if (!algorithmId.isEmpty())
    {
        aerial[QStringLiteral("algorithm_id")] = algorithmId;
    }
    aerial[QStringLiteral("lightglue_tensorrt_engine")] =
        source.value(QStringLiteral("lightglue_tensorrt_engine")).toString().trimmed();
    aerial[QStringLiteral("loma_r_tensorrt_package")] =
        source.value(QStringLiteral("loma_r_tensorrt_package")).toString().trimmed();
    const int lomaRKeypointBudget =
        source.value(QStringLiteral("loma_r_keypoint_budget")).toInt();
    aerial[QStringLiteral("loma_r_keypoint_budget")] =
        lomaRKeypointBudget == 1024 || lomaRKeypointBudget == 2048 ||
                lomaRKeypointBudget == 3840
            ? lomaRKeypointBudget
            : 0;
    return aerial;
}

QJsonObject WorkflowSettingsDialog::modelGenerationSettings(
    const QJsonObject &settings)
{
    const QJsonObject workflows = settings.value(
        QStringLiteral("workflows")).toObject();
    const QJsonObject source = workflows.value(
        QString::fromLatin1(kModelGenerationWorkflowId)).toObject();

    QJsonObject model_settings = defaultModelGenerationSettings();
    const QString compute_mode = source.value(QStringLiteral("compute_mode"))
        .toString()
        .trimmed()
        .toLower();
    if (compute_mode == QLatin1String(kCudaComputeMode) ||
        compute_mode == QLatin1String(kOpenClComputeMode) ||
        compute_mode == QLatin1String(kHybridComputeMode))
    {
        model_settings[QStringLiteral("compute_mode")] = compute_mode;
    }
    return model_settings;
}

void WorkflowSettingsDialog::setupUi()
{
    setWindowTitle(QStringLiteral("工作流程设置"));
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);
    setModal(true);
    resize(720, 500);
    setMinimumSize(620, 420);

    auto *rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(20, 18, 20, 18);
    rootLayout->setSpacing(14);

    auto *workflowForm = new QFormLayout();
    workflowForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    _workflowCombo = new QComboBox(this);
    _workflowCombo->setObjectName(QStringLiteral("workflowSelector"));
    xjw::gui::dialogs::configureWorkflowComboBox(_workflowCombo);
    for (const WorkflowEntry &entry : kWorkflowEntries)
    {
        _workflowCombo->addItem(QString::fromUtf8(entry.displayName),
                                QString::fromLatin1(entry.id));
    }
    workflowForm->addRow(QStringLiteral("工作流程:"), _workflowCombo);
    rootLayout->addLayout(workflowForm);

    _workflowPages = new QStackedWidget(this);
    _workflowPages->setObjectName(QStringLiteral("workflowPages"));

    auto *aerialPage = new QWidget(_workflowPages);
    aerialPage->setObjectName(QStringLiteral("workflowPage_aerial_triangulation"));
    auto *aerialLayout = new QVBoxLayout(aerialPage);
    aerialLayout->setContentsMargins(0, 0, 0, 0);
    auto *aerialGroup = new QGroupBox(QStringLiteral("空中三角测量"), aerialPage);
    auto *aerialForm = new QFormLayout(aerialGroup);
    aerialForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    _matchingAlgorithmCombo = new QComboBox(aerialGroup);
    _matchingAlgorithmCombo->setObjectName(QStringLiteral("aerialMatchingAlgorithmCombo"));
    xjw::gui::dialogs::configureWorkflowComboBox(_matchingAlgorithmCombo);
    populateMatchingAlgorithms();
    aerialForm->addRow(QStringLiteral("匹配算法:"), _matchingAlgorithmCombo);

    _lomaRKeypointBudgetCombo = new QComboBox(aerialGroup);
    _lomaRKeypointBudgetCombo->setObjectName(
        QStringLiteral("aerialLoMaRKeypointBudgetCombo"));
    xjw::gui::dialogs::configureWorkflowComboBox(_lomaRKeypointBudgetCombo);
    _lomaRKeypointBudgetCombo->addItem(QStringLiteral("自动（按显存）"), 0);
    _lomaRKeypointBudgetCombo->addItem(QStringLiteral("标准（1024）"), 1024);
    _lomaRKeypointBudgetCombo->addItem(QStringLiteral("高（2048）"), 2048);
    _lomaRKeypointBudgetCombo->addItem(QStringLiteral("最高（3840）"), 3840);
    aerialForm->addRow(QStringLiteral("LoMa-R 特征档位:"),
                       _lomaRKeypointBudgetCombo);

    auto *enginePathRow = new QWidget(aerialGroup);
    auto *enginePathLayout = new QHBoxLayout(enginePathRow);
    enginePathLayout->setContentsMargins(0, 0, 0, 0);
    enginePathLayout->setSpacing(6);
    _matchingResourceEdit = new QLineEdit(enginePathRow);
    _matchingResourceEdit->setObjectName(QStringLiteral("aerialMatchingResourceEdit"));
    xjw::gui::dialogs::configureWorkflowInputWidget(_matchingResourceEdit);
    _matchingResourceEdit->setPlaceholderText(QStringLiteral("自动选择本机模型资源"));
    _matchingResourceEdit->setClearButtonEnabled(true);
    _matchingResourceBrowseButton = new QToolButton(enginePathRow);
    _matchingResourceBrowseButton->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
    _matchingResourceBrowseButton->setToolTip(
        QStringLiteral("选择 ONNX 模型或兼容的本机 TensorRT 资源"));
    _matchingResourceBrowseButton->setFixedSize(32, 30);
    _downloadModelButton = new QPushButton(QStringLiteral("下载模型"), enginePathRow);
    _downloadModelButton->setObjectName(QStringLiteral("aerialDownloadModelButton"));
    _downloadModelButton->setToolTip(QStringLiteral("从 PlaScan GitHub Release 下载已校验模型"));
    enginePathLayout->addWidget(_matchingResourceEdit, 1);
    enginePathLayout->addWidget(_matchingResourceBrowseButton);
    enginePathLayout->addWidget(_downloadModelButton);
    aerialForm->addRow(QStringLiteral("模型资源:"), enginePathRow);

    _matchingResourceStatusLabel = new QLabel(aerialGroup);
    _matchingResourceStatusLabel->setObjectName(
        QStringLiteral("aerialMatchingResourceStatusLabel"));
    _matchingResourceStatusLabel->setWordWrap(true);
    _matchingResourceStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    aerialForm->addRow(QStringLiteral("当前生效:"), _matchingResourceStatusLabel);
    aerialLayout->addWidget(aerialGroup);
    aerialLayout->addStretch(1);
    _workflowPages->addWidget(aerialPage);

    auto *model_page = new QWidget(_workflowPages);
    model_page->setObjectName(QStringLiteral("workflowPage_generate_model"));
    auto *model_layout = new QVBoxLayout(model_page);
    model_layout->setContentsMargins(0, 0, 0, 0);
    auto *model_group = new QGroupBox(QStringLiteral("生成模型"), model_page);
    auto *model_form = new QFormLayout(model_group);
    model_form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    _modelComputeModeCombo = new QComboBox(model_group);
    _modelComputeModeCombo->setObjectName(
        QStringLiteral("modelComputeModeCombo"));
    xjw::gui::dialogs::configureWorkflowComboBox(_modelComputeModeCombo);
    _modelComputeModeCombo->addItem(
        QStringLiteral("只用 CUDA"), QString::fromLatin1(kCudaComputeMode));
    _modelComputeModeCombo->addItem(
        QStringLiteral("只用 OpenCL"), QString::fromLatin1(kOpenClComputeMode));
    _modelComputeModeCombo->addItem(
        QStringLiteral("混合使用 CUDA + OpenCL"),
        QString::fromLatin1(kHybridComputeMode));
    model_form->addRow(QStringLiteral("计算模式:"), _modelComputeModeCombo);

    _cudaDeviceStatusLabel = new QLabel(model_group);
    _cudaDeviceStatusLabel->setObjectName(
        QStringLiteral("modelCudaDeviceStatusLabel"));
    _cudaDeviceStatusLabel->setWordWrap(true);
    _cudaDeviceStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    model_form->addRow(QStringLiteral("CUDA 设备:"), _cudaDeviceStatusLabel);

    _openClDeviceStatusLabel = new QLabel(model_group);
    _openClDeviceStatusLabel->setObjectName(
        QStringLiteral("modelOpenClDeviceStatusLabel"));
    _openClDeviceStatusLabel->setWordWrap(true);
    _openClDeviceStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    model_form->addRow(QStringLiteral("OpenCL 设备:"), _openClDeviceStatusLabel);

    _modelComputePolicyLabel = new QLabel(model_group);
    _modelComputePolicyLabel->setObjectName(
        QStringLiteral("modelComputePolicyLabel"));
    _modelComputePolicyLabel->setWordWrap(true);
    _modelComputePolicyLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    model_form->addRow(QStringLiteral("实际策略:"), _modelComputePolicyLabel);

    _detectComputeDevicesButton = new QPushButton(
        QStringLiteral("重新检测设备"), model_group);
    _detectComputeDevicesButton->setObjectName(
        QStringLiteral("modelDetectComputeDevicesButton"));
    model_form->addRow(QString(), _detectComputeDevicesButton);
    model_layout->addWidget(model_group);
    model_layout->addStretch(1);
    _workflowPages->addWidget(model_page);

    _workflowPages->addWidget(makeUnavailableWorkflowPage(
        QString::fromLatin1(kDemWorkflowId), QStringLiteral("创建 DEM"), _workflowPages));
    _workflowPages->addWidget(makeUnavailableWorkflowPage(
        QString::fromLatin1(kOrthomosaicWorkflowId), QStringLiteral("生成正射影像"), _workflowPages));
    rootLayout->addWidget(_workflowPages, 1);

    connect(_workflowCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &WorkflowSettingsDialog::setCurrentWorkflow);
    connect(_matchingAlgorithmCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this]() { switchAlgorithmResource(); });
    connect(_matchingResourceEdit, &QLineEdit::textChanged,
            this, [this]() { refreshMatchingResourceStatus(); });
    connect(_lomaRKeypointBudgetCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this]() { refreshMatchingResourceStatus(); });
    connect(_matchingResourceBrowseButton, &QToolButton::clicked, this, [this]()
    {
        const QString algorithmId = _matchingAlgorithmCombo->currentData().toString();
        const bool loMaR = algorithmId == QLatin1String(kLoMaRAlgorithmId);
        const QString currentPath = _matchingResourceEdit->text().trimmed();
        const QString startPath = currentPath.isEmpty()
            ? QString()
            : QFileInfo(currentPath).absolutePath();
        const QString selected = QFileDialog::getOpenFileName(
            this,
            loMaR ? QStringLiteral("选择 LoMa-R ONNX 模型清单")
                  : QStringLiteral("选择 LightGlue ONNX 模型"),
            startPath,
            loMaR ? QStringLiteral("LoMa-R manifest (*.json);;所有文件 (*)")
                  : QStringLiteral("ONNX 模型 (*.onnx);;兼容的本机 engine (*.engine);;所有文件 (*)"));
        if (!selected.isEmpty())
        {
            _matchingResourceEdit->setText(QFileInfo(selected).absoluteFilePath());
        }
    });
    connect(_downloadModelButton, &QPushButton::clicked,
            this, &WorkflowSettingsDialog::downloadCurrentModelPackage);
    connect(_modelComputeModeCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            [this]() { refreshModelComputePolicy(); });
    connect(_detectComputeDevicesButton,
            &QPushButton::clicked,
            this,
            &WorkflowSettingsDialog::refreshModelComputeDevices);

    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
        this);
    xjw::gui::dialogs::configureWorkflowButtonBox(buttons);
    buttons->button(QDialogButtonBox::RestoreDefaults)->setText(QStringLiteral("恢复默认值"));
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(buttons->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked,
            this, [this]() { applySettings(defaultSettings()); });
    rootLayout->addWidget(buttons);
}

void WorkflowSettingsDialog::populateMatchingAlgorithms()
{
    _matchingAlgorithmCombo->clear();
    const auto descriptors = xjw::image_matching::ImageMatchingRegistry::descriptors();
    for (const auto &descriptor : descriptors)
    {
        _matchingAlgorithmCombo->addItem(descriptor.displayName, descriptor.id);
    }
    if (_matchingAlgorithmCombo->count() == 0)
    {
        _matchingAlgorithmCombo->addItem(QStringLiteral("无可用算法"), QString());
        _matchingAlgorithmCombo->setEnabled(false);
    }
}

void WorkflowSettingsDialog::applySettings(const QJsonObject &requestedSettings)
{
    _appliedSettings = normalizedSettings(requestedSettings);
    const QString selectedWorkflow = _appliedSettings
        .value(QStringLiteral("selected_workflow"))
        .toString();
    const int workflowIndex = _workflowCombo->findData(selectedWorkflow);
    _workflowCombo->setCurrentIndex(workflowIndex >= 0 ? workflowIndex : 0);

    const QJsonObject aerial = aerialTriangulationSettings(_appliedSettings);
    const QString algorithmId = aerial.value(QStringLiteral("algorithm_id")).toString();
    const int algorithmIndex = _matchingAlgorithmCombo->findData(algorithmId);
    _matchingAlgorithmCombo->setCurrentIndex(algorithmIndex >= 0 ? algorithmIndex : 0);
    _lightGlueEnginePath =
        aerial.value(QStringLiteral("lightglue_tensorrt_engine")).toString();
    _lomaRPackagePath =
        aerial.value(QStringLiteral("loma_r_tensorrt_package")).toString();
    const int budgetIndex = _lomaRKeypointBudgetCombo->findData(
        aerial.value(QStringLiteral("loma_r_keypoint_budget")).toInt());
    _lomaRKeypointBudgetCombo->setCurrentIndex(budgetIndex >= 0 ? budgetIndex : 0);

    const QJsonObject model_settings = modelGenerationSettings(_appliedSettings);
    const int compute_mode_index = _modelComputeModeCombo->findData(
        model_settings.value(QStringLiteral("compute_mode")).toString());
    _modelComputeModeCombo->setCurrentIndex(
        compute_mode_index >= 0 ? compute_mode_index : 2);
    _currentAlgorithmId.clear();
    switchAlgorithmResource();
    refreshModelComputeDevices();
    setCurrentWorkflow(_workflowCombo->currentIndex());
}

QJsonObject WorkflowSettingsDialog::collectSettings() const
{
    QJsonObject settings = normalizedSettings(_appliedSettings);
    settings[QStringLiteral("workflow_settings_version")] = kWorkflowSettingsVersion;
    settings[QStringLiteral("selected_workflow")] =
        _workflowCombo->currentData().toString();

    QJsonObject workflows = settings.value(QStringLiteral("workflows")).toObject();
    QJsonObject aerial = workflows.value(QString::fromLatin1(kAerialWorkflowId)).toObject();
    const QString algorithmId = _matchingAlgorithmCombo->currentData().toString();
    aerial[QStringLiteral("algorithm_id")] = algorithmId.isEmpty()
        ? QString::fromLatin1(kSiftLightGlueAlgorithmId)
        : algorithmId;
    QString lightGluePath = _lightGlueEnginePath;
    QString loMaRPath = _lomaRPackagePath;
    if (_currentAlgorithmId == QLatin1String(kSiftLightGlueAlgorithmId))
    {
        lightGluePath = _matchingResourceEdit->text().trimmed();
    }
    else if (_currentAlgorithmId == QLatin1String(kLoMaRAlgorithmId))
    {
        loMaRPath = _matchingResourceEdit->text().trimmed();
    }
    aerial[QStringLiteral("lightglue_tensorrt_engine")] = lightGluePath;
    aerial[QStringLiteral("loma_r_tensorrt_package")] = loMaRPath;
    aerial[QStringLiteral("loma_r_keypoint_budget")] =
        _lomaRKeypointBudgetCombo->currentData().toInt();
    workflows[QString::fromLatin1(kAerialWorkflowId)] = aerial;
    QJsonObject model_settings = workflows.value(
        QString::fromLatin1(kModelGenerationWorkflowId)).toObject();
    model_settings[QStringLiteral("compute_mode")] =
        _modelComputeModeCombo->currentData().toString();
    workflows[QString::fromLatin1(kModelGenerationWorkflowId)] =
        model_settings;
    settings[QStringLiteral("workflows")] = workflows;
    return settings;
}

void WorkflowSettingsDialog::setCurrentWorkflow(int index)
{
    if (_workflowPages && index >= 0 && index < _workflowPages->count())
    {
        _workflowPages->setCurrentIndex(index);
    }
}

void WorkflowSettingsDialog::refreshAlgorithmControls()
{
    const QString algorithmId = _matchingAlgorithmCombo->currentData().toString();
    const bool modelBacked = algorithmId == QLatin1String(kSiftLightGlueAlgorithmId) ||
        algorithmId == QLatin1String(kLoMaRAlgorithmId);
    const bool cudaSift = algorithmId == QLatin1String(kCudaSiftAlgorithmId);
    _matchingResourceEdit->setEnabled(modelBacked);
    _matchingResourceBrowseButton->setEnabled(modelBacked);
    _downloadModelButton->setEnabled(modelBacked);
    _lomaRKeypointBudgetCombo->setEnabled(
        algorithmId == QLatin1String(kLoMaRAlgorithmId));
    if (modelBacked)
    {
        refreshMatchingResourceStatus();
    }
    else if (cudaSift)
    {
        QPalette palette = _matchingResourceStatusLabel->palette();
        palette.setColor(QPalette::WindowText, QColor(35, 110, 70));
        _matchingResourceStatusLabel->setPalette(palette);
        _matchingResourceStatusLabel->setText(
            QStringLiteral("内置 SIFT，无需下载模型；自动优先 CUDA，"
                           "CUDA 不可用时回退 OpenCV CPU"));
        _downloadModelButton->setVisible(false);
    }
    else
    {
        _matchingResourceStatusLabel->setText(QStringLiteral("不适用"));
        _downloadModelButton->setVisible(false);
    }
}

void WorkflowSettingsDialog::switchAlgorithmResource()
{
    if (!_matchingResourceEdit || !_matchingAlgorithmCombo)
    {
        return;
    }
    if (_currentAlgorithmId == QLatin1String(kSiftLightGlueAlgorithmId))
    {
        _lightGlueEnginePath = _matchingResourceEdit->text().trimmed();
    }
    else if (_currentAlgorithmId == QLatin1String(kLoMaRAlgorithmId))
    {
        _lomaRPackagePath = _matchingResourceEdit->text().trimmed();
    }
    _currentAlgorithmId = _matchingAlgorithmCombo->currentData().toString();
    const QSignalBlocker block(_matchingResourceEdit);
    QString resourcePath;
    if (_currentAlgorithmId == QLatin1String(kLoMaRAlgorithmId))
    {
        resourcePath = _lomaRPackagePath;
    }
    else if (_currentAlgorithmId == QLatin1String(kSiftLightGlueAlgorithmId))
    {
        resourcePath = _lightGlueEnginePath;
    }
    _matchingResourceEdit->setText(resourcePath);
    refreshAlgorithmControls();
}

void WorkflowSettingsDialog::refreshMatchingResourceStatus()
{
    if (!_matchingResourceEdit || !_matchingResourceStatusLabel)
    {
        return;
    }

    xjw::matchphotos::MatchPhotosOptions options;
    const QString algorithmId = _matchingAlgorithmCombo->currentData().toString();
    QString resolvedPath;
    QString detail;
    if (algorithmId == QLatin1String(kLoMaRAlgorithmId))
    {
        options.lomaRTensorRtPackagePath = _matchingResourceEdit->text().trimmed();
        options.lomaRKeypointBudget = _lomaRKeypointBudgetCombo->currentData().toInt();
        const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(
            options, 40000, false);
        resolvedPath = resolved.manifestPath;
        detail = resolved.isValid()
            ? QStringLiteral("  [K=%1, D=%2, %3x%4]")
                  .arg(resolved.keypointCount)
                  .arg(resolved.descriptorDimension)
                  .arg(resolved.inputWidth)
                  .arg(resolved.inputHeight)
            : resolved.errorMessage;
    }
    else if (algorithmId == QLatin1String(kSiftLightGlueAlgorithmId))
    {
        options.lightGlueTensorRtEnginePath = _matchingResourceEdit->text().trimmed();
        const auto resolved = xjw::matchphotos::resolveLightGlueTensorRtEngine(
            options, 4096, false);
        resolvedPath = resolved.path;
        detail = resolved.bucketKeypoints > 0
            ? QStringLiteral("  [K=%1]").arg(resolved.bucketKeypoints)
            : QString();
    }
    else if (algorithmId == QLatin1String(kCudaSiftAlgorithmId))
    {
        QPalette palette = _matchingResourceStatusLabel->palette();
        palette.setColor(QPalette::WindowText, QColor(35, 110, 70));
        _matchingResourceStatusLabel->setPalette(palette);
        _matchingResourceStatusLabel->setText(
            QStringLiteral("内置 SIFT，无需下载模型；自动优先 CUDA，"
                           "CUDA 不可用时回退 OpenCV CPU"));
        _downloadModelButton->setVisible(false);
        return;
    }

    QPalette palette = _matchingResourceStatusLabel->palette();
    if (resolvedPath.isEmpty())
    {
        palette.setColor(QPalette::WindowText, QColor(180, 45, 45));
        _matchingResourceStatusLabel->setPalette(palette);
        _matchingResourceStatusLabel->setText(
            detail.isEmpty() ? QStringLiteral("未找到自动模型资源") : detail);
        _downloadModelButton->setVisible(true);
        return;
    }

    palette.setColor(QPalette::WindowText, QColor(35, 110, 70));
    _matchingResourceStatusLabel->setPalette(palette);
    _matchingResourceStatusLabel->setText(
        QDir::toNativeSeparators(resolvedPath) + detail);
    _downloadModelButton->setVisible(false);
}

void WorkflowSettingsDialog::downloadCurrentModelPackage()
{
    const QString algorithmId = _matchingAlgorithmCombo->currentData().toString();
    xjw::common::model::ModelAssetPackage package;
    if (algorithmId == QLatin1String(kLoMaRAlgorithmId))
    {
        const int requestedBudget = _lomaRKeypointBudgetCombo->currentData().toInt();
        const int effectiveBudget = xjw::matchphotos::resolveLoMaRKeypointBudget(
            40000,
            requestedBudget,
            xjw::matchphotos::queryMatchPhotosGpuMemory(0));
        package = xjw::common::model::loMaRTensorRtPackage(effectiveBudget);
    }
    else if (algorithmId == QLatin1String(kSiftLightGlueAlgorithmId))
    {
        package = xjw::common::model::lightGlueTensorRtPackage();
    }
    else
    {
        return;
    }

    const xjw::common::model::ModelFileResolver resolver;
    const QString targetDirectory =
        xjw::common::model::modelPackageInstallDirectory(package, resolver);
    QString errorMessage;
    if (!ModelPackageDownloadDialog::downloadPackage(
            package, targetDirectory, this, &errorMessage))
    {
        return;
    }

    _matchingResourceEdit->setText(
        xjw::common::model::modelPackageEntryPoint(package, resolver));
    refreshMatchingResourceStatus();
}

void WorkflowSettingsDialog::refreshModelComputeDevices()
{
    if (!_modelComputeModeCombo || !_cudaDeviceStatusLabel ||
        !_openClDeviceStatusLabel)
    {
        return;
    }

    QStringList cuda_lines;
    QSet<QString> cuda_identities;
    const int cuda_device_count =
        xjw::mvs::PatchMatchDepthEstimator::cudaDeviceCount();
    for (int device_index = 0;
         device_index < cuda_device_count;
         ++device_index)
    {
        const QString name = QString::fromStdString(
            xjw::mvs::PatchMatchDepthEstimator::cudaDeviceName(device_index));
        if (name.isEmpty())
        {
            continue;
        }
        cuda_lines.append(QStringLiteral("[%1] %2").arg(device_index).arg(name));
        const QString identity = QString::fromStdString(
            xjw::mvs::PatchMatchDepthEstimator::cudaDeviceIdentity(device_index));
        if (!identity.isEmpty())
        {
            cuda_identities.insert(identity);
        }
    }
    _cudaAvailable = !cuda_lines.isEmpty();

    QStringList opencl_lines;
    int eligible_opencl_devices = 0;
    int independent_opencl_devices = 0;
    const std::vector<xjw::mvs::OpenClDeviceInfo> opencl_devices =
        xjw::mvs::PatchMatchDepthEstimator::openClDevices();
    for (const xjw::mvs::OpenClDeviceInfo &device : opencl_devices)
    {
        const QString identity = QString::fromStdString(
            device.physicalDeviceIdentity);
        const bool nvidia_opencl =
            xjw::mvs::isNvidiaOpenClVendor(device.vendor);
        const bool duplicates_cuda = !identity.isEmpty() &&
            cuda_identities.contains(identity);
        const bool auto_deduplicated_nvidia = _cudaAvailable && nvidia_opencl;
        ++eligible_opencl_devices;
        if (!duplicates_cuda && !auto_deduplicated_nvidia)
        {
            ++independent_opencl_devices;
        }
        const double memory_gib = static_cast<double>(device.globalMemoryBytes) /
            (1024.0 * 1024.0 * 1024.0);
        QString detail = QStringLiteral("[%1] %2 · %3 · %4 CU")
            .arg(device.index)
            .arg(QString::fromStdString(device.vendor))
            .arg(QString::fromStdString(device.name))
            .arg(device.computeUnits);
        if (memory_gib > 0.0)
        {
            detail += QStringLiteral(" · %1 GiB").arg(memory_gib, 0, 'f', 1);
        }
        if (duplicates_cuda || auto_deduplicated_nvidia)
        {
            detail += QStringLiteral("（可用于显式 OpenCL；混合模式与 CUDA 去重）");
        }
        opencl_lines.append(detail);
    }
    _openClAvailable = eligible_opencl_devices > 0;
    _hybridAvailable = _cudaAvailable && independent_opencl_devices > 0;

    QPalette cuda_palette = _cudaDeviceStatusLabel->palette();
    cuda_palette.setColor(
        QPalette::WindowText,
        _cudaAvailable ? QColor(35, 110, 70) : QColor(180, 45, 45));
    _cudaDeviceStatusLabel->setPalette(cuda_palette);
    _cudaDeviceStatusLabel->setText(
        _cudaAvailable
            ? QStringLiteral("可用，共 %1 个设备\n%2")
                  .arg(cuda_lines.size())
                  .arg(cuda_lines.join(QLatin1Char('\n')))
            : QStringLiteral("不可用：当前构建或驱动未提供可用 CUDA 设备"));

    QPalette opencl_palette = _openClDeviceStatusLabel->palette();
    opencl_palette.setColor(
        QPalette::WindowText,
        _openClAvailable ? QColor(35, 110, 70) : QColor(180, 45, 45));
    _openClDeviceStatusLabel->setPalette(opencl_palette);
    _openClDeviceStatusLabel->setText(
        _openClAvailable
            ? QStringLiteral("可用，共 %1 个设备\n%2")
                  .arg(eligible_opencl_devices)
                  .arg(opencl_lines.join(QLatin1Char('\n')))
            : QStringLiteral(
                  "不可用：未检测到同时支持运行和在线编译的 OpenCL GPU"));

    auto set_mode_enabled = [this](const char *mode, bool enabled)
    {
        const int index = _modelComputeModeCombo->findData(
            QString::fromLatin1(mode));
        auto *model = qobject_cast<QStandardItemModel *>(
            _modelComputeModeCombo->model());
        if (model && index >= 0)
        {
            if (QStandardItem *item = model->item(index))
            {
                item->setEnabled(enabled);
            }
        }
    };
    set_mode_enabled(kCudaComputeMode, _cudaAvailable);
    set_mode_enabled(kOpenClComputeMode, _openClAvailable);
    set_mode_enabled(kHybridComputeMode, _hybridAvailable);

    const QString selected_mode = _modelComputeModeCombo->currentData().toString();
    const bool selected_available =
        (selected_mode == QLatin1String(kCudaComputeMode) && _cudaAvailable) ||
        (selected_mode == QLatin1String(kOpenClComputeMode) && _openClAvailable) ||
        (selected_mode == QLatin1String(kHybridComputeMode) && _hybridAvailable);
    if (!selected_available)
    {
        const QString fallback_mode = _hybridAvailable
            ? QString::fromLatin1(kHybridComputeMode)
            : (_cudaAvailable
                   ? QString::fromLatin1(kCudaComputeMode)
                   : (_openClAvailable
                          ? QString::fromLatin1(kOpenClComputeMode)
                          : selected_mode));
        const int fallback_index = _modelComputeModeCombo->findData(fallback_mode);
        if (fallback_index >= 0)
        {
            _modelComputeModeCombo->setCurrentIndex(fallback_index);
        }
    }
    _modelComputeModeCombo->setEnabled(_cudaAvailable || _openClAvailable);
    refreshModelComputePolicy();
}

void WorkflowSettingsDialog::refreshModelComputePolicy()
{
    if (!_modelComputeModeCombo || !_modelComputePolicyLabel)
    {
        return;
    }

    const QString mode = _modelComputeModeCombo->currentData().toString();
    QString policy;
    bool available = false;
    if (mode == QLatin1String(kCudaComputeMode))
    {
        available = _cudaAvailable;
        policy = QStringLiteral(
            "深度图估计、点云预处理和可用的模型求解阶段仅使用 CUDA；"
            "不会调度 OpenCL 设备。");
    }
    else if (mode == QLatin1String(kOpenClComputeMode))
    {
        available = _openClAvailable;
        policy = QStringLiteral(
            "深度图估计、点云预处理和 Poisson 稀疏求解仅使用 OpenCL；"
            "自动忽略 NVIDIA 设备且不会调用 CUDA。网格提取等串行阶段仍在 CPU 执行。");
    }
    else
    {
        available = _hybridAvailable;
        policy = QStringLiteral(
            "深度图估计会并行调度 CUDA 与独立 OpenCL GPU；"
            "同一物理显卡的重复接口会自动去重。模型预处理优先 OpenCL，"
            "Poisson 求解使用 CUDA。");
    }

    QPalette palette = _modelComputePolicyLabel->palette();
    palette.setColor(
        QPalette::WindowText,
        available ? QColor(35, 80, 120) : QColor(180, 45, 45));
    _modelComputePolicyLabel->setPalette(palette);
    _modelComputePolicyLabel->setText(
        available
            ? policy
            : QStringLiteral("当前机器不满足该模式：") + policy);
}
