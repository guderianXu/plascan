/**
 * @file WorkflowSettingsDialog.cpp
 * @brief 多工作流程项目设置对话框实现。
 */

#include "application/WorkflowSettingsDialog.h"

#include "application/ModelPackageDownloadDialog.h"

#include "ImageMatchingRegistry.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosRuntime.h"
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
#include <QSignalBlocker>
#include <QStackedWidget>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidget>

namespace
{

constexpr auto kAerialWorkflowId = "aerial_triangulation";
constexpr auto kReconstructionWorkflowId = "reconstruction";
constexpr auto kDemWorkflowId = "dem";
constexpr auto kOrthomosaicWorkflowId = "orthomosaic";
constexpr auto kSiftLightGlueAlgorithmId = "sift_lightglue";
constexpr auto kLoMaRAlgorithmId = "loma_r";
constexpr int kWorkflowSettingsVersion = 5;

struct WorkflowEntry
{
    const char *id;
    const char *displayName;
};

constexpr WorkflowEntry kWorkflowEntries[] = {
    {kAerialWorkflowId, "空中三角测量"},
    {kReconstructionWorkflowId, "三维重建"},
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
        if (it.value().isObject())
        {
            normalizedWorkflows.insert(it.key(), it.value());
        }
    }
    normalizedWorkflows[QString::fromLatin1(kAerialWorkflowId)] =
        WorkflowSettingsDialog::aerialTriangulationSettings(settings);
    normalized[QStringLiteral("workflows")] = normalizedWorkflows;
    return normalized;
}

QWidget *makeReadOnlyWorkflowPage(const QString &workflowId,
                                  const QString &title,
                                  QWidget *parent)
{
    auto *page = new QWidget(parent);
    page->setObjectName(QStringLiteral("workflowPage_%1").arg(workflowId));
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);

    auto *group = new QGroupBox(title, page);
    auto *form = new QFormLayout(group);
    form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    auto *profile = new QComboBox(group);
    profile->addItem(QStringLiteral("默认"));
    form->addRow(QStringLiteral("参数方案:"), profile);
    layout->addWidget(group);
    layout->addStretch(1);

    // 页面仍可通过顶部选择器查看，但整个设置区域不可编辑。
    page->setEnabled(false);
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
    workflows[QString::fromLatin1(kReconstructionWorkflowId)] = QJsonObject();
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

void WorkflowSettingsDialog::setupUi()
{
    setWindowTitle(QStringLiteral("工作流程设置"));
    setModal(true);
    resize(680, 360);
    setMinimumSize(580, 320);

    auto *rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(20, 18, 20, 18);
    rootLayout->setSpacing(14);

    auto *workflowForm = new QFormLayout();
    workflowForm->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    _workflowCombo = new QComboBox(this);
    _workflowCombo->setObjectName(QStringLiteral("workflowSelector"));
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
    populateMatchingAlgorithms();
    aerialForm->addRow(QStringLiteral("匹配算法:"), _matchingAlgorithmCombo);

    _lomaRKeypointBudgetCombo = new QComboBox(aerialGroup);
    _lomaRKeypointBudgetCombo->setObjectName(
        QStringLiteral("aerialLoMaRKeypointBudgetCombo"));
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
    _matchingResourceEdit->setPlaceholderText(QStringLiteral("自动选择本机模型资源"));
    _matchingResourceEdit->setClearButtonEnabled(true);
    _matchingResourceBrowseButton = new QToolButton(enginePathRow);
    _matchingResourceBrowseButton->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
    _matchingResourceBrowseButton->setToolTip(QStringLiteral("选择 TensorRT 模型资源"));
    _matchingResourceBrowseButton->setFixedSize(32, 30);
    _downloadModelButton = new QPushButton(QStringLiteral("下载模型"), enginePathRow);
    _downloadModelButton->setObjectName(QStringLiteral("aerialDownloadModelButton"));
    _downloadModelButton->setToolTip(QStringLiteral("从 PlaScan GitHub Release 下载已校验模型"));
    enginePathLayout->addWidget(_matchingResourceEdit, 1);
    enginePathLayout->addWidget(_matchingResourceBrowseButton);
    enginePathLayout->addWidget(_downloadModelButton);
    aerialForm->addRow(QStringLiteral("模型资源:"), enginePathRow);

    _matchingResourceStatusLabel = new QLabel(aerialGroup);
    _matchingResourceStatusLabel->setWordWrap(true);
    _matchingResourceStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    aerialForm->addRow(QStringLiteral("当前生效:"), _matchingResourceStatusLabel);
    aerialLayout->addWidget(aerialGroup);
    aerialLayout->addStretch(1);
    _workflowPages->addWidget(aerialPage);

    _workflowPages->addWidget(makeReadOnlyWorkflowPage(
        QString::fromLatin1(kReconstructionWorkflowId), QStringLiteral("三维重建"), _workflowPages));
    _workflowPages->addWidget(makeReadOnlyWorkflowPage(
        QString::fromLatin1(kDemWorkflowId), QStringLiteral("创建 DEM"), _workflowPages));
    _workflowPages->addWidget(makeReadOnlyWorkflowPage(
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
            loMaR ? QStringLiteral("选择 LoMa-R TensorRT 模型清单")
                  : QStringLiteral("选择 TensorRT LightGlue 引擎"),
            startPath,
            loMaR ? QStringLiteral("LoMa-R manifest (*.json);;所有文件 (*)")
                  : QStringLiteral("TensorRT engine (*.engine);;所有文件 (*)"));
        if (!selected.isEmpty())
        {
            _matchingResourceEdit->setText(QFileInfo(selected).absoluteFilePath());
        }
    });
    connect(_downloadModelButton, &QPushButton::clicked,
            this, &WorkflowSettingsDialog::downloadCurrentModelPackage);

    auto *buttons = new QDialogButtonBox(
        QDialogButtonBox::RestoreDefaults | QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
        this);
    buttons->button(QDialogButtonBox::RestoreDefaults)->setText(QStringLiteral("恢复默认值"));
    buttons->button(QDialogButtonBox::Ok)->setText(QStringLiteral("确定"));
    buttons->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("取消"));
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
    _currentAlgorithmId.clear();
    switchAlgorithmResource();
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
    const bool supported = algorithmId == QLatin1String(kSiftLightGlueAlgorithmId) ||
        algorithmId == QLatin1String(kLoMaRAlgorithmId);
    _matchingResourceEdit->setEnabled(supported);
    _matchingResourceBrowseButton->setEnabled(supported);
    _downloadModelButton->setEnabled(supported);
    _lomaRKeypointBudgetCombo->setEnabled(
        algorithmId == QLatin1String(kLoMaRAlgorithmId));
    if (supported)
    {
        refreshMatchingResourceStatus();
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
    _matchingResourceEdit->setText(
        _currentAlgorithmId == QLatin1String(kLoMaRAlgorithmId)
            ? _lomaRPackagePath
            : _lightGlueEnginePath);
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
        const auto resolved = xjw::matchphotos::resolveLoMaRTensorRtPackage(options, 40000);
        resolvedPath = resolved.manifestPath;
        detail = resolved.isValid()
            ? QStringLiteral("  [K=%1, D=%2, %3x%4]")
                  .arg(resolved.keypointCount)
                  .arg(resolved.descriptorDimension)
                  .arg(resolved.inputWidth)
                  .arg(resolved.inputHeight)
            : resolved.errorMessage;
    }
    else
    {
        options.lightGlueTensorRtEnginePath = _matchingResourceEdit->text().trimmed();
        const auto resolved = xjw::matchphotos::resolveLightGlueTensorRtEngine(options, 4096);
        resolvedPath = resolved.path;
        detail = resolved.bucketKeypoints > 0
            ? QStringLiteral("  [K=%1]").arg(resolved.bucketKeypoints)
            : QString();
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
