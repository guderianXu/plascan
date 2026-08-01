/**
 * @file WorkflowSettingsDialog.cpp
 * @brief 多工作流程项目设置对话框实现。
 */

#include "application/WorkflowSettingsDialog.h"

#include "ImageMatchingRegistry.h"
#include "MatchPhotosOptions.h"
#include "MatchPhotosRuntime.h"

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
    settings[QStringLiteral("workflow_settings_version")] = 3;
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

    auto *enginePathRow = new QWidget(aerialGroup);
    auto *enginePathLayout = new QHBoxLayout(enginePathRow);
    enginePathLayout->setContentsMargins(0, 0, 0, 0);
    enginePathLayout->setSpacing(6);
    _lightGlueEngineEdit = new QLineEdit(enginePathRow);
    _lightGlueEngineEdit->setObjectName(QStringLiteral("aerialLightGlueEngineEdit"));
    _lightGlueEngineEdit->setPlaceholderText(QStringLiteral("自动选择本机引擎"));
    _lightGlueEngineEdit->setClearButtonEnabled(true);
    _lightGlueEngineBrowseButton = new QToolButton(enginePathRow);
    _lightGlueEngineBrowseButton->setIcon(style()->standardIcon(QStyle::SP_DialogOpenButton));
    _lightGlueEngineBrowseButton->setToolTip(QStringLiteral("选择 TensorRT 引擎"));
    _lightGlueEngineBrowseButton->setFixedSize(32, 30);
    enginePathLayout->addWidget(_lightGlueEngineEdit, 1);
    enginePathLayout->addWidget(_lightGlueEngineBrowseButton);
    aerialForm->addRow(QStringLiteral("TensorRT 引擎:"), enginePathRow);

    _lightGlueEngineStatusLabel = new QLabel(aerialGroup);
    _lightGlueEngineStatusLabel->setWordWrap(true);
    _lightGlueEngineStatusLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    aerialForm->addRow(QStringLiteral("当前生效:"), _lightGlueEngineStatusLabel);
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
            this, [this]() { refreshAlgorithmControls(); });
    connect(_lightGlueEngineEdit, &QLineEdit::textChanged,
            this, [this]() { refreshLightGlueEngineStatus(); });
    connect(_lightGlueEngineBrowseButton, &QToolButton::clicked, this, [this]()
    {
        const QString currentPath = _lightGlueEngineEdit->text().trimmed();
        const QString startPath = currentPath.isEmpty()
            ? QString()
            : QFileInfo(currentPath).absolutePath();
        const QString selected = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("选择 TensorRT LightGlue 引擎"),
            startPath,
            QStringLiteral("TensorRT engine (*.engine);;所有文件 (*)"));
        if (!selected.isEmpty())
        {
            _lightGlueEngineEdit->setText(QFileInfo(selected).absoluteFilePath());
        }
    });

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
    _lightGlueEngineEdit->setText(
        aerial.value(QStringLiteral("lightglue_tensorrt_engine")).toString());
    setCurrentWorkflow(_workflowCombo->currentIndex());
    refreshAlgorithmControls();
}

QJsonObject WorkflowSettingsDialog::collectSettings() const
{
    QJsonObject settings = normalizedSettings(_appliedSettings);
    settings[QStringLiteral("workflow_settings_version")] = 3;
    settings[QStringLiteral("selected_workflow")] =
        _workflowCombo->currentData().toString();

    QJsonObject workflows = settings.value(QStringLiteral("workflows")).toObject();
    QJsonObject aerial = workflows.value(QString::fromLatin1(kAerialWorkflowId)).toObject();
    const QString algorithmId = _matchingAlgorithmCombo->currentData().toString();
    aerial[QStringLiteral("algorithm_id")] = algorithmId.isEmpty()
        ? QString::fromLatin1(kSiftLightGlueAlgorithmId)
        : algorithmId;
    aerial[QStringLiteral("lightglue_tensorrt_engine")] =
        _lightGlueEngineEdit->text().trimmed();
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
    const bool usesLightGlue = _matchingAlgorithmCombo->currentData().toString() ==
        QString::fromLatin1(kSiftLightGlueAlgorithmId);
    _lightGlueEngineEdit->setEnabled(usesLightGlue);
    _lightGlueEngineBrowseButton->setEnabled(usesLightGlue);
    if (usesLightGlue)
    {
        refreshLightGlueEngineStatus();
    }
    else
    {
        _lightGlueEngineStatusLabel->setText(QStringLiteral("不适用"));
    }
}

void WorkflowSettingsDialog::refreshLightGlueEngineStatus()
{
    if (!_lightGlueEngineEdit || !_lightGlueEngineStatusLabel ||
        _matchingAlgorithmCombo->currentData().toString() !=
            QString::fromLatin1(kSiftLightGlueAlgorithmId))
    {
        return;
    }

    xjw::matchphotos::MatchPhotosOptions options;
    options.lightGlueTensorRtEnginePath = _lightGlueEngineEdit->text().trimmed();
    const auto resolved = xjw::matchphotos::resolveLightGlueTensorRtEngine(options, 4096);

    QPalette palette = _lightGlueEngineStatusLabel->palette();
    if (!resolved.isValid())
    {
        palette.setColor(QPalette::WindowText, QColor(180, 45, 45));
        _lightGlueEngineStatusLabel->setPalette(palette);
        _lightGlueEngineStatusLabel->setText(
            options.lightGlueTensorRtEnginePath.isEmpty()
                ? QStringLiteral("未找到自动引擎")
                : QStringLiteral("指定引擎不可用"));
        return;
    }

    palette.setColor(QPalette::WindowText, QColor(35, 110, 70));
    _lightGlueEngineStatusLabel->setPalette(palette);
    const QString bucketLabel = resolved.bucketKeypoints > 0
        ? QStringLiteral("  [K=%1]").arg(resolved.bucketKeypoints)
        : QString();
    _lightGlueEngineStatusLabel->setText(
        QDir::toNativeSeparators(resolved.path) + bucketLabel);
}
