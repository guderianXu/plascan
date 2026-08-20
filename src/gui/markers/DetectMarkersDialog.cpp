#include "DetectMarkersDialog.h"

#include "MarkerDetectionJobBuilder.h"
#include "MarkerWorkspaceController.h"
#include "project/ProjectSessionModel.h"

#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QThread>
#include <QVBoxLayout>

#include <algorithm>

namespace xjw::gui::markers
{

namespace
{

using control_points::MarkerTargetFamily;

bool isCircularCodedFamily(MarkerTargetFamily family)
{
    return family >= MarkerTargetFamily::Circular12Bit
        && family <= MarkerTargetFamily::Circular20Bit;
}

} // namespace

DetectMarkersDialog::DetectMarkersDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    populateTargetFamilies();
    connect(&_runner, &MarkerTaskRunner::progressChanged,
            this, &DetectMarkersDialog::handleProgress);
    connect(&_runner, &MarkerTaskRunner::finished,
            this, &DetectMarkersDialog::handleFinished);
}

bool DetectMarkersDialog::setContext(MarkerWorkspaceController *controller,
                                     ProjectData *projectData)
{
    _controller = controller;
    _projectData = projectData;
    const int image_count = projectData
        ? projectData->coreFilesMeta().value(QStringLiteral("images")).toArray().size()
        : 0;
    _progressBar->setRange(0, image_count);
    _progressBar->setValue(0);
    _startButton->setEnabled(controller && projectData && projectData->hasProject() && image_count > 0);
    if (image_count > 0)
    {
        _statusLabel->setText(QStringLiteral("待检测 %1 张影像").arg(image_count));
    }
    else
    {
        _statusLabel->setText(QStringLiteral("项目中没有可检测影像"));
    }
    updateFamilyAvailability();
    return _startButton->isEnabled();
}

void DetectMarkersDialog::closeEvent(QCloseEvent *event)
{
    if (_runner.isRunning())
    {
        _closeAfterCancel = true;
        cancelDetection();
        event->ignore();
        return;
    }
    QDialog::closeEvent(event);
}

void DetectMarkersDialog::setupUi()
{
    setWindowTitle(QStringLiteral("检测标靶"));
    setMinimumWidth(520);
    setModal(true);

    auto *root = new QVBoxLayout(this);
    auto *parameters = new QGroupBox(QStringLiteral("检测参数"), this);
    auto *form = new QFormLayout(parameters);

    _familyCombo = new QComboBox(parameters);
    _familyCombo->setObjectName(QStringLiteral("markerTargetFamilyCombo"));
    form->addRow(QStringLiteral("标靶类型:"), _familyCombo);

    _decisionMarginSpin = new QDoubleSpinBox(parameters);
    _decisionMarginSpin->setRange(0.0, 1000.0);
    _decisionMarginSpin->setDecimals(1);
    _decisionMarginSpin->setValue(20.0);
    _decisionMarginSpin->setSuffix(QStringLiteral(" margin"));
    form->addRow(QStringLiteral("最小判决裕量:"), _decisionMarginSpin);

    _maxHammingSpin = new QSpinBox(parameters);
    _maxHammingSpin->setRange(0, 3);
    _maxHammingSpin->setValue(1);
    form->addRow(QStringLiteral("最大纠错位数:"), _maxHammingSpin);

    _concurrentImagesSpin = new QSpinBox(parameters);
    _concurrentImagesSpin->setRange(1, std::max(1, QThread::idealThreadCount()));
    _concurrentImagesSpin->setValue(std::max(1, QThread::idealThreadCount()));
    form->addRow(QStringLiteral("并行影像数:"), _concurrentImagesSpin);
    root->addWidget(parameters);

    _progressBar = new QProgressBar(this);
    _progressBar->setObjectName(QStringLiteral("markerDetectionProgress"));
    _progressBar->setRange(0, 0);
    _progressBar->setValue(0);
    _progressBar->setFormat(QStringLiteral("%v / %m（%p%）"));
    root->addWidget(_progressBar);

    _statusLabel = new QLabel(QStringLiteral("请选择项目后开始检测"), this);
    _statusLabel->setObjectName(QStringLiteral("markerDetectionStatus"));
    _statusLabel->setWordWrap(true);
    root->addWidget(_statusLabel);

    auto *buttons = new QHBoxLayout();
    buttons->addStretch();
    _startButton = new QPushButton(QStringLiteral("开始检测"), this);
    _startButton->setObjectName(QStringLiteral("startMarkerDetectionButton"));
    _startButton->setDefault(true);
    _startButton->setEnabled(false);
    _cancelButton = new QPushButton(QStringLiteral("取消"), this);
    _cancelButton->setObjectName(QStringLiteral("cancelMarkerDetectionButton"));
    _cancelButton->setEnabled(false);
    _closeButton = new QPushButton(QStringLiteral("关闭"), this);
    buttons->addWidget(_startButton);
    buttons->addWidget(_cancelButton);
    buttons->addWidget(_closeButton);
    root->addLayout(buttons);

    connect(_familyCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int) { updateFamilyAvailability(); });
    connect(_startButton, &QPushButton::clicked, this, &DetectMarkersDialog::startDetection);
    connect(_cancelButton, &QPushButton::clicked, this, &DetectMarkersDialog::cancelDetection);
    connect(_closeButton, &QPushButton::clicked, this, &QDialog::close);
}

void DetectMarkersDialog::populateTargetFamilies()
{
    const QVector<MarkerTargetFamily> families = {
        MarkerTargetFamily::AprilTag16h5,
        MarkerTargetFamily::AprilTag25h9,
        MarkerTargetFamily::AprilTag36h10,
        MarkerTargetFamily::AprilTag36h11,
        MarkerTargetFamily::AprilTagCircle21h7,
        MarkerTargetFamily::AprilTagStandard41h12,
        MarkerTargetFamily::AprilTagStandard52h13,
        MarkerTargetFamily::Circular12Bit,
        MarkerTargetFamily::Circular14Bit,
        MarkerTargetFamily::Circular16Bit,
        MarkerTargetFamily::Circular20Bit,
        MarkerTargetFamily::NonCodedCircle,
        MarkerTargetFamily::NonCodedFourQuadrant,
    };
    for (const MarkerTargetFamily family : families)
    {
        _familyCombo->addItem(control_points::markerTargetFamilyName(family),
                              static_cast<int>(family));
        const int index = _familyCombo->count() - 1;
        if (isCircularCodedFamily(family))
        {
            _familyCombo->setItemData(
                index,
                QStringLiteral("需要先导入经许可导出的 Metashape 官方圆形标靶语料"),
                Qt::ToolTipRole);
            if (auto *model = qobject_cast<QStandardItemModel *>(_familyCombo->model()))
            {
                model->item(index)->setEnabled(false);
            }
        }
    }
    const int default_index = _familyCombo->findData(
        static_cast<int>(MarkerTargetFamily::AprilTag36h11));
    _familyCombo->setCurrentIndex(default_index);
}

void DetectMarkersDialog::updateFamilyAvailability()
{
    const bool has_context = _controller && _projectData && _projectData->hasProject()
        && _progressBar->maximum() > 0;
    const auto family = static_cast<MarkerTargetFamily>(_familyCombo->currentData().toInt());
    const bool supported = !isCircularCodedFamily(family);
    _startButton->setEnabled(!_runner.isRunning() && has_context && supported);
}

void DetectMarkersDialog::startDetection()
{
    if (!_controller || !_projectData || _runner.isRunning())
    {
        return;
    }

    MarkerDetectionJobBuildOptions options;
    options.baseRevision = _controller->markerRevision();
    options.targetFamilies = {
        static_cast<MarkerTargetFamily>(_familyCombo->currentData().toInt())
    };
    options.detectorOptions.minDecisionMargin = _decisionMarginSpin->value();
    options.detectorOptions.maxHamming = _maxHammingSpin->value();
    options.maxConcurrentImages = _concurrentImagesSpin->value();
    const MarkerDetectionJobBuildResult built = MarkerDetectionJobBuilder::build(*_projectData, options);
    if (!built.ok)
    {
        _statusLabel->setText(built.errors.join(QLatin1Char('\n')));
        return;
    }

    _preflightWarnings = built.errors;
    _progressBar->setRange(0, built.job.images.size());
    _progressBar->setValue(0);
    setRunning(true);
    _statusLabel->setText(QStringLiteral("正在检测 0 / %1 张影像...").arg(built.job.images.size()));
    if (!_runner.start(built.job))
    {
        setRunning(false);
        _statusLabel->setText(QStringLiteral("无法启动标靶检测任务"));
    }
}

void DetectMarkersDialog::cancelDetection()
{
    if (_runner.isRunning())
    {
        _runner.cancel();
        _cancelButton->setEnabled(false);
        _statusLabel->setText(QStringLiteral("正在取消，等待已启动的影像检测结束..."));
    }
}

void DetectMarkersDialog::handleProgress(const MarkerDetectionProgress &progress)
{
    _progressBar->setRange(0, progress.imageCount);
    _progressBar->setValue(progress.imagesCompleted);
    _statusLabel->setText(
        QStringLiteral("已访问 %1 / %2 张影像，发现 %3 个候选；当前：%4")
            .arg(progress.imagesCompleted)
            .arg(progress.imageCount)
            .arg(progress.candidatesDetected)
            .arg(QFileInfo(progress.currentImage).fileName()));
}

void DetectMarkersDialog::handleFinished(const MarkerDetectionTaskResult &result)
{
    setRunning(false);
    if (result.cancelled)
    {
        _statusLabel->setText(QStringLiteral("检测已取消，未修改标记点"));
    }
    else if (!_controller)
    {
        _statusLabel->setText(QStringLiteral("项目已关闭，检测结果未写入"));
    }
    else
    {
        control_points::DetectionIntegrationResult integration;
        QString error;
        if (!_controller->applyDetectionTaskResult(result, &integration, &error))
        {
            _statusLabel->setText(error);
        }
        else
        {
            const int error_count = _preflightWarnings.size() + result.errors.size();
            _statusLabel->setText(
                QStringLiteral("检测完成：候选 %1，写入 %2，新增标记 %3，待复核 %4，冲突 %5，错误 %6")
                    .arg(result.observations.size())
                    .arg(integration.appliedDetections)
                    .arg(integration.createdMarkers)
                    .arg(integration.pendingReview.size())
                    .arg(integration.conflicts.size())
                    .arg(error_count));
        }
    }
    _preflightWarnings.clear();
    if (_closeAfterCancel)
    {
        _closeAfterCancel = false;
        accept();
    }
}

void DetectMarkersDialog::setRunning(bool running)
{
    _familyCombo->setEnabled(!running);
    _decisionMarginSpin->setEnabled(!running);
    _maxHammingSpin->setEnabled(!running);
    _concurrentImagesSpin->setEnabled(!running);
    _startButton->setEnabled(false);
    _cancelButton->setEnabled(running);
    _closeButton->setEnabled(!running);
    if (!running)
    {
        updateFamilyAvailability();
    }
}

} // namespace xjw::gui::markers
