#include "reconstruction/MapProjectDialog.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QSet>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTimer>
#include <QToolButton>
#include <QWidget>

#include <algorithm>

namespace
{

QString normalizedImagePath(const QString &path)
{
    QString normalized = QDir::cleanPath(QDir::fromNativeSeparators(path));
#ifdef Q_OS_WIN
    normalized = normalized.toCaseFolded();
#endif
    return normalized;
}

} // namespace

MapProjectDialog::MapProjectDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    connectUi();

    _estimateTimer = new QTimer(this);
    _estimateTimer->setSingleShot(true);
    _estimateTimer->setInterval(250);
    connect(_estimateTimer, &QTimer::timeout, this, [this]()
    {
        runEstimate(false);
    });

    updateControlAvailability();
    updateImageSummary();
    updateLocalEstimateSummary();
}

void MapProjectDialog::connectUi()
{
    connect(_demBrowseButton, &QPushButton::clicked, this, &MapProjectDialog::onChooseDem);
    connect(_outputBrowseButton, &QPushButton::clicked, this, &MapProjectDialog::onChooseOutput);
    connect(_runButton, &QPushButton::clicked, this, &MapProjectDialog::onRun);
    connect(_cancelButton, &QPushButton::clicked, this, &MapProjectDialog::onCancelRequested);
    connect(_estimateButton, &QPushButton::clicked, this, &MapProjectDialog::estimateNow);
    connect(_restorePixelSizeButton, &QPushButton::clicked, this, &MapProjectDialog::restoreDemPixelSize);
    connect(_resetBoundsButton, &QPushButton::clicked, this, &MapProjectDialog::resetBoundsToDem);
    connect(_imageToggleButton, &QToolButton::toggled, this, &MapProjectDialog::onImageListToggled);
    connect(_imageList, &QListWidget::itemChanged, this, &MapProjectDialog::onImageSelectionChanged);

    connect(_demEdit, &QLineEdit::textChanged, this, &MapProjectDialog::onDemPathChanged);
    connect(_outputEdit, &QLineEdit::textChanged, this, &MapProjectDialog::onSettingsModified);
    connect(_blendCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MapProjectDialog::onSettingsModified);
    for (QCheckBox *checkBox : {
             _fillHolesCheck, _ghostFilterCheck, _colorCorrectionCheck,
             _sharpnessWeightingCheck, _useProjectMasksCheck})
    {
        connect(checkBox, &QCheckBox::toggled, this, &MapProjectDialog::onSettingsModified);
    }

    connect(_pixelSizeRadio, &QRadioButton::toggled, this, &MapProjectDialog::onResolutionModeChanged);
    connect(_maximumDimensionRadio, &QRadioButton::toggled,
            this, &MapProjectDialog::onResolutionModeChanged);
    connect(_pixelSizeXSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MapProjectDialog::onPixelSizeEdited);
    connect(_pixelSizeYSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MapProjectDialog::onPixelSizeEdited);
    connect(_maximumDimensionSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MapProjectDialog::onSettingsModified);
    connect(_boundsEnabledCheck, &QCheckBox::toggled, this, &MapProjectDialog::onBoundsEnabledChanged);
    for (QDoubleSpinBox *spinBox : {_minXSpin, _maxXSpin, _minYSpin, _maxYSpin})
    {
        connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, &MapProjectDialog::onBoundsEdited);
    }
}

void MapProjectDialog::setAvailableImages(const QStringList &images)
{
    const QSignalBlocker blocker(_imageList);
    _imageList->clear();
    for (const QString &path : images)
    {
        auto *item = new QListWidgetItem(path, _imageList);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
        item->setToolTip(QDir::toNativeSeparators(path));
    }
    applyPendingImageSelection();
    applyImageReadiness();
    updateImageSummary();
}

void MapProjectDialog::setProjectRoot(const QString &projectRoot)
{
    _projectRoot = QDir::cleanPath(projectRoot);
    if (_outputEdit->text().trimmed().isEmpty() && !_projectRoot.isEmpty())
    {
        _outputEdit->setText(
            QDir(_projectRoot).filePath(QStringLiteral("assets/ortho/relative_dom.tif")));
    }
}

void MapProjectDialog::setDefaultDemPath(const QString &demPath)
{
    const QString cleanPath = demPath.trimmed();
    if (_demEdit->text().trimmed().isEmpty() && !cleanPath.isEmpty())
    {
        _demEdit->setText(cleanPath);
    }
}

void MapProjectDialog::setImageReadiness(const QStringList &cameraReadyImages, int maskCount)
{
    _cameraReadyImages = cameraReadyImages;
    _cameraReadyCount = cameraReadyImages.size();
    _maskCount = std::max(0, maskCount);
    _imageReadinessSet = true;

    applyImageReadiness();
    _imageReadinessLabel->setText(
        tr("相机参数就绪 %1 / %2 张；项目蒙版 %3 张")
            .arg(_cameraReadyCount)
            .arg(_imageList->count())
            .arg(_maskCount));
    _useProjectMasksCheck->setEnabled(!_running && _maskCount > 0);
    const QSignalBlocker blocker(_useProjectMasksCheck);
    if (_maskCount > 0)
    {
        _useProjectMasksCheck->setToolTip(tr("投影时排除项目蒙版标记的无效像素。"));
        _useProjectMasksCheck->setChecked(_requestedUseProjectMasks);
    }
    else
    {
        _useProjectMasksCheck->setChecked(false);
        _useProjectMasksCheck->setToolTip(tr("当前项目没有可用蒙版，因此不能启用此选项。"));
    }
    updateImageSummary();
}

void MapProjectDialog::applyImageReadiness()
{
    if (!_imageReadinessSet || !_imageList)
    {
        return;
    }

    QSet<QString> readyImages;
    for (const QString &path : _cameraReadyImages)
    {
        readyImages.insert(normalizedImagePath(path));
    }

    const QSignalBlocker blocker(_imageList);
    for (int index = 0; index < _imageList->count(); ++index)
    {
        QListWidgetItem *item = _imageList->item(index);
        if (!item)
        {
            continue;
        }
        const bool ready = readyImages.contains(normalizedImagePath(item->text()));
        item->setData(Qt::UserRole, ready);
        if (ready)
        {
            item->setFlags(item->flags() | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
        }
        else
        {
            item->setCheckState(Qt::Unchecked);
            item->setFlags(item->flags() & ~Qt::ItemIsEnabled & ~Qt::ItemIsUserCheckable);
            item->setToolTip(tr("%1\n缺少有效相机参数，不能用于正射投影。")
                                 .arg(QDir::toNativeSeparators(item->text())));
        }
    }
}

void MapProjectDialog::onChooseDem()
{
    const QString startPath = _demEdit->text().trimmed().isEmpty()
        ? _projectRoot
        : _demEdit->text().trimmed();
    const QString path = QFileDialog::getOpenFileName(
        this,
        tr("选择 DEM"),
        startPath,
        tr("DEM 栅格 (*.tif *.tiff *.img *.vrt);;所有文件 (*)"));
    if (!path.isEmpty())
    {
        _demEdit->setText(QDir::cleanPath(path));
    }
}

void MapProjectDialog::onChooseOutput()
{
    QString path = QFileDialog::getSaveFileName(
        this,
        tr("选择正射影像输出路径"),
        _outputEdit->text(),
        tr("GeoTIFF (*.tif *.tiff);;PNG (*.png)"));
    if (path.isEmpty())
    {
        return;
    }
    if (QFileInfo(path).suffix().isEmpty())
    {
        path += QStringLiteral(".tif");
    }
    _outputEdit->setText(QDir::cleanPath(path));
}

void MapProjectDialog::onRun()
{
    if (_running)
    {
        return;
    }

    if (_estimateTimer)
    {
        _estimateTimer->stop();
    }
    if (!runEstimate(true))
    {
        return;
    }

    const QJsonObject settings = currentSettings();
    QString errorMessage;
    if (!validateSettings(settings, &errorMessage))
    {
        QMessageBox::warning(this, tr("参数错误"), errorMessage);
        return;
    }
    const QString outputPath =
        settings.value(QStringLiteral("output_path")).toString();
    if (QFileInfo::exists(outputPath)
        && QMessageBox::question(
               this,
               tr("覆盖已有正射影像"),
               tr("输出文件已存在：\n%1\n\n是否覆盖？")
                   .arg(QDir::toNativeSeparators(outputPath)),
               QMessageBox::Yes | QMessageBox::No,
               QMessageBox::No)
            != QMessageBox::Yes)
    {
        return;
    }

    _hasRunFinished = false;
    _cancelRequested = false;
    setRunning(true);
    emit settingsChanged(settings);
    emit requestRunMapProject(settings);
}

void MapProjectDialog::onCancelRequested()
{
    if (!_running)
    {
        QDialog::reject();
        return;
    }
    if (_cancelRequested)
    {
        return;
    }

    _cancelRequested = true;
    _cancelButton->setText(tr("正在取消..."));
    _cancelButton->setEnabled(false);
    _stageLabel->setText(tr("正在请求取消"));
    _statusLabel->setText(tr("当前瓦片或处理步骤结束后将安全停止，请稍候。"));
    emit requestCancelMapProject();
}

void MapProjectDialog::onPipelineStarted()
{
    _hasRunFinished = false;
    _cancelRequested = false;
    setRunning(true);
    _runButton->setText(tr("生成"));
    _stageLabel->setText(tr("正在准备正射影像"));
    _statusLabel->setText(tr("正在检查 DEM、相机参数和输出范围。"));
    _progressBar->setValue(0);
}

void MapProjectDialog::onPipelineProgress(const QString &stage, int percent)
{
    if (!_running)
    {
        setRunning(true);
    }
    _stageLabel->setText(stage.trimmed().isEmpty() ? tr("正在生成正射影像") : stage);
    _progressBar->setValue(std::clamp(percent, 0, 100));
}

void MapProjectDialog::onPipelineFinished(bool success, const QString &message)
{
    const bool wasCancelled = _cancelRequested;
    _hasRunFinished = true;
    setRunning(false);
    _stageLabel->setText(success ? tr("生成完成") : (wasCancelled ? tr("任务已取消") : tr("生成失败")));
    _statusLabel->setText(message.trimmed().isEmpty()
                             ? (success ? tr("正射影像已生成。") : tr("正射影像生成未完成。"))
                             : message);
    _runButton->setText(success ? tr("再次生成") : tr("重试"));
    if (success)
    {
        _progressBar->setValue(100);
    }
    _cancelRequested = false;
}

void MapProjectDialog::reject()
{
    if (_running)
    {
        onCancelRequested();
        return;
    }
    QDialog::reject();
}

void MapProjectDialog::closeEvent(QCloseEvent *event)
{
    if (_running)
    {
        event->ignore();
        onCancelRequested();
        return;
    }
    QDialog::closeEvent(event);
}

void MapProjectDialog::setRunning(bool running)
{
    _running = running;
    for (QWidget *widget : {
             static_cast<QWidget *>(_projectionGroup),
             static_cast<QWidget *>(_parametersGroup),
             static_cast<QWidget *>(_regionGroup),
             static_cast<QWidget *>(_outputGroup)})
    {
        widget->setEnabled(!running);
    }

    _runButton->setEnabled(!running);
    _cancelButton->setEnabled(true);
    _cancelButton->setText(running
                               ? tr("取消任务")
                               : (_hasRunFinished ? tr("关闭") : tr("取消")));
    if (running)
    {
        _progressBar->setValue(0);
        _stageLabel->setText(tr("正在启动"));
        _statusLabel->setText(tr("参数已锁定，正在创建正射影像任务。"));
    }
    else
    {
        updateControlAvailability();
    }
}
