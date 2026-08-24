#include "reconstruction/CreateDemDialog.h"
#include "ui_CreateDemDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSizePolicy>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTimer>

CreateDemDialog::CreateDemDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
}

void CreateDemDialog::reject()
{
    if (_running && _runningCancelable)
    {
        if (!_cancelRequested)
        {
            _cancelRequested = true;
            _closeBtn->setEnabled(false);
            _stageLabel->setText(tr("正在取消..."));
            emit requestCancel();
        }
        return;
    }
    QDialog::reject();
}

void CreateDemDialog::setupUi()
{
    Ui::CreateDemDialog ui;
    ui.setupUi(this);

    _modeCombo = ui.m_modeCombo;
    _optionsStack = ui.m_optionsStack;
    _optionsStack->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    _hintLabel = ui.m_hintLabel;

    _denseEdit = ui.m_denseEdit;
    _browseDenseBtn = ui.browseDenseBtn;

    _stereoImageList = ui.m_stereoImageList;
    _stereoSelectionLabel = ui.m_stereoSelectionLabel;
    _selectAllImagesBtn = ui.m_selectAllImagesBtn;
    _clearImagesBtn = ui.m_clearImagesBtn;
    _rpcResolutionSpin = ui.m_rpcResolutionSpin;
    _rpcMaximumFeaturesSpin = ui.m_rpcMaximumFeaturesSpin;
    _rpcMaximumErrorSpin = ui.m_rpcMaximumErrorSpin;

    _surfaceEdit = ui.m_surfaceEdit;
    _browseSurfaceBtn = ui.browseSurfaceBtn;
    _surfaceUnitCombo = ui.m_surfaceUnitCombo;
    _targetNameEdit = ui.m_targetNameEdit;
    _bodyFixedFrameEdit = ui.m_bodyFixedFrameEdit;
    _angularResolutionSpin = ui.m_angularResolutionSpin;
    _referenceRadiusSpin = ui.m_referenceRadiusSpin;
    _automaticCenterCheck = ui.m_automaticCenterCheck;
    _centerXSpin = ui.m_centerXSpin;
    _centerYSpin = ui.m_centerYSpin;
    _centerZSpin = ui.m_centerZSpin;
    _centralMeridianSpin = ui.m_centralMeridianSpin;

    _stageLabel = ui.m_stageLabel;
    _progressBar = ui.m_progressBar;
    _runBtn = ui.m_runBtn;
    _closeBtn = ui.m_closeBtn;

    connect(_modeCombo, &QComboBox::currentIndexChanged, this,
            [this](int)
            {
                refreshModeUi();
            });
    connect(_browseDenseBtn, &QPushButton::clicked,
            this, &CreateDemDialog::onBrowseDenseCloud);
    connect(_selectAllImagesBtn, &QPushButton::clicked, this, &CreateDemDialog::selectAllStereoImages);
    connect(_clearImagesBtn, &QPushButton::clicked, this, &CreateDemDialog::clearStereoImages);
    connect(_stereoImageList, &QListWidget::itemChanged, this,
            [this](QListWidgetItem *) { refreshStereoSelection(); });
    connect(_browseSurfaceBtn, &QPushButton::clicked,
            this, &CreateDemDialog::onBrowseSurface);
    connect(_denseEdit, &QLineEdit::textChanged,
            this, &CreateDemDialog::refreshRunButton);
    connect(_surfaceEdit, &QLineEdit::textChanged,
            this, &CreateDemDialog::refreshRunButton);
    connect(_targetNameEdit, &QLineEdit::textChanged,
            this, &CreateDemDialog::refreshRunButton);
    connect(_bodyFixedFrameEdit, &QLineEdit::textChanged,
            this, &CreateDemDialog::refreshRunButton);
    connect(_automaticCenterCheck, &QCheckBox::toggled, this,
            [this](bool)
            {
                refreshModeUi();
            });
    connect(_runBtn, &QPushButton::clicked, this, &CreateDemDialog::onRunClicked);
    connect(_closeBtn, &QPushButton::clicked, this,
            [this]()
            {
                if (_running && _runningCancelable)
                {
                    reject();
                    return;
                }
                accept();
            });

    resize(900, 430);
    refreshModeUi();
}

bool CreateDemDialog::isSmallBodyGlobalMode() const
{
    return _modeCombo && _modeCombo->currentIndex() == 2;
}

bool CreateDemDialog::isImageStereoMode() const
{
    return _modeCombo && _modeCombo->currentIndex() == 1;
}

void CreateDemDialog::setAvailableImages(const QStringList &images)
{
    _stereoImageList->clear();
    for (const QString &path : images)
    {
        auto *item = new QListWidgetItem(QFileInfo(path).fileName(), _stereoImageList);
        item->setData(Qt::UserRole, path);
        item->setToolTip(path);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
    }
    refreshStereoSelection();
}

void CreateDemDialog::onBrowseDenseCloud()
{
    const QString file_path = QFileDialog::getOpenFileName(
        this, tr("选择点云文件"), QString(),
        tr("点云文件 (*.ply *.las *.laz);;所有文件 (*)"));
    if (!file_path.isEmpty())
    {
        _denseEdit->setText(file_path);
    }
}

QStringList CreateDemDialog::selectedStereoImages() const
{
    QStringList images;
    for (int index = 0; index < _stereoImageList->count(); ++index)
    {
        const auto *item = _stereoImageList->item(index);
        if (item->checkState() == Qt::Checked)
        {
            images.append(item->data(Qt::UserRole).toString());
        }
    }
    return images;
}

void CreateDemDialog::selectAllStereoImages()
{
    for (int index = 0; index < _stereoImageList->count(); ++index)
    {
        _stereoImageList->item(index)->setCheckState(Qt::Checked);
    }
}

void CreateDemDialog::clearStereoImages()
{
    for (int index = 0; index < _stereoImageList->count(); ++index)
    {
        _stereoImageList->item(index)->setCheckState(Qt::Unchecked);
    }
}

void CreateDemDialog::refreshStereoSelection()
{
    _stereoSelectionLabel->setText(tr("已选择 %1 / %2 张")
                                       .arg(selectedStereoImages().size())
                                       .arg(_stereoImageList->count()));
    refreshRunButton();
}

void CreateDemDialog::onBrowseSurface()
{
    const QString file_path = QFileDialog::getOpenFileName(
        this, tr("选择体固连三角网格"), QString(),
        tr("三角网格 (*.ply *.obj);;PLY 网格 (*.ply);;OBJ 网格 (*.obj);;所有文件 (*)"));
    if (!file_path.isEmpty())
    {
        _surfaceEdit->setText(file_path);
    }
}

void CreateDemDialog::onRunClicked()
{
    xjw::gui::project::DemGenerationRequest request;
    if (isImageStereoMode())
    {
        request.mode = xjw::gui::project::DemGenerationMode::ImageStereo;
        request.imageStereoOptions.sourceImages = selectedStereoImages();
        request.imageStereoOptions.gridResolutionMeters = _rpcResolutionSpin->value();
        request.imageStereoOptions.maximumFeatures = _rpcMaximumFeaturesSpin->value();
        request.imageStereoOptions.maximumReprojectionErrorPixels =
            _rpcMaximumErrorSpin->value();
    }
    else if (isSmallBodyGlobalMode())
    {
        request.mode = xjw::gui::project::DemGenerationMode::SmallBodyGlobal;
        request.sourceSurfacePath = _surfaceEdit->text().trimmed();
        request.smallBodyOptions.targetName = _targetNameEdit->text().trimmed();
        request.smallBodyOptions.bodyFixedFrame = _bodyFixedFrameEdit->text().trimmed();
        request.smallBodyOptions.surfaceCoordinateUnit =
            _surfaceUnitCombo->currentIndex() == 1
                ? QStringLiteral("km") : QStringLiteral("m");
        request.smallBodyOptions.angularResolutionDeg = _angularResolutionSpin->value();
        request.smallBodyOptions.referenceRadiusM = _referenceRadiusSpin->value();
        request.smallBodyOptions.automaticCenter = _automaticCenterCheck->isChecked();
        request.smallBodyOptions.bodyCenterX = _centerXSpin->value();
        request.smallBodyOptions.bodyCenterY = _centerYSpin->value();
        request.smallBodyOptions.bodyCenterZ = _centerZSpin->value();
        request.smallBodyOptions.centralMeridianDeg = _centralMeridianSpin->value();
        request.smallBodyOptions.writeReportPreview = true;
    }
    else
    {
        request.sourcePointCloudPath = _denseEdit->text().trimmed();
    }

    QString error_message;
    if (!request.validate(&error_message))
    {
        QMessageBox::warning(this, tr("生成 DEM"), error_message);
        return;
    }
    _runningCancelable = isSmallBodyGlobalMode() || isImageStereoMode();
    setRunning(true);
    emit requestRun(request);
}

void CreateDemDialog::refreshModeUi()
{
    const bool global_mode = isSmallBodyGlobalMode();
    const bool rpc_mode = isImageStereoMode();
    _optionsStack->setCurrentIndex(global_mode ? 2 : (rpc_mode ? 1 : 0));
    _hintLabel->setText(global_mode
        ? tr("从体固连闭合三角网格生成全球径向 DEM、参考半径高程 DEM、DOM 和质量报告。")
        : (rpc_mode
               ? tr("从当前项目勾选多张重叠影像，自动使用 RPC 或框幅式相机进行匹配与前方交会。")
               : tr("选择已有点云文件生成局部平面相对 DEM。")));
    _runBtn->setText(global_mode
        ? tr("生成全球 DEM + DOM")
        : (rpc_mode ? tr("生成影像立体 DEM") : tr("一键生成 DEM")));

    const bool manual_center = global_mode
        && !_automaticCenterCheck->isChecked()
        && !_running;
    _centerXSpin->setEnabled(manual_center);
    _centerYSpin->setEnabled(manual_center);
    _centerZSpin->setEnabled(manual_center);
    refreshRunButton();
    if (!_running)
    {
        QTimer::singleShot(0, this, [this, global_mode]()
        {
            resize(width(), global_mode ? 650 : 430);
        });
    }
}

void CreateDemDialog::refreshRunButton()
{
    if (_running)
    {
        _runBtn->setEnabled(false);
        return;
    }

    if (isSmallBodyGlobalMode())
    {
        _runBtn->setEnabled(
            !_surfaceEdit->text().trimmed().isEmpty()
            && !_targetNameEdit->text().trimmed().isEmpty()
            && !_bodyFixedFrameEdit->text().trimmed().isEmpty());
        return;
    }
    if (isImageStereoMode())
    {
        _runBtn->setEnabled(selectedStereoImages().size() >= 2);
        return;
    }
    _runBtn->setEnabled(!_denseEdit->text().trimmed().isEmpty());
}

void CreateDemDialog::setRunning(bool running)
{
    _running = running;
    _cancelRequested = false;
    _modeCombo->setEnabled(!running);
    _optionsStack->setEnabled(!running);
    _closeBtn->setEnabled(true);
    _closeBtn->setText(running && _runningCancelable ? tr("取消任务") : tr("关闭"));
    _stageLabel->setVisible(running);
    _progressBar->setVisible(running);
    if (running)
    {
        _stageLabel->setText(tr("准备中..."));
        _progressBar->setValue(0);
    }
    refreshModeUi();
}

void CreateDemDialog::onPipelineProgress(const QString &stage, int percent)
{
    _stageLabel->setText(stage);
    _progressBar->setValue(percent);
}

void CreateDemDialog::onPipelineFinished(bool success, const QString &message)
{
    setRunning(false);
    if (success)
    {
        _stageLabel->setText(tr("完成"));
        _stageLabel->setVisible(true);
        _progressBar->setValue(100);
        _progressBar->setVisible(true);
    }
    else
    {
        _stageLabel->setText(tr("失败：%1").arg(message));
        _stageLabel->setVisible(true);
    }
    refreshRunButton();
}
