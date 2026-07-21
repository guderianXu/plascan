#include "CreateDemDialog.h"
#include "ui_CreateDemDialog.h"
#include "ProjectManager.h"

#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QListWidget>
#include <QStackedWidget>
#include <QProgressBar>
#include <QFileDialog>
#include <QJsonObject>
#include <QJsonArray>
#include <QFileInfo>
#include <QMessageBox>
#include <QButtonGroup>

CreateDemDialog::CreateDemDialog(ProjectManager *projectManager, QWidget *parent)
    : QDialog(parent)
    , _projectManager(projectManager)
{
    setWindowTitle(tr("生成 DEM"));
    setMinimumWidth(520);
    setupUi();
}

void CreateDemDialog::setupUi()
{
    Ui::CreateDemDialog ui;
    ui.setupUi(this);

    _autoModeBtn = ui.m_autoModeBtn;
    _manualModeBtn = ui.m_manualModeBtn;
    _modeStack = ui.m_modeStack;
    _imageList = ui.m_imageList;
    _camStatusLabel = ui.m_camStatusLabel;
    _denseEdit = ui.m_denseEdit;
    _stageLabel = ui.m_stageLabel;
    _progressBar = ui.m_progressBar;
    _runBtn = ui.m_runBtn;
    _closeBtn = ui.m_closeBtn;

    const QString modeStyle = QStringLiteral(
        "QPushButton:checked { font-weight: bold; border-bottom: 2px solid palette(highlight); }");
    _autoModeBtn->setStyleSheet(modeStyle);
    _manualModeBtn->setStyleSheet(modeStyle);

    auto *modeGroup = new QButtonGroup(this);
    modeGroup->addButton(_autoModeBtn);
    modeGroup->addButton(_manualModeBtn);
    modeGroup->setExclusive(true);

    connect(ui.addImagesBtn, &QPushButton::clicked, this, &CreateDemDialog::onBrowseImages);
    connect(ui.clearImagesBtn, &QPushButton::clicked, _imageList, &QListWidget::clear);
    connect(ui.clearImagesBtn, &QPushButton::clicked, this, &CreateDemDialog::refreshRunButton);
    connect(_imageList->model(), &QAbstractItemModel::rowsInserted,
            this, &CreateDemDialog::refreshRunButton);
    connect(_imageList->model(), &QAbstractItemModel::rowsRemoved,
            this, &CreateDemDialog::refreshRunButton);

    connect(ui.browseDenseBtn, &QPushButton::clicked, this, &CreateDemDialog::onBrowseDenseCloud);
    connect(_autoModeBtn, &QPushButton::toggled, this, &CreateDemDialog::onModeToggled);
    connect(_runBtn, &QPushButton::clicked, this, &CreateDemDialog::onRunClicked);
    connect(_closeBtn, &QPushButton::clicked, this, &QDialog::accept);

    refreshRunButton();
}

void CreateDemDialog::onModeToggled(bool autoMode)
{
    _modeStack->setCurrentIndex(autoMode ? 0 : 1);
    refreshRunButton();
}

void CreateDemDialog::onBrowseImages()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, tr("选择立体影像"), QString(),
        tr("影像文件 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;所有文件 (*)"));
    for (const QString &f : files)
    {
        bool found = false;
        for (int i = 0; i < _imageList->count(); ++i)
        {
            if (_imageList->item(i)->text() == f)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            _imageList->addItem(f);
        }
    }

    // 检查相机状态
    if (_projectManager && _imageList->count() > 0)
    {
        QStringList images;
        for (int i = 0; i < _imageList->count(); ++i)
        {
            images << _imageList->item(i)->text();
        }
        bool hasCams = false;
        _projectManager->getCamerasForImages(images, &hasCams);
        _camStatusLabel->setText(hasCams
            ? tr("✓ 相机参数已从项目中加载")
            : tr("⚠ 未检测到相机参数，请先导入 .tsai 文件"));
        _camStatusLabel->setStyleSheet(hasCams
            ? QStringLiteral("color: green;")
            : QStringLiteral("color: orange;"));
    }
}

void CreateDemDialog::onBrowseDenseCloud()
{
    const QString f = QFileDialog::getOpenFileName(
        this, tr("选择密集点云文件"), QString(),
        tr("点云文件 (*.ply *.las *.laz);;所有文件 (*)"));
    if (!f.isEmpty())
    {
        _denseEdit->setText(f);
    }
}

void CreateDemDialog::onRunClicked()
{
    if (_autoModeBtn->isChecked())
    {
        QStringList images;
        for (int i = 0; i < _imageList->count(); ++i)
        {
            images << _imageList->item(i)->text();
        }

        if (images.size() < 2)
        {
            QMessageBox::warning(this, tr("生成 DEM"), tr("请至少选择 2 张立体影像。"));
            return;
        }

        bool hasCams = false;
        if (_projectManager)
        {
            _projectManager->getCamerasForImages(images, &hasCams);
        }
        if (!hasCams)
        {
            QMessageBox::warning(this, tr("生成 DEM"),
                tr("所选影像没有相机参数，请先通过[导入相机]菜单导入 .tsai 文件。"));
            return;
        }

        QJsonObject settings;
        settings[QStringLiteral("dem_resolution")] = 0.0;
        settings[QStringLiteral("dem_type")]       = QStringLiteral("float32");

        setRunning(true);
        emit requestRunFullPipeline(images, QString(), settings);
    }
    else
    {
        const QString dense = _denseEdit ? _denseEdit->text().trimmed() : QString();
        setRunning(true);
        emit requestRunFromDenseCloud(dense, QString(), 0.0, QStringLiteral("float32"));
    }
}

void CreateDemDialog::refreshRunButton()
{
    if (_running)
    {
        _runBtn->setEnabled(false);
        return;
    }
    if (_autoModeBtn->isChecked())
    {
        _runBtn->setEnabled(_imageList && _imageList->count() >= 2);
    }
    else
    {
        _runBtn->setEnabled(true);
    }
}

void CreateDemDialog::setRunning(bool running)
{
    _running = running;
    _stageLabel->setVisible(running);
    _progressBar->setVisible(running);
    _runBtn->setEnabled(!running);
    if (running)
    {
        _stageLabel->setText(tr("准备中..."));
        _progressBar->setValue(0);
    }
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

void CreateDemDialog::setAvailableImages(const QStringList &images)
{
    _availableImages = images;
    _imageList->clear();
    for (const QString &img : images)
    {
        _imageList->addItem(img);
    }
    refreshRunButton();
}

void CreateDemDialog::setDefaultOutput(const QString &)
{
    // 输出目录现在完全自动，不需要用户指定
}
