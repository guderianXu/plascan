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
    , m_projectManager(projectManager)
{
    setWindowTitle(tr("生成 DEM"));
    setMinimumWidth(520);
    setupUi();
}

void CreateDemDialog::setupUi()
{
    Ui::CreateDemDialog ui;
    ui.setupUi(this);

    m_autoModeBtn = ui.m_autoModeBtn;
    m_manualModeBtn = ui.m_manualModeBtn;
    m_modeStack = ui.m_modeStack;
    m_imageList = ui.m_imageList;
    m_camStatusLabel = ui.m_camStatusLabel;
    m_denseEdit = ui.m_denseEdit;
    m_stageLabel = ui.m_stageLabel;
    m_progressBar = ui.m_progressBar;
    m_runBtn = ui.m_runBtn;
    m_closeBtn = ui.m_closeBtn;

    const QString modeStyle = QStringLiteral(
        "QPushButton:checked { font-weight: bold; border-bottom: 2px solid palette(highlight); }");
    m_autoModeBtn->setStyleSheet(modeStyle);
    m_manualModeBtn->setStyleSheet(modeStyle);

    auto *modeGroup = new QButtonGroup(this);
    modeGroup->addButton(m_autoModeBtn);
    modeGroup->addButton(m_manualModeBtn);
    modeGroup->setExclusive(true);

    connect(ui.addImagesBtn, &QPushButton::clicked, this, &CreateDemDialog::onBrowseImages);
    connect(ui.clearImagesBtn, &QPushButton::clicked, m_imageList, &QListWidget::clear);
    connect(ui.clearImagesBtn, &QPushButton::clicked, this, &CreateDemDialog::refreshRunButton);
    connect(m_imageList->model(), &QAbstractItemModel::rowsInserted,
            this, &CreateDemDialog::refreshRunButton);
    connect(m_imageList->model(), &QAbstractItemModel::rowsRemoved,
            this, &CreateDemDialog::refreshRunButton);

    connect(ui.browseDenseBtn, &QPushButton::clicked, this, &CreateDemDialog::onBrowseDenseCloud);
    connect(m_autoModeBtn, &QPushButton::toggled, this, &CreateDemDialog::onModeToggled);
    connect(m_runBtn, &QPushButton::clicked, this, &CreateDemDialog::onRunClicked);
    connect(m_closeBtn, &QPushButton::clicked, this, &QDialog::accept);

    refreshRunButton();
}

void CreateDemDialog::onModeToggled(bool autoMode)
{
    m_modeStack->setCurrentIndex(autoMode ? 0 : 1);
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
        for (int i = 0; i < m_imageList->count(); ++i)
            if (m_imageList->item(i)->text() == f) { found = true; break; }
        if (!found)
            m_imageList->addItem(f);
    }

    // 检查相机状态
    if (m_projectManager && m_imageList->count() > 0)
    {
        QStringList images;
        for (int i = 0; i < m_imageList->count(); ++i)
            images << m_imageList->item(i)->text();
        bool hasCams = false;
        m_projectManager->getCamerasForImages(images, &hasCams);
        m_camStatusLabel->setText(hasCams
            ? tr("✓ 相机参数已从项目中加载")
            : tr("⚠ 未检测到相机参数，请先导入 .tsai 文件"));
        m_camStatusLabel->setStyleSheet(hasCams
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
        m_denseEdit->setText(f);
}

void CreateDemDialog::onRunClicked()
{
    if (m_autoModeBtn->isChecked())
    {
        QStringList images;
        for (int i = 0; i < m_imageList->count(); ++i)
            images << m_imageList->item(i)->text();

        if (images.size() < 2)
        {
            QMessageBox::warning(this, tr("生成 DEM"), tr("请至少选择 2 张立体影像。"));
            return;
        }

        bool hasCams = false;
        if (m_projectManager)
            m_projectManager->getCamerasForImages(images, &hasCams);
        if (!hasCams)
        {
            QMessageBox::warning(this, tr("生成 DEM"),
                tr("所选影像没有相机参数，请先通过[导入相机]菜单导入 .tsai 文件。"));
            return;
        }

        QJsonObject settings;
        settings[QStringLiteral("dem_resolution")] = 0.0;
        settings[QStringLiteral("dem_type")]       = QStringLiteral("float32");
        settings[QStringLiteral("matcher")]        = QStringLiteral("disk_lightglue");

        setRunning(true);
        emit requestRunFullPipeline(images, QString(), settings);
    }
    else
    {
        const QString dense = m_denseEdit ? m_denseEdit->text().trimmed() : QString();
        setRunning(true);
        emit requestRunFromDenseCloud(dense, QString(), 0.0, QStringLiteral("float32"));
    }
}

void CreateDemDialog::refreshRunButton()
{
    if (m_running)
    {
        m_runBtn->setEnabled(false);
        return;
    }
    if (m_autoModeBtn->isChecked())
        m_runBtn->setEnabled(m_imageList && m_imageList->count() >= 2);
    else
        m_runBtn->setEnabled(true);
}

void CreateDemDialog::setRunning(bool running)
{
    m_running = running;
    m_stageLabel->setVisible(running);
    m_progressBar->setVisible(running);
    m_runBtn->setEnabled(!running);
    if (running)
    {
        m_stageLabel->setText(tr("准备中..."));
        m_progressBar->setValue(0);
    }
}

void CreateDemDialog::onPipelineProgress(const QString &stage, int percent)
{
    m_stageLabel->setText(stage);
    m_progressBar->setValue(percent);
}

void CreateDemDialog::onPipelineFinished(bool success, const QString &message)
{
    setRunning(false);
    if (success)
    {
        m_stageLabel->setText(tr("完成"));
        m_stageLabel->setVisible(true);
        m_progressBar->setValue(100);
        m_progressBar->setVisible(true);
    }
    else
    {
        m_stageLabel->setText(tr("失败：%1").arg(message));
        m_stageLabel->setVisible(true);
    }
    refreshRunButton();
}

void CreateDemDialog::setAvailableImages(const QStringList &images)
{
    m_availableImages = images;
    m_imageList->clear();
    for (const QString &img : images)
        m_imageList->addItem(img);
    refreshRunButton();
}

void CreateDemDialog::setDefaultOutput(const QString &)
{
    // 输出目录现在完全自动，不需要用户指定
}
