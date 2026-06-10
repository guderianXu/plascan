#include "CreateDemDialog.h"
#include "ui_CreateDemDialog.h"
#include "ProjectManager.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
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
        return;
    }

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);

    // ── 模式切换（两个扁平按钮，像 tab） ──────────────────────────
    auto *modeRow = new QHBoxLayout();
    m_autoModeBtn  = new QPushButton(tr("自动模式"), this);
    m_manualModeBtn = new QPushButton(tr("手动模式"), this);
    m_autoModeBtn->setCheckable(true);
    m_manualModeBtn->setCheckable(true);
    m_autoModeBtn->setChecked(true);
    m_autoModeBtn->setFlat(true);
    m_manualModeBtn->setFlat(true);
    m_autoModeBtn->setStyleSheet(QStringLiteral("QPushButton:checked { font-weight: bold; border-bottom: 2px solid palette(highlight); }"));
    m_manualModeBtn->setStyleSheet(m_autoModeBtn->styleSheet());
    auto *modeGroup = new QButtonGroup(this);
    modeGroup->addButton(m_autoModeBtn);
    modeGroup->addButton(m_manualModeBtn);
    modeGroup->setExclusive(true);
    modeRow->addWidget(m_autoModeBtn);
    modeRow->addWidget(m_manualModeBtn);
    modeRow->addStretch(1);
    mainLayout->addLayout(modeRow);

    // ── 模式面板 ────────────────────────────────────────────────
    m_modeStack = new QStackedWidget(this);

    // 自动模式页
    auto *autoPage = new QWidget(this);
    {
        auto *layout = new QVBoxLayout(autoPage);
        layout->setSpacing(6);

        auto *hint = new QLabel(tr("选择 2 张立体影像，其余全自动完成。"), autoPage);
        hint->setWordWrap(true);
        layout->addWidget(hint);

        m_imageList = new QListWidget(autoPage);
        m_imageList->setSelectionMode(QAbstractItemView::ExtendedSelection);
        m_imageList->setMaximumHeight(110);
        layout->addWidget(m_imageList);

        auto *btnRow = new QHBoxLayout();
        auto *addBtn   = new QPushButton(tr("添加影像..."), autoPage);
        auto *clearBtn = new QPushButton(tr("清空"), autoPage);
        btnRow->addWidget(addBtn);
        btnRow->addWidget(clearBtn);
        btnRow->addStretch(1);
        layout->addLayout(btnRow);

        m_camStatusLabel = new QLabel(autoPage);
        m_camStatusLabel->setWordWrap(true);
        layout->addWidget(m_camStatusLabel);

        connect(addBtn,   &QPushButton::clicked, this, &CreateDemDialog::onBrowseImages);
        connect(clearBtn, &QPushButton::clicked, m_imageList, &QListWidget::clear);
        connect(clearBtn, &QPushButton::clicked, this, &CreateDemDialog::refreshRunButton);
        connect(m_imageList->model(), &QAbstractItemModel::rowsInserted,
                this, &CreateDemDialog::refreshRunButton);
        connect(m_imageList->model(), &QAbstractItemModel::rowsRemoved,
                this, &CreateDemDialog::refreshRunButton);
    }
    m_modeStack->addWidget(autoPage);

    // 手动模式页
    auto *manualPage = new QWidget(this);
    {
        auto *layout = new QVBoxLayout(manualPage);
        layout->setSpacing(6);

        auto *hint = new QLabel(tr("已有密集点云？直接指定文件路径，跳过特征提取/MVS 步骤。"), manualPage);
        hint->setWordWrap(true);
        layout->addWidget(hint);

        auto *row = new QHBoxLayout();
        m_denseEdit = new QLineEdit(manualPage);
        m_denseEdit->setPlaceholderText(tr("（留空则自动使用项目最新密集点云）"));
        auto *browseBtn = new QPushButton(tr("浏览..."), manualPage);
        browseBtn->setFixedWidth(64);
        row->addWidget(m_denseEdit, 1);
        row->addWidget(browseBtn);
        layout->addLayout(row);
        layout->addStretch(1);

        connect(browseBtn, &QPushButton::clicked, this, &CreateDemDialog::onBrowseDenseCloud);
    }
    m_modeStack->addWidget(manualPage);

    mainLayout->addWidget(m_modeStack);

    // ── 进度区域（运行时显示） ──────────────────────────────────
    m_stageLabel = new QLabel(this);
    m_stageLabel->setVisible(false);
    mainLayout->addWidget(m_stageLabel);

    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setVisible(false);
    mainLayout->addWidget(m_progressBar);

    // ── 按钮 ────────────────────────────────────────────────────
    auto *btnRow = new QHBoxLayout();
    btnRow->addStretch(1);
    m_runBtn   = new QPushButton(tr("运行"), this);
    m_closeBtn = new QPushButton(tr("关闭"), this);
    m_runBtn->setDefault(true);
    m_runBtn->setMinimumWidth(80);
    m_closeBtn->setMinimumWidth(80);
    btnRow->addWidget(m_runBtn);
    btnRow->addWidget(m_closeBtn);
    mainLayout->addLayout(btnRow);

    connect(m_autoModeBtn,  &QPushButton::toggled, this, &CreateDemDialog::onModeToggled);
    connect(m_runBtn,   &QPushButton::clicked, this, &CreateDemDialog::onRunClicked);
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
