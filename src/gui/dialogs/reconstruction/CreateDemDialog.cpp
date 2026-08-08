#include "reconstruction/CreateDemDialog.h"
#include "ui_CreateDemDialog.h"

#include <QFileDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>

CreateDemDialog::CreateDemDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
}

void CreateDemDialog::setupUi()
{
    Ui::CreateDemDialog ui;
    ui.setupUi(this);

    _denseEdit = ui.m_denseEdit;
    _stageLabel = ui.m_stageLabel;
    _progressBar = ui.m_progressBar;
    _runBtn = ui.m_runBtn;
    _closeBtn = ui.m_closeBtn;

    connect(ui.browseDenseBtn, &QPushButton::clicked, this, &CreateDemDialog::onBrowseDenseCloud);
    connect(_denseEdit, &QLineEdit::textChanged, this, &CreateDemDialog::refreshRunButton);
    connect(_runBtn, &QPushButton::clicked, this, &CreateDemDialog::onRunClicked);
    connect(_closeBtn, &QPushButton::clicked, this, &QDialog::accept);

    refreshRunButton();
}

void CreateDemDialog::onBrowseDenseCloud()
{
    const QString f = QFileDialog::getOpenFileName(
        this, tr("选择点云文件"), QString(),
        tr("点云文件 (*.ply *.las *.laz);;所有文件 (*)"));
    if (!f.isEmpty())
    {
        _denseEdit->setText(f);
    }
}

void CreateDemDialog::onRunClicked()
{
    xjw::gui::project::DemGenerationRequest request;
    request.sourcePointCloudPath = _denseEdit ? _denseEdit->text().trimmed() : QString();
    QString error_message;
    if (!request.validate(&error_message))
    {
        QMessageBox::warning(this, tr("生成 DEM"), error_message);
        return;
    }
    setRunning(true);
    emit requestRun(request);
}

void CreateDemDialog::refreshRunButton()
{
    if (_running)
    {
        _runBtn->setEnabled(false);
        return;
    }
    _runBtn->setEnabled(_denseEdit && !_denseEdit->text().trimmed().isEmpty());
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
