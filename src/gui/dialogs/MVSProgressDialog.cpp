// =============================================================================
// 文件: MVSProgressDialog.cpp
// 模块: GUI / Dialogs
// =============================================================================
#include "MVSProgressDialog.h"

#include "ui_MVSProgressDialog.h"

#include <algorithm>

#include <QApplication>

namespace xjw {

// ─────────────────────────────────────────────────────────────────────────────
MVSProgressDialog::MVSProgressDialog(QWidget *parent, Qt::WindowFlags f)
    : QDialog(parent, f)
{
    setWindowTitle("MVS 深度图生成");
    setWindowModality(Qt::ApplicationModal);
    // 不允许关闭按钮，防止意外取消
    setWindowFlags(windowFlags() & ~Qt::WindowCloseButtonHint);
    setMinimumWidth(460);

    setupUi();

    // 每秒更新已用时
    connect(&_timer, &QTimer::timeout, this, &MVSProgressDialog::updateElapsed);
    _timer.setInterval(1000);
    _elapsed.start();
    _timer.start();
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::setupUi()
{
    Ui::MVSProgressDialog ui;
    ui.setupUi(this);

    _stageLabel = ui.m_stageLabel;
    _progressBar = ui.m_progressBar;
    _elapsedLabel = ui.m_elapsedLabel;
    _cancelBtn = ui.m_cancelBtn;

    connect(_cancelBtn, &QPushButton::clicked,
            this, &MVSProgressDialog::onCancelClicked);
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::setTotalSteps(int total)
{
    _totalSteps = total;
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onProgress(const QString &stage, float progress)
{
    if (_finished)
    {
        return;
    }
    _stageLabel->setText(stage);
    const int pct = static_cast<int>(std::max(0.f, std::min(1.f, progress)) * 100.f);
    _progressBar->setValue(pct);
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onError(const QString &error)
{
    if (_finished)
    {
        return;
    }
    _stageLabel->setText("<font color='red'>错误: " + error + "</font>");
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onFinished(bool success)
{
    _finished = true;
    _timer.stop();

    if (success)
    {
        _progressBar->setValue(100);
        _stageLabel->setText("深度图生成完成！");
        _cancelBtn->setText("关闭");
    }
    else
    {
        _stageLabel->setText(_stageLabel->text() + "\n（任务已终止）");
        _cancelBtn->setText("关闭");
    }
    // 完成后取消按钮改为普通关闭
    disconnect(_cancelBtn, &QPushButton::clicked,
               this, &MVSProgressDialog::onCancelClicked);
    connect(_cancelBtn, &QPushButton::clicked,
            this, &MVSProgressDialog::accept);
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::updateElapsed()
{
    if (_finished)
    {
        return;
    }
    const qint64 secs = _elapsed.elapsed() / 1000;
    if (secs < 60)
    {
        _elapsedLabel->setText(QString("已用时: %1 秒").arg(secs));
    }
    else
    {
        _elapsedLabel->setText(QString("已用时: %1 分 %2 秒")
                                    .arg(secs / 60).arg(secs % 60));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onCancelClicked()
{
    if (_finished)
    {
        return;
    }
    _stageLabel->setText("正在取消…");
    _cancelBtn->setEnabled(false);
    emit cancelled();
}

} // namespace xjw
