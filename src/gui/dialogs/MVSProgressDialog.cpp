// =============================================================================
// 文件: MVSProgressDialog.cpp
// 模块: GUI / Dialogs
// =============================================================================
#include "MVSProgressDialog.h"

#include "ui_MVSProgressDialog.h"

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
    connect(&m_timer, &QTimer::timeout, this, &MVSProgressDialog::updateElapsed);
    m_timer.setInterval(1000);
    m_elapsed.start();
    m_timer.start();
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::setupUi()
{
    Ui::MVSProgressDialog ui;
    ui.setupUi(this);

    m_stageLabel = ui.m_stageLabel;
    m_progressBar = ui.m_progressBar;
    m_elapsedLabel = ui.m_elapsedLabel;
    m_cancelBtn = ui.m_cancelBtn;

    connect(m_cancelBtn, &QPushButton::clicked,
            this, &MVSProgressDialog::onCancelClicked);
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::setTotalSteps(int total)
{
    m_totalSteps = total;
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onProgress(const QString &stage, float progress)
{
    if (m_finished) return;
    m_stageLabel->setText(stage);
    const int pct = static_cast<int>(std::max(0.f, std::min(1.f, progress)) * 100.f);
    m_progressBar->setValue(pct);
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onError(const QString &error)
{
    if (m_finished) return;
    m_stageLabel->setText("<font color='red'>错误: " + error + "</font>");
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onFinished(bool success)
{
    m_finished = true;
    m_timer.stop();

    if (success) {
        m_progressBar->setValue(100);
        m_stageLabel->setText("深度图生成完成！");
        m_cancelBtn->setText("关闭");
    } else {
        m_stageLabel->setText(m_stageLabel->text() + "\n（任务已终止）");
        m_cancelBtn->setText("关闭");
    }
    // 完成后取消按钮改为普通关闭
    disconnect(m_cancelBtn, &QPushButton::clicked,
               this, &MVSProgressDialog::onCancelClicked);
    connect(m_cancelBtn, &QPushButton::clicked,
            this, &MVSProgressDialog::accept);
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::updateElapsed()
{
    if (m_finished) return;
    const qint64 secs = m_elapsed.elapsed() / 1000;
    if (secs < 60)
        m_elapsedLabel->setText(QString("已用时: %1 秒").arg(secs));
    else
        m_elapsedLabel->setText(QString("已用时: %1 分 %2 秒")
                                     .arg(secs / 60).arg(secs % 60));
}

// ─────────────────────────────────────────────────────────────────────────────
void MVSProgressDialog::onCancelClicked()
{
    if (m_finished) return;
    m_stageLabel->setText("正在取消…");
    m_cancelBtn->setEnabled(false);
    emit cancelled();
}

} // namespace xjw
