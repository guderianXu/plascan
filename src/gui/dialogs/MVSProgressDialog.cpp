// =============================================================================
// 文件: MVSProgressDialog.cpp
// 模块: GUI / Dialogs
// =============================================================================
#include "MVSProgressDialog.h"

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
    // ── 阶段文字 ─────────────────────────────────────────────────────────
    m_stageLabel = new QLabel("正在初始化…", this);
    m_stageLabel->setWordWrap(true);
    m_stageLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);

    // ── 进度条 ────────────────────────────────────────────────────────────
    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_progressBar->setTextVisible(true);
    m_progressBar->setMinimumHeight(22);

    // ── 已用时 ────────────────────────────────────────────────────────────
    m_elapsedLabel = new QLabel("已用时: 0 秒", this);
    m_elapsedLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    // ── 底部行（已用时 + 取消按钮） ───────────────────────────────────────
    m_cancelBtn = new QPushButton("取消", this);
    m_cancelBtn->setFixedWidth(80);
    connect(m_cancelBtn, &QPushButton::clicked,
            this, &MVSProgressDialog::onCancelClicked);

    auto *bottomRow = new QHBoxLayout();
    bottomRow->addWidget(m_elapsedLabel, 1);
    bottomRow->addWidget(m_cancelBtn);

    // ── 主布局 ────────────────────────────────────────────────────────────
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);
    mainLayout->setContentsMargins(16, 16, 16, 14);
    mainLayout->addWidget(m_stageLabel);
    mainLayout->addWidget(m_progressBar);
    mainLayout->addLayout(bottomRow);

    setLayout(mainLayout);
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
