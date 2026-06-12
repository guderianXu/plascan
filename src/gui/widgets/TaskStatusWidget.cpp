#include "TaskStatusWidget.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QToolButton>

TaskStatusWidget::TaskStatusWidget(QWidget *parent)
    : QWidget(parent)
    , m_statusLabel(new QLabel(this))
    , m_progressBar(new QProgressBar(this))
    , m_cancelButton(new QToolButton(this))
    , m_cancelText(tr("取消"))
    , m_cancellingText(tr("正在取消..."))
{
    setObjectName(QStringLiteral("taskStatusWidget"));

    m_statusLabel->setObjectName(QStringLiteral("statusLabel"));
    m_statusLabel->setMinimumWidth(180);

    m_progressBar->setObjectName(QStringLiteral("progressBar"));
    m_progressBar->setRange(0, 100);
    m_progressBar->setFixedWidth(200);
    m_progressBar->setFixedHeight(16);
    m_progressBar->setTextVisible(false);

    m_cancelButton->setObjectName(QStringLiteral("cancelButton"));
    m_cancelButton->setText(m_cancelText);
    m_cancelButton->setFixedHeight(18);

    auto *layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    layout->addWidget(m_statusLabel);
    layout->addWidget(m_progressBar);
    layout->addWidget(m_cancelButton);

    connect(m_cancelButton, &QToolButton::clicked, this, [this]()
    {
        markCancelling();
        emit cancelRequested();
    });

    hide();
}

void TaskStatusWidget::setCancellable(bool cancellable)
{
    m_cancelButton->setVisible(cancellable);
}

void TaskStatusWidget::setCancellingText(const QString &text)
{
    m_cancellingText = text;
}

void TaskStatusWidget::setLabelMinimumWidth(int width)
{
    m_statusLabel->setMinimumWidth(width);
}

void TaskStatusWidget::setProgressFixedWidth(int width)
{
    m_progressBar->setFixedWidth(width);
}

void TaskStatusWidget::begin(const QString &statusText, int minimum, int maximum)
{
    m_active = true;
    m_cancelling = false;
    m_statusLabel->setText(statusText);
    m_progressBar->setRange(minimum, maximum);
    m_progressBar->setValue(minimum);
    m_cancelButton->setEnabled(true);
    m_cancelButton->setText(m_cancelText);
    show();
}

void TaskStatusWidget::updateProgress(const QString &statusText, int value)
{
    if (!m_active)
    {
        begin(statusText, m_progressBar->minimum(), m_progressBar->maximum());
    }

    m_progressBar->setValue(value);
    if (!m_cancelling)
    {
        m_statusLabel->setText(statusText);
    }
}

void TaskStatusWidget::finish()
{
    m_active = false;
    m_cancelling = false;
    m_cancelButton->setEnabled(true);
    m_cancelButton->setText(m_cancelText);
    hide();
}

bool TaskStatusWidget::isActive() const
{
    return m_active;
}

bool TaskStatusWidget::isCancelling() const
{
    return m_cancelling;
}

QString TaskStatusWidget::statusText() const
{
    return m_statusLabel->text();
}

int TaskStatusWidget::progressValue() const
{
    return m_progressBar->value();
}

int TaskStatusWidget::progressMaximum() const
{
    return m_progressBar->maximum();
}

void TaskStatusWidget::markCancelling()
{
    if (m_cancelling)
    {
        return;
    }

    m_cancelling = true;
    m_statusLabel->setText(m_cancellingText);
    m_cancelButton->setEnabled(false);
    m_cancelButton->setText(tr("正在取消"));
}
