#include "TaskStatusWidget.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QToolButton>

TaskStatusWidget::TaskStatusWidget(QWidget *parent)
    : QWidget(parent)
    , _statusLabel(new QLabel(this))
    , _progressBar(new QProgressBar(this))
    , _cancelButton(new QToolButton(this))
    , _cancelText(tr("取消"))
    , _cancellingText(tr("正在取消..."))
{
    setObjectName(QStringLiteral("taskStatusWidget"));

    _statusLabel->setObjectName(QStringLiteral("statusLabel"));
    _statusLabel->setMinimumWidth(180);

    _progressBar->setObjectName(QStringLiteral("progressBar"));
    _progressBar->setRange(0, 100);
    _progressBar->setFixedWidth(200);
    _progressBar->setFixedHeight(16);
    _progressBar->setTextVisible(false);

    _cancelButton->setObjectName(QStringLiteral("cancelButton"));
    _cancelButton->setText(_cancelText);
    _cancelButton->setFixedHeight(18);

    auto *layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    layout->addWidget(_statusLabel);
    layout->addWidget(_progressBar);
    layout->addWidget(_cancelButton);

    connect(_cancelButton, &QToolButton::clicked, this, [this]()
    {
        markCancelling();
        emit cancelRequested();
    });

    hide();
}

void TaskStatusWidget::setCancellable(bool cancellable)
{
    _cancelButton->setVisible(cancellable);
}

void TaskStatusWidget::setCancellingText(const QString &text)
{
    _cancellingText = text;
}

void TaskStatusWidget::setLabelMinimumWidth(int width)
{
    _statusLabel->setMinimumWidth(width);
}

void TaskStatusWidget::begin(const QString &statusText, int minimum, int maximum)
{
    if (!_active)
    {
        _elapsedTimer.start();
    }
    _active = true;
    _cancelling = false;
    _statusLabel->setText(statusText);
    _progressBar->setRange(minimum, maximum);
    _progressBar->setValue(minimum);
    _cancelButton->setEnabled(true);
    _cancelButton->setText(_cancelText);
    show();
}

void TaskStatusWidget::updateProgress(const QString &statusText, int value)
{
    if (!_active)
    {
        begin(statusText, _progressBar->minimum(), _progressBar->maximum());
    }

    _progressBar->setValue(value);
    if (!_cancelling)
    {
        _statusLabel->setText(statusText);
    }
}

void TaskStatusWidget::finish()
{
    _active = false;
    _cancelling = false;
    _elapsedTimer.invalidate();
    _cancelButton->setEnabled(true);
    _cancelButton->setText(_cancelText);
    hide();
}

bool TaskStatusWidget::isActive() const
{
    return _active;
}

bool TaskStatusWidget::isCancelling() const
{
    return _cancelling;
}

QString TaskStatusWidget::statusText() const
{
    return _statusLabel->text();
}

int TaskStatusWidget::progressValue() const
{
    return _progressBar->value();
}

int TaskStatusWidget::progressMaximum() const
{
    return _progressBar->maximum();
}

qint64 TaskStatusWidget::elapsedMilliseconds() const
{
    return _elapsedTimer.isValid() ? _elapsedTimer.elapsed() : 0;
}

void TaskStatusWidget::markCancelling()
{
    if (_cancelling)
    {
        return;
    }

    _cancelling = true;
    _statusLabel->setText(_cancellingText);
    _cancelButton->setEnabled(false);
    _cancelButton->setText(tr("正在取消"));
}
