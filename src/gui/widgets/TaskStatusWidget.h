#pragma once

#include <QWidget>

class QLabel;
class QProgressBar;
class QToolButton;

class TaskStatusWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TaskStatusWidget(QWidget *parent = nullptr);

    void setCancellable(bool cancellable);
    void setCancellingText(const QString &text);
    void setLabelMinimumWidth(int width);
    void setProgressFixedWidth(int width);

    void begin(const QString &statusText, int minimum, int maximum);
    void updateProgress(const QString &statusText, int value);
    void finish();

    bool isActive() const;
    bool isCancelling() const;
    QString statusText() const;
    int progressValue() const;
    int progressMaximum() const;

signals:
    void cancelRequested();

private:
    void markCancelling();

    QLabel *m_statusLabel = nullptr;
    QProgressBar *m_progressBar = nullptr;
    QToolButton *m_cancelButton = nullptr;
    QString m_cancelText;
    QString m_cancellingText;
    bool m_active = false;
    bool m_cancelling = false;
};
