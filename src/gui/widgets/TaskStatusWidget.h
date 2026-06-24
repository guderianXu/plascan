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

    QLabel *_statusLabel = nullptr;
    QProgressBar *_progressBar = nullptr;
    QToolButton *_cancelButton = nullptr;
    QString _cancelText;
    QString _cancellingText;
    bool _active = false;
    bool _cancelling = false;
};
