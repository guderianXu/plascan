#pragma once

#include <QElapsedTimer>
#include <QJsonArray>
#include <QWidget>

class QLabel;
class QStackedWidget;
class QTableWidget;
class QTimer;

/**
 * @brief 下方“工作”面板，集中展示当前正在执行或取消中的后台任务。
 */
class WorkPanelWidget final : public QWidget
{
    Q_OBJECT

public:
    explicit WorkPanelWidget(QWidget *parent = nullptr);

public slots:
    void setTaskSnapshots(const QJsonArray &tasks);

private:
    void updateElapsedTimes();

    QLabel *_emptyLabel{};
    QStackedWidget *_stack{};
    QTableWidget *_taskTable{};
    QTimer *_elapsedRefreshTimer{};
    QElapsedTimer _snapshotAge;
};
