#pragma once

#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonObject>
#include <QWidget>

class QLabel;
class QComboBox;
class QPushButton;
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
    explicit WorkPanelWidget(QWidget* parent = nullptr);
    QJsonArray taskSnapshots() const;

public slots:
    void setTaskSnapshots(const QJsonArray& tasks);

signals:
    void logRangeRequested(qulonglong firstSequence, qulonglong lastSequence, const QString& taskId);
    void clearHistoryRequested();
    void taskCommandRequested(
        const QString& action, const QString& runId, const QString& referenceRunId, int priority, qulonglong revision);

private:
    void updateElapsedTimes();
    void rebuildTable();
    void updateTaskActions();
    QJsonObject selectedTask() const;
    QString adjacentRunId(bool previous) const;

    QLabel* _emptyLabel{};
    QComboBox* _filterCombo{};
    QPushButton* _clearHistoryButton{};
    QPushButton* _pauseResumeButton{};
    QPushButton* _cancelButton{};
    QPushButton* _moveUpButton{};
    QPushButton* _moveDownButton{};
    QStackedWidget* _stack{};
    QTableWidget* _taskTable{};
    QTimer* _elapsedRefreshTimer{};
    QElapsedTimer _snapshotAge;
    QJsonArray _taskSnapshots;
};
