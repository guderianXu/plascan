#pragma once

#include <QHash>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QVector>

#include <memory>

class QEvent;
class QWidget;

namespace xjw::gui::platform
{

enum class TaskbarProgressState
{
    NoProgress,
    Indeterminate,
    Normal
};

struct TaskbarProgressItem
{
    int value{0};
    int maximum{0};
};

struct TaskbarProgressSnapshot
{
    TaskbarProgressState state{TaskbarProgressState::NoProgress};
    int value{0};
    int maximum{0};
};

TaskbarProgressSnapshot aggregateTaskbarProgress(
    const QVector<TaskbarProgressItem> &items);

class TaskbarProgressController final : public QObject
{
public:
    explicit TaskbarProgressController(QWidget *window, QObject *parent = nullptr);
    ~TaskbarProgressController() override;

    void updateTask(const QString &taskId, int value, int maximum);
    void finishTask(const QString &taskId);
    void clearTasks();
    bool hasTask(const QString &taskId) const;
    TaskbarProgressSnapshot currentProgress() const;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    struct NativeData;

    void refreshProgress();
    void setProgress(const TaskbarProgressSnapshot &progress);
    void applyProgress();
    void handleTaskbarButtonCreated(quintptr windowId);

    QPointer<QWidget> _window;
    QHash<QString, TaskbarProgressItem> _tasks;
    TaskbarProgressSnapshot _progress;
    std::unique_ptr<NativeData> _native;
};

} // namespace xjw::gui::platform
