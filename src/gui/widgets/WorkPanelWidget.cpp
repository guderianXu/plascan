#include "WorkPanelWidget.h"

#include <QAbstractItemView>
#include <QHeaderView>
#include <QComboBox>
#include <QHBoxLayout>
#include <QJsonObject>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>

namespace
{
    QTableWidgetItem* readOnlyItem(const QString& text)
    {
        auto* item = new QTableWidgetItem(text);
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        return item;
    }

    QString formatElapsedTime(qint64 elapsed_ms)
    {
        const qint64 total_seconds = std::max<qint64>(0, elapsed_ms) / 1000;
        const qint64 hours = total_seconds / 3600;
        const qint64 minutes = (total_seconds % 3600) / 60;
        const qint64 seconds = total_seconds % 60;
        return QStringLiteral("%1:%2:%3")
            .arg(hours, 2, 10, QLatin1Char('0'))
            .arg(minutes, 2, 10, QLatin1Char('0'))
            .arg(seconds, 2, 10, QLatin1Char('0'));
    }

    QString stateText(const QJsonObject& task)
    {
        if (task.value(QStringLiteral("cancelling")).toBool(false))
        {
            return QObject::tr("正在取消");
        }
        const QString state = task.value(QStringLiteral("state")).toString();
        if (state == QStringLiteral("succeeded"))
        {
            return QObject::tr("已完成");
        }
        if (state == QStringLiteral("failed"))
        {
            return QObject::tr("失败");
        }
        if (state == QStringLiteral("cancelled"))
        {
            return QObject::tr("已取消");
        }
        if (state == QStringLiteral("queued"))
        {
            return QObject::tr("排队中");
        }
        if (state == QStringLiteral("blocked"))
        {
            return QObject::tr("等待资源");
        }
        if (state == QStringLiteral("pause_requested"))
        {
            return QObject::tr("正在暂停");
        }
        if (state == QStringLiteral("paused"))
        {
            return QObject::tr("已暂停");
        }
        if (state == QStringLiteral("interrupted"))
        {
            return QObject::tr("异常中断");
        }
        return QObject::tr("运行中");
    }
} // namespace

WorkPanelWidget::WorkPanelWidget(QWidget* parent) : QWidget(parent)
{
    setObjectName(QStringLiteral("workPanel"));

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    auto* toolbar = new QWidget(this);
    toolbar->setObjectName(QStringLiteral("workPanelToolbar"));
    auto* toolbar_layout = new QHBoxLayout(toolbar);
    toolbar_layout->setContentsMargins(4, 3, 4, 3);
    toolbar_layout->setSpacing(4);
    _filterCombo = new QComboBox(toolbar);
    _filterCombo->setObjectName(QStringLiteral("workPanelFilterCombo"));
    _filterCombo->addItem(tr("全部工作"), QStringLiteral("all"));
    _filterCombo->addItem(tr("运行中"), QStringLiteral("active"));
    _filterCombo->addItem(tr("历史"), QStringLiteral("history"));
    _clearHistoryButton = new QPushButton(tr("清除历史"), toolbar);
    _clearHistoryButton->setObjectName(QStringLiteral("workPanelClearHistoryButton"));
    _moveUpButton = new QPushButton(tr("上移"), toolbar);
    _moveUpButton->setObjectName(QStringLiteral("workPanelMoveUpButton"));
    _moveDownButton = new QPushButton(tr("下移"), toolbar);
    _moveDownButton->setObjectName(QStringLiteral("workPanelMoveDownButton"));
    _pauseResumeButton = new QPushButton(tr("暂停"), toolbar);
    _pauseResumeButton->setObjectName(QStringLiteral("workPanelPauseResumeButton"));
    _cancelButton = new QPushButton(tr("取消"), toolbar);
    _cancelButton->setObjectName(QStringLiteral("workPanelCancelButton"));
    toolbar_layout->addWidget(_filterCombo);
    toolbar_layout->addStretch(1);
    toolbar_layout->addWidget(_moveUpButton);
    toolbar_layout->addWidget(_moveDownButton);
    toolbar_layout->addWidget(_pauseResumeButton);
    toolbar_layout->addWidget(_cancelButton);
    toolbar_layout->addWidget(_clearHistoryButton);
    layout->addWidget(toolbar);

    _stack = new QStackedWidget(this);
    _stack->setObjectName(QStringLiteral("workPanelStack"));
    layout->addWidget(_stack);

    _emptyLabel = new QLabel(tr("暂无工作记录"), _stack);
    _emptyLabel->setObjectName(QStringLiteral("workPanelEmptyLabel"));
    _emptyLabel->setAlignment(Qt::AlignCenter);
    _emptyLabel->setStyleSheet(QStringLiteral("color: palette(mid); padding: 12px;"));
    _stack->addWidget(_emptyLabel);

    _taskTable = new QTableWidget(_stack);
    _taskTable->setObjectName(QStringLiteral("workPanelTaskTable"));
    _taskTable->setColumnCount(4);
    _taskTable->setHorizontalHeaderLabels({tr("工作"), tr("状态"), tr("用时"), tr("进度")});
    _taskTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _taskTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _taskTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _taskTable->setShowGrid(false);
    _taskTable->setAlternatingRowColors(true);
    _taskTable->setWordWrap(false);
    _taskTable->verticalHeader()->setVisible(false);
    _taskTable->verticalHeader()->setDefaultSectionSize(28);
    _taskTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    _taskTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    _taskTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    _taskTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    _stack->addWidget(_taskTable);

    _elapsedRefreshTimer = new QTimer(this);
    connect(_elapsedRefreshTimer, &QTimer::timeout, this, &WorkPanelWidget::updateElapsedTimes);
    _elapsedRefreshTimer->start(1000);

    connect(_filterCombo, &QComboBox::currentIndexChanged, this, [this](int) { rebuildTable(); });
    connect(_clearHistoryButton, &QPushButton::clicked, this, &WorkPanelWidget::clearHistoryRequested);
    connect(_taskTable, &QTableWidget::itemSelectionChanged, this, &WorkPanelWidget::updateTaskActions);
    connect(_pauseResumeButton,
            &QPushButton::clicked,
            this,
            [this]()
            {
                const QJsonObject task = selectedTask();
                if (task.isEmpty())
                {
                    return;
                }
                emit taskCommandRequested(task.value(QStringLiteral("can_resume")).toBool(false)
                                              ? QStringLiteral("resume")
                                              : QStringLiteral("pause"),
                                          task.value(QStringLiteral("run_id")).toString(),
                                          {},
                                          task.value(QStringLiteral("priority")).toInt(),
                                          task.value(QStringLiteral("revision")).toVariant().toULongLong());
            });
    connect(_cancelButton,
            &QPushButton::clicked,
            this,
            [this]()
            {
                const QJsonObject task = selectedTask();
                if (!task.isEmpty())
                {
                    emit taskCommandRequested(QStringLiteral("cancel"),
                                              task.value(QStringLiteral("run_id")).toString(),
                                              {},
                                              task.value(QStringLiteral("priority")).toInt(),
                                              task.value(QStringLiteral("revision")).toVariant().toULongLong());
                }
            });
    const auto connect_move = [this](QPushButton* button, bool previous)
    {
        connect(button,
                &QPushButton::clicked,
                this,
                [this, previous]()
                {
                    const QJsonObject task = selectedTask();
                    const QString reference_run_id = adjacentRunId(previous);
                    if (task.isEmpty() || reference_run_id.isEmpty())
                    {
                        return;
                    }
                    emit taskCommandRequested(previous ? QStringLiteral("move_before") : QStringLiteral("move_after"),
                                              task.value(QStringLiteral("run_id")).toString(),
                                              reference_run_id,
                                              task.value(QStringLiteral("priority")).toInt(),
                                              task.value(QStringLiteral("revision")).toVariant().toULongLong());
                });
    };
    connect_move(_moveUpButton, true);
    connect_move(_moveDownButton, false);
    connect(_taskTable,
            &QTableWidget::cellDoubleClicked,
            this,
            [this](int row, int)
            {
                const QTableWidgetItem* item = _taskTable->item(row, 0);
                if (!item)
                {
                    return;
                }
                emit logRangeRequested(item->data(Qt::UserRole + 1).toULongLong(),
                                       item->data(Qt::UserRole + 2).toULongLong(),
                                       item->data(Qt::UserRole + 3).toString());
            });

    setTaskSnapshots({});
}

QJsonArray WorkPanelWidget::taskSnapshots() const
{
    return _taskSnapshots;
}

void WorkPanelWidget::setTaskSnapshots(const QJsonArray& tasks)
{
    _snapshotAge.start();
    _taskSnapshots = tasks;
    rebuildTable();
}

void WorkPanelWidget::rebuildTable()
{
    _taskTable->setRowCount(0);
    const QString filter = _filterCombo->currentData().toString();
    for (const QJsonValue& snapshot : _taskSnapshots)
    {
        if (!snapshot.isObject())
        {
            continue;
        }

        const QJsonObject task = snapshot.toObject();
        const bool active = task.value(QStringLiteral("active")).toBool(false) ||
                            task.value(QStringLiteral("cancelling")).toBool(false);
        if ((filter == QStringLiteral("active") && !active) || (filter == QStringLiteral("history") && active))
        {
            continue;
        }

        const int row = _taskTable->rowCount();
        _taskTable->insertRow(row);
        auto* name_item = readOnlyItem(task.value(QStringLiteral("name")).toString());
        name_item->setToolTip(task.value(QStringLiteral("status_text")).toString());
        name_item->setData(Qt::UserRole + 1, task.value(QStringLiteral("start_sequence")).toVariant());
        name_item->setData(Qt::UserRole + 2, task.value(QStringLiteral("end_sequence")).toVariant());
        name_item->setData(Qt::UserRole + 3, task.value(QStringLiteral("task_id")).toString());
        name_item->setData(Qt::UserRole + 4, task.toVariantMap());
        _taskTable->setItem(row, 0, name_item);
        auto* state_item = readOnlyItem(stateText(task));
        state_item->setTextAlignment(Qt::AlignCenter);
        _taskTable->setItem(row, 1, state_item);
        auto* elapsed_item = readOnlyItem(QString());
        elapsed_item->setTextAlignment(Qt::AlignCenter);
        elapsed_item->setData(Qt::UserRole, task.value(QStringLiteral("elapsed_ms")).toVariant());
        elapsed_item->setData(Qt::UserRole + 1, active);
        _taskTable->setItem(row, 2, elapsed_item);

        const int progress_value = task.value(QStringLiteral("progress_value")).toInt(-1);
        const int maximum = task.value(QStringLiteral("progress_maximum")).toInt(-1);
        auto* progress = new QProgressBar(_taskTable);
        progress->setObjectName(QStringLiteral("workPanelProgress"));
        progress->setTextVisible(true);
        progress->setMinimumWidth(130);
        const QString state = task.value(QStringLiteral("state")).toString();
        const bool running = state == QStringLiteral("running") || state == QStringLiteral("pause_requested") ||
                             state == QStringLiteral("cancel_requested") || state.isEmpty();
        if (maximum > 0)
        {
            progress->setRange(0, maximum);
            progress->setValue(std::clamp(progress_value, 0, maximum));
        }
        else if (running)
        {
            progress->setRange(0, 0);
        }
        else
        {
            progress->setRange(0, 1);
            progress->setValue(0);
            progress->setFormat(stateText(task));
        }
        if (!active)
        {
            progress->setRange(0, 1);
            progress->setValue(task.value(QStringLiteral("state")).toString() == QStringLiteral("succeeded") ? 1 : 0);
            progress->setFormat(stateText(task));
        }
        _taskTable->setCellWidget(row, 3, progress);
    }

    updateElapsedTimes();

    _stack->setCurrentWidget(_taskTable->rowCount() > 0 ? static_cast<QWidget*>(_taskTable)
                                                        : static_cast<QWidget*>(_emptyLabel));
    updateTaskActions();
}

void WorkPanelWidget::updateElapsedTimes()
{
    const qint64 snapshot_age_ms = _snapshotAge.isValid() ? _snapshotAge.elapsed() : 0;
    for (int row = 0; row < _taskTable->rowCount(); ++row)
    {
        QTableWidgetItem* elapsed_item = _taskTable->item(row, 2);
        if (!elapsed_item)
        {
            continue;
        }
        const qint64 elapsed_ms = elapsed_item->data(Qt::UserRole).toLongLong();
        const bool active = elapsed_item->data(Qt::UserRole + 1).toBool();
        elapsed_item->setText(formatElapsedTime(elapsed_ms + (active ? snapshot_age_ms : 0)));
    }
}

QJsonObject WorkPanelWidget::selectedTask() const
{
    const int row = _taskTable->currentRow();
    const QTableWidgetItem* item = row >= 0 ? _taskTable->item(row, 0) : nullptr;
    return item ? QJsonObject::fromVariantMap(item->data(Qt::UserRole + 4).toMap()) : QJsonObject();
}

QString WorkPanelWidget::adjacentRunId(bool previous) const
{
    const int row = _taskTable->currentRow();
    const int adjacent_row = previous ? row - 1 : row + 1;
    const QTableWidgetItem* item =
        adjacent_row >= 0 && adjacent_row < _taskTable->rowCount() ? _taskTable->item(adjacent_row, 0) : nullptr;
    if (!item)
    {
        return {};
    }
    const QJsonObject task = QJsonObject::fromVariantMap(item->data(Qt::UserRole + 4).toMap());
    return task.value(QStringLiteral("scheduler_managed")).toBool(false) &&
                   task.value(QStringLiteral("can_reorder")).toBool(false)
               ? task.value(QStringLiteral("run_id")).toString()
               : QString();
}

void WorkPanelWidget::updateTaskActions()
{
    const QJsonObject task = selectedTask();
    const bool scheduler_managed = task.value(QStringLiteral("scheduler_managed")).toBool(false);
    const bool can_resume = task.value(QStringLiteral("can_resume")).toBool(false);
    _pauseResumeButton->setText(can_resume ? tr("恢复") : tr("暂停"));
    _pauseResumeButton->setEnabled(scheduler_managed &&
                                   (can_resume || task.value(QStringLiteral("can_pause")).toBool(false)));
    _cancelButton->setEnabled(scheduler_managed && task.value(QStringLiteral("can_cancel")).toBool(false));
    const bool can_reorder = scheduler_managed && task.value(QStringLiteral("can_reorder")).toBool(false);
    _moveUpButton->setEnabled(can_reorder && !adjacentRunId(true).isEmpty());
    _moveDownButton->setEnabled(can_reorder && !adjacentRunId(false).isEmpty());
}
