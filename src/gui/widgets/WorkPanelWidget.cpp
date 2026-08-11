#include "WorkPanelWidget.h"

#include <QAbstractItemView>
#include <QHeaderView>
#include <QJsonObject>
#include <QLabel>
#include <QProgressBar>
#include <QStackedWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>

#include <algorithm>

namespace
{
QTableWidgetItem *readOnlyItem(const QString &text)
{
    auto *item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    return item;
}
}

WorkPanelWidget::WorkPanelWidget(QWidget *parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("workPanel"));

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    _stack = new QStackedWidget(this);
    _stack->setObjectName(QStringLiteral("workPanelStack"));
    layout->addWidget(_stack);

    _emptyLabel = new QLabel(tr("当前没有正在运行的工作"), _stack);
    _emptyLabel->setObjectName(QStringLiteral("workPanelEmptyLabel"));
    _emptyLabel->setAlignment(Qt::AlignCenter);
    _emptyLabel->setStyleSheet(QStringLiteral("color: palette(mid); padding: 12px;"));
    _stack->addWidget(_emptyLabel);

    _taskTable = new QTableWidget(_stack);
    _taskTable->setObjectName(QStringLiteral("workPanelTaskTable"));
    _taskTable->setColumnCount(3);
    _taskTable->setHorizontalHeaderLabels({tr("工作"), tr("状态"), tr("进度")});
    _taskTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _taskTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    _taskTable->setSelectionMode(QAbstractItemView::SingleSelection);
    _taskTable->setShowGrid(false);
    _taskTable->setAlternatingRowColors(true);
    _taskTable->setWordWrap(false);
    _taskTable->verticalHeader()->setVisible(false);
    _taskTable->verticalHeader()->setDefaultSectionSize(28);
    _taskTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    _taskTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
    _taskTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    _stack->addWidget(_taskTable);

    setTaskSnapshots({});
}

void WorkPanelWidget::setTaskSnapshots(const QJsonArray &tasks)
{
    _taskTable->setRowCount(0);
    for (const QJsonValue &snapshot : tasks)
    {
        if (!snapshot.isObject())
        {
            continue;
        }

        const QJsonObject task = snapshot.toObject();
        if (!task.value(QStringLiteral("active")).toBool(false)
            && !task.value(QStringLiteral("cancelling")).toBool(false))
        {
            continue;
        }

        const int row = _taskTable->rowCount();
        _taskTable->insertRow(row);
        _taskTable->setItem(row,
                            0,
                            readOnlyItem(task.value(QStringLiteral("name")).toString()));
        _taskTable->setItem(row,
                            1,
                            readOnlyItem(task.value(QStringLiteral("status_text")).toString()));

        const int progress_value = task.value(QStringLiteral("progress_value")).toInt(-1);
        const int maximum = task.value(QStringLiteral("progress_maximum")).toInt(-1);
        auto *progress = new QProgressBar(_taskTable);
        progress->setObjectName(QStringLiteral("workPanelProgress"));
        progress->setTextVisible(true);
        progress->setMinimumWidth(130);
        if (maximum > 0)
        {
            progress->setRange(0, maximum);
            progress->setValue(std::clamp(progress_value, 0, maximum));
        }
        else
        {
            progress->setRange(0, 0);
        }
        _taskTable->setCellWidget(row, 2, progress);
    }

    _stack->setCurrentWidget(_taskTable->rowCount() > 0 ? static_cast<QWidget *>(_taskTable)
                                                        : static_cast<QWidget *>(_emptyLabel));
}
