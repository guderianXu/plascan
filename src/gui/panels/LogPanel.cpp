#include "LogPanel.h"

#include "LogEntryModel.h"
#include "ui_LogPanel.h"

#include <QAbstractItemView>
#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QKeySequence>
#include <QMenu>
#include <QMessageBox>
#include <QMetaObject>
#include <QPointer>
#include <QScrollBar>
#include <QShowEvent>
#include <QStyle>
#include <QTableView>
#include <QTextStream>
#include <QToolButton>
#include <QUrl>

#include <algorithm>

LogPanel::LogPanel(QWidget* parent) : QWidget(parent), _pendingQueue(std::make_shared<PendingQueue>())
{
    Ui::LogPanel ui;
    ui.setupUi(this);

    _table = ui.logTable;
    _sessionCombo = ui.sessionCombo;
    _levelCombo = ui.levelCombo;
    _searchEdit = ui.searchEdit;
    _countLabel = ui.countLabel;
    _followButton = ui.followButton;
    _clearButton = ui.clearButton;
    _saveButton = ui.saveButton;
    _openDirectoryButton = ui.openDirectoryButton;

    _model = new LogEntryModel(this);
    _proxy = new LogFilterProxyModel(this);
    _proxy->setSourceModel(_model);
    _table->setModel(_proxy);
    _table->setObjectName(QStringLiteral("consoleLogTable"));
    _table->setAlternatingRowColors(true);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setSelectionBehavior(QAbstractItemView::SelectRows);
    _table->setSelectionMode(QAbstractItemView::ExtendedSelection);
    _table->setShowGrid(false);
    _table->setWordWrap(false);
    _table->verticalHeader()->setVisible(false);
    _table->verticalHeader()->setDefaultSectionSize(24);
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->horizontalHeader()->setSectionResizeMode(LogEntryModel::TimeColumn, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(LogEntryModel::LevelColumn, QHeaderView::ResizeToContents);
    _table->horizontalHeader()->setSectionResizeMode(LogEntryModel::ContextColumn, QHeaderView::ResizeToContents);

    auto* copy_action = new QAction(tr("复制选中日志"), _table);
    copy_action->setObjectName(QStringLiteral("copyConsoleSelectionAction"));
    copy_action->setShortcut(QKeySequence::Copy);
    copy_action->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    _table->addAction(copy_action);
    connect(copy_action,
            &QAction::triggered,
            this,
            [this]()
            {
                QModelIndexList rows = _table->selectionModel()->selectedRows();
                std::sort(rows.begin(),
                          rows.end(),
                          [](const QModelIndex& left, const QModelIndex& right) { return left.row() < right.row(); });
                QString text;
                for (const QModelIndex& row : rows)
                {
                    text += row.data(LogEntryModel::FormattedRole).toString();
                }
                if (!text.isEmpty())
                {
                    QApplication::clipboard()->setText(text);
                }
            });

    const QString current_session = QString::fromStdString(Logger::instance()->sessionId());
    _sessionCombo->setObjectName(QStringLiteral("consoleSessionCombo"));
    _sessionCombo->addItem(tr("当前会话"), current_session);
    _sessionCombo->addItem(tr("全部会话"), QString());
    _sessionCombo->setToolTip(tr("选择要显示的运行会话"));
    _proxy->setSessionId(current_session);

    _levelCombo->setObjectName(QStringLiteral("consoleLevelCombo"));
    _levelCombo->addItem(tr("常规"), static_cast<int>(Logger::Info));
    _levelCombo->addItem(tr("警告及以上"), static_cast<int>(Logger::Warn));
    _levelCombo->addItem(tr("仅错误"), static_cast<int>(Logger::Error));
    _levelCombo->addItem(tr("全部（含调试）"), static_cast<int>(Logger::Debug));
    _levelCombo->setToolTip(tr("最低显示级别"));

    _searchEdit->setObjectName(QStringLiteral("consoleSearchEdit"));
    _searchEdit->setClearButtonEnabled(true);
    _searchEdit->setPlaceholderText(tr("搜索消息或上下文"));
    _searchEdit->setMinimumWidth(160);

    _followButton->setObjectName(QStringLiteral("consoleFollowButton"));
    _followButton->setCheckable(true);
    _followButton->setChecked(true);
    _followButton->setText(tr("跟随"));
    _followButton->setToolTip(tr("自动滚动到最新日志"));

    _clearButton->setObjectName(QStringLiteral("clearConsoleButton"));
    _clearButton->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));
    _clearButton->setToolTip(tr("清空当前视图（不删除日志文件）"));
    _clearButton->setPopupMode(QToolButton::MenuButtonPopup);
    auto* clear_menu = new QMenu(_clearButton);
    clear_menu->addAction(tr("删除磁盘日志…"), this, &LogPanel::deleteDiskHistory);
    _clearButton->setMenu(clear_menu);

    _saveButton->setObjectName(QStringLiteral("saveConsoleButton"));
    _saveButton->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    _saveButton->setToolTip(tr("导出当前筛选结果"));
    _openDirectoryButton->setObjectName(QStringLiteral("openLogDirectoryButton"));
    _openDirectoryButton->setIcon(style()->standardIcon(QStyle::SP_DirOpenIcon));
    _openDirectoryButton->setToolTip(tr("打开日志目录"));

    connect(_sessionCombo,
            &QComboBox::currentIndexChanged,
            this,
            [this](int index)
            {
                _proxy->clearTaskRange();
                _proxy->setSessionId(_sessionCombo->itemData(index).toString());
                updateCountLabel();
            });
    connect(_levelCombo,
            &QComboBox::currentIndexChanged,
            this,
            [this](int index)
            {
                _proxy->setMinimumLevel(static_cast<Logger::Level>(_levelCombo->itemData(index).toInt()));
                updateCountLabel();
            });
    connect(_searchEdit, &QLineEdit::textChanged, _proxy, &LogFilterProxyModel::setSearchText);
    connect(_searchEdit, &QLineEdit::textChanged, this, &LogPanel::updateCountLabel);
    connect(_followButton,
            &QToolButton::toggled,
            this,
            [this](bool enabled)
            {
                if (enabled)
                {
                    scrollToLatest();
                    markRead();
                }
            });
    connect(_table->verticalScrollBar(),
            &QScrollBar::valueChanged,
            this,
            [this](int value)
            {
                if (value < _table->verticalScrollBar()->maximum() && _followButton->isChecked())
                {
                    _followButton->setChecked(false);
                }
            });
    connect(_clearButton, &QToolButton::clicked, this, &LogPanel::clearLogs);
    connect(_saveButton,
            &QToolButton::clicked,
            this,
            [this]()
            {
                const QString path = QFileDialog::getSaveFileName(
                    this, tr("导出控制台筛选结果"), QDir::homePath(), tr("文本文件 (*.txt);;所有文件 (*)"));
                if (!path.isEmpty())
                {
                    saveLogsToFile(path);
                }
            });
    connect(_openDirectoryButton,
            &QToolButton::clicked,
            this,
            []() {
                QDesktopServices::openUrl(
                    QUrl::fromLocalFile(QString::fromStdString(Logger::instance()->logDirectory())));
            });

    QPointer<LogPanel> self(this);
    const std::shared_ptr<PendingQueue> pending = _pendingQueue;
    _sinkId = Logger::instance()->registerSink(
        [self, pending](const Logger::Entry& entry)
        {
            bool schedule_drain = false;
            {
                std::lock_guard<std::mutex> lock(pending->mutex);
                pending->entries.push_back(entry);
                if (!pending->drainScheduled)
                {
                    pending->drainScheduled = true;
                    schedule_drain = true;
                }
            }
            if (schedule_drain && self)
            {
                QMetaObject::invokeMethod(
                    self,
                    [self]()
                    {
                        if (self)
                        {
                            self->drainPendingEntries();
                        }
                    },
                    Qt::QueuedConnection);
            }
        });

    appendEntries(Logger::instance()->recentEntries());
}

LogPanel::~LogPanel()
{
    if (_sinkId != 0)
    {
        Logger::instance()->unregisterSink(_sinkId);
    }
}

QSize LogPanel::minimumSizeHint() const
{
    return QSize(360, 110);
}

QSize LogPanel::sizeHint() const
{
    return QSize(760, 220);
}

void LogPanel::append(const QString& text)
{
    appendLog(text + QLatin1Char('\n'), static_cast<int>(Logger::Info));
}

void LogPanel::appendLog(const QString& formatted, int level)
{
    Logger::Entry entry;
    entry.level =
        static_cast<Logger::Level>(std::clamp(level, static_cast<int>(Logger::Debug), static_cast<int>(Logger::Error)));
    entry.timestamp = QStringLiteral("manual").toStdString();
    entry.sessionId = Logger::instance()->sessionId();
    entry.message = formatted.trimmed().toStdString();
    entry.formatted = formatted.toStdString();
    appendEntries({entry});
}

void LogPanel::clearLogs()
{
    _model->clear();
    markRead();
    updateCountLabel();
}

bool LogPanel::saveLogsToFile(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        return false;
    }
    QTextStream stream(&file);
    for (int row = 0; row < _proxy->rowCount(); ++row)
    {
        stream << _proxy->index(row, 0).data(LogEntryModel::FormattedRole).toString();
    }
    return stream.status() == QTextStream::Ok;
}

void LogPanel::focusLogRange(qulonglong firstSequence, qulonglong lastSequence, const QString& taskId)
{
    _sessionCombo->setCurrentIndex(0);
    _levelCombo->setCurrentIndex(3);
    _searchEdit->clear();
    _proxy->setSessionId(QString::fromStdString(Logger::instance()->sessionId()));
    _proxy->setTaskRange(firstSequence, lastSequence, taskId);
    updateCountLabel();
    if (_proxy->rowCount() > 0)
    {
        _table->scrollTo(_proxy->index(0, 0), QAbstractItemView::PositionAtTop);
        _table->selectRow(0);
    }
}

void LogPanel::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    if (_followButton->isChecked())
    {
        scrollToLatest();
        markRead();
    }
}

void LogPanel::drainPendingEntries()
{
    std::vector<Logger::Entry> entries;
    {
        std::lock_guard<std::mutex> lock(_pendingQueue->mutex);
        entries.swap(_pendingQueue->entries);
        _pendingQueue->drainScheduled = false;
    }
    appendEntries(entries);
}

void LogPanel::appendEntries(const std::vector<Logger::Entry>& entries)
{
    if (entries.empty())
    {
        return;
    }
    _model->appendEntries(entries);
    if (!isVisible() || !_followButton->isChecked())
    {
        for (const Logger::Entry& entry : entries)
        {
            _unreadWarnings += entry.level == Logger::Warn ? 1 : 0;
            _unreadErrors += entry.level == Logger::Error ? 1 : 0;
        }
        emit unreadCountsChanged(_unreadWarnings, _unreadErrors);
    }
    else
    {
        scrollToLatest();
    }
    updateCountLabel();
}

void LogPanel::updateCountLabel()
{
    _countLabel->setText(tr("%1 / %2").arg(_proxy->rowCount()).arg(_model->rowCount()));
    _countLabel->setToolTip(tr("当前筛选结果 / 已加载记录"));
}

void LogPanel::scrollToLatest()
{
    if (_proxy->rowCount() > 0)
    {
        _table->scrollToBottom();
    }
}

void LogPanel::markRead()
{
    if (_unreadWarnings == 0 && _unreadErrors == 0)
    {
        return;
    }
    _unreadWarnings = 0;
    _unreadErrors = 0;
    emit unreadCountsChanged(0, 0);
}

void LogPanel::deleteDiskHistory()
{
    const QMessageBox::StandardButton result =
        QMessageBox::warning(this,
                             tr("删除磁盘日志"),
                             tr("这会永久清空当前日志文件，但不会清空正在显示的记录。"
                                "是否继续？"),
                             QMessageBox::Yes | QMessageBox::No,
                             QMessageBox::No);
    if (result == QMessageBox::Yes)
    {
        Logger::instance()->clearLogFile();
    }
}

void LogPanel::loadFromLogFile()
{
    QFile file(QString::fromStdString(Logger::instance()->logFilePath()));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return;
    }

    constexpr qint64 limit = 2 * 1024 * 1024;
    QByteArray data;
    if (file.size() <= limit)
    {
        data = file.readAll();
    }
    else
    {
        file.seek(file.size() - limit);
        file.readLine();
        data = file.readAll();
    }

    const std::string current_session = Logger::instance()->sessionId();
    std::vector<Logger::Entry> history = LogEntryModel::parseLegacyText(data, current_session);
    const std::vector<Logger::Entry> recent = Logger::instance()->recentEntries();
    if (!recent.empty())
    {
        history.erase(std::remove_if(history.begin(),
                                     history.end(),
                                     [&current_session](const Logger::Entry& entry)
                                     { return entry.sessionId == current_session; }),
                      history.end());
    }
    _model->clear();
    _model->appendEntries(history);
    _model->appendEntries(recent);
    updateCountLabel();
    scrollToLatest();
}
