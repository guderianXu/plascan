#include "LogEntryModel.h"

#include <QBrush>
#include <QColor>
#include <QRegularExpression>
#include <QStringList>

#include <algorithm>

namespace
{
    QString levelText(Logger::Level level)
    {
        switch (level)
        {
        case Logger::Debug:
            return QObject::tr("调试");
        case Logger::Info:
            return QObject::tr("信息");
        case Logger::Warn:
            return QObject::tr("警告");
        case Logger::Error:
            return QObject::tr("错误");
        }
        return QObject::tr("信息");
    }

    Logger::Level parseLevel(const QString& text)
    {
        if (text == QStringLiteral("DEBUG"))
        {
            return Logger::Debug;
        }
        if (text == QStringLiteral("WARN"))
        {
            return Logger::Warn;
        }
        if (text == QStringLiteral("ERROR"))
        {
            return Logger::Error;
        }
        return Logger::Info;
    }

    QString inferredCategory(const QString& message)
    {
        static const QRegularExpression prefix(QStringLiteral("^\\[([^\\]]+)\\]"));
        const QRegularExpressionMatch match = prefix.match(message);
        return match.hasMatch() ? match.captured(1) : QString();
    }
} // namespace

LogEntryModel::LogEntryModel(QObject* parent) : QAbstractTableModel(parent)
{
}

int LogEntryModel::rowCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : static_cast<int>(_entries.size());
}

int LogEntryModel::columnCount(const QModelIndex& parent) const
{
    return parent.isValid() ? 0 : ColumnCount;
}

QVariant LogEntryModel::data(const QModelIndex& index, int role) const
{
    const Logger::Entry* entry = entryAt(index.row());
    if (!entry || !index.isValid())
    {
        return {};
    }

    if (role == LevelRole)
    {
        return static_cast<int>(entry->level);
    }
    if (role == SessionRole)
    {
        return QString::fromStdString(entry->sessionId);
    }
    if (role == SequenceRole)
    {
        return QVariant::fromValue<qulonglong>(entry->sequence);
    }
    if (role == TaskIdRole)
    {
        return QString::fromStdString(entry->taskId);
    }
    if (role == FormattedRole)
    {
        return QString::fromStdString(entry->formatted);
    }
    if (role == Qt::ForegroundRole)
    {
        if (entry->level == Logger::Error)
        {
            return QBrush(QColor(QStringLiteral("#c62828")));
        }
        if (entry->level == Logger::Warn)
        {
            return QBrush(QColor(QStringLiteral("#9a6700")));
        }
    }
    if (role != Qt::DisplayRole && role != Qt::ToolTipRole)
    {
        return {};
    }

    if (role == Qt::ToolTipRole)
    {
        return QString::fromStdString(entry->formatted).trimmed();
    }

    switch (index.column())
    {
    case TimeColumn:
        return QString::fromStdString(entry->timestamp).mid(11);
    case LevelColumn:
        return levelText(entry->level);
    case ContextColumn:
    {
        QStringList parts;
        if (!entry->category.empty())
        {
            parts << QString::fromStdString(entry->category);
        }
        if (!entry->stage.empty())
        {
            parts << QString::fromStdString(entry->stage);
        }
        return parts.join(QStringLiteral(" / "));
    }
    case MessageColumn:
        return QString::fromStdString(entry->message);
    default:
        return {};
    }
}

QVariant LogEntryModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (orientation != Qt::Horizontal || role != Qt::DisplayRole)
    {
        return {};
    }
    switch (section)
    {
    case TimeColumn:
        return tr("时间");
    case LevelColumn:
        return tr("级别");
    case ContextColumn:
        return tr("上下文");
    case MessageColumn:
        return tr("消息");
    default:
        return {};
    }
}

void LogEntryModel::appendEntries(const std::vector<Logger::Entry>& entries)
{
    std::vector<Logger::Entry> accepted;
    accepted.reserve(entries.size());
    for (const Logger::Entry& entry : entries)
    {
        if (entry.sequence != 0 && entry.sequence <= _lastLiveSequence)
        {
            continue;
        }
        accepted.push_back(entry);
        _lastLiveSequence = std::max(_lastLiveSequence, entry.sequence);
    }
    if (accepted.empty())
    {
        return;
    }

    const int overflow = std::max(0, static_cast<int>(_entries.size() + accepted.size()) - _maximumEntries);
    if (overflow > 0)
    {
        const int removable = std::min(overflow, static_cast<int>(_entries.size()));
        if (removable > 0)
        {
            beginRemoveRows(QModelIndex(), 0, removable - 1);
            _entries.erase(_entries.begin(), _entries.begin() + removable);
            endRemoveRows();
        }
        if (overflow > removable)
        {
            accepted.erase(accepted.begin(), accepted.begin() + (overflow - removable));
        }
    }
    if (accepted.empty())
    {
        return;
    }

    const int first = static_cast<int>(_entries.size());
    beginInsertRows(QModelIndex(), first, first + static_cast<int>(accepted.size()) - 1);
    _entries.insert(_entries.end(), accepted.begin(), accepted.end());
    endInsertRows();
}

void LogEntryModel::clear()
{
    if (_entries.empty())
    {
        return;
    }
    beginResetModel();
    _entries.clear();
    _lastLiveSequence = 0;
    endResetModel();
}

const Logger::Entry* LogEntryModel::entryAt(int row) const
{
    if (row < 0 || row >= static_cast<int>(_entries.size()))
    {
        return nullptr;
    }
    return &_entries[static_cast<std::size_t>(row)];
}

std::vector<Logger::Entry> LogEntryModel::parseLegacyText(const QByteArray& data, const std::string& currentSessionId)
{
    static const QRegularExpression linePattern(QStringLiteral("^\\[([^\\]]+)\\] \\[(DEBUG|INFO|WARN|ERROR)\\] (.*)$"));
    const QStringList lines = QString::fromUtf8(data).split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    std::vector<Logger::Entry> entries;
    entries.reserve(static_cast<std::size_t>(lines.size()));
    int session_number = 1;
    QString session = QStringLiteral("history-%1").arg(session_number);
    for (const QString& line : lines)
    {
        const QRegularExpressionMatch match = linePattern.match(line);
        if (!match.hasMatch())
        {
            continue;
        }
        if (match.captured(3).contains(QStringLiteral("PlaScan GUI started")) && !entries.empty())
        {
            session = QStringLiteral("history-%1").arg(++session_number);
        }

        Logger::Entry entry;
        entry.level = parseLevel(match.captured(2));
        entry.timestamp = match.captured(1).toStdString();
        entry.sessionId = session.toStdString();
        entry.message = match.captured(3).toStdString();
        entry.category = inferredCategory(match.captured(3)).toStdString();
        entry.formatted = (line + QLatin1Char('\n')).toStdString();
        entries.push_back(std::move(entry));
    }
    if (!entries.empty())
    {
        const std::string latest_session = entries.back().sessionId;
        for (Logger::Entry& entry : entries)
        {
            if (entry.sessionId == latest_session)
            {
                entry.sessionId = currentSessionId;
            }
        }
    }
    return entries;
}

LogFilterProxyModel::LogFilterProxyModel(QObject* parent) : QSortFilterProxyModel(parent)
{
    setDynamicSortFilter(true);
}

void LogFilterProxyModel::setMinimumLevel(Logger::Level level)
{
    _minimumLevel = level;
    refreshFilter();
}

void LogFilterProxyModel::setSessionId(const QString& sessionId)
{
    _sessionId = sessionId;
    refreshFilter();
}

void LogFilterProxyModel::setSearchText(const QString& text)
{
    _searchText = text.trimmed();
    refreshFilter();
}

void LogFilterProxyModel::setTaskRange(std::uint64_t firstSequence, std::uint64_t lastSequence, const QString& taskId)
{
    _firstSequence = firstSequence;
    _lastSequence = lastSequence;
    _taskId = taskId;
    refreshFilter();
}

void LogFilterProxyModel::clearTaskRange()
{
    setTaskRange(0, 0, QString());
}

void LogFilterProxyModel::refreshFilter()
{
#if QT_VERSION >= QT_VERSION_CHECK(6, 10, 0)
    beginFilterChange();
    endFilterChange(Direction::Rows);
#else
    invalidateFilter();
#endif
}

bool LogFilterProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const
{
    const QModelIndex level_index = sourceModel()->index(sourceRow, 0, sourceParent);
    if (level_index.data(LogEntryModel::LevelRole).toInt() < static_cast<int>(_minimumLevel))
    {
        return false;
    }
    if (!_sessionId.isEmpty() && level_index.data(LogEntryModel::SessionRole).toString() != _sessionId)
    {
        return false;
    }

    const qulonglong sequence = level_index.data(LogEntryModel::SequenceRole).toULongLong();
    if (_firstSequence != 0 && sequence < _firstSequence)
    {
        return false;
    }
    if (_lastSequence != 0 && sequence > _lastSequence)
    {
        return false;
    }
    if (!_taskId.isEmpty() && level_index.data(LogEntryModel::TaskIdRole).toString() != _taskId)
    {
        return false;
    }

    if (_searchText.isEmpty())
    {
        return true;
    }
    for (int column = 0; column < LogEntryModel::ColumnCount; ++column)
    {
        if (sourceModel()
                ->index(sourceRow, column, sourceParent)
                .data(Qt::DisplayRole)
                .toString()
                .contains(_searchText, Qt::CaseInsensitive))
        {
            return true;
        }
    }
    return false;
}
