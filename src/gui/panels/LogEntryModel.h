#pragma once

#include "Logger.h"

#include <QAbstractTableModel>
#include <QSortFilterProxyModel>

#include <cstdint>
#include <vector>

class LogEntryModel final : public QAbstractTableModel
{
    Q_OBJECT

public:
    enum Column
    {
        TimeColumn,
        LevelColumn,
        ContextColumn,
        MessageColumn,
        ColumnCount
    };

    enum Role
    {
        LevelRole = Qt::UserRole + 1,
        SessionRole,
        SequenceRole,
        TaskIdRole,
        FormattedRole
    };

    explicit LogEntryModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

    void appendEntries(const std::vector<Logger::Entry>& entries);
    void clear();
    const Logger::Entry* entryAt(int row) const;
    static std::vector<Logger::Entry> parseLegacyText(const QByteArray& data, const std::string& currentSessionId);

private:
    std::vector<Logger::Entry> _entries;
    std::uint64_t _lastLiveSequence{0};
    int _maximumEntries{20000};
};

class LogFilterProxyModel final : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    explicit LogFilterProxyModel(QObject* parent = nullptr);

    void setMinimumLevel(Logger::Level level);
    void setSessionId(const QString& sessionId);
    void setSearchText(const QString& text);
    void setTaskRange(std::uint64_t firstSequence, std::uint64_t lastSequence, const QString& taskId);
    void clearTaskRange();

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex& sourceParent) const override;

private:
    void refreshFilter();

    Logger::Level _minimumLevel{Logger::Info};
    QString _sessionId;
    QString _searchText;
    QString _taskId;
    std::uint64_t _firstSequence{0};
    std::uint64_t _lastSequence{0};
};
