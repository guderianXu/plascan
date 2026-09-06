#pragma once

#include "Logger.h"

#include <QSize>
#include <QWidget>

#include <memory>
#include <mutex>
#include <vector>

class LogEntryModel;
class LogFilterProxyModel;
class QComboBox;
class QLabel;
class QLineEdit;
class QShowEvent;
class QTableView;
class QToolButton;

class LogPanel final : public QWidget
{
    Q_OBJECT

public:
    explicit LogPanel(QWidget* parent = nullptr);
    ~LogPanel() override;

    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;
    void append(const QString& text);
    void loadFromLogFile();

public slots:
    void appendLog(const QString& formatted, int level);
    void clearLogs();
    bool saveLogsToFile(const QString& filePath);
    void focusLogRange(qulonglong firstSequence, qulonglong lastSequence, const QString& taskId);

signals:
    void unreadCountsChanged(int warningCount, int errorCount);

protected:
    void showEvent(QShowEvent* event) override;

private:
    struct PendingQueue
    {
        std::mutex mutex;
        std::vector<Logger::Entry> entries;
        bool drainScheduled{false};
    };

    void drainPendingEntries();
    void appendEntries(const std::vector<Logger::Entry>& entries);
    void updateCountLabel();
    void scrollToLatest();
    void markRead();
    void deleteDiskHistory();

    LogEntryModel* _model{};
    LogFilterProxyModel* _proxy{};
    QTableView* _table{};
    QComboBox* _sessionCombo{};
    QComboBox* _levelCombo{};
    QLineEdit* _searchEdit{};
    QLabel* _countLabel{};
    QToolButton* _followButton{};
    QToolButton* _clearButton{};
    QToolButton* _saveButton{};
    QToolButton* _openDirectoryButton{};
    std::shared_ptr<PendingQueue> _pendingQueue;
    int _sinkId{0};
    int _unreadWarnings{0};
    int _unreadErrors{0};
};
