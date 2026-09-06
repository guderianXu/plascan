#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QWidget>

class QLabel;
class QTableWidget;

class ProjectDashboardWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ProjectDashboardWidget(QWidget *parent = nullptr);
    ~ProjectDashboardWidget() override;

public slots:
    void loadFromJson(const QJsonObject &meta);
    void setTaskSnapshots(const QJsonArray &tasks);
    void clear();

signals:
    void taskSnapshotsChanged(const QJsonArray &tasks);

private:
    void setupUi();
    void updateTables(const QJsonObject &meta);
    void updateTaskSummary();

    QLabel *_summaryLabel = nullptr;
    QLabel *_referenceLabel = nullptr;
    QLabel* _taskLabel = nullptr;
    QTableWidget *_referenceTable = nullptr;
    QTableWidget *_workflowTable = nullptr;
    QTableWidget *_qualityTable = nullptr;
    QTableWidget *_qualityAlertTable = nullptr;
    QTableWidget *_reportTable = nullptr;
    QJsonArray _taskSnapshots;
};
