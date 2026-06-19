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

private:
    void setupUi();
    void updateTables(const QJsonObject &meta);
    void updateTaskTable();

    QLabel *m_summaryLabel = nullptr;
    QLabel *m_referenceLabel = nullptr;
    QLabel *m_taskLabel = nullptr;
    QTableWidget *m_taskTable = nullptr;
    QTableWidget *m_referenceTable = nullptr;
    QTableWidget *m_workflowTable = nullptr;
    QTableWidget *m_qualityTable = nullptr;
    QTableWidget *m_qualityAlertTable = nullptr;
    QTableWidget *m_reportTable = nullptr;
    QJsonArray m_taskSnapshots;
};
