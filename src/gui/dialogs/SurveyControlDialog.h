#pragma once

#include <QDialog>
#include <QJsonObject>

class QLabel;
class QPushButton;
class QTableWidget;

class SurveyControlDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SurveyControlDialog(QWidget *parent = nullptr);

    void setSurveyControlMetadata(const QJsonObject &metadata);

signals:
    void importCsvRequested();

public slots:
    void setStatusMessage(const QString &message);

private:
    void setupUi();
    void refreshTables();
    void populatePointTable(QTableWidget *table, const QJsonArray &points, bool includeResidual);
    void populateScaleBarTable(QTableWidget *table, const QJsonArray &scaleBars);

    QLabel *m_summaryLabel = nullptr;
    QLabel *m_sourceLabel = nullptr;
    QLabel *m_statusLabel = nullptr;
    QPushButton *m_importCsvButton = nullptr;
    QTableWidget *m_controlPointTable = nullptr;
    QTableWidget *m_checkPointTable = nullptr;
    QTableWidget *m_scaleBarTable = nullptr;
    QJsonObject m_metadata;
};
