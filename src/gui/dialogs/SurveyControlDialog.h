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

    QLabel *_summaryLabel = nullptr;
    QLabel *_sourceLabel = nullptr;
    QLabel *_statusLabel = nullptr;
    QPushButton *_importCsvButton = nullptr;
    QTableWidget *_controlPointTable = nullptr;
    QTableWidget *_checkPointTable = nullptr;
    QTableWidget *_scaleBarTable = nullptr;
    QJsonObject _metadata;
};
