#pragma once

#include <QDialog>
#include <QJsonArray>

class ProjectManager;
class QComboBox;
class QTableWidget;

class ForwardIntersectionResultsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ForwardIntersectionResultsDialog(ProjectManager *projectManager, QWidget *parent = nullptr);
    ~ForwardIntersectionResultsDialog() override;

private slots:
    void onPairChanged();
    void onRowChanged(int row, int column);

private:
    void setupUi();
    void loadResults();
    void fillTableForPair(const QString &pairKey);
    void fillDetailTable(const QJsonObject &batchResult);
    QString makePairKey(const QJsonObject &result) const;

    ProjectManager *_projectManager{};
    QComboBox *_pairCombo{};
    QTableWidget *_table{};
    QTableWidget *_detailTable{};
    QJsonArray _allResults;
};
