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

    ProjectManager *m_projectManager{};
    QComboBox *m_pairCombo{};
    QTableWidget *m_table{};
    QTableWidget *m_detailTable{};
    QJsonArray m_allResults;
};
