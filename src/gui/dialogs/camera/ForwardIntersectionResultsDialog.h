#pragma once

#include <QDialog>
#include <QJsonArray>

class QComboBox;
class QTableWidget;

namespace xjw::gui::project
{
class ProjectSession;
}

class ForwardIntersectionResultsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ForwardIntersectionResultsDialog(xjw::gui::project::ProjectSession *session,
                                              QWidget *parent = nullptr);
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

    xjw::gui::project::ProjectSession *_session{};
    QComboBox *_pairCombo{};
    QTableWidget *_table{};
    QTableWidget *_detailTable{};
    QJsonArray _allResults;
};
