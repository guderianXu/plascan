#pragma once

#include <QDialog>
#include <QPointer>

class QComboBox;
class QLabel;
class QPushButton;
class QTableWidget;

namespace xjw::gui::markers
{

class MarkerWorkspaceController;

class MarkerDetectionReviewDialog final : public QDialog
{
    Q_OBJECT

public:
    explicit MarkerDetectionReviewDialog(QWidget *parent = nullptr);

    bool setController(MarkerWorkspaceController *controller);

private slots:
    void refresh();
    void updateSelection();
    void acceptSelected();
    void discardSelected();

private:
    QString selectedEntryId() const;
    void refreshAssignmentOptions();
    void updatePreview();
    void showOperationError(const QString &message);

    QPointer<MarkerWorkspaceController> _controller;
    QTableWidget *_table = nullptr;
    QLabel *_preview = nullptr;
    QLabel *_details = nullptr;
    QComboBox *_assignment = nullptr;
    QPushButton *_accept = nullptr;
    QPushButton *_discard = nullptr;
};

} // namespace xjw::gui::markers
