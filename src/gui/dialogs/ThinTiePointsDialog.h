#pragma once

#include <QDialog>

class QLineEdit;

class ThinTiePointsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ThinTiePointsDialog(QWidget *parent = nullptr);

    int tiePointLimit() const;

private:
    void buildUi();

    QLineEdit *_tiePointLimitEdit{};
};
