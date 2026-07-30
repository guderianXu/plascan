#include "tie_points/ThinTiePointsDialog.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QVBoxLayout>

ThinTiePointsDialog::ThinTiePointsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("稀释连接点"));
    setMinimumWidth(480);
    buildUi();
}

int ThinTiePointsDialog::tiePointLimit() const
{
    if (!_tiePointLimitEdit)
    {
        return 500;
    }

    QString text = _tiePointLimitEdit->text();
    text.remove(QLatin1Char(','));
    text.remove(QLatin1Char(' '));

    bool ok = false;
    const int value = text.toInt(&ok);
    return ok ? value : 500;
}

void ThinTiePointsDialog::buildUi()
{
    auto *mainLayout = new QVBoxLayout(this);

    auto *generalGroup = new QGroupBox(tr("一般"), this);
    auto *formLayout = new QFormLayout(generalGroup);
    formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    _tiePointLimitEdit = new QLineEdit(QStringLiteral("500"), generalGroup);
    _tiePointLimitEdit->setValidator(
        new QRegularExpressionValidator(QRegularExpression(QStringLiteral("[0-9, ]+")), _tiePointLimitEdit));
    formLayout->addRow(tr("连接点限制:"), _tiePointLimitEdit);
    mainLayout->addWidget(generalGroup);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    if (QPushButton *okButton = buttonBox->button(QDialogButtonBox::Ok))
    {
        okButton->setText(QStringLiteral("OK"));
    }
    if (QPushButton *cancelButton = buttonBox->button(QDialogButtonBox::Cancel))
    {
        cancelButton->setText(QStringLiteral("Cancel"));
    }
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(buttonBox);
}
