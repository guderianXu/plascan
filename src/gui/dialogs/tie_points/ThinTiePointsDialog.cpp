#include "tie_points/ThinTiePointsDialog.h"

#include "shared/WorkflowParameterDialogStyle.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLineEdit>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QVBoxLayout>

ThinTiePointsDialog::ThinTiePointsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("稀释连接点"));
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);
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
    xjw::gui::dialogs::configureWorkflowForm(formLayout);

    _tiePointLimitEdit = new QLineEdit(QStringLiteral("500"), generalGroup);
    _tiePointLimitEdit->setValidator(
        new QRegularExpressionValidator(QRegularExpression(QStringLiteral("[0-9, ]+")), _tiePointLimitEdit));
    xjw::gui::dialogs::configureWorkflowInputWidget(_tiePointLimitEdit);
    formLayout->addRow(tr("连接点限制:"), _tiePointLimitEdit);
    mainLayout->addWidget(generalGroup);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttonBox->setObjectName(QStringLiteral("workflowButtonBox"));
    xjw::gui::dialogs::configureWorkflowButtonBox(buttonBox, tr("稀释"));
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(buttonBox);
}
