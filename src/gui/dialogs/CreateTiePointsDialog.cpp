#include "CreateTiePointsDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QVBoxLayout>

namespace
{

void setEnglishDialogButtons(QDialogButtonBox *buttonBox)
{
    if (!buttonBox)
    {
        return;
    }

    if (QPushButton *okButton = buttonBox->button(QDialogButtonBox::Ok))
    {
        okButton->setText(QStringLiteral("OK"));
    }
    if (QPushButton *cancelButton = buttonBox->button(QDialogButtonBox::Cancel))
    {
        cancelButton->setText(QStringLiteral("Cancel"));
    }
}

QLineEdit *makeIntegerEdit(QWidget *parent, const QString &text)
{
    auto *edit = new QLineEdit(text, parent);
    edit->setValidator(new QRegularExpressionValidator(QRegularExpression(QStringLiteral("[0-9, ]+")), edit));
    edit->setMinimumWidth(200);
    return edit;
}

} // namespace

CreateTiePointsDialog::CreateTiePointsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("创建连接点"));
    setMinimumWidth(480);
    buildUi();
}

QString CreateTiePointsDialog::accuracy() const
{
    return _accuracyCombo ? _accuracyCombo->currentData().toString() : QStringLiteral("highest");
}

int CreateTiePointsDialog::keypointLimit() const
{
    return intFromEdit(_keypointLimitEdit, 40000);
}

int CreateTiePointsDialog::tiePointLimit() const
{
    return intFromEdit(_tiePointLimitEdit, 4000);
}

bool CreateTiePointsDialog::useGenericPreselection() const
{
    return _genericPreselectionCheck && _genericPreselectionCheck->isChecked();
}

bool CreateTiePointsDialog::useReferencePreselection() const
{
    return _referencePreselectionCheck && _referencePreselectionCheck->isChecked();
}

bool CreateTiePointsDialog::useGuidedMatching() const
{
    return _guidedMatchingCheck && _guidedMatchingCheck->isChecked();
}

bool CreateTiePointsDialog::excludePinnedTiePoints() const
{
    return _excludePinnedTiePointsCheck && _excludePinnedTiePointsCheck->isChecked();
}

int CreateTiePointsDialog::intFromEdit(const QLineEdit *edit, int fallback) const
{
    if (!edit)
    {
        return fallback;
    }

    QString text = edit->text();
    text.remove(QLatin1Char(','));
    text.remove(QLatin1Char(' '));

    bool ok = false;
    const int value = text.toInt(&ok);
    return ok ? value : fallback;
}

void CreateTiePointsDialog::buildUi()
{
    auto *mainLayout = new QVBoxLayout(this);

    auto *generalGroup = new QGroupBox(tr("一般"), this);
    auto *generalLayout = new QFormLayout(generalGroup);
    generalLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    _accuracyCombo = new QComboBox(generalGroup);
    _accuracyCombo->addItem(tr("最低"), QStringLiteral("lowest"));
    _accuracyCombo->addItem(tr("低"), QStringLiteral("low"));
    _accuracyCombo->addItem(tr("中"), QStringLiteral("medium"));
    _accuracyCombo->addItem(tr("高"), QStringLiteral("high"));
    _accuracyCombo->addItem(tr("最高"), QStringLiteral("highest"));
    _accuracyCombo->setCurrentIndex(_accuracyCombo->findData(QStringLiteral("highest")));
    generalLayout->addRow(tr("精度:"), _accuracyCombo);

    _genericPreselectionCheck = new QCheckBox(tr("通用预选"), generalGroup);
    _genericPreselectionCheck->setChecked(true);
    generalLayout->addRow(QString(), _genericPreselectionCheck);

    _referencePreselectionCheck = new QCheckBox(tr("参考预选"), generalGroup);
    _referencePreselectionCheck->setEnabled(false);
    generalLayout->addRow(QString(), _referencePreselectionCheck);
    mainLayout->addWidget(generalGroup);

    auto *advancedGroup = new QGroupBox(tr("高级"), this);
    auto *advancedLayout = new QFormLayout(advancedGroup);
    advancedLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    _keypointLimitEdit = makeIntegerEdit(advancedGroup, QStringLiteral("40,000"));
    advancedLayout->addRow(tr("关键点限制:"), _keypointLimitEdit);

    _tiePointLimitEdit = makeIntegerEdit(advancedGroup, QStringLiteral("4,000"));
    advancedLayout->addRow(tr("连接点限制:"), _tiePointLimitEdit);

    _maskModeCombo = new QComboBox(advancedGroup);
    _maskModeCombo->addItem(tr("无"), QStringLiteral("none"));
    _maskModeCombo->setEnabled(false);
    advancedLayout->addRow(tr("将掩膜应用于:"), _maskModeCombo);

    _guidedMatchingCheck = new QCheckBox(tr("指导图像匹配"), advancedGroup);
    advancedLayout->addRow(QString(), _guidedMatchingCheck);

    _excludePinnedTiePointsCheck = new QCheckBox(tr("不包括固定的连接点"), advancedGroup);
    _excludePinnedTiePointsCheck->setChecked(true);
    advancedLayout->addRow(QString(), _excludePinnedTiePointsCheck);
    mainLayout->addWidget(advancedGroup);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    setEnglishDialogButtons(buttonBox);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(buttonBox);
}
