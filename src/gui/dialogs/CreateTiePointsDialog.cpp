#include "CreateTiePointsDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QSizePolicy>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>

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
    edit->setMinimumHeight(28);
    edit->setMinimumWidth(240);
    edit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    return edit;
}

void stabilizeComboBox(QComboBox *combo)
{
    if (!combo)
    {
        return;
    }
    combo->setMinimumHeight(28);
    combo->setMinimumWidth(240);
    combo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void stabilizeCheckBox(QCheckBox *checkBox)
{
    if (!checkBox)
    {
        return;
    }
    checkBox->setMinimumHeight(24);
    checkBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

} // namespace

CreateTiePointsDialog::CreateTiePointsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("创建连接点"));
    setMinimumWidth(560);
    buildUi();
}

QString CreateTiePointsDialog::accuracy() const
{
    return _accuracyCombo ? _accuracyCombo->currentData().toString() : QStringLiteral("highest");
}

int CreateTiePointsDialog::keypointLimit() const
{
    if (useGuidedMatching())
    {
        return _keypointLimit;
    }
    return intFromEdit(_keypointLimitEdit, 40000);
}

int CreateTiePointsDialog::keypointLimitPerMegapixel() const
{
    if (useGuidedMatching())
    {
        return intFromEdit(_keypointLimitEdit, 1000);
    }
    return _keypointLimitPerMegapixel;
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

QString CreateTiePointsDialog::maskApplyMode() const
{
    return _maskModeCombo ? _maskModeCombo->currentData().toString() : QStringLiteral("none");
}

void CreateTiePointsDialog::setReferencePreselectionAvailable(bool available,
                                                              int cameraCount,
                                                              int imageCount)
{
    if (_referencePreselectionCheck)
    {
        _referencePreselectionCheck->setEnabled(available);
        _referencePreselectionCheck->setChecked(false);
        _referencePreselectionCheck->setToolTip(
            available
                ? tr("使用已导入相机模型/外参生成候选匹配对。")
                : tr("当前项目没有完整可用的相机参考，只能使用通用预选或全量两两匹配。"));
    }

    if (_preselectionStatusLabel)
    {
        _preselectionStatusLabel->setText(
            available
                ? tr("已检测到 %1/%2 个相机参考，可启用参考预选。").arg(cameraCount).arg(imageCount)
                : tr("未检测到完整相机参考；参考预选不可用。"));
        _preselectionStatusLabel->setVisible(false);
    }
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

QString CreateTiePointsDialog::formattedInteger(int value) const
{
    QString number = QString::number(std::max(0, value));
    for (int pos = number.size() - 3; pos > 0; pos -= 3)
    {
        number.insert(pos, QLatin1Char(','));
    }
    return number;
}

void CreateTiePointsDialog::updateKeypointLimitMode(bool guided)
{
    if (!_keypointLimitEdit || !_keypointLimitLabel)
    {
        return;
    }

    if (guided)
    {
        _keypointLimit = intFromEdit(_keypointLimitEdit, _keypointLimit);
        _keypointLimitLabel->setText(tr("每百万像素的关键点限制:"));
        _keypointLimitEdit->setText(formattedInteger(_keypointLimitPerMegapixel));
    }
    else
    {
        _keypointLimitPerMegapixel =
            intFromEdit(_keypointLimitEdit, _keypointLimitPerMegapixel);
        _keypointLimitLabel->setText(tr("关键点限制:"));
        _keypointLimitEdit->setText(formattedInteger(_keypointLimit));
    }
}

void CreateTiePointsDialog::setAdvancedExpanded(bool expanded)
{
    if (!_advancedToggle || !_advancedContent)
    {
        return;
    }

    _advancedToggle->setChecked(expanded);
    _advancedToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    _advancedContent->setVisible(expanded);
    if (_advancedGroup)
    {
        _advancedGroup->setVisible(expanded);
    }
    if (layout())
    {
        layout()->invalidate();
        layout()->activate();
    }

    setMinimumSize(QSize(0, 0));
    setMaximumSize(QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX));
    const QSize nextSize = sizeHint();
    setFixedSize(nextSize);
}

void CreateTiePointsDialog::buildUi()
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(18, 16, 18, 14);
    mainLayout->setSpacing(12);
    mainLayout->setSizeConstraint(QLayout::SetFixedSize);

    _generalGroup = new QGroupBox(tr("一般"), this);
    _generalGroup->setObjectName(QStringLiteral("m_generalGroup"));
    _generalGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    auto *generalLayout = new QFormLayout(_generalGroup);
    generalLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    generalLayout->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    generalLayout->setContentsMargins(16, 14, 16, 14);
    generalLayout->setHorizontalSpacing(24);
    generalLayout->setVerticalSpacing(8);

    _accuracyCombo = new QComboBox(_generalGroup);
    _accuracyCombo->setObjectName(QStringLiteral("m_accuracyCombo"));
    _accuracyCombo->addItem(tr("最低"), QStringLiteral("lowest"));
    _accuracyCombo->addItem(tr("低"), QStringLiteral("low"));
    _accuracyCombo->addItem(tr("中"), QStringLiteral("medium"));
    _accuracyCombo->addItem(tr("高"), QStringLiteral("high"));
    _accuracyCombo->addItem(tr("最高"), QStringLiteral("highest"));
    _accuracyCombo->setCurrentIndex(_accuracyCombo->findData(QStringLiteral("highest")));
    stabilizeComboBox(_accuracyCombo);
    generalLayout->addRow(tr("精度:"), _accuracyCombo);

    _genericPreselectionCheck = new QCheckBox(tr("通用预选"), _generalGroup);
    _genericPreselectionCheck->setObjectName(QStringLiteral("m_genericPreselectionCheck"));
    _genericPreselectionCheck->setChecked(true);
    stabilizeCheckBox(_genericPreselectionCheck);
    generalLayout->addRow(QString(), _genericPreselectionCheck);

    _referencePreselectionCheck = new QCheckBox(tr("参考预选"), _generalGroup);
    _referencePreselectionCheck->setObjectName(QStringLiteral("m_referencePreselectionCheck"));
    _referencePreselectionCheck->setEnabled(false);
    _referencePreselectionCheck->setToolTip(
        tr("当前项目没有完整可用的相机参考，只能使用通用预选或全量两两匹配。"));
    stabilizeCheckBox(_referencePreselectionCheck);
    generalLayout->addRow(QString(), _referencePreselectionCheck);

    _preselectionStatusLabel = new QLabel(_generalGroup);
    _preselectionStatusLabel->setObjectName(QStringLiteral("m_preselectionStatusLabel"));
    _preselectionStatusLabel->setWordWrap(true);
    _preselectionStatusLabel->setText(tr("未检测到完整相机参考；参考预选不可用。"));
    _preselectionStatusLabel->setVisible(false);
    generalLayout->addRow(QString(), _preselectionStatusLabel);
    mainLayout->addWidget(_generalGroup);

    auto *advancedHeader = new QWidget(this);
    advancedHeader->setObjectName(QStringLiteral("m_advancedHeader"));
    auto *advancedHeaderLayout = new QHBoxLayout(advancedHeader);
    advancedHeaderLayout->setContentsMargins(28, 0, 28, 0);
    advancedHeaderLayout->setSpacing(6);

    _advancedToggle = new QToolButton(advancedHeader);
    _advancedToggle->setObjectName(QStringLiteral("m_advancedToggle"));
    _advancedToggle->setText(tr("高级"));
    _advancedToggle->setCheckable(true);
    _advancedToggle->setChecked(true);
    _advancedToggle->setAutoRaise(true);
    _advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    _advancedToggle->setArrowType(Qt::DownArrow);
    advancedHeaderLayout->addWidget(_advancedToggle, 0, Qt::AlignLeft);

    auto *advancedLine = new QFrame(advancedHeader);
    advancedLine->setFrameShape(QFrame::HLine);
    advancedLine->setFrameShadow(QFrame::Sunken);
    advancedHeaderLayout->addWidget(advancedLine, 1);
    mainLayout->addWidget(advancedHeader);

    _advancedGroup = new QGroupBox(this);
    _advancedGroup->setObjectName(QStringLiteral("m_advancedGroup"));
    _advancedGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    auto *advancedOuterLayout = new QVBoxLayout(_advancedGroup);
    advancedOuterLayout->setContentsMargins(16, 14, 16, 14);

    _advancedContent = new QWidget(_advancedGroup);
    _advancedContent->setObjectName(QStringLiteral("m_advancedContent"));
    auto *advancedLayout = new QFormLayout(_advancedContent);
    advancedLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    advancedLayout->setLabelAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    advancedLayout->setContentsMargins(0, 0, 8, 0);
    advancedLayout->setHorizontalSpacing(24);
    advancedLayout->setVerticalSpacing(8);

    _keypointLimitLabel = new QLabel(tr("关键点限制:"), _advancedContent);
    _keypointLimitLabel->setObjectName(QStringLiteral("m_keypointLimitLabel"));
    _keypointLimitEdit = makeIntegerEdit(_advancedContent, QStringLiteral("40,000"));
    _keypointLimitEdit->setObjectName(QStringLiteral("m_keypointLimitEdit"));
    advancedLayout->addRow(_keypointLimitLabel, _keypointLimitEdit);

    _tiePointLimitEdit = makeIntegerEdit(_advancedContent, QStringLiteral("4,000"));
    _tiePointLimitEdit->setObjectName(QStringLiteral("m_tiePointLimitEdit"));
    advancedLayout->addRow(tr("连接点限制:"), _tiePointLimitEdit);

    _maskModeCombo = new QComboBox(_advancedContent);
    _maskModeCombo->setObjectName(QStringLiteral("m_maskModeCombo"));
    _maskModeCombo->addItem(tr("无"), QStringLiteral("none"));
    _maskModeCombo->addItem(tr("关键点"), QStringLiteral("keypoints"));
    _maskModeCombo->addItem(tr("连接点"), QStringLiteral("tiepoints"));
    _maskModeCombo->setToolTip(
        tr("使用项目蒙版约束连接点流程：0 为有效区域，非 0 为排除区域。"));
    stabilizeComboBox(_maskModeCombo);
    advancedLayout->addRow(tr("将掩膜应用于:"), _maskModeCombo);

    _guidedMatchingCheck = new QCheckBox(tr("指导图像匹配"), _advancedContent);
    _guidedMatchingCheck->setObjectName(QStringLiteral("m_guidedMatchingCheck"));
    stabilizeCheckBox(_guidedMatchingCheck);
    connect(_guidedMatchingCheck,
            &QCheckBox::toggled,
            this,
            &CreateTiePointsDialog::updateKeypointLimitMode);
    advancedLayout->addRow(QString(), _guidedMatchingCheck);

    _excludePinnedTiePointsCheck = new QCheckBox(tr("不包括固定的连接点"), _advancedContent);
    _excludePinnedTiePointsCheck->setObjectName(QStringLiteral("m_excludePinnedTiePointsCheck"));
    _excludePinnedTiePointsCheck->setChecked(true);
    stabilizeCheckBox(_excludePinnedTiePointsCheck);
    advancedLayout->addRow(QString(), _excludePinnedTiePointsCheck);
    advancedOuterLayout->addWidget(_advancedContent);
    mainLayout->addWidget(_advancedGroup);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    setEnglishDialogButtons(buttonBox);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(_advancedToggle, &QToolButton::toggled, this, [this](bool expanded)
    {
        setAdvancedExpanded(expanded);
    });

    auto *buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(buttonBox);
    buttonLayout->addStretch(1);
    mainLayout->addLayout(buttonLayout);
    setAdvancedExpanded(true);
}
