#include "CleanTiePointsDialog.h"

#include <QComboBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>

CleanTiePointsDialog::CleanTiePointsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(QStringLiteral("Clean Tie Points"));
    setMinimumWidth(480);
    buildUi();
    updateCriterionState();
}

CleanTiePointsDialog::Criterion CleanTiePointsDialog::criterion() const
{
    if (!_criterionCombo)
    {
        return Criterion::None;
    }

    return static_cast<Criterion>(_criterionCombo->currentData().toInt());
}

QString CleanTiePointsDialog::criterionText() const
{
    return _criterionCombo ? _criterionCombo->currentText() : QString();
}

double CleanTiePointsDialog::level() const
{
    if (!_levelEdit)
    {
        return 0.0;
    }

    bool ok = false;
    const double value = _levelEdit->text().trimmed().toDouble(&ok);
    return ok ? value : 0.0;
}

bool CleanTiePointsDialog::deleteRequested() const
{
    return _deleteRequested;
}

void CleanTiePointsDialog::buildUi()
{
    auto *mainLayout = new QVBoxLayout(this);

    auto *generalGroup = new QGroupBox(tr("一般"), this);
    auto *formLayout = new QFormLayout(generalGroup);
    formLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

    _criterionCombo = new QComboBox(generalGroup);
    _criterionCombo->addItem(tr("请选择..."), static_cast<int>(Criterion::None));
    _criterionCombo->addItem(tr("重投影误差"), static_cast<int>(Criterion::ReprojectionError));
    _criterionCombo->addItem(tr("重建不确定性"), static_cast<int>(Criterion::ReconstructionUncertainty));
    _criterionCombo->addItem(tr("图像计数"), static_cast<int>(Criterion::ImageCount));
    _criterionCombo->addItem(tr("投影精度"), static_cast<int>(Criterion::ProjectionAccuracy));
    formLayout->addRow(tr("标准:"), _criterionCombo);

    _levelEdit = new QLineEdit(generalGroup);
    formLayout->addRow(tr("级别:"), _levelEdit);
    mainLayout->addWidget(generalGroup);

    _levelSlider = new QSlider(Qt::Horizontal, this);
    _levelSlider->setRange(0, 100);
    _levelSlider->setValue(0);
    mainLayout->addWidget(_levelSlider);

    auto *buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch(1);
    _okButton = new QPushButton(QStringLiteral("OK"), this);
    _deleteButton = new QPushButton(tr("删除"), this);
    _cancelButton = new QPushButton(QStringLiteral("Cancel"), this);
    buttonLayout->addWidget(_okButton);
    buttonLayout->addWidget(_deleteButton);
    buttonLayout->addWidget(_cancelButton);
    buttonLayout->addStretch(1);
    mainLayout->addLayout(buttonLayout);

    connect(_criterionCombo, &QComboBox::currentIndexChanged, this, &CleanTiePointsDialog::updateCriterionState);
    connect(_okButton, &QPushButton::clicked, this, [this]()
    {
        _deleteRequested = false;
        accept();
    });
    connect(_deleteButton, &QPushButton::clicked, this, [this]()
    {
        _deleteRequested = true;
        accept();
    });
    connect(_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
}

void CleanTiePointsDialog::updateCriterionState()
{
    const bool hasCriterion = criterion() != Criterion::None;
    if (_levelEdit)
    {
        _levelEdit->setEnabled(hasCriterion);
        if (hasCriterion && _levelEdit->text().trimmed().isEmpty())
        {
            _levelEdit->setText(QStringLiteral("1.0"));
        }
        if (!hasCriterion)
        {
            _levelEdit->clear();
        }
    }
    if (_levelSlider)
    {
        _levelSlider->setEnabled(hasCriterion);
    }
    if (_okButton)
    {
        _okButton->setEnabled(hasCriterion);
    }
    if (_deleteButton)
    {
        _deleteButton->setEnabled(hasCriterion);
    }
}
