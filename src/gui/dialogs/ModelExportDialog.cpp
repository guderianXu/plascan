#include "ModelExportDialog.h"
#include "ui_ModelExportDialog.h"

#include <QComboBox>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>

ModelExportDialog::ModelExportDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::ModelExportDialog form;
    form.setupUi(this);

    m_formatCombo        = form.m_formatCombo;
    m_coordSysCombo      = form.m_coordSysCombo;
    m_includeTexCheck    = form.m_includeTexCheck;
    m_includeNormalCheck = form.m_includeNormalCheck;
    m_includeColorCheck  = form.m_includeColorCheck;
    m_simplifyCheck      = form.m_simplifyCheck;
    m_simplifyRatioSpin  = form.m_simplifyRatioSpin;
    m_upAxisCombo        = form.m_upAxisCombo;
    m_outputPathEdit     = form.m_outputPathEdit;
    m_browseBtn          = form.m_browseBtn;

    connect(m_browseBtn, &QPushButton::clicked, this, &ModelExportDialog::onBrowseOutput);

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_formatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_coordSysCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_upAxisCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_includeTexCheck, &QCheckBox::toggled, this, changed);
    connect(m_includeNormalCheck, &QCheckBox::toggled, this, changed);
    connect(m_includeColorCheck, &QCheckBox::toggled, this, changed);
    connect(m_simplifyCheck, &QCheckBox::toggled, this, changed);
    connect(m_simplifyRatioSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &ModelExportDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

void ModelExportDialog::onBrowseOutput()
{
    QString path = QFileDialog::getSaveFileName(
        this, tr("选择导出路径"), QString(),
        tr("所有文件 (*)"));
    if (!path.isEmpty())
    {
        m_outputPathEdit->setText(path);
    }
}

QJsonObject ModelExportDialog::collectSettings() const
{
    QJsonObject o;
    o["format"]       = m_formatCombo->currentText();
    o["coordSystem"]  = m_coordSysCombo->currentText();
    o["upAxis"]       = m_upAxisCombo->currentText();
    o["includeTexture"] = m_includeTexCheck->isChecked();
    o["includeNormals"] = m_includeNormalCheck->isChecked();
    o["includeColor"]   = m_includeColorCheck->isChecked();
    o["simplify"]       = m_simplifyCheck->isChecked();
    o["simplifyRatio"]  = m_simplifyRatioSpin->value();
    o["outputPath"]     = m_outputPathEdit->text();
    return o;
}

void ModelExportDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("format"))
    {
        int i = m_formatCombo->findText(s["format"].toString());
        if (i >= 0) m_formatCombo->setCurrentIndex(i);
    }
    if (s.contains("coordSystem"))
    {
        int i = m_coordSysCombo->findText(s["coordSystem"].toString());
        if (i >= 0) m_coordSysCombo->setCurrentIndex(i);
    }
    if (s.contains("upAxis"))
    {
        int i = m_upAxisCombo->findText(s["upAxis"].toString());
        if (i >= 0) m_upAxisCombo->setCurrentIndex(i);
    }
    if (s.contains("includeTexture")) m_includeTexCheck->setChecked(s["includeTexture"].toBool());
    if (s.contains("includeNormals")) m_includeNormalCheck->setChecked(s["includeNormals"].toBool());
    if (s.contains("includeColor"))   m_includeColorCheck->setChecked(s["includeColor"].toBool());
    if (s.contains("simplify"))       m_simplifyCheck->setChecked(s["simplify"].toBool());
    if (s.contains("simplifyRatio"))  m_simplifyRatioSpin->setValue(s["simplifyRatio"].toDouble());
    if (s.contains("outputPath"))     m_outputPathEdit->setText(s["outputPath"].toString());
}

void ModelExportDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void ModelExportDialog::onRun() { emit runRequested(collectSettings()); accept(); }
