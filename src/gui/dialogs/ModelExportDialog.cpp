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

    _formatCombo = form.m_formatCombo;
    _coordSysCombo = form.m_coordSysCombo;
    _includeTexCheck = form.m_includeTexCheck;
    _includeNormalCheck = form.m_includeNormalCheck;
    _includeColorCheck = form.m_includeColorCheck;
    _simplifyCheck = form.m_simplifyCheck;
    _simplifyRatioSpin = form.m_simplifyRatioSpin;
    _upAxisCombo = form.m_upAxisCombo;
    _outputPathEdit = form.m_outputPathEdit;
    _browseBtn = form.m_browseBtn;

    connect(_browseBtn, &QPushButton::clicked, this, &ModelExportDialog::onBrowseOutput);

    auto changed = [this]() { emitSettingsNow(); };
    connect(_formatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_coordSysCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_upAxisCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_includeTexCheck, &QCheckBox::toggled, this, changed);
    connect(_includeNormalCheck, &QCheckBox::toggled, this, changed);
    connect(_includeColorCheck, &QCheckBox::toggled, this, changed);
    connect(_simplifyCheck, &QCheckBox::toggled, this, changed);
    connect(_simplifyRatioSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);

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
        _outputPathEdit->setText(path);
    }
}

QJsonObject ModelExportDialog::collectSettings() const
{
    QJsonObject o;
    o["format"] = _formatCombo->currentText();
    o["coordSystem"] = _coordSysCombo->currentText();
    o["upAxis"] = _upAxisCombo->currentText();
    o["includeTexture"] = _includeTexCheck->isChecked();
    o["includeNormals"] = _includeNormalCheck->isChecked();
    o["includeColor"] = _includeColorCheck->isChecked();
    o["simplify"] = _simplifyCheck->isChecked();
    o["simplifyRatio"] = _simplifyRatioSpin->value();
    o["outputPath"] = _outputPathEdit->text();
    return o;
}

void ModelExportDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("format"))
    {
        const int i = _formatCombo->findText(s["format"].toString());
        if (i >= 0)
        {
            _formatCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("coordSystem"))
    {
        const int i = _coordSysCombo->findText(s["coordSystem"].toString());
        if (i >= 0)
        {
            _coordSysCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("upAxis"))
    {
        const int i = _upAxisCombo->findText(s["upAxis"].toString());
        if (i >= 0)
        {
            _upAxisCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("includeTexture"))
    {
        _includeTexCheck->setChecked(s["includeTexture"].toBool());
    }
    if (s.contains("includeNormals"))
    {
        _includeNormalCheck->setChecked(s["includeNormals"].toBool());
    }
    if (s.contains("includeColor"))
    {
        _includeColorCheck->setChecked(s["includeColor"].toBool());
    }
    if (s.contains("simplify"))
    {
        _simplifyCheck->setChecked(s["simplify"].toBool());
    }
    if (s.contains("simplifyRatio"))
    {
        _simplifyRatioSpin->setValue(s["simplifyRatio"].toDouble());
    }
    if (s.contains("outputPath"))
    {
        _outputPathEdit->setText(s["outputPath"].toString());
    }
}

void ModelExportDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void ModelExportDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
