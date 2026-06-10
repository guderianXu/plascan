#include "MeshReconstructionDialog.h"
#include "ui_MeshReconstructionDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>

MeshReconstructionDialog::MeshReconstructionDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::MeshReconstructionDialog form;
    form.setupUi(this);

    m_denseCloudCombo     = form.m_denseCloudCombo;
    m_browseDenseBtn      = form.m_browseDenseBtn;
    m_methodCombo         = form.m_methodCombo;
    m_outputFormatCombo   = form.m_outputFormatCombo;
    m_qualityProfileCombo = form.m_qualityProfileCombo;
    m_octreeDepthSpin     = form.m_octreeDepthSpin;
    m_meshResSpin         = form.m_meshResSpin;
    m_smoothIterSpin      = form.m_smoothIterSpin;
    m_holeFillCheck       = form.m_holeFillCheck;
    m_maxHoleSizeSpin     = form.m_maxHoleSizeSpin;
    m_cleanCheck          = form.m_cleanCheck;
    m_minFacesSpin        = form.m_minFacesSpin;
    m_voxelDensityCombo   = form.m_voxelDensityCombo;
    m_decimateCheck       = form.m_decimateCheck;
    m_decimateRatioSpin   = form.m_decimateRatioSpin;
    m_threadsSpin         = form.m_threadsSpin;
    m_infoLabel           = form.m_infoLabel;

    m_qualityProfileCombo->setItemData(0, QStringLiteral("detail"));
    m_qualityProfileCombo->setItemData(1, QStringLiteral("balanced"));
    m_qualityProfileCombo->setItemData(2, QStringLiteral("lite"));
    m_voxelDensityCombo->setItemData(0, QStringLiteral("coarse"));
    m_voxelDensityCombo->setItemData(1, QStringLiteral("medium"));
    m_voxelDensityCombo->setItemData(2, QStringLiteral("fine"));

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_denseCloudCombo, &QComboBox::editTextChanged, this, changed);
    connect(m_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_qualityProfileCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_octreeDepthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_meshResSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_smoothIterSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_holeFillCheck, &QCheckBox::toggled, this, changed);
    connect(m_cleanCheck, &QCheckBox::toggled, this, changed);
    connect(m_voxelDensityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_decimateCheck, &QCheckBox::toggled, this, changed);
    connect(m_decimateRatioSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(m_browseDenseBtn, &QPushButton::clicked, this, [this]() {
        const QString initialPath = m_denseCloudCombo->currentText().trimmed();
        const QString initialDir = initialPath.isEmpty()
            ? QDir::homePath()
            : QFileInfo(initialPath).absolutePath();

        const QString chosenPath = QFileDialog::getOpenFileName(
            this,
            tr("选择密集点云"),
            initialDir,
            tr("点云文件 (*.xyz *.ply *.txt);;所有文件 (*)"));

        if (chosenPath.isEmpty())
        {
            return;
        }

        const QString normalized = QDir::cleanPath(chosenPath);
        const int index = m_denseCloudCombo->findData(normalized);
        if (index < 0)
        {
            m_denseCloudCombo->addItem(normalized, normalized);
        }
        m_denseCloudCombo->setCurrentText(normalized);
        emitSettingsNow();
    });

    connect(form.m_runBtn, &QPushButton::clicked, this, &MeshReconstructionDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject MeshReconstructionDialog::collectSettings() const
{
    QJsonObject o;
    o["method"]        = m_methodCombo->currentText();
    o["denseCloudPath"] = m_denseCloudCombo->currentText().trimmed();
    o["export_format"] = m_outputFormatCombo->currentText();
    o["qualityProfile"] = m_qualityProfileCombo->currentData().toString();
    o["octreeDepth"]   = m_octreeDepthSpin->value();
    o["meshResolution"]= m_meshResSpin->value();
    o["smoothIter"]    = m_smoothIterSpin->value();
    o["holeFill"]      = m_holeFillCheck->isChecked();
    o["maxHoleSize"]   = m_maxHoleSizeSpin->value();
    o["cleanSmall"]    = m_cleanCheck->isChecked();
    o["minFaces"]      = m_minFacesSpin->value();
    o["voxelDensity"]  = m_voxelDensityCombo->currentData().toString();
    o["decimate"]      = m_decimateCheck->isChecked();
    o["decimateRatio"] = m_decimateRatioSpin->value();
    o["threads"]       = m_threadsSpin->value();
    return o;
}

void MeshReconstructionDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("method"))
    {
        int i = m_methodCombo->findText(s["method"].toString());
        if (i >= 0) m_methodCombo->setCurrentIndex(i);
    }
    if (s.contains("denseCloudPath"))
    {
        m_denseCloudCombo->setCurrentText(QDir::cleanPath(s["denseCloudPath"].toString()));
    }
    if (s.contains("export_format"))
    {
        const int i = m_outputFormatCombo->findText(s["export_format"].toString());
        if (i >= 0) m_outputFormatCombo->setCurrentIndex(i);
    }
    if (s.contains("qualityProfile"))
    {
        const QString qualityProfile = s["qualityProfile"].toString();
        const int i = m_qualityProfileCombo->findData(qualityProfile);
        if (i >= 0)
        {
            m_qualityProfileCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("octreeDepth"))    m_octreeDepthSpin->setValue(s["octreeDepth"].toInt());
    if (s.contains("meshResolution")) m_meshResSpin->setValue(s["meshResolution"].toDouble());
    if (s.contains("smoothIter"))     m_smoothIterSpin->setValue(s["smoothIter"].toInt());
    if (s.contains("holeFill"))       m_holeFillCheck->setChecked(s["holeFill"].toBool());
    if (s.contains("maxHoleSize"))    m_maxHoleSizeSpin->setValue(s["maxHoleSize"].toDouble());
    if (s.contains("cleanSmall"))     m_cleanCheck->setChecked(s["cleanSmall"].toBool());
    if (s.contains("minFaces"))       m_minFacesSpin->setValue(s["minFaces"].toInt());
    if (s.contains("voxelDensity"))
    {
        const QString density = s["voxelDensity"].toString();
        const int index = m_voxelDensityCombo->findData(density);
        if (index >= 0)
        {
            m_voxelDensityCombo->setCurrentIndex(index);
        }
    }
    if (s.contains("decimate"))       m_decimateCheck->setChecked(s["decimate"].toBool());
    if (s.contains("decimateRatio"))  m_decimateRatioSpin->setValue(s["decimateRatio"].toDouble());
    if (s.contains("threads"))        m_threadsSpin->setValue(s["threads"].toInt());
}

void MeshReconstructionDialog::setDenseCloudCandidates(const QStringList &paths)
{
    const QString current = QDir::cleanPath(m_denseCloudCombo->currentText().trimmed());

    m_denseCloudCombo->clear();
    for (const QString &path : paths)
    {
        const QString normalized = QDir::cleanPath(path);
        if (!normalized.isEmpty() && m_denseCloudCombo->findData(normalized) < 0)
        {
            m_denseCloudCombo->addItem(normalized, normalized);
        }
    }

    if (!current.isEmpty())
    {
        const int index = m_denseCloudCombo->findData(current);
        if (index >= 0)
        {
            m_denseCloudCombo->setCurrentIndex(index);
        }
        else
        {
            m_denseCloudCombo->setCurrentText(current);
        }
    }
    else if (m_denseCloudCombo->count() > 0)
    {
        m_denseCloudCombo->setCurrentIndex(0);
    }
}

void MeshReconstructionDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void MeshReconstructionDialog::onRun() { emit runRequested(collectSettings()); accept(); }
