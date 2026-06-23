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

    _denseCloudCombo = form.m_denseCloudCombo;
    _browseDenseBtn = form.m_browseDenseBtn;
    _methodCombo = form.m_methodCombo;
    _outputFormatCombo = form.m_outputFormatCombo;
    _qualityProfileCombo = form.m_qualityProfileCombo;
    _octreeDepthSpin = form.m_octreeDepthSpin;
    _meshResSpin = form.m_meshResSpin;
    _smoothIterSpin = form.m_smoothIterSpin;
    _holeFillCheck = form.m_holeFillCheck;
    _maxHoleSizeSpin = form.m_maxHoleSizeSpin;
    _cleanCheck = form.m_cleanCheck;
    _minFacesSpin = form.m_minFacesSpin;
    _voxelDensityCombo = form.m_voxelDensityCombo;
    _decimateCheck = form.m_decimateCheck;
    _decimateRatioSpin = form.m_decimateRatioSpin;
    _threadsSpin = form.m_threadsSpin;
    _infoLabel = form.m_infoLabel;

    _qualityProfileCombo->setItemData(0, QStringLiteral("detail"));
    _qualityProfileCombo->setItemData(1, QStringLiteral("balanced"));
    _qualityProfileCombo->setItemData(2, QStringLiteral("lite"));
    _voxelDensityCombo->setItemData(0, QStringLiteral("coarse"));
    _voxelDensityCombo->setItemData(1, QStringLiteral("medium"));
    _voxelDensityCombo->setItemData(2, QStringLiteral("fine"));

    auto changed = [this]() { emitSettingsNow(); };
    connect(_denseCloudCombo, &QComboBox::editTextChanged, this, changed);
    connect(_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_qualityProfileCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_octreeDepthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_meshResSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_smoothIterSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_holeFillCheck, &QCheckBox::toggled, this, changed);
    connect(_cleanCheck, &QCheckBox::toggled, this, changed);
    connect(_voxelDensityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_decimateCheck, &QCheckBox::toggled, this, changed);
    connect(_decimateRatioSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(_browseDenseBtn, &QPushButton::clicked, this, [this]() {
        const QString initialPath = _denseCloudCombo->currentText().trimmed();
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
        const int index = _denseCloudCombo->findData(normalized);
        if (index < 0)
        {
            _denseCloudCombo->addItem(normalized, normalized);
        }
        _denseCloudCombo->setCurrentText(normalized);
        emitSettingsNow();
    });

    connect(form.m_runBtn, &QPushButton::clicked, this, &MeshReconstructionDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject MeshReconstructionDialog::collectSettings() const
{
    QJsonObject o;
    o["method"] = _methodCombo->currentText();
    o["denseCloudPath"] = _denseCloudCombo->currentText().trimmed();
    o["export_format"] = _outputFormatCombo->currentText();
    o["qualityProfile"] = _qualityProfileCombo->currentData().toString();
    o["octreeDepth"] = _octreeDepthSpin->value();
    o["meshResolution"] = _meshResSpin->value();
    o["smoothIter"] = _smoothIterSpin->value();
    o["holeFill"] = _holeFillCheck->isChecked();
    o["maxHoleSize"] = _maxHoleSizeSpin->value();
    o["cleanSmall"] = _cleanCheck->isChecked();
    o["minFaces"] = _minFacesSpin->value();
    o["voxelDensity"] = _voxelDensityCombo->currentData().toString();
    o["decimate"] = _decimateCheck->isChecked();
    o["decimateRatio"] = _decimateRatioSpin->value();
    o["threads"] = _threadsSpin->value();
    return o;
}

void MeshReconstructionDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("method"))
    {
        const int i = _methodCombo->findText(s["method"].toString());
        if (i >= 0)
        {
            _methodCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("denseCloudPath"))
    {
        _denseCloudCombo->setCurrentText(QDir::cleanPath(s["denseCloudPath"].toString()));
    }
    if (s.contains("export_format"))
    {
        const int i = _outputFormatCombo->findText(s["export_format"].toString());
        if (i >= 0)
        {
            _outputFormatCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("qualityProfile"))
    {
        const QString qualityProfile = s["qualityProfile"].toString();
        const int i = _qualityProfileCombo->findData(qualityProfile);
        if (i >= 0)
        {
            _qualityProfileCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("octreeDepth"))
    {
        _octreeDepthSpin->setValue(s["octreeDepth"].toInt());
    }
    if (s.contains("meshResolution"))
    {
        _meshResSpin->setValue(s["meshResolution"].toDouble());
    }
    if (s.contains("smoothIter"))
    {
        _smoothIterSpin->setValue(s["smoothIter"].toInt());
    }
    if (s.contains("holeFill"))
    {
        _holeFillCheck->setChecked(s["holeFill"].toBool());
    }
    if (s.contains("maxHoleSize"))
    {
        _maxHoleSizeSpin->setValue(s["maxHoleSize"].toDouble());
    }
    if (s.contains("cleanSmall"))
    {
        _cleanCheck->setChecked(s["cleanSmall"].toBool());
    }
    if (s.contains("minFaces"))
    {
        _minFacesSpin->setValue(s["minFaces"].toInt());
    }
    if (s.contains("voxelDensity"))
    {
        const QString density = s["voxelDensity"].toString();
        const int index = _voxelDensityCombo->findData(density);
        if (index >= 0)
        {
            _voxelDensityCombo->setCurrentIndex(index);
        }
    }
    if (s.contains("decimate"))
    {
        _decimateCheck->setChecked(s["decimate"].toBool());
    }
    if (s.contains("decimateRatio"))
    {
        _decimateRatioSpin->setValue(s["decimateRatio"].toDouble());
    }
    if (s.contains("threads"))
    {
        _threadsSpin->setValue(s["threads"].toInt());
    }
}

void MeshReconstructionDialog::setDenseCloudCandidates(const QStringList &paths)
{
    const QString current = QDir::cleanPath(_denseCloudCombo->currentText().trimmed());

    _denseCloudCombo->clear();
    for (const QString &path : paths)
    {
        const QString normalized = QDir::cleanPath(path);
        if (!normalized.isEmpty() && _denseCloudCombo->findData(normalized) < 0)
        {
            _denseCloudCombo->addItem(normalized, normalized);
        }
    }

    if (!current.isEmpty())
    {
        const int index = _denseCloudCombo->findData(current);
        if (index >= 0)
        {
            _denseCloudCombo->setCurrentIndex(index);
        }
        else
        {
            _denseCloudCombo->setCurrentText(current);
        }
    }
    else if (_denseCloudCombo->count() > 0)
    {
        _denseCloudCombo->setCurrentIndex(0);
    }
}

void MeshReconstructionDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void MeshReconstructionDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
