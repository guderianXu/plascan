#include "ThreeDReconstructionDialog.h"

#include "ui_ThreeDReconstructionDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QThread>
#include <QtGlobal>

namespace
{

constexpr int kDefaultFeatureGrayscaleMinPx = 5;

double grayscalePixelToNormalized(int value)
{
    return static_cast<double>(qBound(0, value, 255)) / 255.0;
}

int grayscaleSettingToPixel(const QJsonObject &settings,
                            const QString &pixelKey,
                            const QString &normalizedKey,
                            int fallback)
{
    if (settings.contains(pixelKey))
    {
        return qBound(0, settings.value(pixelKey).toInt(fallback), 255);
    }
    if (settings.contains(normalizedKey))
    {
        const double normalized = settings.value(normalizedKey).toDouble(grayscalePixelToNormalized(fallback));
        if (normalized > 1.0)
        {
            return qBound(0, qRound(normalized), 255);
        }
        return qBound(0, qRound(normalized * 255.0), 255);
    }
    return fallback;
}

} // namespace

ThreeDReconstructionDialog::ThreeDReconstructionDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
}

void ThreeDReconstructionDialog::setupUi()
{
    Ui::ThreeDReconstructionDialog form;
    form.setupUi(this);
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);

    _titleLabel = form.m_titleLabel;
    _statusLabel = form.m_statusLabel;
    _qualityCombo = form.m_qualityCombo;
    _deviceCombo = form.m_deviceCombo;
    _featureGrayMinSpin = form.m_featureGrayMinSpin;
    _threadsSpin = form.m_threadsSpin;
    _outputDirEdit = form.m_outputDirEdit;
    _exportObjCheck = form.m_exportObjCheck;
    _browseBtn = form.m_browseBtn;
    _startBtn = form.m_startBtn;
    _cancelBtn = form.m_cancelBtn;
    _exportObjCheck->setChecked(true);

    _qualityCombo->setItemData(0, QStringLiteral("standard"));
    _qualityCombo->setItemData(1, QStringLiteral("fast"));
    _qualityCombo->setItemData(2, QStringLiteral("quality"));
    _deviceCombo->setItemData(0, QStringLiteral("auto"));
    _deviceCombo->setItemData(1, QStringLiteral("cuda"));
    _deviceCombo->setItemData(2, QStringLiteral("cpu"));
    _threadsSpin->setValue(qMax(1, QThread::idealThreadCount()));

    connect(_browseBtn, &QPushButton::clicked, this, &ThreeDReconstructionDialog::browseOutputDir);
    connect(_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
    connect(_startBtn, &QPushButton::clicked, this, &ThreeDReconstructionDialog::start);

    connect(_qualityCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(_deviceCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(_featureGrayMinSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(_threadsSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(_outputDirEdit, &QLineEdit::textChanged,
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(_exportObjCheck, &QCheckBox::toggled,
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
}

void ThreeDReconstructionDialog::setMode(Mode mode)
{
    _mode = mode;
    if (mode == Mode::AerialTriangulation)
    {
        setWindowTitle(QStringLiteral("空中三角测量"));
        if (_titleLabel)
        {
            _titleLabel->setText(QStringLiteral("<b>一键生成正式 SfM/BA 稀疏云</b>"));
        }
        if (_exportObjCheck)
        {
            _exportObjCheck->setChecked(false);
            _exportObjCheck->setVisible(false);
        }
        if (_startBtn)
        {
            _startBtn->setText(QStringLiteral("开始空三"));
        }
        return;
    }

    setWindowTitle(QStringLiteral("三维重建"));
    if (_titleLabel)
    {
        _titleLabel->setText(QStringLiteral("<b>一键生成三维模型</b>"));
    }
    if (_exportObjCheck)
    {
        _exportObjCheck->setChecked(true);
        _exportObjCheck->setVisible(true);
    }
    if (_startBtn)
    {
        _startBtn->setText(QStringLiteral("开始重建"));
    }
}

void ThreeDReconstructionDialog::setImageCount(int count)
{
    _statusLabel->setText(tr("当前项目影像：%1 张").arg(count));
    _startBtn->setEnabled(count >= 2);
}

void ThreeDReconstructionDialog::setDefaultOutputDir(const QString &dir)
{
    if (_outputDirEdit && _outputDirEdit->text().trimmed().isEmpty())
    {
        _outputDirEdit->setText(QDir::cleanPath(dir));
    }
}

void ThreeDReconstructionDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty())
    {
        return;
    }

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    const int qualityIndex = _qualityCombo->findData(quality);
    if (qualityIndex >= 0)
    {
        _qualityCombo->setCurrentIndex(qualityIndex);
    }

    const QString device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    const int deviceIndex = _deviceCombo->findData(device);
    if (deviceIndex >= 0)
    {
        _deviceCombo->setCurrentIndex(deviceIndex);
    }

    if (settings.contains(QStringLiteral("threads")))
    {
        _threadsSpin->setValue(settings.value(QStringLiteral("threads")).toInt(_threadsSpin->value()));
    }
    _featureGrayMinSpin->setValue(grayscaleSettingToPixel(settings,
                                                          QStringLiteral("feature_grayscale_min_px"),
                                                          QStringLiteral("feature_grayscale_min"),
                                                          kDefaultFeatureGrayscaleMinPx));
    if (settings.contains(QStringLiteral("output_dir")))
    {
        _outputDirEdit->setText(settings.value(QStringLiteral("output_dir")).toString());
    }
    _exportObjCheck->setChecked(settings.value(QStringLiteral("export_obj")).toBool(_exportObjCheck->isChecked()));
}

QJsonObject ThreeDReconstructionDialog::collectSettings() const
{
    QJsonObject settings;
    QString quality = _qualityCombo->currentData().toString();
    if (quality.isEmpty())
    {
        quality = QStringLiteral("standard");
    }
    QString device = _deviceCombo->currentData().toString();
    if (device.isEmpty())
    {
        device = QStringLiteral("auto");
    }
    settings[QStringLiteral("quality")] = quality;
    settings[QStringLiteral("device")] = device;
    const int featureGrayscaleMinPx = qBound(0, _featureGrayMinSpin->value(), 255);
    settings[QStringLiteral("feature_grayscale_min_px")] = featureGrayscaleMinPx;
    settings[QStringLiteral("feature_grayscale_min")] = grayscalePixelToNormalized(featureGrayscaleMinPx);
    settings[QStringLiteral("threads")] = _threadsSpin->value();
    settings[QStringLiteral("output_dir")] = QDir::cleanPath(_outputDirEdit->text().trimmed());
    settings[QStringLiteral("export_obj")] = _exportObjCheck->isChecked();
    settings[QStringLiteral("workflow_kind")] =
        _mode == Mode::AerialTriangulation
            ? QStringLiteral("aerial_triangulation")
            : QStringLiteral("three_d_reconstruction");
    if (_mode == Mode::AerialTriangulation)
    {
        settings[QStringLiteral("export_obj")] = false;
    }
    return settings;
}

void ThreeDReconstructionDialog::emitSettingsChanged()
{
    emit settingsChanged(collectSettings());
}

void ThreeDReconstructionDialog::browseOutputDir()
{
    const QString dir = QFileDialog::getExistingDirectory(this,
                                                          tr("选择输出目录"),
                                                          _outputDirEdit->text().trimmed());
    if (!dir.isEmpty())
    {
        _outputDirEdit->setText(QDir::cleanPath(dir));
    }
}

void ThreeDReconstructionDialog::start()
{
    emit runRequested(collectSettings());
    accept();
}
