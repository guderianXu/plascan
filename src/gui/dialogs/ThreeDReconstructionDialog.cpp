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

    m_titleLabel = form.m_titleLabel;
    m_statusLabel = form.m_statusLabel;
    m_qualityCombo = form.m_qualityCombo;
    m_deviceCombo = form.m_deviceCombo;
    m_featureGrayMinSpin = form.m_featureGrayMinSpin;
    m_threadsSpin = form.m_threadsSpin;
    m_outputDirEdit = form.m_outputDirEdit;
    m_exportObjCheck = form.m_exportObjCheck;
    m_browseBtn = form.m_browseBtn;
    m_startBtn = form.m_startBtn;
    m_cancelBtn = form.m_cancelBtn;
    m_exportObjCheck->setChecked(true);

    m_qualityCombo->setItemData(0, QStringLiteral("standard"));
    m_qualityCombo->setItemData(1, QStringLiteral("fast"));
    m_qualityCombo->setItemData(2, QStringLiteral("quality"));
    m_deviceCombo->setItemData(0, QStringLiteral("auto"));
    m_deviceCombo->setItemData(1, QStringLiteral("cuda"));
    m_deviceCombo->setItemData(2, QStringLiteral("cpu"));
    m_threadsSpin->setValue(qMax(1, QThread::idealThreadCount()));

    connect(m_browseBtn, &QPushButton::clicked, this, &ThreeDReconstructionDialog::browseOutputDir);
    connect(m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
    connect(m_startBtn, &QPushButton::clicked, this, &ThreeDReconstructionDialog::start);

    connect(m_qualityCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(m_deviceCombo, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(m_featureGrayMinSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(m_threadsSpin, qOverload<int>(&QSpinBox::valueChanged),
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(m_outputDirEdit, &QLineEdit::textChanged,
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
    connect(m_exportObjCheck, &QCheckBox::toggled,
            this, &ThreeDReconstructionDialog::emitSettingsChanged);
}

void ThreeDReconstructionDialog::setMode(Mode mode)
{
    m_mode = mode;
    if (mode == Mode::AerialTriangulation)
    {
        setWindowTitle(QStringLiteral("空中三角测量"));
        if (m_titleLabel)
        {
            m_titleLabel->setText(QStringLiteral("<b>一键生成正式 SfM/BA 稀疏云</b>"));
        }
        if (m_exportObjCheck)
        {
            m_exportObjCheck->setChecked(false);
            m_exportObjCheck->setVisible(false);
        }
        if (m_startBtn)
        {
            m_startBtn->setText(QStringLiteral("开始空三"));
        }
        return;
    }

    setWindowTitle(QStringLiteral("三维重建"));
    if (m_titleLabel)
    {
        m_titleLabel->setText(QStringLiteral("<b>一键生成三维模型</b>"));
    }
    if (m_exportObjCheck)
    {
        m_exportObjCheck->setChecked(true);
        m_exportObjCheck->setVisible(true);
    }
    if (m_startBtn)
    {
        m_startBtn->setText(QStringLiteral("开始重建"));
    }
}

void ThreeDReconstructionDialog::setImageCount(int count)
{
    m_statusLabel->setText(tr("当前项目影像：%1 张").arg(count));
    m_startBtn->setEnabled(count >= 2);
}

void ThreeDReconstructionDialog::setDefaultOutputDir(const QString &dir)
{
    if (m_outputDirEdit && m_outputDirEdit->text().trimmed().isEmpty())
    {
        m_outputDirEdit->setText(QDir::cleanPath(dir));
    }
}

void ThreeDReconstructionDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty())
    {
        return;
    }

    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("standard"));
    const int qualityIndex = m_qualityCombo->findData(quality);
    if (qualityIndex >= 0)
    {
        m_qualityCombo->setCurrentIndex(qualityIndex);
    }

    const QString device = settings.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    const int deviceIndex = m_deviceCombo->findData(device);
    if (deviceIndex >= 0)
    {
        m_deviceCombo->setCurrentIndex(deviceIndex);
    }

    if (settings.contains(QStringLiteral("threads")))
    {
        m_threadsSpin->setValue(settings.value(QStringLiteral("threads")).toInt(m_threadsSpin->value()));
    }
    m_featureGrayMinSpin->setValue(grayscaleSettingToPixel(settings,
                                                           QStringLiteral("feature_grayscale_min_px"),
                                                           QStringLiteral("feature_grayscale_min"),
                                                           kDefaultFeatureGrayscaleMinPx));
    if (settings.contains(QStringLiteral("output_dir")))
    {
        m_outputDirEdit->setText(settings.value(QStringLiteral("output_dir")).toString());
    }
    m_exportObjCheck->setChecked(settings.value(QStringLiteral("export_obj")).toBool(m_exportObjCheck->isChecked()));
}

QJsonObject ThreeDReconstructionDialog::collectSettings() const
{
    QJsonObject settings;
    QString quality = m_qualityCombo->currentData().toString();
    if (quality.isEmpty())
    {
        quality = QStringLiteral("standard");
    }
    QString device = m_deviceCombo->currentData().toString();
    if (device.isEmpty())
    {
        device = QStringLiteral("auto");
    }
    settings[QStringLiteral("quality")] = quality;
    settings[QStringLiteral("device")] = device;
    const int featureGrayscaleMinPx = qBound(0, m_featureGrayMinSpin->value(), 255);
    settings[QStringLiteral("feature_grayscale_min_px")] = featureGrayscaleMinPx;
    settings[QStringLiteral("feature_grayscale_min")] = grayscalePixelToNormalized(featureGrayscaleMinPx);
    settings[QStringLiteral("threads")] = m_threadsSpin->value();
    settings[QStringLiteral("output_dir")] = QDir::cleanPath(m_outputDirEdit->text().trimmed());
    settings[QStringLiteral("export_obj")] = m_exportObjCheck->isChecked();
    settings[QStringLiteral("workflow_kind")] =
        m_mode == Mode::AerialTriangulation
            ? QStringLiteral("aerial_triangulation")
            : QStringLiteral("three_d_reconstruction");
    if (m_mode == Mode::AerialTriangulation)
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
                                                          m_outputDirEdit->text().trimmed());
    if (!dir.isEmpty())
    {
        m_outputDirEdit->setText(QDir::cleanPath(dir));
    }
}

void ThreeDReconstructionDialog::start()
{
    emit runRequested(collectSettings());
    accept();
}
