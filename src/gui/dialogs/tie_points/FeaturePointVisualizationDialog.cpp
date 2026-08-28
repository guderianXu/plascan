#include "tie_points/FeaturePointVisualizationDialog.h"

#include "ui_FeaturePointVisualizationDialog.h"

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>

FeaturePointVisualizationDialog::FeaturePointVisualizationDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    setupConnections();
}

FeaturePointVisualizationDialog::~FeaturePointVisualizationDialog() = default;

void FeaturePointVisualizationDialog::setupUi()
{
    Ui::FeaturePointVisualizationDialog ui;
    ui.setupUi(this);

    _pointSourceCombo = ui.m_pointSourceCombo;
    _showPointsChk = ui.m_showPointsChk;
    _pointSizeSpin = ui.m_pointSizeSpin;
    _pointColorBtn = ui.m_pointColorBtn;
    _opacitySlider = ui.m_opacitySlider;
    _opacityLabel = ui.m_opacityLabel;
    _maxDisplaySpin = ui.m_maxDisplaySpin;
    _pointStatusLabel = ui.m_pointStatusLabel;
    _showResidualsChk = ui.m_showResidualsChk;
    _residualScaleSpin = ui.m_residualScaleSpin;
    _residualStatusLabel = ui.m_residualStatusLabel;
    _applyBtn = ui.m_applyBtn;
    _resetBtn = ui.m_resetBtn;
    _closeBtn = ui.m_closeBtn;

    updateColorButton(_pointColorBtn, _pointColor);
    _applyBtn->setDefault(true);
}

void FeaturePointVisualizationDialog::setupConnections()
{
    connect(_opacitySlider, &QSlider::valueChanged, this, [this](int value)
    {
        _opacityLabel->setText(QString::number(value));
    });
    connect(_pointColorBtn, &QPushButton::clicked, this, [this]()
    {
        const QColor color = QColorDialog::getColor(_pointColor, this, tr("选择点颜色"));
        if (color.isValid())
        {
            _pointColor = color;
            updateColorButton(_pointColorBtn, color);
        }
    });
    connect(_applyBtn, &QPushButton::clicked, this, &FeaturePointVisualizationDialog::onApply);
    connect(_closeBtn, &QPushButton::clicked, this, &FeaturePointVisualizationDialog::onClose);
    connect(_resetBtn,
            &QPushButton::clicked,
            this,
            &FeaturePointVisualizationDialog::onResetDefaults);
}

LayerRenderer::FeatureDisplayOptions FeaturePointVisualizationDialog::getDisplayOptions() const
{
    LayerRenderer::FeatureDisplayOptions options;
    switch (_pointSourceCombo->currentIndex())
    {
    case 0:
        options.pointSource = xjw::gui::views::FeaturePointSource::ExtractedFeatures;
        break;
    case 1:
        options.pointSource = xjw::gui::views::FeaturePointSource::RawMatches;
        break;
    default:
        options.pointSource = xjw::gui::views::FeaturePointSource::ValidTiePoints;
        break;
    }
    options.showPoints = _showPointsChk->isChecked();
    options.pointSize = _pointSizeSpin->value();
    options.pointColor = _pointColor;
    options.opacity = _opacitySlider->value();
    options.maxDisplayCount = _maxDisplaySpin->value();
    options.showResiduals = _showResidualsChk->isChecked();
    options.residualScale = _residualScaleSpin->value();
    return options;
}

void FeaturePointVisualizationDialog::setDisplayOptions(
    const LayerRenderer::FeatureDisplayOptions &options)
{
    switch (options.pointSource)
    {
    case xjw::gui::views::FeaturePointSource::ExtractedFeatures:
        _pointSourceCombo->setCurrentIndex(0);
        break;
    case xjw::gui::views::FeaturePointSource::RawMatches:
        _pointSourceCombo->setCurrentIndex(1);
        break;
    case xjw::gui::views::FeaturePointSource::ValidTiePoints:
        _pointSourceCombo->setCurrentIndex(2);
        break;
    }
    _showPointsChk->setChecked(options.showPoints);
    _pointSizeSpin->setValue(options.pointSize);
    _pointColor = options.pointColor;
    updateColorButton(_pointColorBtn, _pointColor);
    _opacitySlider->setValue(options.opacity);
    _opacityLabel->setText(QString::number(options.opacity));
    _maxDisplaySpin->setValue(options.maxDisplayCount);
    _showResidualsChk->setChecked(options.showResiduals);
    _residualScaleSpin->setValue(options.residualScale);
}

void FeaturePointVisualizationDialog::setPointStatus(
    const QString &message,
    bool available,
    int count)
{
    _pointStatusLabel->setText(available
        ? tr("当前照片：%1 个点。%2").arg(count).arg(message)
        : tr("当前照片：%1").arg(message));
    _pointStatusLabel->setStyleSheet(available
        ? QStringLiteral("color: palette(text);")
        : QStringLiteral("color: #c06a00;"));
}

void FeaturePointVisualizationDialog::setResidualStatus(
    const QString &message,
    bool available,
    int count)
{
    _residualStatusLabel->setText(available
        ? tr("当前照片：%1 条残差箭头。%2").arg(count).arg(message)
        : tr("当前照片：%1").arg(message));
    _residualStatusLabel->setStyleSheet(available
        ? QStringLiteral("color: palette(text);")
        : QStringLiteral("color: #c06a00;"));
}

void FeaturePointVisualizationDialog::onApply()
{
    emit displayOptionsChanged(getDisplayOptions());
}

void FeaturePointVisualizationDialog::onClose()
{
    accept();
}

void FeaturePointVisualizationDialog::onResetDefaults()
{
    setDisplayOptions(LayerRenderer::FeatureDisplayOptions{});
}

void FeaturePointVisualizationDialog::updateColorButton(
    QPushButton *button,
    const QColor &color)
{
    if (!button)
    {
        return;
    }
    button->setStyleSheet(QStringLiteral("background-color: rgb(%1, %2, %3);")
                              .arg(color.red())
                              .arg(color.green())
                              .arg(color.blue()));
}
