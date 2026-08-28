#pragma once

#include "LayerRenderer.h"

#include <QDialog>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;

class FeaturePointVisualizationDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FeaturePointVisualizationDialog(QWidget *parent = nullptr);
    ~FeaturePointVisualizationDialog() override;

    LayerRenderer::FeatureDisplayOptions getDisplayOptions() const;
    void setDisplayOptions(const LayerRenderer::FeatureDisplayOptions &options);
    void setPointStatus(const QString &message, bool available, int count);
    void setResidualStatus(const QString &message, bool available, int count);

signals:
    void displayOptionsChanged(const LayerRenderer::FeatureDisplayOptions &options);

private slots:
    void onApply();
    void onClose();
    void onResetDefaults();

private:
    void setupUi();
    void setupConnections();
    void updateColorButton(QPushButton *button, const QColor &color);

    QComboBox *_pointSourceCombo = nullptr;
    QCheckBox *_showPointsChk = nullptr;
    QSpinBox *_pointSizeSpin = nullptr;
    QPushButton *_pointColorBtn = nullptr;
    QSlider *_opacitySlider = nullptr;
    QLabel *_opacityLabel = nullptr;
    QSpinBox *_maxDisplaySpin = nullptr;
    QLabel *_pointStatusLabel = nullptr;

    QCheckBox *_showResidualsChk = nullptr;
    QDoubleSpinBox *_residualScaleSpin = nullptr;
    QLabel *_residualStatusLabel = nullptr;

    QPushButton *_applyBtn = nullptr;
    QPushButton *_resetBtn = nullptr;
    QPushButton *_closeBtn = nullptr;

    QColor _pointColor{0, 120, 255};
};
