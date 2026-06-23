// =============================================================================
// 文件: FeaturePointVisualizationDialog.h
// 功能: 特征点可视化配置对话框声明
// 职责:
//   - 提供特征点文件后缀切换（.sp / .dsk / .alk / ...）
//   - 提供特征点显示内容的多维控制
//   - 提供样式设置：标记形状、点大小、尺度倍数、整体透明度
//   - 提供颜色设置：点颜色、尺度圈颜色、方向箭头颜色
//   - 提供显示过滤：最大显示数量限制、优先显示高分点
//   - 通过 displayOptionsChanged 信号实时通知外部刷新视图
// =============================================================================

#pragma once

#include <QDialog>
#include "LayerRenderer.h"  // 包含 FeatureDisplayOptions 结构体定义

// Qt 控件前向声明
class QCheckBox;
class QSpinBox;
class QDoubleSpinBox;
class QSlider;
class QLabel;
class QPushButton;
class QGroupBox;
class QComboBox;

// =============================================================================
// FeaturePointVisualizationDialog — 特征点可视化参数配置对话框
//
// 典型用法：
//   auto *dlg = new FeaturePointVisualizationDialog(suffixes, this);
//   dlg->setCurrentSuffix(currentSuffix);
//   connect(dlg, &FeaturePointVisualizationDialog::displayOptionsChanged,
//           layerRenderer, &LayerRenderer::setDisplayOptions);
//   connect(dlg, &FeaturePointVisualizationDialog::featureSuffixChanged,
//           canvasWidget, &CanvasWidget::setActiveFeatureSuffix);
//   dlg->show(); // 非模态，参数修改实时预览
// =============================================================================
class FeaturePointVisualizationDialog : public QDialog
{
    Q_OBJECT
public:
    explicit FeaturePointVisualizationDialog(const QStringList &availableSuffixes,
                                             QWidget *parent = nullptr);
    ~FeaturePointVisualizationDialog() override;

    LayerRenderer::FeatureDisplayOptions getDisplayOptions() const;
    void setDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts);

    // 特征文件后缀
    QString currentSuffix() const;
    void setCurrentSuffix(const QString &suffix);
    void setAvailableSuffixes(const QStringList &suffixes);

signals:
    void displayOptionsChanged(const LayerRenderer::FeatureDisplayOptions &opts);
    // 用户切换了特征文件后缀
    void featureSuffixChanged(const QString &suffix);

private slots:
    void onApply();
    void onClose();
    void onResetDefaults();
    void updatePreview();

private:
    void setupUi();
    void setupConnections();
    void emitCurrentOptions();

    // ── 特征文件选择 ─────────────────────────────────────────────
    QComboBox *_suffixCombo{nullptr};

    // ── 显示内容控制 ─────────────────────────────────────────────
    QCheckBox *_showPointsChk{nullptr};
    QCheckBox *_showScaleChk{nullptr};
    QCheckBox *_showOrientationChk{nullptr};
    QCheckBox *_useFillChk{nullptr};

    // ── 尺寸和透明度 ──────────────────────────────────────────────
    QSpinBox *_pointSizeSpin{nullptr};
    QDoubleSpinBox *_scaleMultiplierSpin{nullptr};
    QSlider *_opacitySlider{nullptr};
    QLabel *_opacityLabel{nullptr};

    // ── 颜色选择按钮 ──────────────────────────────────────────────
    QPushButton *_pointColorBtn{nullptr};
    QPushButton *_scaleColorBtn{nullptr};
    QPushButton *_orientColorBtn{nullptr};

    // ── 形状和过滤 ────────────────────────────────────────────────
    QComboBox *_markerShapeCombo{nullptr};
    QSpinBox *_maxDisplaySpin{nullptr};
    QCheckBox *_showTopScoresChk{nullptr};

    // ── 预览与底部按钮 ────────────────────────────────────────────
    QLabel *_previewLabel{nullptr};
    QPushButton *_applyBtn{nullptr};
    QPushButton *_resetBtn{nullptr};
    QPushButton *_closeBtn{nullptr};

    // ── 当前颜色缓存 ──────────────────────────────────────────────
    QColor _pointColor{0, 120, 255};
    QColor _scaleColor{255, 255, 0};
    QColor _orientColor{255, 0, 0};
};
