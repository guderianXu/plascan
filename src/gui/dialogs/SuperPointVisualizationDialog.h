// =============================================================================
// 文件: SuperPointVisualizationDialog.h
// 功能: SuperPoint 特征点可视化配置对话框声明
// 职责:
//   - 提供特征点显示内容的多维控制：是否显示点、尺度圈、方向箭头、实心填充
//   - 提供样式设置：标记形状（点/圆/十字/加号）、点大小、尺度倍数、整体透明度
//   - 提供颜色设置：点颜色、尺度圈颜色、方向箭头颜色
//   - 提供显示过滤：最大显示数量限制、优先显示高分点
//   - 实时预览描述及应用/重置/关闭按钮
//   - 通过 displayOptionsChanged 信号实时通知外部（LayerRenderer）刷新视图
// =============================================================================
// SuperPoint 特征点可视化配置对话框
// 专门用于配置SuperPoint检测结果的显示方式

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
// SuperPointVisualizationDialog — SuperPoint 特征点可视化参数配置对话框
//
// 典型用法：
//   auto *dlg = new SuperPointVisualizationDialog(this);
//   connect(dlg, &SuperPointVisualizationDialog::displayOptionsChanged,
//           layerRenderer, &LayerRenderer::setDisplayOptions);
//   dlg->show(); // 非模态，参数修改实时预览
// =============================================================================
class SuperPointVisualizationDialog : public QDialog
{
    Q_OBJECT
public:
    // 构造函数，初始化界面并建立信号槽连接
    explicit SuperPointVisualizationDialog(QWidget *parent = nullptr);
    // 析构函数（默认）
    ~SuperPointVisualizationDialog() override;

    // 从对话框控件中读取并返回当前完整的显示选项结构体
    LayerRenderer::FeatureDisplayOptions getDisplayOptions() const;
    
    // 将外部保存的显示选项恢复到对话框控件（如从项目配置载入时调用）
    // opts — 要应用的显示选项结构体
    void setDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts);

signals:
    // 任意显示参数改变（通过"应用"按钮或实时更新）时发出
    // opts — 最新的显示选项，外部 LayerRenderer 收到后立即重绘
    void displayOptionsChanged(const LayerRenderer::FeatureDisplayOptions &opts);

private slots:
    // 点击"应用"按钮时触发，发出 displayOptionsChanged 信号
    void onApply();
    // 点击"关闭"按钮时触发，接受对话框（accept）
    void onClose();
    // 点击"恢复默认"按钮时触发，将所有控件恢复到内置默认值
    void onResetDefaults();
    // 任意控件值改变时更新预览区文字描述（不发出外部信号，仅刷新文字）
    void updatePreview();

private:
    // 构建对话框所有控件及布局
    void setupUi();
    // 连接各控件的变化信号到 updatePreview 及颜色/按钮槽
    void setupConnections();
    // 收集当前选项并通过 displayOptionsChanged 信号发出
    void emitCurrentOptions();

    // ── 显示内容控制 ─────────────────────────────────────────────
    QCheckBox*      m_showPointsChk{nullptr};      // 是否显示关键点中心标记
    QCheckBox*      m_showScaleChk{nullptr};       // 是否显示关键点尺度圆圈
    QCheckBox*      m_showOrientationChk{nullptr}; // 是否显示关键点方向箭头
    QCheckBox*      m_useFillChk{nullptr};         // 是否使用实心填充（默认空心）

    // ── 尺寸和透明度 ──────────────────────────────────────────────
    QSpinBox*       m_pointSizeSpin{nullptr};        // 标记大小（像素，1–20）
    QDoubleSpinBox* m_scaleMultiplierSpin{nullptr};  // 尺度圈相对 KeyPoint.size 的倍数
    QSlider*        m_opacitySlider{nullptr};        // 整体透明度（0–255）
    QLabel*         m_opacityLabel{nullptr};         // 显示当前透明度数值的标签

    // ── 颜色选择按钮 ──────────────────────────────────────────────
    QPushButton*    m_pointColorBtn{nullptr};   // 点颜色选择按钮（按钮背景 = 当前颜色）
    QPushButton*    m_scaleColorBtn{nullptr};   // 尺度圈颜色选择按钮
    QPushButton*    m_orientColorBtn{nullptr};  // 方向箭头颜色选择按钮

    // ── 形状和过滤 ────────────────────────────────────────────────
    QComboBox*      m_markerShapeCombo{nullptr};   // 标记形状：点/圆形/十字/加号
    QSpinBox*       m_maxDisplaySpin{nullptr};     // 最大显示特征点数（0 = 全部）
    QCheckBox*      m_showTopScoresChk{nullptr};   // 限制数量时是否优先显示高 score 点

    // ── 预览与底部按钮 ────────────────────────────────────────────
    QLabel*         m_previewLabel{nullptr};   // 预览区文字描述标签
    QPushButton*    m_applyBtn{nullptr};       // "应用"按钮（发出信号触发外部重绘）
    QPushButton*    m_resetBtn{nullptr};       // "恢复默认"按钮
    QPushButton*    m_closeBtn{nullptr};       // "关闭"按钮

    // ── 当前颜色缓存 ──────────────────────────────────────────────
    QColor m_pointColor{255, 200, 0};  // 当前特征点颜色（默认黄色）
    QColor m_scaleColor{255, 255, 0};  // 当前尺度圈颜色（默认亮黄色）
    QColor m_orientColor{255, 0, 0};   // 当前方向箭头颜色（默认红色）
};
