// =============================================================================
// 文件: SuperPointVisualizationDialog.cpp
// 功能: SuperPoint 特征点可视化配置对话框实现
// 职责:
//   - setupUi()         : 构建所有控件和分组布局
//   - setupConnections(): 连接控件变化信号到更新/颜色选择槽
//   - getDisplayOptions(): 将控件状态序列化为 FeatureDisplayOptions 结构体
//   - setDisplayOptions(): 将 FeatureDisplayOptions 结构体反序列化回控件状态
//   - onResetDefaults() : 将所有控件恢复为内置默认值
//   - updatePreview()   : 根据控件状态生成预览区文字描述
// =============================================================================
// SuperPoint 特征点可视化配置对话框实现

#include "SuperPointVisualizationDialog.h"

#include "ui_SuperPointVisualizationDialog.h"

#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QColorDialog>

SuperPointVisualizationDialog::SuperPointVisualizationDialog(const QStringList &availableSuffixes,
                                                           QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    if (!availableSuffixes.isEmpty())
    {
        m_suffixCombo->addItems(availableSuffixes);
    }
    setupConnections();
}

SuperPointVisualizationDialog::~SuperPointVisualizationDialog() = default;

QString SuperPointVisualizationDialog::currentSuffix() const
{
    return m_suffixCombo->currentText();
}

void SuperPointVisualizationDialog::setCurrentSuffix(const QString &suffix)
{
    const int idx = m_suffixCombo->findText(suffix);
    if (idx >= 0)
        m_suffixCombo->setCurrentIndex(idx);
}

void SuperPointVisualizationDialog::setAvailableSuffixes(const QStringList &suffixes)
{
    const QString current = m_suffixCombo->currentText();
    m_suffixCombo->blockSignals(true);
    m_suffixCombo->clear();
    m_suffixCombo->addItems(suffixes);
    const int idx = m_suffixCombo->findText(current);
    if (idx >= 0)
        m_suffixCombo->setCurrentIndex(idx);
    m_suffixCombo->blockSignals(false);
}

// setupUi: 构建对话框全部控件和分组布局
// 布局结构（从上到下）：
//   1. 显示内容组  —— 4 个 QCheckBox（显示点/尺度/方向/实心）
//   2. 样式设置组  —— 形状下拉、点大小、尺度倍数、透明度滑块+标签
//   3. 颜色设置组  —— 3 个颜色选择按钮（点/尺度圈/方向箭头）
//   4. 显示过滤组  —— 最大显示数量、高分优先复选框
//   5. 预览区      —— 文字描述标签
//   6. 底部按钮    —— 恢复默认 | 应用 | 关闭
void SuperPointVisualizationDialog::setupUi()
{
    Ui::SuperPointVisualizationDialog ui;
    ui.setupUi(this);

    m_suffixCombo = ui.m_suffixCombo;
    m_showPointsChk = ui.m_showPointsChk;
    m_showScaleChk = ui.m_showScaleChk;
    m_showOrientationChk = ui.m_showOrientationChk;
    m_useFillChk = ui.m_useFillChk;
    m_pointSizeSpin = ui.m_pointSizeSpin;
    m_scaleMultiplierSpin = ui.m_scaleMultiplierSpin;
    m_opacitySlider = ui.m_opacitySlider;
    m_opacityLabel = ui.m_opacityLabel;
    m_pointColorBtn = ui.m_pointColorBtn;
    m_scaleColorBtn = ui.m_scaleColorBtn;
    m_orientColorBtn = ui.m_orientColorBtn;
    m_markerShapeCombo = ui.m_markerShapeCombo;
    m_maxDisplaySpin = ui.m_maxDisplaySpin;
    m_showTopScoresChk = ui.m_showTopScoresChk;
    m_previewLabel = ui.m_previewLabel;
    m_applyBtn = ui.m_applyBtn;
    m_resetBtn = ui.m_resetBtn;
    m_closeBtn = ui.m_closeBtn;

    m_pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_pointColor.red()).arg(m_pointColor.green()).arg(m_pointColor.blue()));
    m_scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_scaleColor.red()).arg(m_scaleColor.green()).arg(m_scaleColor.blue()));
    m_orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_orientColor.red()).arg(m_orientColor.green()).arg(m_orientColor.blue()));
    m_applyBtn->setDefault(true);
}

// setupConnections: 连接所有控件的变化信号
// 变化信号 → updatePreview() : 实时更新预览文字（不触发外部重绘）
// 颜色按钮 → lambda          : 弹出颜色对话框、更新缓存色和按钮背景
// 底部按钮 → 对应槽函数
void SuperPointVisualizationDialog::setupConnections()
{
    // 特征文件后缀切换 → 通知外部
    connect(m_suffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this]() {
        emit featureSuffixChanged(m_suffixCombo->currentText());
    });

    // 显示选项变化 → 实时刷新预览文字
    connect(m_showPointsChk, &QCheckBox::toggled, this, &SuperPointVisualizationDialog::updatePreview);
    connect(m_showScaleChk, &QCheckBox::toggled, this, &SuperPointVisualizationDialog::updatePreview);
    connect(m_showOrientationChk, &QCheckBox::toggled, this, &SuperPointVisualizationDialog::updatePreview);
    connect(m_useFillChk, &QCheckBox::toggled, this, &SuperPointVisualizationDialog::updatePreview);
    
    // 样式变化
    connect(m_markerShapeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &SuperPointVisualizationDialog::updatePreview);
    connect(m_pointSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SuperPointVisualizationDialog::updatePreview);
    connect(m_scaleMultiplierSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &SuperPointVisualizationDialog::updatePreview);
    connect(m_opacitySlider, &QSlider::valueChanged, this, [this](int value) {
        m_opacityLabel->setText(QString::number(value));
        updatePreview();
    });
    
    // 颜色选择
    connect(m_pointColorBtn, &QPushButton::clicked, this, [this]() {
        QColor color = QColorDialog::getColor(m_pointColor, this, tr("选择点颜色"));
        if (color.isValid()) {
            m_pointColor = color;
            m_pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
                .arg(color.red()).arg(color.green()).arg(color.blue()));
            updatePreview();
        }
    });
    
    connect(m_scaleColorBtn, &QPushButton::clicked, this, [this]() {
        QColor color = QColorDialog::getColor(m_scaleColor, this, tr("选择尺度圈颜色"));
        if (color.isValid()) {
            m_scaleColor = color;
            m_scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
                .arg(color.red()).arg(color.green()).arg(color.blue()));
            updatePreview();
        }
    });
    
    connect(m_orientColorBtn, &QPushButton::clicked, this, [this]() {
        QColor color = QColorDialog::getColor(m_orientColor, this, tr("选择方向箭头颜色"));
        if (color.isValid()) {
            m_orientColor = color;
            m_orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
                .arg(color.red()).arg(color.green()).arg(color.blue()));
            updatePreview();
        }
    });
    
    // 过滤变化
    connect(m_maxDisplaySpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SuperPointVisualizationDialog::updatePreview);
    connect(m_showTopScoresChk, &QCheckBox::toggled, this, &SuperPointVisualizationDialog::updatePreview);
    
    // 按钮
    connect(m_applyBtn, &QPushButton::clicked, this, &SuperPointVisualizationDialog::onApply);
    connect(m_closeBtn, &QPushButton::clicked, this, &SuperPointVisualizationDialog::onClose);
    connect(m_resetBtn, &QPushButton::clicked, this, &SuperPointVisualizationDialog::onResetDefaults);
}

// getDisplayOptions: 将当前控件状态序列化为 FeatureDisplayOptions 结构体
// 标记形状（markerShape）映射关系：
//   下拉框 index 0 → "point"   (单像素点)
//   下拉框 index 1 → "circle"  (空心/实心圆形)
//   下拉框 index 2 → "cross"   (45° 叉号)
//   下拉框 index 3 (默认) → "plus" (十字加号)
LayerRenderer::FeatureDisplayOptions SuperPointVisualizationDialog::getDisplayOptions() const
{
    LayerRenderer::FeatureDisplayOptions opts;
    
    opts.showPoints = m_showPointsChk->isChecked();
    opts.showScale = m_showScaleChk->isChecked();
    opts.showOrientation = m_showOrientationChk->isChecked();
    opts.useFill = m_useFillChk->isChecked();
    
    opts.pointSize = m_pointSizeSpin->value();
    opts.scaleMultiplier = m_scaleMultiplierSpin->value();
    opts.opacity = m_opacitySlider->value();
    
    opts.pointColor = m_pointColor;
    opts.scaleColor = m_scaleColor;
    opts.orientColor = m_orientColor;
    
    // 标记形状
    int shapeIndex = m_markerShapeCombo->currentIndex();
    if (shapeIndex == 0) opts.markerShape = QStringLiteral("point");
    else if (shapeIndex == 1) opts.markerShape = QStringLiteral("circle");
    else if (shapeIndex == 2) opts.markerShape = QStringLiteral("cross");
    else opts.markerShape = QStringLiteral("plus");
    
    opts.maxDisplayCount = m_maxDisplaySpin->value();
    opts.showTopScores = m_showTopScoresChk->isChecked();
    
    return opts;
}

// setDisplayOptions: 将外部传入的 FeatureDisplayOptions 反序列化回各控件
// 同时更新颜色按钮的背景色样式表，以直观展示当前颜色
// markerShape 字符串映射关系与 getDisplayOptions() 对称
void SuperPointVisualizationDialog::setDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts)
{
    m_showPointsChk->setChecked(opts.showPoints);
    m_showScaleChk->setChecked(opts.showScale);
    m_showOrientationChk->setChecked(opts.showOrientation);
    m_useFillChk->setChecked(opts.useFill);
    
    m_pointSizeSpin->setValue(opts.pointSize);
    m_scaleMultiplierSpin->setValue(opts.scaleMultiplier);
    m_opacitySlider->setValue(opts.opacity);
    m_opacityLabel->setText(QString::number(opts.opacity));
    
    m_pointColor = opts.pointColor;
    m_scaleColor = opts.scaleColor;
    m_orientColor = opts.orientColor;
    
    // 更新颜色按钮
    m_pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_pointColor.red()).arg(m_pointColor.green()).arg(m_pointColor.blue()));
    m_scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_scaleColor.red()).arg(m_scaleColor.green()).arg(m_scaleColor.blue()));
    m_orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_orientColor.red()).arg(m_orientColor.green()).arg(m_orientColor.blue()));
    
    // 标记形状
    if (opts.markerShape == QStringLiteral("point")) m_markerShapeCombo->setCurrentIndex(0);
    else if (opts.markerShape == QStringLiteral("circle")) m_markerShapeCombo->setCurrentIndex(1);
    else if (opts.markerShape == QStringLiteral("cross")) m_markerShapeCombo->setCurrentIndex(2);
    else m_markerShapeCombo->setCurrentIndex(3);
    
    m_maxDisplaySpin->setValue(opts.maxDisplayCount);
    m_showTopScoresChk->setChecked(opts.showTopScores);
}

// onApply: 点击"应用"按钮时触发，将当前控件参数通过信号发出
void SuperPointVisualizationDialog::onApply()
{
    emitCurrentOptions();
}

// onClose: 点击"关闭"按钮时触发，调用 accept() 关闭对话框
void SuperPointVisualizationDialog::onClose()
{
    accept();
}

// onResetDefaults: 将所有控件恢复为内置默认值，然后刷新预览区
// 默认配置：仅显示点（1px 十字）、透明度180/255、
//           点颜色黄(255,200,0)、尺度圈亮黄(255,255,0)、方向箭头红(255,0,0)
//           最大显示0（全部）、优先高分开启
void SuperPointVisualizationDialog::onResetDefaults()
{
    // 恢复默认值
    m_showPointsChk->setChecked(true);
    m_showScaleChk->setChecked(false);
    m_showOrientationChk->setChecked(false);
    m_useFillChk->setChecked(false);
    
    m_markerShapeCombo->setCurrentIndex(2); // 默认：十字
    m_pointSizeSpin->setValue(1);
    m_scaleMultiplierSpin->setValue(1.0);
    m_opacitySlider->setValue(180);
    
    m_pointColor = QColor(255, 200, 0);
    m_scaleColor = QColor(255, 255, 0);
    m_orientColor = QColor(255, 0, 0);
    
    m_pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_pointColor.red()).arg(m_pointColor.green()).arg(m_pointColor.blue()));
    m_scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_scaleColor.red()).arg(m_scaleColor.green()).arg(m_scaleColor.blue()));
    m_orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_orientColor.red()).arg(m_orientColor.green()).arg(m_orientColor.blue()));
    
    m_maxDisplaySpin->setValue(0);
    m_showTopScoresChk->setChecked(true);
    
    updatePreview();
}

// updatePreview: 根据当前控件状态合成预览区文字描述（不发出外部信号）
// 拼接启用的显示项名称，若有数量限制则附加说明，若全部关闭显示"无显示内容"
void SuperPointVisualizationDialog::updatePreview()
{
    // 拼接当前启用的显示项名称
    QString desc;
    if (m_showPointsChk->isChecked()) desc += tr("显示点 ");
    if (m_showScaleChk->isChecked()) desc += tr("显示尺度 ");
    if (m_showOrientationChk->isChecked()) desc += tr("显示方向 ");
    
    if (m_maxDisplaySpin->value() > 0) {
        desc += tr("(最多 %1 个)").arg(m_maxDisplaySpin->value());
    }
    
    if (desc.isEmpty()) desc = tr("无显示内容");
    
    m_previewLabel->setText(desc);
}

// emitCurrentOptions: 收集当前选项并通过 displayOptionsChanged 信号通知外部
// 外部（LayerRenderer）收到信号后立即按新参数重绘特征点覆盖层
void SuperPointVisualizationDialog::emitCurrentOptions()
{
    emit displayOptionsChanged(getDisplayOptions());
}
