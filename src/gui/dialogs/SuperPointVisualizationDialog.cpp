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

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QColorDialog>
#include <QFrame>

// 构造函数：调用 setupUi() 构建界面，再调用 setupConnections() 连接信号槽
SuperPointVisualizationDialog::SuperPointVisualizationDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    setupConnections();
}

SuperPointVisualizationDialog::~SuperPointVisualizationDialog() = default;

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
    setWindowTitle(tr("SuperPoint 特征点显示设置"));
    resize(500, 600);  // 固定初始尺寸（宽×高）

    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // ==================== 显示内容组 ====================
    QGroupBox *displayGroup = new QGroupBox(tr("显示内容"), this);
    QVBoxLayout *displayLayout = new QVBoxLayout(displayGroup);
    
    m_showPointsChk = new QCheckBox(tr("显示特征点"), displayGroup);
    m_showPointsChk->setChecked(true);
    m_showPointsChk->setToolTip(tr("显示关键点中心标记"));
    displayLayout->addWidget(m_showPointsChk);
    
    m_showScaleChk = new QCheckBox(tr("显示尺度圈"), displayGroup);
    m_showScaleChk->setChecked(false);
    m_showScaleChk->setToolTip(tr("显示关键点的尺度圆圈（基于KeyPoint.size）"));
    displayLayout->addWidget(m_showScaleChk);
    
    m_showOrientationChk = new QCheckBox(tr("显示方向箭头"), displayGroup);
    m_showOrientationChk->setChecked(false);
    m_showOrientationChk->setToolTip(tr("显示关键点的方向（基于KeyPoint.angle）\\nSuperPoint默认无方向"));
    displayLayout->addWidget(m_showOrientationChk);
    
    m_useFillChk = new QCheckBox(tr("实心填充"), displayGroup);
    m_useFillChk->setChecked(false);
    m_useFillChk->setToolTip(tr("使用实心标记（默认空心）"));
    displayLayout->addWidget(m_useFillChk);
    
    mainLayout->addWidget(displayGroup);

    // ==================== 样式设置组 ====================
    QGroupBox *styleGroup = new QGroupBox(tr("样式设置"), this);
    QFormLayout *styleForm = new QFormLayout(styleGroup);
    
    // 标记形状
    m_markerShapeCombo = new QComboBox(styleGroup);
    // 增加“点”选项并调整顺序：点, 圆形, 十字, 加号
    m_markerShapeCombo->addItems({tr("点"), tr("圆形"), tr("十字"), tr("加号")});
    m_markerShapeCombo->setToolTip(tr("特征点标记的形状"));
    styleForm->addRow(tr("标记形状:"), m_markerShapeCombo);
    
    // 点大小
    m_pointSizeSpin = new QSpinBox(styleGroup);
    m_pointSizeSpin->setRange(1, 20);
    m_pointSizeSpin->setValue(3);
    m_pointSizeSpin->setToolTip(tr("特征点标记的大小（像素）"));
    styleForm->addRow(tr("点大小:"), m_pointSizeSpin);
    
    // 尺度倍数
    m_scaleMultiplierSpin = new QDoubleSpinBox(styleGroup);
    m_scaleMultiplierSpin->setRange(0.1, 10.0);
    m_scaleMultiplierSpin->setValue(1.0);
    m_scaleMultiplierSpin->setDecimals(1);
    m_scaleMultiplierSpin->setSingleStep(0.1);
    m_scaleMultiplierSpin->setToolTip(tr("尺度圈相对于KeyPoint.size的倍数"));
    styleForm->addRow(tr("尺度倍数:"), m_scaleMultiplierSpin);
    
    // 透明度
    QHBoxLayout *opacityLayout = new QHBoxLayout();
    m_opacitySlider = new QSlider(Qt::Horizontal, styleGroup);
    m_opacitySlider->setRange(0, 255);
    m_opacitySlider->setValue(180);
    m_opacitySlider->setToolTip(tr("显示透明度 (0=完全透明, 255=完全不透明)"));
    m_opacityLabel = new QLabel(tr("180"), styleGroup);
    m_opacityLabel->setMinimumWidth(40);
    opacityLayout->addWidget(m_opacitySlider);
    opacityLayout->addWidget(m_opacityLabel);
    styleForm->addRow(tr("透明度:"), opacityLayout);
    
    mainLayout->addWidget(styleGroup);

    // ==================== 颜色设置组 ====================
    QGroupBox *colorGroup = new QGroupBox(tr("颜色设置"), this);
    QFormLayout *colorForm = new QFormLayout(colorGroup);
    
    // 点颜色
    m_pointColorBtn = new QPushButton(styleGroup);
    m_pointColorBtn->setText("    ");
    m_pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_pointColor.red()).arg(m_pointColor.green()).arg(m_pointColor.blue()));
    m_pointColorBtn->setToolTip(tr("点击选择特征点颜色"));
    colorForm->addRow(tr("点颜色:"), m_pointColorBtn);
    
    // 尺度圈颜色
    m_scaleColorBtn = new QPushButton(styleGroup);
    m_scaleColorBtn->setText("    ");
    m_scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_scaleColor.red()).arg(m_scaleColor.green()).arg(m_scaleColor.blue()));
    m_scaleColorBtn->setToolTip(tr("点击选择尺度圈颜色"));
    colorForm->addRow(tr("尺度圈颜色:"), m_scaleColorBtn);
    
    // 方向箭头颜色
    m_orientColorBtn = new QPushButton(styleGroup);
    m_orientColorBtn->setText("    ");
    m_orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(m_orientColor.red()).arg(m_orientColor.green()).arg(m_orientColor.blue()));
    m_orientColorBtn->setToolTip(tr("点击选择方向箭头颜色"));
    colorForm->addRow(tr("方向箭头颜色:"), m_orientColorBtn);
    
    mainLayout->addWidget(colorGroup);

    // ==================== 过滤设置组 ====================
    QGroupBox *filterGroup = new QGroupBox(tr("显示过滤"), this);
    QFormLayout *filterForm = new QFormLayout(filterGroup);
    
    // 最大显示数量
    m_maxDisplaySpin = new QSpinBox(filterGroup);
    m_maxDisplaySpin->setRange(0, 100000);
    m_maxDisplaySpin->setValue(0);
    m_maxDisplaySpin->setSpecialValueText(tr("全部显示"));
    m_maxDisplaySpin->setToolTip(tr("限制显示的特征点数量\\n0 = 全部显示"));
    filterForm->addRow(tr("最大显示数:"), m_maxDisplaySpin);
    
    // 显示高分优先
    m_showTopScoresChk = new QCheckBox(tr("优先显示高分点"), filterGroup);
    m_showTopScoresChk->setChecked(true);
    m_showTopScoresChk->setToolTip(tr("当限制显示数量时，优先显示score最高的点"));
    filterForm->addRow("", m_showTopScoresChk);
    
    mainLayout->addWidget(filterGroup);

    // ==================== 预览区 ====================
    QGroupBox *previewGroup = new QGroupBox(tr("预览"), this);
    QVBoxLayout *previewLayout = new QVBoxLayout(previewGroup);
    
    m_previewLabel = new QLabel(tr("调整参数后点击应用查看效果"), previewGroup);
    m_previewLabel->setAlignment(Qt::AlignCenter);
    m_previewLabel->setFrameStyle(QFrame::Panel | QFrame::Sunken);
    m_previewLabel->setMinimumHeight(80);
    previewLayout->addWidget(m_previewLabel);
    
    mainLayout->addWidget(previewGroup);

    // ==================== 底部按钮 ====================
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    m_resetBtn = new QPushButton(tr("恢复默认"), this);
    buttonLayout->addWidget(m_resetBtn);
    buttonLayout->addStretch();
    m_applyBtn = new QPushButton(tr("应用"), this);
    m_closeBtn = new QPushButton(tr("关闭"), this);
    buttonLayout->addWidget(m_applyBtn);
    buttonLayout->addWidget(m_closeBtn);
    
    mainLayout->addLayout(buttonLayout);

    m_applyBtn->setDefault(true);
}

// setupConnections: 连接所有控件的变化信号
// 变化信号 → updatePreview() : 实时更新预览文字（不触发外部重绘）
// 颜色按钮 → lambda          : 弹出颜色对话框、更新缓存色和按钮背景
// 底部按钮 → 对应槽函数
void SuperPointVisualizationDialog::setupConnections()
{
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
// 默认配置：仅显示点（空心圆点）、大小3px、透明度180/255、
//           点颜色黄(255,200,0)、尺度圈亮黄(255,255,0)、方向箭头红(255,0,0)
//           最大显示0（全部）、优先高分开启
void SuperPointVisualizationDialog::onResetDefaults()
{
    // 恢复默认值
    m_showPointsChk->setChecked(true);
    m_showScaleChk->setChecked(false);
    m_showOrientationChk->setChecked(false);
    m_useFillChk->setChecked(false);
    
    m_markerShapeCombo->setCurrentIndex(0); // 默认：点
    m_pointSizeSpin->setValue(3);
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
