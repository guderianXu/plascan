// =============================================================================
// 文件: FeaturePointVisualizationDialog.cpp
// 功能: 特征点可视化配置对话框实现
// 职责:
//   - setupUi()         : 构建所有控件和分组布局
//   - setupConnections(): 连接控件变化信号到更新/颜色选择槽
//   - getDisplayOptions(): 将控件状态序列化为 FeatureDisplayOptions 结构体
//   - setDisplayOptions(): 将 FeatureDisplayOptions 结构体反序列化回控件状态
//   - onResetDefaults() : 将所有控件恢复为内置默认值
//   - updatePreview()   : 根据控件状态生成预览区文字描述
// =============================================================================
// 特征点可视化配置对话框实现

#include "FeaturePointVisualizationDialog.h"

#include "ui_FeaturePointVisualizationDialog.h"

#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QColorDialog>

FeaturePointVisualizationDialog::FeaturePointVisualizationDialog(const QStringList &availableSuffixes,
                                                                 QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    if (!availableSuffixes.isEmpty())
    {
        _suffixCombo->addItems(availableSuffixes);
    }
    setupConnections();
}

FeaturePointVisualizationDialog::~FeaturePointVisualizationDialog() = default;

QString FeaturePointVisualizationDialog::currentSuffix() const
{
    return _suffixCombo->currentText();
}

void FeaturePointVisualizationDialog::setCurrentSuffix(const QString &suffix)
{
    const int idx = _suffixCombo->findText(suffix);
    if (idx >= 0)
        _suffixCombo->setCurrentIndex(idx);
}

void FeaturePointVisualizationDialog::setAvailableSuffixes(const QStringList &suffixes)
{
    const QString current = _suffixCombo->currentText();
    _suffixCombo->blockSignals(true);
    _suffixCombo->clear();
    _suffixCombo->addItems(suffixes);
    const int idx = _suffixCombo->findText(current);
    if (idx >= 0)
        _suffixCombo->setCurrentIndex(idx);
    _suffixCombo->blockSignals(false);
}

// setupUi: 构建对话框全部控件和分组布局
// 布局结构（从上到下）：
//   1. 显示内容组  —— 4 个 QCheckBox（显示点/尺度/方向/实心）
//   2. 样式设置组  —— 形状下拉、点大小、尺度倍数、透明度滑块+标签
//   3. 颜色设置组  —— 3 个颜色选择按钮（点/尺度圈/方向箭头）
//   4. 显示过滤组  —— 最大显示数量、高分优先复选框
//   5. 预览区      —— 文字描述标签
//   6. 底部按钮    —— 恢复默认 | 应用 | 关闭
void FeaturePointVisualizationDialog::setupUi()
{
    Ui::FeaturePointVisualizationDialog ui;
    ui.setupUi(this);

    _suffixCombo = ui.m_suffixCombo;
    _showPointsChk = ui.m_showPointsChk;
    _showScaleChk = ui.m_showScaleChk;
    _showOrientationChk = ui.m_showOrientationChk;
    _useFillChk = ui.m_useFillChk;
    _pointSizeSpin = ui.m_pointSizeSpin;
    _scaleMultiplierSpin = ui.m_scaleMultiplierSpin;
    _opacitySlider = ui.m_opacitySlider;
    _opacityLabel = ui.m_opacityLabel;
    _pointColorBtn = ui.m_pointColorBtn;
    _scaleColorBtn = ui.m_scaleColorBtn;
    _orientColorBtn = ui.m_orientColorBtn;
    _markerShapeCombo = ui.m_markerShapeCombo;
    _maxDisplaySpin = ui.m_maxDisplaySpin;
    _showTopScoresChk = ui.m_showTopScoresChk;
    _previewLabel = ui.m_previewLabel;
    _applyBtn = ui.m_applyBtn;
    _resetBtn = ui.m_resetBtn;
    _closeBtn = ui.m_closeBtn;

    _pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_pointColor.red()).arg(_pointColor.green()).arg(_pointColor.blue()));
    _scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_scaleColor.red()).arg(_scaleColor.green()).arg(_scaleColor.blue()));
    _orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_orientColor.red()).arg(_orientColor.green()).arg(_orientColor.blue()));
    _applyBtn->setDefault(true);
}

// setupConnections: 连接所有控件的变化信号
// 变化信号 → updatePreview() : 实时更新预览文字（不触发外部重绘）
// 颜色按钮 → lambda          : 弹出颜色对话框、更新缓存色和按钮背景
// 底部按钮 → 对应槽函数
void FeaturePointVisualizationDialog::setupConnections()
{
    // 特征文件后缀切换 → 通知外部
    connect(_suffixCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this]() {
        emit featureSuffixChanged(_suffixCombo->currentText());
    });

    // 显示选项变化 → 实时刷新预览文字
    connect(_showPointsChk, &QCheckBox::toggled, this, &FeaturePointVisualizationDialog::updatePreview);
    connect(_showScaleChk, &QCheckBox::toggled, this, &FeaturePointVisualizationDialog::updatePreview);
    connect(_showOrientationChk, &QCheckBox::toggled, this, &FeaturePointVisualizationDialog::updatePreview);
    connect(_useFillChk, &QCheckBox::toggled, this, &FeaturePointVisualizationDialog::updatePreview);
    
    // 样式变化
    connect(_markerShapeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FeaturePointVisualizationDialog::updatePreview);
    connect(_pointSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeaturePointVisualizationDialog::updatePreview);
    connect(_scaleMultiplierSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &FeaturePointVisualizationDialog::updatePreview);
    connect(_opacitySlider, &QSlider::valueChanged, this, [this](int value) {
        _opacityLabel->setText(QString::number(value));
        updatePreview();
    });
    
    // 颜色选择
    connect(_pointColorBtn, &QPushButton::clicked, this, [this]() {
        QColor color = QColorDialog::getColor(_pointColor, this, tr("选择点颜色"));
        if (color.isValid()) {
            _pointColor = color;
            _pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
                .arg(color.red()).arg(color.green()).arg(color.blue()));
            updatePreview();
        }
    });
    
    connect(_scaleColorBtn, &QPushButton::clicked, this, [this]() {
        QColor color = QColorDialog::getColor(_scaleColor, this, tr("选择尺度圈颜色"));
        if (color.isValid()) {
            _scaleColor = color;
            _scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
                .arg(color.red()).arg(color.green()).arg(color.blue()));
            updatePreview();
        }
    });
    
    connect(_orientColorBtn, &QPushButton::clicked, this, [this]() {
        QColor color = QColorDialog::getColor(_orientColor, this, tr("选择方向箭头颜色"));
        if (color.isValid()) {
            _orientColor = color;
            _orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
                .arg(color.red()).arg(color.green()).arg(color.blue()));
            updatePreview();
        }
    });
    
    // 过滤变化
    connect(_maxDisplaySpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &FeaturePointVisualizationDialog::updatePreview);
    connect(_showTopScoresChk, &QCheckBox::toggled, this, &FeaturePointVisualizationDialog::updatePreview);
    
    // 按钮
    connect(_applyBtn, &QPushButton::clicked, this, &FeaturePointVisualizationDialog::onApply);
    connect(_closeBtn, &QPushButton::clicked, this, &FeaturePointVisualizationDialog::onClose);
    connect(_resetBtn, &QPushButton::clicked, this, &FeaturePointVisualizationDialog::onResetDefaults);
}

// getDisplayOptions: 将当前控件状态序列化为 FeatureDisplayOptions 结构体
// 标记形状（markerShape）映射关系：
//   下拉框 index 0 → "point"   (单像素点)
//   下拉框 index 1 → "circle"  (空心/实心圆形)
//   下拉框 index 2 → "cross"   (45° 叉号)
//   下拉框 index 3 (默认) → "plus" (十字加号)
LayerRenderer::FeatureDisplayOptions FeaturePointVisualizationDialog::getDisplayOptions() const
{
    LayerRenderer::FeatureDisplayOptions opts;
    
    opts.showPoints = _showPointsChk->isChecked();
    opts.showScale = _showScaleChk->isChecked();
    opts.showOrientation = _showOrientationChk->isChecked();
    opts.useFill = _useFillChk->isChecked();
    
    opts.pointSize = _pointSizeSpin->value();
    opts.scaleMultiplier = _scaleMultiplierSpin->value();
    opts.opacity = _opacitySlider->value();
    
    opts.pointColor = _pointColor;
    opts.scaleColor = _scaleColor;
    opts.orientColor = _orientColor;
    
    // 标记形状
    int shapeIndex = _markerShapeCombo->currentIndex();
    if (shapeIndex == 0) opts.markerShape = QStringLiteral("point");
    else if (shapeIndex == 1) opts.markerShape = QStringLiteral("circle");
    else if (shapeIndex == 2) opts.markerShape = QStringLiteral("cross");
    else opts.markerShape = QStringLiteral("plus");
    
    opts.maxDisplayCount = _maxDisplaySpin->value();
    opts.showTopScores = _showTopScoresChk->isChecked();
    
    return opts;
}

// setDisplayOptions: 将外部传入的 FeatureDisplayOptions 反序列化回各控件
// 同时更新颜色按钮的背景色样式表，以直观展示当前颜色
// markerShape 字符串映射关系与 getDisplayOptions() 对称
void FeaturePointVisualizationDialog::setDisplayOptions(const LayerRenderer::FeatureDisplayOptions &opts)
{
    _showPointsChk->setChecked(opts.showPoints);
    _showScaleChk->setChecked(opts.showScale);
    _showOrientationChk->setChecked(opts.showOrientation);
    _useFillChk->setChecked(opts.useFill);
    
    _pointSizeSpin->setValue(opts.pointSize);
    _scaleMultiplierSpin->setValue(opts.scaleMultiplier);
    _opacitySlider->setValue(opts.opacity);
    _opacityLabel->setText(QString::number(opts.opacity));
    
    _pointColor = opts.pointColor;
    _scaleColor = opts.scaleColor;
    _orientColor = opts.orientColor;
    
    // 更新颜色按钮
    _pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_pointColor.red()).arg(_pointColor.green()).arg(_pointColor.blue()));
    _scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_scaleColor.red()).arg(_scaleColor.green()).arg(_scaleColor.blue()));
    _orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_orientColor.red()).arg(_orientColor.green()).arg(_orientColor.blue()));
    
    // 标记形状
    if (opts.markerShape == QStringLiteral("point")) _markerShapeCombo->setCurrentIndex(0);
    else if (opts.markerShape == QStringLiteral("circle")) _markerShapeCombo->setCurrentIndex(1);
    else if (opts.markerShape == QStringLiteral("cross")) _markerShapeCombo->setCurrentIndex(2);
    else _markerShapeCombo->setCurrentIndex(3);
    
    _maxDisplaySpin->setValue(opts.maxDisplayCount);
    _showTopScoresChk->setChecked(opts.showTopScores);
}

// onApply: 点击"应用"按钮时触发，将当前控件参数通过信号发出
void FeaturePointVisualizationDialog::onApply()
{
    emitCurrentOptions();
}

// onClose: 点击"关闭"按钮时触发，调用 accept() 关闭对话框
void FeaturePointVisualizationDialog::onClose()
{
    accept();
}

// onResetDefaults: 将所有控件恢复为内置默认值，然后刷新预览区
// 默认配置：仅显示点（1px 十字）、透明度180/255、
//           点颜色蓝(0,120,255)、尺度圈亮黄(255,255,0)、方向箭头红(255,0,0)
//           最大显示0（全部）、优先高分开启
void FeaturePointVisualizationDialog::onResetDefaults()
{
    // 恢复默认值
    _showPointsChk->setChecked(true);
    _showScaleChk->setChecked(false);
    _showOrientationChk->setChecked(false);
    _useFillChk->setChecked(false);
    
    _markerShapeCombo->setCurrentIndex(2); // 默认：十字
    _pointSizeSpin->setValue(1);
    _scaleMultiplierSpin->setValue(1.0);
    _opacitySlider->setValue(180);
    
    _pointColor = QColor(0, 120, 255);
    _scaleColor = QColor(255, 255, 0);
    _orientColor = QColor(255, 0, 0);
    
    _pointColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_pointColor.red()).arg(_pointColor.green()).arg(_pointColor.blue()));
    _scaleColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_scaleColor.red()).arg(_scaleColor.green()).arg(_scaleColor.blue()));
    _orientColorBtn->setStyleSheet(QString("background-color: rgb(%1, %2, %3);")
        .arg(_orientColor.red()).arg(_orientColor.green()).arg(_orientColor.blue()));
    
    _maxDisplaySpin->setValue(0);
    _showTopScoresChk->setChecked(true);
    
    updatePreview();
}

// updatePreview: 根据当前控件状态合成预览区文字描述（不发出外部信号）
// 拼接启用的显示项名称，若有数量限制则附加说明，若全部关闭显示"无显示内容"
void FeaturePointVisualizationDialog::updatePreview()
{
    // 拼接当前启用的显示项名称
    QString desc;
    if (_showPointsChk->isChecked()) desc += tr("显示点 ");
    if (_showScaleChk->isChecked()) desc += tr("显示尺度 ");
    if (_showOrientationChk->isChecked()) desc += tr("显示方向 ");
    
    if (_maxDisplaySpin->value() > 0) {
        desc += tr("(最多 %1 个)").arg(_maxDisplaySpin->value());
    }
    
    if (desc.isEmpty()) desc = tr("无显示内容");
    
    _previewLabel->setText(desc);
}

// emitCurrentOptions: 收集当前选项并通过 displayOptionsChanged 信号通知外部
// 外部（LayerRenderer）收到信号后立即按新参数重绘特征点覆盖层
void FeaturePointVisualizationDialog::emitCurrentOptions()
{
    emit displayOptionsChanged(getDisplayOptions());
}
