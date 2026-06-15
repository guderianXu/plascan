// =============================================================================
// 文件: MatchViewerDialog.cpp
// 功能: MatchViewerDialog 的实现
// 职责:
//   - 绑定 .ui 中的工具栏、显示选项控件组、状态栏
//   - 通过 DualImageViewer 加载并展示两张影像及匹配连线
//   - 将用户操作（同步缩放、颜色、宽度等）实时转发给 MatchLineOverlay
//   - 通过 project_dialog.json 持久化显示配置（项目级）
// =============================================================================
#include "MatchViewerDialog.h"
#include "DualImageViewer.h"        // 双图并列查看器
#include "ImageViewWidget.h"        // 单张影像可缩放/平移控件
#include "MatchLineOverlay.h"       // 匹配连线覆盖层（负责绘制连接线）
#include "DisparityHeatmapOverlay.h" // 视差热力图覆盖层
#include "settings/DialogSettingStore.h" // 项目级记忆化
#include "settings/DialogSettingKeys.h"
#include "ui_MatchViewerDialog.h"

#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QColorDialog>
#include <QFileInfo>
#include <QStatusBar>
#include <QGroupBox>
#include <QTabWidget>
#include <QComboBox>

// 构造函数
// imgA      — 左侧影像路径
// imgB      — 右侧影像路径
// matchFile — .match 格式匹配文件路径
// parent    — 父窗口
MatchViewerDialog::MatchViewerDialog(const QString &imgA, const QString &imgB,
                                     const QString &matchFile, QWidget *parent)
    : QDialog(parent)
    , m_matchFile(matchFile)  // 保存匹配文件路径
    , m_totalMatches(0)       // 初始化匹配点计数
{
    m_sparseMatchFileMissing = m_matchFile.trimmed().isEmpty();

    // 窗口标题显示两张影像的文件名（去除目录部分）
    setWindowTitle(tr("匹配查看：%1 <-> %2")
                   .arg(QFileInfo(imgA).fileName())
                   .arg(QFileInfo(imgB).fileName()));
    resize(1400, 800);  // 初始窗口尺寸：宽 1400px，高 800px

    Ui::MatchViewerDialog form;
    form.setupUi(this);

    m_tabWidget = form.m_tabWidget;
    m_sparseTab = form.m_sparseTab;
    m_denseTab = form.m_denseTab;
    m_statusLabel = form.m_statusLabel;

    m_syncModeChk = form.m_syncModeChk;
    m_fitBtn = form.m_fitBtn;
    m_resetBtn = form.m_resetBtn;
    m_zoomInBtn = form.m_zoomInBtn;
    m_zoomOutBtn = form.m_zoomOutBtn;

    m_lineColorBtn = form.m_lineColorBtn;
    m_lineWidthSpin = form.m_lineWidthSpin;
    m_opacitySlider = form.m_opacitySlider;
    m_maxCountSpin = form.m_maxCountSpin;
    m_showEndPointsChk = form.m_showEndPointsChk;
    m_showOnlyInliersChk = form.m_showOnlyInliersChk;
    m_rainbowChk = form.m_rainbowChk;

    m_denseDisplayGroup = form.m_denseDisplayGroup;
    m_denseOpacitySlider = form.m_denseOpacitySlider;
    m_denseColormapCombo = form.m_denseColormapCombo;
    m_denseAutoRangeChk = form.m_denseAutoRangeChk;
    m_denseMinSpin = form.m_denseMinSpin;
    m_denseMaxSpin = form.m_denseMaxSpin;

    m_denseColormapCombo->clear();
    m_denseColormapCombo->addItem(tr("Jet"), 2);
    m_denseColormapCombo->addItem(tr("Hot"), 11);
    m_denseColormapCombo->addItem(tr("Turbo"), 20);

    m_viewer = new DualImageViewer(m_sparseTab);
    form.sparseLayout->addWidget(m_viewer);

    connect(m_syncModeChk, &QCheckBox::toggled, this, &MatchViewerDialog::onSyncModeToggled);
    connect(m_fitBtn, &QPushButton::clicked, this, &MatchViewerDialog::onFitToView);
    connect(m_resetBtn, &QPushButton::clicked, this, &MatchViewerDialog::onResetView);
    connect(m_zoomInBtn, &QPushButton::clicked, this, &MatchViewerDialog::onZoomIn);
    connect(m_zoomOutBtn, &QPushButton::clicked, this, &MatchViewerDialog::onZoomOut);

    connect(m_lineColorBtn, &QPushButton::clicked, this, &MatchViewerDialog::onLineColorChanged);
    connect(m_lineWidthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MatchViewerDialog::onLineWidthChanged);
    connect(m_opacitySlider, &QSlider::valueChanged, this, &MatchViewerDialog::onOpacityChanged);
    connect(m_maxCountSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MatchViewerDialog::onMaxCountChanged);
    connect(m_showEndPointsChk, &QCheckBox::toggled, this, &MatchViewerDialog::onShowEndPointsToggled);
    connect(m_showOnlyInliersChk, &QCheckBox::toggled, this, &MatchViewerDialog::onShowOnlyInliersToggled);
    connect(m_rainbowChk, &QCheckBox::toggled, this, &MatchViewerDialog::onRainbowToggled);

    connect(m_denseOpacitySlider, &QSlider::valueChanged, this,
            &MatchViewerDialog::onDenseOpacityChanged);
    connect(m_denseColormapCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchViewerDialog::onDenseColormapChanged);
    connect(m_denseAutoRangeChk, &QCheckBox::toggled, this, [this](bool checked)
    {
        m_denseMinSpin->setEnabled(!checked);
        m_denseMaxSpin->setEnabled(!checked);
        if (m_viewer && m_viewer->disparityOverlay())
        {
            m_viewer->disparityOverlay()->setAutoRange(checked);
        }
    });
    connect(m_denseMinSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MatchViewerDialog::onDenseRangeChanged);
    connect(m_denseMaxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MatchViewerDialog::onDenseRangeChanged);

    // 标签页切换处理
    connect(m_tabWidget, &QTabWidget::currentChanged, this, [this](int idx)
    {
        if (m_viewer)
        {
            m_viewer->setOverlayMode(idx);
            // 切换密集显示选项可见性
            if (m_denseDisplayGroup)
                m_denseDisplayGroup->setVisible(idx == 1);
        }
    });
    
    // 设置初始标签页
    m_tabWidget->setCurrentIndex(m_initialTab);

    // 连接 DualImageViewer 的数据加载信号到本对话框的回调槽
    connect(m_viewer, &DualImageViewer::matchDataLoaded,
            this, &MatchViewerDialog::onMatchDataLoaded);
    connect(m_viewer, &DualImageViewer::loadFailed,
            this, &MatchViewerDialog::onLoadFailed);
    
    // 从 project_dialog.json 恢复上次保存的显示参数（若已设置项目路径）
    loadSettings();
    
    // 异步加载影像对及匹配文件（加载完成后触发 matchDataLoaded/loadFailed 信号）
    m_viewer->loadMatchPair(imgA, imgB, matchFile);
}

// 析构函数：在对话框关闭前将当前显示参数持久化到 project_dialog.json
MatchViewerDialog::~MatchViewerDialog()
{
    saveSettings();
}

// forDenseMatch: 工厂方法，创建密集匹配查看对话框
MatchViewerDialog* MatchViewerDialog::forDenseMatch(
    const QString &imgA, const QString &imgB,
    const QString &disparityFile, QWidget *parent)
{
    auto *dlg = new MatchViewerDialog(imgA, imgB, QString(), parent);
    dlg->m_disparityFile = disparityFile;
    dlg->setInitialTab(1);
    if (!disparityFile.isEmpty())
        dlg->m_viewer->disparityOverlay()->loadDisparity(disparityFile);
    return dlg;
}

// setInitialTab: 设置初始显示的标签页索引
void MatchViewerDialog::setInitialTab(int tabIndex)
{
    m_initialTab = tabIndex;
    if (m_tabWidget)
        m_tabWidget->setCurrentIndex(tabIndex);
}

// setProjectPath: 设置项目路径以启用记忆化，并立即加载已保存的设置
void MatchViewerDialog::setProjectPath(const QString &plascanPath)
{
    if (plascanPath.isEmpty()) return;
    if (!m_setting) m_setting = new DialogSettingStore(DialogSettingKeys::MatchViewer, this);
    m_setting->setProjectPath(plascanPath);
    loadSettings();
}

// onSyncModeToggled: 同步缩放/平移开关切换，转发给 DualImageViewer
void MatchViewerDialog::onSyncModeToggled(bool checked)
{
    m_viewer->setSyncMode(checked);
}

// onFitToView: 将左右两张图像都缩放到适合当前视口大小
void MatchViewerDialog::onFitToView()
{
    m_viewer->fitBothViews();
}

// onResetView: 将左右两张图像缩放重置为 100%（原始像素大小）
void MatchViewerDialog::onResetView()
{
    m_viewer->resetBothViews();
}

// onZoomIn: 同时对左右视图执行放大操作
void MatchViewerDialog::onZoomIn()
{
    if (m_viewer->leftView()) m_viewer->leftView()->zoomIn();
    if (m_viewer->rightView()) m_viewer->rightView()->zoomIn();
}

// onZoomOut: 同时对左右视图执行缩小操作
void MatchViewerDialog::onZoomOut()
{
    if (m_viewer->leftView()) m_viewer->leftView()->zoomOut();
    if (m_viewer->rightView()) m_viewer->rightView()->zoomOut();
}

// onLineColorChanged: 弹出颜色选择对话框，将选定颜色应用到 MatchLineOverlay
// 同时更新颜色按钮的背景色以直观显示当前颜色
void MatchViewerDialog::onLineColorChanged()
{
    QColor color = QColorDialog::getColor(m_viewer->overlay()->lineColor(), this, tr("选择线条颜色"));
    if (color.isValid()) {
        m_viewer->overlay()->setLineColor(color);
        // 用内联样式将按钮背景设为所选颜色，方便用户一眼看到当前颜色
        m_lineColorBtn->setStyleSheet(QString("background-color: %1;").arg(color.name()));
    }
}

// onRainbowToggled: 切换五彩斑斓（每条线不同颜色）模式
// 启用时禁用单色颜色按钮，避免与彩虹模式冲突
void MatchViewerDialog::onRainbowToggled(bool checked)
{
    m_viewer->overlay()->setRainbowMode(checked);
    // 当启用五彩斑斓时，将禁用单色选择按钮
    m_lineColorBtn->setEnabled(!checked);
}

// onLineWidthChanged: 更新 MatchLineOverlay 中连线的绘制宽度
void MatchViewerDialog::onLineWidthChanged(double value)
{
    m_viewer->overlay()->setLineWidth(value);
}

// onOpacityChanged: 将滑块的 0–100 整数值转换为 0.0–1.0 浮点透明度并更新覆盖层
void MatchViewerDialog::onOpacityChanged(int value)
{
    m_viewer->overlay()->setOpacity(value / 100.0);
}

// onMaxCountChanged: 限制 MatchLineOverlay 最多显示的连线条数（0 = 无限制）
void MatchViewerDialog::onMaxCountChanged(int value)
{
    m_viewer->overlay()->setMaxDisplayCount(value);
}

// onShowEndPointsToggled: 控制是否在连线两端绘制关键点圆圈
void MatchViewerDialog::onShowEndPointsToggled(bool checked)
{
    m_viewer->overlay()->setShowEndPoints(checked);
}

// onShowOnlyInliersToggled: 控制是否仅显示内点连线（需要先加载内点标记）
void MatchViewerDialog::onShowOnlyInliersToggled(bool checked)
{
    m_viewer->overlay()->setShowOnlyInliers(checked);
}

// onMatchDataLoaded: 匹配数据加载成功的回调，更新总匹配数并刷新状态栏
void MatchViewerDialog::onMatchDataLoaded(int count)
{
    m_totalMatches = count;  // 保存总匹配点数

    // 兼容旧版本默认值（500）：若匹配总数超过500且当前仍为500，
    // 自动切换到“全部显示”，避免用户误以为匹配点丢失。
    if (m_maxCountSpin && m_maxCountSpin->value() == 500 && count > 500)
    {
        m_maxCountSpin->setValue(0);
        if (m_viewer && m_viewer->overlay())
        {
            m_viewer->overlay()->setMaxDisplayCount(0);
        }
    }

    updateStatusBar();       // 立即刷新状态栏文字
}

// onLoadFailed: 匹配数据加载失败的回调，直接将错误信息显示在状态栏
void MatchViewerDialog::onLoadFailed(const QString &error)
{
    m_statusLabel->setText(tr("加载失败：%1").arg(error));
}

// updateStatusBar: 刷新底部状态栏文字，显示当前总匹配点数
// 若后续需要显示可见点数、内点数等，可在此处扩展
void MatchViewerDialog::updateStatusBar()
{
    if (m_sparseMatchFileMissing)
    {
        m_statusLabel->setText(tr("尚未生成匹配：当前仅显示重叠候选影像对"));
        return;
    }

    QString status = tr("总匹配点数：%1").arg(m_totalMatches);
    
    // 可以添加更多统计信息（如可见点数）：
    // int visible = m_viewer->visibleMatchCount();
    // if (visible >= 0) {
    //     status += tr(" | 可见：%1").arg(visible);
    // }
    
    m_statusLabel->setText(status);
}

// loadSettings: 从 project_dialog.json 恢复上次保存的显示参数（项目级记忆化）
// 若记忆化管理器未初始化或无已保存数据，则使用控件默认值
void MatchViewerDialog::loadSettings()
{
    if (!m_setting) return;
    const QJsonObject cfg = m_setting->load();
    if (cfg.isEmpty()) return;

    // 逐项恢复显示选项
    bool syncMode = cfg.value(QStringLiteral("syncMode")).toBool(false);
    m_syncModeChk->setChecked(syncMode);
    
    // 颜色从 "#RRGGBB" 字符串反序列化
    const QString colorStr = cfg.value(QStringLiteral("lineColor")).toString();
    QColor lineColor = colorStr.isEmpty() ? QColor(Qt::yellow) : QColor(colorStr);
    m_viewer->overlay()->setLineColor(lineColor);
    m_lineColorBtn->setStyleSheet(QString("background-color: %1;").arg(lineColor.name()));
    bool rainbow = cfg.value(QStringLiteral("rainbowMode")).toBool(false);
    m_rainbowChk->setChecked(rainbow);
    m_viewer->overlay()->setRainbowMode(rainbow);
    m_lineColorBtn->setEnabled(!rainbow);
    
    double lineWidth = cfg.value(QStringLiteral("lineWidth")).toDouble(1.5);
    m_lineWidthSpin->setValue(lineWidth);
    
    int opacity = cfg.value(QStringLiteral("opacity")).toInt(70);
    m_opacitySlider->setValue(opacity);
    
    int maxCount = cfg.value(QStringLiteral("maxCount")).toInt(0);
    // 兼容旧版本默认值：历史默认是 500，现版本默认应为“全部(0)”。
    if (maxCount == 500)
    {
        maxCount = 0;
    }
    m_maxCountSpin->setValue(maxCount);
    
    bool showEndPoints = cfg.value(QStringLiteral("showEndPoints")).toBool(true);
    m_showEndPointsChk->setChecked(showEndPoints);
}

// saveSettings: 将当前界面上的显示参数持久化到 project_dialog.json（析构时调用）
void MatchViewerDialog::saveSettings()
{
    if (!m_setting) return;

    QJsonObject cfg;
    cfg[QStringLiteral("syncMode")]      = m_syncModeChk->isChecked();
    cfg[QStringLiteral("lineColor")]     = m_viewer->overlay()->lineColor().name();  // "#RRGGBB"
    cfg[QStringLiteral("rainbowMode")]   = m_rainbowChk->isChecked();
    cfg[QStringLiteral("lineWidth")]     = m_lineWidthSpin->value();
    cfg[QStringLiteral("opacity")]       = m_opacitySlider->value();
    cfg[QStringLiteral("maxCount")]      = m_maxCountSpin->value();
    cfg[QStringLiteral("showEndPoints")] = m_showEndPointsChk->isChecked();

    m_setting->save(cfg);
}

// onDenseOpacityChanged: 密集匹配透明度滑块变化
void MatchViewerDialog::onDenseOpacityChanged(int value)
{
    if (m_viewer && m_viewer->disparityOverlay())
        m_viewer->disparityOverlay()->setOpacity(value / 100.0f);
}

// onDenseColormapChanged: 密集匹配色彩映射变化
void MatchViewerDialog::onDenseColormapChanged(int index)
{
    if (m_viewer && m_viewer->disparityOverlay() && m_denseColormapCombo)
    {
        int cmap = m_denseColormapCombo->itemData(index).toInt();
        m_viewer->disparityOverlay()->setColormap(cmap);
    }
}

// onDenseRangeChanged: 密集匹配视差范围微调
void MatchViewerDialog::onDenseRangeChanged()
{
    if (m_viewer && m_viewer->disparityOverlay() && m_denseMinSpin && m_denseMaxSpin)
        m_viewer->disparityOverlay()->setDisparityRange(
            static_cast<float>(m_denseMinSpin->value()),
            static_cast<float>(m_denseMaxSpin->value()));
}
