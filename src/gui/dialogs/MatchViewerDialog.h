// =============================================================================
// 文件: MatchViewerDialog.h
// 功能: 特征点匹配查看器对话框声明
// 职责:
//   - 使用 DualImageViewer 并排展示两张影像及其匹配连接线
//   - 提供工具栏控制同步缩放、适应窗口、放大/缩小等操作
//   - 提供显示选项组：线条颜色、宽度、透明度、最大显示数、五彩斑斓模式
//   - 状态栏实时显示当前匹配点总数
//   - 支持通过 project_dialog.json 持久化用户配置（项目级）
// =============================================================================
#pragma once

#include <QDialog>
#include <QString>

// 前向声明，减少头文件依赖
class DualImageViewer;  // 双图并列查看器（左右各一个 ImageViewWidget）
class QLabel;           // 状态栏文字标签
class QCheckBox;        // 复选框控件
class QPushButton;      // 按钮控件
class QSlider;          // 滑块控件（透明度）
class QSpinBox;         // 整型数值输入框
class QDoubleSpinBox;   // 浮点数值输入框（线宽）
class DialogSettingStore; // 项目级记忆化管理器
class QTabWidget;       // 标签页控件
class QComboBox;        // 下拉框控件
class QGroupBox;        // 分组框控件

// =============================================================================
// MatchViewerDialog — 特征点匹配结果双图查看对话框
//
// 使用方法：
//   MatchViewerDialog *dlg = new MatchViewerDialog(imgA, imgB, matchFile, this);
//   dlg->show();
//
// 主要功能分组：
//   1. 工具栏：同步缩放开关、适应窗口、重置、放大/缩小
//   2. 显示选项区（嵌入工具栏）：线条颜色/宽度/透明度/最大显示数/端点/彩虹模式/内点过滤
//   3. 状态栏：显示总匹配点数及错误信息
// =============================================================================
class MatchViewerDialog : public QDialog
{
    Q_OBJECT

public:
    // 构造函数
    // imgA      — 左侧影像的完整路径
    // imgB      — 右侧影像的完整路径
    // matchFile — 二进制 .match 格式的匹配结果文件路径
    // parent    — 父窗口指针
    explicit MatchViewerDialog(const QString &imgA, const QString &imgB,
                               const QString &matchFile, QWidget *parent = nullptr);

    static MatchViewerDialog* forDenseMatch(
        const QString &imgA, const QString &imgB,
        const QString &disparityFile, QWidget *parent = nullptr);

    void setInitialTab(int tabIndex);

    // 析构函数：保存当前显示设置到 project_dialog.json
    ~MatchViewerDialog() override;

    /**
     * @brief 设置当前项目的 .plascan 路径以启用项目级记忆化。
     *
     * 若未调用此方法，对话框仍可正常使用，只是不会持久化显示参数。
     *
     * @param plascanPath 当前项目 .plascan 文件的绝对路径。
     */
    void setProjectPath(const QString &plascanPath);

private slots:
    // ── 工具栏操作槽 ──────────────────────────────────────────────
    // 同步模式切换：checked=true 时左右视图缩放/平移联动
    void onSyncModeToggled(bool checked);
    // 将左右图像同时缩放到适合当前窗口大小
    void onFitToView();
    // 重置左右图像缩放比例到 100%
    void onResetView();
    // 同时放大左右图像一档
    void onZoomIn();
    // 同时缩小左右图像一档
    void onZoomOut();

    // ── 显示选项槽 ────────────────────────────────────────────────
    // 弹出颜色选择对话框，更新匹配连线颜色
    void onLineColorChanged();
    // 连线宽度变化（单位：像素，由 m_lineWidthSpin 触发）
    void onLineWidthChanged(double value);
    // 连线整体透明度变化（0–100，由 m_opacitySlider 触发）
    void onOpacityChanged(int value);
    // 最大显示匹配点数限制变化（0 表示全部显示，由 m_maxCountSpin 触发）
    void onMaxCountChanged(int value);
    // 是否显示匹配点端点圆圈（由 m_showEndPointsChk 触发）
    void onShowEndPointsToggled(bool checked);
    // 是否仅显示内点（需要 bundle_adjust 后的内点标记，由 m_showOnlyInliersChk 触发）
    void onShowOnlyInliersToggled(bool checked);
    // 五彩斑斓模式开关：每条匹配线使用不同颜色（由 m_rainbowChk 触发）
    void onRainbowToggled(bool checked);

    // ── 数据加载回调槽 ────────────────────────────────────────────
    // 匹配数据加载成功后触发，count 为总匹配点数
    void onMatchDataLoaded(int count);
    // 匹配数据加载失败后触发，error 为错误描述文字
    void onLoadFailed(const QString &error);

    // 刷新底部状态栏文字（含总匹配点数等统计信息）
    void updateStatusBar();

    // ── 密集显示选项槽 ────────────────────────────────────────────
    void onDenseOpacityChanged(int value);
    void onDenseColormapChanged(int index);
    void onDenseRangeChanged();

private:
    // ── 设置持久化 ────────────────────────────────────────────────
    // 从 project_dialog.json 恢复上次保存的显示参数（项目级）
    void loadSettings();
    // 将当前显示参数保存到 project_dialog.json（项目级）
    void saveSettings();

    // ── 核心控件 ──────────────────────────────────────────────────
    DualImageViewer *m_viewer;  // 双图查看器，持有左右 ImageViewWidget 和 MatchLineOverlay

    // ── 标签页控件 ──────────────────────────────────────────────────
    QTabWidget        *m_tabWidget     = nullptr;
    QWidget           *m_sparseTab     = nullptr;
    QWidget           *m_denseTab      = nullptr;
    int                m_initialTab    = 0;
    QString            m_disparityFile;
    bool               m_sparseMatchFileMissing = false;

    // ── 工具栏及工具按钮 ──────────────────────────────────────────
    QCheckBox   *m_syncModeChk;    // 同步缩放/平移开关
    QPushButton *m_fitBtn;         // 适应窗口按钮
    QPushButton *m_resetBtn;       // 重置到 100% 按钮
    QPushButton *m_zoomInBtn;      // 放大按钮（显示 "+"）
    QPushButton *m_zoomOutBtn;     // 缩小按钮（显示 "-"）

    // ── 显示选项控件（嵌入工具栏的 QGroupBox） ────────────────────
    QPushButton  *m_lineColorBtn;       // 连线颜色选择按钮（背景色即当前颜色）
    QDoubleSpinBox *m_lineWidthSpin;    // 连线宽度（0.5–10.0，步长 0.5）
    QSlider      *m_opacitySlider;      // 透明度滑块（0–100）
    QSpinBox     *m_maxCountSpin;       // 最大显示连线数（0 = 全部）
    QCheckBox    *m_showEndPointsChk;   // 显示端点开关
    QCheckBox    *m_showOnlyInliersChk; // 仅显示内点开关（默认禁用）
    QCheckBox    *m_rainbowChk;         // 五彩斑斓模式开关

    // ── 密集显示选项控件 ──────────────────────────────────────────
    QSlider           *m_denseOpacitySlider = nullptr;
    QComboBox         *m_denseColormapCombo = nullptr;
    QCheckBox         *m_denseAutoRangeChk  = nullptr;
    QDoubleSpinBox    *m_denseMinSpin       = nullptr;
    QDoubleSpinBox    *m_denseMaxSpin       = nullptr;
    QGroupBox         *m_denseDisplayGroup  = nullptr;

    // ── 状态栏 ────────────────────────────────────────────────────
    QLabel *m_statusLabel;  // 底部状态信息标签

    // ── 数据成员 ──────────────────────────────────────────────────
    QString m_matchFile;    // 当前加载的 .match 文件路径
    int m_totalMatches;     // 已加载的总匹配点数（加载成功后更新）
    DialogSettingStore *m_setting = nullptr; ///< 项目级记忆化管理器（可空）
};
