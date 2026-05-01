// =============================================================================
// 文件: WorkflowReportDialog.h
// 功能: 工作流程历史报告查看器
//
//   显示三个标签页：
//   - 空中三角测量（AT）：含平差精度对比折线/柱状图、相机注册率、重投影误差等
//   - 稠密点云生成：点数统计、文件路径
//   - 三维模型生成：顶点/面片数、文件路径
//
//   报告数据从项目资产目录下的 reports/*.json 文件读取，
//   由工作流完成时由 MenuWorkflowController 写入。
// =============================================================================
#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QJsonArray>
#include <QColor>
#include <QFont>
#include <QString>
#include <vector>

class QTabWidget;
class QLabel;
class QWidget;
class QScrollArea;
class QPushButton;
class QVBoxLayout;
class QComboBox;

// ─────────────────────────────────────────────────────────────────────────────
// ReportChartWidget — 轻量级内联图表控件（QPainter 自绘，无须 Qt Charts 依赖）
// ─────────────────────────────────────────────────────────────────────────────
class ReportChartWidget : public QWidget
{
    Q_OBJECT
public:
    enum class ChartType {
        BarComparison,   // 横向双柱比较（平差前/后）
        BarSeries,       // 纵向柱状序列
        ArcProgress,     // 弧形进度（注册率）
    };

    explicit ReportChartWidget(ChartType type, QWidget *parent = nullptr);

    // 设置横向双柱比较数据
    // labels: 各列名称, before/after: 对应数值, unit: 数值单位
    void setComparisonData(const QStringList &labels,
                           const std::vector<double> &before,
                           const std::vector<double> &after,
                           const QString &unit = "px");

    // 设置弧形进度
    void setArcData(double value, double total, const QString &label);

    // 设置柱状序列
    void setBarSeriesData(const QStringList &labels,
                          const std::vector<double> &values,
                          const QString &unit = "");

    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void drawBarComparison(QPainter &p);
    void drawArcProgress(QPainter &p);
    void drawBarSeries(QPainter &p);

    ChartType               m_type;
    QStringList             m_labels;
    std::vector<double>     m_before;
    std::vector<double>     m_after;
    std::vector<double>     m_values;
    double                  m_arcValue  = 0.0;
    double                  m_arcTotal  = 1.0;
    QString                 m_arcLabel;
    QString                 m_unit;
};

// ─────────────────────────────────────────────────────────────────────────────
// WorkflowReportDialog
// ─────────────────────────────────────────────────────────────────────────────
class WorkflowReportDialog : public QDialog
{
    Q_OBJECT
public:
    explicit WorkflowReportDialog(const QString &projectAssetsDir,
                                  QWidget *parent = nullptr);

    // 强制刷新（重新从磁盘读取报告文件）
    void refresh();

private:
    void buildUi();
    QWidget *buildAtTab();
    QWidget *buildAtReportPage(const QJsonObject &report);
    QWidget *buildDenseTab();
    QWidget *buildMeshTab();

    // 从 assets/reports/at_report.json 等读取
    QJsonObject loadReport(const QString &name) const;
    QJsonArray loadReportHistory(const QString &name) const;

    // 辅助：创建带标题、数值和颜色的统计卡片
    static QWidget *makeStatCard(const QString &title,
                                  const QString &value,
                                  const QString &subtitle = QString(),
                                  const QColor &accent = QColor(70, 130, 210));

    // 辅助：创建横向分割线
    static QWidget *makeSeparator();

    // 辅助：创建形如 "键: 值" 的行
    static QLabel *makeKVLabel(const QString &key, const QString &value);

    QString         m_assetsDir;
    QTabWidget     *m_tabs      = nullptr;
    QPushButton    *m_refreshBtn= nullptr;
};
