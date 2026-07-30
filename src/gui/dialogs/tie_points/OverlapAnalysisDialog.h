// =============================================================================
// 文件: OverlapAnalysisDialog.h
// 说明: 影像重叠度分析对话框的声明。
//       根据相机参数和 DEM（或固定高程面），计算项目中任意两张影像的投影覆盖
//       区域，给出中心距离、重叠评分及是否重叠等指标。
// =============================================================================
#pragma once

#include <QDialog>

// 前向声明，避免头文件循环依赖
class ProjectManager;
class QListWidget;
class QLineEdit;
class QCheckBox;
class QDoubleSpinBox;
class QTableWidget;
class QLabel;

// OverlapAnalysisDialog — 影像重叠度获取对话框
// 从项目元数据中读取影像及相机参数，结合 DEM 或固定高程计算两两影像的重叠度。
class OverlapAnalysisDialog : public QDialog
{
    Q_OBJECT
public:
    // 构造函数，传入项目管理器以访问影像和相机元数据
    // projectManager — 项目管理器指针，用于获取影像列表和相机参数
    // parent         — 父窗口指针
    explicit OverlapAnalysisDialog(ProjectManager *projectManager, QWidget *parent = nullptr);

private slots:
    // 点击"浏览"按钮时，弹出文件选择对话框，用于选取 DEM（XYZ 格式）文件路径
    void browseDemPath();
    // 点击"计算重叠度"按钮时，收集参数并执行 OverlapAnalyzer 分析
    void runAnalysis();

private:
    // 从项目元数据中加载影像列表，填充 _imageList 复选框列表
    void loadProjectImages();

    // 项目管理器指针，用于读取影像路径和相机参数
    ProjectManager *_projectManager = nullptr;

    // 影像复选框列表控件，支持多选；每项 UserRole 存储完整路径
    QListWidget *_imageList = nullptr;
    // DEM 文件路径输入框（XYZ 格式点云文件）
    QLineEdit *_demPathEdit = nullptr;
    // 是否使用固定高程 Z 的复选框（勾选后忽略 DEM，改用固定 Z 值）
    QCheckBox *_useFixedZCheck = nullptr;
    // 固定高程 Z 值微调框（_useFixedZCheck 打勾时生效）
    QDoubleSpinBox *_fixedZSpin = nullptr;
    // 邻域系数微调框，用于控制重叠评分的邻域范围系数
    QDoubleSpinBox *_neighborSpin = nullptr;

    // 分析结果摘要标签，显示统计描述文字
    QLabel *_summaryLabel = nullptr;
    // 分析结果表格，列：影像A | 影像B | 中心距(m) | 重叠评分 | 是否重叠
    QTableWidget *_resultTable = nullptr;
};
