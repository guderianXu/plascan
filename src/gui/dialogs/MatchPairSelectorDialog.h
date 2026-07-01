// =============================================================================
// 文件: MatchPairSelectorDialog.h
// 说明: 匹配对选择器对话框的声明（类似 Metashape 在线比对面板）。
//       通过下拉框选择当前查看的影像，在表格中展示与其存在匹配关系的所有
//       其他影像及匹配点统计（总数/有效/无效），支持双击或按钮打开详细
//       匹配视图（MatchViewerDialog）。
// =============================================================================
#pragma once

#include <QDialog>
#include <QString>
#include <QStringList>
#include <QJsonObject>
#include <QTimer>
#include <QMap>
#include <QSet>
#include <QVector>

#include "MatchResultCatalog.h"

// Qt 控件前向声明
class QComboBox;
class QTableWidget;
class QPushButton;
class QLabel;
class ProjectManager;

// MatchPairSelectorDialog: 匹配对选择器（类似 Metashape）
// 功能：
// - 顶部下拉框：选择当前查看的图像
// - 中间表格：显示与该图像匹配的所有其他图像
//   列：图像名称、最佳算法、有效内点、总匹配、可用算法、状态
// - 底部按钮：查看详细匹配（打开 MatchViewerDialog）
class MatchPairSelectorDialog : public QDialog
{
    Q_OBJECT

public:
    // 构造函数，传入项目管理器以读取影像列表和匹配元数据
    // projectManager — 项目管理器指针
    // parent         — 父窗口指针
    explicit MatchPairSelectorDialog(ProjectManager *projectManager, QWidget *parent = nullptr);
    ~MatchPairSelectorDialog() override;

private slots:
    // 当前图像下拉框索引改变时，重新加载该影像的匹配对列表
    void onCurrentImageChanged(int index);
    
    // 表格行被单击时，更新选中状态并启用"查看详细匹配"按钮
    void onMatchPairSelected(int row, int column);
    
    // 表格行被双击时，直接打开 MatchViewerDialog 查看详细匹配
    void onMatchPairDoubleClicked(int row, int column);
    
    // 点击"查看详细匹配"按钮时，打开 MatchViewerDialog
    void onViewDetailedMatch();
    
    // 点击"刷新"按钮时，重新加载项目影像列表及当前影像的匹配对
    void onRefresh();

    // 收到 ProjectManager::matchPairReady 信号时，触发防抖刷新
    void scheduleRefresh();

private:
    // 构建并初始化整体界面布局（顶部、中间、底部）
    void setupUI();
    // 初始化匹配对表格（列头、列宽、选择模式等）
    void setupTable();
    // 从项目管理器加载所有影像并填充下拉框
    void loadProjectImages();
    // 为指定影像路径加载其所有匹配对，并填充到表格中
    void loadMatchPairsForImage(const QString &imagePath);
    
    // 匹配信息结构体，描述与某张影像之间的匹配统计
    struct MatchInfo {
        QString imagePath;       // 匹配影像的完整路径
        QString imageName;       // 匹配影像的文件名（用于显示）
        QString algorithm;       // 匹配算法名 (superglue/lightglue/loftr/...)
        int totalPoints;         // 总匹配点数
        int validPoints;         // 有效（内点）匹配点数
        int invalidPoints;       // 无效（外点）匹配点数
        QString matchFilePath;   // 对应 .match 文件的完整路径
        QVector<xjw::pipeline::MatchVariant> variants; // 同一影像对的全部算法结果
        bool hasInlierStats = false; // true 表示 validPoints 来自几何验证内点统计
        int compatibleVariantCount = 0; // 可由查看器加载的算法结果数量
        QString availableAlgorithms;    // 可用算法列表（用于表格显示）
        QString status;                 // 当前行状态说明
        bool overlapCandidate = false; // true 表示来自重叠对规划，尚无匹配文件
        double overlapScore = 0.0;      // 重叠评分（若输出中提供）
        QString overlapSource;          // overlap json/lis 文件路径
    };
    
    // 解析指定影像与其他影像的所有匹配信息
    // imagePath — 当前选中影像的完整路径
    // 返回值    — 所有与之存在匹配关系的 MatchInfo 列表
    QList<MatchInfo> parseMatchDataForImage(const QString &imagePath);

    QList<MatchInfo> loadOverlapCandidatesForImage(const QString &imagePath,
                                                   const QSet<QString> &seenPairKeys,
                                                   const QMap<QString, QString> &baseToPath) const;
    
    // 在元数据和文件系统中查找两张影像对应的 .match 文件路径
    // imgA, imgB — 两张影像的完整路径
    // 返回值     — 找到的 .match 文件路径，未找到返回空字符串
    QString findMatchFile(const QString &imgA, const QString &imgB);
    
    // 读取 .match 文件，统计匹配点数量并返回 MatchInfo
    // imgA      — 参考影像路径（仅用于上下文，不直接使用）
    // imgB      — 匹配影像路径
    // matchFile — .match 文件路径
    // 返回值    — 包含统计结果的 MatchInfo 结构体
    MatchInfo getMatchStatistics(const QString &imgA, const QString &imgB, 
                                 const QString &matchFile);

private:
    // 项目管理器指针，用于获取影像列表和匹配元数据
    ProjectManager *_projectManager;
    
    // 顶部影像选择下拉框
    QComboBox *_imageComboBox;
    // 匹配对信息表格（图像名 | 总计 | 有效 | 无效）
    QTableWidget *_matchTable;
    // 打开详细匹配查看器的按钮（有选中行时启用）
    QPushButton *_viewDetailBtn;
    // 刷新数据的按钮
    QPushButton *_refreshBtn;
    // 底部状态标签，显示当前匹配对数或选中信息
    QLabel *_statusLabel;
    
    // 所有影像的完整路径列表
    QStringList _allImages;
    // 当前选中的影像路径
    QString _currentImage;
    // 当前影像的所有匹配信息列表
    QList<MatchInfo> _currentMatches;
    // 当前在表格中选中的匹配对索引（-1 表示无选中）
    int _selectedMatchIndex;

    // matches 目录路径（用于文件系统扫描，替代读取大 JSON 元数据）
    QString _matchDir;
    // 防抖刷新计时器（300ms 内多次触发只刷新一次）
    QTimer *_refreshTimer = nullptr;
};
