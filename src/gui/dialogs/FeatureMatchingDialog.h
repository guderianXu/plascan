// =============================================================================
// 文件: FeatureMatchingDialog.h
// 功能: 特征匹配配置对话框声明
// 职责:
//   - 管理输入影像对（通过 lis 文件或自动生成全组合）
//   - 提供基础参数：模型类型（室内/室外）、粗差剔除算法、匹配阈值、最大关键点数
//   - 提供高级参数（可折叠）：Sinkhorn 迭代、批次大小、RANSAC 参数、输入尺寸
//   - 提供系统参数（可折叠）：运行设备（CPU/CUDA）、线程数
//   - 提供调试参数（可折叠）：保存 CSV/可视化、冗长日志
//   - 通过 runRequested 信号将参数上传给 ProjectManager 异步执行
//   - 通过 settingsChanged 信号实时持久化 UI 配置
// =============================================================================
// 特征匹配配置对话框
// 提供基础、高级、系统、调试参数的分组配置界面

#pragma once

#include <QDialog>
#include <QJsonObject>

class QListWidget;
class QLineEdit;
class QPushButton;
class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QGroupBox;
class QTextEdit;
class QStackedWidget;

namespace Ui { class FeatureMatchingDialog; }

class FeatureMatchingDialog : public QDialog 
{
    Q_OBJECT
public:
    explicit FeatureMatchingDialog(QWidget *parent = nullptr);
    ~FeatureMatchingDialog() override;

signals:
    // 用户点击"运行"时发出
    // 对话框只负责收集参数，不直接执行匹配
    // 上层（ProjectManager）收到后负责异步执行和持久化
    void runRequested(const QJsonObject &config, const QStringList &imagePairs);
    
    // 当参数被修改时立即发出（用于项目配置持久化）
    void settingsChanged(const QJsonObject &settings);

    // 用户点击"查看匹配"时发出（上层负责打开 MatchPairSelectorDialog）
    void viewMatchesRequested();

private slots:
    // 全选所有特征点文件
    void onSelectAll();
    // 清除所有特征点文件的选中状态
    void onDeselectAll();
    // 点击"添加 lis 文件"按钮，弹出文件选择对话框并解析 lis 文件中的影像路径
    void onAddLisFile();
    // 清空当前已加载的 lis 文件路径及生成的匹配对列表
    void onClearLis();
    // 根据 m_availableFeatures 自动生成所有两两影像组合并填入 m_currentPairs
    void onGeneratePairs();
    // 点击"浏览"按钮，弹出目录选择对话框并设置匹配结果输出路径
    void onBrowseOutput();
    // 点击"运行"按钮，收集参数发出 runRequested 信号（不在此处执行匹配）
    void onRun();
    // 点击"取消"按钮，关闭对话框（reject）
    void onCancel();
    // 点击"恢复默认"按钮，将所有控件恢复为内置默认参数值
    void onResetDefaults();
    // 点击"查看匹配"按钮，在项目视图中打开当前选定影像对的匹配结果
    void onViewMatches();
    // 算法切换时更新参数控件的启用状态
    void onAlgorithmChanged(int index);

public slots:
    // 将外部传入的 JSON 设置对象应用到各控件（由 ProjectManager 加载项目时调用）
    // settings — 与 collectSettings() 返回格式相同的 JSON 对象
    void applySettings(const QJsonObject &settings);

    // 将项目中已检测到的特征点文件路径列表传入对话框，填充 m_featureList
    // featurePaths — SuperPoint 输出的 .pt 特征文件路径列表
    void setProjectFeatures(const QStringList &featurePaths);

    // 将项目中的原始影像路径传入对话框，供 LoFTR/RoMa 使用
    void setProjectImages(const QStringList &imagePaths);

private:
    // 构建对话框全部控件和分组布局（基础/高级/系统/调试参数组 + 输入输出区 + 按钮）
    void setupUi();
    // 连接各控件的变化信号到 emitSettingsNow 等槽，实现实时配置持久化
    void setupConnections();
    // 根据 m_currentPairs 刷新 m_pairPreview 文本框的预览内容
    void updatePreview();
    // 收集当前全部控件状态并通过 settingsChanged 信号立即发出（用于自动保存）
    void emitSettingsNow();
    // 将界面所有控件状态序列化为 JSON 对象（用于持久化和 runRequested 信号）
    QJsonObject collectSettings() const;
    // 解析指定 lis 文件，返回其中包含的影像路径列表（每行一个路径，忽略空行和注释）
    // lisPath — lis 文件的绝对路径
    QStringList parseLisFile(const QString &lisPath) const;
    // 将 m_availableFeatures 中所有特征文件两两组合，返回 "img1__img2" 格式字符串列表
    QStringList generateAllPairs() const;

    // ── 输入输出控件 ──────────────────────────────────────────────
    QPushButton*      m_selectAllBtn{nullptr};     // 全选按钮
    QPushButton*      m_deselectAllBtn{nullptr};   // 清除按钮
    QListWidget*      m_featureList{nullptr};      // 项目中可用特征点文件的列表控件
    QListWidget*      m_imageList{nullptr};        // 原始影像列表（LoFTR/RoMa 使用）
    QWidget*          m_featureInputWidget{nullptr};  // 特征文件输入区（可切换显示）
    QWidget*          m_imageInputWidget{nullptr};    // 影像输入区（LoFTR/RoMa）
    QTextEdit*        m_pairPreview{nullptr};      // 当前匹配对的预览文本框（只读）
    QLineEdit*        m_lisFileLine{nullptr};      // lis 文件路径输入框
    QPushButton*      m_addLisBtn{nullptr};        // "添加 lis 文件"按钮
    QPushButton*      m_clearLisBtn{nullptr};      // "清空"匹配对按钮
    QPushButton*      m_generatePairsBtn{nullptr}; // "自动生成全组合"按钮
    QLineEdit*        m_outputLine{nullptr};       // 匹配结果输出目录输入框
    QPushButton*      m_browseOutBtn{nullptr};     // "浏览"输出目录按钮

    // ── 基础参数控件（Basic） ─────────────────────────────────────
    QComboBox*        m_matchAlgorithmCombo{nullptr}; // 匹配算法选择
    QComboBox*        m_outlierMethodCombo{nullptr};  // 粗差剔除算法（始终显示）
    QSpinBox*         m_maxKeypointsSpin{nullptr};    // 最大关键点数（始终显示）

    // ── 算法参数切换面板 ──────────────────────────────────────────
    QStackedWidget*   m_paramStack{nullptr};          // Page 0=SuperGlue, 1=LightGlue, 2=传统, 3=LoFTR, 4=RoMa

    // SuperGlue 面板（Page 0）
    QComboBox*        m_modelTypeCombo{nullptr};      // 模型类型：indoor / outdoor
    QDoubleSpinBox*   m_matchThresholdSpin{nullptr};  // 匹配置信度阈值
    QSpinBox*         m_sinkhornIterSpin{nullptr};    // Sinkhorn 正则化迭代次数
    QSpinBox*         m_batchSizeSpin{nullptr};       // 推理批次大小
    QSpinBox*         m_inputWidthSpin{nullptr};      // 推理输入宽度
    QSpinBox*         m_inputHeightSpin{nullptr};     // 推理输入高度

    // LightGlue 面板（Page 1）— 独立控件，避免与 SuperGlue 面板共享状态
    QDoubleSpinBox*   m_lgMatchThresholdSpin{nullptr};
    QSpinBox*         m_lgBatchSizeSpin{nullptr};
    QSpinBox*         m_lgInputWidthSpin{nullptr};
    QSpinBox*         m_lgInputHeightSpin{nullptr};

    // LoFTR 面板（Page 3）
    QComboBox*        m_loftrModelTypeCombo{nullptr};
    QDoubleSpinBox*   m_loftrMatchThresholdSpin{nullptr};

    // RoMa 面板（Page 4）
    QComboBox*        m_romaModelTypeCombo{nullptr};
    QDoubleSpinBox*   m_romaMatchThresholdSpin{nullptr};
    QSpinBox*         m_romaMaxKeypointsSpin{nullptr};

    // ── 高级参数控件（Advanced，可折叠 QGroupBox） ────────────────
    QGroupBox*        m_advancedGroup{nullptr};
    QDoubleSpinBox*   m_outlierReprojSpin{nullptr};
    QDoubleSpinBox*   m_outlierConfidenceSpin{nullptr};
    QSpinBox*         m_outlierMaxItersSpin{nullptr};
    QSpinBox*         m_outlierMinInliersSpin{nullptr};

    // ── 系统参数控件（System，可折叠 QGroupBox） ──────────────────
    QGroupBox*        m_systemGroup{nullptr};     // 系统参数折叠组
    QComboBox*        m_deviceCombo{nullptr};     // 推理设备选择：cpu / cuda:0 等
    QSpinBox*         m_numThreadsSpin{nullptr};     // CPU 推理线程数
    QSpinBox*         m_cudaParallelSpin{nullptr};  // CUDA 模式并行对数（1-8）
    // ── 调试参数控件（Debug，可折叠 QGroupBox） ───────────────────
    QGroupBox*        m_debugGroup{nullptr};   // 调试参数折叠组
    QCheckBox*        m_saveCsvChk{nullptr};   // 是否保存匹配结果为 CSV 文件
    QCheckBox*        m_saveVisChk{nullptr};   // 是否保存匹配可视化图像
    QCheckBox*        m_verboseChk{nullptr};   // 是否输出详细调试日志

    // ── 底部操作按钮 ──────────────────────────────────────────────
    QPushButton*      m_runBtn{nullptr};         // "运行"按钮：发出 runRequested 信号
    QPushButton*      m_viewMatchesBtn{nullptr}; // "查看匹配"按钮：在项目视图中打开匹配结果
    QPushButton*      m_cancelBtn{nullptr};      // "取消"按钮：关闭对话框
    QPushButton*      m_resetBtn{nullptr};       // "恢复默认"按钮：重置所有参数
    
    // ── 数据成员 ──────────────────────────────────────────────────
    QStringList       m_availableFeatures;  // 项目中已探测到的特征点文件路径列表
    QStringList       m_currentPairs;       // 当前待匹配的影像对列表（格式："img1__img2"）
};
