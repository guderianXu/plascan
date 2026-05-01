#pragma once

#include <QDialog>
#include <QJsonObject>
#include <QStringList>

// 前置声明，减少头文件依赖
class QListWidget;
class QComboBox;
class QCheckBox;
class QLineEdit;
class QPushButton;
class QLabel;
class QDialogButtonBox;
class QDoubleSpinBox;
class QSpinBox;

/**
 * @brief 空中三角测量（SFM）对话框类
 * @details 保留原始核心逻辑：线程数手动选择、影像列表复选框；
 *          仅新增：输出目录浏览按钮、影像数量提示、信号防抖、布局优化
 */
class AerialTriangulationDialog : public QDialog
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数
     * @param parent 父窗口指针
     */
    explicit AerialTriangulationDialog(QWidget *parent = nullptr);

    /**
     * @brief 设置可用影像列表
     * @param images 影像文件路径列表
     */
    void setAvailableImages(const QStringList &images);

    /**
     * @brief 设置默认输出目录
     * @param outputDir 默认输出目录路径
     */
    void setDefaultOutputDir(const QString &outputDir);

    /**
     * @brief 应用保存的配置参数
     * @param settings 配置参数JSON对象
     */
    void applySettings(const QJsonObject &settings);

signals:
    /**
     * @brief 配置参数变化信号（仅在参数真变化时触发）
     * @param settings 最新的配置参数
     */
    void settingsChanged(const QJsonObject &settings);

    /**
     * @brief 开始处理请求信号
     * @param settings 最终确认的配置参数
     */
    void runRequested(const QJsonObject &settings);

private slots:
    /**
     * @brief 点击OK按钮触发的处理函数
     * @details 校验参数合法性，触发处理请求，关闭对话框
     */
    void onRun();

    /**
     * @brief 任意参数变化时的统一处理函数
     * @details 收集最新配置并触发settingsChanged信号（做防抖处理）
     */
    void onAnyChanged();

    /**
     * @brief 浏览输出目录按钮点击事件
     * @details 弹出目录选择对话框，更新输出目录输入框
     */
    void onBrowseOutputDir();

    /**
     * @brief 影像列表复选状态变化事件
     * @details 更新已选影像数量提示，更新OK按钮启用状态
     */
    void onImageCheckStateChanged();

    /**
     * @brief 一键全选/取消槽
     */
    void onSelectAll();
    void onDeselectAll();

    /**
     * @brief 影像列表右键菜单处理
     */
    void onImageListContextMenu(const QPoint &pos);

    /**
     * @brief 在文件管理器中打开所选影像的所在目录
     */
    void onOpenContainingFolder();

    /**
     * @brief 使用系统默认程序打开所选影像
     */
    void onOpenImageExternally();

private:
    /**
     * @brief 收集当前所有配置参数
     * @return 包含所有配置的JSON对象
     */
    QJsonObject collectSettings() const;

    /**
     * @brief 初始化界面布局
     * @details 封装布局初始化逻辑，保留原始核心控件
     */
    void initUI();

    // 界面控件成员变量（严格保留原始逻辑）
    QListWidget *m_imageList = nullptr;          // 影像列表控件（复选框模式）
    QPushButton *m_selectAllBtn = nullptr;      // 一键全选按钮
    QPushButton *m_deselectAllBtn = nullptr;    // 一键取消按钮
    QComboBox *m_qualityCombo = nullptr;         // 重建精度下拉框（低/中/高/最高）
    QComboBox *m_cameraOptCombo = nullptr;       // 相机优化路径下拉框（SFM/光束法平差/自动）
    QCheckBox *m_stepCompletionCheck = nullptr;  // 步骤完成保存复选框（保留）
    QCheckBox *m_resetCurrentAlignCheck = nullptr; // 重置当前对齐复选框
    QLineEdit *m_outputDirEdit = nullptr;        // 输出目录输入框
    QPushButton *m_browseBtn = nullptr;          // 输出目录浏览按钮（新增）
    QLabel *m_imageCountLabel = nullptr;         // 已选影像数量提示标签（新增）
    QComboBox *m_threadsCombo = nullptr;         // 线程数下拉框（保留手动选择）
    QDialogButtonBox *m_btnBox = nullptr;        // 对话框按钮盒（OK/Cancel）
    QJsonObject m_lastSettings;                  // 上一次发送的配置参数（防抖用）

    // 内方位元素控件
    QCheckBox  *m_intrinsicsCheck = nullptr;      // 启用手动输入内参
    QDoubleSpinBox *m_fuSpin  = nullptr;
    QDoubleSpinBox *m_fvSpin  = nullptr;
    QDoubleSpinBox *m_cuSpin  = nullptr;
    QDoubleSpinBox *m_cvSpin  = nullptr;
    QDoubleSpinBox *m_pitchSpin = nullptr;        // 像元大小 (mm)

    // 光束法平差参数控件（高级选项）
    QCheckBox      *m_baRefinePoseCheck   = nullptr; ///< 是否优化相机位姿
    QSpinBox       *m_baMaxIterSpin       = nullptr; ///< 外层最大迭代次数
    QDoubleSpinBox *m_baHuberDeltaSpin    = nullptr; ///< Huber 鲁棒损失阈值(px)
    QDoubleSpinBox *m_baFilterReprojSpin  = nullptr; ///< 离群点过滤阈值(px)
    QDoubleSpinBox *m_baDampingSpin       = nullptr; ///< LM 阻尼因子
};
