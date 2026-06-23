// =============================================================================
// 文件: DenseCloudDialog.h
// 模块: GUI / 对话框
// 说明:
//   稠密点云生成对话框（MVS 流程入口）。
//
//   功能:
//     - 选择输入影像对 / 影像列表（来自项目）
//     - 配置 SGBM 深度图参数（视差范围、块大小、唯一性等）
//     - 配置稠密点云参数（最低置信度、法向量、颜色）
//     - 配置网格重建参数（可选）
//     - 实时显示进度和日志
//     - 结果保存路径配置
//
//   对应处理流程：
//     影像对 + 相机位姿（来自 AT 结果）
//       → StereoRectifier → StereoDepthEstimator → DenseCloudBuilder
//       → （可选）MeshReconstructor → 输出 PLY / XYZ
// =============================================================================
#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLineEdit;
class QPushButton;
class QProgressBar;
class QTextEdit;
class ProjectManager;

class DenseCloudDialog : public QDialog
{
    Q_OBJECT
public:
    explicit DenseCloudDialog(ProjectManager *projectManager, QWidget *parent = nullptr);

    /// 应用外部持久化设置（记忆化）
    void applySettings(const QJsonObject &settings);
    /// 收集当前界面参数为 JSON
    QJsonObject collectSettings() const;

signals:
    void settingsChanged(const QJsonObject &settings);
    void runRequested(const QJsonObject &settings);

private slots:
    void onBrowseOutput();
    void onRun();
    void onCancel();
    void onAnyChanged();
    void onPresetChanged(int index);

private:
    void setupUi();
    void appendLog(const QString &line);

    ProjectManager *_projectManager = nullptr;

    // ---- 基础参数 ----
    QComboBox *_imagePairCombo = nullptr;  ///< 影像对选择
    QComboBox *_atResultCombo = nullptr;  ///< AT 结果选择（提供相机位姿）
    QLineEdit *_outputDirEdit = nullptr;  ///< 输出目录
    QComboBox *_presetCombo = nullptr;  ///< 参数预设（快速/标准/精细）

    // ---- SGBM 参数 ----
    QSpinBox *_numDispSpin = nullptr;  ///< 视差搜索范围（16 的倍数）
    QSpinBox *_blockSizeSpin = nullptr;  ///< 匹配块大小
    QSpinBox *_uniquenessSpin = nullptr;  ///< 唯一性比率(%)
    QSpinBox *_speckleSizeSpin = nullptr;  ///< 噪斑消除窗口大小
    QCheckBox *_wlsFilterCheck = nullptr;  ///< WLS 视差滤波
    QCheckBox *_fullDpCheck = nullptr;  ///< 使用 SGBM_3WAY 模式
    QDoubleSpinBox *_minDepthSpin = nullptr;  ///< 最小有效深度
    QDoubleSpinBox *_maxDepthSpin = nullptr;  ///< 最大有效深度

    // ---- 点云参数 ----
    QDoubleSpinBox *_minConfSpin = nullptr;  ///< 最小置信度
    QCheckBox *_normalsCheck = nullptr;  ///< 估计法向量
    QCheckBox *_colorsCheck = nullptr;  ///< 采样颜色
    QCheckBox *_multiViewCheck = nullptr;  ///< 多视图一致性融合
    QSpinBox *_minConsistentViewsSpin = nullptr;  ///< 最小一致视图数
    QCheckBox *_geomConsistencyCheck = nullptr;  ///< 几何一致性过滤
    QDoubleSpinBox *_maxReprojErrorSpin = nullptr;  ///< 最大重投影误差
    QSpinBox *_speckleMinAreaSpin = nullptr;  ///< 小连通域过滤阈值
    QSpinBox *_fusionMaxImageDimSpin = nullptr;  ///< 融合最长边
    QSpinBox *_normalKnnSpin = nullptr;  ///< 法向量 KNN

    // ---- 网格重建（可选） ----
    QCheckBox *_buildMeshCheck = nullptr;  ///< 是否重建网格
    QComboBox *_meshMethodCombo = nullptr;  ///< 网格方法
    QSpinBox *_voxelResSpin = nullptr;  ///< 体素分辨率
    QSpinBox *_smoothIterSpin = nullptr;  ///< 平滑迭代次数

    // ---- 进度 & 日志 ----
    QProgressBar *_progressBar = nullptr;
    QTextEdit *_logEdit = nullptr;
    QPushButton *_runButton = nullptr;
    QPushButton *_cancelButton = nullptr;
};
