#pragma once

/**
 * @file WorkflowSettingsDialog.h
 * @brief 按工作流程组织的项目级设置对话框。
 *
 * 对话框只呈现工作流程真正需要用户选择的策略，不承载线程数、USAC 迭代数
 * 等实现细节。设置按 workflow ID 分组，后续新增流程页面时不会污染空中三角
 * 测量的配置命名空间。
 */

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QFormLayout;
class QLabel;
class QLineEdit;
class QPushButton;
class QStackedWidget;
class QToolButton;

class WorkflowSettingsDialog final : public QDialog
{
public:
    explicit WorkflowSettingsDialog(QWidget* parent = nullptr);

    /// 返回 v9 工作流程分组配置，包含算法专属参数、资源和生成模型计算模式。
    static QJsonObject defaultSettings();

    /**
     * @brief 提取空中三角测量设置。
     *
     * 同时接受旧版 workflows.aerial_triangulation 对象和旧 v2 扁平字段，
     * 供项目无损升级以及空三启动参数合并使用。
     */
    static QJsonObject aerialTriangulationSettings(const QJsonObject& settings);

    /// 提取并规范化生成模型设置；计算模式为 cuda/opencl/hybrid 之一。
    static QJsonObject modelGenerationSettings(const QJsonObject& settings);

    /// 从项目级 JSON 恢复控件，旧版配置会在内存中迁移到 v9。
    void applySettings(const QJsonObject& settings);

    /// 收集按工作流程分组的 v9 配置。
    QJsonObject collectSettings() const;

private:
    /// 创建工作流程选择器、分页区域和对话框按钮。
    void setupUi();
    /// 从统一算法注册表填充匹配算法，避免 GUI 维护另一份算法清单。
    void populateMatchingAlgorithms();
    /// 切换右侧设置页；暂未开放的页面显示明确说明。
    void setCurrentWorkflow(int index);
    /// 根据当前算法刷新其专用资源控件。
    void refreshAlgorithmControls();
    /// 保存旧算法资源并加载新算法资源，保证切换时两套路径互不覆盖。
    void switchAlgorithmResource();
    /// 解析显式路径或自动搜索结果，让用户看到真正生效的模型包。
    void refreshMatchingResourceStatus();
    /// 下载当前算法所需的预构建 TensorRT 包，并将入口文件写回设置。
    void downloadCurrentModelPackage();
    /// 枚举 CUDA/OpenCL 设备并刷新模式可用性和设备摘要。
    void refreshModelComputeDevices();
    /// 根据选中的生成模型计算模式显示实际阶段路由。
    void refreshModelComputePolicy();

    QComboBox* _workflowCombo = nullptr;
    QStackedWidget* _workflowPages = nullptr;
    QFormLayout* _aerialForm = nullptr;
    QComboBox* _matchingAlgorithmCombo = nullptr;
    QComboBox* _lomaRKeypointBudgetCombo = nullptr;
    QDoubleSpinBox* _siftMaximumRatioSpin = nullptr;
    QCheckBox* _adaptiveSiftRatioCheck = nullptr;
    QWidget* _matchingResourceRow = nullptr;
    QLineEdit* _matchingResourceEdit = nullptr;
    QToolButton* _matchingResourceBrowseButton = nullptr;
    QPushButton* _downloadModelButton = nullptr;
    QLabel* _matchingResourceStatusLabel = nullptr;
    QComboBox* _modelComputeModeCombo = nullptr;
    QLabel* _cudaDeviceStatusLabel = nullptr;
    QLabel* _openClDeviceStatusLabel = nullptr;
    QLabel* _modelComputePolicyLabel = nullptr;
    QPushButton* _detectComputeDevicesButton = nullptr;
    bool _cudaAvailable = false;
    bool _openClAvailable = false;
    bool _hybridAvailable = false;
    QString _currentAlgorithmId;
    QString _lightGlueEnginePath;
    QString _lomaRPackagePath;
    QJsonObject _appliedSettings;
};
