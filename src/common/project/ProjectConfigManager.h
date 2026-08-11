#pragma once

/**
 * @file ProjectConfigManager.h
 * @brief 项目配置管理器的声明文件。
 *
 * ProjectConfigManager 负责管理 Chunk doc.json 的 project_config 字段，
 * 该 JSON 文件只保存影响处理结果、需要随项目复现的工作流参数。
 * 项目视图状态单独存储在根 doc.json 的 ui_state，应用窗口状态存储在
 * QSettings，避免把机器相关状态混入处理配置。
 *
 * 设计原则：
 *   - 本类拥有底层 QJsonObject 数据，子管理器作为"视图"操作特定段落。
 *   - 对外提供语义化的相机模型和 workflowSettings 访问接口，隐藏内部 JSON 键名。
 *   - defaultConfig() + mergeWithDefaults() 保证新建项目或旧版配置文件
 *     升级时所有字段均具备合理默认值。
 */

#include <QJsonObject>
#include <QString>

#include <optional>

#include "ProjectWorkflowConfigManager.h"

/**
 * @brief 项目处理时采用的相机模型策略。
 *
 * 该策略随项目持久化，只负责选择几何模型类别；具体相机参数仍由
 * 各自的相机模型和处理流程管理。
 */
enum class ProjectCameraModelPolicy
{
    FramePinhole,
    IsisUsgsCsmLineScan
};

/** @brief 将相机模型策略转换为稳定的项目配置 token。 */
QString projectCameraModelPolicyToken(ProjectCameraModelPolicy policy);

/**
 * @brief 解析项目配置中的相机模型 token。
 *
 * 缺失或空 token 按旧项目兼容规则解释为面阵针孔模型；未知的非空
 * token 返回 std::nullopt，防止项目被静默按错误模型处理。
 */
std::optional<ProjectCameraModelPolicy> parseProjectCameraModelPolicy(
    const QString &token);

/**
 * @class ProjectConfigManager
 * @brief 管理 project_config 字段的工作流配置数据。
 *
 * 典型用法：
 * @code
 *   ProjectConfigManager cfg;
 *   cfg.setData(ProjectConfigManager::mergeWithDefaults(loadedJson));
 *   auto matching = cfg.workflowSettings("ipmatch");
 * @endcode
 */
class ProjectConfigManager
{
public:
    /** @brief 默认构造，内部 JSON 对象为空，需通过 setData 或 mergeWithDefaults 初始化。 */
    ProjectConfigManager() = default;

    /**
     * @brief 获取完整的底层 QJsonObject 数据。
     * @return 当前配置的 JSON 对象副本。
     */
    QJsonObject data() const { return _config; }

    /**
     * @brief 替换全部底层配置数据。
     * @param data 新的 JSON 配置对象。
     */
    void setData(const QJsonObject &data)
    {
        _config = data;
        _config.remove(QStringLiteral("ui"));
    }

    /**
     * @brief 生成包含所有字段默认值的标准配置对象。
     *
     * 汇集 ProjectWorkflowConfigManager::defaultWorkflowSettings()，
     * 与默认相机模型策略共同形成标准 project_config 字段结构。
     *
     * @return 含有 "camera_model_policy" 和 "workflow" 的默认配置对象。
     */
    static QJsonObject defaultConfig();

    /**
     * @brief 将外部输入与默认配置深度合并，补全缺失字段。
     *
     * 策略：以 defaultConfig() 为基础，将 input 中的值覆盖到对应位置；
     * 若 input 中某子对象缺少某键，则保留默认值，防止旧版配置文件升级后丢字段。
     *
     * @param input 从磁盘加载的原始 JSON 对象（可能缺少部分键）。
     * @return      合并后包含所有键的完整配置对象。
     */
    static QJsonObject mergeWithDefaults(const QJsonObject &input);

    /**
     * @brief 获取当前项目的相机模型策略。
     * @return 已识别的策略；未知的非空 token 返回 std::nullopt。
     */
    std::optional<ProjectCameraModelPolicy> cameraModelPolicy() const;

    /** @brief 设置当前项目的相机模型策略。 */
    void setCameraModelPolicy(ProjectCameraModelPolicy policy);

    /**
     * @brief 获取指定处理步骤的工作流参数。
     * @param step  步骤名称，例如 "bundle_adjust"、"dem"、"ortho"。
     * @return      该步骤对应的参数 QJsonObject；若不存在则返回空对象。
     */
    QJsonObject workflowSettings(const QString &step) const;

    /**
     * @brief 更新指定处理步骤的工作流参数（深度合并补丁）。
     * @param step     步骤名称。
     * @param settings 仅包含需要改动字段的 JSON 补丁对象。
     */
    void setWorkflowSettings(const QString &step, const QJsonObject &settings);

private:
    /** @brief 存储完整项目工作流配置的底层 JSON 对象。 */
    QJsonObject _config;
};
