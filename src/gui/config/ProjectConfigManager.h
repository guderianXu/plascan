#pragma once

/**
 * @file ProjectConfigManager.h
 * @brief 项目配置管理器的声明文件。
 *
 * ProjectConfigManager 负责管理存储在 project_config.json 中的项目级配置，
 * 该 JSON 文件分为两大顶层段落：
 *   - "ui"       : 视图显示相关设置（委托给 ProjectUiConfigManager）
 *   - "workflow" : 各处理步骤的参数配置（委托给 ProjectWorkflowConfigManager）
 *
 * 设计原则：
 *   - 本类拥有底层 QJsonObject 数据，子管理器作为"视图"操作特定段落。
 *   - 对外提供语义化的 uiSettings/workflowSettings 访问接口，隐藏内部 JSON 键名。
 *   - defaultConfig() + mergeWithDefaults() 保证新建项目或旧版配置文件
 *     升级时所有字段均具备合理默认值。
 */

#include <QJsonObject>

#include "ProjectUiConfigManager.h"
#include "ProjectWorkflowConfigManager.h"

/**
 * @class ProjectConfigManager
 * @brief 管理 project_config.json 的完整配置数据。
 *
 * 典型用法：
 * @code
 *   ProjectConfigManager cfg;
 *   cfg.setData(ProjectConfigManager::mergeWithDefaults(loadedJson));
 *   auto uiCfg = cfg.uiSettings();
 * @endcode
 */
class ProjectConfigManager
{
public:
    /** @brief 默认构造，内部 JSON 对象为空，需通过 setData 或 mergeWithDefaults 初始化。 */
    ProjectConfigManager() = default;

    /**
     * @brief 获取完整的底层 QJsonObject 数据（包含 "ui" 和 "workflow" 段）。
     * @return 当前配置的 JSON 对象副本。
     */
    QJsonObject data() const { return m_config; }

    /**
     * @brief 替换全部底层配置数据。
     * @param data 新的 JSON 配置对象，应包含 "ui" 和 "workflow" 两个顶层键。
     */
    void setData(const QJsonObject &data) { m_config = data; }

    /**
     * @brief 生成包含所有字段默认值的标准配置对象。
     *
     * 汇集 ProjectUiConfigManager::defaultUiSettings() 和
     * ProjectWorkflowConfigManager::defaultWorkflowSettings() 的结果，
     * 形成完整的 project_config.json 默认结构。
     *
     * @return 含有 "ui" 和 "workflow" 段的默认配置 QJsonObject。
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
     * @brief 获取 "ui" 段的显示设置。
     * @return "ui" 顶层键对应的 QJsonObject。
     */
    QJsonObject uiSettings() const;

    /**
     * @brief 部分更新 "ui" 段（深度合并补丁）。
     * @param settings 仅包含需要改动字段的 JSON 补丁对象。
     */
    void setUiSettings(const QJsonObject &settings);

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
    /** @brief 存储完整项目配置的底层 JSON 对象（包含 "ui" 和 "workflow" 段）。 */
    QJsonObject m_config;
};
