/**
 * @file ProjectConfigManager.cpp
 * @brief ProjectConfigManager 的实现文件。
 *
 * 核心逻辑包含两部分：
 * 1. 静态深度合并辅助函数 mergeObject，用于将两个 JSON 对象递归合并。
 * 2. ProjectConfigManager 通过工作流子管理器操作 workflow 段。
 */
#include "project/ProjectConfigManager.h"

#include "json/JsonObjectMerge.h"

/**
 * @brief 生成含有所有默认值的标准项目配置对象。
 *
 * 通过调用各子管理器的 default*Settings() 静态方法，
 * 将 "ui" 和 "workflow" 段聚合为完整配置，
 * 确保新建项目无需任何手动配置即可正常运行。
 *
 * @return 包含 "ui" 和 "workflow" 两大顶层段的默认配置对象。
 */
QJsonObject ProjectConfigManager::defaultConfig()
{
    QJsonObject config;

    config["workflow"] = ProjectWorkflowConfigManager::defaultWorkflowSettings();

    return config;
}

/**
 * @brief 将用户配置与默认配置深度合并，确保所有字段均存在且有效。
 *
 * 适用场景：
 * - 打开旧版本项目文件（缺少新增的配置字段）；
 * - 初始化新项目（input 为空对象）。
 *
 * @param input 从磁盘加载的原始配置 JSON，可能缺字段。
 * @return      以 defaultConfig() 为底，input 覆盖后的完整配置。
 */
QJsonObject ProjectConfigManager::mergeWithDefaults(const QJsonObject &input)
{
    QJsonObject merged =
        xjw::common::json::deepMergeObjects(defaultConfig(), input);
    // UI 状态由根 doc.json 的 ui_state 管理，不进入可复现工作流配置。
    merged.remove(QStringLiteral("ui"));
    return merged;
}

/**
 * @brief 获取指定工作流步骤的参数配置。
 *
 * @param step  步骤名称（例如 "bundle_adjust"、"dem"、"ortho"）。
 * @return      该步骤对应的参数对象；不存在时返回空 QJsonObject。
 */
QJsonObject ProjectConfigManager::workflowSettings(const QString &step) const
{
    // 临时创建子管理器，加载 "workflow" 段后查询指定步骤
    ProjectWorkflowConfigManager workflowManager;
    workflowManager.setData(_config.value("workflow").toObject());
    return workflowManager.settings(step);
}

/**
 * @brief 更新指定工作流步骤的参数配置（深度合并补丁）。
 *
 * @param step     步骤名称。
 * @param settings 仅含需要修改字段的 JSON 补丁对象。
 */
void ProjectConfigManager::setWorkflowSettings(const QString &step, const QJsonObject &settings)
{
    // 临时创建子管理器，加载 "workflow" 段后更新指定步骤
    ProjectWorkflowConfigManager workflowManager;
    workflowManager.setData(_config.value("workflow").toObject());
    workflowManager.setSettings(step, settings);
    // 将更新后的 "workflow" 写回完整配置
    _config["workflow"] = workflowManager.data();
}
