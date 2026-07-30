/**
 * @file ProjectWorkflowConfigManager.cpp
 * @brief ProjectWorkflowConfigManager 的实现文件。
 *
 * 包含工作流配置的默认值定义、步骤参数读取与合并更新逻辑。
 */
#include "ProjectWorkflowConfigManager.h"

#include "json/JsonObjectMerge.h"

/**
 * @brief 生成 "workflow" 段各步骤的默认参数结构。
 *
 * 目前注册了三个步骤键（均初始化为空对象），
 * 后续各业务模块在运行时通过 setSettings 填充具体参数：
 *   - "bundle_adjust" : 光束法平差参数占位
 *   - "dem"           : DEM 生成参数占位
 *   - "ortho"         : 正射影像生成参数占位
 *
 * @return 含三个步骤键的默认工作流配置对象。
 */
QJsonObject ProjectWorkflowConfigManager::defaultWorkflowSettings()
{
    QJsonObject workflow;
    workflow["bundle_adjust"] = QJsonObject(); // 光束法平差参数（默认空，由业务层填充）
    workflow["dem"]           = QJsonObject(); // DEM 生成参数（默认空）
    workflow["ortho"]         = QJsonObject(); // 正射影像参数（默认空）
    return workflow;
}

/**
 * @brief 获取指定步骤的参数配置。
 *
 * @param step  步骤名称（例如 "bundle_adjust"）。
 * @return      对应步骤的 QJsonObject；步骤不存在时返回空对象。
 */
QJsonObject ProjectWorkflowConfigManager::settings(const QString &step) const
{
    return _workflow.value(step).toObject();
}

/**
 * @brief 深度合并更新指定步骤的参数配置。
 *
 * 实现细节：
 * 1. 取出该步骤的当前配置作为基准；
 * 2. 将 settings 补丁深度合并到基准中，仅覆盖指定字段；
 * 3. 将合并结果写回 _workflow 对应键。
 *
 * @param step     步骤名称。
 * @param settings 仅含需要修改字段的 JSON 补丁对象。
 */
void ProjectWorkflowConfigManager::setSettings(const QString &step, const QJsonObject &settings)
{
    // 以当前步骤配置为基准进行深度合并，保留未被补丁覆盖的原有字段
    QJsonObject current = _workflow.value(step).toObject();
    _workflow[step] = xjw::common::json::deepMergeObjects(current, settings);
}
