#pragma once

/**
 * @file ProjectWorkflowConfigManager.h
 * @brief 项目工作流参数配置子管理器的声明文件。
 *
 * ProjectWorkflowConfigManager 负责管理 project_config.json 中 "workflow" 顶层段落，
 * 该段落以处理步骤名称为键，存储各步骤的输入参数，例如：
 *   - "bundle_adjust" : 光束法平差的各项参数
 *   - "dem"           : DEM 生成参数
 *   - "ortho"         : 正射影像生成参数
 *
 * 设计要点：
 * - 本类不关心各步骤具体包含哪些字段，完全由上层业务代码决定。
 * - 提供 settings/setSettings 接口，以步骤名称为粒度读写参数。
 * - setSettings 采用深度合并策略，支持增量更新，不会覆盖未指定的字段。
 */

#include <QJsonObject>
#include <QString>

/**
 * @class ProjectWorkflowConfigManager
 * @brief 管理 project_config.json 中 "workflow" 段的各步骤参数配置。
 */
class ProjectWorkflowConfigManager
{
public:
    /** @brief 默认构造，内部 JSON 为空对象，需由调用方通过 setData 初始化。 */
    ProjectWorkflowConfigManager() = default;

    /**
     * @brief 生成各工作流步骤的默认参数结构（初始均为空对象）。
     *
     * 目前包含的步骤键：bundle_adjust、dem、ortho。
     * 返回的各步骤对象为空，具体参数字段在用户配置界面填写后才写入。
     *
     * @return 含各步骤键的 "workflow" 配置对象（值均为 {}）。
     */
    static QJsonObject defaultWorkflowSettings();

    /**
     * @brief 设置（替换）当前管理的工作流配置数据。
     * @param data 通常来自 ProjectConfigManager 中 "workflow" 键对应的 JSON 子对象。
     */
    void setData(const QJsonObject &data) { m_workflow = data; }

    /**
     * @brief 获取当前工作流配置数据。
     * @return "workflow" 段的 QJsonObject 副本。
     */
    QJsonObject data() const { return m_workflow; }

    /**
     * @brief 获取指定步骤的参数配置。
     * @param step  步骤名称（例如 "bundle_adjust"、"dem"、"ortho"）。
     * @return      该步骤的 QJsonObject；若步骤不存在则返回空对象。
     */
    QJsonObject settings(const QString &step) const;

    /**
     * @brief 更新指定步骤的参数（深度合并补丁）。
     *
     * 将 settings 补丁深度合并到该步骤的现有配置中，
     * 只修改 settings 中指定的字段，其余字段保持不变。
     *
     * @param step     步骤名称。
     * @param settings 仅含需要修改字段的 JSON 补丁对象。
     */
    void setSettings(const QString &step, const QJsonObject &settings);

private:
    /** @brief 存储 "workflow" 段 JSON 数据的内部成员。 */
    QJsonObject m_workflow;
};
