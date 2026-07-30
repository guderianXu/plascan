#pragma once

/**
 * @file ProjectUiConfigManager.h
 * @brief 项目 UI 显示配置子管理器的声明文件。
 *
 * ProjectUiConfigManager 负责管理根 doc.json 的 ui_state 中
 * "display_settings" 段，
 * 该段落存储用户在视图中调整的显示参数，主要包括：
 *   - feature_display : 特征点的显示样式（大小、形状、不透明度、最大数量等）
 *   - match_display   : 匹配连线的显示样式（线宽、是否仅显示内点等）
 *   - show_interest_points : 是否显示兴趣点覆盖层
 *
 * 本类不直接持久化数据，数据的加载/保存由 ProjectData 协调。
 */

#include <QJsonObject>

/**
 * @class ProjectUiConfigManager
 * @brief 管理 ui_state.display_settings 段。
 *
 * 主要职责：
 * 1. 提供各字段的默认值（通过静态方法 defaultUiSettings）。
 * 2. 支持通过 applyPatch 对当前配置进行深度合并式的局部更新。
 */
class ProjectUiConfigManager
{
public:
    /** @brief 默认构造，内部 JSON 为空对象，需由调用方通过 setData 初始化。 */
    ProjectUiConfigManager() = default;

    /**
     * @brief 生成 "ui" 段所有字段的默认值配置对象。
     *
     * 包含 feature_display、match_display 以及顶层的 show_interest_points 等字段，
     * 保证新项目或升级旧项目时视图显示参数均合理可用。
     *
     * @return 含默认值的 "ui" 配置 QJsonObject。
     */
    static QJsonObject defaultUiSettings();

    /**
     * @brief 设置（替换）当前管理的 UI 配置数据。
     * @param data 通常来自 ProjectConfigManager 中 "ui" 键对应的 JSON 子对象。
     */
    void setData(const QJsonObject &data) { _ui = data; }

    /**
     * @brief 获取当前 UI 配置数据。
     * @return "ui" 段的 QJsonObject 副本。
     */
    QJsonObject data() const { return _ui; }

    /**
     * @brief 将 partial 补丁深度合并到当前配置中。
     *
     * 只修改 partial 中明确指定的字段，其余字段保持不变，
     * 适用于用户在 UI 对话框中仅调整部分参数的场景。
     *
     * @param partial 仅含需要变更字段的 JSON 补丁对象。
     */
    void applyPatch(const QJsonObject &partial);

private:
    /** @brief 存储 "ui" 段 JSON 数据的内部成员。 */
    QJsonObject _ui;
};
