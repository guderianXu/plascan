/**
 * @file ProjectUiConfigManager.cpp
 * @brief ProjectUiConfigManager 的实现文件。
 *
 * 提供 "ui" 段的默认值定义和深度合并补丁逻辑。
 * 所有字段名称需与前端 JSON 读取代码保持一致。
 */
#include "project/ProjectUiConfigManager.h"

#include "json/JsonObjectMerge.h"

/**
 * @brief 生成 "ui" 段所有字段的默认值配置对象。
 *
 * 各字段含义：
 *
 * feature_display（特征点显示参数）：
 *   - pointSource       : 点数据来源（提取特征/原始匹配/有效连接点）
 *   - showPoints        : 是否显示特征点（默认开启）
 *   - pointSize         : 标记大小（像素，默认 1）
 *   - pointColor        : 点颜色（默认蓝色 RGB 0/120/255）
 *   - opacity           : 不透明度 0-255（默认 180，约 70%）
 *   - maxDisplayCount   : 最大显示数量（0 表示不限制）
 *
 * match_display（匹配线显示参数）：
 *   - showLines         : 是否显示连接线（默认开启）
 *   - lineWidth         : 线宽（像素，默认 1）
 *   - opacity           : 不透明度 0-255（默认 180）
 *   - showOnlyInliers   : 是否仅显示经几何验证的内点匹配（默认关闭）
 *   - maxDisplayCount   : 最大显示数量（0 表示不限制）
 *
 * show_interest_points（顶层开关）：
 *   - 是否在图像视图上叠加兴趣点（默认开启）
 *
 * @return 包含上述所有字段及其默认值的 QJsonObject。
 */
QJsonObject ProjectUiConfigManager::defaultUiSettings()
{
    QJsonObject ui;

    // ---- 特征点显示默认参数 ----
    QJsonObject featureDisplay;
    featureDisplay["pointSource"]       = "valid_tie_points";
    featureDisplay["showPoints"]        = true;    // 默认显示特征点
    featureDisplay["showResiduals"]     = false;
    featureDisplay["residualScale"]     = 50.0;
    featureDisplay["minimumResidualPx"] = 0.0;
    featureDisplay["maximumResidualLengthPx"] = 80.0;
    featureDisplay["pointSize"]         = 1;        // 默认标记大小为 1 像素
    QJsonObject pointColor;
    pointColor["r"] = 0;
    pointColor["g"] = 120;
    pointColor["b"] = 255;
    featureDisplay["pointColor"]        = pointColor; // 默认蓝色
    featureDisplay["opacity"]           = 180;     // 约 70% 不透明
    featureDisplay["maxDisplayCount"]   = 0;       // 0 = 不限数量

    // ---- 匹配线显示默认参数 ----
    QJsonObject matchDisplay;
    matchDisplay["showLines"]           = true;    // 默认显示匹配连线
    matchDisplay["lineWidth"]           = 1;        // 默认线宽 1 像素
    matchDisplay["opacity"]             = 180;     // 约 70% 不透明
    matchDisplay["showOnlyInliers"]     = false;   // 默认显示所有匹配（含外点）
    matchDisplay["maxDisplayCount"]     = 0;       // 0 = 不限数量

    ui["feature_display"]     = featureDisplay;
    ui["match_display"]       = matchDisplay;
    ui["show_interest_points"] = true; // 默认在视图中叠加兴趣点

    return ui;
}

/**
 * @brief 将补丁对象深度合并到当前 UI 配置中。
 *
 * 调用文件内 mergeObjectUi 实现递归合并，
 * 只修改 partial 中指定的字段，其余字段保持原值不变。
 *
 * @param partial 仅含需要更新字段的 JSON 补丁对象。
 */
void ProjectUiConfigManager::applyPatch(const QJsonObject &partial)
{
    _ui = xjw::common::json::deepMergeObjects(_ui, partial);
}
