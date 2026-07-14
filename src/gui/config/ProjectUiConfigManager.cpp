/**
 * @file ProjectUiConfigManager.cpp
 * @brief ProjectUiConfigManager 的实现文件。
 *
 * 提供 "ui" 段的默认值定义和深度合并补丁逻辑。
 * 所有字段名称需与前端 JSON 读取代码保持一致。
 */
#include "ProjectUiConfigManager.h"
#include "JsonMergeUtil.h"

/**
 * @brief 生成 "ui" 段所有字段的默认值配置对象。
 *
 * 各字段含义：
 *
 * feature_display（特征点显示参数）：
 *   - showPoints        : 是否显示特征点（默认开启）
 *   - showScale         : 是否显示特征点尺度圆（默认关闭）
 *   - showOrientation   : 是否显示特征点方向线（默认关闭）
 *   - useFill           : 是否填充特征点标记（默认关闭）
 *   - pointSize         : 标记大小（像素，默认 1）
 *   - pointColor        : 点颜色（默认蓝色 RGB 0/120/255）
 *   - scaleMultiplier   : 尺度圆半径倍数（默认 1.0）
 *   - opacity           : 不透明度 0-255（默认 180，约 70%）
 *   - markerShape       : 标记形状（"point"/"circle"/"cross" 等）
 *   - maxDisplayCount   : 最大显示数量（0 表示不限制）
 *   - showTopScores     : 是否优先显示得分最高的特征点（默认开启）
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
    featureDisplay["showPoints"]        = true;    // 默认显示特征点
    featureDisplay["showScale"]         = false;   // 默认不显示尺度圆
    featureDisplay["showOrientation"]   = false;   // 默认不显示方向线
    featureDisplay["showResiduals"]     = false;
    featureDisplay["residualScale"]     = 10.0;
    featureDisplay["minimumResidualPx"] = 0.0;
    featureDisplay["maximumResidualLengthPx"] = 80.0;
    featureDisplay["useFill"]           = false;   // 默认不填充
    featureDisplay["pointSize"]         = 1;        // 默认标记大小为 1 像素
    QJsonObject pointColor;
    pointColor["r"] = 0;
    pointColor["g"] = 120;
    pointColor["b"] = 255;
    featureDisplay["pointColor"]        = pointColor; // 默认蓝色
    QJsonObject residualColor;
    residualColor["r"] = 255;
    residualColor["g"] = 80;
    residualColor["b"] = 80;
    featureDisplay["residualColor"] = residualColor;
    featureDisplay["scaleMultiplier"]   = 1.0;     // 尺度圆半径不做额外放大
    featureDisplay["opacity"]           = 180;     // 约 70% 不透明
    featureDisplay["markerShape"]       = QStringLiteral("cross"); // 默认十字标记
    featureDisplay["maxDisplayCount"]   = 0;       // 0 = 不限数量
    featureDisplay["showTopScores"]     = true;    // 默认优先显示高分特征点

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
    _ui = JsonMergeUtil::deepMerge(_ui, partial);
}
