#pragma once

/**
 * @file MarkerBaAdapter.h
 * @brief 将 Metashape 风格人工标记、检查点和标尺接入 BA 输入。
 *
 * 适配器先由至少两幅有效投影生成标记轨迹，再用控制点网络估计相似变换，
 * 最后仅把通过控制网鲁棒检查的控制点/标尺写成物方约束。检查点只用于报告，
 * 不参与求解。
 */

#include "project/BaInputBuilder.h"

#include <QMap>
#include <QString>

namespace xjw::core::project
{

/**
 * @brief 追加人工标记相关轨迹、约束和回写绑定。
 *
 * `cameraIndexByPath` 是工程路径到 BA 相机块的唯一映射。无法解析影像、重复相机
 * 投影、少于两视或无法三角化的标记会计入 rejectedMarkerTrackCount。
 */
void appendMarkerBaInput(const MarkerBaInput *input,
                         const QMap<QString, int> &cameraIndexByPath,
                         BaInputBuildResult *result);

} // namespace xjw::core::project
