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

namespace xjw::core::project
{

/**
 * @brief 追加人工标记相关轨迹、约束和回写绑定。
 *
 * canonical ImageId 是投影唯一主键，路径快照只用于展示和审计，不能参与求解绑定。
 * 无法解析影像、重复相机投影、少于两视或无法三角化的标记会计入
 * rejectedMarkerTrackCount；缺少或未知 ImageId 会使整个输入返回 InvalidInput。
 */
void appendMarkerBaInput(const MarkerBaInput *input,
                         const QMap<QString, int> &cameraIndexByImageId,
                         BaInputBuildResult *result);

} // namespace xjw::core::project
