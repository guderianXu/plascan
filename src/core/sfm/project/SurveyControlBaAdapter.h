#pragma once

/**
 * @file SurveyControlBaAdapter.h
 * @brief 读取工程 JSON 中的测量控制点和比例尺约束。
 *
 * 该适配器处理旧 `survey_control` 数据格式；新标记系统由 MarkerBaAdapter 处理。
 * 两者最终都转换为相同的 BATrack/BAControlPointConstraint/BAScaleBarConstraint。
 */

#include "project/BaInputBuilder.h"

#include <QJsonObject>
#include <QMap>
#include <QString>

namespace xjw::core::project
{

/**
 * @brief 从 project meta 追加启用且可解析的控制观测与标尺。
 *
 * 每个控制点必须具有有限物方坐标和至少两台不同相机上的有限像点。标尺端点
 * 通过控制点 ID 解析到最终 track 索引，失败记录计数但不会使整次 BA 中止。
 */
void appendSurveyControlBaInput(const QJsonObject &meta,
                                const QMap<QString, int> &cameraIndexByPath,
                                BaInputBuildResult *result);

} // namespace xjw::core::project
