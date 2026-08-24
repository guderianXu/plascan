#pragma once

/**
 * @file BundleAdjustAdaptiveCameraModel.h
 * @brief 根据观测覆盖和摄影几何估计共享相机模型各参数的可靠性。
 *
 * 该策略不把工程硬分类为“航测”或“环拍”。它从当前粗解构造内参信息矩阵，
 * 再结合视轴多样性、交会角、多视轨迹率和像面覆盖，为 9 个共享内参分别生成
 * [0, 1] 可靠性和实际优化掩码。底层联合求解器使用同一套 SfM/BA 残差。
 */

#include "BundleAdjustOptions.h"
#include "BundleAdjustProblem.h"
#include "BundleAdjustTypes.h"

#include <array>
#include <string>

namespace xjw
{

struct BAAdaptiveCameraModelAssessment
{
    bool valid = false;
    int cameraCount = 0;
    int activeCameraCount = 0;
    int trackCount = 0;
    int observationCount = 0;
    int multiViewTrackCount = 0;
    int occupiedPeripheralSectors = 0;
    bool hasAbsoluteGeometryConstraint = false;
    bool unanchoredParallelAerialGuardApplied = false;
    double opticalAxisConcentration = 1.0;
    double medianTriangulationAngleDegrees = 0.0;
    double multiViewTrackRatio = 0.0;
    double normalizedRadiusP90 = 0.0;
    double maximumNormalizedRadius = 0.0;
    double peripheralRadiusThreshold = 0.30;
    double lowOrderDistortionScale = 1.0;
    double geometryStrength = 0.0;
    double observationSupport = 0.0;
    double peripheralCoverage = 0.0;
    double sectorCoverage = 0.0;
    double imageAxisBalance = 0.0;
    std::array<double, kBAIntrinsicParameterCount> incrementalInformationScore{};
    std::array<double, kBAIntrinsicParameterCount> sensitivity{};
    std::array<double, kBAIntrinsicParameterCount> reliability{};
    BAIntrinsicParameterMask enabled{};
    std::string modelName = "fixed";
    std::string reason;
};

/// 返回报告和日志使用的稳定参数名。
const char *baIntrinsicParameterName(BAIntrinsicParameter parameter);

/// 返回掩码中实际释放的自由度数量。
int enabledIntrinsicParameterCount(const BAIntrinsicParameterMask &mask);

/// 把掩码格式化为 `f+k1`、`f+aspect+cx+...` 等稳定名称。
std::string adaptiveCameraModelName(const BAIntrinsicParameterMask &mask);

/// 从粗略相机/点解评估各共享内参的可观测性与可靠性。
BAAdaptiveCameraModelAssessment assessAdaptiveCameraModel(
    const std::vector<FramePinholeCamera> &cameras,
    const std::vector<BATrack> &tracks,
    const BAOptions *options = nullptr);

/**
 * @brief 将可靠性评估与调用方最大模型求交，写入逐参数掩码，并为弱平行几何收紧低阶参数先验。
 * @return 至少有一个共享内参自由度被保留时返回 true。
 */
bool applyAdaptiveCameraModel(
    const BAAdaptiveCameraModelAssessment &assessment,
    BAOptions *options);

/**
 * @brief 把自适应模型已关闭的参数恢复到稳定输入标定，避免跨轮残留旧自由度。
 * @return 输入尺寸有效且完成恢复时返回 true。
 */
bool restoreInactiveAdaptiveIntrinsics(
    std::vector<FramePinholeCamera> *cameras,
    const std::vector<FramePinholeCamera> &stableReferences,
    const BAIntrinsicParameterMask &activeMask);

} // namespace xjw
