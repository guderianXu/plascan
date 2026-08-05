#pragma once

/**
 * @file BundleAdjustValidation.h
 * @brief BA 公共入口的输入校验与 gauge 规范化。
 *
 * 后端只能接收已经规范化的 BAOptions。本层负责拒绝非法数值、越界相机索引和
 * 不完整的标定分组，并在 AutoAnchor 策略下补足单目 BA 的相似变换自由度。
 * 它不修改调用方配置，也不执行任何数值优化。
 */

#include "BundleAdjust.h"

#include <cstddef>
#include <string>
#include <vector>

namespace xjw::detail
{

struct BundleAdjustValidationResult
{
    bool ok = true; ///< false 表示不得进入任何 BA 后端。
    BASolveStatus status = BASolveStatus::NotRun; ///< 失败时应写入 BAResult 的标准状态。
    std::string message; ///< 面向日志和 GUI 的可定位诊断信息。
};

/// 将非法、非正的观测权重统一视为零，所有后端必须使用同一语义。
double sanitizedObservationWeight(const BAObservation &observation);

/// 检查像点坐标和权重是否合法，不检查相机索引。
bool observationDataIsUsable(const BAObservation &observation);

/// 检查观测索引、像点坐标和权重是否可以进入 BA 残差。
bool observationIsUsable(const BAObservation &observation,
                         std::size_t cameraCount);

/// 统计至少由两台相机提供有效观测的实际可用 BA 问题规模。
BAProblemStats summarizeUsableProblem(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks);

/**
 * @brief 校验 BA 输入，并按 gauge 策略补齐自动锚定相机。
 *
 * 联合优化相机位姿和三维点时，纯重投影目标存在 7 自由度相似变换模糊性。
 * 若调用方没有提供绝对位姿、绝对尺度或足够的固定相机，本函数会按策略：
 * - `CallerManaged`：完全信任调用方，不自动补锚；
 * - `RequireExplicitGauge`：约束不足时直接拒绝；
 * - `AutoAnchor`：固定首台相机，并以最远的非退化基线相机固定尺度。
 *
 * @param cameras 原始相机。相机中心用于判断自动锚定基线是否退化。
 * @param tracks 原始轨迹。只读取其中的控制点、激光平面等绝对约束。
 * @param requestedOptions 调用方请求，不会被修改。
 * @param normalizedOptions 输出的后端配置；成功时首先复制 requestedOptions，
 *        然后仅补充必要的固定相机索引。
 * @return `ok=true` 表示输入可进入后端；否则 status/message 说明拒绝原因。
 */
BundleAdjustValidationResult validateAndNormalizeBundleAdjustOptions(
    const std::vector<Camera> &cameras,
    const std::vector<BATrack> &tracks,
    const BAOptions &requestedOptions,
    BAOptions *normalizedOptions);

} // namespace xjw::detail
