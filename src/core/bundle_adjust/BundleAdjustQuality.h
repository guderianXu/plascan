#pragma once

/**
 * @file BundleAdjustQuality.h
 * @brief 所有 BA 后端共享的结果复核、离群点过滤与约束质量门控。
 *
 * Ceres、CPU 和 native CUDA 的内部收敛判据并不等价。公共入口必须在后端返回后
 * 用同一投影模型重新计算严格 RMS，才能让 GUI、CLI 和 SfM 获得可比较的结果。
 */

#include "BundleAdjust.h"

#include <string>
#include <vector>

namespace xjw::detail
{

/**
 * @brief 计算全局离群点阈值。
 *
 * 仅统计有限且非负的点 RMS，阈值为
 * `max(absoluteFloor, medianFactor * median(finite RMS))`。因此
 * absoluteFloor 是不会被自适应统计收紧的绝对像素下限，而 medianFactor
 * 控制长尾数据允许的相对上限。
 *
 * @return 输入没有有效 RMS 或 medianFactor 非法时，仅返回合法化后的 absoluteFloor。
 */
double adaptivePointFilterThreshold(const std::vector<double> &pointRms,
                                    double absoluteFloor,
                                    double medianFactor);

/**
 * @brief 对所有 BA 后端执行统一的最终正深度、离群点和统计检查。
 *
 * 本函数会就地更新 BAResult：为每条原始 track 对齐一个 BARefinedPoint，
 * 分别在输入相机和优化相机上计算严格 RMS；任一有效观测投影失败、位于相机后方、
 * 观测不足或三维坐标非有限时，整条 track 失效。随后执行可选的全局 RMS 过滤，
 * 并重算 totalTracks、optimizedTracks、meanRmsBefore/After 和 validTrackRatio。
 *
 * 若后端声称成功但最终没有任何有效 track，结果会被降级为 NumericalFailure。
 * 这样可防止“数值求解成功、摄影测量结果不可用”继续流入 SfM。
 */
void finalizeBundleAdjustResult(const std::vector<FramePinholeCamera> &inputCameras,
                                const std::vector<BATrack> &tracks,
                                const BAOptions &options,
                                BAResult *result);

/**
 * @brief 检查一类物方约束的 RMS 是否在质量门控允许范围内。
 *
 * 约束数量为零时直接通过；否则前后 RMS 必须有限，且优化后 RMS 不得超过
 * `max(1, maxGrowth) * rmsBefore`。若优化前近似为零，则优化后也必须保持近零，
 * 避免纯重投影目标破坏已经精确满足的控制约束。
 *
 * @param constraintName 写入失败诊断的约束名称，可为空。
 * @param message 可选失败原因输出；通过时不改写。
 */
bool constraintRmsPassesQualityGate(int constraintCount,
                                    double rmsBefore,
                                    double rmsAfter,
                                    double maxGrowth,
                                    const char *constraintName,
                                    std::string *message);

} // namespace xjw::detail
