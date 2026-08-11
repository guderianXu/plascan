#pragma once

/**
 * @file BundleAdjustNativeCudaWorkset.h
 * @brief 将通用 BA 输入压缩为 native CUDA 后端的连续工作集。
 *
 * 通用 BATrack 允许无效轨迹和多类物方约束，而 CUDA 核函数要求相机、有效点和观测
 * 连续存放。构建器会过滤不可优化轨迹，并保留原始 track 到压缩点索引的双向映射，
 * 便于下载结果后恢复与输入完全一致的 BAResult 顺序。
 */

#include "BundleAdjust.h"
#include "BundleAdjustNativeCudaTypes.h"

#include <string>
#include <vector>

namespace xjw::detail::native_cuda
{

struct WorksetBuildResult
{
    bool ok = false; ///< true 表示至少存在一条可优化轨迹且约束类型受支持。
    std::string message; ///< 构建失败或后端不支持时的具体原因。
    Workset workset; ///< 仅在 ok=true 时交给 CUDA 执行器。
};

/**
 * @brief 过滤无效观测并建立 CUDA 连续工作集。
 *
 * 轨迹至少需要两个正权重有效观测且来自两台不同相机。零权重、非有限权重、
 * 非有限坐标和越界相机观测都会被跳过。
 */
WorksetBuildResult buildWorkset(const std::vector<FramePinholeCamera> &cameras,
                                const std::vector<BATrack> &tracks,
                                const BAOptions &options);

/**
 * @brief 检查 native CUDA 首期实现无法表达的残差块。
 *
 * 当前 native CUDA 仅支持重投影点优化；控制点、比例尺、激光平面和相机位姿先验
 * 必须交给 Ceres/CPU 后端，不能静默丢弃。
 */
bool hasUnsupportedConstraints(const std::vector<BATrack> &tracks,
                               const BAOptions &options,
                               std::string *message);

} // namespace xjw::detail::native_cuda
