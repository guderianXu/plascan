#pragma once

#include "BundleAdjustOptions.h"
#include "BundleAdjustProblem.h"
#include "BundleAdjustResult.h"
#include "BundleAdjustTypes.h"
#include "FramePinholeCamera.h"

#include <vector>

namespace xjw
{

    /**
     * @brief 光束法平差核心类（静态接口）。
     *
     * 提供一个静态方法 optimizePoints，负责接收相机列表和轨迹列表，
     * 返回优化后的点坐标、相机位姿及统计信息。
     */
    class BundleAdjust
    {
    public:
        /// 返回指定 BA 后端在当前编译产物中是否可用。
        static bool isBackendAvailable(BABackend backend);

        /// 返回后端稳定名称，供日志、JSON 和测试输出使用。
        static const char* backendName(BABackend backend);

        /// 返回求解状态稳定名称，供日志、JSON 和测试输出使用。
        static const char* solveStatusName(BASolveStatus status);

        /// 返回后端当前实现的真实能力，不能根据名称推断 CUDA 后端一定支持联合 BA。
        static BABackendCapabilities backendCapabilities(BABackend backend);

        /// 统计 BA 实际可用的问题规模。
        static BAProblemStats summarizeProblem(const std::vector<FramePinholeCamera>& cameras,
                                               const std::vector<BATrack>& tracks);

        /// 根据问题规模与配置选择实际执行后端。
        static BABackend selectBackendForProblem(const BAProblemStats& stats, const BAOptions& options);

        /// 只判断问题规模是否达到指定设备后端的 Auto 门槛，不检查设备运行时可用性。
        static bool
        autoBackendMeetsScaleThreshold(BABackend backend, const BAProblemStats& stats, const BAOptions& options);

        /// 返回自动后端及机器可读原因，供日志解释 CPU/CUDA 选择。
        static BABackendDecision decideBackendForProblem(const BAProblemStats& stats, const BAOptions& options);

        /**
         * @brief 在同一 Schur 问题中联合优化点、相机位姿和可选共享内参。
         *
         * @param cameras  初始相机列表（优化过程中将作副本修改）
         * @param tracks   轨迹列表，每条轨迹包含初始三维点和多相机观测
         * @param options  优化选项（可选，默认使用 BAOptions）
         * @return         BAResult，包含优化后点坐标、相机位姿及误差统计
         */
        static BAResult optimizePoints(const std::vector<FramePinholeCamera>& cameras,
                                       const std::vector<BATrack>& tracks,
                                       const BAOptions& options = BAOptions());
    };

} // namespace xjw
