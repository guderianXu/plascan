#pragma once

/**
 * @file BundleAdjustNativeCudaKernels.cuh
 * @brief native CUDA 点优化执行器的主机调用契约。
 *
 * CUDA 代码只更新 Workset::points，不改变相机或原始轨迹。计时字段把设备选择、
 * 上传、核执行、下载和释放分开，便于判断小问题规模为何无法获得 GPU 加速收益。
 */

#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

struct KernelRunSummary
{
    bool ok = false; ///< CUDA API 和数值迭代均完成，不等价于最终摄影测量质量通过。
    double initialCost = 0.0; ///< 首次线性化的加权平方残差和。
    double finalCost = 0.0; ///< 最后一次接受步后的加权平方残差和。
    int activeObservations = 0; ///< 至少参与过一次有效线性化的观测数量。
    int acceptedSteps = 0; ///< 使目标下降并被提交的 LM 步数。
    int rejectedSteps = 0; ///< 因目标不降或数值失败而回滚的 LM 步数。
    double uploadSeconds = 0.0; ///< 主机到设备传输时间。
    double kernelSeconds = 0.0; ///< CUDA 核与必要设备同步时间。
    double downloadSeconds = 0.0; ///< 优化点下载时间。
    double hostCostSeconds = 0.0; ///< 主机侧代价复核时间。
    double deviceSelectSeconds = 0.0; ///< 选择/初始化 CUDA 设备时间。
    double stagingSeconds = 0.0; ///< 主机 POD 打包时间。
    double releaseSeconds = 0.0; ///< 设备资源释放时间。
    char message[256] = {}; ///< 固定长度错误信息，避免跨 CUDA 边界抛异常。
};

/**
 * @brief 在指定设备上执行独立点块 LM 优化。
 *
 * 每个三维点只依赖固定相机中的自身观测，因此点块之间可并行。`initialDamping`
 * 是 LM 对角阻尼，`maxPointStepNorm` 限制单步世界坐标位移，防止弱交会点发散。
 * 调用成功后 workset 中的压缩点坐标被原位更新。
 */
KernelRunSummary runNativeCudaBundleAdjust(Workset *workset,
                                           int deviceId,
                                           int maxIterations,
                                           double huberDelta,
                                           double initialDamping,
                                           double maxPointStepNorm);

} // namespace xjw::detail::native_cuda
