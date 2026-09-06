#pragma once

/**
 * @file ImageRegistrationEngine.h
 * @brief 无外部位姿时的增量影像注册循环。
 *
 * 每轮按可见已三角化点数选择候选影像，构造 2D-3D 对应并执行 PnP RANSAC；
 * 注册成功后立即扩展轨迹、三角化新点，并按参考阶段稳定当前完整相机块。
 */

#include "IncrementalSfm.h"

namespace xjw
{

    /// 增量注册执行器，操作 owner 中的重建、对应图和三角化器。
    class ImageRegistrationEngine
    {
    public:
        explicit ImageRegistrationEngine(IncrementalSfm& owner);

        /**
         * @brief 从已完成初始对的 owner 状态继续注册剩余影像。
         * @param totalImages 用于进度和覆盖率统计，不改变 owner 的影像集合。
         * @param initialPairTrial true 时采用初始像对 evaluator 的 20 次/净增量 100 BA 策略。
         */
        IncrementalSfmResult run(int totalImages,
                                 SfmProgressCallback progressCb,
                                 int registrationLimit = 0,
                                 bool runFinalRefinement = true,
                                 bool initialPairTrial = false,
                                 bool stabilizeModel = true,
                                 bool runInitialGrowthRefinement = true);

        /**
         * @brief 查找近垂直、高光轴集中度相机块中的小规模姿态离群项。
         *
         * 该判据只看相机光轴，不把低重投影误差误当成全局位姿正确性；调用方还需
         * 结合序列模式、绝对约束和离群比例决定是否执行自动修复。
         */
        static std::vector<ImageId> findParallelAerialPoseOutliers(const SfmReconstruction& reconstruction,
                                                                   double minimumAxisConcentration = 0.98,
                                                                   double minimumAngularDeviationDegrees = 20.0);

    private:
        IncrementalSfm& _owner;
    };

} // namespace xjw
