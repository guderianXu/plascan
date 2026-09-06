#pragma once

/**
 * @file SfmBundleAdjustCoordinator.h
 * @brief SfM 状态与独立 bundle_adjust 模块之间的适配/调度层。
 *
 * 协调器从当前 SfmReconstruction 构造 FramePinholeCamera/BATrack，选择局部或全局相机集合，
 * 调用 BundleAdjust 公共入口，再将通过质量门控的相机和点原子写回重建。
 */

#include "IncrementalSfm.h"

namespace xjw
{

    struct SfmAdaptiveCameraModelDiagnosticSnapshot
    {
        bool evaluated = false;
        bool applied = false;
        BAIntrinsicParameterMask parameterMask{};
        std::string modelName;
    };

    struct SfmAdaptiveCameraModelDiagnosticMergeResult
    {
        SfmAdaptiveCameraModelDiagnosticSnapshot accumulated;
        bool shouldReplaceEvidence = false;
    };

    struct SfmReferenceBaPolicy
    {
        int maxIterations = 10;
        int maxRounds = 5;
        int minimumNetPointGain = 1;
        bool useVarianceFiltering = false;
        bool fixedRounds = false;
    };

    /// 负责 BA 调度，不拥有 IncrementalSfm。
    class SfmBundleAdjustCoordinator
    {
    public:
        explicit SfmBundleAdjustCoordinator(IncrementalSfm& owner);

        /**
         * @brief 执行一次局部或全局 BA。
         * @param localOnly true 时只释放当前局部窗口相机，窗口外相机作为 gauge 锚。
         * @param anchorIds 调用方额外要求固定的影像 ID。
         */
        void run(bool localOnly = false, const std::vector<ImageId>& anchorIds = {});

        /// 按“BA -> 重三角化 -> 过滤”的稳定化循环执行全局优化。
        /// bool 入口保留给既有调用方；false=正式增量，true=最终方差精化。
        void iterative(bool finalRefinement = false);

        /// 按参考阶段执行对应的 20/10 次内层预算、五轮结构刷新和退出条件。
        void iterative(SfmBundleAdjustmentStage stage);

        /// 返回已恢复的参考 BA 阶段策略，供调度和契约测试共用。
        static SfmReferenceBaPolicy referenceBaPolicy(SfmBundleAdjustmentStage stage);

        /// 将参考 8 参数相机掩码映射到 PlaScan 的 9 参数块（宽高比始终固定）。
        static BAIntrinsicParameterMask referenceIntrinsicParameterMask(SfmBundleAdjustmentStage stage);

        /// 根据本轮新增/删除点数判断是否继续；最终方差阶段始终跑满固定轮数。
        static bool shouldContinueReferenceBa(SfmBundleAdjustmentStage stage, int addedPoints, int removedPoints);

        /// 用保留的输入多视组件原子重建最终点网；质量不足时恢复原点网。
        bool consolidateInputTracksForFinalBa();

        /// 删除在任一有效轨迹观测中落到相机后方的三维点，并清理影像关联。
        int filterNegativeDepthPoints();

        /// 共享镜头内参在完整注册或大型工程达到 98% 注册率后的全局 BA 中更新。
        static bool shouldRefineSharedIntrinsics(bool localOnly,
                                                 int activeCameraCount,
                                                 int registeredImageCount,
                                                 int totalImageCount);

        /// 同时检查点网和共享镜头参数，避免点数稳定但自标定仍在漂移时提前结束。
        static bool hasIterativeGlobalBaConverged(int completedRoundCount,
                                                  double pointChangeRate,
                                                  bool sharedIntrinsicsRefined,
                                                  double maximumIntrinsicChange);

        /// 接近注册终点时跳过周期全局 BA，避免数张影像后立刻重复最终全局 BA。
        static bool shouldRunPeriodicGlobalBa(int registeredImageCount,
                                              int registrationTarget,
                                              int iterationsSinceGlobalBa,
                                              int globalBaInterval);

        /// 仅在多视支撑充分的超大问题中缩减两视图轨迹；弱网保留全部相机约束。
        static bool shouldUseMultiViewOnlyGlobalBa(
            bool localOnly, int activeCameraCount, int totalTrackCount, int twoViewTrackCount, int multiViewTrackCount);

        /// 最终点网重建只有在观测覆盖基本保持且多视冗余真实增加时才替换增量点网。
        static bool shouldAcceptTrackConsolidation(std::size_t oldPointCount,
                                                   std::size_t oldObservationCount,
                                                   std::size_t oldLongTrackCount,
                                                   std::size_t newPointCount,
                                                   std::size_t newObservationCount,
                                                   std::size_t newLongTrackCount);

        /// 仅在最终共享内参自标定时保留当前相机层，不按影像数量或场景形状推断穹顶。
        static bool shouldPreserveCameraLayer(bool localOnly,
                                              bool hasAbsoluteConstraint,
                                              bool completeRegistration,
                                              bool refiningSharedIntrinsics);

        /// 近乎平行的完整对地航摄块采用低阶镜头自标定，避免高阶畸变吸收场景穹顶。
        static bool shouldUseLowOrderAerialSelfCalibration(bool localOnly,
                                                           bool hasAbsoluteConstraint,
                                                           bool completeRegistration,
                                                           bool refiningSharedRadialDistortion,
                                                           int activeCameraCount,
                                                           double opticalAxisConcentration);

        /// 大型工程周期全局 BA 限制为一轮；小工程最多两轮，最终精化保留配置轮数。
        static int iterativeGlobalBaRoundLimit(int configuredRounds, bool finalRefinement, int activeCameraCount);

        /// 累计多轮自适应模型诊断；已应用模型不会被后续 no-op 或跳过轮覆盖。
        static SfmAdaptiveCameraModelDiagnosticMergeResult
        mergeAdaptiveCameraModelDiagnostics(const SfmAdaptiveCameraModelDiagnosticSnapshot& previous,
                                            const SfmAdaptiveCameraModelDiagnosticSnapshot& current);

        /// 首次见到影像时保存内参参考，后续独立全局 BA 调用仍按 ImageId 复用原锚点。
        static std::vector<FramePinholeCamera>
        buildPersistentIntrinsicReferences(const std::vector<ImageId>& imageIds,
                                           const std::vector<FramePinholeCamera>& current,
                                           std::unordered_map<ImageId, FramePinholeCamera>* referencesByImageId);

        /// 按相机计算两轮内参最大归一化变化，避免多标定组反向变化在全局平均中抵消。
        static double maximumCameraIntrinsicChange(const std::vector<FramePinholeCamera>& previous,
                                                   const std::vector<FramePinholeCamera>& current,
                                                   const std::vector<FramePinholeCamera>& stableReferences);

    private:
        IncrementalSfm& _owner;
    };

} // namespace xjw
