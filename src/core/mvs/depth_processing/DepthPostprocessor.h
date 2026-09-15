#pragma once

#include "../MvsTypes.h"

namespace xjw::mvs
{

    /// Synchronous pixel processing shared by estimation, artifact replay and fusion.
    /// Mutates only supplied maps; never owns a task or writes files.
    class DepthPostprocessor final
    {
    public:
        static void
        applySparseSupportPrior(cv::Mat& depthMap, cv::Mat& confidenceMap, const cv::Mat& supportMask, int refIdx);
        static int removeLocalDepthOutliers(cv::Mat& depthMap,
                                            cv::Mat& confidenceMap,
                                            int kernelSize,
                                            float relDepthThreshold,
                                            float maxRemovalRatio,
                                            int refIdx,
                                            int sameLayerRadiusPixels = 1);
        static int removeSmallDepthComponents(
            cv::Mat& depthMap, cv::Mat& confidenceMap, int minComponentArea, float maxRemovalRatio, int refIdx);
        static DepthPostProcessStats postprocessFusionDepthMap(cv::Mat& depthMap,
                                                               cv::Mat& confidenceMap,
                                                               const FusionConfig& config,
                                                               int refIdx,
                                                               int viewCount,
                                                               cv::Mat* missingReasonMap = nullptr,
                                                               const DepthPostProcessEvidence* evidence = nullptr,
                                                               const cv::Size& rasterPixelDomainSize = {});
    };

} // namespace xjw::mvs
