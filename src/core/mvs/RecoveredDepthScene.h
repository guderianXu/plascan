#pragma once

#include "MvsTypes.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace xjw::mvs
{

    /// Reference public d4 calibration: f/cx/cy divided by 4, no Brown terms.
    /// This is not OpenCV half-pixel image resizing.
    FramePinholeCamera recoveredPublicD4Camera(const FramePinholeCamera& source);

    struct RecoveredDepthFrame
    {
        int viewIndex = -1;
        cv::Mat depth;
        cv::Mat confidence;
        cv::Mat validMask;
        cv::Mat photometricSourceMask;
        FramePinholeCamera camera;
        std::vector<int> sourceViewIndices;
    };

    struct RecoveredDepthSceneResult
    {
        std::vector<RecoveredDepthFrame> frames;
        double patchMatchSeconds = 0.0;
        double votingSeconds = 0.0;
    };

    bool runRecoveredDepthScene(const std::vector<CameraView>& views,
                                const SparseCloud& sparseCloud,
                                int cudaDeviceIndex,
                                const std::string& workspaceRoot,
                                RecoveredDepthSceneResult* result,
                                std::string* errorMessage);

} // namespace xjw::mvs
