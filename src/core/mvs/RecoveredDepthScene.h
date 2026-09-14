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

    struct RecoveredSourceMask
    {
        std::vector<std::uint8_t> bytes;
        std::string source = "full_image";
        float coverage = 1.0f;
    };

    /// Converts PlaScan's project/prepared mask conventions to the recovered
    /// PatchMatch convention (zero rejects, non-zero permits).
    bool prepareRecoveredSourceMask(const CameraView& view, RecoveredSourceMask* result, std::string* errorMessage);

    struct RecoveredDepthFrame
    {
        int viewIndex = -1;
        cv::Mat depth;
        cv::Mat confidence;
        cv::Mat validMask;
        cv::Mat supportRegionMask;
        cv::Mat photometricSourceMask;
        /// Conservative bridge from recovered weighted voting to PlaScan's
        /// persisted fusion contract. A retained pixel proves at least the
        /// reference plus one cross-view observation, but the recovered
        /// producer does not expose an exact unweighted observation count.
        cv::Mat geometrySupportCount;
        /// CV_32FC1. Retained pixels are NaN because recovered voting does not
        /// expose the samples needed to measure inverse-depth dispersion.
        cv::Mat inverseDepthRelativeSpread;
        FramePinholeCamera camera;
        std::vector<int> sourceViewIndices;
        std::string maskSource = "full_image";
        float maskCoverage = 1.0f;
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
                                const std::string& qualityProfile,
                                RecoveredDepthSceneResult* result,
                                std::string* errorMessage);

} // namespace xjw::mvs
