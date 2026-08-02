#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace xjw::mvs
{

struct DepthPoseAlignmentSample
{
    int cameraIndex = -1;
    int targetCameraIndex = -1;
    cv::Vec3d sourcePointWorld{0.0, 0.0, 0.0};
    cv::Vec3d targetPointWorld{0.0, 0.0, 0.0};
    cv::Vec3d targetNormalWorld{0.0, 0.0, 1.0};
    double confidence = 1.0;
    bool occluded = false;
};

struct DepthPoseAlignmentOptions
{
    bool enabled = false;
    int anchorCameraIndex = 0;
    int maximumIterations = 12;
    int minimumCorrespondences = 24;
    double huberDelta = 0.002;
    double damping = 1.0e-8;
    double convergenceTranslation = 1.0e-7;
    double convergenceRotationRadians = 1.0e-7;
    double maximumTranslation = 0.02;
    double maximumRotationDegrees = 2.0;
    double requiredP90ImprovementRatio = 0.995;
};

struct DepthPoseAlignmentCorrection
{
    int cameraIndex = -1;
    bool accepted = false;
    int correspondenceCount = 0;
    int iterationCount = 0;
    cv::Vec3d pivotWorld{0.0, 0.0, 0.0};
    cv::Matx33d rotation = cv::Matx33d::eye();
    cv::Vec3d translation{0.0, 0.0, 0.0};
    double residualMedianBefore = 0.0;
    double residualMedianAfter = 0.0;
    double residualP90Before = 0.0;
    double residualP90After = 0.0;
    std::string reason;
};

struct DepthPoseAlignmentResult
{
    bool enabled = false;
    bool acceptedAny = false;
    std::vector<DepthPoseAlignmentCorrection> corrections;
};

class DepthPoseAlignmentRefiner
{
public:
    static DepthPoseAlignmentResult refine(
        const std::vector<DepthPoseAlignmentSample> &samples,
        const DepthPoseAlignmentOptions &options = {});

    static cv::Vec3d applyCorrection(
        const DepthPoseAlignmentCorrection &correction,
        const cv::Vec3d &pointWorld);
};

} // namespace xjw::mvs
