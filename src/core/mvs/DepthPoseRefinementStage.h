#pragma once

#include "DepthPoseAlignmentRefiner.h"
#include "camera/FramePinholeCamera.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace xjw::mvs
{

struct DepthPoseRefinementOptions
{
    bool enabled = false;
    bool emitDerivedCameraCandidates = true;
    int samplingStridePixels = 12;
    int maximumSamplesPerCamera = 2048;
    int maximumSourceFramesPerCamera = 4;
    double minimumAdaptiveSupportWeight = 0.45;
    double minimumAdaptiveEffectiveViewCount = 1.25;
    double maximumAdaptiveConflictRatio = 0.35;
    double maximumCorrespondenceRelativeDepthError = 0.04;
    double occlusionRelativeDepthTolerance = 0.02;
    double minimumEvidenceSampleCoverage = 0.01;
    double minimumProjectionRetentionRatio = 0.98;
    DepthPoseAlignmentOptions optimizer;
};

struct DepthPoseRefinementFrame
{
    int cameraIndex = -1;
    FramePinholeCamera camera;
    cv::Mat depthMap;
    cv::Mat normalMap;
    cv::Mat confidence;
    cv::Mat adaptiveSupportWeight;
    cv::Mat adaptiveEffectiveViewCount;
    cv::Mat adaptiveConflictRatio;
    std::vector<int> sourceCameraIndices;
};

struct DepthPoseRefinementCandidate
{
    int cameraIndex = -1;
    bool evidenceComplete = false;
    bool accepted = false;
    int evidencePixelCount = 0;
    int generatedCorrespondenceCount = 0;
    int occludedCandidateCount = 0;
    int depthConflictCandidateCount = 0;
    double evidenceSampleCoverage = 0.0;
    double projectionRetentionRatio = 0.0;
    double correctionTranslation = 0.0;
    double correctionRotationDegrees = 0.0;
    std::string reason;
    DepthPoseAlignmentCorrection correction;
    FramePinholeCamera derivedCamera;
};

struct DepthPoseRefinementStageResult
{
    bool enabled = false;
    bool candidateOnly = true;
    bool acceptedAny = false;
    int anchorCameraIndex = 0;
    std::vector<DepthPoseRefinementCandidate> candidates;
};

class DepthPoseRefinementStage
{
public:
    static DepthPoseRefinementStageResult buildCandidates(
        const std::vector<DepthPoseRefinementFrame> &frames,
        const DepthPoseRefinementOptions &options = {});

    static FramePinholeCamera deriveCameraCandidate(
        const FramePinholeCamera &camera,
        const DepthPoseAlignmentCorrection &correction);
};

} // namespace xjw::mvs
