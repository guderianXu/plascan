#pragma once

#include "DepthPyramidPropagation.h"

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace xjw
{
namespace mvs
{

struct PatchMatchBackendRequest
{
    cv::Mat referenceImage;
    cv::Mat referenceValidMask;
    std::vector<cv::Mat> sourceImages;
    std::vector<cv::Mat> sourceValidMasks;
    std::vector<cv::Mat> sourceDepthMaps;
    FramePinholeCamera referenceCamera;
    std::vector<FramePinholeCamera> sourceCameras;
    float zNear = 0.0f;
    float zFar = 0.0f;
    DepthPyramidLevelConfig levelConfig;
    const DepthSearchPrior *prior = nullptr;
};

class IPatchMatchBackend
{
public:
    virtual ~IPatchMatchBackend() = default;

    virtual bool estimate(const PatchMatchBackendRequest &request,
                          DepthLevelResult &result,
                          std::string *errorMessage) = 0;
};

struct DepthLevelSummary
{
    int level = 1;
    int downsampleFactor = 1;
    int validPixelCount = 0;
    float validCoverage = 0.0f;
    float meanConfidence = 0.0f;
    float meanSupportViews = 0.0f;
    float depthDiscontinuityRatio = 0.0f;
    double elapsedMs = 0.0;
    bool success = false;
    std::string errorMessage;
};

struct DepthPyramidRequest
{
    cv::Mat referenceImage;
    cv::Mat referenceValidMask;
    std::vector<cv::Mat> sourceImages;
    std::vector<cv::Mat> sourceValidMasks;
    std::vector<cv::Mat> sourceDepthMaps;
    cv::Mat guideImage;
    FramePinholeCamera referenceCamera;
    std::vector<FramePinholeCamera> sourceCameras;
    float zNear = 0.0f;
    float zFar = 0.0f;
    DepthPyramidConfig pyramidConfig;
    std::array<cv::Mat, 3> sparseDepthHints;
    float sparseDepthHintRelativeRadius = 0.0f; ///< >0 时为提示深度显式设置相对搜索半径
    const std::atomic_bool *cancelFlag = nullptr;
    /// Optional calibration gate invoked after the first successful pyramid
    /// level. Rejecting it aborts the complete frame; the coarse result is
    /// diagnostic-only and must never be returned as a parent fallback.
    std::function<bool(const DepthLevelSummary &, std::string *)>
        firstLevelCompletionGate;
};

struct DepthPyramidResult
{
    bool success = false;
    DepthLevelResult finalLevel;
    std::vector<DepthLevelResult> intermediateLevels;
    std::vector<DepthLevelSummary> levelSummaries;
    std::string errorMessage;
};

class DepthPyramidEstimator
{
public:
    explicit DepthPyramidEstimator(IPatchMatchBackend *backend = nullptr);
    ~DepthPyramidEstimator();

    DepthPyramidResult estimate(const DepthPyramidRequest &request);

private:
    std::unique_ptr<IPatchMatchBackend> _ownedBackend;
    IPatchMatchBackend *_backend = nullptr;
};

} // namespace mvs
} // namespace xjw
