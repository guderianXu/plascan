#pragma once

#include "DepthPyramidPropagation.h"

#include <array>
#include <atomic>
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
    Camera referenceCamera;
    std::vector<Camera> sourceCameras;
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

struct DepthPyramidRequest
{
    cv::Mat referenceImage;
    cv::Mat referenceValidMask;
    std::vector<cv::Mat> sourceImages;
    std::vector<cv::Mat> sourceValidMasks;
    cv::Mat guideImage;
    Camera referenceCamera;
    std::vector<Camera> sourceCameras;
    float zNear = 0.0f;
    float zFar = 0.0f;
    DepthPyramidConfig pyramidConfig;
    std::array<cv::Mat, 3> sparseDepthHints;
    const std::atomic_bool *cancelFlag = nullptr;
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
