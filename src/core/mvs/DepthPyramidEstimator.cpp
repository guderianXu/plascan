#include "DepthPyramidEstimator.h"

#include "PatchMatchCUDA.h"
#include "DepthPyramidPolicy.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace xjw
{
namespace mvs
{
namespace
{

class NativePatchMatchBackend final : public IPatchMatchBackend
{
public:
    bool estimate(const PatchMatchBackendRequest &request,
                  DepthLevelResult &result,
                  std::string *error_message) override
    {
        cv::Mat confidence;
        const cv::Mat *hint = request.prior && !request.prior->center.empty()
            ? &request.prior->center
            : nullptr;
        const cv::Mat *radius = request.prior && !request.prior->radius.empty()
            ? &request.prior->radius
            : nullptr;
        if (!PatchMatchDepthEstimator::estimate(request.referenceImage,
                                                request.sourceImages,
                                                request.referenceCamera,
                                                request.sourceCameras,
                                                request.zNear,
                                                request.zFar,
                                                request.levelConfig.patchMatch,
                                                result.depth,
                                                &confidence,
                                                error_message,
                                                hint,
                                                radius))
        {
            return false;
        }

        result.level = request.levelConfig.level;
        result.downsampleFactor = request.levelConfig.patchMatch.downsampleFactor;
        result.confidence = confidence;
        result.validMask = result.depth > 0.0f;
        result.supportCount = cv::Mat(result.depth.size(), CV_16U, cv::Scalar(0));
        result.supportCount.setTo(
            cv::Scalar(static_cast<int>(request.sourceImages.size())), result.validMask);
        result.uncertainty = cv::Mat(result.depth.size(), CV_32F, cv::Scalar(0.0f));
        for (int row = 0; row < result.depth.rows; ++row)
        {
            for (int column = 0; column < result.depth.cols; ++column)
            {
                const float depth = result.depth.at<float>(row, column);
                if (depth <= 0.0f || !std::isfinite(depth))
                {
                    continue;
                }
                const float confidence_value = confidence.empty()
                    ? 0.5f
                    : std::clamp(confidence.at<float>(row, column), 0.0f, 1.0f);
                result.uncertainty.at<float>(row, column) =
                    std::max(0.001f, depth * 0.05f * (1.0f - confidence_value));
            }
        }
        return true;
    }
};

void mergeSparseHint(DepthSearchPrior &prior, const cv::Mat &sparse_hint, cv::Size target_size)
{
    if (sparse_hint.empty())
    {
        return;
    }

    cv::Mat hint;
    if (sparse_hint.size() == target_size)
    {
        hint = sparse_hint;
    }
    else
    {
        cv::resize(sparse_hint, hint, target_size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    if (prior.center.empty())
    {
        prior.center = cv::Mat(target_size, CV_32F, cv::Scalar(0.0f));
        prior.radius = cv::Mat(target_size, CV_32F, cv::Scalar(0.0f));
        prior.validMask = cv::Mat(target_size, CV_8U, cv::Scalar(0));
    }

    const cv::Mat sparse_mask = hint > 0.0f;
    hint.copyTo(prior.center, sparse_mask);
    prior.radius.setTo(cv::Scalar(0.0f), sparse_mask);
    prior.validMask.setTo(cv::Scalar(255), sparse_mask);
}

cv::Mat nativeLevelArtifact(const cv::Mat &artifact, const cv::Size &working_size)
{
    if (artifact.empty() || artifact.size() == working_size)
    {
        return artifact;
    }

    cv::Mat native_artifact;
    cv::resize(artifact, native_artifact, working_size, 0.0, 0.0, cv::INTER_NEAREST);
    return native_artifact;
}

DepthLevelSummary summarizeLevel(const DepthLevelResult &level,
                                 const cv::Size &full_resolution,
                                 double elapsed_ms)
{
    DepthLevelSummary summary;
    summary.level = level.level;
    summary.downsampleFactor = level.downsampleFactor;
    summary.elapsedMs = elapsed_ms;
    summary.success = !level.depth.empty();
    if (level.depth.empty())
    {
        return summary;
    }

    const cv::Size working_size = depthPyramidWorkingSize(
        full_resolution.width,
        full_resolution.height,
        level.downsampleFactor);
    const cv::Mat native_depth = nativeLevelArtifact(level.depth, working_size);
    const cv::Mat valid_mask = native_depth > 0.0f;
    summary.validPixelCount = cv::countNonZero(valid_mask);
    summary.validCoverage = static_cast<float>(summary.validPixelCount) /
                            std::max(1, native_depth.rows * native_depth.cols);
    if (!level.confidence.empty() && summary.validPixelCount > 0)
    {
        const cv::Mat native_confidence = nativeLevelArtifact(level.confidence, working_size);
        summary.meanConfidence = static_cast<float>(cv::mean(native_confidence, valid_mask)[0]);
    }
    return summary;
}

} // namespace

DepthPyramidEstimator::DepthPyramidEstimator(IPatchMatchBackend *backend)
{
    if (backend)
    {
        _backend = backend;
    }
    else
    {
        _ownedBackend = std::make_unique<NativePatchMatchBackend>();
        _backend = _ownedBackend.get();
    }
}

DepthPyramidEstimator::~DepthPyramidEstimator() = default;

DepthPyramidResult DepthPyramidEstimator::estimate(const DepthPyramidRequest &request)
{
    DepthPyramidResult result;
    if (!_backend || request.referenceImage.empty() || request.sourceImages.empty())
    {
        result.errorMessage = "depth pyramid request is missing images or backend";
        return result;
    }

    const int level_count = std::clamp(request.pyramidConfig.activeLevelCount, 1, 3);
    DepthLevelResult parent;
    bool has_parent = false;
    for (int index = 0; index < level_count; ++index)
    {
        if (request.cancelFlag && request.cancelFlag->load(std::memory_order_relaxed))
        {
            result.errorMessage = "depth pyramid cancelled";
            return result;
        }

        const DepthPyramidLevelConfig &level_config = request.pyramidConfig.levels[index];
        const cv::Size target_size = depthPyramidWorkingSize(
            request.referenceImage.cols,
            request.referenceImage.rows,
            level_config.patchMatch.downsampleFactor);
        DepthSearchPrior prior;
        if (has_parent)
        {
            const cv::Mat &guide = request.guideImage.empty()
                ? request.referenceImage
                : request.guideImage;
            prior = propagateDepthPrior(parent, guide, target_size);
        }
        mergeSparseHint(prior, request.sparseDepthHints[index], target_size);

        PatchMatchBackendRequest backend_request;
        backend_request.referenceImage = request.referenceImage;
        backend_request.sourceImages = request.sourceImages;
        backend_request.referenceCamera = request.referenceCamera;
        backend_request.sourceCameras = request.sourceCameras;
        backend_request.zNear = request.zNear;
        backend_request.zFar = request.zFar;
        backend_request.levelConfig = level_config;
        backend_request.prior = prior.center.empty() ? nullptr : &prior;

        DepthLevelResult level_result;
        std::string level_error;
        const auto start = std::chrono::steady_clock::now();
        const bool level_ok = _backend->estimate(backend_request, level_result, &level_error);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        DepthLevelSummary summary = summarizeLevel(
            level_result,
            request.referenceImage.size(),
            elapsed_ms);
        summary.success = level_ok && !level_result.depth.empty();
        summary.errorMessage = level_error;
        result.levelSummaries.push_back(summary);

        if (!summary.success)
        {
            result.errorMessage = level_error.empty() ? "depth pyramid level failed" : level_error;
            if (!has_parent)
            {
                return result;
            }
            break;
        }

        if (request.pyramidConfig.saveIntermediateLevels && index + 1 < level_count)
        {
            result.intermediateLevels.push_back(level_result);
        }
        parent = std::move(level_result);
        has_parent = true;
    }

    if (has_parent)
    {
        result.finalLevel = std::move(parent);
        result.success = true;
    }
    return result;
}

} // namespace mvs
} // namespace xjw
