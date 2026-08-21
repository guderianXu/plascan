#include "DepthPyramidEstimator.h"

#include "PatchMatchCUDA.h"
#include "DepthPyramidPolicy.h"
#include "MvsQualityReport.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

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
                                                radius,
                                                request.referenceValidMask.empty()
                                                    ? nullptr
                                                    : &request.referenceValidMask,
                                                request.sourceValidMasks.empty()
                                                    ? nullptr
                                                    : &request.sourceValidMasks))
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

std::vector<int> virtualNearestRoundTripIndices(int native_extent,
                                                int logical_extent,
                                                int target_extent)
{
    std::vector<int> indices(static_cast<std::size_t>(target_extent));
    const double target_to_logical_scale =
        static_cast<double>(logical_extent) / target_extent;
    const double logical_to_native_scale =
        static_cast<double>(native_extent) / logical_extent;
    for (int target_index = 0; target_index < target_extent; ++target_index)
    {
        const int logical_index = std::min(
            static_cast<int>(target_index * target_to_logical_scale),
            logical_extent - 1);
        indices[static_cast<std::size_t>(target_index)] = std::min(
            static_cast<int>(logical_index * logical_to_native_scale),
            native_extent - 1);
    }
    return indices;
}

cv::Mat legacySummaryArtifact(const cv::Mat &artifact,
                              const cv::Size &full_resolution,
                              const cv::Size &working_size)
{
    if (artifact.empty())
    {
        return artifact;
    }

    const std::vector<int> source_columns = virtualNearestRoundTripIndices(
        artifact.cols, full_resolution.width, working_size.width);
    const std::vector<int> source_rows = virtualNearestRoundTripIndices(
        artifact.rows, full_resolution.height, working_size.height);
    bool identity_columns = artifact.cols == working_size.width;
    for (int column = 0; identity_columns && column < working_size.width; ++column)
    {
        identity_columns = source_columns[static_cast<std::size_t>(column)] == column;
    }
    bool identity_rows = artifact.rows == working_size.height;
    for (int row = 0; identity_rows && row < working_size.height; ++row)
    {
        identity_rows = source_rows[static_cast<std::size_t>(row)] == row;
    }
    if (identity_columns && identity_rows)
    {
        return artifact;
    }

    cv::Mat sampled(working_size, artifact.type());
    const std::size_t element_size = artifact.elemSize();
    cv::parallel_for_(cv::Range(0, working_size.height), [&](const cv::Range &rows)
    {
        for (int row = rows.start; row < rows.end; ++row)
        {
            const std::uint8_t *source = artifact.ptr(
                source_rows[static_cast<std::size_t>(row)]);
            std::uint8_t *destination = sampled.ptr(row);
            for (int column = 0; column < working_size.width; ++column)
            {
                std::memcpy(destination + static_cast<std::size_t>(column) * element_size,
                            source + static_cast<std::size_t>(
                                source_columns[static_cast<std::size_t>(column)]) * element_size,
                            element_size);
            }
        }
    });
    return sampled;
}

cv::Mat resizedValidMask(const cv::Mat &valid_mask, const cv::Size &target_size)
{
    if (valid_mask.empty() || target_size.width <= 0 || target_size.height <= 0)
    {
        return cv::Mat();
    }

    cv::Mat byte_mask;
    if (valid_mask.type() == CV_8U)
    {
        byte_mask = valid_mask;
    }
    else
    {
        cv::Mat converted;
        valid_mask.convertTo(converted, CV_8U);
        cv::compare(converted, 0, byte_mask, cv::CMP_GT);
    }
    if (byte_mask.size() == target_size)
    {
        return byte_mask;
    }

    cv::Mat resized_mask;
    cv::resize(byte_mask, resized_mask, target_size, 0.0, 0.0, cv::INTER_NEAREST);
    return resized_mask;
}

void resizeLevelResultArtifacts(DepthLevelResult &result, const cv::Size &target_size)
{
    auto resize_artifact = [&target_size](cv::Mat &artifact)
    {
        if (artifact.empty() || artifact.size() == target_size)
        {
            return;
        }

        cv::Mat resized;
        cv::resize(artifact, resized, target_size, 0.0, 0.0, cv::INTER_NEAREST);
        artifact = std::move(resized);
    };

    resize_artifact(result.depth);
    resize_artifact(result.normalMap);
    resize_artifact(result.confidence);
    resize_artifact(result.supportCount);
    resize_artifact(result.uncertainty);
    resize_artifact(result.validMask);
}

void applyValidMaskToPrior(DepthSearchPrior &prior, const cv::Mat &valid_mask)
{
    if (valid_mask.empty() || prior.center.empty())
    {
        return;
    }

    const cv::Mat invalid_mask = valid_mask == 0;
    prior.center.setTo(cv::Scalar(0.0f), invalid_mask);
    if (!prior.radius.empty())
    {
        prior.radius.setTo(cv::Scalar(0.0f), invalid_mask);
    }
    if (!prior.normalMap.empty())
    {
        prior.normalMap.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), invalid_mask);
    }
    if (!prior.validMask.empty())
    {
        prior.validMask.setTo(cv::Scalar(0), invalid_mask);
    }
}

void applyValidMaskToLevelResult(DepthLevelResult &result, const cv::Mat &valid_mask)
{
    if (result.depth.empty() || valid_mask.empty())
    {
        return;
    }

    const cv::Mat output_mask = resizedValidMask(valid_mask, result.depth.size());
    const cv::Mat invalid_mask = output_mask == 0;
    result.depth.setTo(cv::Scalar(0.0f), invalid_mask);
    if (!result.normalMap.empty())
    {
        result.normalMap.setTo(cv::Scalar(0.0f, 0.0f, 0.0f), invalid_mask);
    }
    if (!result.confidence.empty())
    {
        result.confidence.setTo(cv::Scalar(0.0f), invalid_mask);
    }
    if (!result.supportCount.empty())
    {
        result.supportCount.setTo(cv::Scalar(0), invalid_mask);
    }
    if (!result.uncertainty.empty())
    {
        result.uncertainty.setTo(cv::Scalar(0.0f), invalid_mask);
    }
    if (result.validMask.empty())
    {
        result.validMask = result.depth > 0.0f;
    }
    result.validMask.setTo(cv::Scalar(0), invalid_mask);
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
    const cv::Mat native_depth = legacySummaryArtifact(
        level.depth, full_resolution, working_size);
    const cv::Mat valid_mask = native_depth > 0.0f;
    summary.validPixelCount = cv::countNonZero(valid_mask);
    summary.validCoverage = static_cast<float>(summary.validPixelCount) /
                            std::max(1, native_depth.rows * native_depth.cols);
    if (!level.confidence.empty() && summary.validPixelCount > 0)
    {
        const cv::Mat native_confidence = legacySummaryArtifact(
            level.confidence, full_resolution, working_size);
        summary.meanConfidence = static_cast<float>(cv::mean(native_confidence, valid_mask)[0]);
    }
    if (!level.supportCount.empty() && summary.validPixelCount > 0)
    {
        const cv::Mat native_support = legacySummaryArtifact(
            level.supportCount, full_resolution, working_size);
        summary.meanSupportViews = static_cast<float>(cv::mean(native_support, valid_mask)[0]);
    }
    summary.depthDiscontinuityRatio = measureDepthDiscontinuityRatio(native_depth);
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
    if (!request.sourceValidMasks.empty() &&
        request.sourceValidMasks.size() != request.sourceImages.size())
    {
        result.errorMessage = "depth pyramid source valid mask count mismatch";
        return result;
    }

    const int level_count = std::clamp(request.pyramidConfig.activeLevelCount, 1, 3);
    const DepthPyramidLevelConfig &finest_level =
        request.pyramidConfig.levels[level_count - 1];
    if ((finest_level.patchMatch.backend == PatchMatchBackend::Cuda ||
         finest_level.patchMatch.backend == PatchMatchBackend::Auto) &&
        PatchMatchDepthEstimator::isCudaAvailable())
    {
        const cv::Size finest_size = depthPyramidWorkingSize(
            request.referenceImage.cols,
            request.referenceImage.rows,
            finest_level.patchMatch.downsampleFactor);
        std::string reserve_error;
        if (!PatchMatchDepthEstimator::reserveGpuWorkspace(
                static_cast<std::size_t>(finest_size.width) *
                    static_cast<std::size_t>(finest_size.height),
                static_cast<int>(request.sourceImages.size()),
                !request.referenceValidMask.empty(),
                !request.sourceValidMasks.empty(),
                &reserve_error,
                finest_level.patchMatch.cudaDeviceIndex))
        {
            result.errorMessage = reserve_error.empty()
                ? "failed to reserve CUDA depth-pyramid workspace"
                : reserve_error;
            return result;
        }
    }
    DepthLevelResult parent;
    bool has_parent = false;
    for (int index = 0; index < level_count; ++index)
    {
        if (request.cancelFlag && request.cancelFlag->load(std::memory_order_relaxed))
        {
            result.errorMessage = "depth pyramid cancelled";
            return result;
        }

        DepthPyramidLevelConfig level_config = request.pyramidConfig.levels[index];
        const bool is_final_level = index + 1 == level_count;
        level_config.patchMatch.returnNativeResolution =
            !is_final_level || request.pyramidConfig.returnNativeFinalResolution;
        const cv::Size target_size = depthPyramidWorkingSize(
            request.referenceImage.cols,
            request.referenceImage.rows,
            level_config.patchMatch.downsampleFactor);
        const cv::Mat level_valid_mask = resizedValidMask(request.referenceValidMask, target_size);
        std::vector<cv::Mat> level_source_valid_masks;
        if (!request.sourceValidMasks.empty())
        {
            level_source_valid_masks.reserve(request.sourceValidMasks.size());
            for (const cv::Mat &source_mask : request.sourceValidMasks)
            {
                level_source_valid_masks.push_back(resizedValidMask(source_mask, target_size));
            }
        }
        DepthSearchPrior prior;
        if (has_parent)
        {
            const cv::Mat &guide = request.guideImage.empty()
                ? request.referenceImage
                : request.guideImage;
            prior = propagateDepthPrior(
                parent,
                guide,
                target_size,
                request.referenceImage.size());
        }
        mergeSparseHint(prior, request.sparseDepthHints[index], target_size);
        applyValidMaskToPrior(prior, level_valid_mask);

        PatchMatchBackendRequest backend_request;
        backend_request.referenceImage = request.referenceImage;
        backend_request.referenceValidMask = level_valid_mask;
        backend_request.sourceImages = request.sourceImages;
        backend_request.sourceValidMasks = std::move(level_source_valid_masks);
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
        if (level_ok)
        {
            if (level_config.patchMatch.returnNativeResolution)
            {
                // Backends which support native output avoid a full-resolution upscale entirely.
                // Normalizing here also keeps legacy/custom backends on the same pyramid contract.
                resizeLevelResultArtifacts(level_result, target_size);
            }
            applyValidMaskToLevelResult(level_result, level_valid_mask);
        }
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        DepthLevelSummary summary = summarizeLevel(
            level_result,
            request.referenceImage.size(),
            elapsed_ms);
        summary.success = level_ok && !level_result.depth.empty();
        summary.errorMessage = level_error;
        const float parent_coverage = result.levelSummaries.empty()
            ? 0.0f
            : result.levelSummaries.back().validCoverage;
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
        if (index == 0 && request.firstLevelCompletionGate)
        {
            std::string gate_error;
            if (!request.firstLevelCompletionGate(summary, &gate_error))
            {
                result.errorMessage = gate_error.empty()
                    ? "first pyramid level rejected by completion gate"
                    : std::move(gate_error);
                result.levelSummaries.back().success = false;
                result.levelSummaries.back().errorMessage = result.errorMessage;
                return result;
            }
        }

        constexpr float kMinimumCoverageRetention = 0.60f;
        if (has_parent && parent_coverage > 0.0f &&
            summary.validCoverage < parent_coverage * kMinimumCoverageRetention)
        {
            std::ostringstream message;
            message << "Level " << summary.level
                    << " coverage regression: " << summary.validCoverage
                    << " < " << kMinimumCoverageRetention
                    << " * parent " << parent_coverage;
            result.errorMessage = message.str();
            result.levelSummaries.back().success = false;
            result.levelSummaries.back().errorMessage = result.errorMessage;
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
        if (!request.pyramidConfig.returnNativeFinalResolution)
        {
            // The default public estimator contract remains full-sized even when a finer level
            // fails and the selected fallback is an internally native-resolution parent.
            resizeLevelResultArtifacts(parent, request.referenceImage.size());
        }
        result.finalLevel = std::move(parent);
        result.success = true;
    }
    return result;
}

} // namespace mvs
} // namespace xjw
