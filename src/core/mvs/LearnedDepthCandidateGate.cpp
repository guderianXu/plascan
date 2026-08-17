#include "LearnedDepthCandidateGate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace xjw::mvs
{

LearnedDepthCandidateGateStats gateLearnedDepthCandidate(
    cv::Mat &depth,
    cv::Mat &confidence,
    const cv::Mat &candidateDepth,
    const cv::Mat &candidateConfidence,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthMean,
    const cv::Mat &inverseDepthRelativeSpread,
    cv::Mat *acceptedMask,
    const LearnedDepthCandidateGateOptions &options)
{
    LearnedDepthCandidateGateStats stats;
    const cv::Size size = depth.size();
    stats.validInputs = !depth.empty() && depth.type() == CV_32FC1
        && confidence.type() == CV_32FC1 && confidence.size() == size
        && candidateDepth.type() == CV_32FC1 && candidateDepth.size() == size
        && candidateConfidence.type() == CV_32FC1
        && candidateConfidence.size() == size
        && geometrySupportCount.type() == CV_16UC1
        && geometrySupportCount.size() == size
        && inverseDepthMean.type() == CV_32FC1
        && inverseDepthMean.size() == size
        && inverseDepthRelativeSpread.type() == CV_32FC1
        && inverseDepthRelativeSpread.size() == size;
    if (!stats.validInputs)
    {
        if (acceptedMask)
        {
            acceptedMask->release();
        }
        return stats;
    }

    cv::Mat accepted = cv::Mat::zeros(size, CV_8UC1);
    for (int y = 0; y < depth.rows; ++y)
    {
        float *depth_row = depth.ptr<float>(y);
        float *confidence_row = confidence.ptr<float>(y);
        const float *candidate_row = candidateDepth.ptr<float>(y);
        const float *candidate_confidence_row =
            candidateConfidence.ptr<float>(y);
        const std::uint16_t *support_row =
            geometrySupportCount.ptr<std::uint16_t>(y);
        const float *inverse_mean_row = inverseDepthMean.ptr<float>(y);
        const float *spread_row = inverseDepthRelativeSpread.ptr<float>(y);
        std::uint8_t *accepted_row = accepted.ptr<std::uint8_t>(y);
        for (int x = 0; x < depth.cols; ++x)
        {
            const float candidate = candidate_row[x];
            if (!std::isfinite(candidate) || candidate <= 0.0f)
            {
                continue;
            }
            ++stats.candidatePixelCount;
            const float candidate_confidence = candidate_confidence_row[x];
            if (!std::isfinite(candidate_confidence)
                || candidate_confidence < options.minimumCandidateConfidence)
            {
                ++stats.rejectedConfidenceCount;
                continue;
            }
            const float inverse_mean = inverse_mean_row[x];
            const float spread = spread_row[x];
            if (support_row[x] < options.minimumGeometryObservationCount
                || !std::isfinite(inverse_mean) || inverse_mean <= 0.0f
                || !std::isfinite(spread) || spread < 0.0f
                || spread > options.maximumInverseDepthRelativeSpread)
            {
                ++stats.rejectedGeometryCount;
                continue;
            }
            ++stats.geometrySupportedPixelCount;
            const float expected_depth = 1.0f / inverse_mean;
            const float candidate_error = std::fabs(candidate - expected_depth)
                / std::max(expected_depth, 1.0e-6f);
            if (candidate_error > options.maximumRelativeDepthDifference)
            {
                ++stats.rejectedDepthDifferenceCount;
                continue;
            }

            const float current_depth = depth_row[x];
            bool accept = current_depth <= 0.0f || !std::isfinite(current_depth);
            if (!accept)
            {
                const float current_error = std::fabs(
                    current_depth - expected_depth)
                    / std::max(expected_depth, 1.0e-6f);
                accept = candidate_error < current_error
                    && candidate_confidence >= confidence_row[x]
                        + options.replacementConfidenceMargin;
            }
            if (!accept)
            {
                continue;
            }
            if (current_depth > 0.0f && std::isfinite(current_depth))
            {
                ++stats.replacedPixelCount;
            }
            else
            {
                ++stats.filledPixelCount;
            }
            depth_row[x] = candidate;
            const float geometry_scale = std::clamp(
                1.0f - spread
                    / std::max(options.maximumInverseDepthRelativeSpread,
                               1.0e-6f),
                0.0f,
                1.0f);
            confidence_row[x] = std::min(
                candidate_confidence,
                0.50f + 0.50f * geometry_scale);
            accepted_row[x] = 255;
            ++stats.acceptedPixelCount;
        }
    }
    if (acceptedMask)
    {
        *acceptedMask = std::move(accepted);
    }
    return stats;
}

} // namespace xjw::mvs
