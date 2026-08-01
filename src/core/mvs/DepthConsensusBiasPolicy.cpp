#include "DepthConsensusBiasPolicy.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace xjw
{
namespace mvs
{
namespace
{

bool compatible(const cv::Mat &matrix, const cv::Mat &reference, int type)
{
    return !matrix.empty() && matrix.type() == type &&
           matrix.size() == reference.size();
}

bool validDepth(float depth)
{
    return std::isfinite(depth) && depth > 0.0f;
}

float weightedMedian(std::vector<std::pair<float, float>> *samples)
{
    if (!samples || samples->empty())
    {
        return 0.0f;
    }
    std::sort(samples->begin(), samples->end(),
              [](const auto &left, const auto &right)
              {
                  return left.first < right.first;
              });
    float total_weight = 0.0f;
    for (const auto &[value, weight] : *samples)
    {
        (void)value;
        total_weight += std::max(0.0f, weight);
    }
    if (!(total_weight > 0.0f))
    {
        return samples->at(samples->size() / 2).first;
    }
    const float target = total_weight * 0.5f;
    float accumulated = 0.0f;
    for (const auto &[value, weight] : *samples)
    {
        accumulated += std::max(0.0f, weight);
        if (accumulated >= target)
        {
            return value;
        }
    }
    return samples->back().first;
}

float evidenceWeight(float confidence,
                     std::uint16_t geometry_support,
                     float spread,
                     const ReferenceAnchoredDepthConsensusOptions &options)
{
    const float confidence_weight = std::clamp(confidence, 0.0f, 1.0f);
    const float support_weight = std::clamp(
        (static_cast<float>(geometry_support) - 1.0f) / 3.0f, 0.50f, 1.0f);
    const float spread_weight = options.maximumInverseDepthSpread > 0.0f
        ? std::clamp(1.0f - spread / options.maximumInverseDepthSpread,
                     0.10f,
                     1.0f)
        : 1.0f;
    return std::clamp(options.blendWeight, 0.0f, 1.0f) *
           confidence_weight * support_weight * spread_weight;
}

} // namespace

ReferenceAnchoredDepthConsensusResult makeReferenceAnchoredDepthConsensus(
    const cv::Mat &rawDepth,
    const cv::Mat &inverseDepthMean,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthRelativeSpread,
    const cv::Mat &confidence,
    const cv::Mat &crossViewRepairedMask,
    const cv::Mat &supportMask,
    const cv::Mat &depthValidMask,
    const ReferenceAnchoredDepthConsensusOptions &options)
{
    ReferenceAnchoredDepthConsensusResult result;
    if (rawDepth.empty() || rawDepth.type() != CV_32FC1 ||
        !compatible(inverseDepthMean, rawDepth, CV_32FC1) ||
        !compatible(geometrySupportCount, rawDepth, CV_16UC1) ||
        !compatible(inverseDepthRelativeSpread, rawDepth, CV_32FC1) ||
        !compatible(confidence, rawDepth, CV_32FC1) ||
        !compatible(supportMask, rawDepth, CV_8UC1) ||
        !compatible(depthValidMask, rawDepth, CV_8UC1))
    {
        return result;
    }
    const bool has_repaired_mask = compatible(
        crossViewRepairedMask, rawDepth, CV_8UC1);
    result.depth = rawDepth.clone();
    result.eligibleMask = cv::Mat(rawDepth.size(), CV_8UC1, cv::Scalar(0));
    result.appliedMask = cv::Mat(rawDepth.size(), CV_8UC1, cv::Scalar(0));

    cv::Mat interior_mask;
    cv::compare(supportMask, 0, interior_mask, cv::CMP_GT);
    const int contour_exclusion = std::max(0, options.contourExclusionPixels);
    if (contour_exclusion > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(contour_exclusion * 2 + 1,
                     contour_exclusion * 2 + 1));
        cv::erode(interior_mask, interior_mask, kernel);
    }
    cv::bitwise_and(interior_mask, depthValidMask, interior_mask);

    std::vector<std::pair<float, float>> bias_samples;
    bias_samples.reserve(static_cast<std::size_t>(cv::countNonZero(interior_mask)));
    for (int row = 0; row < rawDepth.rows; ++row)
    {
        const float *raw_row = rawDepth.ptr<float>(row);
        const float *inverse_row = inverseDepthMean.ptr<float>(row);
        const std::uint16_t *support_row = geometrySupportCount.ptr<std::uint16_t>(row);
        const float *spread_row = inverseDepthRelativeSpread.ptr<float>(row);
        const float *confidence_row = confidence.ptr<float>(row);
        const std::uint8_t *interior_row = interior_mask.ptr<std::uint8_t>(row);
        const std::uint8_t *repaired_row = has_repaired_mask
            ? crossViewRepairedMask.ptr<std::uint8_t>(row) : nullptr;
        std::uint8_t *eligible_row = result.eligibleMask.ptr<std::uint8_t>(row);
        for (int column = 0; column < rawDepth.cols; ++column)
        {
            const float raw_depth = raw_row[column];
            const float inverse_depth = inverse_row[column];
            const float spread = spread_row[column];
            const float confidence_value = confidence_row[column];
            if (interior_row[column] == 0 ||
                (repaired_row && repaired_row[column] != 0) ||
                support_row[column] < std::max(2, options.minimumGeometrySupport) ||
                !validDepth(raw_depth) || !std::isfinite(inverse_depth) ||
                inverse_depth <= 1.0e-12f || !std::isfinite(spread) || spread < 0.0f ||
                spread > options.maximumInverseDepthSpread ||
                !std::isfinite(confidence_value) ||
                confidence_value < options.minimumConfidence)
            {
                continue;
            }
            const float consensus_depth = 1.0f / inverse_depth;
            if (!validDepth(consensus_depth) ||
                std::fabs(consensus_depth - raw_depth) / raw_depth >
                    std::max(0.0f, options.maximumCandidateRelativeDifference))
            {
                continue;
            }
            eligible_row[column] = 255;
            bias_samples.emplace_back(
                consensus_depth - raw_depth,
                evidenceWeight(confidence_value,
                               support_row[column],
                               spread,
                               options));
        }
    }

    result.calibration.sampleCount = static_cast<int>(bias_samples.size());
    if (result.calibration.sampleCount <
        std::max(1, options.minimumBiasSampleCount))
    {
        return result;
    }
    result.calibration.additiveDepthBias = weightedMedian(&bias_samples);
    result.calibration.valid = std::isfinite(
        result.calibration.additiveDepthBias);
    if (!result.calibration.valid)
    {
        return result;
    }

    for (int row = 0; row < rawDepth.rows; ++row)
    {
        const float *raw_row = rawDepth.ptr<float>(row);
        const float *inverse_row = inverseDepthMean.ptr<float>(row);
        const std::uint16_t *support_row = geometrySupportCount.ptr<std::uint16_t>(row);
        const float *spread_row = inverseDepthRelativeSpread.ptr<float>(row);
        const float *confidence_row = confidence.ptr<float>(row);
        const std::uint8_t *eligible_row = result.eligibleMask.ptr<std::uint8_t>(row);
        float *output_row = result.depth.ptr<float>(row);
        std::uint8_t *applied_row = result.appliedMask.ptr<std::uint8_t>(row);
        for (int column = 0; column < rawDepth.cols; ++column)
        {
            if (eligible_row[column] == 0)
            {
                continue;
            }
            const float raw_depth = raw_row[column];
            const float consensus_depth = 1.0f / inverse_row[column];
            float correction = consensus_depth - raw_depth -
                               result.calibration.additiveDepthBias;
            const float maximum_correction = raw_depth * std::max(
                0.0f, options.maximumAppliedRelativeCorrection);
            correction = std::clamp(
                correction, -maximum_correction, maximum_correction);
            correction *= evidenceWeight(confidence_row[column],
                                         support_row[column],
                                         spread_row[column],
                                         options);
            const float refined_depth = raw_depth + correction;
            if (!validDepth(refined_depth))
            {
                continue;
            }
            output_row[column] = refined_depth;
            if (std::fabs(correction) > 1.0e-8f)
            {
                applied_row[column] = 255;
                ++result.appliedPixelCount;
            }
        }
    }
    return result;
}

} // namespace mvs
} // namespace xjw
