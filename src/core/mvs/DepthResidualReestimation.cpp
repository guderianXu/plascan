#include "DepthResidualReestimation.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace xjw::mvs
{
namespace
{

bool compatibleDepth(const cv::Mat &depth, const cv::Size &size)
{
    return !depth.empty() && depth.type() == CV_32FC1 && depth.size() == size;
}

int distinctSectorCount(const std::vector<int> &sector_ids)
{
    std::vector<int> unique;
    unique.reserve(sector_ids.size());
    for (const int sector_id : sector_ids)
    {
        if (std::find(unique.begin(), unique.end(), sector_id) == unique.end())
        {
            unique.push_back(sector_id);
        }
    }
    return static_cast<int>(unique.size());
}

float relativeSpread(const std::vector<float> &values, int begin, int end)
{
    if (begin < 0 || end <= begin || end > static_cast<int>(values.size()))
    {
        return std::numeric_limits<float>::infinity();
    }
    float sum = 0.0f;
    for (int index = begin; index < end; ++index)
    {
        sum += values[static_cast<std::size_t>(index)];
    }
    const float mean = sum / static_cast<float>(end - begin);
    float squared_difference_sum = 0.0f;
    for (int index = begin; index < end; ++index)
    {
        const float difference = values[static_cast<std::size_t>(index)] - mean;
        squared_difference_sum += difference * difference;
    }
    return std::sqrt(
               squared_difference_sum / static_cast<float>(end - begin)) /
        std::max(mean, 1.0e-6f);
}

} // namespace

DepthResidualReestimationTarget buildDepthResidualReestimationTarget(
    const cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const std::vector<int> &sourceSectorIds,
    const DepthResidualReestimationOptions &options)
{
    DepthResidualReestimationTarget target;
    if (referenceDepth.empty() || referenceDepth.type() != CV_32FC1 ||
        supportMask.empty() || supportMask.type() != CV_8UC1 ||
        supportMask.size() != referenceDepth.size() ||
        projectedSourceDepths.size() != sourceSectorIds.size() ||
        projectedSourceDepths.empty() ||
        !std::all_of(
            projectedSourceDepths.cbegin(),
            projectedSourceDepths.cend(),
            [&referenceDepth](const cv::Mat &depth)
            {
                return compatibleDepth(depth, referenceDepth.size());
            }))
    {
        target.skippedReason = QStringLiteral("invalid_inputs");
        return target;
    }

    cv::Mat normalized_support;
    cv::compare(supportMask, 0, normalized_support, cv::CMP_GT);
    target.supportPixelCount = cv::countNonZero(normalized_support);
    cv::bitwise_and(
        normalized_support,
        referenceDepth <= 0.0f,
        target.residualMask);
    target.requestedResidualPixelCount = cv::countNonZero(target.residualMask);
    target.requestedResidualRatio = target.supportPixelCount > 0
        ? static_cast<float>(target.requestedResidualPixelCount) /
            static_cast<float>(target.supportPixelCount)
        : 0.0f;
    if (target.requestedResidualPixelCount <
            std::max(1, options.minimumResidualPixelCount) ||
        target.requestedResidualRatio <
            std::max(0.0f, options.minimumResidualRatio))
    {
        target.skippedReason = QStringLiteral("residual_below_threshold");
        return target;
    }

    target.hintDepth = cv::Mat(
        referenceDepth.size(), CV_32FC1, cv::Scalar(0.0f));
    target.hintRadius = cv::Mat(
        referenceDepth.size(), CV_32FC1, cv::Scalar(0.0f));
    target.layerSourceCount = cv::Mat(
        referenceDepth.size(), CV_8UC1, cv::Scalar(0));
    target.layerSectorCount = cv::Mat(
        referenceDepth.size(), CV_8UC1, cv::Scalar(0));
    const int minimum_source_count = std::max(
        2, options.minimumLayerSourceCount);
    const int minimum_sector_count = std::max(
        2, options.minimumLayerSectorCount);
    const float maximum_spread = std::max(
        0.0f, options.maximumLayerInverseDepthRelativeSpread);
    const float radius_ratio = std::clamp(
        options.maximumPriorRadiusRatio, 0.005f, 0.25f);

    struct Observation
    {
        float inverseDepth = 0.0f;
        int sectorId = 0;
    };
    std::vector<Observation> observations;
    observations.reserve(projectedSourceDepths.size());
    std::vector<float> inverse_depths;
    std::vector<int> sector_ids;
    for (int row = 0; row < referenceDepth.rows; ++row)
    {
        const std::uint8_t *residual_row =
            target.residualMask.ptr<std::uint8_t>(row);
        for (int column = 0; column < referenceDepth.cols; ++column)
        {
            if (residual_row[column] == 0)
            {
                continue;
            }
            observations.clear();
            for (std::size_t source = 0;
                 source < projectedSourceDepths.size();
                 ++source)
            {
                const float projected_depth =
                    projectedSourceDepths[source].at<float>(row, column);
                if (std::isfinite(projected_depth) && projected_depth > 0.0f)
                {
                    observations.push_back(Observation{
                        1.0f / projected_depth,
                        sourceSectorIds[source]});
                }
            }
            if (static_cast<int>(observations.size()) < minimum_source_count)
            {
                ++target.insufficientSourcePixelCount;
                continue;
            }
            std::sort(
                observations.begin(),
                observations.end(),
                [](const Observation &first, const Observation &second)
                {
                    return first.inverseDepth < second.inverseDepth;
                });
            inverse_depths.clear();
            inverse_depths.reserve(observations.size());
            for (const Observation &observation : observations)
            {
                inverse_depths.push_back(observation.inverseDepth);
            }

            int best_begin = -1;
            int best_end = -1;
            float best_spread = std::numeric_limits<float>::infinity();
            bool ambiguous_best = false;
            for (int begin = 0;
                 begin < static_cast<int>(observations.size());
                 ++begin)
            {
                for (int end = begin + minimum_source_count;
                     end <= static_cast<int>(observations.size());
                     ++end)
                {
                    const float spread = relativeSpread(
                        inverse_depths, begin, end);
                    if (!std::isfinite(spread) || spread > maximum_spread)
                    {
                        continue;
                    }
                    const int count = end - begin;
                    const int best_count = best_end - best_begin;
                    if (count > best_count ||
                        (count == best_count && spread < best_spread))
                    {
                        ambiguous_best = false;
                        best_begin = begin;
                        best_end = end;
                        best_spread = spread;
                    }
                    else if (count == best_count &&
                             std::fabs(spread - best_spread) < 1.0e-6f &&
                             begin != best_begin)
                    {
                        ambiguous_best = true;
                    }
                }
            }
            if (best_begin < 0 || ambiguous_best)
            {
                ++target.layerSpreadRejectedPixelCount;
                continue;
            }
            sector_ids.clear();
            float inverse_depth_sum = 0.0f;
            for (int index = best_begin; index < best_end; ++index)
            {
                inverse_depth_sum +=
                    observations[static_cast<std::size_t>(index)].inverseDepth;
                sector_ids.push_back(
                    observations[static_cast<std::size_t>(index)].sectorId);
            }
            const int sector_count = distinctSectorCount(sector_ids);
            if (sector_count < minimum_sector_count)
            {
                ++target.insufficientSectorPixelCount;
                continue;
            }
            const int layer_source_count = best_end - best_begin;
            const float inverse_depth_mean = inverse_depth_sum /
                static_cast<float>(layer_source_count);
            const float prior_depth = 1.0f / inverse_depth_mean;
            target.hintDepth.at<float>(row, column) = prior_depth;
            target.hintRadius.at<float>(row, column) =
                prior_depth * radius_ratio;
            target.layerSourceCount.at<std::uint8_t>(row, column) =
                static_cast<std::uint8_t>(
                    std::min(layer_source_count, 255));
            target.layerSectorCount.at<std::uint8_t>(row, column) =
                static_cast<std::uint8_t>(std::min(sector_count, 255));
            ++target.layerCoveredPixelCount;
        }
    }

    cv::bitwise_and(
        target.residualMask,
        target.hintDepth > 0.0f,
        target.residualMask);
    if (target.layerCoveredPixelCount <= 0)
    {
        target.skippedReason = QStringLiteral("no_multiview_residual_layer");
        return target;
    }
    const int dilation = std::max(0, options.estimationMaskDilationPixels);
    if (dilation > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(dilation * 2 + 1, dilation * 2 + 1));
        cv::dilate(target.residualMask, target.estimationMask, kernel);
    }
    else
    {
        target.estimationMask = target.residualMask.clone();
    }
    cv::bitwise_and(
        target.estimationMask,
        normalized_support,
        target.estimationMask);
    target.valid = true;
    return target;
}

DepthResidualReestimationStats mergeDepthResidualReestimationCandidates(
    cv::Mat &depth,
    cv::Mat &confidence,
    const std::vector<cv::Mat> &candidateDepths,
    const std::vector<cv::Mat> &candidateConfidences,
    const DepthResidualReestimationTarget &target,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const std::vector<int> &sourceSectorIds,
    cv::Mat *recoveredMask,
    const DepthResidualReestimationOptions &options)
{
    DepthResidualReestimationStats stats;
    stats.supportPixelCount = target.supportPixelCount;
    stats.requestedResidualPixelCount = target.requestedResidualPixelCount;
    stats.layerCoveredPixelCount = target.layerCoveredPixelCount;
    stats.insufficientSourcePixelCount = target.insufficientSourcePixelCount;
    stats.insufficientSectorPixelCount = target.insufficientSectorPixelCount;
    stats.layerSpreadRejectedPixelCount = target.layerSpreadRejectedPixelCount;
    stats.skippedReason = target.skippedReason;
    if (!target.valid || depth.empty() || depth.type() != CV_32FC1 ||
        candidateDepths.size() != candidateConfidences.size() ||
        candidateDepths.empty() ||
        projectedSourceDepths.size() != sourceSectorIds.size() ||
        !std::all_of(
            candidateDepths.cbegin(),
            candidateDepths.cend(),
            [&depth](const cv::Mat &candidate)
            {
                return compatibleDepth(candidate, depth.size());
            }) ||
        !std::all_of(
            candidateConfidences.cbegin(),
            candidateConfidences.cend(),
            [&depth](const cv::Mat &candidate)
            {
                return compatibleDepth(candidate, depth.size());
            }) ||
        !std::all_of(
            projectedSourceDepths.cbegin(),
            projectedSourceDepths.cend(),
            [&depth](const cv::Mat &projected)
            {
                return compatibleDepth(projected, depth.size());
            }))
    {
        if (stats.skippedReason.isEmpty())
        {
            stats.skippedReason = QStringLiteral("invalid_candidates");
        }
        return stats;
    }

    stats.attempted = true;
    cv::Mat recovered(depth.size(), CV_8UC1, cv::Scalar(0));
    if (confidence.empty() || confidence.type() != CV_32FC1 ||
        confidence.size() != depth.size())
    {
        confidence = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    }
    const int minimum_hypothesis_count = std::max(
        2, options.minimumCandidateHypothesisCount);
    const float minimum_confidence = std::clamp(
        options.minimumCandidateConfidence, 0.0f, 1.0f);
    const float maximum_hypothesis_spread = std::max(
        0.0f, options.maximumCandidateInverseDepthRelativeSpread);
    const float maximum_prior_difference = std::max(
        0.0f, options.maximumCandidatePriorRelativeDifference);
    const float maximum_geometry_difference = std::max(
        0.0f, options.maximumGeometryRelativeDifference);
    const float free_space_difference = std::max(
        maximum_geometry_difference,
        options.freeSpaceConflictRelativeDifference);
    std::vector<float> inverse_depths;
    std::vector<float> candidate_confidences;
    std::vector<int> confirmed_sectors;
    inverse_depths.reserve(candidateDepths.size());
    candidate_confidences.reserve(candidateDepths.size());
    confirmed_sectors.reserve(projectedSourceDepths.size());

    for (int row = 0; row < depth.rows; ++row)
    {
        float *depth_row = depth.ptr<float>(row);
        float *confidence_row = confidence.ptr<float>(row);
        const std::uint8_t *target_row =
            target.residualMask.ptr<std::uint8_t>(row);
        std::uint8_t *recovered_row = recovered.ptr<std::uint8_t>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (target_row[column] == 0 || depth_row[column] > 0.0f)
            {
                continue;
            }
            inverse_depths.clear();
            candidate_confidences.clear();
            bool has_candidate = false;
            for (std::size_t hypothesis = 0;
                 hypothesis < candidateDepths.size();
                 ++hypothesis)
            {
                const float candidate_depth =
                    candidateDepths[hypothesis].at<float>(row, column);
                if (!std::isfinite(candidate_depth) || candidate_depth <= 0.0f)
                {
                    continue;
                }
                has_candidate = true;
                const float candidate_confidence =
                    candidateConfidences[hypothesis].at<float>(row, column);
                if (!std::isfinite(candidate_confidence) ||
                    candidate_confidence < minimum_confidence)
                {
                    continue;
                }
                inverse_depths.push_back(1.0f / candidate_depth);
                candidate_confidences.push_back(candidate_confidence);
            }
            if (!has_candidate)
            {
                continue;
            }
            ++stats.candidatePixelCount;
            if (inverse_depths.empty())
            {
                ++stats.rejectedConfidencePixelCount;
                continue;
            }
            if (static_cast<int>(inverse_depths.size()) <
                minimum_hypothesis_count)
            {
                ++stats.rejectedInsufficientHypothesisPixelCount;
                continue;
            }
            float inverse_depth_sum = 0.0f;
            float minimum_merged_confidence = 1.0f;
            for (std::size_t index = 0; index < inverse_depths.size(); ++index)
            {
                inverse_depth_sum += inverse_depths[index];
                minimum_merged_confidence = std::min(
                    minimum_merged_confidence,
                    candidate_confidences[index]);
            }
            const float inverse_depth_mean = inverse_depth_sum /
                static_cast<float>(inverse_depths.size());
            float squared_difference_sum = 0.0f;
            for (const float inverse_depth : inverse_depths)
            {
                const float difference = inverse_depth - inverse_depth_mean;
                squared_difference_sum += difference * difference;
            }
            const float hypothesis_spread = std::sqrt(
                squared_difference_sum /
                static_cast<float>(inverse_depths.size())) /
                std::max(inverse_depth_mean, 1.0e-6f);
            if (!std::isfinite(hypothesis_spread) ||
                hypothesis_spread > maximum_hypothesis_spread)
            {
                ++stats.rejectedHypothesisSpreadPixelCount;
                continue;
            }
            ++stats.consensusCandidatePixelCount;
            const float candidate_depth = 1.0f / inverse_depth_mean;
            const float prior_depth = target.hintDepth.at<float>(row, column);
            const float prior_difference =
                std::fabs(candidate_depth - prior_depth) /
                std::max(prior_depth, 1.0e-6f);
            if (!std::isfinite(prior_depth) || prior_depth <= 0.0f ||
                prior_difference > maximum_prior_difference)
            {
                ++stats.rejectedPriorPixelCount;
                continue;
            }

            int geometry_confirmation_count = 0;
            bool free_space_conflict = false;
            confirmed_sectors.clear();
            for (std::size_t source = 0;
                 source < projectedSourceDepths.size();
                 ++source)
            {
                const float projected_depth =
                    projectedSourceDepths[source].at<float>(row, column);
                if (!std::isfinite(projected_depth) || projected_depth <= 0.0f)
                {
                    continue;
                }
                const float signed_difference =
                    (candidate_depth - projected_depth) /
                    std::max(projected_depth, 1.0e-6f);
                if (std::fabs(signed_difference) <= maximum_geometry_difference)
                {
                    ++geometry_confirmation_count;
                    confirmed_sectors.push_back(sourceSectorIds[source]);
                }
                else if (signed_difference < -free_space_difference)
                {
                    free_space_conflict = true;
                    break;
                }
            }
            if (free_space_conflict)
            {
                ++stats.rejectedFreeSpacePixelCount;
                continue;
            }
            if (geometry_confirmation_count <
                std::max(2, options.minimumGeometryConfirmationCount))
            {
                ++stats.rejectedGeometryPixelCount;
                continue;
            }
            if (distinctSectorCount(confirmed_sectors) <
                std::max(2, options.minimumGeometrySectorCount))
            {
                ++stats.rejectedGeometrySectorPixelCount;
                continue;
            }
            depth_row[column] = candidate_depth;
            confidence_row[column] = std::max(
                minimum_merged_confidence,
                std::clamp(options.recoveredConfidence, 0.0f, 1.0f));
            recovered_row[column] = 255;
            ++stats.recoveredPixelCount;
        }
    }
    if (stats.layerCoveredPixelCount > 0)
    {
        stats.recoveryRatio = static_cast<float>(stats.recoveredPixelCount) /
            static_cast<float>(stats.layerCoveredPixelCount);
    }
    if (recoveredMask)
    {
        *recoveredMask = std::move(recovered);
    }
    return stats;
}

QJsonObject depthResidualReestimationStatsToJson(
    const DepthResidualReestimationStats &stats)
{
    return QJsonObject{
        {QStringLiteral("attempted"), stats.attempted},
        {QStringLiteral("support_pixel_count"), stats.supportPixelCount},
        {QStringLiteral("requested_residual_pixel_count"),
         stats.requestedResidualPixelCount},
        {QStringLiteral("layer_covered_pixel_count"),
         stats.layerCoveredPixelCount},
        {QStringLiteral("insufficient_source_pixel_count"),
         stats.insufficientSourcePixelCount},
        {QStringLiteral("insufficient_sector_pixel_count"),
         stats.insufficientSectorPixelCount},
        {QStringLiteral("layer_spread_rejected_pixel_count"),
         stats.layerSpreadRejectedPixelCount},
        {QStringLiteral("candidate_pixel_count"), stats.candidatePixelCount},
        {QStringLiteral("consensus_candidate_pixel_count"),
         stats.consensusCandidatePixelCount},
        {QStringLiteral("rejected_confidence_pixel_count"),
         stats.rejectedConfidencePixelCount},
        {QStringLiteral("rejected_insufficient_hypothesis_pixel_count"),
         stats.rejectedInsufficientHypothesisPixelCount},
        {QStringLiteral("rejected_hypothesis_spread_pixel_count"),
         stats.rejectedHypothesisSpreadPixelCount},
        {QStringLiteral("rejected_prior_pixel_count"),
         stats.rejectedPriorPixelCount},
        {QStringLiteral("rejected_geometry_pixel_count"),
         stats.rejectedGeometryPixelCount},
        {QStringLiteral("rejected_geometry_sector_pixel_count"),
         stats.rejectedGeometrySectorPixelCount},
        {QStringLiteral("rejected_free_space_pixel_count"),
         stats.rejectedFreeSpacePixelCount},
        {QStringLiteral("recovered_pixel_count"), stats.recoveredPixelCount},
        {QStringLiteral("attempted_hypothesis_count"),
         stats.attemptedHypothesisCount},
        {QStringLiteral("successful_hypothesis_count"),
         stats.successfulHypothesisCount},
        {QStringLiteral("failed_hypothesis_count"),
         stats.failedHypothesisCount},
        {QStringLiteral("source_count"), stats.sourceCount},
        {QStringLiteral("recovery_ratio"), stats.recoveryRatio},
        {QStringLiteral("skipped_reason"), stats.skippedReason},
        {QStringLiteral("schema_version"), 1}};
}

} // namespace xjw::mvs
