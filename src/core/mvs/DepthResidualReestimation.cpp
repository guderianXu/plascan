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

std::vector<std::vector<int>> buildDepthResidualPatchMatchSourceGroups(
    const std::vector<int> &sourceSectorIds)
{
    std::vector<std::vector<int>> groups(2);
    if (sourceSectorIds.size() < 4)
    {
        return groups;
    }

    const int count = static_cast<int>(sourceSectorIds.size());
    int first_excluded = 0;
    int second_excluded = count - 1;
    int best_distance = -1;
    for (int first = 0; first < count; ++first)
    {
        for (int second = first + 1; second < count; ++second)
        {
            int sector_distance = 0;
            if (sourceSectorIds[first] >= 0 && sourceSectorIds[first] < 6 &&
                sourceSectorIds[second] >= 0 && sourceSectorIds[second] < 6)
            {
                const int raw_distance = std::abs(
                    sourceSectorIds[first] - sourceSectorIds[second]);
                sector_distance = std::min(raw_distance, 6 - raw_distance);
            }
            if (sector_distance > best_distance)
            {
                best_distance = sector_distance;
                first_excluded = first;
                second_excluded = second;
            }
        }
    }
    for (int source = 0; source < count; ++source)
    {
        if (source != first_excluded)
        {
            groups[0].push_back(source);
        }
        if (source != second_excluded)
        {
            groups[1].push_back(source);
        }
    }
    return groups;
}

DepthResidualReestimationPreflight inspectDepthResidualReestimationNeed(
    const cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const DepthResidualReestimationOptions &options)
{
    DepthResidualReestimationPreflight preflight;
    if (referenceDepth.empty() || referenceDepth.type() != CV_32FC1 ||
        supportMask.empty() || supportMask.type() != CV_8UC1 ||
        supportMask.size() != referenceDepth.size())
    {
        preflight.skippedReason = QStringLiteral("invalid_inputs");
        return preflight;
    }

    cv::compare(
        supportMask, 0, preflight.normalizedSupport, cv::CMP_GT);
    preflight.supportPixelCount = cv::countNonZero(
        preflight.normalizedSupport);
    cv::bitwise_and(
        preflight.normalizedSupport,
        referenceDepth <= 0.0f,
        preflight.residualMask);
    preflight.requestedResidualPixelCount = cv::countNonZero(
        preflight.residualMask);
    preflight.requestedResidualRatio = preflight.supportPixelCount > 0
        ? static_cast<float>(preflight.requestedResidualPixelCount) /
            static_cast<float>(preflight.supportPixelCount)
        : 0.0f;
    if (preflight.requestedResidualPixelCount <
            std::max(1, options.minimumResidualPixelCount) ||
        preflight.requestedResidualRatio <
            std::max(0.0f, options.minimumResidualRatio))
    {
        preflight.skippedReason = QStringLiteral("residual_below_threshold");
        return preflight;
    }

    preflight.shouldProjectSources = true;
    return preflight;
}

DepthResidualReestimationPreflight inspectDepthReestimationMask(
    const cv::Mat &support_mask,
    const cv::Mat &requested_mask,
    const DepthResidualReestimationOptions &options)
{
    DepthResidualReestimationPreflight preflight;
    if (support_mask.empty() || support_mask.type() != CV_8UC1 ||
        requested_mask.empty() || requested_mask.type() != CV_8UC1 ||
        requested_mask.size() != support_mask.size())
    {
        preflight.skippedReason = QStringLiteral("invalid_inputs");
        return preflight;
    }
    cv::compare(support_mask, 0, preflight.normalizedSupport, cv::CMP_GT);
    cv::Mat normalized_requested;
    cv::compare(requested_mask, 0, normalized_requested, cv::CMP_GT);
    cv::bitwise_and(
        normalized_requested,
        preflight.normalizedSupport,
        normalized_requested);
    cv::Mat labels;
    cv::Mat component_stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        normalized_requested,
        labels,
        component_stats,
        centroids,
        8,
        CV_32S);
    preflight.residualMask = cv::Mat(
        requested_mask.size(), CV_8UC1, cv::Scalar(0));
    const int minimum_component_area = std::max(
        1, options.minimumResidualPixelCount);
    for (int component = 1; component < component_count; ++component)
    {
        if (component_stats.at<int>(component, cv::CC_STAT_AREA) >=
            minimum_component_area)
        {
            preflight.residualMask.setTo(255, labels == component);
        }
    }
    preflight.supportPixelCount = cv::countNonZero(preflight.normalizedSupport);
    preflight.requestedResidualPixelCount = cv::countNonZero(preflight.residualMask);
    preflight.requestedResidualRatio = preflight.supportPixelCount > 0
        ? static_cast<float>(preflight.requestedResidualPixelCount) /
              static_cast<float>(preflight.supportPixelCount)
        : 0.0f;
    if (preflight.requestedResidualPixelCount <
            std::max(1, options.minimumResidualPixelCount) ||
        preflight.requestedResidualRatio < std::max(0.0f, options.minimumResidualRatio))
    {
        preflight.skippedReason = QStringLiteral("residual_below_threshold");
        return preflight;
    }
    preflight.shouldProjectSources = true;
    return preflight;
}

DepthResidualReestimationTarget buildDepthResidualReestimationTarget(
    const cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const std::vector<int> &sourceSectorIds,
    const DepthResidualReestimationOptions &options)
{
    return buildDepthResidualReestimationTarget(
        referenceDepth,
        supportMask,
        projectedSourceDepths,
        sourceSectorIds,
        options,
        inspectDepthResidualReestimationNeed(
            referenceDepth, supportMask, options));
}

DepthResidualReestimationTarget buildDepthResidualReestimationTarget(
    const cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const std::vector<int> &sourceSectorIds,
    const DepthResidualReestimationOptions &options,
    DepthResidualReestimationPreflight preflight)
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

    target.supportPixelCount = preflight.supportPixelCount;
    target.requestedResidualPixelCount =
        preflight.requestedResidualPixelCount;
    target.requestedResidualRatio = preflight.requestedResidualRatio;
    target.skippedReason = preflight.skippedReason;
    if (!preflight.shouldProjectSources)
    {
        return target;
    }
    if (preflight.normalizedSupport.empty() ||
        preflight.normalizedSupport.type() != CV_8UC1 ||
        preflight.normalizedSupport.size() != referenceDepth.size() ||
        preflight.residualMask.empty() ||
        preflight.residualMask.type() != CV_8UC1 ||
        preflight.residualMask.size() != referenceDepth.size())
    {
        target.skippedReason = QStringLiteral("invalid_preflight");
        return target;
    }
    cv::Mat normalized_support = std::move(preflight.normalizedSupport);
    target.residualMask = std::move(preflight.residualMask);

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
    const DepthResidualReestimationOptions &options,
    const cv::Mat *native_reliability_class_map,
    const std::vector<ProjectedDepthEvidence> *projected_source_evidence,
    const DepthResidualReestimationEvidenceOutputs *evidence_outputs)
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
    DepthGeometryHypothesisRerankMaps *rerank_maps =
        evidence_outputs ? evidence_outputs->rerankMaps : nullptr;
    if (rerank_maps && !rerank_maps->compatible(depth.size()))
    {
        rerank_maps->initialize(depth.size());
    }
    const auto compatible_output = [&depth](const cv::Mat *output, int type)
    {
        return output && output->type() == type && output->size() == depth.size();
    };
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
            const bool native_valid = std::isfinite(depth_row[column]) &&
                depth_row[column] > 0.0f;
            if (target_row[column] == 0 ||
                (native_valid && !options.allowValidDepthReplacement))
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
            float accepted_confidence = std::max(
                minimum_merged_confidence,
                std::clamp(options.recoveredConfidence, 0.0f, 1.0f));
            if (native_valid)
            {
                ++stats.replacementCandidatePixelCount;
                const bool has_reliability = native_reliability_class_map &&
                    native_reliability_class_map->type() == CV_8UC1 &&
                    native_reliability_class_map->size() == depth.size();
                const auto reliability = has_reliability
                    ? static_cast<DepthLayerReliabilityClass>(
                          native_reliability_class_map->at<std::uint8_t>(
                              row, column))
                    : DepthLayerReliabilityClass::Unobservable;
                if (reliability != DepthLayerReliabilityClass::AmbiguousLowTexture &&
                    reliability != DepthLayerReliabilityClass::RejectedLayer)
                {
                    ++stats.rejectedReplacementReliabilityPixelCount;
                    continue;
                }
                if (!projected_source_evidence ||
                    projected_source_evidence->size() != projectedSourceDepths.size())
                {
                    ++stats.rejectedReplacementCostPixelCount;
                    continue;
                }
                DepthGeometryHypothesisRerankOptions rerank_options;
                rerank_options.minimumLayerSwitchCostAdvantage =
                    std::max(0.0f, options.minimumReplacementCostAdvantage);
                rerank_options.ambiguousMaximumRelativeCorrection =
                    std::max(0.0f, options.ambiguousMaximumRelativeCorrection);
                const DepthGeometryHypothesisDecision native_score =
                    scoreMeasuredDepthHypothesis(
                        depth_row[column],
                        row,
                        column,
                        *projected_source_evidence,
                        rerank_options);
                const DepthGeometryHypothesisDecision candidate_score =
                    scoreMeasuredDepthHypothesis(
                        candidate_depth,
                        row,
                        column,
                        *projected_source_evidence,
                        rerank_options);
                const float cost_advantage =
                    native_score.candidateCost - candidate_score.candidateCost;
                const float relative_correction =
                    std::fabs(candidate_depth - depth_row[column]) /
                    std::max(depth_row[column], 1.0e-6f);
                if (rerank_maps)
                {
                    rerank_maps->nativeCost.at<float>(row, column) =
                        native_score.candidateCost;
                    rerank_maps->candidateCost.at<float>(row, column) =
                        candidate_score.candidateCost;
                    rerank_maps->costAdvantage.at<float>(row, column) =
                        cost_advantage;
                    rerank_maps->effectiveSourceWeight.at<float>(row, column) =
                        candidate_score.effectiveSourceWeight;
                    rerank_maps->relativeCorrection.at<float>(row, column) =
                        relative_correction;
                    rerank_maps->weakestSourceConfidence.at<float>(row, column) =
                        candidate_score.weakestSourceConfidence;
                    rerank_maps->supportingSourceCount.at<std::uint8_t>(
                        row, column) = static_cast<std::uint8_t>(std::clamp(
                            candidate_score.supportingSourceCount, 0, 255));
                    rerank_maps->baselineSectorCount.at<std::uint8_t>(
                        row, column) = static_cast<std::uint8_t>(std::clamp(
                            candidate_score.baselineSectorCount, 0, 255));
                    rerank_maps->decisionAction.at<std::uint8_t>(row, column) =
                        static_cast<std::uint8_t>(
                            DepthGeometryHypothesisAction::None);
                }
                if (!candidate_score.validEvidence ||
                    candidate_score.rejectedInsufficientSources ||
                    candidate_score.rejectedInsufficientBaseline ||
                    candidate_score.rejectedInsufficientWeight ||
                    cost_advantage < options.minimumReplacementCostAdvantage)
                {
                    ++stats.rejectedReplacementCostPixelCount;
                    continue;
                }
                if (reliability == DepthLayerReliabilityClass::AmbiguousLowTexture &&
                    relative_correction > options.ambiguousMaximumRelativeCorrection)
                {
                    ++stats.rejectedReplacementCorrectionPixelCount;
                    continue;
                }
                const float evidence_confidence = std::clamp(
                    0.35f + 0.35f * std::min(1.0f, cost_advantage / 0.20f) +
                        0.30f * candidate_score.weakestSourceConfidence,
                    0.0f,
                    0.90f);
                accepted_confidence = std::sqrt(
                    std::clamp(minimum_merged_confidence, 0.0f, 1.0f) *
                    evidence_confidence);
                if (rerank_maps)
                {
                    const bool refinement =
                        reliability ==
                            DepthLayerReliabilityClass::AmbiguousLowTexture ||
                        relative_correction <= 0.025f;
                    rerank_maps->decisionAction.at<std::uint8_t>(row, column) =
                        static_cast<std::uint8_t>(
                            refinement
                                ? DepthGeometryHypothesisAction::Refine
                                : DepthGeometryHypothesisAction::SwitchLayer);
                }
                if (evidence_outputs)
                {
                    if (compatible_output(
                            evidence_outputs->geometrySourceMask, CV_16UC1))
                    {
                        evidence_outputs->geometrySourceMask->at<std::uint16_t>(
                            row, column) = candidate_score.supportingSourceMask;
                    }
                    if (compatible_output(
                            evidence_outputs->sourceInverseDepthSum, CV_32FC1))
                    {
                        evidence_outputs->sourceInverseDepthSum->at<float>(
                            row, column) =
                            candidate_score.supportingInverseDepthSum;
                    }
                    if (compatible_output(
                            evidence_outputs->sourceInverseDepthSquaredSum,
                            CV_32FC1))
                    {
                        evidence_outputs->sourceInverseDepthSquaredSum->at<float>(
                            row, column) =
                            candidate_score.supportingInverseDepthSquaredSum;
                    }
                    if (compatible_output(
                            evidence_outputs->confirmedSourceCount, CV_16UC1))
                    {
                        evidence_outputs->confirmedSourceCount->at<std::uint16_t>(
                            row, column) = static_cast<std::uint16_t>(std::clamp(
                                candidate_score.supportingSourceCount,
                                0,
                                65535));
                    }
                }
                ++stats.replacedPixelCount;
            }
            depth_row[column] = candidate_depth;
            confidence_row[column] = accepted_confidence;
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
        {QStringLiteral("replacement_candidate_pixel_count"),
         stats.replacementCandidatePixelCount},
        {QStringLiteral("replaced_pixel_count"), stats.replacedPixelCount},
        {QStringLiteral("rejected_replacement_reliability_pixel_count"),
         stats.rejectedReplacementReliabilityPixelCount},
        {QStringLiteral("rejected_replacement_cost_pixel_count"),
         stats.rejectedReplacementCostPixelCount},
        {QStringLiteral("rejected_replacement_correction_pixel_count"),
         stats.rejectedReplacementCorrectionPixelCount},
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
