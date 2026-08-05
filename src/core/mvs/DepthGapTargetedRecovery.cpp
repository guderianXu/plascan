#include "DepthGapTargetedRecovery.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xjw::mvs
{
namespace
{

bool compatibleDepthAndMask(const cv::Mat &depth, const cv::Mat &mask)
{
    return !depth.empty() && depth.type() == CV_32FC1 &&
        !mask.empty() && mask.type() == CV_8UC1 && depth.size() == mask.size();
}

struct DirectionalAnchorMaps
{
    cv::Mat depth;
    cv::Mat distance;
};

std::array<DirectionalAnchorMaps, 4> buildDirectionalAnchorMaps(
    const cv::Mat &depth,
    const cv::Mat &support,
    int maximum_distance)
{
    std::array<DirectionalAnchorMaps, 4> result;
    for (DirectionalAnchorMaps &maps : result)
    {
        maps.depth = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
        maps.distance = cv::Mat(depth.size(), CV_32SC1, cv::Scalar(0));
    }
    const auto valid_depth = [](float value)
    {
        return std::isfinite(value) && value > 0.0f;
    };

    for (int row = 0; row < depth.rows; ++row)
    {
        float anchor_depth = 0.0f;
        int anchor_column = -1;
        for (int column = 0; column < depth.cols; ++column)
        {
            if (support.at<std::uint8_t>(row, column) == 0)
            {
                anchor_depth = 0.0f;
                anchor_column = -1;
                continue;
            }
            const float value = depth.at<float>(row, column);
            if (valid_depth(value))
            {
                anchor_depth = value;
                anchor_column = column;
                continue;
            }
            const int distance = column - anchor_column;
            if (anchor_column >= 0 && distance <= maximum_distance)
            {
                result[0].depth.at<float>(row, column) = anchor_depth;
                result[0].distance.at<int>(row, column) = distance;
            }
        }
        anchor_depth = 0.0f;
        anchor_column = -1;
        for (int column = depth.cols - 1; column >= 0; --column)
        {
            if (support.at<std::uint8_t>(row, column) == 0)
            {
                anchor_depth = 0.0f;
                anchor_column = -1;
                continue;
            }
            const float value = depth.at<float>(row, column);
            if (valid_depth(value))
            {
                anchor_depth = value;
                anchor_column = column;
                continue;
            }
            const int distance = anchor_column - column;
            if (anchor_column >= 0 && distance <= maximum_distance)
            {
                result[1].depth.at<float>(row, column) = anchor_depth;
                result[1].distance.at<int>(row, column) = distance;
            }
        }
    }

    for (int column = 0; column < depth.cols; ++column)
    {
        float anchor_depth = 0.0f;
        int anchor_row = -1;
        for (int row = 0; row < depth.rows; ++row)
        {
            if (support.at<std::uint8_t>(row, column) == 0)
            {
                anchor_depth = 0.0f;
                anchor_row = -1;
                continue;
            }
            const float value = depth.at<float>(row, column);
            if (valid_depth(value))
            {
                anchor_depth = value;
                anchor_row = row;
                continue;
            }
            const int distance = row - anchor_row;
            if (anchor_row >= 0 && distance <= maximum_distance)
            {
                result[2].depth.at<float>(row, column) = anchor_depth;
                result[2].distance.at<int>(row, column) = distance;
            }
        }
        anchor_depth = 0.0f;
        anchor_row = -1;
        for (int row = depth.rows - 1; row >= 0; --row)
        {
            if (support.at<std::uint8_t>(row, column) == 0)
            {
                anchor_depth = 0.0f;
                anchor_row = -1;
                continue;
            }
            const float value = depth.at<float>(row, column);
            if (valid_depth(value))
            {
                anchor_depth = value;
                anchor_row = row;
                continue;
            }
            const int distance = anchor_row - row;
            if (anchor_row >= 0 && distance <= maximum_distance)
            {
                result[3].depth.at<float>(row, column) = anchor_depth;
                result[3].distance.at<int>(row, column) = distance;
            }
        }
    }
    return result;
}

void applySurfaceAwarePrior(
    const cv::Mat &depth,
    const cv::Mat &support,
    const DepthGapTargetedRecoveryOptions &options,
    DepthGapTarget *target)
{
    if (!target || !options.enableSurfaceAwarePrior ||
        target->gapMask.empty())
    {
        return;
    }
    const int maximum_distance = std::max(
        1, options.maximumPriorDistancePixels);
    const auto anchors = buildDirectionalAnchorMaps(
        depth, support, maximum_distance);
    target->surfacePriorMask = cv::Mat(
        depth.size(), CV_8UC1, cv::Scalar(0));
    target->priorSupportCount = cv::Mat(
        depth.size(), CV_8UC1, cv::Scalar(0));
    target->priorRelativeResidual = cv::Mat(
        depth.size(), CV_32FC1, cv::Scalar(0.0f));

    const int minimum_anchor_count = std::clamp(
        options.minimumSurfacePriorAnchorCount, 3, 4);
    const float maximum_anchor_spread = std::max(
        0.0f, options.maximumSurfaceAnchorInverseDepthRelativeSpread);
    const float maximum_fit_residual = std::max(
        0.0f, options.maximumSurfacePriorFitRelativeResidual);
    const float envelope_expansion = std::clamp(
        options.maximumSurfacePriorEnvelopeExpansion, 0.0f, 0.5f);

    for (int row = 0; row < depth.rows; ++row)
    {
        const std::uint8_t *gap_row = target->gapMask.ptr<std::uint8_t>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (gap_row[column] == 0)
            {
                continue;
            }
            std::array<float, 4> inverse_depths{};
            std::array<int, 4> distances{};
            int anchor_count = 0;
            float inverse_depth_sum = 0.0f;
            float minimum_inverse_depth = std::numeric_limits<float>::max();
            float maximum_inverse_depth = 0.0f;
            for (int direction = 0; direction < 4; ++direction)
            {
                const float anchor_depth =
                    anchors[static_cast<std::size_t>(direction)]
                        .depth.at<float>(row, column);
                const int distance =
                    anchors[static_cast<std::size_t>(direction)]
                        .distance.at<int>(row, column);
                if (!std::isfinite(anchor_depth) || anchor_depth <= 0.0f ||
                    distance <= 0)
                {
                    continue;
                }
                const float inverse_depth = 1.0f / anchor_depth;
                inverse_depths[static_cast<std::size_t>(direction)] =
                    inverse_depth;
                distances[static_cast<std::size_t>(direction)] = distance;
                inverse_depth_sum += inverse_depth;
                minimum_inverse_depth = std::min(
                    minimum_inverse_depth, inverse_depth);
                maximum_inverse_depth = std::max(
                    maximum_inverse_depth, inverse_depth);
                ++anchor_count;
            }
            target->priorSupportCount.at<std::uint8_t>(row, column) =
                static_cast<std::uint8_t>(anchor_count);
            const bool has_opposing_pair =
                (distances[0] > 0 && distances[1] > 0) ||
                (distances[2] > 0 && distances[3] > 0);
            if (anchor_count < minimum_anchor_count || !has_opposing_pair)
            {
                ++target->surfacePriorInsufficientAnchorPixelCount;
                continue;
            }

            const float inverse_depth_mean = inverse_depth_sum /
                static_cast<float>(anchor_count);
            float inverse_depth_variance = 0.0f;
            for (int direction = 0; direction < 4; ++direction)
            {
                if (distances[static_cast<std::size_t>(direction)] <= 0)
                {
                    continue;
                }
                const float difference =
                    inverse_depths[static_cast<std::size_t>(direction)] -
                    inverse_depth_mean;
                inverse_depth_variance += difference * difference;
            }
            const float anchor_spread = std::sqrt(
                inverse_depth_variance / static_cast<float>(anchor_count)) /
                std::max(inverse_depth_mean, 1.0e-6f);
            if (!std::isfinite(anchor_spread) ||
                anchor_spread > maximum_anchor_spread)
            {
                ++target->surfacePriorAnchorSpreadRejectedPixelCount;
                continue;
            }

            cv::Matx33f normal = cv::Matx33f::zeros();
            cv::Vec3f right_hand_side(0.0f, 0.0f, 0.0f);
            float weight_sum = 0.0f;
            for (int direction = 0; direction < 4; ++direction)
            {
                const int distance =
                    distances[static_cast<std::size_t>(direction)];
                if (distance <= 0)
                {
                    continue;
                }
                const float delta_x = direction == 0
                    ? -static_cast<float>(distance)
                    : (direction == 1 ? static_cast<float>(distance) : 0.0f);
                const float delta_y = direction == 2
                    ? -static_cast<float>(distance)
                    : (direction == 3 ? static_cast<float>(distance) : 0.0f);
                const cv::Vec3f basis(delta_x, delta_y, 1.0f);
                const float weight = 1.0f /
                    std::sqrt(static_cast<float>(distance));
                for (int first = 0; first < 3; ++first)
                {
                    right_hand_side[first] += weight * basis[first] *
                        inverse_depths[static_cast<std::size_t>(direction)];
                    for (int second = 0; second < 3; ++second)
                    {
                        normal(first, second) +=
                            weight * basis[first] * basis[second];
                    }
                }
                weight_sum += weight;
            }
            const double determinant = cv::determinant(normal);
            if (!std::isfinite(determinant) || std::fabs(determinant) < 1.0e-6)
            {
                ++target->surfacePriorFitRejectedPixelCount;
                continue;
            }
            const cv::Vec3f coefficients =
                normal.inv(cv::DECOMP_SVD) * right_hand_side;
            const float predicted_inverse_depth = coefficients[2];
            if (!std::isfinite(predicted_inverse_depth) ||
                predicted_inverse_depth <= 0.0f ||
                predicted_inverse_depth <
                    minimum_inverse_depth * (1.0f - envelope_expansion) ||
                predicted_inverse_depth >
                    maximum_inverse_depth * (1.0f + envelope_expansion))
            {
                ++target->surfacePriorFitRejectedPixelCount;
                continue;
            }

            float weighted_squared_residual = 0.0f;
            for (int direction = 0; direction < 4; ++direction)
            {
                const int distance =
                    distances[static_cast<std::size_t>(direction)];
                if (distance <= 0)
                {
                    continue;
                }
                const float delta_x = direction == 0
                    ? -static_cast<float>(distance)
                    : (direction == 1 ? static_cast<float>(distance) : 0.0f);
                const float delta_y = direction == 2
                    ? -static_cast<float>(distance)
                    : (direction == 3 ? static_cast<float>(distance) : 0.0f);
                const float fitted = coefficients[0] * delta_x +
                    coefficients[1] * delta_y + coefficients[2];
                const float difference = fitted -
                    inverse_depths[static_cast<std::size_t>(direction)];
                const float weight = 1.0f /
                    std::sqrt(static_cast<float>(distance));
                weighted_squared_residual += weight * difference * difference;
            }
            const float relative_residual = std::sqrt(
                weighted_squared_residual / std::max(weight_sum, 1.0e-6f)) /
                predicted_inverse_depth;
            if (!std::isfinite(relative_residual) ||
                relative_residual > maximum_fit_residual)
            {
                ++target->surfacePriorResidualRejectedPixelCount;
                continue;
            }

            const float prior_depth = 1.0f / predicted_inverse_depth;
            target->hintDepth.at<float>(row, column) = prior_depth;
            target->hintRadius.at<float>(row, column) = prior_depth *
                std::max(0.0f, options.missingPriorRadiusRatio);
            target->surfacePriorMask.at<std::uint8_t>(row, column) = 255;
            target->priorRelativeResidual.at<float>(row, column) =
                relative_residual;
            ++target->surfacePriorPixelCount;
        }
    }
}

} // namespace

DepthGapTarget buildDepthGapTarget(
    const cv::Mat &depth,
    const cv::Mat &supportMask,
    const DepthGapTargetedRecoveryOptions &options)
{
    DepthGapTarget target;
    if (!compatibleDepthAndMask(depth, supportMask))
    {
        target.skippedReason = QStringLiteral("invalid_inputs");
        return target;
    }

    cv::Mat normalized_support;
    cv::compare(supportMask, 0, normalized_support, cv::CMP_GT);
    target.supportPixelCount = cv::countNonZero(normalized_support);
    if (target.supportPixelCount <= 0)
    {
        target.skippedReason = QStringLiteral("empty_support_mask");
        return target;
    }

    cv::bitwise_and(depth <= 0.0f, normalized_support, target.gapMask);
    target.requestedGapPixelCount = cv::countNonZero(target.gapMask);
    target.requestedGapRatio = static_cast<float>(target.requestedGapPixelCount) /
        static_cast<float>(target.supportPixelCount);
    if (target.requestedGapPixelCount < std::max(1, options.minimumGapPixelCount) ||
        target.requestedGapRatio < std::max(0.0f, options.minimumGapRatio))
    {
        target.skippedReason = QStringLiteral("gap_below_trigger");
        return target;
    }
    if (target.requestedGapRatio >
        std::clamp(options.maximumGapRatio, 0.0f, 1.0f))
    {
        target.skippedReason = QStringLiteral("gap_above_safety_limit");
        return target;
    }

    const cv::Mat valid_depth = (depth > 0.0f) & normalized_support;
    if (cv::countNonZero(valid_depth) <= 0)
    {
        target.skippedReason = QStringLiteral("no_depth_anchors");
        return target;
    }

    cv::Mat distance_input(depth.size(), CV_8UC1, cv::Scalar(255));
    distance_input.setTo(cv::Scalar(0), valid_depth);
    cv::Mat distance;
    cv::Mat labels;
    cv::distanceTransform(
        distance_input, distance, labels, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);
    double maximum_label_value = 0.0;
    cv::minMaxLoc(labels, nullptr, &maximum_label_value);
    std::vector<float> depth_by_label(
        static_cast<std::size_t>(std::max(0.0, maximum_label_value)) + 1U,
        0.0f);
    for (int row = 0; row < depth.rows; ++row)
    {
        const float *depth_row = depth.ptr<float>(row);
        const std::uint8_t *valid_row = valid_depth.ptr<std::uint8_t>(row);
        const int *label_row = labels.ptr<int>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (valid_row[column] == 0)
            {
                continue;
            }
            const int label = label_row[column];
            if (label > 0 &&
                label < static_cast<int>(depth_by_label.size()))
            {
                depth_by_label[static_cast<std::size_t>(label)] =
                    depth_row[column];
            }
        }
    }

    target.hintDepth = depth.clone();
    target.hintRadius = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    const int maximum_distance = std::max(1, options.maximumPriorDistancePixels);
    for (int row = 0; row < depth.rows; ++row)
    {
        const std::uint8_t *gap_row = target.gapMask.ptr<std::uint8_t>(row);
        const float *distance_row = distance.ptr<float>(row);
        const int *label_row = labels.ptr<int>(row);
        float *hint_row = target.hintDepth.ptr<float>(row);
        float *radius_row = target.hintRadius.ptr<float>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (gap_row[column] == 0)
            {
                if (hint_row[column] > 0.0f)
                {
                    radius_row[column] = hint_row[column] *
                        std::max(0.0f, options.anchoredPriorRadiusRatio);
                }
                continue;
            }
            const int label = label_row[column];
            const float prior = label > 0 &&
                    label < static_cast<int>(depth_by_label.size())
                ? depth_by_label[static_cast<std::size_t>(label)]
                : 0.0f;
            if (distance_row[column] > static_cast<float>(maximum_distance) ||
                !std::isfinite(prior) || prior <= 0.0f)
            {
                hint_row[column] = 0.0f;
                continue;
            }
            hint_row[column] = prior;
            radius_row[column] = prior *
                std::max(0.0f, options.missingPriorRadiusRatio);
            ++target.priorCoveredGapPixelCount;
        }
    }

    target.nearestHintDepth = target.hintDepth.clone();
    applySurfaceAwarePrior(
        depth, normalized_support, options, &target);

    cv::Mat prior_covered_gap;
    cv::bitwise_and(target.gapMask, target.hintDepth > 0.0f,
                    prior_covered_gap);
    target.gapMask = prior_covered_gap;
    target.priorCoveredGapPixelCount = cv::countNonZero(target.gapMask);
    if (target.priorCoveredGapPixelCount <= 0)
    {
        target.skippedReason = QStringLiteral("gap_has_no_bounded_prior");
        return target;
    }

    const int dilation = std::max(0, options.estimationMaskDilationPixels);
    if (dilation > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(dilation * 2 + 1, dilation * 2 + 1));
        cv::dilate(target.gapMask, target.estimationMask, kernel);
    }
    else
    {
        target.estimationMask = target.gapMask.clone();
    }
    cv::bitwise_and(target.estimationMask,
                    normalized_support,
                    target.estimationMask);
    target.hintDepth.setTo(0.0f, target.estimationMask == 0);
    target.nearestHintDepth.setTo(0.0f, target.estimationMask == 0);
    target.hintRadius.setTo(0.0f, target.estimationMask == 0);
    target.valid = true;
    return target;
}

DepthGapTargetedRecoveryStats mergeTargetedDepthGapCandidates(
    cv::Mat &depth,
    cv::Mat &confidence,
    const cv::Mat &candidateDepth,
    const cv::Mat &candidateConfidence,
    const DepthGapTarget &target,
    cv::Mat *recoveredMask,
    const DepthGapTargetedRecoveryOptions &options)
{
    return mergeMultiHypothesisTargetedDepthGapCandidates(
        depth,
        confidence,
        std::vector<cv::Mat>{candidateDepth},
        std::vector<cv::Mat>{candidateConfidence},
        target,
        recoveredMask,
        options);
}

DepthGapTargetedRecoveryStats mergeMultiHypothesisTargetedDepthGapCandidates(
    cv::Mat &depth,
    cv::Mat &confidence,
    const std::vector<cv::Mat> &candidateDepths,
    const std::vector<cv::Mat> &candidateConfidences,
    const DepthGapTarget &target,
    cv::Mat *recoveredMask,
    const DepthGapTargetedRecoveryOptions &options)
{
    DepthGapTargetedRecoveryStats stats;
    stats.supportPixelCount = target.supportPixelCount;
    stats.requestedGapPixelCount = target.requestedGapPixelCount;
    stats.priorCoveredGapPixelCount = target.priorCoveredGapPixelCount;
    stats.surfacePriorPixelCount = target.surfacePriorPixelCount;
    stats.surfacePriorInsufficientAnchorPixelCount =
        target.surfacePriorInsufficientAnchorPixelCount;
    stats.surfacePriorAnchorSpreadRejectedPixelCount =
        target.surfacePriorAnchorSpreadRejectedPixelCount;
    stats.surfacePriorFitRejectedPixelCount =
        target.surfacePriorFitRejectedPixelCount;
    stats.surfacePriorResidualRejectedPixelCount =
        target.surfacePriorResidualRejectedPixelCount;
    stats.skippedReason = target.skippedReason;
    stats.hypothesisCount = static_cast<int>(candidateDepths.size());
    if (!target.valid || depth.empty() || depth.type() != CV_32FC1 ||
        candidateDepths.empty() ||
        candidateDepths.size() != candidateConfidences.size() ||
        depth.size() != target.gapMask.size() ||
        !std::all_of(candidateDepths.begin(), candidateDepths.end(),
                     [&depth](const cv::Mat &candidate)
                     {
                         return candidate.type() == CV_32FC1 &&
                             candidate.size() == depth.size();
                     }) ||
        !std::all_of(candidateConfidences.begin(), candidateConfidences.end(),
                     [&depth](const cv::Mat &candidate)
                     {
                         return candidate.type() == CV_32FC1 &&
                             candidate.size() == depth.size();
                     }))
    {
        if (stats.skippedReason.isEmpty())
        {
            stats.skippedReason = QStringLiteral("invalid_candidates");
        }
        return stats;
    }

    stats.attempted = true;
    if (confidence.empty() || confidence.type() != CV_32FC1 ||
        confidence.size() != depth.size())
    {
        confidence = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    }
    cv::Mat recovered(depth.size(), CV_8UC1, cv::Scalar(0));
    const float minimum_confidence = std::clamp(
        options.minimumCandidateConfidence, 0.0f, 1.0f);
    const float maximum_relative_difference = std::max(
        0.0f, options.maximumCandidatePriorRelativeDifference);
    const float maximum_consensus_relative_difference = std::max(
        maximum_relative_difference,
        options.maximumConsensusPriorRelativeDifference);
    const float maximum_consensus_spread = std::max(
        0.0f, options.maximumConsensusInverseDepthRelativeSpread);
    const int minimum_consensus_count = std::max(
        2, options.minimumConsensusHypothesisCount);
    struct EligibleHypothesis
    {
        float inverseDepth = 0.0f;
        float confidence = 0.0f;
    };
    std::vector<EligibleHypothesis> eligible;
    eligible.reserve(candidateDepths.size());
    for (int row = 0; row < depth.rows; ++row)
    {
        float *depth_row = depth.ptr<float>(row);
        float *confidence_row = confidence.ptr<float>(row);
        const float *prior_row = target.hintDepth.ptr<float>(row);
        const float *nearest_prior_row = target.nearestHintDepth.empty()
            ? prior_row
            : target.nearestHintDepth.ptr<float>(row);
        const std::uint8_t *gap_row = target.gapMask.ptr<std::uint8_t>(row);
        std::uint8_t *recovered_row = recovered.ptr<std::uint8_t>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (gap_row[column] == 0 || depth_row[column] > 0.0f)
            {
                continue;
            }
            eligible.clear();
            bool has_candidate_depth = false;
            for (std::size_t hypothesis_index = 0;
                 hypothesis_index < candidateDepths.size();
                 ++hypothesis_index)
            {
                const float candidate_depth =
                    candidateDepths[hypothesis_index].at<float>(row, column);
                if (!std::isfinite(candidate_depth) || candidate_depth <= 0.0f)
                {
                    continue;
                }
                has_candidate_depth = true;
                const float candidate_confidence =
                    candidateConfidences[hypothesis_index].at<float>(row, column);
                if (!std::isfinite(candidate_confidence) ||
                    candidate_confidence < minimum_confidence)
                {
                    continue;
                }
                eligible.push_back(EligibleHypothesis{
                    1.0f / candidate_depth,
                    candidate_confidence});
            }
            if (!has_candidate_depth)
            {
                continue;
            }
            ++stats.candidatePixelCount;
            if (eligible.empty())
            {
                ++stats.rejectedConfidencePixelCount;
                continue;
            }

            const bool consensus_requested =
                static_cast<int>(candidateDepths.size()) >= minimum_consensus_count;
            const bool has_consensus =
                static_cast<int>(eligible.size()) >= minimum_consensus_count;
            if (consensus_requested && !has_consensus)
            {
                ++stats.rejectedInsufficientHypothesisPixelCount;
            }

            float inverse_depth_weight_sum = 0.0f;
            float confidence_sum = 0.0f;
            float merged_confidence = 1.0f;
            for (const EligibleHypothesis &hypothesis : eligible)
            {
                inverse_depth_weight_sum +=
                    hypothesis.inverseDepth * hypothesis.confidence;
                confidence_sum += hypothesis.confidence;
                merged_confidence = std::min(
                    merged_confidence, hypothesis.confidence);
            }
            const float inverse_depth_mean =
                inverse_depth_weight_sum / std::max(confidence_sum, 1.0e-6f);
            if (!std::isfinite(inverse_depth_mean) || inverse_depth_mean <= 0.0f)
            {
                continue;
            }
            if (has_consensus)
            {
                float variance_sum = 0.0f;
                for (const EligibleHypothesis &hypothesis : eligible)
                {
                    const float difference =
                        hypothesis.inverseDepth - inverse_depth_mean;
                    variance_sum += hypothesis.confidence * difference * difference;
                }
                const float relative_spread = std::sqrt(
                    variance_sum / std::max(confidence_sum, 1.0e-6f)) /
                    inverse_depth_mean;
                if (!std::isfinite(relative_spread) ||
                    relative_spread > maximum_consensus_spread)
                {
                    ++stats.rejectedHypothesisSpreadPixelCount;
                    continue;
                }
                ++stats.consensusCandidatePixelCount;
            }

            const float candidate_depth = 1.0f / inverse_depth_mean;
            const float prior = has_consensus
                ? prior_row[column]
                : nearest_prior_row[column];
            const float relative_difference = std::fabs(candidate_depth - prior) /
                std::max(prior, 1.0e-6f);
            const float allowed_relative_difference = has_consensus
                ? maximum_consensus_relative_difference
                : maximum_relative_difference;
            if (!std::isfinite(prior) || prior <= 0.0f ||
                relative_difference > allowed_relative_difference)
            {
                ++stats.rejectedPriorPixelCount;
                continue;
            }
            depth_row[column] = candidate_depth;
            confidence_row[column] = merged_confidence;
            recovered_row[column] = 255;
            if (has_consensus && !target.surfacePriorMask.empty() &&
                target.surfacePriorMask.at<std::uint8_t>(row, column) != 0)
            {
                ++stats.surfacePriorAcceptedPixelCount;
            }
            ++stats.recoveredPixelCount;
        }
    }
    if (stats.priorCoveredGapPixelCount > 0)
    {
        stats.recoveryRatio = static_cast<float>(stats.recoveredPixelCount) /
            static_cast<float>(stats.priorCoveredGapPixelCount);
    }
    if (recoveredMask)
    {
        *recoveredMask = std::move(recovered);
    }
    return stats;
}

QJsonObject depthGapTargetedRecoveryStatsToJson(
    const DepthGapTargetedRecoveryStats &stats)
{
    return QJsonObject{
        {QStringLiteral("attempted"), stats.attempted},
        {QStringLiteral("support_pixel_count"), stats.supportPixelCount},
        {QStringLiteral("requested_gap_pixel_count"),
         stats.requestedGapPixelCount},
        {QStringLiteral("prior_covered_gap_pixel_count"),
         stats.priorCoveredGapPixelCount},
        {QStringLiteral("candidate_pixel_count"), stats.candidatePixelCount},
        {QStringLiteral("recovered_pixel_count"), stats.recoveredPixelCount},
        {QStringLiteral("rejected_confidence_pixel_count"),
         stats.rejectedConfidencePixelCount},
        {QStringLiteral("rejected_prior_pixel_count"),
         stats.rejectedPriorPixelCount},
        {QStringLiteral("source_count"), stats.sourceCount},
        {QStringLiteral("attempted_hypothesis_count"),
         stats.attemptedHypothesisCount},
        {QStringLiteral("hypothesis_count"), stats.hypothesisCount},
        {QStringLiteral("failed_hypothesis_count"),
         stats.failedHypothesisCount},
        {QStringLiteral("consensus_candidate_pixel_count"),
         stats.consensusCandidatePixelCount},
        {QStringLiteral("rejected_insufficient_hypothesis_pixel_count"),
         stats.rejectedInsufficientHypothesisPixelCount},
        {QStringLiteral("rejected_hypothesis_spread_pixel_count"),
         stats.rejectedHypothesisSpreadPixelCount},
        {QStringLiteral("surface_prior_pixel_count"),
         stats.surfacePriorPixelCount},
        {QStringLiteral("surface_prior_accepted_pixel_count"),
         stats.surfacePriorAcceptedPixelCount},
        {QStringLiteral("surface_prior_insufficient_anchor_pixel_count"),
         stats.surfacePriorInsufficientAnchorPixelCount},
        {QStringLiteral("surface_prior_anchor_spread_rejected_pixel_count"),
         stats.surfacePriorAnchorSpreadRejectedPixelCount},
        {QStringLiteral("surface_prior_fit_rejected_pixel_count"),
         stats.surfacePriorFitRejectedPixelCount},
        {QStringLiteral("surface_prior_residual_rejected_pixel_count"),
         stats.surfacePriorResidualRejectedPixelCount},
        {QStringLiteral("recovery_ratio"), stats.recoveryRatio},
        {QStringLiteral("skipped_reason"), stats.skippedReason},
        {QStringLiteral("schema_version"), 3}};
}

} // namespace xjw::mvs
