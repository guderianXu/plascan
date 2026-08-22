#include "DepthCrossViewHoleRepair.h"

#include "concurrency/SafeWorkerGroup.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <queue>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xjw::mvs
{
namespace
{

struct SourceDepthCandidate
{
    float depth = 0.0f;
    int sourceOrdinal = -1;
};

template <typename Fn>
void parallelForRows(int row_count,
                     int worker_count,
                     const std::atomic<bool> *cancelled,
                     Fn &&fn)
{
    if (row_count <= 0 ||
        (cancelled && cancelled->load(std::memory_order_relaxed)))
    {
        return;
    }
    const int workers = std::clamp(std::max(1, worker_count), 1, row_count);
    if (workers == 1)
    {
        for (int row = 0; row < row_count; ++row)
        {
            if (cancelled && cancelled->load(std::memory_order_relaxed))
            {
                break;
            }
            fn(row);
        }
        return;
    }

#if defined(_OPENMP)
    if (!omp_in_parallel())
    {
        xjw::common::concurrency::WorkerFailureState failure_state;
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
        for (int row = 0; row < row_count; ++row)
        {
            if (failure_state.stopRequested() ||
                (cancelled && cancelled->load(std::memory_order_relaxed)))
            {
                continue;
            }
            try
            {
                fn(row);
            }
            catch (...)
            {
                // captureCurrentException is noexcept, so no exception crosses
                // the OpenMP runtime boundary.
                failure_state.captureCurrentException();
            }
        }
        failure_state.rethrowIfFailed();
        return;
    }
#endif

    std::atomic<int> next_row{0};
    xjw::common::concurrency::runWorkerGroup(
        static_cast<std::size_t>(workers),
        [&](std::stop_token stop_token)
    {
        for (;;)
        {
            if (stop_token.stop_requested() ||
                (cancelled && cancelled->load(std::memory_order_relaxed)))
            {
                break;
            }
            const int row = next_row.fetch_add(1, std::memory_order_relaxed);
            if (row >= row_count)
            {
                break;
            }
            fn(row);
        }
    });
}

void addStats(DominantDepthLayerSelectionStats &target,
              const DominantDepthLayerSelectionStats &source)
{
    target.consideredPixelCount += source.consideredPixelCount;
    target.stableLayerPixelCount += source.stableLayerPixelCount;
    target.refinedNativePixelCount += source.refinedNativePixelCount;
    target.switchedNativePixelCount += source.switchedNativePixelCount;
    target.transferredMissingPixelCount += source.transferredMissingPixelCount;
    target.ambiguousNativePixelCount += source.ambiguousNativePixelCount;
    target.unresolvedMissingPixelCount += source.unresolvedMissingPixelCount;
    target.reliabilityGuidedCandidatePixelCount +=
        source.reliabilityGuidedCandidatePixelCount;
    target.reliabilityGuidedStablePixelCount +=
        source.reliabilityGuidedStablePixelCount;
    target.reliabilityGuidedRefinedPixelCount +=
        source.reliabilityGuidedRefinedPixelCount;
    target.reliabilityGuidedSwitchedPixelCount +=
        source.reliabilityGuidedSwitchedPixelCount;
    target.reliabilityGuidedInsufficientSourcePixelCount +=
        source.reliabilityGuidedInsufficientSourcePixelCount;
    target.geometryRerankEvaluatedPixelCount +=
        source.geometryRerankEvaluatedPixelCount;
    target.geometryRerankValidEvidencePixelCount +=
        source.geometryRerankValidEvidencePixelCount;
    target.geometryRerankRefinedPixelCount +=
        source.geometryRerankRefinedPixelCount;
    target.geometryRerankSwitchedPixelCount +=
        source.geometryRerankSwitchedPixelCount;
    target.geometryRerankRejectedSourceCount +=
        source.geometryRerankRejectedSourceCount;
    target.geometryRerankRejectedBaselineCount +=
        source.geometryRerankRejectedBaselineCount;
    target.geometryRerankRejectedWeightCount +=
        source.geometryRerankRejectedWeightCount;
    target.geometryRerankRejectedCostCount +=
        source.geometryRerankRejectedCostCount;
    target.geometryRerankNativeCostSum += source.geometryRerankNativeCostSum;
    target.geometryRerankCandidateCostSum +=
        source.geometryRerankCandidateCostSum;
    target.geometryRerankCostAdvantageSum +=
        source.geometryRerankCostAdvantageSum;
    target.geometryRerankCorrectionSum +=
        source.geometryRerankCorrectionSum;
    target.geometryRerankWeakestConfidenceSum +=
        source.geometryRerankWeakestConfidenceSum;
}

void addStats(CrossViewHoleRepairStats &target,
              const CrossViewHoleRepairStats &source)
{
    target.projectedCandidateCount += source.projectedCandidateCount;
    target.consideredHolePixelCount += source.consideredHolePixelCount;
    target.rejectedInsufficientSourceCount += source.rejectedInsufficientSourceCount;
    target.rejectedDepthSpreadCount += source.rejectedDepthSpreadCount;
    target.rejectedLocalDepthCount += source.rejectedLocalDepthCount;
    target.repairedPixelCount += source.repairedPixelCount;
    target.twoSourceCandidatePixelCount += source.twoSourceCandidatePixelCount;
}

bool validDepth(float depth)
{
    return std::isfinite(depth) && depth > 0.0f;
}

float relativeDifference(float first, float second)
{
    return std::fabs(first - second) /
        std::max(1.0e-6f, std::min(first, second));
}

int bitCount(std::uint16_t mask)
{
    int count = 0;
    while (mask != 0)
    {
        mask = static_cast<std::uint16_t>(mask & (mask - 1));
        ++count;
    }
    return count;
}

bool surfaceNormalAt(const cv::Mat &depth,
                     int row,
                     int column,
                     const FramePinholeCamera &camera,
                     cv::Vec3f *normal)
{
    if (!normal || depth.type() != CV_32FC1 || !camera.isValid() ||
        row <= 0 || row + 1 >= depth.rows ||
        column <= 0 || column + 1 >= depth.cols)
    {
        return false;
    }
    const float left_depth = depth.at<float>(row, column - 1);
    const float right_depth = depth.at<float>(row, column + 1);
    const float upper_depth = depth.at<float>(row - 1, column);
    const float lower_depth = depth.at<float>(row + 1, column);
    if (!validDepth(left_depth) || !validDepth(right_depth) ||
        !validDepth(upper_depth) || !validDepth(lower_depth))
    {
        return false;
    }
    auto unproject = [&](double x, double y, float value, cv::Vec3f *point)
    {
        const double pixel[2] = {x, y};
        double world[3] = {};
        if (!camera.unprojectPixel(pixel, value, world))
        {
            return false;
        }
        *point = cv::Vec3f(static_cast<float>(world[0]),
                          static_cast<float>(world[1]),
                          static_cast<float>(world[2]));
        return true;
    };
    cv::Vec3f left;
    cv::Vec3f right;
    cv::Vec3f upper;
    cv::Vec3f lower;
    if (!unproject(column - 1, row, left_depth, &left) ||
        !unproject(column + 1, row, right_depth, &right) ||
        !unproject(column, row - 1, upper_depth, &upper) ||
        !unproject(column, row + 1, lower_depth, &lower))
    {
        return false;
    }
    cv::Vec3f value = (right - left).cross(lower - upper);
    const float length = std::sqrt(value.dot(value));
    if (!std::isfinite(length) || length <= 1.0e-8f)
    {
        return false;
    }
    *normal = value / length;
    return true;
}

bool normalsAgree(const cv::Mat &surface,
                  int first_row,
                  int first_column,
                  int second_row,
                  int second_column,
                  const FramePinholeCamera &camera,
                  float maximum_angle_degrees)
{
    cv::Vec3f first;
    cv::Vec3f second;
    if (!surfaceNormalAt(surface, first_row, first_column, camera, &first) ||
        !surfaceNormalAt(surface, second_row, second_column, camera, &second))
    {
        return false;
    }
    const float cosine = std::clamp(std::fabs(first.dot(second)), 0.0f, 1.0f);
    const float angle = std::acos(cosine) * 180.0f /
        static_cast<float>(CV_PI);
    return angle <= maximum_angle_degrees;
}

cv::Mat guideGradient(const cv::Mat *guide_gray, const cv::Size &size)
{
    if (!guide_gray || guide_gray->empty())
    {
        return {};
    }
    cv::Mat gray;
    if (guide_gray->channels() == 1)
    {
        gray = *guide_gray;
    }
    else
    {
        cv::cvtColor(*guide_gray, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.size() != size)
    {
        cv::resize(gray, gray, size, 0.0, 0.0, cv::INTER_AREA);
    }
    cv::Mat gradient_x;
    cv::Mat gradient_y;
    cv::Sobel(gray, gradient_x, CV_32FC1, 1, 0, 3);
    cv::Sobel(gray, gradient_y, CV_32FC1, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gradient_x, gradient_y, magnitude);
    return magnitude;
}

bool agreesWithLocalReference(const cv::Mat &reference_depth,
                              int row,
                              int column,
                              float candidate,
                              int radius,
                              float relative_threshold)
{
    int valid_neighbor_count = 0;
    int agreeing_neighbor_count = 0;
    for (int delta_row = -radius; delta_row <= radius; ++delta_row)
    {
        for (int delta_column = -radius; delta_column <= radius; ++delta_column)
        {
            if (delta_row == 0 && delta_column == 0)
            {
                continue;
            }
            const int neighbor_row = row + delta_row;
            const int neighbor_column = column + delta_column;
            if (neighbor_row < 0 || neighbor_row >= reference_depth.rows ||
                neighbor_column < 0 || neighbor_column >= reference_depth.cols)
            {
                continue;
            }
            const float neighbor = reference_depth.at<float>(neighbor_row, neighbor_column);
            if (!validDepth(neighbor))
            {
                continue;
            }
            ++valid_neighbor_count;
            if (relativeDifference(candidate, neighbor) <= relative_threshold)
            {
                ++agreeing_neighbor_count;
            }
        }
    }
    return valid_neighbor_count < 3 || agreeing_neighbor_count > 0;
}

} // namespace

cv::Mat projectSourceDepthToReference(
    const cv::Mat &source_depth,
    const FramePinholeCamera &source_camera,
    const FramePinholeCamera &reference_camera,
    const cv::Size &reference_size,
    float maximum_projection_distance_pixels,
    std::uint64_t *projected_candidate_count,
    int row_worker_count,
    const std::atomic<bool> *cancelled)
{
    if (projected_candidate_count)
    {
        *projected_candidate_count = 0;
    }
    if (source_depth.empty() || source_depth.type() != CV_32FC1 ||
        !source_camera.isValid() || !reference_camera.isValid() ||
        reference_size.width <= 0 || reference_size.height <= 0)
    {
        return {};
    }

    cv::Mat projected(reference_size, CV_32FC1, cv::Scalar(0.0f));
    const float maximum_distance = std::clamp(
        maximum_projection_distance_pixels, 0.25f, 1.5f);
    std::atomic<std::uint64_t> candidate_count{0};
    parallelForRows(
        source_depth.rows,
        row_worker_count,
        cancelled,
        [&](int source_row)
        {
        std::uint64_t row_candidate_count = 0;
        const float *source_values = source_depth.ptr<float>(source_row);
        for (int source_column = 0; source_column < source_depth.cols; ++source_column)
        {
            if ((source_column & 63) == 0 && cancelled &&
                cancelled->load(std::memory_order_relaxed))
            {
                break;
            }
            const float source_value = source_values[source_column];
            if (!validDepth(source_value))
            {
                continue;
            }
            const double source_pixel[2] = {
                static_cast<double>(source_column), static_cast<double>(source_row)};
            double world[3] = {};
            if (!source_camera.unprojectPixel(source_pixel, source_value, world))
            {
                continue;
            }
            double reference_pixel[2] = {};
            double reference_value = 0.0;
            if (!reference_camera.projectWorldPointWithDepth(
                    world, reference_pixel, reference_value) ||
                !std::isfinite(reference_value) || reference_value <= 0.0)
            {
                continue;
            }

            const int first_column = static_cast<int>(std::floor(reference_pixel[0]));
            const int first_row = static_cast<int>(std::floor(reference_pixel[1]));
            for (int delta_row = 0; delta_row <= 1; ++delta_row)
            {
                for (int delta_column = 0; delta_column <= 1; ++delta_column)
                {
                    const int column = first_column + delta_column;
                    const int row = first_row + delta_row;
                    if (column < 0 || column >= projected.cols ||
                        row < 0 || row >= projected.rows)
                    {
                        continue;
                    }
                    const double offset_x = reference_pixel[0] - column;
                    const double offset_y = reference_pixel[1] - row;
                    if (std::sqrt(offset_x * offset_x + offset_y * offset_y) >
                        maximum_distance)
                    {
                        continue;
                    }
                    const float candidate = static_cast<float>(reference_value);
                    float &stored_value = projected.ptr<float>(row)[column];
                    std::atomic_ref<float> stored(stored_value);
                    float observed = stored.load(std::memory_order_relaxed);
                    while ((!validDepth(observed) || candidate < observed) &&
                           !stored.compare_exchange_weak(
                               observed,
                               candidate,
                               std::memory_order_relaxed,
                               std::memory_order_relaxed))
                    {
                    }
                    ++row_candidate_count;
                }
            }
        }
        candidate_count.fetch_add(row_candidate_count, std::memory_order_relaxed);
    });
    if (projected_candidate_count)
    {
        *projected_candidate_count = candidate_count.load(std::memory_order_relaxed);
    }
    return projected;
}

DominantDepthLayerSelectionStats selectDominantProjectedDepthLayer(
    cv::Mat &reference_depth,
    const cv::Mat &support_mask,
    const std::vector<cv::Mat> &projected_source_depths,
    const cv::Mat &consistent_source_votes,
    const cv::Mat &contradicted_source_votes,
    const DominantDepthLayerSelectionOptions &options,
    cv::Mat *reference_confidence,
    cv::Mat *selected_layer_mask,
    cv::Mat *geometry_source_mask,
    cv::Mat *source_inverse_depth_sum,
    cv::Mat *source_inverse_depth_squared_sum,
    cv::Mat *selected_source_votes,
    int row_worker_count,
    const std::atomic<bool> *cancelled,
    const cv::Mat *native_reliability_class_map,
    cv::Mat *reliability_guided_changed_mask,
    const std::vector<ProjectedDepthEvidence> *projected_source_evidence,
    DepthGeometryHypothesisRerankMaps *geometry_rerank_maps)
{
    DominantDepthLayerSelectionStats result;
    result.reliabilityGuidedOnlyMode =
        options.restrictToReliabilityGuidedCandidates;
    if (reference_depth.type() != CV_32FC1 ||
        projected_source_depths.empty())
    {
        return result;
    }

    const cv::Size size = reference_depth.size();
    const bool has_support = support_mask.type() == CV_8UC1 &&
        support_mask.size() == size;
    const bool has_confidence = reference_confidence &&
        reference_confidence->type() == CV_32FC1 &&
        reference_confidence->size() == size;
    const bool has_consistent_votes = consistent_source_votes.type() == CV_16UC1 &&
        consistent_source_votes.size() == size;
    const bool has_contradicted_votes = contradicted_source_votes.type() == CV_16UC1 &&
        contradicted_source_votes.size() == size;
    const bool update_selection_mask = selected_layer_mask != nullptr;
    if (update_selection_mask &&
        (selected_layer_mask->type() != CV_8UC1 ||
         selected_layer_mask->size() != size))
    {
        *selected_layer_mask = cv::Mat(size, CV_8UC1, cv::Scalar(0));
    }
    const bool update_geometry = geometry_source_mask &&
        source_inverse_depth_sum && source_inverse_depth_squared_sum &&
        geometry_source_mask->type() == CV_16UC1 &&
        source_inverse_depth_sum->type() == CV_32FC1 &&
        source_inverse_depth_squared_sum->type() == CV_32FC1 &&
        geometry_source_mask->size() == size &&
        source_inverse_depth_sum->size() == size &&
        source_inverse_depth_squared_sum->size() == size;
    const bool update_votes = selected_source_votes &&
        selected_source_votes->type() == CV_16UC1 &&
        selected_source_votes->size() == size;
    const bool has_reliability_classes = native_reliability_class_map &&
        native_reliability_class_map->type() == CV_8UC1 &&
        native_reliability_class_map->size() == size;
    const bool update_reliability_guided_changed_mask =
        reliability_guided_changed_mask != nullptr;
    if (update_reliability_guided_changed_mask &&
        (reliability_guided_changed_mask->type() != CV_8UC1 ||
         reliability_guided_changed_mask->size() != size))
    {
        *reliability_guided_changed_mask = cv::Mat(
            size, CV_8UC1, cv::Scalar(0));
    }
    const bool has_projected_source_evidence = projected_source_evidence &&
        projected_source_evidence->size() == projected_source_depths.size();
    if (geometry_rerank_maps && !geometry_rerank_maps->compatible(size))
    {
        geometry_rerank_maps->initialize(size);
    }

    const int minimum_sources = std::max(2, options.minimumDistinctSourceCount);
    const int minimum_replacement_sources = std::max(
        minimum_sources, options.minimumReplacementSourceCount);
    const float maximum_spread = std::clamp(
        options.maximumRelativeDepthSpread, 0.001f, 0.10f);
    const float native_agreement = std::clamp(
        options.maximumNativeAgreementRelativeDifference,
        maximum_spread,
        0.20f);
    const float blend_weight = std::clamp(
        options.nativeConsensusBlendWeight, 0.0f, 1.0f);
    const float maximum_correction = std::clamp(
        options.maximumNativeRelativeCorrection, 0.0f, 0.05f);
    const float selected_confidence = std::clamp(
        options.selectedLayerConfidence, 0.0f, 1.0f);
    const float ambiguous_multiplier = std::clamp(
        options.ambiguousNativeConfidenceMultiplier, 0.0f, 1.0f);
    const int reliability_minimum_sources = std::max(
        minimum_replacement_sources,
        options.reliabilityGuidedMinimumSourceCount);
    const float reliability_blend_weight = std::clamp(
        options.reliabilityGuidedBlendWeight, blend_weight, 1.0f);
    const float reliability_maximum_correction = std::clamp(
        options.reliabilityGuidedMaximumRelativeCorrection,
        maximum_correction,
        native_agreement);

    std::vector<DominantDepthLayerSelectionStats> row_stats(
        static_cast<std::size_t>(reference_depth.rows));
    parallelForRows(
        reference_depth.rows,
        row_worker_count,
        cancelled,
        [&](int row)
        {
        DominantDepthLayerSelectionStats stats;
        std::vector<SourceDepthCandidate> candidates;
        candidates.reserve(projected_source_depths.size());
        float *depth_row = reference_depth.ptr<float>(row);
        float *confidence_row = has_confidence
            ? reference_confidence->ptr<float>(row) : nullptr;
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if ((column & 63) == 0 && cancelled &&
                cancelled->load(std::memory_order_relaxed))
            {
                break;
            }
            if (has_support && support_mask.at<std::uint8_t>(row, column) == 0)
            {
                depth_row[column] = 0.0f;
                if (confidence_row)
                {
                    confidence_row[column] = 0.0f;
                }
                continue;
            }

            ++stats.consideredPixelCount;
            const float native_depth = depth_row[column];
            const bool native_valid = validDepth(native_depth);
            DepthLayerReliabilityClass reliability_class =
                DepthLayerReliabilityClass::Unobservable;
            if (has_reliability_classes)
            {
                reliability_class = static_cast<DepthLayerReliabilityClass>(
                    native_reliability_class_map->at<std::uint8_t>(
                        row, column));
            }
            const bool reliability_guided_candidate =
                options.enableReliabilityGuidedCorrection &&
                has_reliability_classes && native_valid &&
                (reliability_class ==
                     DepthLayerReliabilityClass::AmbiguousLowTexture ||
                 reliability_class == DepthLayerReliabilityClass::RejectedLayer);
            if (reliability_guided_candidate)
            {
                ++stats.reliabilityGuidedCandidatePixelCount;
            }
            if (options.restrictToReliabilityGuidedCandidates &&
                !reliability_guided_candidate)
            {
                continue;
            }
            if (reliability_guided_candidate)
            {
                ++stats.geometryRerankEvaluatedPixelCount;
                DepthGeometryHypothesisDecision decision;
                if (has_projected_source_evidence)
                {
                    decision = rerankMeasuredDepthHypothesis(
                        native_depth,
                        reliability_class,
                        row,
                        column,
                        *projected_source_evidence,
                        options.geometryRerank);
                }
                else
                {
                    decision.rejectedInsufficientSources = true;
                }

                if (geometry_rerank_maps)
                {
                    geometry_rerank_maps->nativeCost.at<float>(row, column) =
                        decision.nativeCost;
                    geometry_rerank_maps->candidateCost.at<float>(row, column) =
                        decision.candidateCost;
                    geometry_rerank_maps->costAdvantage.at<float>(row, column) =
                        decision.costAdvantage;
                    geometry_rerank_maps->effectiveSourceWeight.at<float>(row, column) =
                        decision.effectiveSourceWeight;
                    geometry_rerank_maps->relativeCorrection.at<float>(row, column) =
                        decision.relativeCorrection;
                    geometry_rerank_maps->weakestSourceConfidence.at<float>(row, column) =
                        decision.weakestSourceConfidence;
                    geometry_rerank_maps->supportingSourceCount.at<std::uint8_t>(
                        row, column) = static_cast<std::uint8_t>(std::clamp(
                            decision.supportingSourceCount, 0, 255));
                    geometry_rerank_maps->baselineSectorCount.at<std::uint8_t>(
                        row, column) = static_cast<std::uint8_t>(std::clamp(
                            decision.baselineSectorCount, 0, 255));
                    geometry_rerank_maps->decisionAction.at<std::uint8_t>(
                        row, column) = static_cast<std::uint8_t>(decision.action);
                }
                if (decision.validEvidence)
                {
                    ++stats.geometryRerankValidEvidencePixelCount;
                    stats.geometryRerankNativeCostSum += decision.nativeCost;
                    stats.geometryRerankCandidateCostSum += decision.candidateCost;
                    stats.geometryRerankCostAdvantageSum += decision.costAdvantage;
                    stats.geometryRerankCorrectionSum += decision.relativeCorrection;
                    stats.geometryRerankWeakestConfidenceSum +=
                        decision.weakestSourceConfidence;
                }
                if (decision.rejectedInsufficientSources)
                {
                    ++stats.geometryRerankRejectedSourceCount;
                    ++stats.reliabilityGuidedInsufficientSourcePixelCount;
                }
                if (decision.rejectedInsufficientBaseline)
                {
                    ++stats.geometryRerankRejectedBaselineCount;
                    ++stats.reliabilityGuidedInsufficientSourcePixelCount;
                }
                if (decision.rejectedInsufficientWeight)
                {
                    ++stats.geometryRerankRejectedWeightCount;
                    ++stats.reliabilityGuidedInsufficientSourcePixelCount;
                }
                if (decision.rejectedCostAdvantage)
                {
                    ++stats.geometryRerankRejectedCostCount;
                }

                const bool accepted =
                    decision.action != DepthGeometryHypothesisAction::None;
                if (accepted)
                {
                    ++stats.stableLayerPixelCount;
                    ++stats.reliabilityGuidedStablePixelCount;
                    depth_row[column] = decision.selectedDepth;
                    if (decision.action == DepthGeometryHypothesisAction::Refine)
                    {
                        ++stats.refinedNativePixelCount;
                        ++stats.reliabilityGuidedRefinedPixelCount;
                        ++stats.geometryRerankRefinedPixelCount;
                    }
                    else
                    {
                        ++stats.switchedNativePixelCount;
                        ++stats.reliabilityGuidedSwitchedPixelCount;
                        ++stats.geometryRerankSwitchedPixelCount;
                        if (update_selection_mask)
                        {
                            selected_layer_mask->at<std::uint8_t>(row, column) = 255;
                        }
                    }
                    if (update_reliability_guided_changed_mask)
                    {
                        reliability_guided_changed_mask->at<std::uint8_t>(
                            row, column) = 255;
                    }
                    if (confidence_row)
                    {
                        const float native_confidence = std::clamp(
                            confidence_row[column], 0.0f, 1.0f);
                        confidence_row[column] = std::sqrt(
                            native_confidence * decision.evidenceConfidence);
                    }
                    if (update_geometry)
                    {
                        geometry_source_mask->at<std::uint16_t>(row, column) =
                            decision.supportingSourceMask;
                        source_inverse_depth_sum->at<float>(row, column) =
                            decision.supportingInverseDepthSum;
                        source_inverse_depth_squared_sum->at<float>(row, column) =
                            decision.supportingInverseDepthSquaredSum;
                    }
                    if (update_votes)
                    {
                        selected_source_votes->at<std::uint16_t>(row, column) =
                            static_cast<std::uint16_t>(std::clamp(
                                decision.supportingSourceCount, 0, 65535));
                    }
                }
                else
                {
                    ++stats.ambiguousNativePixelCount;
                }
                // The experimental reliability path is fail-closed: it never
                // falls through to the legacy count-only selector.
                continue;
            }
            candidates.clear();
            for (int source_ordinal = 0;
                 source_ordinal < static_cast<int>(projected_source_depths.size());
                 ++source_ordinal)
            {
                const cv::Mat &projected = projected_source_depths[
                    static_cast<std::size_t>(source_ordinal)];
                if (projected.type() != CV_32FC1 || projected.size() != size)
                {
                    continue;
                }
                const float candidate = projected.at<float>(row, column);
                if (validDepth(candidate))
                {
                    candidates.push_back({candidate, source_ordinal});
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const SourceDepthCandidate &left,
                         const SourceDepthCandidate &right)
                      {
                          return left.depth < right.depth;
                      });

            int best_begin = -1;
            int best_count = 0;
            float best_spread = std::numeric_limits<float>::infinity();
            float best_native_distance = std::numeric_limits<float>::infinity();
            for (int begin = 0; begin < static_cast<int>(candidates.size()); ++begin)
            {
                int end = begin + 1;
                while (end < static_cast<int>(candidates.size()) &&
                       relativeDifference(candidates[begin].depth,
                                          candidates[end].depth) <= maximum_spread)
                {
                    ++end;
                }
                const int count = end - begin;
                const float median = candidates[
                    static_cast<std::size_t>(begin + (count - 1) / 2)].depth;
                const float spread = count > 1
                    ? relativeDifference(candidates[begin].depth,
                                         candidates[end - 1].depth)
                    : 0.0f;
                const float native_distance = native_valid
                    ? relativeDifference(native_depth, median) : 0.0f;
                if (count > best_count ||
                    (count == best_count && spread < best_spread) ||
                    (count == best_count && spread == best_spread &&
                     native_distance < best_native_distance))
                {
                    best_begin = begin;
                    best_count = count;
                    best_spread = spread;
                    best_native_distance = native_distance;
                }
            }

            if (best_count < minimum_sources)
            {
                if (reliability_guided_candidate)
                {
                    ++stats.reliabilityGuidedInsufficientSourcePixelCount;
                }
                if (native_valid)
                {
                    ++stats.ambiguousNativePixelCount;
                    if (confidence_row && has_contradicted_votes &&
                        contradicted_source_votes.at<std::uint16_t>(row, column) > 0)
                    {
                        confidence_row[column] = std::clamp(
                            confidence_row[column] * ambiguous_multiplier,
                            0.0f,
                            1.0f);
                    }
                }
                else
                {
                    ++stats.unresolvedMissingPixelCount;
                }
                continue;
            }

            ++stats.stableLayerPixelCount;
            const bool reliability_guided_stable =
                reliability_guided_candidate &&
                best_count >= reliability_minimum_sources;
            if (reliability_guided_candidate && !reliability_guided_stable)
            {
                ++stats.reliabilityGuidedInsufficientSourcePixelCount;
            }
            if (reliability_guided_stable)
            {
                ++stats.reliabilityGuidedStablePixelCount;
            }
            const int median_index = best_begin + (best_count - 1) / 2;
            const float selected_depth = candidates[
                static_cast<std::size_t>(median_index)].depth;
            bool selected_from_sources = false;
            if (!native_valid)
            {
                if (!options.transferObservedDepthIntoMissingPixels)
                {
                    ++stats.unresolvedMissingPixelCount;
                    continue;
                }
                depth_row[column] = selected_depth;
                selected_from_sources = true;
                ++stats.transferredMissingPixelCount;
            }
            else if (relativeDifference(native_depth, selected_depth) <= native_agreement)
            {
                const float native_inverse = 1.0f / native_depth;
                const float selected_inverse = 1.0f / selected_depth;
                const float effective_blend_weight = reliability_guided_stable
                    ? reliability_blend_weight : blend_weight;
                const float effective_maximum_correction = reliability_guided_stable
                    ? reliability_maximum_correction : maximum_correction;
                float refined_depth = 1.0f /
                    ((1.0f - effective_blend_weight) * native_inverse +
                     effective_blend_weight * selected_inverse);
                const float maximum_delta =
                    native_depth * effective_maximum_correction;
                refined_depth = std::clamp(
                    refined_depth,
                    native_depth - maximum_delta,
                    native_depth + maximum_delta);
                if (std::fabs(refined_depth - native_depth) > 1.0e-7f)
                {
                    depth_row[column] = refined_depth;
                    ++stats.refinedNativePixelCount;
                    if (reliability_guided_stable)
                    {
                        ++stats.reliabilityGuidedRefinedPixelCount;
                        if (update_reliability_guided_changed_mask)
                        {
                            reliability_guided_changed_mask->at<std::uint8_t>(
                                row, column) = 255;
                        }
                    }
                }
            }
            else if (best_count >= minimum_replacement_sources &&
                     ((!has_consistent_votes || !has_contradicted_votes ||
                       contradicted_source_votes.at<std::uint16_t>(row, column) >
                           consistent_source_votes.at<std::uint16_t>(row, column)) ||
                      (reliability_guided_stable &&
                       reliability_class ==
                           DepthLayerReliabilityClass::RejectedLayer)))
            {
                depth_row[column] = selected_depth;
                selected_from_sources = true;
                ++stats.switchedNativePixelCount;
                if (reliability_guided_stable)
                {
                    ++stats.reliabilityGuidedSwitchedPixelCount;
                    if (update_reliability_guided_changed_mask)
                    {
                        reliability_guided_changed_mask->at<std::uint8_t>(
                            row, column) = 255;
                    }
                }
            }
            else
            {
                ++stats.ambiguousNativePixelCount;
                if (confidence_row)
                {
                    confidence_row[column] = std::clamp(
                        confidence_row[column] * ambiguous_multiplier,
                        0.0f,
                        1.0f);
                }
                continue;
            }

            if (selected_from_sources && update_selection_mask)
            {
                selected_layer_mask->at<std::uint8_t>(row, column) = 255;
            }
            if (confidence_row && selected_from_sources)
            {
                confidence_row[column] = std::min(
                    selected_confidence,
                    std::max(confidence_row[column], selected_confidence * 0.75f));
            }

            std::uint16_t source_bits = 0;
            float inverse_sum = 0.0f;
            float inverse_squared_sum = 0.0f;
            for (int candidate_index = best_begin;
                 candidate_index < best_begin + best_count;
                 ++candidate_index)
            {
                const SourceDepthCandidate &candidate = candidates[
                    static_cast<std::size_t>(candidate_index)];
                if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
                {
                    source_bits = static_cast<std::uint16_t>(
                        source_bits |
                        (static_cast<std::uint16_t>(1U) << candidate.sourceOrdinal));
                }
                const float inverse_depth = 1.0f / candidate.depth;
                inverse_sum += inverse_depth;
                inverse_squared_sum += inverse_depth * inverse_depth;
            }
            if (update_geometry)
            {
                geometry_source_mask->at<std::uint16_t>(row, column) = source_bits;
                source_inverse_depth_sum->at<float>(row, column) = inverse_sum;
                source_inverse_depth_squared_sum->at<float>(row, column) =
                    inverse_squared_sum;
            }
            if (update_votes)
            {
                selected_source_votes->at<std::uint16_t>(row, column) =
                    static_cast<std::uint16_t>(std::min(
                        best_count,
                        static_cast<int>(
                            std::numeric_limits<std::uint16_t>::max())));
            }
        }
        row_stats[static_cast<std::size_t>(row)] = stats;
    });
    for (const DominantDepthLayerSelectionStats &stats : row_stats)
    {
        addStats(result, stats);
    }
    return result;
}

CrossViewHoleRepairStats repairDepthHolesFromProjectedSources(
    cv::Mat &reference_depth,
    const cv::Mat &support_mask,
    const std::vector<cv::Mat> &projected_source_depths,
    const CrossViewHoleRepairOptions &options,
    cv::Mat *reference_confidence,
    cv::Mat *consistent_source_votes,
    cv::Mat *repaired_mask,
    cv::Mat *geometry_source_mask,
    cv::Mat *source_inverse_depth_sum,
    cv::Mat *source_inverse_depth_squared_sum,
    const FramePinholeCamera *reference_camera,
    const cv::Mat *guide_gray,
    cv::Mat *anchored_interpolation_mask,
    int row_worker_count,
    const std::atomic<bool> *cancelled,
    const cv::Mat *native_interpolation_anchor_eligibility_mask)
{
    CrossViewHoleRepairStats stats;
    if (reference_depth.empty() || reference_depth.type() != CV_32FC1 ||
        projected_source_depths.empty())
    {
        return stats;
    }
    const bool has_support = support_mask.type() == CV_8UC1 &&
        support_mask.size() == reference_depth.size();
    const bool has_confidence = reference_confidence &&
        reference_confidence->type() == CV_32FC1 &&
        reference_confidence->size() == reference_depth.size();
    const bool has_votes = consistent_source_votes &&
        consistent_source_votes->type() == CV_16UC1 &&
        consistent_source_votes->size() == reference_depth.size();
    const bool has_geometry_evidence = geometry_source_mask &&
        geometry_source_mask->type() == CV_16UC1 &&
        geometry_source_mask->size() == reference_depth.size() &&
        source_inverse_depth_sum && source_inverse_depth_sum->type() == CV_32FC1 &&
        source_inverse_depth_sum->size() == reference_depth.size() &&
        source_inverse_depth_squared_sum &&
        source_inverse_depth_squared_sum->type() == CV_32FC1 &&
        source_inverse_depth_squared_sum->size() == reference_depth.size();
    cv::Mat strong_repaired_mask = repaired_mask &&
            repaired_mask->type() == CV_8UC1 &&
            repaired_mask->size() == reference_depth.size()
        ? repaired_mask->clone()
        : cv::Mat(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    if (repaired_mask)
    {
        if (repaired_mask->type() != CV_8UC1 ||
            repaired_mask->size() != reference_depth.size())
        {
            *repaired_mask = cv::Mat(
                reference_depth.size(), CV_8UC1, cv::Scalar(0));
        }
    }
    if (anchored_interpolation_mask)
    {
        *anchored_interpolation_mask = cv::Mat(
            reference_depth.size(), CV_8UC1, cv::Scalar(0));
    }
    auto interpolate_anchored_components = [&]()
    {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
        {
            return;
        }
        cv::Mat interpolation_anchor_mask = strong_repaired_mask.clone();
        if (options.includeValidNativeInterpolationAnchors)
        {
            const bool restrict_native_anchors =
                native_interpolation_anchor_eligibility_mask != nullptr;
            const bool compatible_native_anchor_mask =
                restrict_native_anchors &&
                native_interpolation_anchor_eligibility_mask->type() == CV_8UC1 &&
                native_interpolation_anchor_eligibility_mask->size() ==
                    reference_depth.size();
            std::vector<std::uint64_t> candidate_counts(
                static_cast<std::size_t>(reference_depth.rows), 0);
            std::vector<std::uint64_t> accepted_counts(
                static_cast<std::size_t>(reference_depth.rows), 0);
            parallelForRows(
                reference_depth.rows,
                row_worker_count,
                cancelled,
                [&](int row)
                {
                const float *depth_row = reference_depth.ptr<float>(row);
                const std::uint8_t *eligibility_row =
                    compatible_native_anchor_mask
                    ? native_interpolation_anchor_eligibility_mask
                          ->ptr<std::uint8_t>(row)
                    : nullptr;
                std::uint8_t *anchor_row =
                    interpolation_anchor_mask.ptr<std::uint8_t>(row);
                std::uint64_t candidates = 0;
                std::uint64_t accepted = 0;
                for (int column = 0; column < reference_depth.cols; ++column)
                {
                    if (validDepth(depth_row[column]))
                    {
                        ++candidates;
                        const bool eligible = !restrict_native_anchors ||
                            (eligibility_row && eligibility_row[column] != 0);
                        if (eligible)
                        {
                            anchor_row[column] = 255;
                            ++accepted;
                        }
                    }
                }
                candidate_counts[static_cast<std::size_t>(row)] = candidates;
                accepted_counts[static_cast<std::size_t>(row)] = accepted;
            });
            for (std::size_t row = 0; row < candidate_counts.size(); ++row)
            {
                stats.nativeInterpolationAnchorCandidateCount +=
                    candidate_counts[row];
                stats.nativeInterpolationAnchorAcceptedCount +=
                    accepted_counts[row];
            }
            stats.nativeInterpolationAnchorRejectedCount =
                stats.nativeInterpolationAnchorCandidateCount -
                stats.nativeInterpolationAnchorAcceptedCount;
        }
        if (cancelled && cancelled->load(std::memory_order_relaxed))
        {
            return;
        }
        cv::Mat *interpolation_output = anchored_interpolation_mask
            ? anchored_interpolation_mask : repaired_mask;
        stats.anchoredInterpolation = interpolateAnchoredInternalDepthHoles(
            reference_depth,
            has_support
                ? support_mask
                : cv::Mat(reference_depth.size(), CV_8UC1, cv::Scalar(255)),
            interpolation_anchor_mask,
            guide_gray,
            options.anchoredInterpolation,
            has_confidence ? reference_confidence : nullptr,
            interpolation_output);
        if (anchored_interpolation_mask && repaired_mask)
        {
            cv::bitwise_or(
                *repaired_mask, *anchored_interpolation_mask, *repaired_mask);
        }
        stats.repairedPixelCount +=
            stats.anchoredInterpolation.interpolatedPixelCount;
    };
    const int minimum_sources = std::max(2, options.minimumDistinctSourceCount);
    const float maximum_spread = std::clamp(
        options.maximumRelativeDepthSpread, 0.001f, 0.10f);
    const int local_radius = std::clamp(options.localDepthRadius, 0, 3);
    const float local_threshold = std::clamp(
        options.maximumLocalRelativeDepthDifference, maximum_spread, 0.20f);
    const cv::Mat original_depth = reference_depth.clone();

    std::vector<CrossViewHoleRepairStats> initial_row_stats(
        static_cast<std::size_t>(reference_depth.rows));
    parallelForRows(
        reference_depth.rows,
        row_worker_count,
        cancelled,
        [&](int row)
        {
        CrossViewHoleRepairStats row_stats;
        std::vector<SourceDepthCandidate> candidates;
        candidates.reserve(projected_source_depths.size());
        float *depth_row = reference_depth.ptr<float>(row);
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if ((column & 63) == 0 && cancelled &&
                cancelled->load(std::memory_order_relaxed))
            {
                break;
            }
            if (validDepth(depth_row[column]) ||
                (has_support && support_mask.at<std::uint8_t>(row, column) == 0))
            {
                continue;
            }
            ++row_stats.consideredHolePixelCount;
            candidates.clear();
            for (int source_ordinal = 0;
                 source_ordinal < static_cast<int>(projected_source_depths.size());
                 ++source_ordinal)
            {
                const cv::Mat &projected = projected_source_depths[
                    static_cast<std::size_t>(source_ordinal)];
                if (projected.type() != CV_32FC1 ||
                    projected.size() != reference_depth.size())
                {
                    continue;
                }
                const float candidate = projected.at<float>(row, column);
                if (validDepth(candidate))
                {
                    candidates.push_back({candidate, source_ordinal});
                    ++row_stats.projectedCandidateCount;
                }
            }
            if (static_cast<int>(candidates.size()) < minimum_sources)
            {
                ++row_stats.rejectedInsufficientSourceCount;
                continue;
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const SourceDepthCandidate &left,
                         const SourceDepthCandidate &right)
                      {
                          return left.depth < right.depth;
                      });
            int best_begin = -1;
            int best_count = 0;
            for (int begin = 0; begin < static_cast<int>(candidates.size()); ++begin)
            {
                int end = begin + 1;
                while (end < static_cast<int>(candidates.size()) &&
                       relativeDifference(candidates[begin].depth, candidates[end].depth) <=
                           maximum_spread)
                {
                    ++end;
                }
                const int count = end - begin;
                if (count > best_count)
                {
                    best_begin = begin;
                    best_count = count;
                }
            }
            if (best_count < minimum_sources)
            {
                ++row_stats.rejectedDepthSpreadCount;
                continue;
            }
            const int reference_candidate_index = best_begin + (best_count - 1) / 2;
            const float repaired_depth = candidates[
                static_cast<std::size_t>(reference_candidate_index)].depth;
            if (!agreesWithLocalReference(original_depth,
                                          row,
                                          column,
                                          repaired_depth,
                                          local_radius,
                                          local_threshold))
            {
                ++row_stats.rejectedLocalDepthCount;
                continue;
            }

            depth_row[column] = repaired_depth;
            if (has_confidence)
            {
                reference_confidence->at<float>(row, column) = std::clamp(
                    options.repairedConfidence + 0.05f * (best_count - minimum_sources),
                    0.0f,
                    0.85f);
            }
            if (has_votes)
            {
                consistent_source_votes->at<std::uint16_t>(row, column) =
                    static_cast<std::uint16_t>(std::min(
                        best_count - 1,
                        static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
            }
            if (has_geometry_evidence)
            {
                std::uint16_t source_bits = 0;
                float inverse_sum = 0.0f;
                float inverse_squared_sum = 0.0f;
                for (int candidate_index = best_begin;
                     candidate_index < best_begin + best_count;
                     ++candidate_index)
                {
                    const SourceDepthCandidate &candidate = candidates[
                        static_cast<std::size_t>(candidate_index)];
                    if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
                    {
                        source_bits = static_cast<std::uint16_t>(
                            source_bits |
                            (static_cast<std::uint16_t>(1U) << candidate.sourceOrdinal));
                    }
                    if (candidate_index == reference_candidate_index)
                    {
                        continue;
                    }
                    const float inverse_depth = 1.0f / candidate.depth;
                    inverse_sum += inverse_depth;
                    inverse_squared_sum += inverse_depth * inverse_depth;
                }
                geometry_source_mask->at<std::uint16_t>(row, column) = source_bits;
                source_inverse_depth_sum->at<float>(row, column) = inverse_sum;
                source_inverse_depth_squared_sum->at<float>(row, column) =
                    inverse_squared_sum;
            }
            if (repaired_mask)
            {
                repaired_mask->at<std::uint8_t>(row, column) = 255;
            }
            strong_repaired_mask.at<std::uint8_t>(row, column) = 255;
            ++row_stats.repairedPixelCount;
        }
        initial_row_stats[static_cast<std::size_t>(row)] = row_stats;
    });
    for (const CrossViewHoleRepairStats &row_stats : initial_row_stats)
    {
        addStats(stats, row_stats);
    }
    if (cancelled && cancelled->load(std::memory_order_relaxed))
    {
        return stats;
    }
    if (!options.enableTwoSourceGrowth || !has_votes || !has_geometry_evidence ||
        !reference_camera || !reference_camera->isValid())
    {
        interpolate_anchored_components();
        return stats;
    }

    const int maximum_growth_distance = std::clamp(
        options.maximumGrowthDistancePixels, 1, 8);
    const float maximum_growth_spread = std::clamp(
        options.maximumGrowthInverseDepthSpread, 0.001f, 0.05f);
    const float maximum_normal_angle = std::clamp(
        options.maximumGrowthNormalAngleDegrees, 5.0f, 45.0f);
    const int maximum_component_area = std::clamp(
        options.maximumGrowthComponentArea, 1, 512);
    const cv::Mat image_gradient = guideGradient(guide_gray, reference_depth.size());
    if (image_gradient.empty())
    {
        interpolate_anchored_components();
        return stats;
    }

    cv::Mat weak_mask(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat weak_depth(reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat weak_source_mask(reference_depth.size(), CV_16UC1, cv::Scalar(0));
    cv::Mat weak_inverse_sum(reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat weak_inverse_squared_sum(
        reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat weak_spread(reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    std::vector<std::uint64_t> weak_candidate_counts(
        static_cast<std::size_t>(reference_depth.rows), 0);
    parallelForRows(
        reference_depth.rows,
        row_worker_count,
        cancelled,
        [&](int row)
        {
        std::vector<SourceDepthCandidate> candidates;
        candidates.reserve(projected_source_depths.size());
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if ((column & 63) == 0 && cancelled &&
                cancelled->load(std::memory_order_relaxed))
            {
                break;
            }
            if (validDepth(reference_depth.at<float>(row, column)) ||
                (has_support && support_mask.at<std::uint8_t>(row, column) == 0))
            {
                continue;
            }
            candidates.clear();
            for (int source_ordinal = 0;
                 source_ordinal < static_cast<int>(projected_source_depths.size());
                 ++source_ordinal)
            {
                const cv::Mat &projected = projected_source_depths[
                    static_cast<std::size_t>(source_ordinal)];
                if (projected.type() != CV_32FC1 ||
                    projected.size() != reference_depth.size())
                {
                    continue;
                }
                const float candidate = projected.at<float>(row, column);
                if (validDepth(candidate))
                {
                    candidates.push_back({candidate, source_ordinal});
                }
            }
            if (candidates.size() < 2)
            {
                continue;
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const SourceDepthCandidate &left,
                         const SourceDepthCandidate &right)
                      {
                          return left.depth < right.depth;
                      });
            int best_begin = -1;
            int best_count = 0;
            for (int begin = 0; begin < static_cast<int>(candidates.size()); ++begin)
            {
                int end = begin + 1;
                while (end < static_cast<int>(candidates.size()) &&
                       relativeDifference(candidates[begin].depth,
                                          candidates[end].depth) <= maximum_spread)
                {
                    ++end;
                }
                if (end - begin > best_count)
                {
                    best_begin = begin;
                    best_count = end - begin;
                }
            }
            if (best_count != 2)
            {
                continue;
            }
            const int reference_candidate_index = best_begin;
            const float candidate_depth = candidates[
                static_cast<std::size_t>(reference_candidate_index)].depth;
            float inverse_sum = 0.0f;
            float inverse_squared_sum = 0.0f;
            float all_inverse_sum = 0.0f;
            float all_inverse_squared_sum = 0.0f;
            std::uint16_t source_bits = 0;
            for (int index = best_begin; index < best_begin + best_count; ++index)
            {
                const SourceDepthCandidate &candidate = candidates[
                    static_cast<std::size_t>(index)];
                if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
                {
                    source_bits = static_cast<std::uint16_t>(
                        source_bits |
                        (static_cast<std::uint16_t>(1U) << candidate.sourceOrdinal));
                }
                const float inverse_depth = 1.0f / candidate.depth;
                all_inverse_sum += inverse_depth;
                all_inverse_squared_sum += inverse_depth * inverse_depth;
                if (index != reference_candidate_index)
                {
                    inverse_sum += inverse_depth;
                    inverse_squared_sum += inverse_depth * inverse_depth;
                }
            }
            if (bitCount(source_bits) < 2)
            {
                continue;
            }
            const float inverse_mean = all_inverse_sum / best_count;
            const float inverse_variance = std::max(
                0.0f, all_inverse_squared_sum / best_count - inverse_mean * inverse_mean);
            const float relative_spread = inverse_mean > 1.0e-12f
                ? std::sqrt(inverse_variance) / inverse_mean : 1.0f;
            if (relative_spread > maximum_growth_spread)
            {
                continue;
            }
            weak_mask.at<std::uint8_t>(row, column) = 255;
            weak_depth.at<float>(row, column) = candidate_depth;
            weak_source_mask.at<std::uint16_t>(row, column) = source_bits;
            weak_inverse_sum.at<float>(row, column) = inverse_sum;
            weak_inverse_squared_sum.at<float>(row, column) = inverse_squared_sum;
            weak_spread.at<float>(row, column) = relative_spread;
            ++weak_candidate_counts[static_cast<std::size_t>(row)];
        }
    });
    for (const std::uint64_t count : weak_candidate_counts)
    {
        stats.twoSourceCandidatePixelCount += count;
    }
    if (cancelled && cancelled->load(std::memory_order_relaxed))
    {
        return stats;
    }
    if (cv::countNonZero(weak_mask) == 0)
    {
        interpolate_anchored_components();
        return stats;
    }

    cv::Mat candidate_surface = reference_depth.clone();
    weak_depth.copyTo(candidate_surface, weak_mask);
    cv::Mat strong_mask(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    parallelForRows(
        reference_depth.rows,
        row_worker_count,
        cancelled,
        [&](int row)
        {
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if (!validDepth(reference_depth.at<float>(row, column)))
            {
                continue;
            }
            if (strong_repaired_mask.at<std::uint8_t>(row, column) != 0 ||
                consistent_source_votes->at<std::uint16_t>(row, column) >= 3)
            {
                strong_mask.at<std::uint8_t>(row, column) = 255;
            }
        }
    });
    if (cancelled && cancelled->load(std::memory_order_relaxed))
    {
        return stats;
    }

    cv::Mat labels;
    cv::Mat component_stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        weak_mask, labels, component_stats, centroids, 8, CV_32S);
    struct GrowthNode
    {
        float cost = 0.0f;
        int distance = 0;
        int row = 0;
        int column = 0;
        int parentRow = 0;
        int parentColumn = 0;
    };
    struct GrowthNodeGreater
    {
        bool operator()(const GrowthNode &left, const GrowthNode &right) const
        {
            return left.cost > right.cost;
        }
    };
    constexpr int neighbor_offsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    cv::Mat accepted(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat accepted_distance(
        reference_depth.size(), CV_16UC1, cv::Scalar(0xffff));
    for (int label = 1; label < component_count; ++label)
    {
        if (cancelled && cancelled->load(std::memory_order_relaxed))
        {
            return stats;
        }
        const int area = component_stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > maximum_component_area)
        {
            ++stats.growthRejectedComponentAreaCount;
            continue;
        }
        std::priority_queue<GrowthNode,
                            std::vector<GrowthNode>,
                            GrowthNodeGreater> queue;
        const int top = component_stats.at<int>(label, cv::CC_STAT_TOP);
        const int left = component_stats.at<int>(label, cv::CC_STAT_LEFT);
        const int height = component_stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int width = component_stats.at<int>(label, cv::CC_STAT_WIDTH);
        for (int row = top; row < top + height; ++row)
        {
            for (int column = left; column < left + width; ++column)
            {
                if (labels.at<int>(row, column) != label)
                {
                    continue;
                }
                for (const auto &offset : neighbor_offsets)
                {
                    const int neighbor_row = row + offset[0];
                    const int neighbor_column = column + offset[1];
                    if (neighbor_row < 0 || neighbor_row >= reference_depth.rows ||
                        neighbor_column < 0 || neighbor_column >= reference_depth.cols ||
                        strong_mask.at<std::uint8_t>(neighbor_row, neighbor_column) == 0)
                    {
                        continue;
                    }
                    const std::uint16_t candidate_sources =
                        weak_source_mask.at<std::uint16_t>(row, column);
                    const std::uint16_t seed_sources =
                        geometry_source_mask->at<std::uint16_t>(neighbor_row,
                                                                neighbor_column);
                    if ((candidate_sources & seed_sources) == 0)
                    {
                        ++stats.growthRejectedSourceOverlapCount;
                        continue;
                    }
                    const float gradient = image_gradient.at<float>(row, column);
                    const float cost = weak_spread.at<float>(row, column) * 1000.0f +
                        gradient * 0.01f;
                    queue.push({cost,
                                1,
                                row,
                                column,
                                neighbor_row,
                                neighbor_column});
                }
            }
        }

        while (!queue.empty())
        {
            if (cancelled && cancelled->load(std::memory_order_relaxed))
            {
                return stats;
            }
            const GrowthNode node = queue.top();
            queue.pop();
            if (accepted.at<std::uint8_t>(node.row, node.column) != 0 ||
                node.distance > maximum_growth_distance)
            {
                continue;
            }
            const float gradient = image_gradient.at<float>(node.row, node.column);
            if (gradient > options.maximumGrowthImageGradient)
            {
                ++stats.growthRejectedImageEdgeCount;
                continue;
            }
            if (!normalsAgree(candidate_surface,
                              node.row,
                              node.column,
                              node.parentRow,
                              node.parentColumn,
                              *reference_camera,
                              maximum_normal_angle))
            {
                ++stats.growthRejectedNormalCount;
                continue;
            }
            const std::uint16_t candidate_sources =
                weak_source_mask.at<std::uint16_t>(node.row, node.column);
            const std::uint16_t parent_sources =
                geometry_source_mask->at<std::uint16_t>(node.parentRow,
                                                        node.parentColumn);
            if ((candidate_sources & parent_sources) == 0)
            {
                ++stats.growthRejectedSourceOverlapCount;
                continue;
            }

            accepted.at<std::uint8_t>(node.row, node.column) = 255;
            accepted_distance.at<std::uint16_t>(node.row, node.column) =
                static_cast<std::uint16_t>(node.distance);
            reference_depth.at<float>(node.row, node.column) =
                weak_depth.at<float>(node.row, node.column);
            consistent_source_votes->at<std::uint16_t>(node.row, node.column) = 1;
            geometry_source_mask->at<std::uint16_t>(node.row, node.column) =
                candidate_sources;
            source_inverse_depth_sum->at<float>(node.row, node.column) =
                weak_inverse_sum.at<float>(node.row, node.column);
            source_inverse_depth_squared_sum->at<float>(node.row, node.column) =
                weak_inverse_squared_sum.at<float>(node.row, node.column);
            if (has_confidence)
            {
                reference_confidence->at<float>(node.row, node.column) =
                    std::min(options.repairedConfidence, 0.60f);
            }
            if (repaired_mask)
            {
                repaired_mask->at<std::uint8_t>(node.row, node.column) = 255;
            }
            ++stats.twoSourceGrownPixelCount;
            ++stats.repairedPixelCount;

            for (const auto &offset : neighbor_offsets)
            {
                const int neighbor_row = node.row + offset[0];
                const int neighbor_column = node.column + offset[1];
                if (neighbor_row < 0 || neighbor_row >= reference_depth.rows ||
                    neighbor_column < 0 || neighbor_column >= reference_depth.cols ||
                    labels.at<int>(neighbor_row, neighbor_column) != label ||
                    accepted.at<std::uint8_t>(neighbor_row, neighbor_column) != 0)
                {
                    continue;
                }
                const float cost = weak_spread.at<float>(neighbor_row, neighbor_column) *
                    1000.0f + image_gradient.at<float>(neighbor_row, neighbor_column) * 0.01f;
                queue.push({cost,
                            node.distance + 1,
                            neighbor_row,
                            neighbor_column,
                            node.row,
                            node.column});
            }
        }
    }
    interpolate_anchored_components();
    return stats;
}

QJsonObject crossViewHoleRepairStatsToJson(
    const CrossViewHoleRepairStats &stats)
{
    QJsonObject object;
    object.insert(
        QStringLiteral("projected_candidate_count"),
        static_cast<double>(stats.projectedCandidateCount));
    object.insert(
        QStringLiteral("considered_hole_pixel_count"),
        static_cast<double>(stats.consideredHolePixelCount));
    object.insert(
        QStringLiteral("rejected_insufficient_source_count"),
        static_cast<double>(stats.rejectedInsufficientSourceCount));
    object.insert(
        QStringLiteral("rejected_depth_spread_count"),
        static_cast<double>(stats.rejectedDepthSpreadCount));
    object.insert(
        QStringLiteral("rejected_local_depth_count"),
        static_cast<double>(stats.rejectedLocalDepthCount));
    object.insert(
        QStringLiteral("repaired_pixel_count"),
        static_cast<double>(stats.repairedPixelCount));
    object.insert(
        QStringLiteral("two_source_candidate_pixel_count"),
        static_cast<double>(stats.twoSourceCandidatePixelCount));
    object.insert(
        QStringLiteral("two_source_grown_pixel_count"),
        static_cast<double>(stats.twoSourceGrownPixelCount));
    object.insert(
        QStringLiteral("growth_rejected_component_area_count"),
        static_cast<double>(stats.growthRejectedComponentAreaCount));
    object.insert(
        QStringLiteral("growth_rejected_source_overlap_count"),
        static_cast<double>(stats.growthRejectedSourceOverlapCount));
    object.insert(
        QStringLiteral("growth_rejected_normal_count"),
        static_cast<double>(stats.growthRejectedNormalCount));
    object.insert(
        QStringLiteral("growth_rejected_image_edge_count"),
        static_cast<double>(stats.growthRejectedImageEdgeCount));
    object.insert(
        QStringLiteral("native_interpolation_anchor_candidate_count"),
        static_cast<double>(stats.nativeInterpolationAnchorCandidateCount));
    object.insert(
        QStringLiteral("native_interpolation_anchor_accepted_count"),
        static_cast<double>(stats.nativeInterpolationAnchorAcceptedCount));
    object.insert(
        QStringLiteral("native_interpolation_anchor_rejected_count"),
        static_cast<double>(stats.nativeInterpolationAnchorRejectedCount));
    object.insert(
        QStringLiteral("anchored_interpolation"),
        depthAnchoredHoleInterpolationStatsToJson(stats.anchoredInterpolation));
    return object;
}

QJsonObject dominantDepthLayerSelectionStatsToJson(
    const DominantDepthLayerSelectionStats &stats)
{
    return {
        {QStringLiteral("considered_pixel_count"),
         static_cast<double>(stats.consideredPixelCount)},
        {QStringLiteral("stable_layer_pixel_count"),
         static_cast<double>(stats.stableLayerPixelCount)},
        {QStringLiteral("refined_native_pixel_count"),
         static_cast<double>(stats.refinedNativePixelCount)},
        {QStringLiteral("switched_native_pixel_count"),
         static_cast<double>(stats.switchedNativePixelCount)},
        {QStringLiteral("transferred_missing_pixel_count"),
         static_cast<double>(stats.transferredMissingPixelCount)},
        {QStringLiteral("ambiguous_native_pixel_count"),
         static_cast<double>(stats.ambiguousNativePixelCount)},
        {QStringLiteral("unresolved_missing_pixel_count"),
         static_cast<double>(stats.unresolvedMissingPixelCount)},
        {QStringLiteral("reliability_guided_candidate_pixel_count"),
         static_cast<double>(stats.reliabilityGuidedCandidatePixelCount)},
        {QStringLiteral("reliability_guided_stable_pixel_count"),
         static_cast<double>(stats.reliabilityGuidedStablePixelCount)},
        {QStringLiteral("reliability_guided_refined_pixel_count"),
         static_cast<double>(stats.reliabilityGuidedRefinedPixelCount)},
        {QStringLiteral("reliability_guided_switched_pixel_count"),
         static_cast<double>(stats.reliabilityGuidedSwitchedPixelCount)},
        {QStringLiteral("reliability_guided_insufficient_source_pixel_count"),
         static_cast<double>(
             stats.reliabilityGuidedInsufficientSourcePixelCount)},
        {QStringLiteral("geometry_rerank_evaluated_pixel_count"),
         static_cast<double>(stats.geometryRerankEvaluatedPixelCount)},
        {QStringLiteral("geometry_rerank_valid_evidence_pixel_count"),
         static_cast<double>(stats.geometryRerankValidEvidencePixelCount)},
        {QStringLiteral("geometry_rerank_refined_pixel_count"),
         static_cast<double>(stats.geometryRerankRefinedPixelCount)},
        {QStringLiteral("geometry_rerank_switched_pixel_count"),
         static_cast<double>(stats.geometryRerankSwitchedPixelCount)},
        {QStringLiteral("geometry_rerank_rejected_source_count"),
         static_cast<double>(stats.geometryRerankRejectedSourceCount)},
        {QStringLiteral("geometry_rerank_rejected_baseline_count"),
         static_cast<double>(stats.geometryRerankRejectedBaselineCount)},
        {QStringLiteral("geometry_rerank_rejected_weight_count"),
         static_cast<double>(stats.geometryRerankRejectedWeightCount)},
        {QStringLiteral("geometry_rerank_rejected_cost_count"),
         static_cast<double>(stats.geometryRerankRejectedCostCount)},
        {QStringLiteral("geometry_rerank_mean_native_cost"),
         stats.geometryRerankValidEvidencePixelCount > 0
             ? stats.geometryRerankNativeCostSum /
                   static_cast<double>(stats.geometryRerankValidEvidencePixelCount)
             : 0.0},
        {QStringLiteral("geometry_rerank_mean_candidate_cost"),
         stats.geometryRerankValidEvidencePixelCount > 0
             ? stats.geometryRerankCandidateCostSum /
                   static_cast<double>(stats.geometryRerankValidEvidencePixelCount)
             : 0.0},
        {QStringLiteral("geometry_rerank_mean_cost_advantage"),
         stats.geometryRerankValidEvidencePixelCount > 0
             ? stats.geometryRerankCostAdvantageSum /
                   static_cast<double>(stats.geometryRerankValidEvidencePixelCount)
             : 0.0},
        {QStringLiteral("geometry_rerank_mean_relative_correction"),
         stats.geometryRerankValidEvidencePixelCount > 0
             ? stats.geometryRerankCorrectionSum /
                   static_cast<double>(stats.geometryRerankValidEvidencePixelCount)
             : 0.0},
        {QStringLiteral("geometry_rerank_mean_weakest_source_confidence"),
         stats.geometryRerankValidEvidencePixelCount > 0
             ? stats.geometryRerankWeakestConfidenceSum /
                   static_cast<double>(stats.geometryRerankValidEvidencePixelCount)
             : 0.0},
        {QStringLiteral("reliability_guided_only_mode"),
         stats.reliabilityGuidedOnlyMode}
    };
}

} // namespace xjw::mvs
