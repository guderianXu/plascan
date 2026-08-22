#include "DepthGeometryHypothesisReranker.h"

#include "concurrency/SafeWorkerGroup.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace xjw::mvs
{
namespace
{

struct Candidate
{
    float depth = 0.0f;
    float confidence = 0.0f;
    float errorPixels = 0.0f;
    int sourceOrdinal = -1;
    int baselineSector = -1;
};

struct CostResult
{
    float cost = 1.0f;
    float effectiveWeight = 0.0f;
};

bool validDepth(float value)
{
    return std::isfinite(value) && value > 0.0f;
}

float relativeDifference(float first, float second)
{
    return std::fabs(first - second) /
        std::max(1.0e-6f, std::min(first, second));
}

float robustResidual(float hypothesis, float observation, float scale)
{
    const float normalized = relativeDifference(hypothesis, observation) /
        std::max(1.0e-6f, scale);
    const float huber = normalized <= 1.0f
        ? 0.5f * normalized * normalized
        : normalized - 0.5f;
    return std::min(1.0f, 0.5f * huber);
}

float evidenceWeight(const Candidate &candidate, float maximum_error)
{
    if (candidate.confidence <= 0.0f || !std::isfinite(candidate.errorPixels))
    {
        return 0.0f;
    }
    const float normalized_error = candidate.errorPixels /
        std::max(0.25f, maximum_error);
    const float projection_weight = std::exp(
        -0.5f * normalized_error * normalized_error);
    return std::clamp(candidate.confidence, 0.0f, 1.0f) * projection_weight;
}

CostResult hypothesisCost(float hypothesis,
                          std::span<const Candidate> candidates,
                          float scale,
                          float maximum_error)
{
    double weighted_cost = 0.0;
    double weight_sum = 0.0;
    for (const Candidate &candidate : candidates)
    {
        float weight = evidenceWeight(candidate, maximum_error);
        if (weight <= 0.0f)
        {
            continue;
        }
        float residual = robustResidual(hypothesis, candidate.depth, scale);
        if (candidate.depth + candidate.depth * scale < hypothesis)
        {
            // A nearer measured surface contradicts a hypothesis behind it.
            residual = std::min(1.0f, residual * 1.25f);
        }
        else if (hypothesis + hypothesis * scale < candidate.depth)
        {
            // A deeper source observation may be hidden by a foreground layer.
            weight *= 0.70f;
        }
        weighted_cost += static_cast<double>(weight) * residual;
        weight_sum += weight;
    }
    if (weight_sum <= 1.0e-8)
    {
        return {};
    }
    return {static_cast<float>(weighted_cost / weight_sum),
            static_cast<float>(weight_sum)};
}

template <typename Fn>
void parallelRows(int row_count,
                  int worker_count,
                  const std::atomic<bool> *cancelled,
                  Fn &&fn)
{
    const int workers = std::clamp(std::max(1, worker_count), 1, row_count);
    if (workers == 1)
    {
        for (int row = 0; row < row_count; ++row)
        {
            if (cancelled && cancelled->load(std::memory_order_relaxed)) break;
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
            try { fn(row); }
            catch (...) { failure_state.captureCurrentException(); }
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
                if (row >= row_count) break;
                fn(row);
            }
        });
}

std::uint16_t quantizeUnit(float value)
{
    return static_cast<std::uint16_t>(std::lround(
        std::clamp(value, 0.0f, 1.0f) * 65535.0f));
}

float dequantizeUnit(std::uint16_t value)
{
    return static_cast<float>(value) / 65535.0f;
}

std::uint64_t packEvidence(float depth, float confidence, float normalized_error)
{
    const std::uint32_t depth_bits = std::bit_cast<std::uint32_t>(depth);
    return static_cast<std::uint64_t>(depth_bits) |
        (static_cast<std::uint64_t>(quantizeUnit(confidence)) << 32U) |
        (static_cast<std::uint64_t>(quantizeUnit(normalized_error)) << 48U);
}

float packedDepth(std::uint64_t packed)
{
    return std::bit_cast<float>(static_cast<std::uint32_t>(packed));
}

bool betterPackedEvidence(std::uint64_t candidate, std::uint64_t current)
{
    const float candidate_depth = packedDepth(candidate);
    const float current_depth = packedDepth(current);
    if (!validDepth(current_depth)) return true;
    if (candidate_depth != current_depth) return candidate_depth < current_depth;
    const std::uint16_t candidate_error = static_cast<std::uint16_t>(candidate >> 48U);
    const std::uint16_t current_error = static_cast<std::uint16_t>(current >> 48U);
    if (candidate_error != current_error) return candidate_error < current_error;
    const std::uint16_t candidate_confidence =
        static_cast<std::uint16_t>((candidate >> 32U) & 0xffffU);
    const std::uint16_t current_confidence =
        static_cast<std::uint16_t>((current >> 32U) & 0xffffU);
    return candidate_confidence > current_confidence;
}

} // namespace

void DepthGeometryHypothesisRerankMaps::initialize(const cv::Size &size)
{
    nativeCost = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    candidateCost = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    costAdvantage = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    effectiveSourceWeight = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    relativeCorrection = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    weakestSourceConfidence = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    supportingSourceCount = cv::Mat(size, CV_8UC1, cv::Scalar(0));
    baselineSectorCount = cv::Mat(size, CV_8UC1, cv::Scalar(0));
    decisionAction = cv::Mat(size, CV_8UC1, cv::Scalar(0));
}

bool DepthGeometryHypothesisRerankMaps::compatible(const cv::Size &size) const
{
    return nativeCost.type() == CV_32FC1 && nativeCost.size() == size &&
        candidateCost.type() == CV_32FC1 && candidateCost.size() == size &&
        costAdvantage.type() == CV_32FC1 && costAdvantage.size() == size &&
        effectiveSourceWeight.type() == CV_32FC1 && effectiveSourceWeight.size() == size &&
        relativeCorrection.type() == CV_32FC1 && relativeCorrection.size() == size &&
        weakestSourceConfidence.type() == CV_32FC1 && weakestSourceConfidence.size() == size &&
        supportingSourceCount.type() == CV_8UC1 && supportingSourceCount.size() == size &&
        baselineSectorCount.type() == CV_8UC1 && baselineSectorCount.size() == size &&
        decisionAction.type() == CV_8UC1 && decisionAction.size() == size;
}

ProjectedDepthEvidence projectSourceDepthEvidenceToReference(
    const cv::Mat &source_depth,
    const cv::Mat &source_confidence,
    const FramePinholeCamera &source_camera,
    const FramePinholeCamera &reference_camera,
    const cv::Size &reference_size,
    float maximum_projection_distance_pixels,
    int baseline_sector,
    std::uint64_t *projected_candidate_count,
    int row_worker_count,
    const std::atomic<bool> *cancelled)
{
    if (projected_candidate_count) *projected_candidate_count = 0;
    ProjectedDepthEvidence result;
    result.baselineSector = baseline_sector;
    if (source_depth.type() != CV_32FC1 || source_depth.empty() ||
        source_confidence.type() != CV_32FC1 ||
        source_confidence.size() != source_depth.size() ||
        !source_camera.isValid() || !reference_camera.isValid() ||
        reference_size.width <= 0 || reference_size.height <= 0)
    {
        return result;
    }

    const float maximum_distance = std::clamp(
        maximum_projection_distance_pixels, 0.25f, 1.5f);
    cv::Mat packed(reference_size, CV_64FC1, cv::Scalar(0.0));
    std::atomic<std::uint64_t> candidate_count{0};
    parallelRows(source_depth.rows, row_worker_count, cancelled,
        [&](int source_row)
        {
            std::uint64_t row_count = 0;
            const float *depth_values = source_depth.ptr<float>(source_row);
            const float *confidence_values = source_confidence.ptr<float>(source_row);
            for (int source_column = 0; source_column < source_depth.cols; ++source_column)
            {
                if ((source_column & 63) == 0 && cancelled &&
                    cancelled->load(std::memory_order_relaxed)) break;
                const float source_value = depth_values[source_column];
                if (!validDepth(source_value)) continue;
                const double source_pixel[2] = {
                    static_cast<double>(source_column), static_cast<double>(source_row)};
                double world[3] = {};
                if (!source_camera.unprojectPixel(source_pixel, source_value, world)) continue;
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
                        if (column < 0 || column >= reference_size.width ||
                            row < 0 || row >= reference_size.height) continue;
                        const float offset_x = static_cast<float>(reference_pixel[0] - column);
                        const float offset_y = static_cast<float>(reference_pixel[1] - row);
                        const float error = std::sqrt(offset_x * offset_x + offset_y * offset_y);
                        if (error > maximum_distance) continue;
                        const std::uint64_t candidate = packEvidence(
                            static_cast<float>(reference_value),
                            confidence_values[source_column],
                            error / maximum_distance);
                        std::uint64_t &stored_value = packed.ptr<std::uint64_t>(row)[column];
                        std::atomic_ref<std::uint64_t> stored(stored_value);
                        std::uint64_t observed = stored.load(std::memory_order_relaxed);
                        while (betterPackedEvidence(candidate, observed) &&
                               !stored.compare_exchange_weak(
                                   observed, candidate,
                                   std::memory_order_relaxed,
                                   std::memory_order_relaxed))
                        {
                        }
                        ++row_count;
                    }
                }
            }
            candidate_count.fetch_add(row_count, std::memory_order_relaxed);
        });

    result.depth = cv::Mat(reference_size, CV_32FC1, cv::Scalar(0.0f));
    result.confidence = cv::Mat(reference_size, CV_32FC1, cv::Scalar(0.0f));
    result.reprojectionErrorPixels = cv::Mat(reference_size, CV_32FC1, cv::Scalar(0.0f));
    parallelRows(reference_size.height, row_worker_count, cancelled,
        [&](int row)
        {
            const std::uint64_t *packed_row = packed.ptr<std::uint64_t>(row);
            float *depth_row = result.depth.ptr<float>(row);
            float *confidence_row = result.confidence.ptr<float>(row);
            float *error_row = result.reprojectionErrorPixels.ptr<float>(row);
            for (int column = 0; column < reference_size.width; ++column)
            {
                const std::uint64_t value = packed_row[column];
                depth_row[column] = packedDepth(value);
                confidence_row[column] = dequantizeUnit(
                    static_cast<std::uint16_t>((value >> 32U) & 0xffffU));
                error_row[column] = maximum_distance * dequantizeUnit(
                    static_cast<std::uint16_t>(value >> 48U));
            }
        });
    if (projected_candidate_count)
    {
        *projected_candidate_count = candidate_count.load(std::memory_order_relaxed);
    }
    return result;
}

DepthGeometryHypothesisDecision scoreMeasuredDepthHypothesis(
    float hypothesis_depth,
    int row,
    int column,
    std::span<const ProjectedDepthEvidence> projected_evidence,
    const DepthGeometryHypothesisRerankOptions &options)
{
    DepthGeometryHypothesisDecision result;
    result.selectedDepth = hypothesis_depth;
    if (!validDepth(hypothesis_depth)) return result;

    std::vector<Candidate> candidates;
    candidates.reserve(projected_evidence.size());
    for (int ordinal = 0; ordinal < static_cast<int>(projected_evidence.size()); ++ordinal)
    {
        const ProjectedDepthEvidence &evidence = projected_evidence[ordinal];
        if (evidence.depth.type() != CV_32FC1 ||
            evidence.confidence.type() != CV_32FC1 ||
            evidence.reprojectionErrorPixels.type() != CV_32FC1 ||
            evidence.confidence.size() != evidence.depth.size() ||
            evidence.reprojectionErrorPixels.size() != evidence.depth.size() ||
            row < 0 || row >= evidence.depth.rows ||
            column < 0 || column >= evidence.depth.cols)
        {
            continue;
        }
        const float depth = evidence.depth.at<float>(row, column);
        const float confidence = evidence.confidence.at<float>(row, column);
        const float error = evidence.reprojectionErrorPixels.at<float>(row, column);
        if (!validDepth(depth) || !std::isfinite(confidence) ||
            confidence < options.minimumProjectedConfidence || !std::isfinite(error))
        {
            continue;
        }
        candidates.push_back({depth, confidence, error, ordinal, evidence.baselineSector});
    }
    if (candidates.empty())
    {
        result.rejectedInsufficientSources = true;
        return result;
    }
    const CostResult cost = hypothesisCost(
        hypothesis_depth, candidates, options.maximumRelativeDepthSpread, 1.0f);
    result.candidateCost = cost.cost;
    result.effectiveSourceWeight = 0.0f;
    result.weakestSourceConfidence = 1.0f;
    std::uint8_t sector_mask = 0;
    for (const Candidate &candidate : candidates)
    {
        if (relativeDifference(hypothesis_depth, candidate.depth) >
            options.maximumRelativeDepthSpread)
        {
            continue;
        }
        ++result.supportingSourceCount;
        result.effectiveSourceWeight += evidenceWeight(candidate, 1.0f);
        result.weakestSourceConfidence = std::min(
            result.weakestSourceConfidence, candidate.confidence);
        if (candidate.baselineSector >= 0 && candidate.baselineSector < 6)
        {
            sector_mask = static_cast<std::uint8_t>(
                sector_mask | (1U << candidate.baselineSector));
        }
        if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
        {
            result.supportingSourceMask = static_cast<std::uint16_t>(
                result.supportingSourceMask | (1U << candidate.sourceOrdinal));
        }
        const float inverse = 1.0f / candidate.depth;
        result.supportingInverseDepthSum += inverse;
        result.supportingInverseDepthSquaredSum += inverse * inverse;
    }
    result.baselineSectorCount = std::popcount(sector_mask);
    result.validEvidence = true;
    result.rejectedInsufficientSources =
        result.supportingSourceCount < options.minimumDistinctSourceCount;
    result.rejectedInsufficientBaseline =
        result.baselineSectorCount < options.minimumBaselineSectorCount;
    result.rejectedInsufficientWeight =
        result.effectiveSourceWeight < options.minimumEffectiveSourceWeight;
    if (result.supportingSourceCount == 0)
    {
        result.weakestSourceConfidence = 0.0f;
    }
    return result;
}

DepthGeometryHypothesisDecision rerankMeasuredDepthHypothesis(
    float native_depth,
    DepthLayerReliabilityClass reliability_class,
    int row,
    int column,
    std::span<const ProjectedDepthEvidence> projected_evidence,
    const DepthGeometryHypothesisRerankOptions &options)
{
    DepthGeometryHypothesisDecision result;
    if (!validDepth(native_depth) ||
        (reliability_class != DepthLayerReliabilityClass::AmbiguousLowTexture &&
         reliability_class != DepthLayerReliabilityClass::RejectedLayer))
    {
        return result;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(projected_evidence.size());
    constexpr float maximum_error = 1.0f;
    for (int ordinal = 0; ordinal < static_cast<int>(projected_evidence.size()); ++ordinal)
    {
        const ProjectedDepthEvidence &evidence = projected_evidence[ordinal];
        if (evidence.depth.type() != CV_32FC1 ||
            evidence.confidence.type() != CV_32FC1 ||
            evidence.reprojectionErrorPixels.type() != CV_32FC1 ||
            row < 0 || row >= evidence.depth.rows ||
            column < 0 || column >= evidence.depth.cols ||
            evidence.confidence.size() != evidence.depth.size() ||
            evidence.reprojectionErrorPixels.size() != evidence.depth.size())
        {
            continue;
        }
        const float depth = evidence.depth.at<float>(row, column);
        const float confidence = evidence.confidence.at<float>(row, column);
        const float error = evidence.reprojectionErrorPixels.at<float>(row, column);
        if (!validDepth(depth) || !std::isfinite(confidence) ||
            confidence < options.minimumProjectedConfidence || !std::isfinite(error))
        {
            continue;
        }
        candidates.push_back({depth, confidence, error, ordinal, evidence.baselineSector});
    }
    if (static_cast<int>(candidates.size()) < options.minimumDistinctSourceCount)
    {
        result.rejectedInsufficientSources = true;
        return result;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate &left, const Candidate &right)
              {
                  if (left.depth != right.depth) return left.depth < right.depth;
                  return left.sourceOrdinal < right.sourceOrdinal;
              });
    const CostResult native_cost = hypothesisCost(
        native_depth, candidates, options.maximumRelativeDepthSpread, maximum_error);

    float best_cost = std::numeric_limits<float>::infinity();
    float best_depth = 0.0f;
    std::vector<int> best_support;
    for (const Candidate &hypothesis : candidates)
    {
        std::vector<int> support;
        for (int index = 0; index < static_cast<int>(candidates.size()); ++index)
        {
            if (relativeDifference(hypothesis.depth, candidates[index].depth) <=
                options.maximumRelativeDepthSpread)
            {
                support.push_back(index);
            }
        }
        if (static_cast<int>(support.size()) < options.minimumDistinctSourceCount) continue;
        double inverse_sum = 0.0;
        double weight_sum = 0.0;
        for (int index : support)
        {
            const float weight = evidenceWeight(candidates[index], maximum_error);
            inverse_sum += weight / candidates[index].depth;
            weight_sum += weight;
        }
        if (weight_sum <= 1.0e-8) continue;
        const float candidate_depth = static_cast<float>(weight_sum / inverse_sum);
        const CostResult cost = hypothesisCost(
            candidate_depth, candidates, options.maximumRelativeDepthSpread, maximum_error);
        if (cost.cost < best_cost ||
            (cost.cost == best_cost &&
             relativeDifference(candidate_depth, native_depth) <
                 relativeDifference(best_depth, native_depth)))
        {
            best_cost = cost.cost;
            best_depth = candidate_depth;
            best_support = std::move(support);
        }
    }
    if (best_support.empty())
    {
        result.rejectedInsufficientSources = true;
        return result;
    }

    std::uint8_t sector_mask = 0;
    float weakest_confidence = 1.0f;
    double effective_weight = 0.0;
    for (int index : best_support)
    {
        const Candidate &candidate = candidates[index];
        if (candidate.baselineSector >= 0 && candidate.baselineSector < 6)
        {
            sector_mask = static_cast<std::uint8_t>(
                sector_mask | (1U << candidate.baselineSector));
        }
        const float weight = evidenceWeight(candidate, maximum_error);
        effective_weight += weight;
        weakest_confidence = std::min(weakest_confidence, candidate.confidence);
        if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
        {
            result.supportingSourceMask = static_cast<std::uint16_t>(
                result.supportingSourceMask | (1U << candidate.sourceOrdinal));
        }
        const float inverse = 1.0f / candidate.depth;
        result.supportingInverseDepthSum += inverse;
        result.supportingInverseDepthSquaredSum += inverse * inverse;
    }
    result.supportingSourceCount = static_cast<int>(best_support.size());
    result.baselineSectorCount = std::popcount(sector_mask);
    result.effectiveSourceWeight = static_cast<float>(effective_weight);
    result.weakestSourceConfidence = weakest_confidence;
    result.nativeCost = native_cost.cost;
    result.candidateCost = best_cost;
    result.costAdvantage = native_cost.cost - best_cost;
    result.relativeCorrection = relativeDifference(native_depth, best_depth);
    result.selectedDepth = best_depth;
    result.validEvidence = true;

    if (result.baselineSectorCount < options.minimumBaselineSectorCount)
    {
        result.rejectedInsufficientBaseline = true;
        return result;
    }
    if (result.effectiveSourceWeight < options.minimumEffectiveSourceWeight)
    {
        result.rejectedInsufficientWeight = true;
        return result;
    }

    const bool ambiguous =
        reliability_class == DepthLayerReliabilityClass::AmbiguousLowTexture;
    const float required_advantage = ambiguous
        ? options.minimumRefinementCostAdvantage
        : (result.relativeCorrection <= options.rejectedMaximumRelativeRefinement
               ? options.minimumRefinementCostAdvantage
               : options.minimumLayerSwitchCostAdvantage);
    if (result.costAdvantage < required_advantage)
    {
        result.rejectedCostAdvantage = true;
        return result;
    }
    if (ambiguous &&
        result.relativeCorrection > options.ambiguousMaximumRelativeCorrection)
    {
        // Ambiguous evidence may refine a mode but never authorizes a layer jump.
        result.rejectedCostAdvantage = true;
        return result;
    }

    if (ambiguous ||
        result.relativeCorrection <= options.rejectedMaximumRelativeRefinement)
    {
        const float blend = std::clamp(options.refinementBlendWeight, 0.0f, 1.0f);
        const float native_inverse = 1.0f / native_depth;
        const float candidate_inverse = 1.0f / best_depth;
        result.selectedDepth = 1.0f /
            ((1.0f - blend) * native_inverse + blend * candidate_inverse);
        const float maximum_relative = ambiguous
            ? options.ambiguousMaximumRelativeCorrection
            : options.rejectedMaximumRelativeRefinement;
        const float maximum_delta = native_depth * maximum_relative;
        result.selectedDepth = std::clamp(
            result.selectedDepth,
            native_depth - maximum_delta,
            native_depth + maximum_delta);
        result.action = DepthGeometryHypothesisAction::Refine;
    }
    else
    {
        result.action = DepthGeometryHypothesisAction::SwitchLayer;
    }
    const float normalized_advantage = std::clamp(
        result.costAdvantage / std::max(0.05f, required_advantage * 2.0f),
        0.0f,
        1.0f);
    const float normalized_weight = std::clamp(
        result.effectiveSourceWeight /
            std::max(1.0f, static_cast<float>(result.supportingSourceCount)),
        0.0f,
        1.0f);
    result.evidenceConfidence = std::clamp(
        0.20f + 0.35f * normalized_advantage +
            0.25f * normalized_weight + 0.20f * weakest_confidence,
        0.0f,
        0.90f);
    return result;
}

} // namespace xjw::mvs
