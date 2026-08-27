#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define PLASCAN_MVS_HOST_DEVICE __host__ __device__
#else
#define PLASCAN_MVS_HOST_DEVICE
#endif

namespace xjw
{
namespace mvs
{

constexpr int kMaxPatchMatchSourceViews = 32;
constexpr float kDefaultMinimumMaskedPatchSupportRatio = 0.35f;
constexpr float kDefaultPatchMatchHintRadiusRatio = 0.05f;
constexpr float kDefaultSourceSelectionNeighborBonus = 0.04f;
constexpr int kMaximumAdaptivePhotometricSupport = 4;
constexpr float kAdaptivePhotometricInlierScoreWindow = 0.25f;

/// Keep CUDA and OpenCL on the same deterministic search budget. The public
/// numIterations value controls both the coarse inverse-depth sweep and a
/// smaller number of expensive plane-propagation passes.
constexpr int patchMatchDepthSampleCount(int num_iterations)
{
    const int samples = num_iterations * 4;
    return samples < 32 ? 32 : (samples > 96 ? 96 : samples);
}

constexpr int patchMatchPropagationIterationCount(int num_iterations)
{
    const int iterations = (num_iterations + 3) / 4;
    return iterations < 2 ? 2 : (iterations > 4 ? 4 : iterations);
}

constexpr int patchMatchScheduledPropagationPassCount(int num_iterations,
                                                      bool enable_final_pass)
{
    return patchMatchPropagationIterationCount(num_iterations) +
        (enable_final_pass ? 1 : 0);
}

struct PatchNccAccumulator
{
    float sumReference = 0.0f;
    float sumSource = 0.0f;
    float sumReferenceSquared = 0.0f;
    float sumSourceSquared = 0.0f;
    float sumReferenceSource = 0.0f;
    int candidateCount = 0;
    int validCount = 0;

    PLASCAN_MVS_HOST_DEVICE void addCandidate(bool valid,
                                               float reference_value = 0.0f,
                                               float source_value = 0.0f)
    {
        ++candidateCount;
        if (!valid)
        {
            return;
        }

        sumReference += reference_value;
        sumSource += source_value;
        sumReferenceSquared += reference_value * reference_value;
        sumSourceSquared += source_value * source_value;
        sumReferenceSource += reference_value * source_value;
        ++validCount;
    }

    PLASCAN_MVS_HOST_DEVICE float score(bool mask_aware,
                                         float minimum_support_ratio =
                                             kDefaultMinimumMaskedPatchSupportRatio) const
    {
        int required_count = 4;
        if (mask_aware && candidateCount > 0)
        {
            const float bounded_ratio = minimum_support_ratio < 0.0f
                ? 0.0f
                : (minimum_support_ratio > 1.0f ? 1.0f : minimum_support_ratio);
            const int ratio_count = static_cast<int>(
                static_cast<float>(candidateCount) * bounded_ratio + 0.999999f);
            required_count = ratio_count > required_count ? ratio_count : required_count;
        }
        if (validCount < required_count)
        {
            return 0.0f;
        }

        const float inverse_count = 1.0f / static_cast<float>(validCount);
        const float mean_reference = sumReference * inverse_count;
        const float mean_source = sumSource * inverse_count;
        float variance_reference =
            sumReferenceSquared * inverse_count - mean_reference * mean_reference;
        float variance_source =
            sumSourceSquared * inverse_count - mean_source * mean_source;
        const float covariance =
            sumReferenceSource * inverse_count - mean_reference * mean_source;
        variance_reference = variance_reference > 0.0f ? variance_reference : 0.0f;
        variance_source = variance_source > 0.0f ? variance_source : 0.0f;
#if defined(__CUDA_ARCH__)
        const float denominator = sqrtf(variance_reference * variance_source);
#else
        const float denominator = std::sqrt(variance_reference * variance_source);
#endif
        if (denominator < 1e-5f)
        {
            return 0.0f;
        }
        return ((covariance / denominator) + 1.0f) * 0.5f;
    }

    PLASCAN_MVS_HOST_DEVICE bool hasSupport(
        bool mask_aware,
        float minimum_support_ratio =
            kDefaultMinimumMaskedPatchSupportRatio) const
    {
        int required_count = 4;
        if (mask_aware && candidateCount > 0)
        {
            const float bounded_ratio = minimum_support_ratio < 0.0f
                ? 0.0f
                : (minimum_support_ratio > 1.0f ? 1.0f : minimum_support_ratio);
            const int ratio_count = static_cast<int>(
                static_cast<float>(candidateCount) * bounded_ratio + 0.999999f);
            required_count = ratio_count > required_count ? ratio_count : required_count;
        }
        return validCount >= required_count;
    }
};

/// Exposure-robust patch evidence shared by CPU and CUDA PatchMatch.
/// Intensity NCC handles affine exposure changes, gradient NCC preserves
/// edges under low-frequency illumination changes, and ternary Census keeps
/// monotonic local ordering when highlights or response curves differ.
struct PatchRobustPhotometricAccumulator
{
    PatchNccAccumulator intensity;
    PatchNccAccumulator gradient;
    int censusCandidateCount = 0;
    int censusValidCount = 0;
    int censusAgreementCount = 0;

    PLASCAN_MVS_HOST_DEVICE void addIntensityCandidate(
        bool valid,
        float reference_value = 0.0f,
        float source_value = 0.0f)
    {
        intensity.addCandidate(valid, reference_value, source_value);
    }

    PLASCAN_MVS_HOST_DEVICE void addGradientCandidate(
        bool valid,
        float reference_x = 0.0f,
        float reference_y = 0.0f,
        float source_x = 0.0f,
        float source_y = 0.0f)
    {
        gradient.addCandidate(valid, reference_x, source_x);
        gradient.addCandidate(valid, reference_y, source_y);
    }

    PLASCAN_MVS_HOST_DEVICE void addCensusCandidate(
        bool valid,
        float reference_delta = 0.0f,
        float source_delta = 0.0f,
        float contrast_threshold = 0.01f)
    {
        ++censusCandidateCount;
        if (!valid)
        {
            return;
        }
        ++censusValidCount;
        const int reference_rank = reference_delta > contrast_threshold
            ? 1
            : (reference_delta < -contrast_threshold ? -1 : 0);
        const int source_rank = source_delta > contrast_threshold
            ? 1
            : (source_delta < -contrast_threshold ? -1 : 0);
        if (reference_rank == source_rank)
        {
            ++censusAgreementCount;
        }
    }

    PLASCAN_MVS_HOST_DEVICE float score(
        bool mask_aware,
        float minimum_support_ratio =
            kDefaultMinimumMaskedPatchSupportRatio) const
    {
        if (!intensity.hasSupport(mask_aware, minimum_support_ratio))
        {
            return 0.0f;
        }

        float weighted_score = 0.50f * intensity.score(
            mask_aware, minimum_support_ratio);
        float weight_sum = 0.50f;
        if (gradient.hasSupport(mask_aware, minimum_support_ratio))
        {
            weighted_score += 0.30f * gradient.score(
                mask_aware, minimum_support_ratio);
            weight_sum += 0.30f;
        }

        int census_required = 4;
        if (mask_aware && censusCandidateCount > 0)
        {
            const int ratio_count = static_cast<int>(
                static_cast<float>(censusCandidateCount)
                    * minimum_support_ratio
                + 0.999999f);
            census_required = ratio_count > census_required
                ? ratio_count
                : census_required;
        }
        if (censusValidCount >= census_required)
        {
            weighted_score += 0.20f
                * static_cast<float>(censusAgreementCount)
                / static_cast<float>(censusValidCount);
            weight_sum += 0.20f;
        }
        return weight_sum > 0.0f ? weighted_score / weight_sum : 0.0f;
    }
};

/// Map the conservative composite score onto a probability-like confidence
/// scale. Gradient and Census evidence lower the raw score even for correct
/// depths, so reusing NCC's identity mapping made the new cost systematically
/// under-confident. This monotonic calibration preserves its ranking.
PLASCAN_MVS_HOST_DEVICE inline float calibrateRobustPhotometricConfidence(
    float score)
{
    const float bounded_score = score < 0.0f
        ? 0.0f
        : (score > 1.0f ? 1.0f : score);
    if (!(bounded_score > 0.0f))
    {
        return 0.0f;
    }
#if defined(__CUDA_ARCH__)
    return powf(bounded_score, 0.20f);
#else
    return std::pow(bounded_score, 0.20f);
#endif
}

PLASCAN_MVS_HOST_DEVICE inline int requiredPhotometricSupport(int source_count)
{
    if (source_count <= 0)
    {
        return 0;
    }
    if (source_count <= 2)
    {
        return source_count;
    }
    if (source_count == 3)
    {
        return 2;
    }
    return 3;
}

/// Aggregate NCC scores while requiring genuine multi-view support.
///
/// Source selection currently uses at most a handful of views. The fixed local
/// array keeps this helper allocation-free inside CUDA kernels and also lets the
/// CPU fallback share exactly the same acceptance rule.
struct JointViewSelection
{
    float photometricScore = 0.0f;
    std::uint32_t sourceMask = 0;
    int sourceCount = 0;
};

/// Select a compact, mutually plausible source set for one pixel/hypothesis.
///
/// The neighbour mask is only a bounded ranking prior. The returned score is
/// always averaged from the original NCC observations, so spatial agreement
/// cannot manufacture photometric confidence. This keeps the bitset useful for
/// PatchMatch propagation without confusing it with independent geometry. A
/// fixed majority couples occlusion tolerance to the global candidate-pool
/// size; instead, require two sources for a three-view pool and three sources
/// for larger pools, reject scores far below the strongest observation, and
/// average at most four inliers.
PLASCAN_MVS_HOST_DEVICE inline JointViewSelection selectJointSourceViews(
    const float *scores,
    int source_count,
    std::uint32_t neighbor_mask = 0,
    float neighbor_bonus = kDefaultSourceSelectionNeighborBonus,
    float minimum_ncc = 0.05f)
{
    JointViewSelection result;
    if (scores == nullptr || source_count <= 0)
    {
        return result;
    }

    const int effective_count = source_count < kMaxPatchMatchSourceViews
                                    ? source_count
                                    : kMaxPatchMatchSourceViews;
    const int required_support = requiredPhotometricSupport(effective_count);
    const int maximum_support = effective_count < kMaximumAdaptivePhotometricSupport
                                    ? effective_count
                                    : kMaximumAdaptivePhotometricSupport;
    float strongest[kMaxPatchMatchSourceViews] = {};
    float strongest_rank[kMaxPatchMatchSourceViews] = {};
    int strongest_index[kMaxPatchMatchSourceViews] = {};
    int support_count = 0;
    int stored_count = 0;

    float best_score = 0.0f;
    for (int source_index = 0; source_index < effective_count; ++source_index)
    {
        const float score = scores[source_index];
        if (score > minimum_ncc && score > best_score)
        {
            best_score = score;
        }
    }
    if (!(best_score > minimum_ncc))
    {
        return result;
    }
    const float relative_inlier_floor =
        best_score - kAdaptivePhotometricInlierScoreWindow;
    const float inlier_floor = relative_inlier_floor > minimum_ncc
        ? relative_inlier_floor
        : minimum_ncc;

    for (int source_index = 0; source_index < effective_count; ++source_index)
    {
        const float score = scores[source_index];
        if (!(score > minimum_ncc) || score < inlier_floor)
        {
            continue;
        }

        ++support_count;
        const float rank_score = score +
            (((neighbor_mask >> source_index) & 1u) != 0u
                 ? (neighbor_bonus > 0.0f ? neighbor_bonus : 0.0f)
                 : 0.0f);
        if (stored_count == maximum_support &&
            (rank_score < strongest_rank[maximum_support - 1] ||
             (rank_score == strongest_rank[maximum_support - 1] &&
              source_index > strongest_index[maximum_support - 1])))
        {
            continue;
        }
        int insert_at = stored_count < maximum_support ? stored_count : maximum_support - 1;
        while (insert_at > 0 &&
               (strongest_rank[insert_at - 1] < rank_score ||
                (strongest_rank[insert_at - 1] == rank_score &&
                 strongest_index[insert_at - 1] > source_index)))
        {
            if (insert_at < maximum_support)
            {
                strongest[insert_at] = strongest[insert_at - 1];
                strongest_rank[insert_at] = strongest_rank[insert_at - 1];
                strongest_index[insert_at] = strongest_index[insert_at - 1];
            }
            --insert_at;
        }
        if (insert_at < maximum_support)
        {
            strongest[insert_at] = score;
            strongest_rank[insert_at] = rank_score;
            strongest_index[insert_at] = source_index;
            if (stored_count < maximum_support)
            {
                ++stored_count;
            }
        }
    }

    if (support_count < required_support || stored_count < required_support)
    {
        return result;
    }

    float score_sum = 0.0f;
    for (int support_index = 0; support_index < stored_count; ++support_index)
    {
        score_sum += strongest[support_index];
        result.sourceMask |= 1u << strongest_index[support_index];
    }
    const float average_score = score_sum / static_cast<float>(stored_count);
    result.photometricScore = average_score < 0.0f
        ? 0.0f
        : (average_score > 1.0f ? 1.0f : average_score);
    result.sourceCount = stored_count;
    return result;
}

PLASCAN_MVS_HOST_DEVICE inline float robustMultiSourceNcc(const float *scores,
                                                          int source_count,
                                                          float minimum_ncc = 0.05f)
{
    return selectJointSourceViews(scores, source_count, 0, 0.0f, minimum_ncc)
        .photometricScore;
}

PLASCAN_MVS_HOST_DEVICE inline int selectedSourceCount(std::uint32_t source_mask)
{
    int count = 0;
    while (source_mask != 0u)
    {
        source_mask &= source_mask - 1u;
        ++count;
    }
    return count;
}

/// Convert a local depth-cost uniqueness margin into a soft confidence scale.
///
/// AliceVision's SGM path compares the best and second-best depth-volume
/// candidates. PatchMatch does not keep a full volume, so PlaScan probes the
/// converged solution at neighbouring depths and applies the same principle:
/// a visually plausible hypothesis is not reliable when a distinct depth has
/// nearly the same score. Keeping this as a soft scale lets later cross-view
/// repair recover coverage without allowing ambiguous native layers to carry
/// full TSDF weight.
PLASCAN_MVS_HOST_DEVICE inline float photometricUniquenessConfidenceScale(
    float best_ncc,
    float competing_ncc,
    float minimum_margin,
    float minimum_scale)
{
    const float bounded_minimum_scale = minimum_scale < 0.0f
        ? 0.0f
        : (minimum_scale > 1.0f ? 1.0f : minimum_scale);
    if (!(minimum_margin > 0.0f))
    {
        return 1.0f;
    }

    const float margin = best_ncc > competing_ncc
        ? best_ncc - competing_ncc
        : 0.0f;
    const float normalized_margin = margin >= minimum_margin
        ? 1.0f
        : margin / minimum_margin;
    return bounded_minimum_scale
        + (1.0f - bounded_minimum_scale) * normalized_margin;
}

} // namespace mvs
} // namespace xjw

#undef PLASCAN_MVS_HOST_DEVICE
