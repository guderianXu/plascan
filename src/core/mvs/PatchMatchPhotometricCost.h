#pragma once

#include <cmath>

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
};

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
    return source_count / 2 + 1;
}

/// Aggregate NCC scores while requiring genuine multi-view support.
///
/// Source selection currently uses at most a handful of views. The fixed local
/// array keeps this helper allocation-free inside CUDA kernels and also lets the
/// CPU fallback share exactly the same acceptance rule.
PLASCAN_MVS_HOST_DEVICE inline float robustMultiSourceNcc(const float *scores,
                                                          int source_count,
                                                          float minimum_ncc = 0.05f)
{
    if (scores == nullptr || source_count <= 0)
    {
        return 0.0f;
    }

    const int effective_count = source_count < kMaxPatchMatchSourceViews
                                    ? source_count
                                    : kMaxPatchMatchSourceViews;
    const int required_support = requiredPhotometricSupport(effective_count);
    float strongest[kMaxPatchMatchSourceViews] = {};
    int support_count = 0;
    int stored_count = 0;

    for (int source_index = 0; source_index < effective_count; ++source_index)
    {
        const float score = scores[source_index];
        if (!(score > minimum_ncc))
        {
            continue;
        }

        ++support_count;
        if (stored_count == required_support
            && score <= strongest[required_support - 1])
        {
            continue;
        }
        int insert_at = stored_count < required_support ? stored_count : required_support - 1;
        while (insert_at > 0 && strongest[insert_at - 1] < score)
        {
            if (insert_at < required_support)
            {
                strongest[insert_at] = strongest[insert_at - 1];
            }
            --insert_at;
        }
        if (insert_at < required_support)
        {
            strongest[insert_at] = score;
            if (stored_count < required_support)
            {
                ++stored_count;
            }
        }
    }

    if (support_count < required_support || stored_count < required_support)
    {
        return 0.0f;
    }

    float score_sum = 0.0f;
    for (int support_index = 0; support_index < required_support; ++support_index)
    {
        score_sum += strongest[support_index];
    }
    return score_sum / static_cast<float>(required_support);
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
