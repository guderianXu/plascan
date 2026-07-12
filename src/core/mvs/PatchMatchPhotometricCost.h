#pragma once

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

} // namespace mvs
} // namespace xjw

#undef PLASCAN_MVS_HOST_DEVICE
