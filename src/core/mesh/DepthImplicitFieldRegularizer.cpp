#include "DepthImplicitFieldRegularizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef MESHING_OPENMP
#include <omp.h>
#endif

namespace xjw::mesh
{
namespace
{

std::size_t sampleIndex(const std::array<int, 3> &dimensions,
                        int x,
                        int y,
                        int z)
{
    return static_cast<std::size_t>(z) *
               static_cast<std::size_t>(dimensions[0]) *
               static_cast<std::size_t>(dimensions[1]) +
        static_cast<std::size_t>(y) *
            static_cast<std::size_t>(dimensions[0]) +
        static_cast<std::size_t>(x);
}

bool signsMatch(float lhs, float rhs)
{
    return (lhs < 0.0f) == (rhs < 0.0f);
}

} // namespace

DepthImplicitFieldRegularizationStatistics
DepthImplicitFieldRegularizer::regularize(
    const std::array<int, 3> &sampleDimensions,
    const std::vector<float> &surfaceEvidenceField,
    const std::vector<float> &observationWeight,
    const std::vector<std::uint16_t> &geometrySourceMask,
    const std::vector<std::uint8_t> &eligible,
    const DepthImplicitFieldRegularizationOptions &options,
    std::vector<float> *field,
    std::vector<std::uint8_t> *supported,
    const std::function<bool()> &isCancelled)
{
    DepthImplicitFieldRegularizationStatistics statistics;
    const auto start = std::chrono::steady_clock::now();
    const auto finish = [&statistics, &start]()
    {
        statistics.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        return statistics;
    };
    if (!field || !supported)
    {
        throw std::invalid_argument(
            "DepthImplicitFieldRegularizer requires output field and support arrays");
    }
    if (sampleDimensions[0] < 3 ||
        sampleDimensions[1] < 3 ||
        sampleDimensions[2] < 3)
    {
        return finish();
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(sampleDimensions[0]) *
        static_cast<std::size_t>(sampleDimensions[1]) *
        static_cast<std::size_t>(sampleDimensions[2]);
    if (field->size() != expected_size ||
        supported->size() != expected_size ||
        surfaceEvidenceField.size() != expected_size ||
        observationWeight.size() != expected_size ||
        geometrySourceMask.size() != expected_size ||
        eligible.size() != expected_size)
    {
        throw std::invalid_argument(
            "DepthImplicitFieldRegularizer input array sizes do not match dimensions");
    }

    statistics.executed = true;
    if (isCancelled && isCancelled())
    {
        statistics.cancelled = true;
        return finish();
    }

    if (options.recoverAxialGaps)
    {
        const std::vector<std::uint8_t> support_snapshot = *supported;
        std::vector<std::size_t> recovered_indices;
        for (int z = 1; z < sampleDimensions[2] - 1; ++z)
        {
            for (int y = 1; y < sampleDimensions[1] - 1; ++y)
            {
                for (int x = 1; x < sampleDimensions[0] - 1; ++x)
                {
                    const std::size_t index =
                        sampleIndex(sampleDimensions, x, y, z);
                    if (support_snapshot[index] != 0 || eligible[index] == 0)
                    {
                        continue;
                    }
                    ++statistics.bridgeCandidateCount;
                    const std::array<std::array<int, 3>, 3> axes = {
                        std::array<int, 3>{1, 0, 0},
                        std::array<int, 3>{0, 1, 0},
                        std::array<int, 3>{0, 0, 1}};
                    int agreeing_axes = 0;
                    float prediction_sum = 0.0f;
                    for (const auto &axis : axes)
                    {
                        const std::size_t negative_index = sampleIndex(
                            sampleDimensions,
                            x - axis[0],
                            y - axis[1],
                            z - axis[2]);
                        const std::size_t positive_index = sampleIndex(
                            sampleDimensions,
                            x + axis[0],
                            y + axis[1],
                            z + axis[2]);
                        if (support_snapshot[negative_index] == 0 ||
                            support_snapshot[positive_index] == 0)
                        {
                            continue;
                        }
                        const std::uint16_t common_sources =
                            geometrySourceMask[index] &
                            geometrySourceMask[negative_index] &
                            geometrySourceMask[positive_index];
                        if (common_sources == 0)
                        {
                            continue;
                        }
                        const float prediction =
                            ((*field)[negative_index] +
                             (*field)[positive_index]) *
                            0.5f;
                        if (!std::isfinite(prediction) ||
                            std::fabs(surfaceEvidenceField[index] - prediction) >
                                std::max(
                                    0.01f,
                                    options.maximumBridgePredictionDelta))
                        {
                            continue;
                        }
                        ++agreeing_axes;
                        prediction_sum += prediction;
                    }
                    if (agreeing_axes <
                        std::clamp(options.minimumBridgeAxes, 1, 3))
                    {
                        continue;
                    }
                    (*field)[index] = std::clamp(
                        prediction_sum / static_cast<float>(agreeing_axes),
                        -1.0f,
                        1.0f);
                    recovered_indices.push_back(index);
                }
            }
        }
        for (const std::size_t index : recovered_indices)
        {
            (*supported)[index] = 1;
        }
        statistics.recoveredSampleCount = recovered_indices.size();
    }

    const int level_count = std::clamp(options.coarseToFineLevels, 1, 3);
    const int passes_per_level = std::clamp(options.passesPerLevel, 1, 4);
    const float smoothness = std::clamp(options.smoothness, 0.0f, 2.0f);
    const float data_fidelity = std::max(0.01f, options.dataFidelity);
    const float maximum_update = std::clamp(options.maximumUpdate, 0.0f, 0.5f);
    const float edge_threshold = std::clamp(options.edgeThreshold, 0.02f, 1.0f);
    double absolute_update_sum = 0.0;

    for (int level = level_count - 1; level >= 0; --level)
    {
        const int stride = 1 << level;
        for (int pass = 0; pass < passes_per_level; ++pass)
        {
            for (int color = 0; color < 2; ++color)
            {
                if (isCancelled && isCancelled())
                {
                    statistics.cancelled = true;
                    return finish();
                }
                unsigned long long level_update_count = 0;
                double level_absolute_update_sum = 0.0;
                float level_maximum_update = 0.0f;
                std::vector<float> slice_maximum_updates(
                    static_cast<std::size_t>(sampleDimensions[2]), 0.0f);
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) \
    reduction(+:level_update_count,level_absolute_update_sum)
#endif
                for (int z = stride;
                     z < sampleDimensions[2] - stride;
                     ++z)
                {
                    for (int y = stride;
                         y < sampleDimensions[1] - stride;
                         ++y)
                    {
                        for (int x = stride;
                             x < sampleDimensions[0] - stride;
                             ++x)
                        {
                            if (((x / stride + y / stride + z / stride) & 1) !=
                                color)
                            {
                                continue;
                            }
                            const std::size_t index =
                                sampleIndex(sampleDimensions, x, y, z);
                            if ((*supported)[index] == 0 ||
                                geometrySourceMask[index] == 0 ||
                                std::fabs((*field)[index]) <= 1.0e-5f ||
                                std::fabs((*field)[index]) > edge_threshold)
                            {
                                continue;
                            }
                            const std::array<std::array<int, 3>, 3> axes = {
                                std::array<int, 3>{stride, 0, 0},
                                std::array<int, 3>{0, stride, 0},
                                std::array<int, 3>{0, 0, stride}};
                            std::array<float, 3> axis_predictions{};
                            int axis_count = 0;
                            for (const auto &axis : axes)
                            {
                                const std::size_t negative_index = sampleIndex(
                                    sampleDimensions,
                                    x - axis[0],
                                    y - axis[1],
                                    z - axis[2]);
                                const std::size_t positive_index = sampleIndex(
                                    sampleDimensions,
                                    x + axis[0],
                                    y + axis[1],
                                    z + axis[2]);
                                if ((*supported)[negative_index] == 0 ||
                                    (*supported)[positive_index] == 0 ||
                                    (geometrySourceMask[index] &
                                     geometrySourceMask[negative_index] &
                                     geometrySourceMask[positive_index]) == 0 ||
                                    std::fabs((*field)[negative_index] -
                                              (*field)[index]) >
                                        edge_threshold ||
                                    std::fabs((*field)[positive_index] -
                                              (*field)[index]) >
                                        edge_threshold)
                                {
                                    continue;
                                }
                                axis_predictions[axis_count++] =
                                    ((*field)[negative_index] +
                                     (*field)[positive_index]) *
                                    0.5f;
                            }
                            if (axis_count < 2)
                            {
                                continue;
                            }
                            std::sort(
                                axis_predictions.begin(),
                                axis_predictions.begin() + axis_count);
                            const float curvature_prediction =
                                axis_count == 2
                                ? (axis_predictions[0] +
                                   axis_predictions[1]) *
                                      0.5f
                                : axis_predictions[1];
                            const float observation_weight = data_fidelity *
                                std::max(0.10f, observationWeight[index]);
                            const float regularization_weight =
                                smoothness * static_cast<float>(axis_count);
                            const float original_value = (*field)[index];
                            float target =
                                (observation_weight * original_value +
                                 regularization_weight *
                                     curvature_prediction) /
                                (observation_weight + regularization_weight);
                            float delta = std::clamp(
                                target - original_value,
                                -maximum_update,
                                maximum_update);
                            target = original_value + delta;
                            if (options.preserveFieldSign &&
                                std::fabs(original_value) > 1.0e-5f &&
                                !signsMatch(target, original_value))
                            {
                                target = std::copysign(
                                    std::numeric_limits<float>::epsilon(),
                                    original_value);
                                delta = target - original_value;
                            }
                            const float absolute_update = std::fabs(delta);
                            if (absolute_update <= 1.0e-7f)
                            {
                                continue;
                            }
                            (*field)[index] = std::clamp(target, -1.0f, 1.0f);
                            ++level_update_count;
                            level_absolute_update_sum += absolute_update;
                            slice_maximum_updates[static_cast<std::size_t>(z)] =
                                std::max(
                                    slice_maximum_updates[
                                        static_cast<std::size_t>(z)],
                                    absolute_update);
                        }
                    }
                }
                for (const float slice_maximum_update :
                     slice_maximum_updates)
                {
                    level_maximum_update = std::max(
                        level_maximum_update, slice_maximum_update);
                }
                statistics.updateOperationCount += level_update_count;
                absolute_update_sum += level_absolute_update_sum;
                statistics.maximumAbsoluteUpdate = std::max(
                    statistics.maximumAbsoluteUpdate,
                    level_maximum_update);
            }
        }
    }
    if (statistics.updateOperationCount > 0)
    {
        statistics.meanAbsoluteUpdate =
            absolute_update_sum /
            static_cast<double>(statistics.updateOperationCount);
    }
    return finish();
}

} // namespace xjw::mesh
