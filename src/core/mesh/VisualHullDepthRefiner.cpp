#include "VisualHullDepthRefiner.h"

#include "DepthFrameQualificationPolicy.h"
#include "DepthProvenance.h"
#include "DepthTsdfSurfaceBuilder.h"
#include "RobustSurfaceDisplacementSolver.h"
#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xjw::mesh
{
namespace
{

float percentile(std::vector<float> values, float fraction)
{
    if (values.empty())
    {
        return 0.0f;
    }
    std::sort(values.begin(), values.end());
    const float position =
        std::clamp(fraction, 0.0f, 1.0f) *
        static_cast<float>(values.size() - 1);
    const std::size_t lower =
        static_cast<std::size_t>(std::floor(position));
    const std::size_t upper =
        static_cast<std::size_t>(std::ceil(position));
    const float blend = position - static_cast<float>(lower);
    return values[lower] * (1.0f - blend) + values[upper] * blend;
}

float median(std::vector<float> values)
{
    return percentile(std::move(values), 0.5f);
}

float weightedMedian(
    const std::vector<float> &values,
    const std::vector<float> &weights)
{
    if (values.empty() || values.size() != weights.size())
    {
        return 0.0f;
    }
    std::vector<std::pair<float, float>> ordered;
    ordered.reserve(values.size());
    float total_weight = 0.0f;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        const float weight = std::max(0.0f, weights[index]);
        ordered.emplace_back(values[index], weight);
        total_weight += weight;
    }
    if (!(total_weight > 0.0f))
    {
        return median(values);
    }
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto &first, const auto &second)
        {
            return first.first < second.first;
        });
    const float target_weight = total_weight * 0.5f;
    float accumulated_weight = 0.0f;
    for (std::size_t index = 0; index < ordered.size(); ++index)
    {
        const auto &[value, weight] = ordered[index];
        accumulated_weight += weight;
        if (accumulated_weight >= target_weight)
        {
            const float plateau_tolerance =
                std::max(1.0f, total_weight) * 1.0e-6f;
            if (std::abs(accumulated_weight - target_weight) <=
                    plateau_tolerance)
            {
                for (std::size_t next = index + 1;
                     next < ordered.size();
                     ++next)
                {
                    if (ordered[next].second > 0.0f)
                    {
                        return 0.5f *
                            (value + ordered[next].first);
                    }
                }
            }
            return value;
        }
    }
    return ordered.back().first;
}

struct ConsensusEstimate
{
    float value = 0.0f;
    bool blended = false;
};

ConsensusEstimate robustContinuousConsensus(
    const std::vector<float> &values,
    const std::vector<float> &weights,
    bool all_native,
    float maximum_median_absolute_deviation)
{
    ConsensusEstimate result;
    result.value = weightedMedian(values, weights);
    if (values.size() < 2 || values.size() != weights.size() ||
        !(maximum_median_absolute_deviation > 0.0f))
    {
        return result;
    }

    float total_weight = 0.0f;
    float weighted_sum = 0.0f;
    for (std::size_t index = 0; index < weights.size(); ++index)
    {
        const float positive_weight = std::max(0.0f, weights[index]);
        total_weight += positive_weight;
        weighted_sum += positive_weight * values[index];
    }
    if (!(total_weight > 0.0f))
    {
        return result;
    }

    // The weighted median is retained only as a robust scale estimator.  Using
    // it as the data target creates a discontinuous winner-takes-all depth
    // layer whenever two orbital views exchange dominant confidence.
    const float scale_center = result.value;
    std::vector<float> deviations;
    deviations.reserve(values.size());
    for (const float value : values)
    {
        deviations.push_back(std::abs(value - scale_center));
    }
    const float median_absolute_deviation =
        weightedMedian(deviations, weights);
    const float robust_scale = std::clamp(
        1.4826f * median_absolute_deviation,
        0.01f * maximum_median_absolute_deviation,
        maximum_median_absolute_deviation);
    float estimate = weighted_sum / total_weight;
    for (int iteration = 0; iteration < 8; ++iteration)
    {
        float robust_weight_sum = 0.0f;
        float robust_value_sum = 0.0f;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            const float weight = std::max(0.0f, weights[index]);
            if (!(weight > 0.0f))
            {
                continue;
            }
            const float normalized =
                (values[index] - estimate) /
                std::max(1.0e-8f, robust_scale);
            const float cauchy_weight =
                1.0f / (1.0f + normalized * normalized);
            const float robust_weight = weight * cauchy_weight;
            robust_weight_sum += robust_weight;
            robust_value_sum += robust_weight * values[index];
        }
        if (!(robust_weight_sum > 0.0f))
        {
            break;
        }
        const float updated = robust_value_sum / robust_weight_sum;
        if (std::abs(updated - estimate) <=
            1.0e-5f * std::max(1.0f, maximum_median_absolute_deviation))
        {
            estimate = updated;
            break;
        }
        estimate = updated;
    }
    result.value = estimate;
    result.blended = true;
    static_cast<void>(all_native);
    return result;
}

bool validNormal(const MeshVertex &vertex)
{
    const float length_squared =
        vertex.nx * vertex.nx +
        vertex.ny * vertex.ny +
        vertex.nz * vertex.nz;
    return std::isfinite(length_squared) && length_squared > 1.0e-12f;
}

struct DepthObservation
{
    float depth = 0.0f;
    float confidence = 0.0f;
    float repairedFraction = 0.0f;
    float inverseDepthRelativeSpread = 0.0f;
    bool hasInverseDepthRelativeSpread = false;
};

struct VertexDepthObservation
{
    int frameIndex = -1;
    float displacement = 0.0f;
    float confidence = 0.0f;
    bool repaired = false;
};

struct PairBiasSamples
{
    int firstFrame = -1;
    int secondFrame = -1;
    std::vector<float> differences;
    std::vector<float> weights;
};

bool validObservationPixel(
    const DepthTsdfFrame &frame,
    int row,
    int column)
{
    if (row < 0 || column < 0 ||
        row >= frame.depth.rows || column >= frame.depth.cols)
    {
        return false;
    }
    if ((!frame.supportMask.empty() &&
         frame.supportMask.at<std::uint8_t>(row, column) == 0) ||
        (!frame.depthValidMask.empty() &&
         frame.depthValidMask.at<std::uint8_t>(row, column) == 0))
    {
        return false;
    }
    const float depth = frame.depth.at<float>(row, column);
    const float confidence = frame.confidence.empty()
        ? 1.0f
        : frame.confidence.at<float>(row, column);
    return std::isfinite(depth) && depth > 0.0f &&
           std::isfinite(confidence);
}

bool nonInterpolatedMeasuredPixel(
    const DepthTsdfFrame &frame,
    int row,
    int column)
{
    if (frame.depthProvenance.type() != CV_8UC1 ||
        frame.depthProvenance.size() != frame.depth.size())
    {
        return false;
    }
    const std::uint8_t provenance =
        frame.depthProvenance.at<std::uint8_t>(row, column);
    return provenance != static_cast<std::uint8_t>(
               xjw::mvs::DepthProvenance::Invalid) &&
        !xjw::mvs::isInterpolatedDepthProvenance(provenance);
}

bool readDepthObservationPixel(
    const DepthTsdfFrame &frame,
    int row,
    int column,
    DepthObservation *observation)
{
    if (!validObservationPixel(frame, row, column))
    {
        return false;
    }
    observation->depth = frame.depth.at<float>(row, column);
    observation->confidence = frame.confidence.empty()
        ? 1.0f
        : frame.confidence.at<float>(row, column);
    observation->repairedFraction =
        frame.crossViewRepairedMask.type() == CV_8UC1 &&
        frame.crossViewRepairedMask.size() == frame.depth.size() &&
        frame.crossViewRepairedMask.at<std::uint8_t>(row, column) != 0
        ? 1.0f
        : 0.0f;
    if (frame.inverseDepthRelativeSpread.type() == CV_32FC1 &&
        frame.inverseDepthRelativeSpread.size() == frame.depth.size())
    {
        const float spread =
            frame.inverseDepthRelativeSpread.at<float>(row, column);
        if (std::isfinite(spread) && spread >= 0.0f)
        {
            observation->inverseDepthRelativeSpread = spread;
            observation->hasInverseDepthRelativeSpread = true;
        }
    }
    return true;
}

bool sampleDepthObservation(
    const DepthTsdfFrame &frame,
    const double pixel[2],
    const VisualHullDepthRefineOptions &options,
    bool *rejectedNonMeasured,
    bool *usedNearestMeasured,
    DepthObservation *observation)
{
    if (rejectedNonMeasured != nullptr)
    {
        *rejectedNonMeasured = false;
    }
    if (usedNearestMeasured != nullptr)
    {
        *usedNearestMeasured = false;
    }
    if (observation == nullptr ||
        pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > static_cast<double>(frame.depth.cols - 1) ||
        pixel[1] > static_cast<double>(frame.depth.rows - 1))
    {
        return false;
    }

    if (options.measuredDepthSamplesOnly)
    {
        const int column = static_cast<int>(std::lround(pixel[0]));
        const int row = static_cast<int>(std::lround(pixel[1]));
        if (!nonInterpolatedMeasuredPixel(frame, row, column))
        {
            if (rejectedNonMeasured != nullptr)
            {
                *rejectedNonMeasured = true;
            }
            return false;
        }
        if (!readDepthObservationPixel(
                frame, row, column, observation))
        {
            return false;
        }
        if (usedNearestMeasured != nullptr)
        {
            *usedNearestMeasured = true;
        }
        return true;
    }

    const int x0 = static_cast<int>(std::floor(pixel[0]));
    const int y0 = static_cast<int>(std::floor(pixel[1]));
    const int x1 = std::min(x0 + 1, frame.depth.cols - 1);
    const int y1 = std::min(y0 + 1, frame.depth.rows - 1);
    const float tx = static_cast<float>(pixel[0] - x0);
    const float ty = static_cast<float>(pixel[1] - y0);
    const int columns[4] = {x0, x1, x0, x1};
    const int rows[4] = {y0, y0, y1, y1};
    const float weights[4] = {
        (1.0f - tx) * (1.0f - ty),
        tx * (1.0f - ty),
        (1.0f - tx) * ty,
        tx * ty};

    bool all_valid = true;
    for (int sample = 0; sample < 4; ++sample)
    {
        if (!validObservationPixel(
                frame, rows[sample], columns[sample]))
        {
            all_valid = false;
            break;
        }
    }
    if (!all_valid)
    {
        const int column = static_cast<int>(std::lround(pixel[0]));
        const int row = static_cast<int>(std::lround(pixel[1]));
        return readDepthObservationPixel(
            frame, row, column, observation);
    }

    float depth = 0.0f;
    float confidence = 0.0f;
    float repaired_fraction = 0.0f;
    float inverse_depth_relative_spread = 0.0f;
    bool has_inverse_depth_relative_spread = false;
    const bool has_repaired_mask =
        frame.crossViewRepairedMask.type() == CV_8UC1 &&
        frame.crossViewRepairedMask.size() == frame.depth.size();
    const bool has_spread =
        frame.inverseDepthRelativeSpread.type() == CV_32FC1 &&
        frame.inverseDepthRelativeSpread.size() == frame.depth.size();
    for (int sample = 0; sample < 4; ++sample)
    {
        const float weight = weights[sample];
        depth += weight * frame.depth.at<float>(
            rows[sample], columns[sample]);
        confidence += weight * (
            frame.confidence.empty()
                ? 1.0f
                : frame.confidence.at<float>(
                      rows[sample], columns[sample]));
        if (has_repaired_mask &&
            frame.crossViewRepairedMask.at<std::uint8_t>(
                rows[sample], columns[sample]) != 0)
        {
            repaired_fraction += weight;
        }
        if (has_spread)
        {
            const float spread =
                frame.inverseDepthRelativeSpread.at<float>(
                    rows[sample], columns[sample]);
            if (std::isfinite(spread) && spread >= 0.0f)
            {
                inverse_depth_relative_spread += weight * spread;
                has_inverse_depth_relative_spread = true;
            }
        }
    }
    observation->depth = depth;
    observation->confidence = confidence;
    observation->repairedFraction = repaired_fraction;
    observation->inverseDepthRelativeSpread =
        inverse_depth_relative_spread;
    observation->hasInverseDepthRelativeSpread =
        has_inverse_depth_relative_spread;
    return true;
}

float inverseDepthSpreadWeight(
    const DepthObservation &observation,
    const VisualHullDepthRefineOptions &options)
{
    if (!options.enableInverseDepthSpreadWeighting ||
        !observation.hasInverseDepthRelativeSpread)
    {
        return 1.0f;
    }
    const float knee = std::clamp(
        options.inverseDepthSpreadWeightKnee, 0.0f, 0.099f);
    const float zero = std::clamp(
        options.inverseDepthSpreadWeightZero,
        knee + 1.0e-6f,
        0.10f);
    const float minimum_weight = std::clamp(
        options.minimumInverseDepthSpreadWeight, 0.0f, 1.0f);
    const float spread = observation.inverseDepthRelativeSpread;
    if (spread <= knee)
    {
        return 1.0f;
    }
    if (spread >= zero)
    {
        return minimum_weight;
    }
    const float normalized = (spread - knee) / (zero - knee);
    const float smooth =
        normalized * normalized * (3.0f - 2.0f * normalized);
    return 1.0f - smooth * (1.0f - minimum_weight);
}

std::vector<VertexDepthObservation> collectVertexObservations(
    const MeshVertex &vertex,
    const QVector<DepthTsdfFrame> &frames,
    const VisualHullDepthRefineOptions &options,
    VisualHullDepthRefineStatistics *statistics)
{
    std::vector<VertexDepthObservation> observations;
    observations.reserve(frames.size());
    if (!validNormal(vertex))
    {
        return observations;
    }
    const double world[3] = {vertex.x, vertex.y, vertex.z};
    for (int frame_index = 0;
         frame_index < frames.size();
         ++frame_index)
    {
        const DepthTsdfFrame &frame = frames[frame_index];
        if (options.primaryFramesOnly && frame.auxiliarySurfaceOnly)
        {
            if (statistics != nullptr)
            {
                ++statistics->rejectedAuxiliaryObservationCount;
            }
            continue;
        }
        double pixel[2]{};
        double projected_depth = 0.0;
        if (!frame.camera.projectWorldPointWithDepth(
                world, pixel, projected_depth))
        {
            continue;
        }
        if (statistics != nullptr)
        {
            ++statistics->projectedObservationCount;
        }
        DepthObservation observation;
        bool rejected_non_measured = false;
        bool used_nearest_measured = false;
        if (!sampleDepthObservation(
                frame,
                pixel,
                options,
                &rejected_non_measured,
                &used_nearest_measured,
                &observation))
        {
            if (statistics != nullptr && rejected_non_measured)
            {
                ++statistics->rejectedNonMeasuredObservationCount;
            }
            continue;
        }
        if (statistics != nullptr && used_nearest_measured)
        {
            ++statistics->nearestMeasuredObservationCount;
        }
        const bool repaired_observation =
            observation.repairedFraction >= 0.5f;
        if (!std::isfinite(observation.depth) ||
            !(observation.depth > 0.0f) ||
            !std::isfinite(observation.confidence) ||
            observation.confidence < options.minimumDepthConfidence ||
            std::abs(
                observation.depth -
                static_cast<float>(projected_depth)) >
                options.maximumEvidenceDistance)
        {
            continue;
        }
        double target_world[3]{};
        if (!frame.camera.unprojectPixel(
                pixel, observation.depth, target_world))
        {
            continue;
        }
        const float displacement =
            static_cast<float>(target_world[0] - world[0]) * vertex.nx +
            static_cast<float>(target_world[1] - world[1]) * vertex.ny +
            static_cast<float>(target_world[2] - world[2]) * vertex.nz;
        const float frame_quality_weight = std::clamp(
            frame.frameQualityWeight, 0.0f, 1.0f);
        const float frame_role_weight = frame.auxiliarySurfaceOnly
            ? xjw::mvs::kCoverageAuxiliaryWeightMultiplier
            : 1.0f;
        if (!std::isfinite(displacement) ||
            frame_quality_weight <= 1.0e-6f)
        {
            continue;
        }
        const float spread_weight =
            inverseDepthSpreadWeight(observation, options);
        const float repaired_fraction =
            std::clamp(observation.repairedFraction, 0.0f, 1.0f);
        const float repaired_weight =
            1.0f +
            repaired_fraction *
                (std::clamp(
                     options.repairedObservationWeight, 0.0f, 1.0f) -
                 1.0f);
        const float confidence =
            std::clamp(observation.confidence, 0.0f, 1.0f) *
            frame_quality_weight *
            frame_role_weight *
            repaired_weight *
            spread_weight;
        if (!(confidence > 0.0f))
        {
            continue;
        }
        observations.push_back(
            {frame_index,
             displacement,
             confidence,
             repaired_observation});
        if (statistics != nullptr)
        {
            ++statistics->acceptedObservationCount;
            if (spread_weight < 1.0f)
            {
                ++statistics->spreadDownweightedObservationCount;
            }
            if (options.enableInverseDepthSpreadWeighting &&
                spread_weight <=
                    options.minimumInverseDepthSpreadWeight + 1.0e-6f)
            {
                ++statistics->spreadVeryWeakObservationCount;
            }
        }
    }
    return observations;
}

std::vector<float> estimateCrossViewNormalBias(
    const std::vector<std::vector<VertexDepthObservation>>
        &vertex_observations,
    int frame_count,
    const VisualHullDepthRefineOptions &options,
    VisualHullDepthRefineStatistics *statistics)
{
    std::vector<float> biases(
        static_cast<std::size_t>(std::max(0, frame_count)), 0.0f);
    if (!options.enableCrossViewBiasCompensation ||
        frame_count < 2)
    {
        return biases;
    }

    std::unordered_map<std::uint64_t, PairBiasSamples> pair_samples;
    const float maximum_pair_difference =
        2.0f * options.maximumViewMedianAbsoluteDeviation;
    for (const auto &observations : vertex_observations)
    {
        for (std::size_t first = 0;
             first < observations.size();
             ++first)
        {
            if (observations[first].repaired)
            {
                continue;
            }
            for (std::size_t second = first + 1;
                 second < observations.size();
                 ++second)
            {
                if (observations[second].repaired)
                {
                    continue;
                }
                const VertexDepthObservation *lower = &observations[first];
                const VertexDepthObservation *upper = &observations[second];
                if (lower->frameIndex > upper->frameIndex)
                {
                    std::swap(lower, upper);
                }
                const float difference =
                    lower->displacement - upper->displacement;
                if (!std::isfinite(difference) ||
                    std::abs(difference) > maximum_pair_difference)
                {
                    continue;
                }
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(
                         static_cast<std::uint32_t>(
                             lower->frameIndex)) << 32U) |
                    static_cast<std::uint32_t>(
                        upper->frameIndex);
                PairBiasSamples &samples = pair_samples[key];
                samples.firstFrame = lower->frameIndex;
                samples.secondFrame = upper->frameIndex;
                samples.differences.push_back(difference);
                samples.weights.push_back(std::min(
                    lower->confidence, upper->confidence));
            }
        }
    }

    struct BiasEdge
    {
        int first = -1;
        int second = -1;
        float difference = 0.0f;
        float weight = 0.0f;
    };
    std::vector<BiasEdge> edges;
    const int minimum_samples =
        std::max(1, options.minimumCrossViewBiasPairSamples);
    for (const auto &[key, samples] : pair_samples)
    {
        (void)key;
        if (static_cast<int>(samples.differences.size()) <
            minimum_samples)
        {
            continue;
        }
        edges.push_back(
            {samples.firstFrame,
             samples.secondFrame,
             weightedMedian(
                 samples.differences, samples.weights),
             std::sqrt(static_cast<float>(
                 samples.differences.size()))});
    }
    if (edges.empty())
    {
        return biases;
    }

    std::vector<float> updated(biases.size(), 0.0f);
    std::vector<float> incident_weight(biases.size(), 0.0f);
    for (int iteration = 0; iteration < 40; ++iteration)
    {
        std::fill(updated.begin(), updated.end(), 0.0f);
        std::fill(
            incident_weight.begin(),
            incident_weight.end(),
            0.0f);
        for (const BiasEdge &edge : edges)
        {
            const std::size_t first =
                static_cast<std::size_t>(edge.first);
            const std::size_t second =
                static_cast<std::size_t>(edge.second);
            updated[first] += edge.weight *
                (biases[second] + edge.difference);
            incident_weight[first] += edge.weight;
            updated[second] += edge.weight *
                (biases[first] - edge.difference);
            incident_weight[second] += edge.weight;
        }
        std::vector<float> connected_biases;
        for (std::size_t index = 0;
             index < biases.size();
             ++index)
        {
            if (incident_weight[index] > 0.0f)
            {
                updated[index] /= incident_weight[index];
                connected_biases.push_back(updated[index]);
            }
            else
            {
                updated[index] = 0.0f;
            }
        }
        const float gauge = median(connected_biases);
        for (std::size_t index = 0;
             index < biases.size();
             ++index)
        {
            if (incident_weight[index] > 0.0f)
            {
                updated[index] -= gauge;
            }
        }
        biases.swap(updated);
    }

    const float maximum_bias =
        options.maximumCrossViewBias > 0.0f
        ? options.maximumCrossViewBias
        : std::min(
              options.maximumDisplacement,
              options.maximumViewMedianAbsoluteDeviation);
    for (std::size_t index = 0; index < biases.size(); ++index)
    {
        if (incident_weight[index] <= 0.0f)
        {
            biases[index] = 0.0f;
            continue;
        }
        biases[index] = std::clamp(
            biases[index], -maximum_bias, maximum_bias);
        if (statistics != nullptr)
        {
            ++statistics->biasCalibratedFrameCount;
            statistics->maximumAbsoluteFrameBias = std::max(
                statistics->maximumAbsoluteFrameBias,
                std::abs(biases[index]));
        }
    }
    if (statistics != nullptr)
    {
        statistics->biasCalibrationPairCount = edges.size();
    }
    return biases;
}

} // namespace

VisualHullDepthRefineStatistics VisualHullDepthRefiner::refine(
    TriMesh *mesh,
    const QVector<DepthTsdfFrame> &frames,
    const VisualHullDepthRefineOptions &options,
    const std::function<bool()> &isCancelled)
{
    VisualHullDepthRefineStatistics statistics;
    if (mesh == nullptr || mesh->empty() ||
        frames.size() < options.minimumViewCount ||
        !(options.maximumEvidenceDistance > 0.0f) ||
        !(options.maximumDisplacement > 0.0f) ||
        !(options.maximumViewMedianAbsoluteDeviation > 0.0f))
    {
        return statistics;
    }

    detail::recomputeNormals(mesh);
    const std::size_t vertex_count = mesh->vertices.size();
    std::vector<float> initial_displacement(vertex_count, 0.0f);
    std::vector<float> anchor_weight(vertex_count, 0.0f);
    std::vector<float> supporting_view_counts(vertex_count, 0.0f);
    std::vector<RobustSurfaceDisplacementObservation>
        solver_observations;
    solver_observations.reserve(vertex_count * 2);
    std::vector<std::vector<VertexDepthObservation>>
        vertex_observations(vertex_count);
    for (std::size_t vertex_index = 0;
         vertex_index < vertex_count;
         ++vertex_index)
    {
        vertex_observations[vertex_index] =
            collectVertexObservations(
                mesh->vertices[vertex_index],
                frames,
                options,
                &statistics);
    }
    const std::vector<float> frame_biases =
        estimateCrossViewNormalBias(
            vertex_observations,
            frames.size(),
            options,
            &statistics);

    const int minimum_view_count = std::max(1, options.minimumViewCount);
    for (std::size_t vertex_index = 0;
         vertex_index < vertex_count;
         ++vertex_index)
    {
        const MeshVertex &vertex = mesh->vertices[vertex_index];
        if (!validNormal(vertex))
        {
            continue;
        }
        std::vector<float> displacements;
        std::vector<float> confidences;
        bool all_observations_native = true;
        displacements.reserve(frames.size());
        confidences.reserve(frames.size());
        int native_view_count = 0;
        for (const VertexDepthObservation &observation :
             vertex_observations[vertex_index])
        {
            const std::size_t frame_index =
                static_cast<std::size_t>(
                    observation.frameIndex);
            const float bias =
                frame_index < frame_biases.size()
                ? frame_biases[frame_index]
                : 0.0f;
            displacements.push_back(
                observation.displacement - bias);
            confidences.push_back(observation.confidence);
            if (!observation.repaired)
            {
                ++native_view_count;
            }
            else
            {
                all_observations_native = false;
            }
        }

        supporting_view_counts[vertex_index] =
            static_cast<float>(displacements.size());
        if (static_cast<int>(displacements.size()) < minimum_view_count)
        {
            continue;
        }
        if (native_view_count <
            std::max(0, options.minimumNativeViewCount))
        {
            continue;
        }
        const ConsensusEstimate displacement_consensus =
            robustContinuousConsensus(
                displacements,
                confidences,
                all_observations_native,
                options.maximumViewMedianAbsoluteDeviation);
        const float displacement_median =
            displacement_consensus.value;
        std::vector<float> absolute_deviations;
        absolute_deviations.reserve(displacements.size());
        for (const float displacement : displacements)
        {
            absolute_deviations.push_back(
                std::abs(displacement - displacement_median));
        }
        const float median_absolute_deviation =
            weightedMedian(absolute_deviations, confidences);
        const float mean_confidence =
            std::accumulate(
                confidences.cbegin(), confidences.cend(), 0.0f) /
            static_cast<float>(confidences.size());
        const float view_weight = std::clamp(
            static_cast<float>(displacements.size()) /
                static_cast<float>(minimum_view_count + 1),
            0.0f,
            1.0f);
        const float agreement_weight = std::clamp(
            1.0f -
                median_absolute_deviation /
                    options.maximumViewMedianAbsoluteDeviation,
            0.0f,
            1.0f);
        const float weight =
            mean_confidence * view_weight * agreement_weight;
        if (weight < options.minimumAnchorWeight)
        {
            continue;
        }
        anchor_weight[vertex_index] = weight;
        initial_displacement[vertex_index] = std::clamp(
            displacement_median,
            -options.maximumDisplacement,
            options.maximumDisplacement);
        const float observation_reliability =
            view_weight * agreement_weight;
        for (std::size_t observation_index = 0;
             observation_index < displacements.size();
             ++observation_index)
        {
            const float observation_weight =
                confidences[observation_index] *
                observation_reliability;
            if (!(observation_weight > 0.0f))
            {
                continue;
            }
            solver_observations.push_back(
                {static_cast<int>(vertex_index),
                 std::clamp(
                     displacements[observation_index],
                     -options.maximumDisplacement,
                     options.maximumDisplacement),
                 observation_weight});
        }
        statistics.blendedConsensusVertexCount +=
            displacement_consensus.blended ? 1U : 0U;
        ++statistics.anchoredVertexCount;
    }

    std::vector<float> displacement = initial_displacement;
    bool use_legacy_regularization = true;
    if (options.enableGlobalRobustSolver &&
        !solver_observations.empty())
    {
        ++statistics.globalSolverAttemptCount;
        RobustSurfaceDisplacementOptions solver_options;
        solver_options.irlsIterations =
            options.globalSolverIrlsIterations;
        solver_options.maximumPcgIterations =
            options.globalSolverMaximumPcgIterations;
        solver_options.convergenceTolerance =
            options.globalSolverConvergenceTolerance;
        solver_options.robustScale =
            options.maximumViewMedianAbsoluteDeviation *
            options.globalSolverRobustScaleMultiplier;
        solver_options.laplacianWeight =
            options.globalSolverLaplacianWeight;
        solver_options.hullPriorWeight =
            options.globalSolverHullPriorWeight;
        solver_options.maximumDisplacement =
            options.maximumDisplacement;
        constexpr float degrees_to_radians =
            3.14159265358979323846f / 180.0f;
        solver_options.minimumNormalDot = std::cos(
            std::clamp(
                options.regularizationMaximumNormalAngleDegrees,
                5.0f,
                89.0f) *
            degrees_to_radians);
        statistics.globalSolverEffectiveRobustScale =
            solver_options.robustScale;
        std::vector<float> candidate_displacement =
            initial_displacement;
        const RobustSurfaceDisplacementStatistics solver =
            RobustSurfaceDisplacementSolver::solve(
                *mesh,
                solver_observations,
                solver_options,
                &candidate_displacement,
                isCancelled);
        statistics.globalSolverCancelled = solver.cancelled;
        statistics.globalSolverIrlsIterationCount =
            solver.irlsIterationCount;
        statistics.globalSolverPcgIterationCount =
            solver.pcgIterationCount;
        statistics.globalSolverObservationCount =
            solver.observationCount;
        statistics.globalSolverRegularizationEdgeCount =
            solver.regularizationEdgeCount;
        statistics.globalSolverAnchoredVertexCount =
            solver.anchoredVertexCount;
        statistics.globalSolverPriorOnlyVertexCount =
            solver.priorOnlyVertexCount;
        statistics.globalSolverInitialEnergy =
            solver.initialEnergy;
        statistics.globalSolverFinalEnergy =
            solver.finalEnergy;
        statistics.globalSolverFinalRelativeResidual =
            solver.finalRelativeResidual;
        statistics.globalSolverSolvedPassCount =
            solver.solved ? 1 : 0;
        statistics.globalSolverConvergedPassCount =
            solver.converged ? 1 : 0;
        if (solver.cancelled)
        {
            return statistics;
        }
        const double energy_tolerance = std::max(
            1.0e-12,
            std::abs(solver.initialEnergy) * 1.0e-5);
        if (solver.solved &&
            solver.finalEnergy <=
                solver.initialEnergy + energy_tolerance)
        {
            displacement = std::move(candidate_displacement);
            use_legacy_regularization = false;
            statistics.globalSolverAppliedPassCount = 1;
        }
        else
        {
            statistics.globalSolverFallbackPassCount = 1;
        }
    }
    std::vector<float> confidence = anchor_weight;
    std::vector<std::vector<int>> neighbors(vertex_count);
    for (const Triangle &face : mesh->faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const int first = face.v[corner];
            const int second = face.v[(corner + 1) % 3];
            if (first < 0 || second < 0 ||
                first >= static_cast<int>(vertex_count) ||
                second >= static_cast<int>(vertex_count) ||
                first == second)
            {
                continue;
            }
            neighbors[static_cast<std::size_t>(first)].push_back(second);
            neighbors[static_cast<std::size_t>(second)].push_back(first);
        }
    }
    for (std::vector<int> &adjacent : neighbors)
    {
        std::sort(adjacent.begin(), adjacent.end());
        adjacent.erase(
            std::unique(adjacent.begin(), adjacent.end()),
            adjacent.end());
    }
    const int regularization_iterations =
        use_legacy_regularization
        ? std::clamp(options.regularizationIterations, 0, 100)
        : 0;
    const float regularization_weight =
        std::clamp(options.regularizationWeight, 0.0f, 20.0f);
    const float propagation_decay =
        std::clamp(options.propagationDecay, 0.0f, 1.0f);
    constexpr float degrees_to_radians =
        3.14159265358979323846f / 180.0f;
    const float minimum_normal_dot = std::cos(
        std::clamp(
            options.regularizationMaximumNormalAngleDegrees,
            5.0f,
            89.0f) *
        degrees_to_radians);
    for (int iteration = 0;
         iteration < regularization_iterations;
         ++iteration)
    {
        std::vector<float> updated(vertex_count, 0.0f);
        std::vector<float> updated_confidence(vertex_count, 0.0f);
        for (std::size_t index = 0; index < vertex_count; ++index)
        {
            float neighbor_displacement_sum = 0.0f;
            float neighbor_confidence_sum = 0.0f;
            float compatibility_sum = 0.0f;
            const MeshVertex &vertex = mesh->vertices[index];
            for (const int adjacent_index : neighbors[index])
            {
                const std::size_t adjacent =
                    static_cast<std::size_t>(adjacent_index);
                const MeshVertex &neighbor = mesh->vertices[adjacent];
                const float normal_dot = std::clamp(
                    vertex.nx * neighbor.nx +
                        vertex.ny * neighbor.ny +
                        vertex.nz * neighbor.nz,
                    -1.0f,
                    1.0f);
                if (normal_dot <= minimum_normal_dot)
                {
                    continue;
                }
                const float compatibility = std::pow(
                    (normal_dot - minimum_normal_dot) /
                        std::max(1.0e-6f, 1.0f - minimum_normal_dot),
                    2.0f);
                const float weighted_confidence =
                    compatibility * confidence[adjacent];
                neighbor_displacement_sum +=
                    weighted_confidence * displacement[adjacent];
                neighbor_confidence_sum += weighted_confidence;
                compatibility_sum += compatibility;
            }
            if (neighbor_confidence_sum <= 1.0e-12f ||
                compatibility_sum <= 1.0e-12f)
            {
                updated[index] = initial_displacement[index];
                updated_confidence[index] = anchor_weight[index];
                continue;
            }
            const float neighbor_average =
                neighbor_displacement_sum /
                neighbor_confidence_sum;
            const float neighbor_confidence =
                neighbor_confidence_sum / compatibility_sum;
            const float propagated_weight =
                regularization_weight *
                propagation_decay *
                neighbor_confidence;
            const float denominator =
                anchor_weight[index] + propagated_weight;
            if (denominator > 1.0e-12f)
            {
                updated[index] = std::clamp(
                    (anchor_weight[index] *
                         initial_displacement[index] +
                     propagated_weight * neighbor_average) /
                        denominator,
                    -options.maximumDisplacement,
                    options.maximumDisplacement);
            }
            updated_confidence[index] = std::max(
                anchor_weight[index],
                propagation_decay * neighbor_confidence);
        }
        displacement.swap(updated);
        confidence.swap(updated_confidence);
    }

    std::vector<float> absolute_displacements;
    absolute_displacements.reserve(vertex_count);
    for (std::size_t index = 0; index < vertex_count; ++index)
    {
        const float value = displacement[index];
        absolute_displacements.push_back(std::abs(value));
        if (std::abs(value) <= 1.0e-12f)
        {
            continue;
        }
        MeshVertex &vertex = mesh->vertices[index];
        vertex.x += value * vertex.nx;
        vertex.y += value * vertex.ny;
        vertex.z += value * vertex.nz;
        ++statistics.displacedVertexCount;
    }
    detail::recomputeNormals(mesh);
    statistics.medianSupportingViewCount =
        percentile(supporting_view_counts, 0.5f);
    statistics.p90SupportingViewCount =
        percentile(std::move(supporting_view_counts), 0.9f);
    statistics.maximumAppliedDisplacement =
        absolute_displacements.empty()
        ? 0.0f
        : *std::max_element(
              absolute_displacements.cbegin(),
              absolute_displacements.cend());
    statistics.medianAppliedDisplacement =
        percentile(absolute_displacements, 0.5f);
    statistics.p90AppliedDisplacement =
        percentile(std::move(absolute_displacements), 0.9f);
    statistics.applied = statistics.displacedVertexCount > 0;
    return statistics;
}

} // namespace xjw::mesh
