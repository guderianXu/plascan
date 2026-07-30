#include "VisualHullDepthRefiner.h"

#include "DepthTsdfSurfaceBuilder.h"
#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
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
    for (const auto &[value, weight] : ordered)
    {
        accumulated_weight += weight;
        if (accumulated_weight >= target_weight)
        {
            return value;
        }
    }
    return ordered.back().first;
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

bool sampleDepthObservation(
    const DepthTsdfFrame &frame,
    const double pixel[2],
    DepthObservation *observation)
{
    if (observation == nullptr ||
        pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > static_cast<double>(frame.depth.cols - 1) ||
        pixel[1] > static_cast<double>(frame.depth.rows - 1))
    {
        return false;
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
            frame.crossViewRepairedMask.at<std::uint8_t>(
                row, column) != 0
            ? 1.0f
            : 0.0f;
        return true;
    }

    float depth = 0.0f;
    float confidence = 0.0f;
    float repaired_fraction = 0.0f;
    const bool has_repaired_mask =
        frame.crossViewRepairedMask.type() == CV_8UC1 &&
        frame.crossViewRepairedMask.size() == frame.depth.size();
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
    }
    observation->depth = depth;
    observation->confidence = confidence;
    observation->repairedFraction = repaired_fraction;
    return true;
}

} // namespace

VisualHullDepthRefineStatistics VisualHullDepthRefiner::refine(
    TriMesh *mesh,
    const QVector<DepthTsdfFrame> &frames,
    const VisualHullDepthRefineOptions &options)
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
        displacements.reserve(frames.size());
        confidences.reserve(frames.size());
        int native_view_count = 0;
        const double world[3] = {vertex.x, vertex.y, vertex.z};
        for (const DepthTsdfFrame &frame : frames)
        {
            double pixel[2]{};
            double projected_depth = 0.0;
            if (!frame.camera.projectWorldPointWithDepth(
                    world, pixel, projected_depth))
            {
                continue;
            }
            ++statistics.projectedObservationCount;
            DepthObservation observation;
            if (!sampleDepthObservation(
                    frame, pixel, &observation))
            {
                continue;
            }
            const float observed_depth = observation.depth;
            const float confidence = observation.confidence;
            const bool repaired_observation =
                observation.repairedFraction >= 0.5f;
            if (!std::isfinite(observed_depth) ||
                !(observed_depth > 0.0f) ||
                !std::isfinite(confidence) ||
                confidence < options.minimumDepthConfidence ||
                std::abs(
                    observed_depth -
                    static_cast<float>(projected_depth)) >
                    options.maximumEvidenceDistance)
            {
                continue;
            }
            double target_world[3]{};
            if (!frame.camera.unprojectPixel(
                    pixel, observed_depth, target_world))
            {
                continue;
            }
            const float displacement =
                static_cast<float>(target_world[0] - world[0]) * vertex.nx +
                static_cast<float>(target_world[1] - world[1]) * vertex.ny +
                static_cast<float>(target_world[2] - world[2]) * vertex.nz;
            if (!std::isfinite(displacement))
            {
                continue;
            }
            const float frame_quality_weight = std::clamp(
                frame.frameQualityWeight, 0.0f, 1.0f);
            if (frame_quality_weight <= 1.0e-6f)
            {
                continue;
            }
            displacements.push_back(displacement);
            confidences.push_back(
                std::clamp(confidence, 0.0f, 1.0f) *
                frame_quality_weight *
                (repaired_observation
                     ? std::clamp(
                           options.repairedObservationWeight,
                           0.0f,
                           1.0f)
                     : 1.0f));
            if (!repaired_observation)
            {
                ++native_view_count;
            }
            ++statistics.acceptedObservationCount;
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
        const float displacement_median =
            weightedMedian(displacements, confidences);
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
        ++statistics.anchoredVertexCount;
    }

    std::vector<float> displacement = initial_displacement;
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
        std::clamp(options.regularizationIterations, 0, 100);
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
