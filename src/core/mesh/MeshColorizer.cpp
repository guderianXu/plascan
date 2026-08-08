#include "MeshColorizer.h"

#include "MeshFaceColorOptimizer.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#ifdef MESHING_OPENMP
#include <omp.h>
#endif

namespace xjw::mesh
{
namespace
{

struct ColorCandidate
{
    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;
    float weight = 0.0f;
};

float median(std::vector<float> values)
{
    if (values.empty())
    {
        return 0.0f;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    float result = values[middle];
    if (values.size() % 2 == 0)
    {
        const auto lower = std::max_element(values.begin(), values.begin() + middle);
        result = (*lower + result) * 0.5f;
    }
    return result;
}

cv::Vec3f bilinearColor(const cv::Mat &image, double x, double y)
{
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, image.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, image.rows - 1);
    const int x1 = std::min(x0 + 1, image.cols - 1);
    const int y1 = std::min(y0 + 1, image.rows - 1);
    const float tx = static_cast<float>(x - x0);
    const float ty = static_cast<float>(y - y0);
    const cv::Vec3f top = cv::Vec3f(image.at<cv::Vec3b>(y0, x0)) * (1.0f - tx)
        + cv::Vec3f(image.at<cv::Vec3b>(y0, x1)) * tx;
    const cv::Vec3f bottom = cv::Vec3f(image.at<cv::Vec3b>(y1, x0)) * (1.0f - tx)
        + cv::Vec3f(image.at<cv::Vec3b>(y1, x1)) * tx;
    return top * (1.0f - ty) + bottom * ty;
}

std::vector<float> exposureGains(const QVector<MeshColorView> &views, bool enabled)
{
    std::vector<float> luminances(static_cast<std::size_t>(views.size()), 0.0f);
    std::vector<float> valid_luminances;
    for (int index = 0; index < views.size(); ++index)
    {
        const MeshColorView &view = views[index];
        if (view.colorBgr.type() != CV_8UC3)
        {
            continue;
        }
        cv::Mat mask;
        if (view.supportMask.type() == CV_8UC1 &&
            view.depthValidMask.type() == CV_8UC1 &&
            view.supportMask.size() == view.colorBgr.size() &&
            view.depthValidMask.size() == view.colorBgr.size())
        {
            cv::bitwise_and(view.supportMask, view.depthValidMask, mask);
        }
        const cv::Scalar mean = cv::mean(view.colorBgr, mask);
        const float luminance = static_cast<float>(
            0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2]);
        if (std::isfinite(luminance) && luminance > 1.0f)
        {
            luminances[static_cast<std::size_t>(index)] = luminance;
            valid_luminances.push_back(luminance);
        }
    }

    const float target = median(valid_luminances);
    std::vector<float> gains(static_cast<std::size_t>(views.size()), 1.0f);
    if (!enabled || target <= 1.0f)
    {
        return gains;
    }
    for (std::size_t index = 0; index < gains.size(); ++index)
    {
        if (luminances[index] > 1.0f)
        {
            gains[index] = std::clamp(target / luminances[index], 0.75f, 1.33f);
        }
    }
    return gains;
}

std::vector<cv::Mat> visibilityDepths(const TriMesh &mesh,
                                      const QVector<MeshColorView> &views,
                                      int worker_count)
{
    std::vector<cv::Mat> depths(static_cast<std::size_t>(views.size()));
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(worker_count) if(worker_count > 1)
#endif
    for (int view_index = 0; view_index < views.size(); ++view_index)
    {
        const MeshColorView &view = views[view_index];
        if (view.colorBgr.empty())
        {
            continue;
        }
        cv::Mat depth(view.colorBgr.size(), CV_32F,
                      cv::Scalar(std::numeric_limits<float>::infinity()));
        for (const MeshVertex &vertex : mesh.vertices)
        {
            const double world[3] = {vertex.x, vertex.y, vertex.z};
            double pixel[2]{};
            double camera_depth = 0.0;
            if (!view.camera.projectWorldPointWithDepth(world, pixel, camera_depth))
            {
                continue;
            }
            const int column = static_cast<int>(std::lround(pixel[0]));
            const int row = static_cast<int>(std::lround(pixel[1]));
            if (row >= 0 && column >= 0 && row < depth.rows && column < depth.cols)
            {
                float &nearest = depth.at<float>(row, column);
                nearest = std::min(nearest, static_cast<float>(camera_depth));
            }
        }
        cv::erode(depth, depth, cv::Mat::ones(3, 3, CV_8UC1));
        depths[static_cast<std::size_t>(view_index)] = std::move(depth);
    }
    return depths;
}

std::vector<ColorCandidate> rejectColorOutliers(const std::vector<ColorCandidate> &input,
                                                std::uint64_t *rejected)
{
    if (input.size() < 3)
    {
        return input;
    }
    std::vector<float> reds;
    std::vector<float> greens;
    std::vector<float> blues;
    reds.reserve(input.size());
    greens.reserve(input.size());
    blues.reserve(input.size());
    for (const ColorCandidate &candidate : input)
    {
        reds.push_back(candidate.red);
        greens.push_back(candidate.green);
        blues.push_back(candidate.blue);
    }
    const float center_red = median(reds);
    const float center_green = median(greens);
    const float center_blue = median(blues);
    std::vector<float> distances;
    distances.reserve(input.size());
    for (const ColorCandidate &candidate : input)
    {
        const float dr = candidate.red - center_red;
        const float dg = candidate.green - center_green;
        const float db = candidate.blue - center_blue;
        distances.push_back(std::sqrt(dr * dr + dg * dg + db * db));
    }
    const float center_distance = median(distances);
    std::vector<float> deviations;
    deviations.reserve(distances.size());
    for (float distance : distances)
    {
        deviations.push_back(std::fabs(distance - center_distance));
    }
    const float threshold = std::max(18.0f,
                                     center_distance + 2.5f * std::max(4.0f, median(deviations)));
    std::vector<ColorCandidate> output;
    output.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        if (distances[index] <= threshold)
        {
            output.push_back(input[index]);
        }
        else if (rejected)
        {
            ++(*rejected);
        }
    }
    return output;
}

void propagateColors(TriMesh *mesh,
                     std::vector<std::uint8_t> *colored,
                     const MeshColorOptions &options,
                     MeshColorStatistics *statistics)
{
    if (!mesh || !colored || !statistics)
    {
        return;
    }
    const std::size_t vertex_count = mesh->vertices.size();
    for (int pass = 0; pass < options.propagationPasses; ++pass)
    {
        std::vector<float> red(vertex_count, 0.0f);
        std::vector<float> green(vertex_count, 0.0f);
        std::vector<float> blue(vertex_count, 0.0f);
        std::vector<int> count(vertex_count, 0);
        const auto accumulate = [&](int target, int source)
        {
            const std::size_t target_index = static_cast<std::size_t>(target);
            const std::size_t source_index = static_cast<std::size_t>(source);
            if ((*colored)[target_index] || !(*colored)[source_index])
            {
                return;
            }
            const MeshVertex &a = mesh->vertices[target_index];
            const MeshVertex &b = mesh->vertices[source_index];
            const float normal_score = a.nx * b.nx + a.ny * b.ny + a.nz * b.nz;
            if (normal_score < options.propagationNormalCosine)
            {
                return;
            }
            red[target_index] += b.r;
            green[target_index] += b.g;
            blue[target_index] += b.b;
            ++count[target_index];
        };
        for (const Triangle &face : mesh->faces)
        {
            accumulate(face.v[0], face.v[1]); accumulate(face.v[1], face.v[0]);
            accumulate(face.v[1], face.v[2]); accumulate(face.v[2], face.v[1]);
            accumulate(face.v[2], face.v[0]); accumulate(face.v[0], face.v[2]);
        }
        int changed = 0;
        for (std::size_t index = 0; index < vertex_count; ++index)
        {
            if ((*colored)[index] || count[index] == 0)
            {
                continue;
            }
            MeshVertex &vertex = mesh->vertices[index];
            vertex.r = static_cast<std::uint8_t>(std::lround(red[index] / count[index]));
            vertex.g = static_cast<std::uint8_t>(std::lround(green[index] / count[index]));
            vertex.b = static_cast<std::uint8_t>(std::lround(blue[index] / count[index]));
            (*colored)[index] = 1;
            ++changed;
        }
        statistics->propagatedVertexCount += changed;
        if (changed == 0)
        {
            break;
        }
    }
}

int cleanIsolatedColorSpeckles(TriMesh *mesh, const MeshColorOptions &options)
{
    if (!mesh || options.speckleCleanupPasses <= 0)
    {
        return 0;
    }
    int cleaned_total = 0;
    const std::size_t vertex_count = mesh->vertices.size();
    for (int pass = 0; pass < options.speckleCleanupPasses; ++pass)
    {
        std::vector<float> red(vertex_count, 0.0f);
        std::vector<float> green(vertex_count, 0.0f);
        std::vector<float> blue(vertex_count, 0.0f);
        std::vector<float> red_squared(vertex_count, 0.0f);
        std::vector<float> green_squared(vertex_count, 0.0f);
        std::vector<float> blue_squared(vertex_count, 0.0f);
        std::vector<int> count(vertex_count, 0);
        const auto accumulate = [&](int target, int source)
        {
            const MeshVertex &a = mesh->vertices[static_cast<std::size_t>(target)];
            const MeshVertex &b = mesh->vertices[static_cast<std::size_t>(source)];
            const float normal_score = a.nx * b.nx + a.ny * b.ny + a.nz * b.nz;
            if (normal_score < options.speckleNormalCosine)
            {
                return;
            }
            const std::size_t index = static_cast<std::size_t>(target);
            red[index] += b.r; green[index] += b.g; blue[index] += b.b;
            red_squared[index] += b.r * b.r;
            green_squared[index] += b.g * b.g;
            blue_squared[index] += b.b * b.b;
            ++count[index];
        };
        for (const Triangle &face : mesh->faces)
        {
            accumulate(face.v[0], face.v[1]); accumulate(face.v[1], face.v[0]);
            accumulate(face.v[1], face.v[2]); accumulate(face.v[2], face.v[1]);
            accumulate(face.v[2], face.v[0]); accumulate(face.v[0], face.v[2]);
        }
        const std::vector<MeshVertex> source = mesh->vertices;
        int cleaned = 0;
        for (std::size_t index = 0; index < vertex_count; ++index)
        {
            if (count[index] < 3)
            {
                continue;
            }
            const float inverse = 1.0f / count[index];
            const float mean_red = red[index] * inverse;
            const float mean_green = green[index] * inverse;
            const float mean_blue = blue[index] * inverse;
            const float variance = std::max(0.0f, red_squared[index] * inverse - mean_red * mean_red)
                + std::max(0.0f, green_squared[index] * inverse - mean_green * mean_green)
                + std::max(0.0f, blue_squared[index] * inverse - mean_blue * mean_blue);
            const float neighbor_deviation = std::sqrt(variance / 3.0f);
            const float dr = source[index].r - mean_red;
            const float dg = source[index].g - mean_green;
            const float db = source[index].b - mean_blue;
            if (std::sqrt(dr * dr + dg * dg + db * db) <
                    options.speckleMinimumColorDistance ||
                neighbor_deviation > options.speckleMaximumNeighborDeviation)
            {
                continue;
            }
            mesh->vertices[index].r = static_cast<std::uint8_t>(
                std::clamp(std::lround(mean_red), 0l, 255l));
            mesh->vertices[index].g = static_cast<std::uint8_t>(
                std::clamp(std::lround(mean_green), 0l, 255l));
            mesh->vertices[index].b = static_cast<std::uint8_t>(
                std::clamp(std::lround(mean_blue), 0l, 255l));
            ++cleaned;
        }
        cleaned_total += cleaned;
        if (cleaned == 0)
        {
            break;
        }
    }
    return cleaned_total;
}

} // namespace

MeshColorStatistics MeshColorizer::colorize(TriMesh *mesh,
                                             const QVector<MeshColorView> &views,
                                             const MeshColorOptions &options)
{
    MeshColorStatistics statistics;
    const auto started_at = std::chrono::steady_clock::now();
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() || views.empty())
    {
        return statistics;
    }
    int maximum_workers = 1;
#ifdef MESHING_OPENMP
    maximum_workers = std::max(1, omp_get_max_threads());
#endif
    const int desired_workers = options.workerCount > 0
        ? options.workerCount
        : std::max(1, maximum_workers - 2);
    statistics.effectiveWorkerCount = std::clamp(
        desired_workers,
        1,
        std::max(1, std::min(
            static_cast<int>(mesh->vertices.size()), maximum_workers)));
    const std::vector<float> gains = exposureGains(views, options.compensateExposure);
    const std::vector<cv::Mat> visibility_depths = visibilityDepths(
        *mesh, views, statistics.effectiveWorkerCount);
    std::vector<std::uint8_t> colored(mesh->vertices.size(), 0);
    std::vector<std::array<std::uint8_t, 3>> reliable_color_by_vertex(
        mesh->vertices.size());
    const float voxel_size = std::max(options.maximumVoxelSize, 1.0e-8f);

    std::uint64_t candidate_observation_count = 0;
    std::uint64_t rejected_projection_count = 0;
    std::uint64_t rejected_mask_count = 0;
    std::uint64_t rejected_depth_count = 0;
    std::uint64_t rejected_visibility_count = 0;
    std::uint64_t rejected_view_angle_count = 0;
    std::uint64_t rejected_color_outlier_count = 0;
    int reliably_colored_vertex_count = 0;
    int best_view_fallback_vertex_count = 0;
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) num_threads(statistics.effectiveWorkerCount) \
    if(statistics.effectiveWorkerCount > 1) \
    reduction(+:candidate_observation_count,rejected_projection_count,rejected_mask_count) \
    reduction(+:rejected_depth_count,rejected_visibility_count,rejected_view_angle_count) \
    reduction(+:rejected_color_outlier_count,reliably_colored_vertex_count) \
    reduction(+:best_view_fallback_vertex_count)
#endif
    for (std::int64_t vertex_offset = 0;
         vertex_offset < static_cast<std::int64_t>(mesh->vertices.size());
         ++vertex_offset)
    {
        const std::size_t vertex_index = static_cast<std::size_t>(vertex_offset);
        MeshVertex &vertex = mesh->vertices[vertex_index];
        std::vector<ColorCandidate> candidates;
        candidates.reserve(static_cast<std::size_t>(views.size()));
        ColorCandidate best_fallback;
        bool has_best_fallback = false;
        const double world[3] = {vertex.x, vertex.y, vertex.z};
        for (int view_index = 0; view_index < views.size(); ++view_index)
        {
            const MeshColorView &view = views[view_index];
            if (view.colorBgr.type() != CV_8UC3 || view.depth.type() != CV_32FC1)
            {
                continue;
            }
            double pixel[2]{};
            double vertex_depth = 0.0;
            if (!view.camera.projectWorldPointWithDepth(world, pixel, vertex_depth))
            {
                ++rejected_projection_count;
                continue;
            }
            const int column = static_cast<int>(std::lround(pixel[0]));
            const int row = static_cast<int>(std::lround(pixel[1]));
            if (row < 0 || column < 0 || row >= view.colorBgr.rows || column >= view.colorBgr.cols)
            {
                ++rejected_projection_count;
                continue;
            }
            if (view.supportMask.at<std::uint8_t>(row, column) == 0 ||
                view.depthValidMask.at<std::uint8_t>(row, column) == 0)
            {
                ++rejected_mask_count;
                continue;
            }
            const float observed_depth = view.depth.at<float>(row, column);
            const float confidence = view.confidence.at<float>(row, column);
            if (!std::isfinite(observed_depth) || observed_depth <= 0.0f ||
                !std::isfinite(confidence) || confidence < options.minimumConfidence)
            {
                ++rejected_depth_count;
                continue;
            }
            const float depth_tolerance = std::max(
                options.depthToleranceVoxels * voxel_size,
                options.relativeDepthTolerance * std::fabs(static_cast<float>(vertex_depth)));
            const float depth_residual = std::fabs(
                observed_depth - static_cast<float>(vertex_depth));
            const float fallback_tolerance = std::max(
                7.5f * voxel_size,
                0.008f * std::fabs(static_cast<float>(vertex_depth)));
            if (depth_residual <= fallback_tolerance)
            {
                const cv::Vec3f fallback_color = bilinearColor(
                    view.colorBgr, pixel[0], pixel[1])
                    * gains[static_cast<std::size_t>(view_index)];
                ColorCandidate fallback;
                fallback.blue = std::clamp(fallback_color[0], 0.0f, 255.0f);
                fallback.green = std::clamp(fallback_color[1], 0.0f, 255.0f);
                fallback.red = std::clamp(fallback_color[2], 0.0f, 255.0f);
                fallback.weight = confidence * std::max(0.0f, view.qualityWeight) /
                    (1.0f + depth_residual / std::max(fallback_tolerance, 1.0e-8f));
                if (!has_best_fallback || fallback.weight > best_fallback.weight)
                {
                    best_fallback = fallback;
                    has_best_fallback = true;
                }
            }
            if (depth_residual > depth_tolerance)
            {
                ++rejected_depth_count;
                continue;
            }
            const float nearest_depth = visibility_depths[static_cast<std::size_t>(view_index)]
                                            .at<float>(row, column);
            if (std::isfinite(nearest_depth) &&
                vertex_depth > nearest_depth + options.visibilityToleranceVoxels * voxel_size)
            {
                ++rejected_visibility_count;
                continue;
            }
            const std::array<double, 3> center = view.camera.cameraCenter();
            float dx = static_cast<float>(center[0]) - vertex.x;
            float dy = static_cast<float>(center[1]) - vertex.y;
            float dz = static_cast<float>(center[2]) - vertex.z;
            const float length = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (length <= 1.0e-8f)
            {
                ++rejected_view_angle_count;
                continue;
            }
            dx /= length; dy /= length; dz /= length;
            const float view_cosine = std::fabs(vertex.nx * dx + vertex.ny * dy + vertex.nz * dz);
            if (view_cosine < options.minimumViewCosine)
            {
                ++rejected_view_angle_count;
                continue;
            }
            const cv::Vec3f color = bilinearColor(view.colorBgr, pixel[0], pixel[1])
                * gains[static_cast<std::size_t>(view_index)];
            const float residual_score = 1.0f /
                std::pow(1.0f + depth_residual / std::max(depth_tolerance, 1.0e-8f), 2.0f);
            ColorCandidate candidate;
            candidate.blue = std::clamp(color[0], 0.0f, 255.0f);
            candidate.green = std::clamp(color[1], 0.0f, 255.0f);
            candidate.red = std::clamp(color[2], 0.0f, 255.0f);
            candidate.weight = confidence * std::max(0.0f, view.qualityWeight)
                * std::pow(view_cosine, 4.0f) * residual_score;
            candidates.push_back(candidate);
            ++candidate_observation_count;
        }

        candidates = rejectColorOutliers(candidates, &rejected_color_outlier_count);
        if (static_cast<int>(candidates.size()) < options.minimumConsistentViews)
        {
            if (has_best_fallback)
            {
                vertex.r = static_cast<std::uint8_t>(std::clamp(
                    std::lround(best_fallback.red), 0l, 255l));
                vertex.g = static_cast<std::uint8_t>(std::clamp(
                    std::lround(best_fallback.green), 0l, 255l));
                vertex.b = static_cast<std::uint8_t>(std::clamp(
                    std::lround(best_fallback.blue), 0l, 255l));
                colored[vertex_index] = 1;
                reliable_color_by_vertex[vertex_index] = {vertex.r, vertex.g, vertex.b};
                ++best_view_fallback_vertex_count;
            }
            continue;
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto &left, const auto &right)
        {
            return left.weight > right.weight;
        });
        if (candidates.size() == 2)
        {
            const float dr = candidates[0].red - candidates[1].red;
            const float dg = candidates[0].green - candidates[1].green;
            const float db = candidates[0].blue - candidates[1].blue;
            if (std::sqrt(dr * dr + dg * dg + db * db) > 50.0f)
            {
                candidates.resize(1);
                ++rejected_color_outlier_count;
            }
        }
        if (static_cast<int>(candidates.size()) > options.maximumBlendedViews)
        {
            candidates.resize(static_cast<std::size_t>(options.maximumBlendedViews));
        }
        float red = 0.0f;
        float green = 0.0f;
        float blue = 0.0f;
        float weight = 0.0f;
        for (const ColorCandidate &candidate : candidates)
        {
            red += candidate.red * candidate.weight;
            green += candidate.green * candidate.weight;
            blue += candidate.blue * candidate.weight;
            weight += candidate.weight;
        }
        if (weight <= 1.0e-8f)
        {
            continue;
        }
        vertex.r = static_cast<std::uint8_t>(std::clamp(std::lround(red / weight), 0l, 255l));
        vertex.g = static_cast<std::uint8_t>(std::clamp(std::lround(green / weight), 0l, 255l));
        vertex.b = static_cast<std::uint8_t>(std::clamp(std::lround(blue / weight), 0l, 255l));
        colored[vertex_index] = 1;
        reliable_color_by_vertex[vertex_index] = {vertex.r, vertex.g, vertex.b};
        ++reliably_colored_vertex_count;
    }

    statistics.candidateObservationCount = candidate_observation_count;
    statistics.rejectedProjectionCount = rejected_projection_count;
    statistics.rejectedMaskCount = rejected_mask_count;
    statistics.rejectedDepthCount = rejected_depth_count;
    statistics.rejectedVisibilityCount = rejected_visibility_count;
    statistics.rejectedViewAngleCount = rejected_view_angle_count;
    statistics.rejectedColorOutlierCount = rejected_color_outlier_count;
    statistics.reliablyColoredVertexCount = reliably_colored_vertex_count;
    statistics.bestViewFallbackVertexCount = best_view_fallback_vertex_count;

    std::vector<std::array<std::uint8_t, 3>> reliable_colors;
    reliable_colors.reserve(mesh->vertices.size());
    for (std::size_t index = 0; index < colored.size(); ++index)
    {
        if (colored[index])
        {
            reliable_colors.push_back(reliable_color_by_vertex[index]);
        }
    }
    propagateColors(mesh, &colored, options, &statistics);
    std::array<std::uint8_t, 3> fallback{180, 180, 180};
    if (!reliable_colors.empty())
    {
        std::vector<float> reds;
        std::vector<float> greens;
        std::vector<float> blues;
        reds.reserve(reliable_colors.size());
        greens.reserve(reliable_colors.size());
        blues.reserve(reliable_colors.size());
        for (const auto &color : reliable_colors)
        {
            reds.push_back(color[0]); greens.push_back(color[1]); blues.push_back(color[2]);
        }
        fallback = {static_cast<std::uint8_t>(median(reds)),
                    static_cast<std::uint8_t>(median(greens)),
                    static_cast<std::uint8_t>(median(blues))};
    }
    int fallback_vertex_count = 0;
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) num_threads(statistics.effectiveWorkerCount) \
    if(statistics.effectiveWorkerCount > 1) reduction(+:fallback_vertex_count)
#endif
    for (std::int64_t vertex_offset = 0;
         vertex_offset < static_cast<std::int64_t>(mesh->vertices.size());
         ++vertex_offset)
    {
        const std::size_t index = static_cast<std::size_t>(vertex_offset);
        if (colored[index])
        {
            continue;
        }
        mesh->vertices[index].r = fallback[0];
        mesh->vertices[index].g = fallback[1];
        mesh->vertices[index].b = fallback[2];
        ++fallback_vertex_count;
    }
    statistics.fallbackVertexCount = fallback_vertex_count;
    if (options.coherentFacePrimaryViews)
    {
        const FaceColorCoherenceStatistics coherence = applyFaceCoherentPrimaryViews(
            mesh, views, options);
        statistics.coherentPrimaryViewFaceCount = coherence.assignedFaceCount;
        statistics.coherentPrimaryViewVertexCount = coherence.recoloredVertexCount;
    }
    statistics.cleanedSpeckleVertexCount = cleanIsolatedColorSpeckles(mesh, options);
    mesh->hasVertexColors = statistics.reliablyColoredVertexCount > 0;
    statistics.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started_at).count();
    return statistics;
}

} // namespace xjw::mesh
