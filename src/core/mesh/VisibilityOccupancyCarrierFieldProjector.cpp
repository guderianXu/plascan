#include "VisibilityOccupancyCarrierFieldProjector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace xjw::mesh
{
namespace
{

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct FaceReference
{
    Vec3 cross;
    double crossLength = 0.0;
};

struct MeshMetrics
{
    double area = 0.0;
    double absoluteVolume = 0.0;
};

struct FieldSample
{
    double value = 0.0;
    Vec3 gradient;
};

struct ResidualMetrics
{
    double mean = 0.0;
    double p90 = 0.0;
    double maximum = 0.0;
};

struct LocalSafetyResult
{
    bool valid = false;
    bool cancelled = false;
};

Vec3 position(const MeshVertex &vertex)
{
    return {vertex.x, vertex.y, vertex.z};
}

Vec3 subtract(const Vec3 &first, const Vec3 &second)
{
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

Vec3 add(const Vec3 &first, const Vec3 &second)
{
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

Vec3 multiply(const Vec3 &value, double scale)
{
    return {value.x * scale, value.y * scale, value.z * scale};
}

double dot(const Vec3 &first, const Vec3 &second)
{
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

Vec3 cross(const Vec3 &first, const Vec3 &second)
{
    return {
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x};
}

double length(const Vec3 &value)
{
    return std::sqrt(dot(value, value));
}

Vec3 faceCross(const TriMesh &mesh, const Triangle &face)
{
    const Vec3 first = position(mesh.vertices[face.v[0]]);
    const Vec3 second = position(mesh.vertices[face.v[1]]);
    const Vec3 third = position(mesh.vertices[face.v[2]]);
    return cross(subtract(second, first), subtract(third, first));
}

bool isCancelled(
    const VisibilityOccupancyCarrierFieldProjectionOptions &options)
{
    return options.isCancelled && options.isCancelled();
}

std::size_t gridIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(dimensions[1]) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(dimensions[0]) +
           static_cast<std::size_t>(x);
}

bool checkedSampleCount(
    const std::array<int, 3> &dimensions,
    std::size_t *sampleCount)
{
    std::size_t count = 1;
    for (const int dimension : dimensions)
    {
        if (dimension < 2)
        {
            return false;
        }
        const std::size_t extent = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / extent)
        {
            return false;
        }
        count *= extent;
    }
    *sampleCount = count;
    return true;
}

bool validOptions(
    const VisibilityOccupancyCarrierFieldProjectionOptions &options)
{
    return options.iterations >= 0 &&
        options.maximumBacktrackingSteps >= 0 &&
        options.maximumBacktrackingSteps <= 30 &&
        std::isfinite(options.relaxation) &&
        options.relaxation > 0.0 && options.relaxation <= 1.0 &&
        std::isfinite(options.maximumStepSpacingRatio) &&
        options.maximumStepSpacingRatio >= 0.0 &&
        std::isfinite(options.maximumCumulativeDisplacementSpacingRatio) &&
        options.maximumCumulativeDisplacementSpacingRatio >= 0.0 &&
        std::isfinite(options.narrowBandWidthSpacingRatio) &&
        options.narrowBandWidthSpacingRatio >= 0.0 &&
        std::isfinite(options.scalarSmoothingRelaxation) &&
        options.scalarSmoothingRelaxation >= 0.0 &&
        options.scalarSmoothingRelaxation <= 1.0 &&
        std::isfinite(options.minimumNormalDot) &&
        options.minimumNormalDot >= -1.0 &&
        options.minimumNormalDot <= 1.0 &&
        std::isfinite(options.minimumFaceAreaRatio) &&
        options.minimumFaceAreaRatio > 0.0 &&
        std::isfinite(options.minimumSurfaceAreaRatio) &&
        std::isfinite(options.maximumSurfaceAreaRatio) &&
        options.minimumSurfaceAreaRatio > 0.0 &&
        options.minimumSurfaceAreaRatio <= 1.0 &&
        options.maximumSurfaceAreaRatio >= 1.0 &&
        options.minimumSurfaceAreaRatio <= options.maximumSurfaceAreaRatio &&
        std::isfinite(options.minimumAbsoluteVolumeRatio) &&
        std::isfinite(options.maximumAbsoluteVolumeRatio) &&
        options.minimumAbsoluteVolumeRatio >= 0.0 &&
        options.minimumAbsoluteVolumeRatio <= 1.0 &&
        options.maximumAbsoluteVolumeRatio >= 1.0 &&
        options.minimumAbsoluteVolumeRatio <=
            options.maximumAbsoluteVolumeRatio;
}

bool validateGrid(
    const std::array<int, 3> &dimensions,
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::vector<float> &field,
    std::array<double, 3> *spacing,
    std::string *error)
{
    std::size_t sample_count = 0;
    if (!checkedSampleCount(dimensions, &sample_count))
    {
        *error = "sample dimensions must be at least two and must not overflow";
        return false;
    }
    if (field.size() != sample_count)
    {
        *error = "signed field sample count does not match dimensions";
        return false;
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(boundsMin[axis]) ||
            !std::isfinite(boundsMax[axis]) ||
            !(boundsMax[axis] > boundsMin[axis]))
        {
            *error = "field bounds must be finite and strictly increasing";
            return false;
        }
        (*spacing)[axis] =
            (static_cast<double>(boundsMax[axis]) - boundsMin[axis]) /
            static_cast<double>(dimensions[axis] - 1);
    }
    if (std::any_of(field.begin(), field.end(), [](float value)
        {
            return !std::isfinite(value);
        }))
    {
        *error = "signed field contains a non-finite sample";
        return false;
    }
    return true;
}

MeshMetrics measureMesh(const TriMesh &mesh)
{
    MeshMetrics metrics;
    const Vec3 origin = position(mesh.vertices.front());
    double signed_volume_six = 0.0;
    for (const Triangle &face : mesh.faces)
    {
        const Vec3 first = subtract(position(mesh.vertices[face.v[0]]), origin);
        const Vec3 second = subtract(position(mesh.vertices[face.v[1]]), origin);
        const Vec3 third = subtract(position(mesh.vertices[face.v[2]]), origin);
        const Vec3 face_cross = cross(
            subtract(second, first), subtract(third, first));
        metrics.area += 0.5 * length(face_cross);
        signed_volume_six += dot(first, cross(second, third));
    }
    metrics.absoluteVolume = std::abs(signed_volume_six) / 6.0;
    return metrics;
}

bool sampleField(
    const std::array<int, 3> &dimensions,
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::array<double, 3> &spacing,
    const std::vector<float> &field,
    const Vec3 &point,
    FieldSample *sample)
{
    const std::array<double, 3> world{point.x, point.y, point.z};
    std::array<int, 3> base{};
    std::array<double, 3> fraction{};
    for (int axis = 0; axis < 3; ++axis)
    {
        if (world[axis] < boundsMin[axis] || world[axis] > boundsMax[axis])
        {
            return false;
        }
        const double coordinate = std::clamp(
            (world[axis] - boundsMin[axis]) / spacing[axis],
            0.0,
            static_cast<double>(dimensions[axis] - 1));
        base[axis] = std::min(
            static_cast<int>(std::floor(coordinate)), dimensions[axis] - 2);
        fraction[axis] = coordinate - static_cast<double>(base[axis]);
    }

    double corner[2][2][2]{};
    for (int z = 0; z < 2; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 2; ++x)
            {
                corner[z][y][x] = field[gridIndex(
                    dimensions, base[0] + x, base[1] + y, base[2] + z)];
            }
        }
    }

    const double tx = fraction[0];
    const double ty = fraction[1];
    const double tz = fraction[2];
    const auto lerp = [](double first, double second, double t)
    {
        return first + (second - first) * t;
    };
    const double c00 = lerp(corner[0][0][0], corner[0][0][1], tx);
    const double c10 = lerp(corner[0][1][0], corner[0][1][1], tx);
    const double c01 = lerp(corner[1][0][0], corner[1][0][1], tx);
    const double c11 = lerp(corner[1][1][0], corner[1][1][1], tx);
    const double c0 = lerp(c00, c10, ty);
    const double c1 = lerp(c01, c11, ty);
    sample->value = lerp(c0, c1, tz);

    const double dx0 = lerp(
        corner[0][0][1] - corner[0][0][0],
        corner[0][1][1] - corner[0][1][0], ty);
    const double dx1 = lerp(
        corner[1][0][1] - corner[1][0][0],
        corner[1][1][1] - corner[1][1][0], ty);
    const double dy0 = lerp(
        corner[0][1][0] - corner[0][0][0],
        corner[0][1][1] - corner[0][0][1], tx);
    const double dy1 = lerp(
        corner[1][1][0] - corner[1][0][0],
        corner[1][1][1] - corner[1][0][1], tx);
    const double dz0 = lerp(corner[1][0][0], corner[1][0][1], tx);
    const double dz1 = lerp(corner[1][1][0], corner[1][1][1], tx);
    sample->gradient = {
        lerp(dx0, dx1, tz) / spacing[0],
        lerp(dy0, dy1, tz) / spacing[1],
        (lerp(dz0, dz1, ty) - c0) / spacing[2]};
    return std::isfinite(sample->value) &&
        std::isfinite(sample->gradient.x) &&
        std::isfinite(sample->gradient.y) &&
        std::isfinite(sample->gradient.z);
}

std::vector<float> smoothField(
    const std::array<int, 3> &dimensions,
    const std::vector<float> &source,
    double minimumSpacing,
    const VisibilityOccupancyCarrierFieldProjectionOptions &options,
    VisibilityOccupancyCarrierFieldProjectionStatistics *statistics,
    bool *cancelled)
{
    if (!options.smoothNarrowBand ||
        options.scalarSmoothingRelaxation == 0.0 ||
        options.narrowBandWidthSpacingRatio == 0.0)
    {
        return source;
    }
    std::vector<float> result = source;
    const double band = options.narrowBandWidthSpacingRatio * minimumSpacing;
    constexpr std::array<std::array<int, 3>, 6> offsets{{
        {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
        {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}}};
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                if (isCancelled(options))
                {
                    *cancelled = true;
                    return source;
                }
                const std::size_t index = gridIndex(dimensions, x, y, z);
                const float original = source[index];
                if (original == 0.0f || std::abs(original) > band)
                {
                    continue;
                }
                double magnitude_sum = std::abs(original);
                int count = 1;
                for (const auto &offset : offsets)
                {
                    const int nx = x + offset[0];
                    const int ny = y + offset[1];
                    const int nz = z + offset[2];
                    if (nx < 0 || nx >= dimensions[0] ||
                        ny < 0 || ny >= dimensions[1] ||
                        nz < 0 || nz >= dimensions[2])
                    {
                        continue;
                    }
                    const float neighbour = source[gridIndex(
                        dimensions, nx, ny, nz)];
                    if (neighbour == 0.0f ||
                        std::signbit(neighbour) != std::signbit(original))
                    {
                        continue;
                    }
                    magnitude_sum += std::abs(neighbour);
                    ++count;
                }
                const double mean_magnitude = magnitude_sum / count;
                double magnitude =
                    (1.0 - options.scalarSmoothingRelaxation) *
                        std::abs(original) +
                    options.scalarSmoothingRelaxation * mean_magnitude;
                magnitude = std::max(
                    magnitude, static_cast<double>(std::numeric_limits<float>::min()));
                result[index] = std::copysign(
                    static_cast<float>(magnitude), original);
                if (result[index] != original)
                {
                    ++statistics->smoothedSampleCount;
                }
            }
        }
    }
    statistics->fieldSmoothingApplied = true;
    return result;
}

bool faceIsSafe(
    const TriMesh &candidate,
    std::size_t faceIndex,
    const FaceReference &reference,
    const VisibilityOccupancyCarrierFieldProjectionOptions &options)
{
    const Vec3 candidate_cross = faceCross(candidate, candidate.faces[faceIndex]);
    const double candidate_length = length(candidate_cross);
    if (!std::isfinite(candidate_length) ||
        candidate_length < reference.crossLength * options.minimumFaceAreaRatio)
    {
        return false;
    }
    const double normal_dot = dot(reference.cross, candidate_cross) /
        (reference.crossLength * candidate_length);
    return std::isfinite(normal_dot) && normal_dot >= options.minimumNormalDot;
}

bool changed(const MeshVertex &first, const MeshVertex &second)
{
    return first.x != second.x || first.y != second.y || first.z != second.z;
}

LocalSafetyResult enforceLocalSafety(
    const TriMesh &current,
    const std::vector<FaceReference> &references,
    const VisibilityOccupancyCarrierFieldProjectionOptions &options,
    std::vector<std::uint8_t> *rejectedFaces,
    std::vector<std::uint8_t> *frozenVertices,
    TriMesh *candidate)
{
    LocalSafetyResult result;
    while (true)
    {
        std::vector<std::uint8_t> freeze(candidate->vertices.size(), 0);
        bool found_violation = false;
        for (std::size_t face_index = 0;
             face_index < candidate->faces.size(); ++face_index)
        {
            if (isCancelled(options))
            {
                result.cancelled = true;
                return result;
            }
            if (faceIsSafe(*candidate, face_index, references[face_index], options))
            {
                continue;
            }
            found_violation = true;
            (*rejectedFaces)[face_index] = 1;
            for (const int vertex_index : candidate->faces[face_index].v)
            {
                if (changed(candidate->vertices[vertex_index],
                            current.vertices[vertex_index]))
                {
                    freeze[vertex_index] = 1;
                }
            }
        }
        if (!found_violation)
        {
            result.valid = true;
            return result;
        }
        bool froze_any = false;
        for (std::size_t index = 0; index < freeze.size(); ++index)
        {
            if (freeze[index] == 0)
            {
                continue;
            }
            candidate->vertices[index].x = current.vertices[index].x;
            candidate->vertices[index].y = current.vertices[index].y;
            candidate->vertices[index].z = current.vertices[index].z;
            (*frozenVertices)[index] = 1;
            froze_any = true;
        }
        if (!froze_any)
        {
            return result;
        }
    }
}

bool globalMetricsSafe(
    const MeshMetrics &initial,
    const MeshMetrics &candidate,
    const VisibilityOccupancyCarrierFieldProjectionOptions &options,
    double volumeTolerance,
    double *areaRatio,
    double *volumeRatio)
{
    *areaRatio = candidate.area / initial.area;
    const bool check_volume = initial.absoluteVolume > volumeTolerance;
    *volumeRatio = check_volume
        ? candidate.absoluteVolume / initial.absoluteVolume
        : 1.0;
    return std::isfinite(*areaRatio) && std::isfinite(*volumeRatio) &&
        *areaRatio >= options.minimumSurfaceAreaRatio &&
        *areaRatio <= options.maximumSurfaceAreaRatio &&
        (!check_volume ||
         (*volumeRatio >= options.minimumAbsoluteVolumeRatio &&
          *volumeRatio <= options.maximumAbsoluteVolumeRatio));
}

ResidualMetrics measureResiduals(
    const TriMesh &mesh,
    const std::array<int, 3> &dimensions,
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::array<double, 3> &spacing,
    const std::vector<float> &field)
{
    ResidualMetrics result;
    std::vector<double> residuals;
    residuals.reserve(mesh.vertices.size());
    double sum = 0.0;
    for (const MeshVertex &vertex : mesh.vertices)
    {
        FieldSample sample;
        if (!sampleField(dimensions, boundsMin, boundsMax, spacing,
                         field, position(vertex), &sample))
        {
            continue;
        }
        const double residual = std::abs(sample.value);
        residuals.push_back(residual);
        sum += residual;
        result.maximum = std::max(result.maximum, residual);
    }
    if (residuals.empty())
    {
        return result;
    }
    result.mean = sum / static_cast<double>(residuals.size());
    const std::size_t p90_index = static_cast<std::size_t>(std::ceil(
        0.90 * static_cast<double>(residuals.size()))) - 1U;
    std::nth_element(
        residuals.begin(), residuals.begin() + p90_index, residuals.end());
    result.p90 = residuals[p90_index];
    return result;
}

bool residualsDoNotWorsen(
    const ResidualMetrics &current,
    const ResidualMetrics &candidate)
{
    const double mean_tolerance = std::max(1.0e-12, current.mean * 1.0e-6);
    const double p90_tolerance = std::max(1.0e-12, current.p90 * 1.0e-6);
    return candidate.mean <= current.mean + mean_tolerance &&
        candidate.p90 <= current.p90 + p90_tolerance;
}

VisibilityOccupancyCarrierFieldProjectionResult atomicExit(
    VisibilityOccupancyCarrierFieldProjectionResult result,
    const TriMesh &original,
    const MeshMetrics &initial,
    bool cancelled,
    std::string message)
{
    result.ok = false;
    result.cancelled = cancelled;
    result.rolledBack = true;
    result.errorMessage = std::move(message);
    result.mesh = original;
    ++result.statistics.rollbackCount;
    result.statistics.finalSurfaceArea = initial.area;
    result.statistics.finalAbsoluteVolume = initial.absoluteVolume;
    result.statistics.finalSurfaceAreaRatio = 1.0;
    result.statistics.finalAbsoluteVolumeRatio = 1.0;
    result.statistics.meanAbsoluteFieldResidualAfter =
        result.statistics.meanAbsoluteFieldResidualBefore;
    result.statistics.p90AbsoluteFieldResidualAfter =
        result.statistics.p90AbsoluteFieldResidualBefore;
    result.statistics.maximumAbsoluteFieldResidualAfter =
        result.statistics.maximumAbsoluteFieldResidualBefore;
    return result;
}

} // namespace

VisibilityOccupancyCarrierFieldProjectionResult
VisibilityOccupancyCarrierFieldProjector::project(
    const TriMesh &mesh,
    const std::array<int, 3> &sampleDimensions,
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::vector<float> &signedWorldDistance,
    const VisibilityOccupancyCarrierFieldProjectionOptions &options)
{
    VisibilityOccupancyCarrierFieldProjectionResult result;
    result.mesh = mesh;
    result.statistics.inputVertexCount = mesh.vertices.size();
    result.statistics.inputFaceCount = mesh.faces.size();
    result.statistics.requestedIterationCount = options.iterations;
    if (!validOptions(options))
    {
        result.errorMessage = "invalid carrier field projection options";
        return result;
    }
    if (mesh.empty())
    {
        result.errorMessage = "carrier mesh must contain vertices and faces";
        return result;
    }
    std::array<double, 3> spacing{};
    if (!validateGrid(sampleDimensions, boundsMin, boundsMax,
                      signedWorldDistance, &spacing, &result.errorMessage))
    {
        return result;
    }
    result.statistics.minimumSpacing =
        std::min({spacing[0], spacing[1], spacing[2]});
    result.statistics.resolvedMaximumStep =
        options.maximumStepSpacingRatio * result.statistics.minimumSpacing;
    result.statistics.resolvedMaximumCumulativeDisplacement =
        options.maximumCumulativeDisplacementSpacingRatio *
        result.statistics.minimumSpacing;

    std::vector<FaceReference> face_references;
    face_references.reserve(mesh.faces.size());
    for (const MeshVertex &vertex : mesh.vertices)
    {
        if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
            !std::isfinite(vertex.z))
        {
            result.errorMessage = "carrier mesh contains a non-finite vertex";
            return result;
        }
        FieldSample unused;
        if (!sampleField(sampleDimensions, boundsMin, boundsMax, spacing,
                         signedWorldDistance, position(vertex), &unused))
        {
            result.errorMessage = "carrier mesh vertex lies outside field bounds";
            return result;
        }
    }
    for (const Triangle &face : mesh.faces)
    {
        for (const int vertex_index : face.v)
        {
            if (vertex_index < 0 ||
                static_cast<std::size_t>(vertex_index) >= mesh.vertices.size())
            {
                result.errorMessage = "carrier mesh has an invalid face index";
                return result;
            }
        }
        if (face.v[0] == face.v[1] || face.v[1] == face.v[2] ||
            face.v[2] == face.v[0])
        {
            result.errorMessage = "carrier mesh has a repeated face index";
            return result;
        }
        const Vec3 face_cross = faceCross(mesh, face);
        const double cross_length = length(face_cross);
        if (!(cross_length > std::numeric_limits<double>::epsilon()) ||
            !std::isfinite(cross_length))
        {
            result.errorMessage = "carrier mesh has a degenerate face";
            return result;
        }
        face_references.push_back({face_cross, cross_length});
    }

    const MeshMetrics initial = measureMesh(mesh);
    result.statistics.initialSurfaceArea = initial.area;
    result.statistics.initialAbsoluteVolume = initial.absoluteVolume;
    if (!(initial.area > std::numeric_limits<double>::epsilon()) ||
        !std::isfinite(initial.area) || !std::isfinite(initial.absoluteVolume))
    {
        result.errorMessage = "carrier mesh has invalid area or volume";
        return result;
    }
    const double extent_x = boundsMax[0] - boundsMin[0];
    const double extent_y = boundsMax[1] - boundsMin[1];
    const double extent_z = boundsMax[2] - boundsMin[2];
    const double volume_tolerance =
        extent_x * extent_y * extent_z * 1.0e-12;

    if (isCancelled(options))
    {
        return atomicExit(std::move(result), mesh, initial, true,
                          "visibility occupancy field projection cancelled");
    }
    bool smoothing_cancelled = false;
    std::vector<float> field = smoothField(
        sampleDimensions, signedWorldDistance,
        result.statistics.minimumSpacing, options, &result.statistics,
        &smoothing_cancelled);
    if (smoothing_cancelled)
    {
        return atomicExit(std::move(result), mesh, initial, true,
                          "visibility occupancy field projection cancelled");
    }
    ResidualMetrics current_residual = measureResiduals(
        mesh, sampleDimensions, boundsMin, boundsMax, spacing, field);
    result.statistics.meanAbsoluteFieldResidualBefore = current_residual.mean;
    result.statistics.p90AbsoluteFieldResidualBefore = current_residual.p90;
    result.statistics.maximumAbsoluteFieldResidualBefore =
        current_residual.maximum;

    TriMesh current = mesh;
    std::vector<std::uint8_t> step_clamped(mesh.vertices.size(), 0);
    std::vector<std::uint8_t> cumulative_clamped(mesh.vertices.size(), 0);
    std::vector<std::uint8_t> rejected_faces(mesh.faces.size(), 0);
    std::vector<std::uint8_t> frozen_vertices(mesh.vertices.size(), 0);
    for (int iteration = 0; iteration < options.iterations; ++iteration)
    {
        TriMesh raw_candidate = current;
        for (std::size_t index = 0; index < current.vertices.size(); ++index)
        {
            if (isCancelled(options))
            {
                return atomicExit(std::move(result), mesh, initial, true,
                                  "visibility occupancy field projection cancelled");
            }
            FieldSample sample;
            if (!sampleField(sampleDimensions, boundsMin, boundsMax, spacing,
                             field, position(current.vertices[index]), &sample))
            {
                continue;
            }
            const double gradient_squared = dot(sample.gradient, sample.gradient);
            if (!(gradient_squared > 1.0e-20) || !std::isfinite(gradient_squared))
            {
                continue;
            }
            Vec3 step = multiply(
                sample.gradient,
                -options.relaxation * sample.value / gradient_squared);
            double step_length = length(step);
            if (step_length > result.statistics.resolvedMaximumStep &&
                step_length > 0.0)
            {
                step = multiply(
                    step, result.statistics.resolvedMaximumStep / step_length);
                step_length = result.statistics.resolvedMaximumStep;
                step_clamped[index] = 1;
            }
            Vec3 proposal = add(position(current.vertices[index]), step);
            Vec3 from_original = subtract(
                proposal, position(mesh.vertices[index]));
            const double cumulative_length = length(from_original);
            if (cumulative_length >
                    result.statistics.resolvedMaximumCumulativeDisplacement &&
                cumulative_length > 0.0)
            {
                from_original = multiply(
                    from_original,
                    result.statistics.resolvedMaximumCumulativeDisplacement /
                        cumulative_length);
                proposal = add(position(mesh.vertices[index]), from_original);
                cumulative_clamped[index] = 1;
            }
            if (proposal.x < boundsMin[0] || proposal.x > boundsMax[0] ||
                proposal.y < boundsMin[1] || proposal.y > boundsMax[1] ||
                proposal.z < boundsMin[2] || proposal.z > boundsMax[2])
            {
                continue;
            }
            raw_candidate.vertices[index].x = static_cast<float>(proposal.x);
            raw_candidate.vertices[index].y = static_cast<float>(proposal.y);
            raw_candidate.vertices[index].z = static_cast<float>(proposal.z);
        }

        bool accepted = false;
        TriMesh accepted_candidate;
        ResidualMetrics accepted_residual;
        for (int backtracking_step = 0;
             backtracking_step <= options.maximumBacktrackingSteps;
             ++backtracking_step)
        {
            if (isCancelled(options))
            {
                return atomicExit(std::move(result), mesh, initial, true,
                                  "visibility occupancy field projection cancelled");
            }
            const double blend = std::ldexp(1.0, -backtracking_step);
            ++result.statistics.attemptedBlendCount;
            if (backtracking_step > 0)
            {
                ++result.statistics.backtrackingAttemptCount;
            }

            TriMesh candidate = raw_candidate;
            for (std::size_t index = 0;
                 index < candidate.vertices.size(); ++index)
            {
                const Vec3 current_position = position(current.vertices[index]);
                const Vec3 raw_position = position(raw_candidate.vertices[index]);
                const Vec3 blended = add(
                    current_position,
                    multiply(subtract(raw_position, current_position), blend));
                candidate.vertices[index].x = static_cast<float>(blended.x);
                candidate.vertices[index].y = static_cast<float>(blended.y);
                candidate.vertices[index].z = static_cast<float>(blended.z);
            }

            const LocalSafetyResult local = enforceLocalSafety(
                current, face_references, options, &rejected_faces,
                &frozen_vertices, &candidate);
            if (local.cancelled)
            {
                return atomicExit(std::move(result), mesh, initial, true,
                                  "visibility occupancy field projection cancelled");
            }
            if (!local.valid)
            {
                ++result.statistics.rejectedBlendCount;
                continue;
            }

            const MeshMetrics metrics = measureMesh(candidate);
            double area_ratio = 1.0;
            double volume_ratio = 1.0;
            if (!globalMetricsSafe(initial, metrics, options, volume_tolerance,
                                   &area_ratio, &volume_ratio))
            {
                ++result.statistics.rejectedBlendCount;
                continue;
            }
            const ResidualMetrics candidate_residual = measureResiduals(
                candidate, sampleDimensions, boundsMin, boundsMax,
                spacing, field);
            if (!residualsDoNotWorsen(current_residual, candidate_residual))
            {
                ++result.statistics.rejectedBlendCount;
                continue;
            }

            accepted = true;
            accepted_candidate = std::move(candidate);
            accepted_residual = candidate_residual;
            result.statistics.minimumAcceptedBlend = std::min(
                result.statistics.minimumAcceptedBlend, blend);
            if (backtracking_step == 0)
            {
                ++result.statistics.acceptedFullStepCount;
            }
            else
            {
                ++result.statistics.acceptedHalfStepCount;
            }
            break;
        }
        if (!accepted)
        {
            return atomicExit(
                std::move(result), mesh, initial, false,
                "all carrier projection blend candidates violated local, global, or residual guards");
        }
        current = std::move(accepted_candidate);
        current_residual = accepted_residual;
        ++result.statistics.completedIterationCount;
    }

    if (isCancelled(options))
    {
        return atomicExit(std::move(result), mesh, initial, true,
                          "visibility occupancy field projection cancelled");
    }

    result.mesh = std::move(current);
    const MeshMetrics final_metrics = measureMesh(result.mesh);
    result.statistics.finalSurfaceArea = final_metrics.area;
    result.statistics.finalAbsoluteVolume = final_metrics.absoluteVolume;
    result.statistics.finalSurfaceAreaRatio = final_metrics.area / initial.area;
    result.statistics.finalAbsoluteVolumeRatio =
        initial.absoluteVolume > volume_tolerance
            ? final_metrics.absoluteVolume / initial.absoluteVolume
            : 1.0;
    for (std::size_t index = 0; index < result.mesh.vertices.size(); ++index)
    {
        if (changed(result.mesh.vertices[index], mesh.vertices[index]))
        {
            ++result.statistics.projectedVertexCount;
        }
    }
    result.statistics.stepClampedVertexCount = static_cast<std::uint64_t>(
        std::count(step_clamped.begin(), step_clamped.end(), 1));
    result.statistics.cumulativeClampedVertexCount = static_cast<std::uint64_t>(
        std::count(cumulative_clamped.begin(), cumulative_clamped.end(), 1));
    result.statistics.locallyRejectedFaceCount = static_cast<std::uint64_t>(
        std::count(rejected_faces.begin(), rejected_faces.end(), 1));
    result.statistics.locallyFrozenVertexCount = static_cast<std::uint64_t>(
        std::count(frozen_vertices.begin(), frozen_vertices.end(), 1));
    result.statistics.meanAbsoluteFieldResidualAfter = current_residual.mean;
    result.statistics.p90AbsoluteFieldResidualAfter = current_residual.p90;
    result.statistics.maximumAbsoluteFieldResidualAfter =
        current_residual.maximum;
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
