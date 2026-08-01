#include "DepthConstrainedSurfaceRefiner.h"

#include "DepthTsdfSurfaceBuilder.h"
#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace xjw::mesh
{
namespace
{

struct MeshMetrics
{
    double area = 0.0;
    double absoluteVolume = 0.0;
};

struct CandidateQuality
{
    bool accepted = false;
    MeshMetrics metrics;
    std::uint64_t flippedFaceCount = 0;
    std::uint64_t degenerateFaceCount = 0;
};

struct LocalSafetyProjection
{
    bool valid = false;
    int iterationCount = 0;
    std::uint64_t rejectedFaceCount = 0;
    std::uint64_t frozenVertexCount = 0;
};

std::array<double, 3> faceCross(
    const TriMesh &mesh,
    const Triangle &face)
{
    const MeshVertex &first = mesh.vertices[
        static_cast<std::size_t>(face.v[0])];
    const MeshVertex &second = mesh.vertices[
        static_cast<std::size_t>(face.v[1])];
    const MeshVertex &third = mesh.vertices[
        static_cast<std::size_t>(face.v[2])];
    const double ab_x = second.x - first.x;
    const double ab_y = second.y - first.y;
    const double ab_z = second.z - first.z;
    const double ac_x = third.x - first.x;
    const double ac_y = third.y - first.y;
    const double ac_z = third.z - first.z;
    return {
        ab_y * ac_z - ab_z * ac_y,
        ab_z * ac_x - ab_x * ac_z,
        ab_x * ac_y - ab_y * ac_x};
}

double vectorLength(const std::array<double, 3> &value)
{
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]);
}

bool sameTopology(const TriMesh &first, const TriMesh &second);
bool sameUnorientedTopology(const TriMesh &first, const TriMesh &second);
bool validTopology(const TriMesh &mesh);

bool faceViolatesLocalSafety(
    const TriMesh &baseline,
    const TriMesh &candidate,
    std::size_t face_index,
    const DepthConstrainedSurfaceRefineOptions &options,
    bool *normal_violation,
    bool *area_violation)
{
    const auto baseline_cross =
        faceCross(baseline, baseline.faces[face_index]);
    const auto candidate_cross =
        faceCross(candidate, candidate.faces[face_index]);
    const double baseline_length = vectorLength(baseline_cross);
    const double candidate_length = vectorLength(candidate_cross);
    const bool invalid_area =
        !(baseline_length > 1.0e-15) ||
        !(candidate_length >=
          baseline_length * options.minimumFaceAreaRatio);
    bool invalid_normal = false;
    if (!invalid_area)
    {
        const double normal_dot =
            (baseline_cross[0] * candidate_cross[0] +
             baseline_cross[1] * candidate_cross[1] +
             baseline_cross[2] * candidate_cross[2]) /
            (baseline_length * candidate_length);
        invalid_normal =
            !std::isfinite(normal_dot) ||
            normal_dot < options.minimumFaceNormalDot;
    }
    if (normal_violation != nullptr)
    {
        *normal_violation = invalid_normal;
    }
    if (area_violation != nullptr)
    {
        *area_violation = invalid_area;
    }
    return invalid_area || invalid_normal;
}

LocalSafetyProjection projectCandidateToLocalSafety(
    const TriMesh &baseline,
    TriMesh *candidate,
    const DepthConstrainedSurfaceRefineOptions &options)
{
    LocalSafetyProjection projection;
    if (candidate == nullptr ||
        !sameTopology(baseline, *candidate) ||
        !validTopology(*candidate))
    {
        return projection;
    }

    std::vector<std::uint8_t> frozen_vertices(
        candidate->vertices.size(), 0);
    std::vector<std::uint8_t> rejected_faces(
        candidate->faces.size(), 0);
    while (true)
    {
        bool found_violation = false;
        std::vector<std::uint8_t> vertices_to_freeze(
            candidate->vertices.size(), 0);
        for (std::size_t face_index = 0;
             face_index < candidate->faces.size();
             ++face_index)
        {
            if (!faceViolatesLocalSafety(
                    baseline,
                    *candidate,
                    face_index,
                    options,
                    nullptr,
                    nullptr))
            {
                continue;
            }
            found_violation = true;
            rejected_faces[face_index] = 1;
            const Triangle &face = candidate->faces[face_index];
            for (const int vertex_index : face.v)
            {
                const std::size_t index =
                    static_cast<std::size_t>(vertex_index);
                vertices_to_freeze[index] = 1;
            }
        }
        if (!found_violation)
        {
            projection.valid = true;
            break;
        }
        bool froze_new_vertex = false;
        for (std::size_t index = 0;
             index < vertices_to_freeze.size();
             ++index)
        {
            if (vertices_to_freeze[index] == 0 ||
                frozen_vertices[index] != 0)
            {
                continue;
            }
            frozen_vertices[index] = 1;
            candidate->vertices[index] = baseline.vertices[index];
            froze_new_vertex = true;
        }
        if (!froze_new_vertex)
        {
            break;
        }
        ++projection.iterationCount;
    }

    projection.rejectedFaceCount = static_cast<std::uint64_t>(
        std::count(rejected_faces.begin(), rejected_faces.end(), 1));
    projection.frozenVertexCount = static_cast<std::uint64_t>(
        std::count(frozen_vertices.begin(), frozen_vertices.end(), 1));
    detail::recomputeNormals(candidate);
    return projection;
}

bool hasPositionChange(
    const TriMesh &baseline,
    const TriMesh &candidate)
{
    constexpr double minimum_squared_displacement = 1.0e-20;
    for (std::size_t index = 0; index < baseline.vertices.size(); ++index)
    {
        const double delta_x =
            candidate.vertices[index].x - baseline.vertices[index].x;
        const double delta_y =
            candidate.vertices[index].y - baseline.vertices[index].y;
        const double delta_z =
            candidate.vertices[index].z - baseline.vertices[index].z;
        if (delta_x * delta_x +
                delta_y * delta_y +
                delta_z * delta_z >
            minimum_squared_displacement)
        {
            return true;
        }
    }
    return false;
}

double robustMedian(std::vector<double> *values)
{
    if (values == nullptr || values->empty())
    {
        return 0.0;
    }
    const std::size_t middle = values->size() / 2;
    std::nth_element(
        values->begin(),
        values->begin() + static_cast<std::ptrdiff_t>(middle),
        values->end());
    const double upper = (*values)[middle];
    if ((values->size() & 1U) != 0U)
    {
        return upper;
    }
    const double lower = *std::max_element(
        values->begin(),
        values->begin() + static_cast<std::ptrdiff_t>(middle));
    return 0.5 * (lower + upper);
}

double removeMedianNormalBias(
    const TriMesh &baseline,
    TriMesh *refined)
{
    if (refined == nullptr ||
        !sameUnorientedTopology(baseline, *refined))
    {
        return 0.0;
    }
    std::vector<double> normal_displacements;
    normal_displacements.reserve(baseline.vertices.size());
    for (std::size_t index = 0; index < baseline.vertices.size(); ++index)
    {
        const MeshVertex &source = baseline.vertices[index];
        const MeshVertex &target = refined->vertices[index];
        const double normal_length = std::sqrt(
            static_cast<double>(source.nx) * source.nx +
            static_cast<double>(source.ny) * source.ny +
            static_cast<double>(source.nz) * source.nz);
        if (!(normal_length > 1.0e-12) ||
            !std::isfinite(normal_length) ||
            !std::isfinite(source.x) ||
            !std::isfinite(source.y) ||
            !std::isfinite(source.z) ||
            !std::isfinite(target.x) ||
            !std::isfinite(target.y) ||
            !std::isfinite(target.z))
        {
            continue;
        }
        const double inverse_normal_length = 1.0 / normal_length;
        const double displacement =
            (static_cast<double>(target.x) - source.x) *
                source.nx * inverse_normal_length +
            (static_cast<double>(target.y) - source.y) *
                source.ny * inverse_normal_length +
            (static_cast<double>(target.z) - source.z) *
                source.nz * inverse_normal_length;
        if (std::isfinite(displacement))
        {
            normal_displacements.push_back(displacement);
        }
    }
    const double median_bias = robustMedian(&normal_displacements);
    if (!std::isfinite(median_bias))
    {
        return 0.0;
    }
    for (std::size_t index = 0; index < baseline.vertices.size(); ++index)
    {
        const MeshVertex &source = baseline.vertices[index];
        MeshVertex &target = refined->vertices[index];
        const double normal_length = std::sqrt(
            static_cast<double>(source.nx) * source.nx +
            static_cast<double>(source.ny) * source.ny +
            static_cast<double>(source.nz) * source.nz);
        if (!(normal_length > 1.0e-12) ||
            !std::isfinite(normal_length) ||
            !std::isfinite(target.x) ||
            !std::isfinite(target.y) ||
            !std::isfinite(target.z))
        {
            continue;
        }
        const double scale = median_bias / normal_length;
        target.x -= static_cast<float>(scale * source.nx);
        target.y -= static_cast<float>(scale * source.ny);
        target.z -= static_cast<float>(scale * source.nz);
    }
    return median_bias;
}

bool sameTopology(const TriMesh &first, const TriMesh &second)
{
    if (first.vertices.size() != second.vertices.size() ||
        first.faces.size() != second.faces.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < first.faces.size(); ++index)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            if (first.faces[index].v[corner] !=
                second.faces[index].v[corner])
            {
                return false;
            }
        }
    }
    return true;
}

bool sameUnorientedTopology(
    const TriMesh &first,
    const TriMesh &second)
{
    if (first.vertices.size() != second.vertices.size() ||
        first.faces.size() != second.faces.size())
    {
        return false;
    }
    for (std::size_t index = 0; index < first.faces.size(); ++index)
    {
        std::array<int, 3> first_indices{
            first.faces[index].v[0],
            first.faces[index].v[1],
            first.faces[index].v[2]};
        std::array<int, 3> second_indices{
            second.faces[index].v[0],
            second.faces[index].v[1],
            second.faces[index].v[2]};
        std::sort(first_indices.begin(), first_indices.end());
        std::sort(second_indices.begin(), second_indices.end());
        if (first_indices != second_indices)
        {
            return false;
        }
    }
    return true;
}

bool validTopology(const TriMesh &mesh)
{
    for (const Triangle &face : mesh.faces)
    {
        for (const int vertex_index : face.v)
        {
            if (vertex_index < 0 ||
                vertex_index >= static_cast<int>(mesh.vertices.size()))
            {
                return false;
            }
        }
    }
    return true;
}

MeshMetrics measureMesh(const TriMesh &mesh)
{
    MeshMetrics metrics;
    double signed_volume_six = 0.0;
    for (const Triangle &face : mesh.faces)
    {
        const MeshVertex &first = mesh.vertices[
            static_cast<std::size_t>(face.v[0])];
        const MeshVertex &second = mesh.vertices[
            static_cast<std::size_t>(face.v[1])];
        const MeshVertex &third = mesh.vertices[
            static_cast<std::size_t>(face.v[2])];
        metrics.area += 0.5 * vectorLength(faceCross(mesh, face));
        signed_volume_six +=
            static_cast<double>(first.x) *
                (static_cast<double>(second.y) * third.z -
                 static_cast<double>(second.z) * third.y) -
            static_cast<double>(first.y) *
                (static_cast<double>(second.x) * third.z -
                 static_cast<double>(second.z) * third.x) +
            static_cast<double>(first.z) *
                (static_cast<double>(second.x) * third.y -
                 static_cast<double>(second.y) * third.x);
    }
    metrics.absoluteVolume = std::abs(signed_volume_six) / 6.0;
    return metrics;
}

CandidateQuality evaluateCandidate(
    const TriMesh &baseline,
    const TriMesh &candidate,
    const MeshMetrics &baseline_metrics,
    const DepthConstrainedSurfaceRefineOptions &options)
{
    CandidateQuality quality;
    if (!sameTopology(baseline, candidate) ||
        !validTopology(candidate))
    {
        quality.degenerateFaceCount = candidate.faces.size();
        return quality;
    }

    quality.metrics = measureMesh(candidate);
    const double area_ratio =
        quality.metrics.area / baseline_metrics.area;
    const double volume_ratio =
        quality.metrics.absoluteVolume /
        baseline_metrics.absoluteVolume;
    for (std::size_t index = 0; index < baseline.faces.size(); ++index)
    {
        const auto baseline_cross =
            faceCross(baseline, baseline.faces[index]);
        const auto candidate_cross =
            faceCross(candidate, candidate.faces[index]);
        const double baseline_length = vectorLength(baseline_cross);
        const double candidate_length = vectorLength(candidate_cross);
        if (!(baseline_length > 1.0e-15) ||
            !(candidate_length >=
              baseline_length * options.minimumFaceAreaRatio))
        {
            ++quality.degenerateFaceCount;
            continue;
        }
        const double normal_dot =
            (baseline_cross[0] * candidate_cross[0] +
             baseline_cross[1] * candidate_cross[1] +
             baseline_cross[2] * candidate_cross[2]) /
            (baseline_length * candidate_length);
        if (!std::isfinite(normal_dot) ||
            normal_dot < options.minimumFaceNormalDot)
        {
            ++quality.flippedFaceCount;
        }
    }
    quality.accepted =
        std::isfinite(area_ratio) &&
        std::isfinite(volume_ratio) &&
        area_ratio >= options.minimumAreaRatio &&
        area_ratio <= options.maximumAreaRatio &&
        volume_ratio >= options.minimumVolumeRatio &&
        volume_ratio <= options.maximumVolumeRatio &&
        quality.flippedFaceCount == 0 &&
        quality.degenerateFaceCount == 0;
    return quality;
}

TriMesh blendMeshes(
    const TriMesh &baseline,
    const TriMesh &refined,
    float blend)
{
    TriMesh candidate = baseline;
    for (std::size_t index = 0; index < candidate.vertices.size(); ++index)
    {
        MeshVertex &vertex = candidate.vertices[index];
        const MeshVertex &target = refined.vertices[index];
        vertex.x += blend * (target.x - vertex.x);
        vertex.y += blend * (target.y - vertex.y);
        vertex.z += blend * (target.z - vertex.z);
    }
    detail::recomputeNormals(&candidate);
    return candidate;
}

void accumulateRefinerStatistics(
    const VisualHullDepthRefineStatistics &pass,
    VisualHullDepthRefineStatistics *total)
{
    const bool first_solver_pass =
        total->globalSolverAttemptCount == 0 &&
        pass.globalSolverAttemptCount > 0;
    total->applied = total->applied || pass.applied;
    total->projectedObservationCount += pass.projectedObservationCount;
    total->acceptedObservationCount += pass.acceptedObservationCount;
    total->spreadDownweightedObservationCount +=
        pass.spreadDownweightedObservationCount;
    total->spreadVeryWeakObservationCount +=
        pass.spreadVeryWeakObservationCount;
    total->anchoredVertexCount += pass.anchoredVertexCount;
    total->blendedConsensusVertexCount +=
        pass.blendedConsensusVertexCount;
    total->biasCalibratedFrameCount = std::max(
        total->biasCalibratedFrameCount,
        pass.biasCalibratedFrameCount);
    total->biasCalibrationPairCount = std::max(
        total->biasCalibrationPairCount,
        pass.biasCalibrationPairCount);
    total->maximumAbsoluteFrameBias = std::max(
        total->maximumAbsoluteFrameBias,
        pass.maximumAbsoluteFrameBias);
    total->displacedVertexCount += pass.displacedVertexCount;
    total->medianSupportingViewCount = pass.medianSupportingViewCount;
    total->p90SupportingViewCount = pass.p90SupportingViewCount;
    total->maximumAppliedDisplacement = std::max(
        total->maximumAppliedDisplacement,
        pass.maximumAppliedDisplacement);
    total->medianAppliedDisplacement = pass.medianAppliedDisplacement;
    total->p90AppliedDisplacement = pass.p90AppliedDisplacement;
    total->globalSolverAttemptCount += pass.globalSolverAttemptCount;
    total->globalSolverSolvedPassCount += pass.globalSolverSolvedPassCount;
    total->globalSolverAppliedPassCount += pass.globalSolverAppliedPassCount;
    total->globalSolverConvergedPassCount +=
        pass.globalSolverConvergedPassCount;
    total->globalSolverFallbackPassCount +=
        pass.globalSolverFallbackPassCount;
    total->globalSolverCancelled =
        total->globalSolverCancelled || pass.globalSolverCancelled;
    total->globalSolverIrlsIterationCount +=
        pass.globalSolverIrlsIterationCount;
    total->globalSolverPcgIterationCount +=
        pass.globalSolverPcgIterationCount;
    total->globalSolverObservationCount += pass.globalSolverObservationCount;
    total->globalSolverRegularizationEdgeCount =
        pass.globalSolverRegularizationEdgeCount;
    total->globalSolverAnchoredVertexCount =
        pass.globalSolverAnchoredVertexCount;
    total->globalSolverPriorOnlyVertexCount =
        pass.globalSolverPriorOnlyVertexCount;
    total->globalSolverEffectiveRobustScale =
        pass.globalSolverEffectiveRobustScale;
    if (first_solver_pass)
    {
        total->globalSolverInitialEnergy = pass.globalSolverInitialEnergy;
    }
    if (pass.globalSolverAttemptCount > 0)
    {
        total->globalSolverFinalEnergy = pass.globalSolverFinalEnergy;
        total->globalSolverFinalRelativeResidual =
            pass.globalSolverFinalRelativeResidual;
    }
}

bool validOptions(const DepthConstrainedSurfaceRefineOptions &options)
{
    return options.minimumAreaRatio > 0.0 &&
           options.maximumAreaRatio >= options.minimumAreaRatio &&
           options.minimumVolumeRatio > 0.0 &&
           options.maximumVolumeRatio >= options.minimumVolumeRatio &&
           options.minimumFaceNormalDot >= -1.0 &&
           options.minimumFaceNormalDot <= 1.0 &&
           options.minimumFaceAreaRatio > 0.0 &&
           options.minimumFaceAreaRatio <= 1.0;
}

} // namespace

DepthConstrainedSurfaceRefineStatistics
DepthConstrainedSurfaceRefiner::refine(
    TriMesh *mesh,
    const QVector<DepthTsdfFrame> &frames,
    const DepthConstrainedSurfaceRefineOptions &options,
    const std::function<bool()> &isCancelled)
{
    DepthConstrainedSurfaceRefineStatistics statistics;
    if (mesh == nullptr || mesh->empty() ||
        !validTopology(*mesh) || !validOptions(options))
    {
        return statistics;
    }
    const MeshMetrics initial_metrics = measureMesh(*mesh);
    statistics.areaBefore = initial_metrics.area;
    statistics.areaAfter = initial_metrics.area;
    statistics.absoluteVolumeBefore = initial_metrics.absoluteVolume;
    statistics.absoluteVolumeAfter = initial_metrics.absoluteVolume;
    if (!(initial_metrics.area > 1.0e-15) ||
        !(initial_metrics.absoluteVolume > 1.0e-15))
    {
        return statistics;
    }

    const int pass_count = std::clamp(options.passes, 1, 4);
    constexpr std::array<float, 4> blends{1.0f, 0.75f, 0.5f, 0.25f};
    for (int pass_index = 0; pass_index < pass_count; ++pass_index)
    {
        if (isCancelled && isCancelled())
        {
            break;
        }
        const TriMesh baseline = *mesh;
        const MeshMetrics baseline_metrics = measureMesh(baseline);
        TriMesh full_refinement = baseline;
        statistics.attempted = true;
        ++statistics.attemptedPassCount;
        const VisualHullDepthRefineStatistics pass_statistics =
            VisualHullDepthRefiner::refine(
                &full_refinement,
                frames,
                options.depthRefine,
                isCancelled);
        accumulateRefinerStatistics(pass_statistics, &statistics.refiner);
        if (!pass_statistics.applied)
        {
            break;
        }
        if ((isCancelled && isCancelled()) ||
            pass_statistics.globalSolverCancelled ||
            !sameUnorientedTopology(baseline, full_refinement))
        {
            statistics.reverted = true;
            ++statistics.revertedPassCount;
            break;
        }
        const double removed_median_normal_bias =
            options.removeMedianNormalBias
            ? removeMedianNormalBias(baseline, &full_refinement)
            : 0.0;

        bool accepted = false;
        for (const float blend : blends)
        {
            TriMesh candidate =
                blendMeshes(baseline, full_refinement, blend);
            const CandidateQuality unprojected_quality = evaluateCandidate(
                baseline,
                candidate,
                baseline_metrics,
                options);
            statistics.flippedFaceCount = std::max(
                statistics.flippedFaceCount,
                unprojected_quality.flippedFaceCount);
            statistics.degenerateFaceCount = std::max(
                statistics.degenerateFaceCount,
                unprojected_quality.degenerateFaceCount);
            const LocalSafetyProjection projection =
                projectCandidateToLocalSafety(
                    baseline,
                    &candidate,
                    options);
            if (projection.frozenVertexCount > 0)
            {
                ++statistics.locallyProjectedCandidateCount;
            }
            statistics.localSafetyProjectionIterationCount +=
                projection.iterationCount;
            statistics.locallyRejectedFaceCount = std::max(
                statistics.locallyRejectedFaceCount,
                projection.rejectedFaceCount);
            statistics.locallyFrozenVertexCount = std::max(
                statistics.locallyFrozenVertexCount,
                projection.frozenVertexCount);
            if (!projection.valid ||
                !hasPositionChange(baseline, candidate))
            {
                continue;
            }
            const CandidateQuality quality = evaluateCandidate(
                baseline,
                candidate,
                baseline_metrics,
                options);
            statistics.flippedFaceCount = std::max(
                statistics.flippedFaceCount,
                quality.flippedFaceCount);
            statistics.degenerateFaceCount = std::max(
                statistics.degenerateFaceCount,
                quality.degenerateFaceCount);
            if (!quality.accepted)
            {
                continue;
            }
            *mesh = std::move(candidate);
            statistics.applied = true;
            ++statistics.appliedPassCount;
            statistics.acceptedBlend =
                statistics.appliedPassCount == 1
                ? blend
                : std::min(statistics.acceptedBlend, blend);
            statistics.acceptedLocallyRejectedFaceCount = std::max(
                statistics.acceptedLocallyRejectedFaceCount,
                projection.rejectedFaceCount);
            statistics.acceptedLocallyFrozenVertexCount = std::max(
                statistics.acceptedLocallyFrozenVertexCount,
                projection.frozenVertexCount);
            statistics.removedMedianNormalBias +=
                removed_median_normal_bias;
            accepted = true;
            break;
        }
        if (!accepted)
        {
            statistics.reverted = true;
            ++statistics.revertedPassCount;
            break;
        }
    }

    const MeshMetrics final_metrics = measureMesh(*mesh);
    statistics.areaAfter = final_metrics.area;
    statistics.absoluteVolumeAfter = final_metrics.absoluteVolume;
    return statistics;
}

} // namespace xjw::mesh
