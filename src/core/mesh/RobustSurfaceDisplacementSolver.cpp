#include "RobustSurfaceDisplacementSolver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace xjw::mesh
{
namespace
{

struct WeightedEdge
{
    int first = -1;
    int second = -1;
    float weight = 0.0f;
};

bool validNormal(const MeshVertex &vertex)
{
    const float length_squared =
        vertex.nx * vertex.nx +
        vertex.ny * vertex.ny +
        vertex.nz * vertex.nz;
    return std::isfinite(length_squared) && length_squared > 1.0e-12f;
}

float normalDot(const MeshVertex &first, const MeshVertex &second)
{
    if (!validNormal(first) || !validNormal(second))
    {
        return 1.0f;
    }
    const float first_length = std::sqrt(
        first.nx * first.nx +
        first.ny * first.ny +
        first.nz * first.nz);
    const float second_length = std::sqrt(
        second.nx * second.nx +
        second.ny * second.ny +
        second.nz * second.nz);
    return std::clamp(
        (first.nx * second.nx +
            first.ny * second.ny +
            first.nz * second.nz) /
            std::max(1.0e-12f, first_length * second_length),
        -1.0f,
        1.0f);
}

bool validObservation(
    const RobustSurfaceDisplacementObservation &observation,
    std::size_t vertex_count)
{
    return observation.vertexIndex >= 0 &&
        observation.vertexIndex < static_cast<int>(vertex_count) &&
        std::isfinite(observation.target) &&
        std::isfinite(observation.weight) &&
        observation.weight > 0.0f;
}

bool finiteVector(const std::vector<float> &values)
{
    return std::all_of(
        values.cbegin(),
        values.cend(),
        [](float value)
        {
            return std::isfinite(value);
        });
}

std::vector<WeightedEdge> buildEdges(
    const TriMesh &mesh,
    float minimum_normal_dot)
{
    std::vector<std::pair<int, int>> unique_edges;
    unique_edges.reserve(mesh.faces.size() * 3);
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(mesh.faces.size() * 3);
    for (const Triangle &face : mesh.faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            int first = face.v[corner];
            int second = face.v[(corner + 1) % 3];
            if (first < 0 || second < 0 || first == second ||
                first >= static_cast<int>(mesh.vertices.size()) ||
                second >= static_cast<int>(mesh.vertices.size()))
            {
                continue;
            }
            if (first > second)
            {
                std::swap(first, second);
            }
            const std::uint64_t key =
                (static_cast<std::uint64_t>(
                     static_cast<std::uint32_t>(first))
                 << 32U) |
                static_cast<std::uint32_t>(second);
            if (seen.insert(key).second)
            {
                unique_edges.emplace_back(first, second);
            }
        }
    }

    std::vector<int> degree(mesh.vertices.size(), 0);
    for (const auto &[first, second] : unique_edges)
    {
        ++degree[static_cast<std::size_t>(first)];
        ++degree[static_cast<std::size_t>(second)];
    }

    std::vector<WeightedEdge> edges;
    edges.reserve(unique_edges.size());
    const float bounded_minimum_dot =
        std::clamp(minimum_normal_dot, -1.0f, 0.999f);
    for (const auto &[first, second] : unique_edges)
    {
        const float dot = normalDot(
            mesh.vertices[static_cast<std::size_t>(first)],
            mesh.vertices[static_cast<std::size_t>(second)]);
        if (dot <= bounded_minimum_dot)
        {
            continue;
        }
        const float feature_weight =
            (dot - bounded_minimum_dot) /
            std::max(1.0e-6f, 1.0f - bounded_minimum_dot);
        const float degree_scale = std::sqrt(
            static_cast<float>(std::max(1, degree[static_cast<std::size_t>(first)])) *
            static_cast<float>(std::max(1, degree[static_cast<std::size_t>(second)])));
        edges.push_back(
            {first,
             second,
             feature_weight * feature_weight /
                 std::max(1.0f, degree_scale)});
    }
    return edges;
}

double dotProduct(
    const std::vector<float> &first,
    const std::vector<float> &second)
{
    double value = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        value +=
            static_cast<double>(first[index]) *
            static_cast<double>(second[index]);
    }
    return value;
}

void applySystem(
    const std::vector<float> &data_diagonal,
    const std::vector<WeightedEdge> &edges,
    float laplacian_weight,
    const std::vector<float> &input,
    std::vector<float> *output)
{
    output->resize(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        (*output)[index] = data_diagonal[index] * input[index];
    }
    for (const WeightedEdge &edge : edges)
    {
        const float weighted_difference =
            laplacian_weight * edge.weight *
            (input[static_cast<std::size_t>(edge.first)] -
             input[static_cast<std::size_t>(edge.second)]);
        (*output)[static_cast<std::size_t>(edge.first)] +=
            weighted_difference;
        (*output)[static_cast<std::size_t>(edge.second)] -=
            weighted_difference;
    }
}

double robustEnergy(
    const std::vector<RobustSurfaceDisplacementObservation> &observations,
    const std::vector<float> &normalized_observation_weight,
    const std::vector<WeightedEdge> &edges,
    const std::vector<float> &displacement,
    float robust_scale,
    float laplacian_weight,
    float hull_prior_weight)
{
    const double scale_squared =
        static_cast<double>(robust_scale) *
        static_cast<double>(robust_scale);
    double energy = 0.0;
    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        const RobustSurfaceDisplacementObservation &observation =
            observations[index];
        const float weight =
            index < normalized_observation_weight.size()
            ? normalized_observation_weight[index]
            : 0.0f;
        if (!validObservation(observation, displacement.size()) ||
            !(weight > 0.0f))
        {
            continue;
        }
        const double residual =
            static_cast<double>(
                displacement[static_cast<std::size_t>(
                    observation.vertexIndex)] -
                observation.target);
        energy += static_cast<double>(weight) *
            scale_squared *
            std::log1p(residual * residual / scale_squared);
    }
    for (const WeightedEdge &edge : edges)
    {
        const double difference =
            static_cast<double>(
                displacement[static_cast<std::size_t>(edge.first)] -
                displacement[static_cast<std::size_t>(edge.second)]);
        energy += static_cast<double>(laplacian_weight * edge.weight) *
            difference * difference;
    }
    for (const float value : displacement)
    {
        energy += static_cast<double>(hull_prior_weight) *
            static_cast<double>(value) *
            static_cast<double>(value);
    }
    return energy;
}

} // namespace

RobustSurfaceDisplacementStatistics
RobustSurfaceDisplacementSolver::solve(
    const TriMesh &mesh,
    const std::vector<RobustSurfaceDisplacementObservation> &observations,
    const RobustSurfaceDisplacementOptions &options,
    std::vector<float> *displacement,
    const std::function<bool()> &isCancelled)
{
    RobustSurfaceDisplacementStatistics statistics;
    if (!displacement || mesh.vertices.empty() ||
        displacement->size() != mesh.vertices.size() ||
        !std::isfinite(options.robustScale) ||
        !std::isfinite(options.laplacianWeight) ||
        !std::isfinite(options.hullPriorWeight) ||
        !std::isfinite(options.maximumDisplacement) ||
        !std::isfinite(options.minimumNormalDot) ||
        !std::isfinite(options.convergenceTolerance) ||
        !(options.maximumDisplacement > 0.0f))
    {
        return statistics;
    }

    const float robust_scale =
        std::max(1.0e-8f, options.robustScale);
    const float laplacian_weight =
        std::max(0.0f, options.laplacianWeight);
    const float hull_prior_weight =
        std::max(1.0e-6f, options.hullPriorWeight);
    const float maximum_displacement =
        std::max(0.0f, options.maximumDisplacement);
    const std::vector<WeightedEdge> edges =
        buildEdges(mesh, options.minimumNormalDot);
    statistics.regularizationEdgeCount = edges.size();

    std::vector<float> normalized_observation_weight(
        observations.size(), 0.0f);
    std::vector<float> vertex_weight(mesh.vertices.size(), 0.0f);
    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        const RobustSurfaceDisplacementObservation &observation =
            observations[index];
        if (!validObservation(
                observation, mesh.vertices.size()))
        {
            continue;
        }
        vertex_weight[static_cast<std::size_t>(
            observation.vertexIndex)] += observation.weight;
        ++statistics.observationCount;
    }
    for (std::size_t index = 0; index < observations.size(); ++index)
    {
        const RobustSurfaceDisplacementObservation &observation =
            observations[index];
        if (!validObservation(
                observation, mesh.vertices.size()))
        {
            continue;
        }
        const float sum = vertex_weight[static_cast<std::size_t>(
            observation.vertexIndex)];
        if (sum > 0.0f)
        {
            normalized_observation_weight[index] =
                observation.weight / std::max(1.0f, sum);
        }
    }
    statistics.anchoredVertexCount = static_cast<std::uint64_t>(
        std::count_if(
            vertex_weight.cbegin(),
            vertex_weight.cend(),
            [](float value)
            {
                return value > 0.0f;
            }));
    statistics.priorOnlyVertexCount =
        mesh.vertices.size() - statistics.anchoredVertexCount;
    if (statistics.observationCount == 0)
    {
        return statistics;
    }

    std::vector<float> working_displacement = *displacement;
    for (float &value : working_displacement)
    {
        value = std::clamp(
            std::isfinite(value) ? value : 0.0f,
            -maximum_displacement,
            maximum_displacement);
    }
    statistics.initialEnergy = robustEnergy(
        observations,
        normalized_observation_weight,
        edges,
        working_displacement,
        robust_scale,
        laplacian_weight,
        hull_prior_weight);

    const int maximum_irls_iterations =
        std::clamp(options.irlsIterations, 1, 10);
    const int maximum_pcg_iterations =
        std::clamp(options.maximumPcgIterations, 1, 500);
    const double convergence_tolerance =
        std::clamp(
            static_cast<double>(options.convergenceTolerance),
            1.0e-9,
            1.0e-2);
    std::vector<float> data_diagonal(mesh.vertices.size());
    std::vector<float> right_hand_side(mesh.vertices.size());
    std::vector<float> system_diagonal(mesh.vertices.size());
    std::vector<float> residual(mesh.vertices.size());
    std::vector<float> preconditioned(mesh.vertices.size());
    std::vector<float> direction(mesh.vertices.size());
    std::vector<float> system_direction(mesh.vertices.size());
    std::vector<float> system_value(mesh.vertices.size());

    for (int irls = 0; irls < maximum_irls_iterations; ++irls)
    {
        if (isCancelled && isCancelled())
        {
            statistics.cancelled = true;
            return statistics;
        }
        std::fill(
            data_diagonal.begin(),
            data_diagonal.end(),
            hull_prior_weight);
        std::fill(
            right_hand_side.begin(),
            right_hand_side.end(),
            0.0f);
        bool pcg_converged = false;
        bool numerical_failure = false;
        for (std::size_t index = 0; index < observations.size(); ++index)
        {
            const RobustSurfaceDisplacementObservation &observation =
                observations[index];
            if (!validObservation(
                    observation, mesh.vertices.size()))
            {
                continue;
            }
            const std::size_t vertex_index =
                static_cast<std::size_t>(observation.vertexIndex);
            const float residual_value =
                working_displacement[vertex_index] -
                observation.target;
            const float normalized =
                residual_value / robust_scale;
            const float robust_weight =
                normalized_observation_weight[index] /
                (1.0f + normalized * normalized);
            data_diagonal[vertex_index] += robust_weight;
            right_hand_side[vertex_index] +=
                robust_weight * observation.target;
        }
        if (!finiteVector(data_diagonal) ||
            !finiteVector(right_hand_side))
        {
            return statistics;
        }

        system_diagonal = data_diagonal;
        for (const WeightedEdge &edge : edges)
        {
            const float value =
                laplacian_weight * edge.weight;
            system_diagonal[static_cast<std::size_t>(edge.first)] += value;
            system_diagonal[static_cast<std::size_t>(edge.second)] += value;
        }
        applySystem(
            data_diagonal,
            edges,
            laplacian_weight,
            working_displacement,
            &system_value);
        if (!finiteVector(system_diagonal) ||
            !finiteVector(system_value))
        {
            return statistics;
        }
        for (std::size_t index = 0;
             index < residual.size();
             ++index)
        {
            residual[index] =
                right_hand_side[index] - system_value[index];
            preconditioned[index] =
                residual[index] /
                std::max(1.0e-8f, system_diagonal[index]);
        }
        direction = preconditioned;
        double residual_preconditioned =
            dotProduct(residual, preconditioned);
        const double initial_residual_norm =
            std::sqrt(std::max(0.0, dotProduct(residual, residual)));
        if (!std::isfinite(residual_preconditioned) ||
            !std::isfinite(initial_residual_norm))
        {
            return statistics;
        }
        if (!(initial_residual_norm > 0.0))
        {
            pcg_converged = true;
            statistics.irlsIterationCount = irls + 1;
        }

        for (int pcg = 0;
             !pcg_converged && pcg < maximum_pcg_iterations;
             ++pcg)
        {
            if (isCancelled && isCancelled())
            {
                statistics.cancelled = true;
                return statistics;
            }
            applySystem(
                data_diagonal,
                edges,
                laplacian_weight,
                direction,
                &system_direction);
            const double denominator =
                dotProduct(direction, system_direction);
            if (!std::isfinite(denominator) ||
                !std::isfinite(residual_preconditioned))
            {
                numerical_failure = true;
                break;
            }
            if (!(denominator > 1.0e-20) ||
                !(residual_preconditioned > 1.0e-30))
            {
                break;
            }
            const double alpha =
                residual_preconditioned / denominator;
            if (!std::isfinite(alpha))
            {
                numerical_failure = true;
                break;
            }
            for (std::size_t index = 0;
                 index < working_displacement.size();
                 ++index)
            {
                working_displacement[index] +=
                    static_cast<float>(alpha) * direction[index];
                residual[index] -=
                    static_cast<float>(alpha) * system_direction[index];
            }
            ++statistics.pcgIterationCount;
            const double residual_norm =
                std::sqrt(std::max(0.0, dotProduct(residual, residual)));
            if (!std::isfinite(residual_norm))
            {
                numerical_failure = true;
                break;
            }
            statistics.finalRelativeResidual =
                residual_norm / initial_residual_norm;
            if (statistics.finalRelativeResidual <=
                convergence_tolerance)
            {
                pcg_converged = true;
                break;
            }
            for (std::size_t index = 0;
                 index < residual.size();
                 ++index)
            {
                preconditioned[index] =
                    residual[index] /
                    std::max(1.0e-8f, system_diagonal[index]);
            }
            const double updated_residual_preconditioned =
                dotProduct(residual, preconditioned);
            if (!std::isfinite(updated_residual_preconditioned))
            {
                numerical_failure = true;
                break;
            }
            if (!(updated_residual_preconditioned > 1.0e-30))
            {
                pcg_converged =
                    updated_residual_preconditioned >= -1.0e-20;
                numerical_failure = !pcg_converged;
                break;
            }
            const double beta =
                updated_residual_preconditioned /
                std::max(1.0e-30, residual_preconditioned);
            if (!std::isfinite(beta))
            {
                numerical_failure = true;
                break;
            }
            for (std::size_t index = 0;
                 index < direction.size();
                 ++index)
            {
                direction[index] =
                    preconditioned[index] +
                    static_cast<float>(beta) * direction[index];
            }
            residual_preconditioned =
                updated_residual_preconditioned;
        }

        if (numerical_failure ||
            !finiteVector(working_displacement))
        {
            return statistics;
        }
        for (float &value : working_displacement)
        {
            value = std::clamp(
                value,
                -maximum_displacement,
                maximum_displacement);
        }
        statistics.irlsIterationCount = irls + 1;
        statistics.converged = pcg_converged;
    }

    statistics.finalEnergy = robustEnergy(
        observations,
        normalized_observation_weight,
        edges,
        working_displacement,
        robust_scale,
        laplacian_weight,
        hull_prior_weight);
    statistics.solved =
        !statistics.cancelled &&
        std::isfinite(statistics.finalEnergy) &&
        finiteVector(working_displacement);
    if (statistics.solved)
    {
        *displacement = std::move(working_displacement);
    }
    return statistics;
}

} // namespace xjw::mesh
