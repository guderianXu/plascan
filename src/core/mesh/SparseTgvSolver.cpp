#include "SparseTgvSolver.h"

#include "ProcessCpuTimer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef MESHING_OPENMP
#include <omp.h>
#endif

namespace xjw::mesh
{
namespace
{

using Vector3 = std::array<float, 3>;
using Symmetric3 = std::array<float, 6>;

float inverseCenterDistance(const AdaptiveTsdfOctreeNode &first,
                            const AdaptiveTsdfOctreeNode &second)
{
    return 2.0f /
        static_cast<float>(std::max(1, first.size + second.size));
}

float forwardDifference(
    const std::vector<float> &values,
    const std::vector<AdaptiveTsdfOctreeNode> &nodes,
    int index,
    int axis)
{
    const int neighbor = nodes[index].faceNeighbors[axis * 2 + 1];
    if (neighbor < 0)
    {
        return 0.0f;
    }
    return (values[neighbor] - values[index]) *
        inverseCenterDistance(nodes[index], nodes[neighbor]);
}

float forwardVectorDifference(
    const std::vector<Vector3> &values,
    const std::vector<AdaptiveTsdfOctreeNode> &nodes,
    int index,
    int component,
    int axis)
{
    const int neighbor = nodes[index].faceNeighbors[axis * 2 + 1];
    if (neighbor < 0)
    {
        return 0.0f;
    }
    return (values[neighbor][component] - values[index][component]) *
        inverseCenterDistance(nodes[index], nodes[neighbor]);
}

float backwardDivergence(
    const std::vector<Vector3> &values,
    const std::vector<AdaptiveTsdfOctreeNode> &nodes,
    int index)
{
    float divergence = 0.0f;
    for (int axis = 0; axis < 3; ++axis)
    {
        const int negative = nodes[index].faceNeighbors[axis * 2];
        const float negative_value = negative >= 0
            ? values[negative][axis]
            : 0.0f;
        const float inverse_spacing = negative >= 0
            ? inverseCenterDistance(nodes[index], nodes[negative])
            : 1.0f / static_cast<float>(std::max(1, nodes[index].size));
        divergence +=
            (values[index][axis] - negative_value) * inverse_spacing;
    }
    return divergence;
}

float symmetricComponent(const Symmetric3 &value,
                         int row,
                         int column)
{
    if (row == column)
    {
        return value[row];
    }
    if ((row == 0 && column == 1) ||
        (row == 1 && column == 0))
    {
        return value[3];
    }
    if ((row == 0 && column == 2) ||
        (row == 2 && column == 0))
    {
        return value[4];
    }
    return value[5];
}

Vector3 backwardSymmetricDivergence(
    const std::vector<Symmetric3> &values,
    const std::vector<AdaptiveTsdfOctreeNode> &nodes,
    int index)
{
    Vector3 divergence{};
    for (int component = 0; component < 3; ++component)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const int negative = nodes[index].faceNeighbors[axis * 2];
            const float negative_value = negative >= 0
                ? symmetricComponent(values[negative], component, axis)
                : 0.0f;
            const float inverse_spacing = negative >= 0
                ? inverseCenterDistance(nodes[index], nodes[negative])
                : 1.0f / static_cast<float>(
                      std::max(1, nodes[index].size));
            divergence[component] +=
                (symmetricComponent(values[index], component, axis) -
                 negative_value) *
                inverse_spacing;
        }
    }
    return divergence;
}

void projectVector(float radius, Vector3 *value)
{
    const float norm = std::sqrt(
        (*value)[0] * (*value)[0] +
        (*value)[1] * (*value)[1] +
        (*value)[2] * (*value)[2]);
    if (norm <= radius || norm <= 1.0e-12f)
    {
        return;
    }
    const float scale = radius / norm;
    for (float &component : *value)
    {
        component *= scale;
    }
}

void projectSymmetric(float radius, Symmetric3 *value)
{
    const float norm = std::sqrt(
        (*value)[0] * (*value)[0] +
        (*value)[1] * (*value)[1] +
        (*value)[2] * (*value)[2] +
        2.0f * ((*value)[3] * (*value)[3] +
                (*value)[4] * (*value)[4] +
                (*value)[5] * (*value)[5]));
    if (norm <= radius || norm <= 1.0e-12f)
    {
        return;
    }
    const float scale = radius / norm;
    for (float &component : *value)
    {
        component *= scale;
    }
}

float l1DataProx(float value,
                 float dataValue,
                 float threshold)
{
    if (value > dataValue + threshold)
    {
        return value - threshold;
    }
    if (value < dataValue - threshold)
    {
        return value + threshold;
    }
    return dataValue;
}

double meanAbsoluteCurvature(
    const std::vector<float> &field,
    const std::vector<AdaptiveTsdfOctreeNode> &nodes,
    int worker_count)
{
    double sum = 0.0;
    unsigned long long count = 0;
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) num_threads(worker_count) if(worker_count > 1) reduction(+:sum,count)
#endif
    for (int index = 0; index < static_cast<int>(nodes.size()); ++index)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const int negative = nodes[index].faceNeighbors[axis * 2];
            const int positive = nodes[index].faceNeighbors[axis * 2 + 1];
            if (negative < 0 || positive < 0)
            {
                continue;
            }
            sum += std::fabs(
                field[negative] - 2.0f * field[index] + field[positive]);
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

} // namespace

SparseTgvStatistics SparseTgvSolver::solve(
    const SparseTgvOptions &options,
    AdaptiveTsdfOctreeResult *octree,
    const std::function<bool()> &isCancelled,
    const std::function<void(int, int)> &progress)
{
    if (!octree)
    {
        throw std::invalid_argument("SparseTgvSolver requires an octree");
    }
    SparseTgvStatistics statistics;
    const auto start = std::chrono::steady_clock::now();
    const double cpu_start = detail::processCpuTimeMilliseconds();
    const int node_count = static_cast<int>(octree->leaves.size());
#ifdef MESHING_OPENMP
    const int worker_count = std::max(
        1,
        std::min(node_count,
                 options.workerCount > 0
                     ? options.workerCount
                     : omp_get_max_threads()));
#else
    const int worker_count = 1;
#endif
    statistics.effectiveWorkerCount = worker_count;
    statistics.activeNodeCount = node_count;
    if (node_count == 0)
    {
        return statistics;
    }
    statistics.executed = true;

    std::vector<float> data_value(node_count, 1.0f);
    std::vector<float> data_weight(node_count, 0.0f);
    std::vector<float> u(node_count, 1.0f);
    std::vector<float> next_u(node_count, 1.0f);
    std::vector<float> extrapolated_u(node_count, 1.0f);
    std::vector<Vector3> v(node_count);
    std::vector<Vector3> next_v(node_count);
    std::vector<Vector3> extrapolated_v(node_count);
    std::vector<Vector3> first_dual(node_count);
    std::vector<Symmetric3> second_dual(node_count);
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) num_threads(worker_count) if(worker_count > 1)
#endif
    for (int index = 0; index < node_count; ++index)
    {
        const AdaptiveTsdfOctreeNode &node = octree->leaves[index];
        u[index] = node.value;
        next_u[index] = node.value;
        extrapolated_u[index] = node.value;
        const float total_weight = node.histogram.totalWeight();
        if (total_weight > 0.0f)
        {
            constexpr float kHalfHistogramBinWidth =
                1.0f / static_cast<float>(
                    kDepthVisibilityHistogramBinCount);
            const float median = node.histogram.weightedMedian();
            data_value[index] = std::clamp(
                node.value,
                median - kHalfHistogramBinWidth,
                median + kHalfHistogramBinWidth);
        }
        else
        {
            data_value[index] = node.value;
        }
        const float samples = static_cast<float>(
            std::max<std::uint32_t>(1, node.activeSampleCount));
        const float mean_weight = total_weight / samples;
        data_weight[index] = std::clamp(
            mean_weight * (0.25f + node.histogram.dominantBinRatio()),
            0.05f,
            8.0f);
    }
    statistics.initialMeanAbsoluteCurvature =
        meanAbsoluteCurvature(u, octree->leaves, worker_count);

    const int maximum_iterations = std::clamp(
        options.maximumIterations, 1, 1000);
    const int minimum_iterations = std::clamp(
        options.minimumIterations, 1, maximum_iterations);
    const float tau = std::clamp(options.primalStep, 0.001f, 0.25f);
    const float sigma = std::clamp(options.dualStep, 0.001f, 0.25f);
    const float theta = std::clamp(options.extrapolation, 0.0f, 1.0f);
    const float alpha_one = std::max(1.0e-5f, options.firstOrderWeight);
    const float alpha_zero = std::max(1.0e-5f, options.secondOrderWeight);

#ifdef MESHING_OPENMP
    bool stop = false;
    double absolute_update_sum = 0.0;
#pragma omp parallel num_threads(worker_count) if(worker_count > 1) shared(stop, absolute_update_sum)
    {
        for (int iteration = 0; iteration < maximum_iterations; ++iteration)
        {
#pragma omp single
            {
                stop = isCancelled && isCancelled();
                statistics.cancelled = stop;
                absolute_update_sum = 0.0;
                if (!stop && progress &&
                    (iteration == 0 || (iteration + 1) % 10 == 0))
                {
                    progress(iteration + 1, maximum_iterations);
                }
            }
            if (!stop)
            {
#pragma omp for schedule(static)
                for (int index = 0; index < node_count; ++index)
                {
                    Vector3 first = first_dual[index];
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        first[axis] += sigma *
                            (forwardDifference(extrapolated_u,
                                               octree->leaves,
                                               index,
                                               axis) -
                             extrapolated_v[index][axis]);
                    }
                    projectVector(alpha_one, &first);
                    first_dual[index] = first;

                    Symmetric3 second = second_dual[index];
                    second[0] += sigma * forwardVectorDifference(
                        extrapolated_v, octree->leaves, index, 0, 0);
                    second[1] += sigma * forwardVectorDifference(
                        extrapolated_v, octree->leaves, index, 1, 1);
                    second[2] += sigma * forwardVectorDifference(
                        extrapolated_v, octree->leaves, index, 2, 2);
                    second[3] += sigma * 0.5f *
                        (forwardVectorDifference(
                             extrapolated_v, octree->leaves, index, 0, 1) +
                         forwardVectorDifference(
                             extrapolated_v, octree->leaves, index, 1, 0));
                    second[4] += sigma * 0.5f *
                        (forwardVectorDifference(
                             extrapolated_v, octree->leaves, index, 0, 2) +
                         forwardVectorDifference(
                             extrapolated_v, octree->leaves, index, 2, 0));
                    second[5] += sigma * 0.5f *
                        (forwardVectorDifference(
                             extrapolated_v, octree->leaves, index, 1, 2) +
                         forwardVectorDifference(
                             extrapolated_v, octree->leaves, index, 2, 1));
                    projectSymmetric(alpha_zero, &second);
                    second_dual[index] = second;
                }

#pragma omp for schedule(static) reduction(+:absolute_update_sum)
                for (int index = 0; index < node_count; ++index)
                {
                    const float unconstrained =
                        u[index] + tau * backwardDivergence(
                            first_dual, octree->leaves, index);
                    next_u[index] = std::clamp(
                        l1DataProx(
                            unconstrained,
                            data_value[index],
                            tau * std::max(0.0f, options.dataFidelity) *
                                data_weight[index]),
                        -1.0f,
                        1.0f);
                    const Vector3 symmetric_divergence =
                        backwardSymmetricDivergence(
                            second_dual, octree->leaves, index);
                    for (int component = 0; component < 3; ++component)
                    {
                        next_v[index][component] =
                            v[index][component] +
                            tau * (first_dual[index][component] +
                                   symmetric_divergence[component]);
                    }
                    absolute_update_sum +=
                        std::fabs(next_u[index] - u[index]);
                }

#pragma omp for schedule(static)
                for (int index = 0; index < node_count; ++index)
                {
                    extrapolated_u[index] =
                        next_u[index] + theta * (next_u[index] - u[index]);
                    for (int component = 0; component < 3; ++component)
                    {
                        extrapolated_v[index][component] =
                            next_v[index][component] +
                            theta * (next_v[index][component] -
                                     v[index][component]);
                    }
                }
            }
#pragma omp single
            {
                if (!stop)
                {
                    u.swap(next_u);
                    v.swap(next_v);
                    statistics.iterationCount = iteration + 1;
                    statistics.finalMeanAbsoluteUpdate =
                        absolute_update_sum / static_cast<double>(node_count);
                    stop = statistics.iterationCount >= minimum_iterations &&
                        statistics.finalMeanAbsoluteUpdate <=
                            std::max(0.0f, options.convergenceTolerance);
                }
            }
            if (stop)
            {
                break;
            }
        }
    }
#else
    for (int iteration = 0; iteration < maximum_iterations; ++iteration)
    {
        if (isCancelled && isCancelled())
        {
            statistics.cancelled = true;
            break;
        }
        if (progress && (iteration == 0 || (iteration + 1) % 10 == 0))
        {
            progress(iteration + 1, maximum_iterations);
        }
        for (int index = 0; index < node_count; ++index)
        {
            Vector3 first = first_dual[index];
            for (int axis = 0; axis < 3; ++axis)
            {
                first[axis] += sigma *
                    (forwardDifference(
                         extrapolated_u, octree->leaves, index, axis) -
                     extrapolated_v[index][axis]);
            }
            projectVector(alpha_one, &first);
            first_dual[index] = first;
            Symmetric3 second = second_dual[index];
            second[0] += sigma * forwardVectorDifference(
                extrapolated_v, octree->leaves, index, 0, 0);
            second[1] += sigma * forwardVectorDifference(
                extrapolated_v, octree->leaves, index, 1, 1);
            second[2] += sigma * forwardVectorDifference(
                extrapolated_v, octree->leaves, index, 2, 2);
            second[3] += sigma * 0.5f *
                (forwardVectorDifference(
                     extrapolated_v, octree->leaves, index, 0, 1) +
                 forwardVectorDifference(
                     extrapolated_v, octree->leaves, index, 1, 0));
            second[4] += sigma * 0.5f *
                (forwardVectorDifference(
                     extrapolated_v, octree->leaves, index, 0, 2) +
                 forwardVectorDifference(
                     extrapolated_v, octree->leaves, index, 2, 0));
            second[5] += sigma * 0.5f *
                (forwardVectorDifference(
                     extrapolated_v, octree->leaves, index, 1, 2) +
                 forwardVectorDifference(
                     extrapolated_v, octree->leaves, index, 2, 1));
            projectSymmetric(alpha_zero, &second);
            second_dual[index] = second;
        }
        double absolute_update_sum = 0.0;
        for (int index = 0; index < node_count; ++index)
        {
            const float unconstrained = u[index] + tau * backwardDivergence(
                first_dual, octree->leaves, index);
            next_u[index] = std::clamp(
                l1DataProx(unconstrained,
                           data_value[index],
                           tau * std::max(0.0f, options.dataFidelity) *
                               data_weight[index]),
                -1.0f,
                1.0f);
            const Vector3 symmetric_divergence = backwardSymmetricDivergence(
                second_dual, octree->leaves, index);
            for (int component = 0; component < 3; ++component)
            {
                next_v[index][component] = v[index][component] +
                    tau * (first_dual[index][component] +
                           symmetric_divergence[component]);
            }
            absolute_update_sum += std::fabs(next_u[index] - u[index]);
        }
        for (int index = 0; index < node_count; ++index)
        {
            extrapolated_u[index] =
                next_u[index] + theta * (next_u[index] - u[index]);
            for (int component = 0; component < 3; ++component)
            {
                extrapolated_v[index][component] =
                    next_v[index][component] +
                    theta * (next_v[index][component] - v[index][component]);
            }
        }
        u.swap(next_u);
        v.swap(next_v);
        statistics.iterationCount = iteration + 1;
        statistics.finalMeanAbsoluteUpdate =
            absolute_update_sum / static_cast<double>(node_count);
        if (statistics.iterationCount >= minimum_iterations &&
            statistics.finalMeanAbsoluteUpdate <=
                std::max(0.0f, options.convergenceTolerance))
        {
            break;
        }
    }
#endif

#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static) num_threads(worker_count) if(worker_count > 1)
#endif
    for (int index = 0; index < node_count; ++index)
    {
        octree->leaves[index].value = u[index];
    }
    statistics.finalMeanAbsoluteCurvature =
        meanAbsoluteCurvature(u, octree->leaves, worker_count);
    statistics.elapsedMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    statistics.cpuTimeMs =
        detail::processCpuTimeMilliseconds() - cpu_start;
    statistics.cpuDuty = statistics.elapsedMs > 0 && worker_count > 0
        ? statistics.cpuTimeMs /
              (static_cast<double>(statistics.elapsedMs) * worker_count)
        : 0.0;
    return statistics;
}

} // namespace xjw::mesh
