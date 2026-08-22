#include "TextureOverlapExposure.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <utility>

namespace xjw::mesh::texture_v4
{
namespace
{

struct PairKey
{
    int first = -1;
    int second = -1;

    bool operator<(const PairKey &other) const
    {
        return first < other.first ||
            (first == other.first && second < other.second);
    }
};

struct ExposureEdge
{
    int first = -1;
    int second = -1;
    double logRatio = 0.0;
    double logMad = 0.0;
    double weight = 0.0;
};

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() % 2U) != 0U)
    {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

bool solveLinearSystem(std::vector<std::vector<double>> matrix,
                       std::vector<double> rightHandSide,
                       std::vector<double> *solution)
{
    if (!solution || matrix.empty() ||
        matrix.size() != rightHandSide.size())
    {
        return false;
    }
    const int size = static_cast<int>(matrix.size());
    for (int row = 0; row < size; ++row)
    {
        if (matrix[row].size() != static_cast<std::size_t>(size))
        {
            return false;
        }
    }

    for (int column = 0; column < size; ++column)
    {
        int pivot = column;
        for (int row = column + 1; row < size; ++row)
        {
            if (std::fabs(matrix[row][column]) >
                std::fabs(matrix[pivot][column]))
            {
                pivot = row;
            }
        }
        if (!std::isfinite(matrix[pivot][column]) ||
            std::fabs(matrix[pivot][column]) <= 1.0e-12)
        {
            return false;
        }
        if (pivot != column)
        {
            std::swap(matrix[pivot], matrix[column]);
            std::swap(rightHandSide[pivot], rightHandSide[column]);
        }

        const double diagonal = matrix[column][column];
        for (int index = column; index < size; ++index)
        {
            matrix[column][index] /= diagonal;
        }
        rightHandSide[column] /= diagonal;
        for (int row = 0; row < size; ++row)
        {
            if (row == column)
            {
                continue;
            }
            const double scale = matrix[row][column];
            if (std::fabs(scale) <= 1.0e-18)
            {
                continue;
            }
            for (int index = column; index < size; ++index)
            {
                matrix[row][index] -= scale * matrix[column][index];
            }
            rightHandSide[row] -= scale * rightHandSide[column];
        }
    }
    if (!std::all_of(rightHandSide.begin(), rightHandSide.end(), [](double value)
        {
            return std::isfinite(value);
        }))
    {
        return false;
    }
    *solution = std::move(rightHandSide);
    return true;
}

std::vector<std::vector<int>> connectedComponents(
    int view_count,
    const std::vector<ExposureEdge> &edges)
{
    std::vector<std::vector<int>> neighbors(
        static_cast<std::size_t>(view_count));
    for (const ExposureEdge &edge : edges)
    {
        neighbors[static_cast<std::size_t>(edge.first)].push_back(edge.second);
        neighbors[static_cast<std::size_t>(edge.second)].push_back(edge.first);
    }
    std::vector<bool> visited(static_cast<std::size_t>(view_count), false);
    std::vector<std::vector<int>> components;
    for (int seed = 0; seed < view_count; ++seed)
    {
        if (visited[static_cast<std::size_t>(seed)])
        {
            continue;
        }
        std::queue<int> pending;
        pending.push(seed);
        visited[static_cast<std::size_t>(seed)] = true;
        components.emplace_back();
        while (!pending.empty())
        {
            const int view = pending.front();
            pending.pop();
            components.back().push_back(view);
            for (const int neighbor :
                 neighbors[static_cast<std::size_t>(view)])
            {
                if (!visited[static_cast<std::size_t>(neighbor)])
                {
                    visited[static_cast<std::size_t>(neighbor)] = true;
                    pending.push(neighbor);
                }
            }
        }
        std::sort(components.back().begin(), components.back().end());
    }
    return components;
}

} // namespace

ExposureSolveResult solveRobustOverlapExposure(
    int viewCount,
    std::vector<ExposureObservation> observations,
    const ExposureSolveOptions &options)
{
    ExposureSolveResult result;
    result.gains.assign(
        static_cast<std::size_t>(std::max(viewCount, 0)), 1.0f);
    if (viewCount <= 0)
    {
        result.status = "no_views";
        return result;
    }
    if (viewCount == 1)
    {
        result.graphConnected = true;
        result.status = "single_view";
        return result;
    }
    if (options.minimumPairSamples < 1 ||
        options.maximumPairSamples < options.minimumPairSamples ||
        !std::isfinite(options.maximumPairLogMad) ||
        options.maximumPairLogMad < 0.0 ||
        !std::isfinite(options.minimumGain) ||
        !std::isfinite(options.maximumGain) ||
        options.minimumGain <= 0.0f ||
        options.minimumGain > 1.0f ||
        options.maximumGain < 1.0f ||
        options.minimumGain > options.maximumGain)
    {
        result.status = "invalid_options";
        return result;
    }

    observations.erase(
        std::remove_if(observations.begin(), observations.end(), [&](const auto &item)
        {
            return item.viewIndex < 0 || item.viewIndex >= viewCount ||
                !std::isfinite(item.linearLuminance) ||
                item.linearLuminance <= 0.0;
        }),
        observations.end());
    std::sort(observations.begin(), observations.end(), [](const auto &left,
                                                           const auto &right)
    {
        if (left.pointId != right.pointId)
        {
            return left.pointId < right.pointId;
        }
        if (left.viewIndex != right.viewIndex)
        {
            return left.viewIndex < right.viewIndex;
        }
        return left.linearLuminance < right.linearLuminance;
    });

    std::map<PairKey, std::vector<double>> pair_log_ratios;
    std::size_t point_begin = 0;
    while (point_begin < observations.size())
    {
        std::size_t point_end = point_begin + 1;
        while (point_end < observations.size() &&
               observations[point_end].pointId ==
                   observations[point_begin].pointId)
        {
            ++point_end;
        }
        std::vector<std::pair<int, double>> per_view;
        std::size_t view_begin = point_begin;
        while (view_begin < point_end)
        {
            std::size_t view_end = view_begin + 1;
            std::vector<double> luminances{
                observations[view_begin].linearLuminance};
            while (view_end < point_end &&
                   observations[view_end].viewIndex ==
                       observations[view_begin].viewIndex)
            {
                luminances.push_back(
                    observations[view_end].linearLuminance);
                ++view_end;
            }
            per_view.push_back({
                observations[view_begin].viewIndex,
                median(std::move(luminances))});
            view_begin = view_end;
        }
        result.observationCount += per_view.size();
        for (std::size_t first = 0; first < per_view.size(); ++first)
        {
            for (std::size_t second = first + 1;
                 second < per_view.size();
                 ++second)
            {
                const double ratio = std::log(
                    per_view[first].second / per_view[second].second);
                if (std::isfinite(ratio))
                {
                    std::vector<double> &pair_ratios = pair_log_ratios[{
                        per_view[first].first,
                        per_view[second].first}];
                    if (pair_ratios.size() < static_cast<std::size_t>(
                            options.maximumPairSamples))
                    {
                        pair_ratios.push_back(ratio);
                    }
                }
            }
        }
        point_begin = point_end;
    }

    result.candidatePairCount = pair_log_ratios.size();
    std::vector<ExposureEdge> edges;
    edges.reserve(pair_log_ratios.size());
    for (const auto &[key, ratios] : pair_log_ratios)
    {
        if (ratios.size() <
            static_cast<std::size_t>(options.minimumPairSamples))
        {
            ++result.rejectedInsufficientPairCount;
            continue;
        }
        const double pair_median = median(ratios);
        std::vector<double> deviations;
        deviations.reserve(ratios.size());
        for (const double ratio : ratios)
        {
            deviations.push_back(std::fabs(ratio - pair_median));
        }
        const double pair_mad = median(std::move(deviations));
        if (!std::isfinite(pair_median) || !std::isfinite(pair_mad) ||
            pair_mad > options.maximumPairLogMad)
        {
            ++result.rejectedHighMadPairCount;
            continue;
        }
        const double sample_weight = static_cast<double>(
            std::min<std::size_t>(ratios.size(), 512U));
        const double weight = sample_weight /
            std::max(pair_mad + 0.01, 0.01);
        edges.push_back(
            {key.first, key.second, pair_median, pair_mad, weight});
        ++result.acceptedPairCount;
        result.maximumAcceptedLogMad = std::max(
            result.maximumAcceptedLogMad, pair_mad);
    }

    const std::vector<std::vector<int>> components = connectedComponents(
        viewCount, edges);
    result.connectedComponentCount = static_cast<int>(components.size());
    result.graphConnected = components.size() == 1U;
    if (edges.empty())
    {
        if (result.rejectedHighMadPairCount > 0)
        {
            result.status = "high_pair_mad";
        }
        else if (result.rejectedInsufficientPairCount > 0)
        {
            result.status = "insufficient_overlap_samples";
        }
        else if (pair_log_ratios.empty())
        {
            result.status = "no_overlap";
        }
        return result;
    }
    std::vector<double> log_gains(static_cast<std::size_t>(viewCount), 0.0);
    int solved_component_count = 0;
    for (const std::vector<int> &component : components)
    {
        if (component.size() < 2U)
        {
            continue;
        }
        std::vector<int> local_index(static_cast<std::size_t>(viewCount), -1);
        for (int index = 0; index < static_cast<int>(component.size()); ++index)
        {
            local_index[static_cast<std::size_t>(component[index])] = index;
        }
        const int unknown_count = static_cast<int>(component.size()) - 1;
        std::vector<std::vector<double>> normal_matrix(
            static_cast<std::size_t>(unknown_count),
            std::vector<double>(static_cast<std::size_t>(unknown_count), 0.0));
        std::vector<double> right_hand_side(
            static_cast<std::size_t>(unknown_count), 0.0);
        for (const ExposureEdge &edge : edges)
        {
            const int first_local =
                local_index[static_cast<std::size_t>(edge.first)];
            const int second_local =
                local_index[static_cast<std::size_t>(edge.second)];
            if (first_local < 0 || second_local < 0)
            {
                continue;
            }
            const int first = first_local - 1;
            const int second = second_local - 1;
            if (first >= 0)
            {
                normal_matrix[first][first] += edge.weight;
                right_hand_side[first] -= edge.weight * edge.logRatio;
            }
            if (second >= 0)
            {
                normal_matrix[second][second] += edge.weight;
                right_hand_side[second] += edge.weight * edge.logRatio;
            }
            if (first >= 0 && second >= 0)
            {
                normal_matrix[first][second] -= edge.weight;
                normal_matrix[second][first] -= edge.weight;
            }
        }
        std::vector<double> solved;
        if (!solveLinearSystem(
                std::move(normal_matrix),
                std::move(right_hand_side),
                &solved))
        {
            result.gains.assign(static_cast<std::size_t>(viewCount), 1.0f);
            result.status = "singular_overlap_component";
            return result;
        }
        std::vector<double> component_log_gains(component.size(), 0.0);
        for (int index = 1; index < static_cast<int>(component.size()); ++index)
        {
            component_log_gains[static_cast<std::size_t>(index)] =
                solved[static_cast<std::size_t>(index - 1)];
        }
        const double component_mean = std::accumulate(
            component_log_gains.begin(), component_log_gains.end(), 0.0) /
            static_cast<double>(component_log_gains.size());
        for (int index = 0; index < static_cast<int>(component.size()); ++index)
        {
            log_gains[static_cast<std::size_t>(component[index])] =
                component_log_gains[static_cast<std::size_t>(index)] -
                component_mean;
        }
        ++solved_component_count;
    }
    for (int view = 0; view < viewCount; ++view)
    {
        const double gain = std::exp(
            log_gains[static_cast<std::size_t>(view)]);
        if (!std::isfinite(gain))
        {
            result.gains.assign(static_cast<std::size_t>(viewCount), 1.0f);
            result.graphConnected = false;
            result.status = "nonfinite_gain_solution";
            return result;
        }
        result.gains[static_cast<std::size_t>(view)] = std::clamp(
            static_cast<float>(gain),
            options.minimumGain,
            options.maximumGain);
    }
    result.applied = std::any_of(
        result.gains.begin(), result.gains.end(), [](float gain)
    {
        return std::fabs(gain - 1.0f) > 1.0e-5f;
    });
    result.correctedViewCount = static_cast<int>(std::count_if(
        result.gains.begin(), result.gains.end(), [](float gain)
        {
            return std::fabs(gain - 1.0f) > 1.0e-5f;
        }));
    result.status = result.applied
        ? (result.graphConnected ? "applied" : "applied_partial_components")
        : (solved_component_count > 0
            ? "identity_estimate"
            : "disconnected_overlap_graph");
    return result;
}

} // namespace xjw::mesh::texture_v4
