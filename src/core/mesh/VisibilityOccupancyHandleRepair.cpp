#include "VisibilityOccupancyHandleRepair.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <unordered_set>

namespace xjw::mesh
{
namespace
{

std::size_t sampleCount(const std::array<int, 3> &dimensions)
{
    if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] <= 0)
    {
        return 0;
    }
    const std::size_t x = static_cast<std::size_t>(dimensions[0]);
    const std::size_t y = static_cast<std::size_t>(dimensions[1]);
    const std::size_t z = static_cast<std::size_t>(dimensions[2]);
    if (x > std::numeric_limits<std::size_t>::max() / y ||
        x * y > std::numeric_limits<std::size_t>::max() / z)
    {
        return 0;
    }
    return x * y * z;
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

bool occupiedAt(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    int x,
    int y,
    int z)
{
    return x >= 0 && x < dimensions[0] &&
           y >= 0 && y < dimensions[1] &&
           z >= 0 && z < dimensions[2] &&
           occupied[gridIndex(dimensions, x, y, z)] != 0;
}

bool anyIncidentVoxel(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    int x0,
    int x1,
    int y0,
    int y1,
    int z0,
    int z1)
{
    for (int z = z0; z <= z1; ++z)
    {
        for (int y = y0; y <= y1; ++y)
        {
            for (int x = x0; x <= x1; ++x)
            {
                if (occupiedAt(dimensions, occupied, x, y, z))
                {
                    return true;
                }
            }
        }
    }
    return false;
}

int bodyEuler(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied)
{
    std::int64_t cube_count = 0;
    std::int64_t shared_face_count = 0;
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                if (!occupiedAt(dimensions, occupied, x, y, z))
                {
                    continue;
                }
                ++cube_count;
                shared_face_count += occupiedAt(
                    dimensions, occupied, x + 1, y, z);
                shared_face_count += occupiedAt(
                    dimensions, occupied, x, y + 1, z);
                shared_face_count += occupiedAt(
                    dimensions, occupied, x, y, z + 1);
            }
        }
    }
    const std::int64_t face_count = 6 * cube_count - shared_face_count;

    std::int64_t edge_count = 0;
    for (int z = 0; z <= dimensions[2]; ++z)
    {
        for (int y = 0; y <= dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                edge_count += anyIncidentVoxel(
                    dimensions, occupied, x, x, y - 1, y, z - 1, z);
            }
        }
    }
    for (int z = 0; z <= dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x <= dimensions[0]; ++x)
            {
                edge_count += anyIncidentVoxel(
                    dimensions, occupied, x - 1, x, y, y, z - 1, z);
            }
        }
    }
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y <= dimensions[1]; ++y)
        {
            for (int x = 0; x <= dimensions[0]; ++x)
            {
                edge_count += anyIncidentVoxel(
                    dimensions, occupied, x - 1, x, y - 1, y, z, z);
            }
        }
    }

    std::int64_t vertex_count = 0;
    for (int z = 0; z <= dimensions[2]; ++z)
    {
        for (int y = 0; y <= dimensions[1]; ++y)
        {
            for (int x = 0; x <= dimensions[0]; ++x)
            {
                vertex_count += anyIncidentVoxel(
                    dimensions,
                    occupied,
                    x - 1,
                    x,
                    y - 1,
                    y,
                    z - 1,
                    z);
            }
        }
    }
    const std::int64_t euler =
        vertex_count - edge_count + face_count - cube_count;
    return static_cast<int>(std::clamp<std::int64_t>(
        euler,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()));
}

template <typename Visitor>
void visitSixNeighbors(
    const std::array<int, 3> &dimensions,
    std::size_t index,
    Visitor visitor)
{
    const int x = static_cast<int>(index % dimensions[0]);
    const int y = static_cast<int>(
        (index / dimensions[0]) % dimensions[1]);
    const int z = static_cast<int>(
        index / (dimensions[0] * dimensions[1]));
    const int offsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    for (const auto &offset : offsets)
    {
        const int nx = x + offset[0];
        const int ny = y + offset[1];
        const int nz = z + offset[2];
        if (nx >= 0 && nx < dimensions[0] &&
            ny >= 0 && ny < dimensions[1] &&
            nz >= 0 && nz < dimensions[2])
        {
            visitor(gridIndex(dimensions, nx, ny, nz));
        }
    }
}

std::vector<std::uint8_t> exteriorReachableEmpty(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied)
{
    std::vector<std::uint8_t> exterior(occupied.size(), 0);
    std::queue<std::size_t> pending;
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                const bool boundary = x == 0 || y == 0 || z == 0 ||
                    x + 1 == dimensions[0] ||
                    y + 1 == dimensions[1] ||
                    z + 1 == dimensions[2];
                const std::size_t index = gridIndex(dimensions, x, y, z);
                if (boundary && occupied[index] == 0 && exterior[index] == 0)
                {
                    exterior[index] = 1;
                    pending.push(index);
                }
            }
        }
    }
    while (!pending.empty())
    {
        const std::size_t current = pending.front();
        pending.pop();
        visitSixNeighbors(
            dimensions,
            current,
            [&](std::size_t neighbor)
            {
                if (occupied[neighbor] == 0 && exterior[neighbor] == 0)
                {
                    exterior[neighbor] = 1;
                    pending.push(neighbor);
                }
            });
    }
    return exterior;
}

int occupiedComponentCount(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied)
{
    std::vector<std::uint8_t> visited(occupied.size(), 0);
    std::queue<std::size_t> pending;
    int component_count = 0;
    for (std::size_t seed = 0; seed < occupied.size(); ++seed)
    {
        if (occupied[seed] == 0 || visited[seed] != 0)
        {
            continue;
        }
        ++component_count;
        visited[seed] = 1;
        pending.push(seed);
        while (!pending.empty())
        {
            const std::size_t current = pending.front();
            pending.pop();
            visitSixNeighbors(
                dimensions,
                current,
                [&](std::size_t neighbor)
                {
                    if (occupied[neighbor] != 0 && visited[neighbor] == 0)
                    {
                        visited[neighbor] = 1;
                        pending.push(neighbor);
                    }
                });
        }
    }
    return component_count;
}

bool touchesOccupiedByFace(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    std::size_t index)
{
    bool touches_occupied = false;
    visitSixNeighbors(
        dimensions,
        index,
        [&](std::size_t neighbor)
        {
            touches_occupied = touches_occupied || occupied[neighbor] != 0;
        });
    return touches_occupied;
}

std::vector<std::vector<std::size_t>> proposalComponents(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const std::vector<std::uint8_t> &proposal)
{
    std::vector<std::vector<std::size_t>> components;
    std::vector<std::uint8_t> visited(occupied.size(), 0);
    std::queue<std::size_t> pending;
    for (std::size_t seed = 0; seed < occupied.size(); ++seed)
    {
        if (occupied[seed] != 0 || proposal[seed] == 0 || visited[seed] != 0)
        {
            continue;
        }
        components.emplace_back();
        visited[seed] = 1;
        pending.push(seed);
        while (!pending.empty())
        {
            const std::size_t current = pending.front();
            pending.pop();
            components.back().push_back(current);
            visitSixNeighbors(
                dimensions,
                current,
                [&](std::size_t neighbor)
                {
                    if (occupied[neighbor] == 0 &&
                        proposal[neighbor] != 0 && visited[neighbor] == 0)
                    {
                        visited[neighbor] = 1;
                        pending.push(neighbor);
                    }
                });
        }
    }
    return components;
}

std::uint64_t vertexId(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::uint64_t>(z) *
                static_cast<std::uint64_t>(dimensions[1] + 1) +
            static_cast<std::uint64_t>(y)) *
               static_cast<std::uint64_t>(dimensions[0] + 1) +
           static_cast<std::uint64_t>(x);
}

std::uint64_t xEdgeId(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::uint64_t>(z) *
                static_cast<std::uint64_t>(dimensions[1] + 1) +
            static_cast<std::uint64_t>(y)) *
               static_cast<std::uint64_t>(dimensions[0]) +
           static_cast<std::uint64_t>(x);
}

std::uint64_t yEdgeId(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    const std::uint64_t x_edge_count =
        static_cast<std::uint64_t>(dimensions[0]) *
        static_cast<std::uint64_t>(dimensions[1] + 1) *
        static_cast<std::uint64_t>(dimensions[2] + 1);
    return x_edge_count +
        (static_cast<std::uint64_t>(z) *
             static_cast<std::uint64_t>(dimensions[1]) +
         static_cast<std::uint64_t>(y)) *
            static_cast<std::uint64_t>(dimensions[0] + 1) +
        static_cast<std::uint64_t>(x);
}

std::uint64_t zEdgeId(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    const std::uint64_t x_edge_count =
        static_cast<std::uint64_t>(dimensions[0]) *
        static_cast<std::uint64_t>(dimensions[1] + 1) *
        static_cast<std::uint64_t>(dimensions[2] + 1);
    const std::uint64_t y_edge_count =
        static_cast<std::uint64_t>(dimensions[0] + 1) *
        static_cast<std::uint64_t>(dimensions[1]) *
        static_cast<std::uint64_t>(dimensions[2] + 1);
    return x_edge_count + y_edge_count +
        (static_cast<std::uint64_t>(z) *
             static_cast<std::uint64_t>(dimensions[1] + 1) +
         static_cast<std::uint64_t>(y)) *
            static_cast<std::uint64_t>(dimensions[0] + 1) +
        static_cast<std::uint64_t>(x);
}

std::uint64_t xFaceId(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::uint64_t>(z) *
                static_cast<std::uint64_t>(dimensions[1]) +
            static_cast<std::uint64_t>(y)) *
               static_cast<std::uint64_t>(dimensions[0] + 1) +
           static_cast<std::uint64_t>(x);
}

std::uint64_t yFaceId(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    const std::uint64_t x_face_count =
        static_cast<std::uint64_t>(dimensions[0] + 1) *
        static_cast<std::uint64_t>(dimensions[1]) *
        static_cast<std::uint64_t>(dimensions[2]);
    return x_face_count +
        (static_cast<std::uint64_t>(z) *
             static_cast<std::uint64_t>(dimensions[1] + 1) +
         static_cast<std::uint64_t>(y)) *
            static_cast<std::uint64_t>(dimensions[0]) +
        static_cast<std::uint64_t>(x);
}

std::uint64_t zFaceId(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    const std::uint64_t x_face_count =
        static_cast<std::uint64_t>(dimensions[0] + 1) *
        static_cast<std::uint64_t>(dimensions[1]) *
        static_cast<std::uint64_t>(dimensions[2]);
    const std::uint64_t y_face_count =
        static_cast<std::uint64_t>(dimensions[0]) *
        static_cast<std::uint64_t>(dimensions[1] + 1) *
        static_cast<std::uint64_t>(dimensions[2]);
    return x_face_count + y_face_count +
        (static_cast<std::uint64_t>(z) *
             static_cast<std::uint64_t>(dimensions[1]) +
         static_cast<std::uint64_t>(y)) *
            static_cast<std::uint64_t>(dimensions[0]) +
        static_cast<std::uint64_t>(x);
}

template <typename Candidate>
int candidateEulerDelta(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const Candidate &candidate)
{
    std::unordered_set<std::uint64_t> vertices;
    std::unordered_set<std::uint64_t> edges;
    std::unordered_set<std::uint64_t> faces;
    vertices.reserve(candidate.size() * 8);
    edges.reserve(candidate.size() * 12);
    faces.reserve(candidate.size() * 6);
    std::int64_t added_vertices = 0;
    std::int64_t added_edges = 0;
    std::int64_t added_faces = 0;
    for (const std::size_t index : candidate)
    {
        const int x = static_cast<int>(index % dimensions[0]);
        const int y = static_cast<int>(
            (index / dimensions[0]) % dimensions[1]);
        const int z = static_cast<int>(
            index / (dimensions[0] * dimensions[1]));
        for (int dz = 0; dz <= 1; ++dz)
        {
            for (int dy = 0; dy <= 1; ++dy)
            {
                for (int dx = 0; dx <= 1; ++dx)
                {
                    const int vx = x + dx;
                    const int vy = y + dy;
                    const int vz = z + dz;
                    if (vertices.insert(
                            vertexId(dimensions, vx, vy, vz)).second &&
                        !anyIncidentVoxel(
                            dimensions,
                            occupied,
                            vx - 1,
                            vx,
                            vy - 1,
                            vy,
                            vz - 1,
                            vz))
                    {
                        ++added_vertices;
                    }
                }
            }
        }
        for (int dz = 0; dz <= 1; ++dz)
        {
            for (int dy = 0; dy <= 1; ++dy)
            {
                const int ey = y + dy;
                const int ez = z + dz;
                if (edges.insert(xEdgeId(dimensions, x, ey, ez)).second &&
                    !anyIncidentVoxel(
                        dimensions, occupied, x, x, ey - 1, ey, ez - 1, ez))
                {
                    ++added_edges;
                }
            }
        }
        for (int dz = 0; dz <= 1; ++dz)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const int ex = x + dx;
                const int ez = z + dz;
                if (edges.insert(yEdgeId(dimensions, ex, y, ez)).second &&
                    !anyIncidentVoxel(
                        dimensions, occupied, ex - 1, ex, y, y, ez - 1, ez))
                {
                    ++added_edges;
                }
            }
        }
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const int ex = x + dx;
                const int ey = y + dy;
                if (edges.insert(zEdgeId(dimensions, ex, ey, z)).second &&
                    !anyIncidentVoxel(
                        dimensions, occupied, ex - 1, ex, ey - 1, ey, z, z))
                {
                    ++added_edges;
                }
            }
        }
        for (int dx = 0; dx <= 1; ++dx)
        {
            const int fx = x + dx;
            if (faces.insert(xFaceId(dimensions, fx, y, z)).second &&
                !occupiedAt(dimensions, occupied, fx - 1, y, z) &&
                !occupiedAt(dimensions, occupied, fx, y, z))
            {
                ++added_faces;
            }
        }
        for (int dy = 0; dy <= 1; ++dy)
        {
            const int fy = y + dy;
            if (faces.insert(yFaceId(dimensions, x, fy, z)).second &&
                !occupiedAt(dimensions, occupied, x, fy - 1, z) &&
                !occupiedAt(dimensions, occupied, x, fy, z))
            {
                ++added_faces;
            }
        }
        for (int dz = 0; dz <= 1; ++dz)
        {
            const int fz = z + dz;
            if (faces.insert(zFaceId(dimensions, x, y, fz)).second &&
                !occupiedAt(dimensions, occupied, x, y, fz - 1) &&
                !occupiedAt(dimensions, occupied, x, y, fz))
            {
                ++added_faces;
            }
        }
    }
    const std::int64_t delta = added_vertices - added_edges + added_faces -
        static_cast<std::int64_t>(candidate.size());
    return static_cast<int>(std::clamp<std::int64_t>(
        delta,
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()));
}

struct TopologySubsetCandidate
{
    std::vector<std::size_t> samples;
    int eulerDelta = 0;
    std::size_t seed = std::numeric_limits<std::size_t>::max();
};

TopologySubsetCandidate growTopologyImprovingSubset(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const std::vector<std::size_t> &component,
    const std::vector<std::uint8_t> &protected_empty,
    const std::vector<std::uint8_t> &rejected_seed,
    std::size_t maximum_subset_sample_count,
    int maximum_seed_count,
    int *attempted_seed_count)
{
    TopologySubsetCandidate best;
    if (maximum_subset_sample_count == 0 ||
        *attempted_seed_count >= maximum_seed_count)
    {
        return best;
    }

    std::unordered_set<std::size_t> component_samples(
        component.cbegin(), component.cend());
    std::vector<std::size_t> seeds = component;
    std::sort(seeds.begin(), seeds.end());
    for (const std::size_t seed : seeds)
    {
        if (*attempted_seed_count >= maximum_seed_count)
        {
            break;
        }
        if (protected_empty[seed] != 0 || rejected_seed[seed] != 0 ||
            !touchesOccupiedByFace(dimensions, occupied, seed))
        {
            continue;
        }
        ++*attempted_seed_count;

        std::vector<std::size_t> patch{seed};
        std::unordered_set<std::size_t> patch_samples{seed};
        std::set<std::size_t> frontier;
        const auto add_frontier_neighbors =
            [&](std::size_t current)
        {
            visitSixNeighbors(
                dimensions,
                current,
                [&](std::size_t neighbor)
                {
                    if (component_samples.contains(neighbor) &&
                        !patch_samples.contains(neighbor) &&
                        protected_empty[neighbor] == 0)
                    {
                        frontier.insert(neighbor);
                    }
                });
        };
        add_frontier_neighbors(seed);

        int current_delta = candidateEulerDelta(
            dimensions, occupied, patch);
        if (current_delta > 0)
        {
            best.samples = std::move(patch);
            best.eulerDelta = current_delta;
            best.seed = seed;
            return best;
        }
        if (current_delta < 0)
        {
            continue;
        }

        while (!frontier.empty() &&
               patch.size() < maximum_subset_sample_count)
        {
            std::size_t selected =
                std::numeric_limits<std::size_t>::max();
            int selected_delta = std::numeric_limits<int>::min();
            for (const std::size_t index : frontier)
            {
                std::vector<std::size_t> trial = patch;
                trial.push_back(index);
                const int delta = candidateEulerDelta(
                    dimensions, occupied, trial);
                if (delta > selected_delta ||
                    (delta == selected_delta && index < selected))
                {
                    selected = index;
                    selected_delta = delta;
                }
            }
            if (selected == std::numeric_limits<std::size_t>::max() ||
                selected_delta < current_delta)
            {
                break;
            }
            frontier.erase(selected);
            patch.push_back(selected);
            patch_samples.insert(selected);
            current_delta = selected_delta;
            if (current_delta > 0)
            {
                if (best.samples.empty() ||
                    patch.size() < best.samples.size() ||
                    (patch.size() == best.samples.size() && seed < best.seed))
                {
                    best.samples = patch;
                    best.eulerDelta = current_delta;
                    best.seed = seed;
                }
                break;
            }
            add_frontier_neighbors(selected);
        }
    }
    return best;
}

} // namespace

int VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied)
{
    if (sampleCount(dimensions) != occupied.size())
    {
        return 0;
    }
    return bodyEuler(dimensions, occupied);
}

VisibilityOccupancyHandleRepairResult VisibilityOccupancyHandleRepair::repair(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const std::vector<std::uint8_t> &closingProposal,
    const std::vector<std::uint8_t> &protectedEmpty,
    const VisibilityOccupancyHandleRepairOptions &options)
{
    VisibilityOccupancyHandleRepairResult result;
    result.occupied = occupied;
    const std::size_t count = sampleCount(dimensions);
    if (count == 0 || occupied.size() != count ||
        closingProposal.size() != count || protectedEmpty.size() != count)
    {
        result.error = "visibility occupancy handle repair input is invalid";
        return result;
    }
    for (std::size_t index = 0; index < count; ++index)
    {
        if (occupied[index] != 0 && closingProposal[index] == 0)
        {
            result.error = "visibility occupancy closing proposal removes full samples";
            return result;
        }
    }
    if (options.isCancelled && options.isCancelled())
    {
        result.cancelled = true;
        return result;
    }

    result.statistics.sampleCount = count;
    const std::vector<std::uint8_t> initial_exterior =
        exteriorReachableEmpty(dimensions, occupied);
    std::vector<std::uint8_t> required_protected(count, 0);
    for (std::size_t index = 0; index < count; ++index)
    {
        result.statistics.proposalAddedSampleCount +=
            occupied[index] == 0 && closingProposal[index] != 0;
        required_protected[index] =
            occupied[index] == 0 && protectedEmpty[index] != 0 &&
            initial_exterior[index] != 0;
        result.statistics.protectedExteriorSampleCountBefore +=
            required_protected[index] != 0;
    }

    const auto components = proposalComponents(
        dimensions, occupied, closingProposal);
    result.statistics.candidateComponentCount =
        static_cast<int>(components.size());
    int current_euler = bodyEuler(dimensions, result.occupied);
    result.statistics.bodyEulerBefore = current_euler;
    const int occupied_component_count =
        occupiedComponentCount(dimensions, result.occupied);
    const int maximum_accepted =
        std::max(0, options.maximumAcceptedCandidateCount);
    std::vector<std::uint8_t> whole_component_eligible(
        components.size(), 1);
    std::vector<std::uint8_t> subset_eligible(components.size(), 1);
    std::vector<std::uint8_t> rejected_subset_sample(count, 0);
    int attempted_subset_seed_count = 0;
    for (std::size_t component_index = 0;
         component_index < components.size();
         ++component_index)
    {
        const auto &component = components[component_index];
        if (component.size() > options.maximumCandidateSampleCount)
        {
            whole_component_eligible[component_index] = 0;
            ++result.statistics.rejectedOversizedCandidateCount;
        }
        const bool contains_protected = std::any_of(
            component.cbegin(),
            component.cend(),
            [&](std::size_t index)
            {
                return protectedEmpty[index] != 0;
            });
        if (contains_protected)
        {
            whole_component_eligible[component_index] = 0;
            ++result.statistics.rejectedProtectedCandidateCount;
        }
    }

    while (result.statistics.acceptedCandidateCount < maximum_accepted)
    {
        if (options.isCancelled && options.isCancelled())
        {
            result.cancelled = true;
            result.occupied = occupied;
            return result;
        }

        std::size_t selected_component = components.size();
        int selected_delta = 0;
        std::vector<std::size_t> selected_candidate;
        bool selected_is_subset = false;
        bool selected_is_plateau_subset = false;
        for (std::size_t component_index = 0;
             component_index < components.size();
             ++component_index)
        {
            if (whole_component_eligible[component_index] == 0)
            {
                continue;
            }
            const int delta = candidateEulerDelta(
                dimensions,
                result.occupied,
                components[component_index]);
            if (delta > selected_delta ||
                (delta == selected_delta && delta > 0 &&
                 (selected_component == components.size() ||
                  components[component_index].size() <
                      components[selected_component].size())))
            {
                selected_component = component_index;
                selected_delta = delta;
            }
        }
        if (selected_component != components.size())
        {
            selected_candidate = components[selected_component];
        }
        else
        {
            std::size_t selected_sample = count;
            for (std::size_t component_index = 0;
                 component_index < components.size();
                 ++component_index)
            {
                if (subset_eligible[component_index] == 0)
                {
                    continue;
                }
                for (const std::size_t index : components[component_index])
                {
                    if (result.occupied[index] != 0 ||
                        protectedEmpty[index] != 0 ||
                        rejected_subset_sample[index] != 0 ||
                        !touchesOccupiedByFace(
                            dimensions, result.occupied, index))
                    {
                        continue;
                    }
                    const std::array<std::size_t, 1> singleton{index};
                    const int delta = candidateEulerDelta(
                        dimensions, result.occupied, singleton);
                    if (delta > selected_delta ||
                        (delta == selected_delta && delta > 0 &&
                         index < selected_sample))
                    {
                        selected_component = component_index;
                        selected_sample = index;
                        selected_delta = delta;
                    }
                }
            }
            if (selected_sample != count)
            {
                selected_candidate.push_back(selected_sample);
                selected_is_subset = true;
            }
            else
            {
                TopologySubsetCandidate best_subset;
                for (std::size_t component_index = 0;
                     component_index < components.size();
                     ++component_index)
                {
                    if (subset_eligible[component_index] == 0)
                    {
                        continue;
                    }
                    TopologySubsetCandidate candidate =
                        growTopologyImprovingSubset(
                            dimensions,
                            result.occupied,
                            components[component_index],
                            protectedEmpty,
                            rejected_subset_sample,
                            std::max<std::size_t>(
                                1,
                                options.maximumSubsetSampleCount),
                            std::max(0, options.maximumSubsetSeedCount),
                            &attempted_subset_seed_count);
                    if (candidate.samples.empty())
                    {
                        continue;
                    }
                    if (best_subset.samples.empty() ||
                        candidate.samples.size() < best_subset.samples.size() ||
                        (candidate.samples.size() ==
                             best_subset.samples.size() &&
                         candidate.seed < best_subset.seed))
                    {
                        selected_component = component_index;
                        best_subset = std::move(candidate);
                    }
                }
                if (!best_subset.samples.empty())
                {
                    selected_candidate = std::move(best_subset.samples);
                    selected_delta = best_subset.eulerDelta;
                    selected_is_subset = true;
                    selected_is_plateau_subset = true;
                }
            }
        }
        if (selected_candidate.empty())
        {
            result.statistics.rejectedTopologyCandidateCount +=
                static_cast<int>(std::count(
                    subset_eligible.cbegin(), subset_eligible.cend(), 1));
            break;
        }

        for (const std::size_t index : selected_candidate)
        {
            result.occupied[index] = 1;
        }

        const int candidate_euler = bodyEuler(dimensions, result.occupied);
        const bool preserves_full_components =
            occupiedComponentCount(dimensions, result.occupied) ==
            occupied_component_count;
        const std::vector<std::uint8_t> candidate_exterior =
            exteriorReachableEmpty(dimensions, result.occupied);
        bool preserves_exterior_reachability = true;
        for (std::size_t index = 0; index < count; ++index)
        {
            if (required_protected[index] != 0 &&
                candidate_exterior[index] == 0)
            {
                preserves_exterior_reachability = false;
                break;
            }
        }
        if (candidate_euler <= current_euler ||
            !preserves_full_components ||
            !preserves_exterior_reachability)
        {
            for (const std::size_t index : selected_candidate)
            {
                result.occupied[index] = 0;
            }
            if (selected_is_subset)
            {
                rejected_subset_sample[selected_candidate.front()] = 1;
            }
            else
            {
                whole_component_eligible[selected_component] = 0;
                if (selected_candidate.size() == 1)
                {
                    subset_eligible[selected_component] = 0;
                    rejected_subset_sample[selected_candidate.front()] = 1;
                }
            }
            if (!preserves_exterior_reachability)
            {
                ++result.statistics
                      .rejectedProtectedReachabilityCandidateCount;
            }
            else
            {
                ++result.statistics.rejectedTopologyCandidateCount;
            }
            continue;
        }

        current_euler = candidate_euler;
        whole_component_eligible[selected_component] = 0;
        subset_eligible[selected_component] = 0;
        ++result.statistics.acceptedCandidateCount;
        result.statistics.acceptedSubsetCandidateCount += selected_is_subset;
        result.statistics.acceptedPlateauSubsetCandidateCount +=
            selected_is_plateau_subset;
        result.statistics.filledSampleCount += selected_candidate.size();
    }

    result.statistics.attemptedSubsetSeedCount =
        attempted_subset_seed_count;

    const std::vector<std::uint8_t> final_exterior =
        exteriorReachableEmpty(dimensions, result.occupied);
    for (std::size_t index = 0; index < count; ++index)
    {
        result.statistics.protectedExteriorSampleCountAfter +=
            required_protected[index] != 0 && final_exterior[index] != 0;
    }
    result.statistics.bodyEulerAfter = bodyEuler(dimensions, result.occupied);
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
