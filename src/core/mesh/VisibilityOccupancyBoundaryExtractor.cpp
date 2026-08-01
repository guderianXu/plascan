#include "VisibilityOccupancyBoundaryExtractor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xjw::mesh
{
namespace
{

bool checkedProduct(
    const std::array<int, 3> &dimensions,
    std::size_t *product)
{
    std::size_t value = 1;
    for (const int dimension : dimensions)
    {
        if (dimension <= 0 ||
            value > std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(dimension))
        {
            return false;
        }
        value *= static_cast<std::size_t>(dimension);
    }
    *product = value;
    return true;
}

std::size_t cellIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) * dimensions[1] + y) *
        dimensions[0] + x;
}

bool isOccupied(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    int x,
    int y,
    int z)
{
    return x >= 0 && y >= 0 && z >= 0 &&
        x < dimensions[0] && y < dimensions[1] && z < dimensions[2] &&
        occupied[cellIndex(dimensions, x, y, z)] != 0;
}

std::size_t cornerIndex(
    const std::array<int, 3> &cornerDimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) * cornerDimensions[1] + y) *
        cornerDimensions[0] + x;
}

std::uint64_t edgeKey(int first, int second)
{
    const std::uint32_t low = static_cast<std::uint32_t>(
        std::min(first, second));
    const std::uint32_t high = static_cast<std::uint32_t>(
        std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

std::uint64_t countNonManifoldVertices(const TriMesh &mesh)
{
    std::vector<std::vector<std::pair<int, int>>> link_edges(
        mesh.vertices.size());
    for (const Triangle &face : mesh.faces)
    {
        link_edges[face.v[0]].emplace_back(face.v[1], face.v[2]);
        link_edges[face.v[1]].emplace_back(face.v[2], face.v[0]);
        link_edges[face.v[2]].emplace_back(face.v[0], face.v[1]);
    }

    std::uint64_t non_manifold_count = 0;
    for (const auto &edges : link_edges)
    {
        std::vector<int> nodes;
        std::vector<int> degrees;
        std::vector<int> parents;
        const auto nodeIndex = [&](int vertex)
        {
            const auto found = std::find(nodes.cbegin(), nodes.cend(), vertex);
            if (found != nodes.cend())
            {
                return static_cast<int>(found - nodes.cbegin());
            }
            const int index = static_cast<int>(nodes.size());
            nodes.push_back(vertex);
            degrees.push_back(0);
            parents.push_back(index);
            return index;
        };
        const auto root = [&parents](int value)
        {
            int current = value;
            while (parents[current] != current)
            {
                current = parents[current];
            }
            while (parents[value] != value)
            {
                const int next = parents[value];
                parents[value] = current;
                value = next;
            }
            return current;
        };
        for (const auto &[first, second] : edges)
        {
            const int first_index = nodeIndex(first);
            const int second_index = nodeIndex(second);
            ++degrees[first_index];
            ++degrees[second_index];
            const int first_root = root(first_index);
            const int second_root = root(second_index);
            if (first_root != second_root)
            {
                parents[second_root] = first_root;
            }
        }
        bool manifold = !nodes.empty() &&
            std::all_of(
                degrees.cbegin(),
                degrees.cend(),
                [](int degree)
                {
                    return degree == 2;
                });
        if (manifold)
        {
            const int first_root = root(0);
            for (int index = 1; index < static_cast<int>(nodes.size()); ++index)
            {
                if (root(index) != first_root)
                {
                    manifold = false;
                    break;
                }
            }
        }
        non_manifold_count += !manifold;
    }
    return non_manifold_count;
}

} // namespace

VisibilityOccupancyBoundaryResult
VisibilityOccupancyBoundaryExtractor::extract(
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::array<int, 3> &cellDimensions,
    const std::vector<std::uint8_t> &occupied,
    const VisibilityOccupancyBoundaryOptions &options)
{
    VisibilityOccupancyBoundaryResult result;
    std::size_t cell_count = 0;
    if (!checkedProduct(cellDimensions, &cell_count) ||
        occupied.size() != cell_count)
    {
        result.errorMessage = "visibility occupancy boundary dimensions are invalid";
        return result;
    }

    std::array<float, 3> spacing{};
    std::array<int, 3> corner_dimensions{};
    for (int axis = 0; axis < 3; ++axis)
    {
        if (cellDimensions[axis] < 2 ||
            !(boundsMax[axis] > boundsMin[axis]))
        {
            result.errorMessage = "visibility occupancy boundary bounds are invalid";
            return result;
        }
        spacing[axis] =
            (boundsMax[axis] - boundsMin[axis]) /
            static_cast<float>(cellDimensions[axis] - 1);
        corner_dimensions[axis] = cellDimensions[axis] + 1;
    }

    std::size_t corner_count = 0;
    if (!checkedProduct(corner_dimensions, &corner_count) ||
        corner_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max()))
    {
        result.errorMessage = "visibility occupancy boundary grid is too large";
        return result;
    }
    std::vector<int> vertex_indices(corner_count, -1);

    const auto vertexIndex =
        [&](int x, int y, int z) -> int
    {
        const std::size_t lookup_index =
            cornerIndex(corner_dimensions, x, y, z);
        int &mesh_index = vertex_indices[lookup_index];
        if (mesh_index >= 0)
        {
            return mesh_index;
        }
        mesh_index = static_cast<int>(result.mesh.vertices.size());
        MeshVertex vertex;
        vertex.x = boundsMin[0] +
            (static_cast<float>(x) - 0.5f) * spacing[0];
        vertex.y = boundsMin[1] +
            (static_cast<float>(y) - 0.5f) * spacing[1];
        vertex.z = boundsMin[2] +
            (static_cast<float>(z) - 0.5f) * spacing[2];
        result.mesh.vertices.push_back(vertex);
        return mesh_index;
    };

    const auto appendQuad =
        [&](const std::array<std::array<int, 3>, 4> &corners)
    {
        const int v0 = vertexIndex(
            corners[0][0], corners[0][1], corners[0][2]);
        const int v1 = vertexIndex(
            corners[1][0], corners[1][1], corners[1][2]);
        const int v2 = vertexIndex(
            corners[2][0], corners[2][1], corners[2][2]);
        const int v3 = vertexIndex(
            corners[3][0], corners[3][1], corners[3][2]);
        result.mesh.faces.push_back(Triangle{{v0, v1, v2}});
        result.mesh.faces.push_back(Triangle{{v0, v2, v3}});
        ++result.statistics.exposedQuadCount;
    };

    for (int z = 0; z < cellDimensions[2]; ++z)
    {
        if ((z & 3) == 0 && options.isCancelled && options.isCancelled())
        {
            result.cancelled = true;
            result.errorMessage = "visibility occupancy boundary extraction cancelled";
            return result;
        }
        for (int y = 0; y < cellDimensions[1]; ++y)
        {
            for (int x = 0; x < cellDimensions[0]; ++x)
            {
                if (!isOccupied(cellDimensions, occupied, x, y, z))
                {
                    continue;
                }
                ++result.statistics.occupiedCellCount;
                if (!isOccupied(cellDimensions, occupied, x - 1, y, z))
                {
                    appendQuad({{{x, y, z},
                                 {x, y, z + 1},
                                 {x, y + 1, z + 1},
                                 {x, y + 1, z}}});
                }
                if (!isOccupied(cellDimensions, occupied, x + 1, y, z))
                {
                    appendQuad({{{x + 1, y, z},
                                 {x + 1, y + 1, z},
                                 {x + 1, y + 1, z + 1},
                                 {x + 1, y, z + 1}}});
                }
                if (!isOccupied(cellDimensions, occupied, x, y - 1, z))
                {
                    appendQuad({{{x, y, z},
                                 {x + 1, y, z},
                                 {x + 1, y, z + 1},
                                 {x, y, z + 1}}});
                }
                if (!isOccupied(cellDimensions, occupied, x, y + 1, z))
                {
                    appendQuad({{{x, y + 1, z},
                                 {x, y + 1, z + 1},
                                 {x + 1, y + 1, z + 1},
                                 {x + 1, y + 1, z}}});
                }
                if (!isOccupied(cellDimensions, occupied, x, y, z - 1))
                {
                    appendQuad({{{x, y, z},
                                 {x, y + 1, z},
                                 {x + 1, y + 1, z},
                                 {x + 1, y, z}}});
                }
                if (!isOccupied(cellDimensions, occupied, x, y, z + 1))
                {
                    appendQuad({{{x, y, z + 1},
                                 {x + 1, y, z + 1},
                                 {x + 1, y + 1, z + 1},
                                 {x, y + 1, z + 1}}});
                }
            }
        }
    }

    if (result.mesh.empty())
    {
        result.errorMessage = "visibility occupancy boundary is empty";
        return result;
    }
    result.statistics.outputVertexCount = result.mesh.vertices.size();
    result.statistics.outputFaceCount = result.mesh.faces.size();
    std::unordered_map<std::uint64_t, int> edge_incidence;
    edge_incidence.reserve(result.mesh.faces.size() * 2U);
    for (const Triangle &face : result.mesh.faces)
    {
        ++edge_incidence[edgeKey(face.v[0], face.v[1])];
        ++edge_incidence[edgeKey(face.v[1], face.v[2])];
        ++edge_incidence[edgeKey(face.v[2], face.v[0])];
    }
    result.statistics.uniqueEdgeCount = edge_incidence.size();
    for (const auto &[key, incidence] : edge_incidence)
    {
        static_cast<void>(key);
        result.statistics.boundaryEdgeCount += incidence == 1;
        result.statistics.nonManifoldEdgeCount += incidence > 2;
    }
    result.statistics.nonManifoldVertexCount =
        countNonManifoldVertices(result.mesh);
    const std::int64_t euler =
        static_cast<std::int64_t>(result.mesh.vertices.size()) -
        static_cast<std::int64_t>(edge_incidence.size()) +
        static_cast<std::int64_t>(result.mesh.faces.size());
    result.statistics.eulerCharacteristic = static_cast<int>(
        std::clamp<std::int64_t>(
            euler,
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::max()));
    result.statistics.closedTwoManifold =
        result.statistics.boundaryEdgeCount == 0 &&
        result.statistics.nonManifoldEdgeCount == 0 &&
        result.statistics.nonManifoldVertexCount == 0;
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
