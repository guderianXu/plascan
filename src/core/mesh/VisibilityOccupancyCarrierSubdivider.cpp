#include "VisibilityOccupancyCarrierSubdivider.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <utility>

namespace xjw::mesh
{
namespace
{

using EdgeKey = std::pair<int, int>;

EdgeKey edgeKey(int first, int second)
{
    return {std::min(first, second), std::max(first, second)};
}

bool isCancelled(
    const VisibilityOccupancyCarrierSubdivisionOptions &options)
{
    return options.isCancelled && options.isCancelled();
}

bool isFinite(const MeshVertex &vertex)
{
    return std::isfinite(vertex.x) &&
        std::isfinite(vertex.y) &&
        std::isfinite(vertex.z) &&
        std::isfinite(vertex.nx) &&
        std::isfinite(vertex.ny) &&
        std::isfinite(vertex.nz);
}

bool hasValidIndices(const Triangle &face, std::size_t vertexCount)
{
    for (const int vertex_index : face.v)
    {
        if (vertex_index < 0 ||
            static_cast<std::size_t>(vertex_index) >= vertexCount)
        {
            return false;
        }
    }
    return face.v[0] != face.v[1] &&
        face.v[1] != face.v[2] &&
        face.v[2] != face.v[0];
}

bool isGeometricallyDegenerate(
    const TriMesh &mesh,
    const Triangle &face)
{
    const MeshVertex &first =
        mesh.vertices[static_cast<std::size_t>(face.v[0])];
    const MeshVertex &second =
        mesh.vertices[static_cast<std::size_t>(face.v[1])];
    const MeshVertex &third =
        mesh.vertices[static_cast<std::size_t>(face.v[2])];

    const double ab_x = static_cast<double>(second.x) - first.x;
    const double ab_y = static_cast<double>(second.y) - first.y;
    const double ab_z = static_cast<double>(second.z) - first.z;
    const double ac_x = static_cast<double>(third.x) - first.x;
    const double ac_y = static_cast<double>(third.y) - first.y;
    const double ac_z = static_cast<double>(third.z) - first.z;
    const double bc_x = static_cast<double>(third.x) - second.x;
    const double bc_y = static_cast<double>(third.y) - second.y;
    const double bc_z = static_cast<double>(third.z) - second.z;

    const double cross_x = ab_y * ac_z - ab_z * ac_y;
    const double cross_y = ab_z * ac_x - ab_x * ac_z;
    const double cross_z = ab_x * ac_y - ab_y * ac_x;
    const double cross_squared =
        cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
    const double maximum_edge_squared = std::max({
        ab_x * ab_x + ab_y * ab_y + ab_z * ab_z,
        ac_x * ac_x + ac_y * ac_y + ac_z * ac_z,
        bc_x * bc_x + bc_y * bc_y + bc_z * bc_z});
    const double relative_tolerance =
        std::numeric_limits<double>::epsilon() *
        maximum_edge_squared * maximum_edge_squared;
    return !std::isfinite(cross_squared) ||
        !(maximum_edge_squared > 0.0) ||
        cross_squared <= relative_tolerance;
}

std::uint8_t midpointColor(std::uint8_t first, std::uint8_t second)
{
    const unsigned int sum =
        static_cast<unsigned int>(first) +
        static_cast<unsigned int>(second);
    return static_cast<std::uint8_t>((sum + 1U) / 2U);
}

MeshVertex interpolateMidpoint(
    const MeshVertex &first,
    const MeshVertex &second)
{
    MeshVertex midpoint;
    midpoint.x = static_cast<float>(
        0.5 * (static_cast<double>(first.x) + second.x));
    midpoint.y = static_cast<float>(
        0.5 * (static_cast<double>(first.y) + second.y));
    midpoint.z = static_cast<float>(
        0.5 * (static_cast<double>(first.z) + second.z));

    const double normal_x =
        0.5 * (static_cast<double>(first.nx) + second.nx);
    const double normal_y =
        0.5 * (static_cast<double>(first.ny) + second.ny);
    const double normal_z =
        0.5 * (static_cast<double>(first.nz) + second.nz);
    const double normal_length = std::sqrt(
        normal_x * normal_x +
        normal_y * normal_y +
        normal_z * normal_z);
    if (normal_length > 1.0e-12)
    {
        midpoint.nx = static_cast<float>(normal_x / normal_length);
        midpoint.ny = static_cast<float>(normal_y / normal_length);
        midpoint.nz = static_cast<float>(normal_z / normal_length);
    }
    else
    {
        midpoint.nx = 0.0f;
        midpoint.ny = 0.0f;
        midpoint.nz = 0.0f;
    }

    midpoint.r = midpointColor(first.r, second.r);
    midpoint.g = midpointColor(first.g, second.g);
    midpoint.b = midpointColor(first.b, second.b);
    return midpoint;
}

VisibilityOccupancyCarrierSubdivisionResult cancelledResult(
    VisibilityOccupancyCarrierSubdivisionResult result)
{
    result.ok = false;
    result.cancelled = true;
    result.errorMessage = "visibility occupancy carrier subdivision cancelled";
    result.mesh = {};
    result.statistics.outputVertexCount = 0;
    result.statistics.outputFaceCount = 0;
    return result;
}

} // namespace

VisibilityOccupancyCarrierSubdivisionResult
VisibilityOccupancyCarrierSubdivider::subdivide(
    const TriMesh &mesh,
    const VisibilityOccupancyCarrierSubdivisionOptions &options)
{
    VisibilityOccupancyCarrierSubdivisionResult result;
    result.statistics.inputVertexCount =
        static_cast<std::uint64_t>(mesh.vertices.size());
    result.statistics.inputFaceCount =
        static_cast<std::uint64_t>(mesh.faces.size());

    if (isCancelled(options))
    {
        return cancelledResult(std::move(result));
    }
    if (mesh.vertices.empty() || mesh.faces.empty())
    {
        result.errorMessage = "carrier mesh must contain vertices and faces";
        return result;
    }
    if (mesh.vertices.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        result.errorMessage = "carrier mesh has too many vertices";
        return result;
    }

    for (const MeshVertex &vertex : mesh.vertices)
    {
        if (isCancelled(options))
        {
            return cancelledResult(std::move(result));
        }
        if (!isFinite(vertex))
        {
            result.errorMessage = "carrier mesh contains a non-finite vertex";
            return result;
        }
    }

    std::map<EdgeKey, int> midpoint_indices;
    for (const Triangle &face : mesh.faces)
    {
        if (isCancelled(options))
        {
            return cancelledResult(std::move(result));
        }
        if (!hasValidIndices(face, mesh.vertices.size()))
        {
            result.errorMessage =
                "carrier mesh contains an invalid or repeated face index";
            return result;
        }
        if (isGeometricallyDegenerate(mesh, face))
        {
            result.errorMessage =
                "carrier mesh contains a geometrically degenerate face";
            return result;
        }
        midpoint_indices.try_emplace(edgeKey(face.v[0], face.v[1]), -1);
        midpoint_indices.try_emplace(edgeKey(face.v[1], face.v[2]), -1);
        midpoint_indices.try_emplace(edgeKey(face.v[2], face.v[0]), -1);
        ++result.statistics.validatedFaceCount;
    }

    result.statistics.uniqueInputEdgeCount =
        static_cast<std::uint64_t>(midpoint_indices.size());
    if (midpoint_indices.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max()) -
            mesh.vertices.size())
    {
        result.errorMessage = "subdivided carrier mesh has too many vertices";
        return result;
    }
    if (mesh.faces.size() > result.mesh.faces.max_size() / 4U)
    {
        result.errorMessage = "subdivided carrier mesh has too many faces";
        return result;
    }

    result.mesh.hasVertexColors = mesh.hasVertexColors;
    result.mesh.vertices = mesh.vertices;
    result.mesh.vertices.reserve(
        mesh.vertices.size() + midpoint_indices.size());
    result.mesh.faces.reserve(mesh.faces.size() * 4U);

    for (auto &[edge, midpoint_index] : midpoint_indices)
    {
        if (isCancelled(options))
        {
            return cancelledResult(std::move(result));
        }
        midpoint_index = static_cast<int>(result.mesh.vertices.size());
        result.mesh.vertices.push_back(interpolateMidpoint(
            mesh.vertices[static_cast<std::size_t>(edge.first)],
            mesh.vertices[static_cast<std::size_t>(edge.second)]));
        ++result.statistics.createdMidpointVertexCount;
    }

    for (const Triangle &face : mesh.faces)
    {
        if (isCancelled(options))
        {
            return cancelledResult(std::move(result));
        }
        const int first = face.v[0];
        const int second = face.v[1];
        const int third = face.v[2];
        const int first_second =
            midpoint_indices.at(edgeKey(first, second));
        const int second_third =
            midpoint_indices.at(edgeKey(second, third));
        const int third_first =
            midpoint_indices.at(edgeKey(third, first));
        result.mesh.faces.push_back(
            {{first, first_second, third_first}});
        result.mesh.faces.push_back(
            {{first_second, second, second_third}});
        result.mesh.faces.push_back(
            {{third_first, second_third, third}});
        result.mesh.faces.push_back(
            {{first_second, second_third, third_first}});
        ++result.statistics.subdividedFaceCount;
    }

    result.statistics.outputVertexCount =
        static_cast<std::uint64_t>(result.mesh.vertices.size());
    result.statistics.outputFaceCount =
        static_cast<std::uint64_t>(result.mesh.faces.size());
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
