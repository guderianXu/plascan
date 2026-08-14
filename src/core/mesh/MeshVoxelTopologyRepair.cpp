#include "MeshVoxelTopologyRepair.h"

#include "Mc33IsoSurfaceExtractor.h"
#include "MeshTopologyQuality.h"
#include "VisibilityOccupancyBoundaryExtractor.h"
#include "VisibilityOccupancyHandleRepair.h"
#include "VisibilityOccupancyWellComposedRepair.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace xjw::mesh
{
namespace
{

constexpr std::array<std::array<int, 3>, 6> kSixNeighbours{{
    {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
    {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}}};

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct GridLayout
{
    std::array<float, 3> boundsMin{};
    std::array<float, 3> boundsMax{};
    std::array<int, 3> dimensions{};
    float spacing = 0.0f;
};

struct SolidFill
{
    std::vector<std::uint8_t> occupied;
    std::uint64_t enclosedInteriorCount = 0;
};

struct ComponentSummary
{
    int count = 0;
    int largestLabel = -1;
    std::uint64_t largestSize = 0;
    std::vector<int> labels;
};

struct Candidate
{
    bool valid = false;
    TriMesh mesh;
    int closingRadius = -1;
    std::uint64_t enclosedInteriorCount = 0;
    std::uint64_t occupiedCount = 0;
    std::uint64_t discardedCount = 0;
    int cubicalEuler = 0;
    bool largestComponentFallback = false;
    bool smoothExtractionPreferred = false;
    bool smoothExtractionAvailable = false;
    bool smoothExtractionAttempted = false;
    bool smoothExtractionAccepted = false;
    bool smoothExtractionRejectedByTopology = false;
    bool cellBoundaryExtractionUsed = false;
    std::uint64_t smoothExtractionVertexCount = 0;
    std::uint64_t smoothExtractionFaceCount = 0;
    bool smoothTriangleOptimizationAttempted = false;
    bool smoothTriangleOptimizationCancelled = false;
    bool smoothTriangleOptimizationAccepted = false;
    MeshTriangleOptimizationStatistics smoothOptimization;
    MeshTopologyQualityStatistics smoothQuality;
    MeshTopologyQualityStatistics smoothQualityBeforeOptimization;
    MeshTopologyQualityStatistics quality;
};

Vec3 operator-(const Vec3 &first, const Vec3 &second)
{
    return {first.x - second.x, first.y - second.y, first.z - second.z};
}

Vec3 operator+(const Vec3 &first, const Vec3 &second)
{
    return {first.x + second.x, first.y + second.y, first.z + second.z};
}

Vec3 operator*(const Vec3 &value, double scale)
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

Vec3 position(const MeshVertex &vertex)
{
    return {vertex.x, vertex.y, vertex.z};
}

bool finite(const MeshVertex &vertex)
{
    return std::isfinite(vertex.x) && std::isfinite(vertex.y) &&
        std::isfinite(vertex.z);
}

std::size_t gridIndex(const std::array<int, 3> &dimensions,
                      int x,
                      int y,
                      int z)
{
    return (static_cast<std::size_t>(z) * dimensions[1] + y) *
        dimensions[0] + x;
}

bool inside(const std::array<int, 3> &dimensions, int x, int y, int z)
{
    return x >= 0 && y >= 0 && z >= 0 &&
        x < dimensions[0] && y < dimensions[1] && z < dimensions[2];
}

bool cancelled(const MeshVoxelTopologyRepairOptions &options)
{
    return options.isCancelled && options.isCancelled();
}

std::uint64_t countOccupied(const std::vector<std::uint8_t> &occupied)
{
    return static_cast<std::uint64_t>(std::count_if(
        occupied.cbegin(), occupied.cend(),
        [](std::uint8_t value) { return value != 0; }));
}

double pointTriangleDistanceSquared(const Vec3 &point,
                                    const Vec3 &a,
                                    const Vec3 &b,
                                    const Vec3 &c)
{
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = point - a;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        return dot(ap, ap);
    }
    const Vec3 bp = point - b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        return dot(bp, bp);
    }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const Vec3 projection = a + ab * (d1 / (d1 - d3));
        const Vec3 delta = point - projection;
        return dot(delta, delta);
    }
    const Vec3 cp = point - c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
    {
        return dot(cp, cp);
    }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const Vec3 projection = a + ac * (d2 / (d2 - d6));
        const Vec3 delta = point - projection;
        return dot(delta, delta);
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0)
    {
        const Vec3 edge = c - b;
        const Vec3 projection = b + edge * ((d4 - d3) /
            ((d4 - d3) + (d5 - d6)));
        const Vec3 delta = point - projection;
        return dot(delta, delta);
    }
    const double denominator = 1.0 / (va + vb + vc);
    const Vec3 projection = a + ab * (vb * denominator) +
        ac * (vc * denominator);
    const Vec3 delta = point - projection;
    return dot(delta, delta);
}

bool makeLayout(const TriMesh &mesh,
                const MeshVoxelTopologyRepairOptions &options,
                GridLayout *layout,
                int *effectiveResolution)
{
    std::array<double, 3> minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
    for (const MeshVertex &vertex : mesh.vertices)
    {
        if (!finite(vertex))
        {
            continue;
        }
        minimum[0] = std::min(minimum[0], static_cast<double>(vertex.x));
        minimum[1] = std::min(minimum[1], static_cast<double>(vertex.y));
        minimum[2] = std::min(minimum[2], static_cast<double>(vertex.z));
        maximum[0] = std::max(maximum[0], static_cast<double>(vertex.x));
        maximum[1] = std::max(maximum[1], static_cast<double>(vertex.y));
        maximum[2] = std::max(maximum[2], static_cast<double>(vertex.z));
    }
    const double longest = std::max(
        {maximum[0] - minimum[0], maximum[1] - minimum[1],
         maximum[2] - minimum[2]});
    if (!(longest > 0.0) || !std::isfinite(longest))
    {
        return false;
    }
    *effectiveResolution = std::clamp(options.targetResolution, 24, 256);
    const int margin = std::max(3, options.maximumClosingRadius + 2);
    const int usable_cells = *effectiveResolution - 2 * margin - 1;
    if (usable_cells < 8)
    {
        return false;
    }
    layout->spacing = static_cast<float>(longest / usable_cells);
    for (int axis = 0; axis < 3; ++axis)
    {
        const double extent = maximum[axis] - minimum[axis];
        layout->dimensions[axis] = std::min(
            *effectiveResolution,
            static_cast<int>(std::ceil(extent / layout->spacing)) +
                2 * margin + 1);
        layout->boundsMin[axis] = static_cast<float>(
            minimum[axis] - margin * layout->spacing);
        layout->boundsMax[axis] = layout->boundsMin[axis] +
            (layout->dimensions[axis] - 1) * layout->spacing;
    }
    return true;
}

bool voxelize(const TriMesh &mesh,
              const GridLayout &layout,
              const MeshVoxelTopologyRepairOptions &options,
              std::vector<std::uint8_t> *surface,
              MeshVoxelTopologyRepairStatistics *statistics)
{
    const std::size_t sample_count =
        static_cast<std::size_t>(layout.dimensions[0]) *
        layout.dimensions[1] * layout.dimensions[2];
    surface->assign(sample_count, 0);
    const double half_diagonal =
        std::sqrt(3.0) * 0.5 * layout.spacing;
    const double maximum_distance_squared =
        half_diagonal * half_diagonal * (1.0 + 1e-9);
    for (std::size_t face_index = 0; face_index < mesh.faces.size(); ++face_index)
    {
        if ((face_index & 1023U) == 0U && cancelled(options))
        {
            return false;
        }
        const Triangle &face = mesh.faces[face_index];
        const bool valid_indices = face.v[0] >= 0 && face.v[1] >= 0 &&
            face.v[2] >= 0 && face.v[0] < mesh.vertexCount() &&
            face.v[1] < mesh.vertexCount() && face.v[2] < mesh.vertexCount();
        if (!valid_indices || !finite(mesh.vertices[face.v[0]]) ||
            !finite(mesh.vertices[face.v[1]]) ||
            !finite(mesh.vertices[face.v[2]]))
        {
            ++statistics->rejectedInputFaceCount;
            continue;
        }
        const Vec3 a = position(mesh.vertices[face.v[0]]);
        const Vec3 b = position(mesh.vertices[face.v[1]]);
        const Vec3 c = position(mesh.vertices[face.v[2]]);
        if (dot(cross(b - a, c - a), cross(b - a, c - a)) <=
            static_cast<double>(layout.spacing) * layout.spacing * 1e-16)
        {
            ++statistics->rejectedInputFaceCount;
            continue;
        }
        ++statistics->validInputFaceCount;
        const std::array<Vec3, 3> vertices{a, b, c};
        std::array<int, 3> lower{};
        std::array<int, 3> upper{};
        for (int axis = 0; axis < 3; ++axis)
        {
            const auto coordinate = [axis](const Vec3 &value)
            {
                return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
            };
            const double triangle_minimum = std::min(
                {coordinate(vertices[0]), coordinate(vertices[1]),
                 coordinate(vertices[2])});
            const double triangle_maximum = std::max(
                {coordinate(vertices[0]), coordinate(vertices[1]),
                 coordinate(vertices[2])});
            lower[axis] = std::clamp(
                static_cast<int>(std::ceil(
                    (triangle_minimum - half_diagonal -
                     layout.boundsMin[axis]) / layout.spacing)),
                0, layout.dimensions[axis] - 1);
            upper[axis] = std::clamp(
                static_cast<int>(std::floor(
                    (triangle_maximum + half_diagonal -
                     layout.boundsMin[axis]) / layout.spacing)),
                0, layout.dimensions[axis] - 1);
        }
        for (int z = lower[2]; z <= upper[2]; ++z)
        {
            for (int y = lower[1]; y <= upper[1]; ++y)
            {
                for (int x = lower[0]; x <= upper[0]; ++x)
                {
                    const Vec3 point{
                        layout.boundsMin[0] + x * layout.spacing,
                        layout.boundsMin[1] + y * layout.spacing,
                        layout.boundsMin[2] + z * layout.spacing};
                    if (pointTriangleDistanceSquared(point, a, b, c) <=
                        maximum_distance_squared)
                    {
                        (*surface)[gridIndex(layout.dimensions, x, y, z)] = 1;
                    }
                }
            }
        }
    }
    statistics->surfaceVoxelCount = countOccupied(*surface);
    return statistics->validInputFaceCount > 0 &&
        statistics->surfaceVoxelCount > 0;
}

std::vector<std::uint8_t> morphologyStep(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &input,
    bool dilate,
    const MeshVoxelTopologyRepairOptions &options,
    bool *wasCancelled)
{
    std::vector<std::uint8_t> output(input.size(), 0);
    for (int z = 0; z < dimensions[2]; ++z)
    {
        if ((z & 7) == 0 && cancelled(options))
        {
            *wasCancelled = true;
            return {};
        }
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                bool value = input[gridIndex(dimensions, x, y, z)] != 0;
                for (const auto &offset : kSixNeighbours)
                {
                    const int nx = x + offset[0];
                    const int ny = y + offset[1];
                    const int nz = z + offset[2];
                    const bool neighbour = inside(dimensions, nx, ny, nz) &&
                        input[gridIndex(dimensions, nx, ny, nz)] != 0;
                    value = dilate ? value || neighbour : value && neighbour;
                }
                output[gridIndex(dimensions, x, y, z)] = value ? 1 : 0;
            }
        }
    }
    return output;
}

std::vector<std::uint8_t> closeGrid(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &input,
    int radius,
    const MeshVoxelTopologyRepairOptions &options,
    bool *wasCancelled)
{
    std::vector<std::uint8_t> current = input;
    for (int pass = 0; pass < radius; ++pass)
    {
        current = morphologyStep(
            dimensions, current, true, options, wasCancelled);
        if (*wasCancelled)
        {
            return {};
        }
    }
    for (int pass = 0; pass < radius; ++pass)
    {
        current = morphologyStep(
            dimensions, current, false, options, wasCancelled);
        if (*wasCancelled)
        {
            return {};
        }
    }
    return current;
}

SolidFill fillSolid(const std::array<int, 3> &dimensions,
                    const std::vector<std::uint8_t> &blocked,
                    const MeshVoxelTopologyRepairOptions &options,
                    bool *wasCancelled)
{
    SolidFill result;
    std::vector<std::uint8_t> exterior(blocked.size(), 0);
    std::vector<std::uint32_t> queue;
    const auto enqueue = [&](int x, int y, int z)
    {
        const std::size_t index = gridIndex(dimensions, x, y, z);
        if (blocked[index] == 0 && exterior[index] == 0)
        {
            exterior[index] = 1;
            queue.push_back(static_cast<std::uint32_t>(index));
        }
    };
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            enqueue(0, y, z);
            enqueue(dimensions[0] - 1, y, z);
        }
    }
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int x = 0; x < dimensions[0]; ++x)
        {
            enqueue(x, 0, z);
            enqueue(x, dimensions[1] - 1, z);
        }
    }
    for (int y = 0; y < dimensions[1]; ++y)
    {
        for (int x = 0; x < dimensions[0]; ++x)
        {
            enqueue(x, y, 0);
            enqueue(x, y, dimensions[2] - 1);
        }
    }
    for (std::size_t head = 0; head < queue.size(); ++head)
    {
        if ((head & 65535U) == 0U && cancelled(options))
        {
            *wasCancelled = true;
            return {};
        }
        const std::size_t index = queue[head];
        const int x = static_cast<int>(index % dimensions[0]);
        const int y = static_cast<int>((index / dimensions[0]) % dimensions[1]);
        const int z = static_cast<int>(index /
            (static_cast<std::size_t>(dimensions[0]) * dimensions[1]));
        for (const auto &offset : kSixNeighbours)
        {
            const int nx = x + offset[0];
            const int ny = y + offset[1];
            const int nz = z + offset[2];
            if (inside(dimensions, nx, ny, nz))
            {
                enqueue(nx, ny, nz);
            }
        }
    }
    result.occupied.resize(blocked.size(), 0);
    for (std::size_t index = 0; index < blocked.size(); ++index)
    {
        result.occupied[index] = exterior[index] == 0 ? 1 : 0;
        result.enclosedInteriorCount +=
            exterior[index] == 0 && blocked[index] == 0;
    }
    return result;
}

ComponentSummary components(const std::array<int, 3> &dimensions,
                            const std::vector<std::uint8_t> &occupied,
                            const MeshVoxelTopologyRepairOptions &options,
                            bool *wasCancelled)
{
    ComponentSummary result;
    result.labels.assign(occupied.size(), -1);
    std::vector<std::uint32_t> queue;
    for (std::size_t seed = 0; seed < occupied.size(); ++seed)
    {
        if (occupied[seed] == 0 || result.labels[seed] >= 0)
        {
            continue;
        }
        if (cancelled(options))
        {
            *wasCancelled = true;
            return {};
        }
        queue.clear();
        queue.push_back(static_cast<std::uint32_t>(seed));
        result.labels[seed] = result.count;
        for (std::size_t head = 0; head < queue.size(); ++head)
        {
            const std::size_t index = queue[head];
            const int x = static_cast<int>(index % dimensions[0]);
            const int y = static_cast<int>((index / dimensions[0]) % dimensions[1]);
            const int z = static_cast<int>(index /
                (static_cast<std::size_t>(dimensions[0]) * dimensions[1]));
            for (const auto &offset : kSixNeighbours)
            {
                const int nx = x + offset[0];
                const int ny = y + offset[1];
                const int nz = z + offset[2];
                if (!inside(dimensions, nx, ny, nz))
                {
                    continue;
                }
                const std::size_t next = gridIndex(dimensions, nx, ny, nz);
                if (occupied[next] != 0 && result.labels[next] < 0)
                {
                    result.labels[next] = result.count;
                    queue.push_back(static_cast<std::uint32_t>(next));
                }
            }
        }
        if (queue.size() > result.largestSize)
        {
            result.largestSize = queue.size();
            result.largestLabel = result.count;
        }
        ++result.count;
    }
    return result;
}

std::vector<std::uint8_t> largestComponent(
    const std::vector<std::uint8_t> &occupied,
    const ComponentSummary &components)
{
    std::vector<std::uint8_t> result(occupied.size(), 0);
    for (std::size_t index = 0; index < occupied.size(); ++index)
    {
        result[index] = occupied[index] != 0 &&
            components.labels[index] == components.largestLabel ? 1 : 0;
    }
    return result;
}

void calculateVertexNormals(TriMesh *mesh)
{
    for (MeshVertex &vertex : mesh->vertices)
    {
        vertex.nx = 0.0f;
        vertex.ny = 0.0f;
        vertex.nz = 0.0f;
    }
    for (const Triangle &face : mesh->faces)
    {
        const Vec3 normal = cross(
            position(mesh->vertices[face.v[1]]) -
                position(mesh->vertices[face.v[0]]),
            position(mesh->vertices[face.v[2]]) -
                position(mesh->vertices[face.v[0]]));
        for (const int vertex_index : face.v)
        {
            MeshVertex &vertex = mesh->vertices[vertex_index];
            vertex.nx += static_cast<float>(normal.x);
            vertex.ny += static_cast<float>(normal.y);
            vertex.nz += static_cast<float>(normal.z);
        }
    }
    for (MeshVertex &vertex : mesh->vertices)
    {
        const double length = std::sqrt(
            static_cast<double>(vertex.nx) * vertex.nx +
            static_cast<double>(vertex.ny) * vertex.ny +
            static_cast<double>(vertex.nz) * vertex.nz);
        if (length > 0.0)
        {
            vertex.nx = static_cast<float>(vertex.nx / length);
            vertex.ny = static_cast<float>(vertex.ny / length);
            vertex.nz = static_cast<float>(vertex.nz / length);
        }
    }
}

bool hasExactGenusZeroTopology(
    const MeshTopologyQualityStatistics &quality)
{
    return quality.componentCount == 1 &&
        quality.boundaryEdgeCount == 0 &&
        quality.nonManifoldEdgeCount == 0 &&
        quality.nonManifoldVertexCount == 0 &&
        quality.eulerCharacteristic == 2 &&
        quality.componentEulerCharacteristics == std::vector<int>{2};
}

std::vector<float> makeLocallyAveragedField(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const MeshVoxelTopologyRepairOptions &options,
    bool *wasCancelled)
{
    constexpr std::array<int, 3> weights{1, 2, 1};
    // The separable kernel has an integer total weight of 64. Keep the
    // threshold halfway between integer values so no grid sample lies exactly
    // on the MC33 iso-level; exact-zero samples otherwise produce vertices on
    // grid corners and can collapse adjacent triangles to zero area.
    constexpr float half_weight = 31.5f;
    std::vector<float> field(occupied.size(), 0.0f);
    for (int z = 0; z < dimensions[2]; ++z)
    {
        if ((z & 3) == 0 && cancelled(options))
        {
            *wasCancelled = true;
            return {};
        }
        for (int y = 0; y < dimensions[1]; ++y)
        {
            for (int x = 0; x < dimensions[0]; ++x)
            {
                int occupied_weight = 0;
                for (int dz = -1; dz <= 1; ++dz)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            const int nx = x + dx;
                            const int ny = y + dy;
                            const int nz = z + dz;
                            if (inside(dimensions, nx, ny, nz) &&
                                occupied[gridIndex(
                                    dimensions, nx, ny, nz)] != 0)
                            {
                                occupied_weight += weights[dx + 1] *
                                    weights[dy + 1] * weights[dz + 1];
                            }
                        }
                    }
                }
                field[gridIndex(dimensions, x, y, z)] =
                    half_weight - static_cast<float>(occupied_weight);
            }
        }
    }
    return field;
}

bool trySmoothExtraction(
    const GridLayout &layout,
    const std::vector<std::uint8_t> &occupied,
    const MeshVoxelTopologyRepairOptions &options,
    Candidate *candidate)
{
    candidate->smoothExtractionPreferred = options.preferSmoothExtraction;
    candidate->smoothExtractionAvailable =
        Mc33IsoSurfaceExtractor::isAvailable();
    if (!options.preferSmoothExtraction ||
        !candidate->smoothExtractionAvailable)
    {
        return false;
    }
    bool was_cancelled = false;
    std::vector<float> field = makeLocallyAveragedField(
        layout.dimensions, occupied, options, &was_cancelled);
    if (was_cancelled)
    {
        return false;
    }
    std::array<int, 3> cells{};
    for (int axis = 0; axis < 3; ++axis)
    {
        cells[axis] = layout.dimensions[axis] - 1;
    }
    Mc33IsoSurfaceOptions extraction_options;
    extraction_options.isoLevel = 0.0f;
    extraction_options.isCancelled = options.isCancelled;
    candidate->smoothExtractionAttempted = true;
    Mc33IsoSurfaceResult extraction = Mc33IsoSurfaceExtractor::extract(
        layout.boundsMin,
        layout.boundsMax,
        cells,
        field,
        {},
        extraction_options);
    candidate->smoothExtractionVertexCount =
        extraction.mesh.vertices.size();
    candidate->smoothExtractionFaceCount = extraction.mesh.faces.size();
    if (!extraction.ok || extraction.cancelled || extraction.mesh.empty())
    {
        candidate->smoothExtractionRejectedByTopology =
            extraction.ok && !extraction.cancelled;
        return false;
    }
    candidate->smoothQualityBeforeOptimization =
        evaluateMeshTopologyQuality(extraction.mesh);
    candidate->smoothQuality = candidate->smoothQualityBeforeOptimization;
    if (!hasExactGenusZeroTopology(candidate->smoothQuality))
    {
        candidate->smoothExtractionRejectedByTopology = true;
        return false;
    }
    if (!candidate->smoothQuality.strictGatePassed)
    {
        MeshTriangleOptimizationOptions optimization_options;
        // First repair MC33 diagonals without changing connectivity. A tightly
        // bounded tangential pass may then move a vertex by at most five per
        // cent of its local mean edge per pass; the exact topology and strict
        // triangle-quality gates below remain transactional.
        optimization_options.maximumPasses = 12;
        optimization_options.minimumWorstAspectImprovementRatio = 0.0;
        optimization_options.maximumFeatureAngleDegrees = 45.0;
        optimization_options.maximumNormalDeviationDegrees = 35.0;
        optimization_options.enableTangentialRelaxation = true;
        optimization_options.tangentialRelaxationPasses = 2;
        optimization_options.tangentialRelaxationLambda = 0.25;
        optimization_options.tangentialMaximumDisplacementEdgeRatio = 0.05;
        optimization_options.enableIsotropicRemeshing = false;
        optimization_options.isCancelled = options.isCancelled;
        candidate->smoothTriangleOptimizationAttempted = true;
        candidate->smoothOptimization = optimizeTriangleQuality(
            &extraction.mesh, optimization_options);
        candidate->smoothTriangleOptimizationCancelled =
            candidate->smoothOptimization.cancelled;
        if (candidate->smoothTriangleOptimizationCancelled)
        {
            return false;
        }
        candidate->smoothQuality =
            evaluateMeshTopologyQuality(extraction.mesh);
        if (!hasExactGenusZeroTopology(candidate->smoothQuality) ||
            !candidate->smoothQuality.strictGatePassed)
        {
            candidate->smoothExtractionRejectedByTopology = true;
            return false;
        }
        candidate->smoothTriangleOptimizationAccepted = true;
    }
    candidate->mesh = std::move(extraction.mesh);
    candidate->quality = candidate->smoothQuality;
    candidate->smoothExtractionAccepted = true;
    calculateVertexNormals(&candidate->mesh);
    return true;
}

Candidate makeCandidate(const GridLayout &layout,
                        const std::vector<std::uint8_t> &occupied,
                        int closingRadius,
                        std::uint64_t enclosedInteriorCount,
                        std::uint64_t discardedCount,
                        bool largestComponentFallback,
                        const MeshVoxelTopologyRepairOptions &options)
{
    Candidate candidate;
    candidate.closingRadius = closingRadius;
    candidate.enclosedInteriorCount = enclosedInteriorCount;
    candidate.occupiedCount = countOccupied(occupied);
    candidate.discardedCount = discardedCount;
    candidate.largestComponentFallback = largestComponentFallback;
    candidate.cubicalEuler =
        VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
            layout.dimensions, occupied);
    if (candidate.cubicalEuler != 1 ||
        (options.requireEnclosedInterior && enclosedInteriorCount == 0))
    {
        return candidate;
    }
    VisibilityOccupancyWellComposedRepairOptions well_composed_options;
    well_composed_options.maximumPasses = 12;
    well_composed_options.maximumFilledSampleCount = 8192;
    well_composed_options.isCancelled = options.isCancelled;
    const VisibilityOccupancyWellComposedRepairResult well_composed =
        VisibilityOccupancyWellComposedRepair::repair(
            layout.dimensions,
            occupied,
            std::vector<std::uint8_t>(occupied.size(), 0),
            well_composed_options);
    if (!well_composed.ok ||
        well_composed.statistics.bodyEulerAfter != 1)
    {
        return candidate;
    }
    candidate.occupiedCount = countOccupied(well_composed.occupied);

    if (trySmoothExtraction(
            layout, well_composed.occupied, options, &candidate))
    {
        candidate.valid = true;
        return candidate;
    }

    VisibilityOccupancyBoundaryOptions boundary_options;
    boundary_options.isCancelled = options.isCancelled;
    candidate.cellBoundaryExtractionUsed = true;
    auto boundary = VisibilityOccupancyBoundaryExtractor::extract(
        layout.boundsMin,
        layout.boundsMax,
        layout.dimensions,
        well_composed.occupied,
        boundary_options);
    if (!boundary.ok || boundary.cancelled)
    {
        return candidate;
    }
    candidate.quality = evaluateMeshTopologyQuality(boundary.mesh);
    if (!hasExactGenusZeroTopology(candidate.quality))
    {
        return candidate;
    }
    candidate.mesh = std::move(boundary.mesh);
    calculateVertexNormals(&candidate.mesh);
    candidate.valid = true;
    return candidate;
}

void assignCandidate(const Candidate &candidate,
                     MeshVoxelTopologyRepairResult *result)
{
    result->ok = true;
    result->mesh = candidate.mesh;
    result->statistics.enclosedInteriorCellCountAfter =
        candidate.enclosedInteriorCount;
    result->statistics.occupiedCellCountAfter = candidate.occupiedCount;
    result->statistics.discardedOccupiedCellCount = candidate.discardedCount;
    result->statistics.occupiedComponentCountAfter = 1;
    result->statistics.cubicalEulerAfter = candidate.cubicalEuler;
    result->statistics.selectedClosingRadius = candidate.closingRadius;
    result->statistics.outputVertexCount = result->mesh.vertices.size();
    result->statistics.outputFaceCount = result->mesh.faces.size();
    result->statistics.outputBoundaryEdgeCount =
        candidate.quality.boundaryEdgeCount;
    result->statistics.outputNonManifoldEdgeCount =
        candidate.quality.nonManifoldEdgeCount;
    result->statistics.outputNonManifoldVertexCount =
        candidate.quality.nonManifoldVertexCount;
    result->statistics.outputComponentCount = candidate.quality.componentCount;
    result->statistics.outputSurfaceEulerCharacteristic =
        candidate.quality.eulerCharacteristic;
    result->statistics.usedLargestComponentFallback =
        candidate.largestComponentFallback;
    result->statistics.smoothExtractionPreferred =
        candidate.smoothExtractionPreferred;
    result->statistics.smoothExtractionAvailable =
        candidate.smoothExtractionAvailable;
    result->statistics.smoothExtractionAttempted =
        candidate.smoothExtractionAttempted;
    result->statistics.smoothExtractionAccepted =
        candidate.smoothExtractionAccepted;
    result->statistics.smoothExtractionRejectedByTopology =
        candidate.smoothExtractionRejectedByTopology;
    result->statistics.cellBoundaryExtractionUsed =
        candidate.cellBoundaryExtractionUsed;
    result->statistics.smoothExtractionVertexCount =
        candidate.smoothExtractionVertexCount;
    result->statistics.smoothExtractionFaceCount =
        candidate.smoothExtractionFaceCount;
    result->statistics.smoothExtractionComponentCount =
        candidate.smoothQuality.componentCount;
    result->statistics.smoothExtractionBoundaryEdgeCount =
        candidate.smoothQuality.boundaryEdgeCount;
    result->statistics.smoothExtractionNonManifoldEdgeCount =
        candidate.smoothQuality.nonManifoldEdgeCount;
    result->statistics.smoothExtractionNonManifoldVertexCount =
        candidate.smoothQuality.nonManifoldVertexCount;
    result->statistics.smoothExtractionSurfaceEulerCharacteristic =
        candidate.smoothQuality.eulerCharacteristic;
    result->statistics.smoothExtractionHighAspectFaceRatio =
        candidate.smoothQuality.highAspectFaceRatio;
    result->statistics.smoothExtractionExtremeAspectFaceRatio =
        candidate.smoothQuality.extremeAspectFaceRatio;
    result->statistics.smoothExtractionStrictGatePassed =
        candidate.smoothQuality.strictGatePassed;
    result->statistics.smoothTriangleOptimizationAttempted =
        candidate.smoothTriangleOptimizationAttempted;
    result->statistics.smoothTriangleOptimizationCancelled =
        candidate.smoothTriangleOptimizationCancelled;
    result->statistics.smoothTriangleOptimizationAccepted =
        candidate.smoothTriangleOptimizationAccepted;
    result->statistics.smoothTriangleOptimizationPassCount =
        candidate.smoothOptimization.passCount;
    result->statistics.smoothTriangleOptimizationFlippedEdgeCount =
        candidate.smoothOptimization.flippedEdgeCount;
    result->statistics.smoothTriangleOptimizationTangentialPassCount =
        candidate.smoothOptimization.tangentialRelaxationPassCount;
    result->statistics.smoothTriangleOptimizationRelaxedVertexCount =
        candidate.smoothOptimization.tangentialRelaxedVertexCount;
    result->statistics
        .smoothExtractionHighAspectFaceRatioBeforeOptimization =
        candidate.smoothQualityBeforeOptimization.highAspectFaceRatio;
    result->statistics
        .smoothExtractionExtremeAspectFaceRatioBeforeOptimization =
        candidate.smoothQualityBeforeOptimization.extremeAspectFaceRatio;
}

} // namespace

MeshVoxelTopologyRepairResult MeshVoxelTopologyRepair::repair(
    const TriMesh &mesh,
    const MeshVoxelTopologyRepairOptions &options)
{
    MeshVoxelTopologyRepairResult result;
    result.statistics.requestedResolution = options.targetResolution;
    if (mesh.empty())
    {
        result.errorMessage = "mesh voxel topology repair input is empty";
        return result;
    }
    GridLayout layout;
    if (!makeLayout(mesh, options, &layout,
                    &result.statistics.effectiveResolution))
    {
        result.errorMessage = "mesh voxel topology repair bounds are invalid";
        return result;
    }
    result.statistics.gridDimensions = layout.dimensions;
    result.statistics.voxelSize = layout.spacing;
    std::vector<std::uint8_t> surface;
    if (!voxelize(mesh, layout, options, &surface, &result.statistics))
    {
        result.cancelled = cancelled(options);
        result.errorMessage = result.cancelled
            ? "mesh voxel topology repair cancelled during voxelization"
            : "mesh voxel topology repair found no valid triangle samples";
        return result;
    }

    bool base_fill_cancelled = false;
    const SolidFill base_fill = fillSolid(
        layout.dimensions, surface, options, &base_fill_cancelled);
    if (base_fill_cancelled)
    {
        result.cancelled = true;
        result.errorMessage =
            "mesh voxel topology repair cancelled during initial exterior fill";
        return result;
    }

    Candidate fallback;
    std::vector<int> attempted_eulers;
    std::vector<int> attempted_components;
    const int maximum_radius = std::clamp(options.maximumClosingRadius, 0, 8);
    for (int radius = 0; radius <= maximum_radius; ++radius)
    {
        bool was_cancelled = false;
        std::vector<std::uint8_t> body = closeGrid(
            layout.dimensions,
            base_fill.occupied,
            radius,
            options,
            &was_cancelled);
        if (was_cancelled)
        {
            result.cancelled = true;
            result.errorMessage =
                "mesh voxel topology repair cancelled during body closing";
            return result;
        }
        SolidFill sealed = fillSolid(
            layout.dimensions, body, options, &was_cancelled);
        if (was_cancelled)
        {
            result.cancelled = true;
            result.errorMessage =
                "mesh voxel topology repair cancelled during cavity fill";
            return result;
        }
        body = std::move(sealed.occupied);
        ComponentSummary summary = components(
            layout.dimensions, body, options, &was_cancelled);
        if (was_cancelled)
        {
            result.cancelled = true;
            result.errorMessage =
                "mesh voxel topology repair cancelled during component analysis";
            return result;
        }
        const std::uint64_t occupied_count = countOccupied(body);
        const int body_euler =
            VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
                layout.dimensions, body);
        attempted_eulers.push_back(body_euler);
        attempted_components.push_back(summary.count);
        if (radius == 0)
        {
            result.statistics.enclosedInteriorCellCountBefore =
                base_fill.enclosedInteriorCount;
            result.statistics.occupiedCellCountBefore = occupied_count;
            result.statistics.occupiedComponentCountBefore = summary.count;
            result.statistics.cubicalEulerBefore = body_euler;
        }
        if (summary.count == 1)
        {
            Candidate candidate = makeCandidate(
                layout, body, radius, base_fill.enclosedInteriorCount,
                0, false, options);
            if (candidate.valid)
            {
                assignCandidate(candidate, &result);
                return result;
            }
        }
        else if (options.allowLargestComponentFallback &&
                 summary.largestLabel >= 0)
        {
            std::vector<std::uint8_t> largest =
                largestComponent(body, summary);
            Candidate candidate = makeCandidate(
                layout,
                largest,
                radius,
                base_fill.enclosedInteriorCount,
                occupied_count - summary.largestSize,
                true,
                options);
            if (candidate.valid && !fallback.valid)
            {
                fallback = std::move(candidate);
            }
        }
    }
    if (fallback.valid)
    {
        assignCandidate(fallback, &result);
        return result;
    }
    std::ostringstream error;
    error << "mesh voxel topology repair could not produce a single closed "
             "genus-zero surface; attempted body Euler=[";
    for (std::size_t index = 0; index < attempted_eulers.size(); ++index)
    {
        if (index > 0)
        {
            error << ',';
        }
        error << attempted_eulers[index];
    }
    error << "]; components=[";
    for (std::size_t index = 0; index < attempted_components.size(); ++index)
    {
        if (index > 0)
        {
            error << ',';
        }
        error << attempted_components[index];
    }
    error << ']';
    result.errorMessage = error.str();
    return result;
}

} // namespace xjw::mesh
