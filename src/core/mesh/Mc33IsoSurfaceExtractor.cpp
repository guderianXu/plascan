#include "Mc33IsoSurfaceExtractor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if PLASCAN_HAS_MC33
#define mc33cpp_implementation
#include <mc33cpp.h>
#endif

namespace xjw::mesh
{
namespace
{

std::size_t checkedSampleCount(const std::array<int, 3> &cells)
{
    std::size_t count = 1u;
    for (int cell_count : cells)
    {
        if (cell_count < 1)
        {
            throw std::invalid_argument("MC33 grid cell counts must be positive");
        }
        const std::size_t samples = static_cast<std::size_t>(cell_count) + 1u;
        if (count > std::numeric_limits<std::size_t>::max() / samples)
        {
            throw std::overflow_error("MC33 grid sample count overflow");
        }
        count *= samples;
    }
    return count;
}

void validateBounds(const std::array<float, 3> &boundsMin,
                    const std::array<float, 3> &boundsMax)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(boundsMin[axis]) ||
            !std::isfinite(boundsMax[axis]) ||
            !(boundsMin[axis] < boundsMax[axis]))
        {
            throw std::invalid_argument("MC33 bounds must be finite and ordered");
        }
    }
}

std::size_t sampleIndex(const std::array<int, 3> &cells,
                        int x,
                        int y,
                        int z)
{
    const std::size_t samples_x =
        static_cast<std::size_t>(cells[0] + 1);
    const std::size_t samples_y =
        static_cast<std::size_t>(cells[1] + 1);
    return (static_cast<std::size_t>(z) * samples_y +
            static_cast<std::size_t>(y)) *
               samples_x +
        static_cast<std::size_t>(x);
}

bool isInsideIsoSurface(float value, float isoLevel)
{
    // Keep exact-iso samples on the non-negative side, matching the TSDF
    // connectivity code (negative: value < iso, positive: value >= iso).
    return value < isoLevel;
}

bool supportedGridEdgeHasSignChange(
    const std::array<int, 3> &cells,
    const std::vector<float> &field,
    const std::vector<std::uint8_t> &support,
    float isoLevel,
    const std::array<int, 3> &first,
    const std::array<int, 3> &second)
{
    const std::size_t first_index = sampleIndex(
        cells, first[0], first[1], first[2]);
    const std::size_t second_index = sampleIndex(
        cells, second[0], second[1], second[2]);
    return support[first_index] != 0u && support[second_index] != 0u &&
        isInsideIsoSurface(field[first_index], isoLevel) !=
            isInsideIsoSurface(field[second_index], isoLevel);
}

bool cellHasSupportedGridEdgeSignChange(
    const std::array<int, 3> &cells,
    const std::vector<float> &field,
    const std::vector<std::uint8_t> &support,
    float isoLevel,
    const std::array<int, 3> &cell)
{
    for (int axis = 0; axis < 3; ++axis)
    {
        const int first_orthogonal_axis = (axis + 1) % 3;
        const int second_orthogonal_axis = (axis + 2) % 3;
        for (int first_offset = 0; first_offset <= 1; ++first_offset)
        {
            for (int second_offset = 0; second_offset <= 1; ++second_offset)
            {
                std::array<int, 3> first = cell;
                first[first_orthogonal_axis] += first_offset;
                first[second_orthogonal_axis] += second_offset;
                std::array<int, 3> second = first;
                ++second[axis];
                if (supportedGridEdgeHasSignChange(
                        cells, field, support, isoLevel, first, second))
                {
                    return true;
                }
            }
        }
    }
    return false;
}

enum class VertexSupportKind
{
    Invalid,
    SupportedGridEdge,
    SupportedCellInterior
};

VertexSupportKind classifyVertexSupport(
    const MeshVertex &vertex,
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &voxelSize,
    const std::array<int, 3> &cells,
    const std::vector<float> &field,
    const std::vector<std::uint8_t> &support,
    float isoLevel)
{
    constexpr double coordinate_tolerance = 2.0e-4;
    std::array<double, 3> grid_coordinate{
        (static_cast<double>(vertex.x) - boundsMin[0]) / voxelSize[0],
        (static_cast<double>(vertex.y) - boundsMin[1]) / voxelSize[1],
        (static_cast<double>(vertex.z) - boundsMin[2]) / voxelSize[2]};
    std::array<int, 3> rounded_coordinate{};
    std::array<bool, 3> is_integer{};
    int integer_axis_count = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (grid_coordinate[axis] < -coordinate_tolerance ||
            grid_coordinate[axis] >
                static_cast<double>(cells[axis]) + coordinate_tolerance)
        {
            return VertexSupportKind::Invalid;
        }
        rounded_coordinate[axis] = static_cast<int>(
            std::llround(grid_coordinate[axis]));
        is_integer[axis] =
            std::fabs(grid_coordinate[axis] - rounded_coordinate[axis]) <=
            coordinate_tolerance;
        integer_axis_count += is_integer[axis] ? 1 : 0;
    }

    if (integer_axis_count == 2)
    {
        int varying_axis = 0;
        while (is_integer[varying_axis])
        {
            ++varying_axis;
        }
        std::array<int, 3> first = rounded_coordinate;
        first[varying_axis] = static_cast<int>(
            std::floor(grid_coordinate[varying_axis]));
        std::array<int, 3> second = first;
        ++second[varying_axis];
        if (first[varying_axis] < 0 ||
            second[varying_axis] > cells[varying_axis])
        {
            return VertexSupportKind::Invalid;
        }
        return supportedGridEdgeHasSignChange(
                   cells, field, support, isoLevel, first, second)
            ? VertexSupportKind::SupportedGridEdge
            : VertexSupportKind::Invalid;
    }

    if (integer_axis_count == 3)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            for (int direction : {-1, 1})
            {
                std::array<int, 3> neighbour = rounded_coordinate;
                neighbour[axis] += direction;
                if (neighbour[axis] < 0 || neighbour[axis] > cells[axis])
                {
                    continue;
                }
                if (supportedGridEdgeHasSignChange(
                        cells,
                        field,
                        support,
                        isoLevel,
                        rounded_coordinate,
                        neighbour))
                {
                    return VertexSupportKind::SupportedGridEdge;
                }
            }
        }
        return VertexSupportKind::Invalid;
    }

    // MC33 may introduce a cube-centre vertex for an ambiguous case. It is
    // not a crossing edge itself, so retain it only when its cube contains an
    // independently supported grid-edge crossing. Every retained triangle is
    // additionally required to contain two supported crossing-edge vertices.
    if (integer_axis_count == 0)
    {
        std::array<int, 3> cell{};
        for (int axis = 0; axis < 3; ++axis)
        {
            cell[axis] = static_cast<int>(std::floor(grid_coordinate[axis]));
            if (cell[axis] < 0 || cell[axis] >= cells[axis] ||
                std::fabs(grid_coordinate[axis] -
                          (static_cast<double>(cell[axis]) + 0.5)) >
                    coordinate_tolerance)
            {
                return VertexSupportKind::Invalid;
            }
        }
        return cellHasSupportedGridEdgeSignChange(
                   cells, field, support, isoLevel, cell)
            ? VertexSupportKind::SupportedCellInterior
            : VertexSupportKind::Invalid;
    }

    return VertexSupportKind::Invalid;
}

void compactReferencedVertices(TriMesh *mesh)
{
    std::vector<std::uint8_t> referenced(mesh->vertices.size(), 0u);
    for (const Triangle &face : mesh->faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            referenced[static_cast<std::size_t>(face.v[corner])] = 1u;
        }
    }

    std::vector<int> remap(mesh->vertices.size(), -1);
    std::vector<MeshVertex> vertices;
    vertices.reserve(mesh->vertices.size());
    for (std::size_t index = 0; index < mesh->vertices.size(); ++index)
    {
        if (referenced[index] == 0u)
        {
            continue;
        }
        remap[index] = static_cast<int>(vertices.size());
        vertices.push_back(mesh->vertices[index]);
    }
    for (Triangle &face : mesh->faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            face.v[corner] =
                remap[static_cast<std::size_t>(face.v[corner])];
        }
    }
    mesh->vertices = std::move(vertices);
}

} // namespace

bool Mc33IsoSurfaceExtractor::isAvailable()
{
#if PLASCAN_HAS_MC33
    return true;
#else
    return false;
#endif
}

Mc33IsoSurfaceResult Mc33IsoSurfaceExtractor::extract(
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::array<int, 3> &cells,
    const std::vector<float> &field,
    const std::vector<std::uint8_t> &extractionSupport,
    const Mc33IsoSurfaceOptions &options)
{
    Mc33IsoSurfaceResult result;
#if !PLASCAN_HAS_MC33
    static_cast<void>(boundsMin);
    static_cast<void>(boundsMax);
    static_cast<void>(cells);
    static_cast<void>(field);
    static_cast<void>(extractionSupport);
    static_cast<void>(options);
    result.errorMessage =
        "MC33 support is unavailable; configure PLASCAN_MC33_INCLUDE_DIR";
    return result;
#else
    try
    {
        validateBounds(boundsMin, boundsMax);
        const std::size_t expected_samples = checkedSampleCount(cells);
        if (field.size() != expected_samples)
        {
            throw std::invalid_argument("MC33 field size does not match grid");
        }
        if (!extractionSupport.empty() &&
            extractionSupport.size() != expected_samples)
        {
            throw std::invalid_argument(
                "MC33 extraction support size does not match grid");
        }
        if (!std::isfinite(options.isoLevel))
        {
            throw std::invalid_argument("MC33 iso-surface level must be finite");
        }
        if (options.isCancelled && options.isCancelled())
        {
            result.cancelled = true;
            result.errorMessage = "MC33 extraction cancelled";
            return result;
        }

        std::vector<float> extraction_field(field);
        result.statistics.inputSampleCount = extraction_field.size();
        const float outside_value = options.isoLevel + 1.0f;
        for (std::size_t index = 0; index < extraction_field.size(); ++index)
        {
            if (!std::isfinite(extraction_field[index]))
            {
                throw std::invalid_argument("MC33 field samples must be finite");
            }
            if (extraction_field[index] == options.isoLevel)
            {
                extraction_field[index] = std::nextafter(
                    options.isoLevel,
                    std::numeric_limits<float>::infinity());
            }
            if (!extractionSupport.empty() && extractionSupport[index] == 0u)
            {
                extraction_field[index] = outside_value;
                ++result.statistics.supportMaskedSampleCount;
            }
        }

        grid3d grid;
        const int grid_status = grid.set_data_pointer(
            static_cast<unsigned int>(cells[0] + 1),
            static_cast<unsigned int>(cells[1] + 1),
            static_cast<unsigned int>(cells[2] + 1),
            extraction_field.data());
        if (grid_status != 0)
        {
            throw std::runtime_error(
                "MC33 failed to bind scalar grid (status " +
                std::to_string(grid_status) + ")");
        }
        grid.set_ratio_aspect(
            (boundsMax[0] - boundsMin[0]) / static_cast<double>(cells[0]),
            (boundsMax[1] - boundsMin[1]) / static_cast<double>(cells[1]),
            (boundsMax[2] - boundsMin[2]) / static_cast<double>(cells[2]));
        grid.set_r0(boundsMin[0], boundsMin[1], boundsMin[2]);

        MC33 extractor;
        const int setup_status = extractor.set_grid3d(grid);
        if (setup_status != 0)
        {
            throw std::runtime_error(
                "MC33 failed to initialize extractor (status " +
                std::to_string(setup_status) + ")");
        }
        surface surface_mesh;
        const int extraction_status =
            extractor.calculate_isosurface(surface_mesh, options.isoLevel);
        if (extraction_status != 0)
        {
            throw std::runtime_error(
                "MC33 failed to calculate iso-surface (status " +
                std::to_string(extraction_status) + ")");
        }
        if (options.isCancelled && options.isCancelled())
        {
            result.cancelled = true;
            result.errorMessage = "MC33 extraction cancelled";
            return result;
        }

        const unsigned int vertex_count = surface_mesh.get_num_vertices();
        const unsigned int face_count = surface_mesh.get_num_triangles();
        result.mesh.vertices.resize(vertex_count);
        result.mesh.faces.resize(face_count);
        for (unsigned int index = 0; index < vertex_count; ++index)
        {
            const MC33_real *source = surface_mesh.getVertex(index);
            MeshVertex &target = result.mesh.vertices[index];
            target.x = static_cast<float>(source[0]);
            target.y = static_cast<float>(source[1]);
            target.z = static_cast<float>(source[2]);
        }
        for (unsigned int index = 0; index < face_count; ++index)
        {
            const unsigned int *source = surface_mesh.getTriangle(index);
            Triangle &target = result.mesh.faces[index];
            target.v[0] = static_cast<int>(source[0]);
            target.v[1] = static_cast<int>(source[1]);
            target.v[2] = static_cast<int>(source[2]);
        }
        if (options.requireSupportedSignChange &&
            !extractionSupport.empty())
        {
            const std::array<float, 3> voxel_size{
                (boundsMax[0] - boundsMin[0]) /
                    static_cast<float>(cells[0]),
                (boundsMax[1] - boundsMin[1]) /
                    static_cast<float>(cells[1]),
                (boundsMax[2] - boundsMin[2]) /
                    static_cast<float>(cells[2])};
            std::vector<Triangle> retained_faces;
            retained_faces.reserve(result.mesh.faces.size());
            std::vector<VertexSupportKind> vertex_support(
                result.mesh.vertices.size(), VertexSupportKind::Invalid);
            for (std::size_t index = 0; index < result.mesh.vertices.size(); ++index)
            {
                vertex_support[index] = classifyVertexSupport(
                    result.mesh.vertices[index],
                    boundsMin,
                    voxel_size,
                    cells,
                    field,
                    extractionSupport,
                    options.isoLevel);
            }
            for (const Triangle &face : result.mesh.faces)
            {
                bool valid = true;
                int supported_edge_vertex_count = 0;
                for (int corner = 0; corner < 3; ++corner)
                {
                    const VertexSupportKind support_kind = vertex_support[
                        static_cast<std::size_t>(face.v[corner])];
                    valid = valid && support_kind != VertexSupportKind::Invalid;
                    supported_edge_vertex_count +=
                        support_kind == VertexSupportKind::SupportedGridEdge
                        ? 1
                        : 0;
                }
                if (valid && supported_edge_vertex_count >= 2)
                {
                    retained_faces.push_back(face);
                }
                else
                {
                    ++result.statistics.rejectedUnsupportedCellFaceCount;
                }
            }
            result.mesh.faces = std::move(retained_faces);
            compactReferencedVertices(&result.mesh);
        }
        result.statistics.outputVertexCount = result.mesh.vertices.size();
        result.statistics.outputFaceCount = result.mesh.faces.size();
        result.ok = true;
    }
    catch (const std::exception &exception)
    {
        result.errorMessage = exception.what();
    }
    return result;
#endif
}

} // namespace xjw::mesh
