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

bool cellHasSupportedSignChange(
    const std::array<int, 3> &cells,
    const std::vector<float> &field,
    const std::vector<std::uint8_t> &support,
    float isoLevel,
    int cellX,
    int cellY,
    int cellZ)
{
    bool has_inside = false;
    bool has_outside = false;
    for (int dz = 0; dz <= 1; ++dz)
    {
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const std::size_t index = sampleIndex(
                    cells, cellX + dx, cellY + dy, cellZ + dz);
                if (support[index] == 0u)
                {
                    continue;
                }
                has_inside = has_inside || field[index] <= isoLevel;
                has_outside = has_outside || field[index] > isoLevel;
            }
        }
    }
    return has_inside && has_outside;
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
            for (const Triangle &face : result.mesh.faces)
            {
                std::array<float, 3> centroid{};
                for (int corner = 0; corner < 3; ++corner)
                {
                    const MeshVertex &vertex = result.mesh.vertices[
                        static_cast<std::size_t>(face.v[corner])];
                    centroid[0] += vertex.x / 3.0f;
                    centroid[1] += vertex.y / 3.0f;
                    centroid[2] += vertex.z / 3.0f;
                }
                const int cell_x = std::clamp(
                    static_cast<int>(std::floor(
                        (centroid[0] - boundsMin[0]) / voxel_size[0])),
                    0,
                    cells[0] - 1);
                const int cell_y = std::clamp(
                    static_cast<int>(std::floor(
                        (centroid[1] - boundsMin[1]) / voxel_size[1])),
                    0,
                    cells[1] - 1);
                const int cell_z = std::clamp(
                    static_cast<int>(std::floor(
                        (centroid[2] - boundsMin[2]) / voxel_size[2])),
                    0,
                    cells[2] - 1);
                if (cellHasSupportedSignChange(
                        cells,
                        field,
                        extractionSupport,
                        options.isoLevel,
                        cell_x,
                        cell_y,
                        cell_z))
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
