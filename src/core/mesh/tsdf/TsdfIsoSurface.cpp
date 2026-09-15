#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    ComparableIsoSurfaceExtraction extractComparableIsoSurface(const std::array<float, 3>& boundsMin,
                                                               const std::array<float, 3>& boundsMax,
                                                               const std::array<int, 3>& cells,
                                                               const std::vector<float>& field,
                                                               const std::vector<std::uint8_t>& support,
                                                               bool useMc33,
                                                               bool requireSupportedSignChange,
                                                               bool useConsistentExtractor,
                                                               const std::function<bool()>& isCancelled)
    {
        ComparableIsoSurfaceExtraction result;
        if (useMc33)
        {
            Mc33IsoSurfaceOptions options;
            options.isoLevel = 0.0f;
            options.requireSupportedSignChange = requireSupportedSignChange;
            options.isCancelled = isCancelled;
            Mc33IsoSurfaceResult extracted =
                Mc33IsoSurfaceExtractor::extract(boundsMin, boundsMax, cells, field, support, options);
            result.ok = extracted.ok;
            result.errorMessage = QString::fromStdString(extracted.errorMessage);
            result.mesh = std::move(extracted.mesh);
            result.mc33SupportMaskedSampleCount = extracted.statistics.supportMaskedSampleCount;
            result.mc33RejectedUnsupportedCellFaceCount = extracted.statistics.rejectedUnsupportedCellFaceCount;
            return result;
        }
        if (useConsistentExtractor)
        {
            ConsistentIsoSurfaceOptions options;
            options.isoLevel = 0.0f;
            options.isCancelled = isCancelled;
            ConsistentIsoSurfaceResult extracted =
                ConsistentIsoSurfaceExtractor::extract(boundsMin, boundsMax, cells, field, support, options);
            result.ok = extracted.ok;
            result.errorMessage = QString::fromStdString(extracted.errorMessage);
            result.mesh = std::move(extracted.mesh);
            result.isoSurfaceAmbiguousFaceCount = extracted.statistics.uniqueAmbiguousFaceCount;
            result.isoSurfaceTopologyAdjustedCellCount = extracted.statistics.topologyAdjustedCellCount;
            result.isoSurfaceDeciderTieCount = extracted.statistics.deciderTieCount;
            result.isoSurfaceMultipleLoopCellCount = extracted.statistics.multipleLoopCellCount;
            result.isoSurfaceEdgeVertexCacheHitCount = extracted.statistics.edgeVertexCacheHitCount;
            result.isoSurfaceEdgeVertexCacheMissCount = extracted.statistics.edgeVertexCacheMissCount;
            result.isoSurfaceInteriorLoopVertexCount = extracted.statistics.interiorLoopVertexCount;
            result.isoSurfaceRejectedDegenerateFaceCount = extracted.statistics.rejectedDegenerateFaceCount;
            result.isoSurfaceUnresolvedCellCount = extracted.statistics.unresolvedCellCount;
            return result;
        }

        plapoint::mesh::MarchingCubes<float> marching_cubes;
        marching_cubes.setBounds({boundsMin[0], boundsMin[1], boundsMin[2]},
                                 {boundsMax[0], boundsMax[1], boundsMax[2]});
        marching_cubes.setResolution(cells[0], cells[1], cells[2]);
        marching_cubes.setIsoLevel(0.0f);
        auto [vertices, faces] = marching_cubes.extract(
            [&](float x, float y, float z)
            {
                const int ix = std::clamp(
                    static_cast<int>(std::lround((x - boundsMin[0]) * cells[0] / (boundsMax[0] - boundsMin[0]))),
                    0,
                    cells[0]);
                const int iy = std::clamp(
                    static_cast<int>(std::lround((y - boundsMin[1]) * cells[1] / (boundsMax[1] - boundsMin[1]))),
                    0,
                    cells[1]);
                const int iz = std::clamp(
                    static_cast<int>(std::lround((z - boundsMin[2]) * cells[2] / (boundsMax[2] - boundsMin[2]))),
                    0,
                    cells[2]);
                const std::size_t index = (static_cast<std::size_t>(iz) * static_cast<std::size_t>(cells[1] + 1) +
                                           static_cast<std::size_t>(iy)) *
                                              static_cast<std::size_t>(cells[0] + 1) +
                                          static_cast<std::size_t>(ix);
                return support[index] != 0 ? field[index] : 1.0f;
            });
        result.mesh.vertices.resize(static_cast<std::size_t>(vertices.rows()));
        for (plamatrix::Index row = 0; row < vertices.rows(); ++row)
        {
            MeshVertex& vertex = result.mesh.vertices[static_cast<std::size_t>(row)];
            vertex.x = vertices(row, 0);
            vertex.y = vertices(row, 1);
            vertex.z = vertices(row, 2);
        }
        result.mesh.faces.resize(static_cast<std::size_t>(faces.rows()));
        for (plamatrix::Index row = 0; row < faces.rows(); ++row)
        {
            Triangle& face = result.mesh.faces[static_cast<std::size_t>(row)];
            face.v[0] = static_cast<int>(std::lround(faces(row, 0)));
            face.v[1] = static_cast<int>(std::lround(faces(row, 1)));
            face.v[2] = static_cast<int>(std::lround(faces(row, 2)));
        }
        result.ok = true;
        return result;
    }
} // namespace xjw::mesh::tsdf_detail
