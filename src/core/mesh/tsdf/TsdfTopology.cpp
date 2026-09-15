#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    MeshBoundaryTopology boundaryTopology(const TriMesh& mesh)
    {
        const auto edge_key = [](int first, int second)
        {
            const auto low = static_cast<std::uint32_t>(std::min(first, second));
            const auto high = static_cast<std::uint32_t>(std::max(first, second));
            return (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
        };
        std::unordered_map<std::uint64_t, int> counts;
        counts.reserve(mesh.faces.size() * 3);
        for (const Triangle& face : mesh.faces)
        {
            ++counts[edge_key(face.v[0], face.v[1])];
            ++counts[edge_key(face.v[1], face.v[2])];
            ++counts[edge_key(face.v[2], face.v[0])];
        }
        MeshBoundaryTopology topology;
        std::vector<int> boundary_degree(mesh.vertices.size(), 0);
        for (const auto& entry : counts)
        {
            if (entry.second == 1)
            {
                ++topology.boundaryEdgeCount;
                const int first = static_cast<int>(entry.first >> 32U);
                const int second = static_cast<int>(entry.first & 0xffffffffU);
                ++boundary_degree[static_cast<std::size_t>(first)];
                ++boundary_degree[static_cast<std::size_t>(second)];
            }
            else if (entry.second > 2)
            {
                ++topology.nonManifoldEdgeCount;
            }
        }
        topology.danglingBoundaryVertexCount =
            static_cast<int>(std::count(boundary_degree.cbegin(), boundary_degree.cend(), 1));
        return topology;
    }

    int boundaryEdgeCount(const TriMesh& mesh)
    {
        return boundaryTopology(mesh).boundaryEdgeCount;
    }

    double meshSurfaceArea(const TriMesh& mesh)
    {
        double area = 0.0;
        for (const Triangle& face : mesh.faces)
        {
            if (face.v[0] < 0 || face.v[1] < 0 || face.v[2] < 0 || face.v[0] >= mesh.vertexCount() ||
                face.v[1] >= mesh.vertexCount() || face.v[2] >= mesh.vertexCount())
            {
                continue;
            }
            const MeshVertex& first = mesh.vertices[static_cast<std::size_t>(face.v[0])];
            const MeshVertex& second = mesh.vertices[static_cast<std::size_t>(face.v[1])];
            const MeshVertex& third = mesh.vertices[static_cast<std::size_t>(face.v[2])];
            const double ab_x = static_cast<double>(second.x) - first.x;
            const double ab_y = static_cast<double>(second.y) - first.y;
            const double ab_z = static_cast<double>(second.z) - first.z;
            const double ac_x = static_cast<double>(third.x) - first.x;
            const double ac_y = static_cast<double>(third.y) - first.y;
            const double ac_z = static_cast<double>(third.z) - first.z;
            const double cross_x = ab_y * ac_z - ab_z * ac_y;
            const double cross_y = ab_z * ac_x - ab_x * ac_z;
            const double cross_z = ab_x * ac_y - ab_y * ac_x;
            const double twice_area = std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
            if (std::isfinite(twice_area))
            {
                area += 0.5 * twice_area;
            }
        }
        return area;
    }

    double meshBoundsDiagonal(const TriMesh& mesh)
    {
        std::array<double, 3> minimum{std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()};
        std::array<double, 3> maximum{-std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity(),
                                      -std::numeric_limits<double>::infinity()};
        bool has_finite_vertex = false;
        for (const MeshVertex& vertex : mesh.vertices)
        {
            const std::array<double, 3> position{vertex.x, vertex.y, vertex.z};
            if (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2]))
            {
                continue;
            }
            has_finite_vertex = true;
            for (int axis = 0; axis < 3; ++axis)
            {
                minimum[axis] = std::min(minimum[axis], position[axis]);
                maximum[axis] = std::max(maximum[axis], position[axis]);
            }
        }
        if (!has_finite_vertex)
        {
            return 0.0;
        }
        const double dx = maximum[0] - minimum[0];
        const double dy = maximum[1] - minimum[1];
        const double dz = maximum[2] - minimum[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    bool meshVerticesInsidePaddedLayout(const TriMesh& mesh, const DepthTsdfLayout& layout, float paddingVoxels)
    {
        const float maximum_voxel_size = std::max({layout.voxelSize[0], layout.voxelSize[1], layout.voxelSize[2]});
        const float padding = std::max(0.0f, paddingVoxels) * maximum_voxel_size;
        for (const MeshVertex& vertex : mesh.vertices)
        {
            const std::array<float, 3> position{{vertex.x, vertex.y, vertex.z}};
            for (int axis = 0; axis < 3; ++axis)
            {
                const float value = position[static_cast<std::size_t>(axis)];
                if (!std::isfinite(value) || value < layout.boundsMin[axis] - padding ||
                    value > layout.boundsMax[axis] + padding)
                {
                    return false;
                }
            }
        }
        return true;
    }

    TriangleQualitySummary triangleQualitySummary(const TriMesh& mesh)
    {
        constexpr double sliver_quality_threshold = 0.05;
        constexpr double normalized_area_factor = 3.4641016151377544;
        TriangleQualitySummary summary;
        for (const Triangle& face : mesh.faces)
        {
            if (face.v[0] < 0 || face.v[1] < 0 || face.v[2] < 0 ||
                static_cast<std::size_t>(face.v[0]) >= mesh.vertices.size() ||
                static_cast<std::size_t>(face.v[1]) >= mesh.vertices.size() ||
                static_cast<std::size_t>(face.v[2]) >= mesh.vertices.size())
            {
                continue;
            }
            const MeshVertex& a = mesh.vertices[static_cast<std::size_t>(face.v[0])];
            const MeshVertex& b = mesh.vertices[static_cast<std::size_t>(face.v[1])];
            const MeshVertex& c = mesh.vertices[static_cast<std::size_t>(face.v[2])];
            const double ab_x = b.x - a.x;
            const double ab_y = b.y - a.y;
            const double ab_z = b.z - a.z;
            const double ac_x = c.x - a.x;
            const double ac_y = c.y - a.y;
            const double ac_z = c.z - a.z;
            const double bc_x = c.x - b.x;
            const double bc_y = c.y - b.y;
            const double bc_z = c.z - b.z;
            const double cross_x = ab_y * ac_z - ab_z * ac_y;
            const double cross_y = ab_z * ac_x - ab_x * ac_z;
            const double cross_z = ab_x * ac_y - ab_y * ac_x;
            const double doubled_area = std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
            const double squared_edge_sum = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z + ac_x * ac_x + ac_y * ac_y +
                                            ac_z * ac_z + bc_x * bc_x + bc_y * bc_y + bc_z * bc_z;
            const double quality =
                squared_edge_sum > 1.0e-20 ? normalized_area_factor * doubled_area / squared_edge_sum : 0.0;
            ++summary.validFaceCount;
            if (!std::isfinite(quality) || quality < sliver_quality_threshold)
            {
                ++summary.sliverFaceCount;
            }
        }
        if (summary.validFaceCount > 0)
        {
            summary.sliverRatio = static_cast<double>(summary.sliverFaceCount) / summary.validFaceCount;
        }
        return summary;
    }

    std::vector<std::uint8_t> boundaryVertexMask(const TriMesh& mesh)
    {
        const auto edge_key = [](int first, int second)
        {
            const auto low = static_cast<std::uint32_t>(std::min(first, second));
            const auto high = static_cast<std::uint32_t>(std::max(first, second));
            return (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
        };
        std::unordered_map<std::uint64_t, int> edge_counts;
        edge_counts.reserve(mesh.faces.size() * 3);
        for (const Triangle& face : mesh.faces)
        {
            ++edge_counts[edge_key(face.v[0], face.v[1])];
            ++edge_counts[edge_key(face.v[1], face.v[2])];
            ++edge_counts[edge_key(face.v[2], face.v[0])];
        }
        std::vector<std::uint8_t> boundary_vertices(mesh.vertices.size(), 0);
        for (const auto& [key, count] : edge_counts)
        {
            if (count != 1)
            {
                continue;
            }
            boundary_vertices[static_cast<std::size_t>(key >> 32U)] = 1;
            boundary_vertices[static_cast<std::size_t>(key & 0xffffffffU)] = 1;
        }
        return boundary_vertices;
    }

    std::vector<std::uint8_t> multiViewSilhouetteBoundaryVertices(const TriMesh& mesh,
                                                                  const QVector<DepthTsdfFrame>& frames,
                                                                  const QVector<cv::Mat>& effectiveDepthValidMasks,
                                                                  int minimumViews,
                                                                  int bandPixels,
                                                                  float depthToleranceVoxels,
                                                                  float maximumVoxelSize,
                                                                  float minimumConfidence)
    {
        std::vector<std::uint8_t> protected_vertices(mesh.vertices.size(), 0);
        if (mesh.empty() || frames.empty())
        {
            return protected_vertices;
        }

        const std::vector<std::uint8_t> boundary_vertices = boundaryVertexMask(mesh);
        QVector<cv::Mat> silhouette_bands;
        silhouette_bands.reserve(frames.size());
        const int radius = std::clamp(bandPixels, 1, 8);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            const DepthTsdfFrame& frame = frames[frame_index];
            if (frame.auxiliarySurfaceOnly)
            {
                silhouette_bands.push_back(cv::Mat());
                continue;
            }
            cv::Mat band;
            cv::morphologyEx(frame.supportMask, band, cv::MORPH_GRADIENT, kernel);
            cv::bitwise_and(band, frame.supportMask, band);
            const cv::Mat& valid_mask = effectiveDepthValidMasks.size() == frames.size()
                                            ? effectiveDepthValidMasks[frame_index]
                                            : frame.depthValidMask;
            cv::bitwise_and(band, valid_mask, band);
            silhouette_bands.push_back(std::move(band));
        }

        const int required_views = std::max(1, minimumViews);
        const float absolute_tolerance = std::max(1.0f, depthToleranceVoxels) * std::max(maximumVoxelSize, 1.0e-8f);
        for (std::size_t vertex_index = 0; vertex_index < mesh.vertices.size(); ++vertex_index)
        {
            if (boundary_vertices[vertex_index] == 0)
            {
                continue;
            }
            const MeshVertex& vertex = mesh.vertices[vertex_index];
            const double world[3] = {vertex.x, vertex.y, vertex.z};
            int agreeing_views = 0;
            for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
            {
                const DepthTsdfFrame& frame = frames[frame_index];
                if (frame.auxiliarySurfaceOnly)
                {
                    continue;
                }
                double pixel[2]{};
                double camera_depth = 0.0;
                if (!frame.camera.projectWorldPointWithDepth(world, pixel, camera_depth))
                {
                    continue;
                }
                const int column = static_cast<int>(std::lround(pixel[0]));
                const int row = static_cast<int>(std::lround(pixel[1]));
                if (row < 0 || column < 0 || row >= silhouette_bands[frame_index].rows ||
                    column >= silhouette_bands[frame_index].cols ||
                    silhouette_bands[frame_index].at<std::uint8_t>(row, column) == 0)
                {
                    continue;
                }
                const float observed_depth = frame.depth.at<float>(row, column);
                const float confidence = frame.confidence.at<float>(row, column);
                const float tolerance =
                    std::max(absolute_tolerance, 0.008f * std::fabs(static_cast<float>(camera_depth)));
                if (!std::isfinite(observed_depth) || observed_depth <= 0.0f || !std::isfinite(confidence) ||
                    confidence < minimumConfidence ||
                    std::fabs(observed_depth - static_cast<float>(camera_depth)) > tolerance)
                {
                    continue;
                }
                if (++agreeing_views >= required_views)
                {
                    protected_vertices[vertex_index] = 1;
                    break;
                }
            }
        }
        return protected_vertices;
    }

    std::pair<int, int> holeFillPatchSampleViewEvidence(const MeshVertex& sample,
                                                        const QVector<DepthTsdfFrame>& frames,
                                                        const QVector<cv::Mat>& effectiveDepthValidMasks,
                                                        float absoluteDepthTolerance,
                                                        float minimumConfidence)
    {
        int supporting_views = 0;
        int conflict_views = 0;
        const double world[3] = {sample.x, sample.y, sample.z};
        for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
        {
            const DepthTsdfFrame& frame = frames[frame_index];
            if (frame.auxiliarySurfaceOnly)
            {
                continue;
            }
            if (frame.depth.empty() || frame.depth.type() != CV_32FC1 || frame.confidence.empty() ||
                frame.confidence.type() != CV_32FC1 || frame.supportMask.empty() ||
                frame.supportMask.type() != CV_8UC1 || frame.depth.size() != frame.confidence.size() ||
                frame.depth.size() != frame.supportMask.size())
            {
                continue;
            }
            const cv::Mat& valid_mask = effectiveDepthValidMasks.size() == frames.size()
                                            ? effectiveDepthValidMasks[frame_index]
                                            : frame.depthValidMask;
            if (valid_mask.empty() || valid_mask.type() != CV_8UC1 || valid_mask.size() != frame.depth.size())
            {
                continue;
            }

            double pixel[2]{};
            double camera_depth = 0.0;
            if (!frame.camera.projectWorldPointWithDepth(world, pixel, camera_depth) || !std::isfinite(camera_depth) ||
                camera_depth <= 0.0)
            {
                continue;
            }
            const int column = static_cast<int>(std::lround(pixel[0]));
            const int row = static_cast<int>(std::lround(pixel[1]));
            if (row < 0 || column < 0 || row >= frame.depth.rows || column >= frame.depth.cols)
            {
                continue;
            }
            if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
            {
                ++conflict_views;
                continue;
            }
            if (valid_mask.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }

            const float observed_depth = frame.depth.at<float>(row, column);
            const float confidence = frame.confidence.at<float>(row, column);
            if (!std::isfinite(observed_depth) || observed_depth <= 0.0f || !std::isfinite(confidence) ||
                confidence < minimumConfidence)
            {
                continue;
            }
            const float projected_depth = static_cast<float>(camera_depth);
            const float tolerance = std::max(absoluteDepthTolerance, 0.008f * std::abs(projected_depth));
            if (std::abs(observed_depth - projected_depth) <= tolerance)
            {
                ++supporting_views;
            }
            else if (projected_depth < observed_depth - tolerance)
            {
                // The proposed patch lies in front of an observed surface and
                // therefore occupies free space in this view.  A sample behind
                // the observed surface is merely occluded and is not a conflict.
                ++conflict_views;
            }
        }
        return {supporting_views, conflict_views};
    }
} // namespace xjw::mesh::tsdf_detail
