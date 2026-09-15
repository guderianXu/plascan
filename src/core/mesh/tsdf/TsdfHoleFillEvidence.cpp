#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    HoleFillPatchEvidenceResult validateAddedHoleFillPatchEvidence(const TriMesh& candidate,
                                                                   std::size_t baselineVertexCount,
                                                                   std::size_t baselineFaceCount,
                                                                   const QVector<DepthTsdfFrame>& frames,
                                                                   const QVector<cv::Mat>& effectiveDepthValidMasks,
                                                                   int minimumSupportingViews,
                                                                   int maximumConflictViews,
                                                                   float depthToleranceVoxels,
                                                                   float maximumVoxelSize,
                                                                   float minimumConfidence)
    {
        HoleFillPatchEvidenceResult result;
        if (baselineVertexCount > candidate.vertices.size() || baselineFaceCount > candidate.faces.size())
        {
            return result;
        }

        result.vertexSampleCount = static_cast<int>(candidate.vertices.size() - baselineVertexCount);
        result.faceCenterSampleCount = static_cast<int>(candidate.faces.size() - baselineFaceCount);
        const int sample_count = result.vertexSampleCount + result.faceCenterSampleCount;
        result.attempted = sample_count > 0;
        if (!result.attempted)
        {
            return result;
        }

        const float absolute_tolerance = std::max(1.0f, depthToleranceVoxels) * std::max(maximumVoxelSize, 1.0e-8f);
        result.minimumSupportingViewCount = std::numeric_limits<int>::max();
        const auto validate_sample = [&](const MeshVertex& sample)
        {
            const auto [supporting_views, conflict_views] = holeFillPatchSampleViewEvidence(
                sample, frames, effectiveDepthValidMasks, absolute_tolerance, minimumConfidence);
            result.minimumSupportingViewCount = std::min(result.minimumSupportingViewCount, supporting_views);
            result.maximumConflictViewCount = std::max(result.maximumConflictViewCount, conflict_views);
            if (supporting_views < std::max(2, minimumSupportingViews))
            {
                ++result.rejectedSupportSampleCount;
            }
            if (conflict_views > std::max(0, maximumConflictViews))
            {
                ++result.rejectedConflictSampleCount;
            }
            if (DepthTsdfSurfaceBuilder::shouldAcceptFinalHoleFillPatchSample(
                    supporting_views, conflict_views, minimumSupportingViews, maximumConflictViews))
            {
                ++result.acceptedSampleCount;
            }
        };

        for (std::size_t vertex_index = baselineVertexCount; vertex_index < candidate.vertices.size(); ++vertex_index)
        {
            validate_sample(candidate.vertices[vertex_index]);
        }
        for (std::size_t face_index = baselineFaceCount; face_index < candidate.faces.size(); ++face_index)
        {
            const Triangle& face = candidate.faces[face_index];
            const MeshVertex& first = candidate.vertices[static_cast<std::size_t>(face.v[0])];
            const MeshVertex& second = candidate.vertices[static_cast<std::size_t>(face.v[1])];
            const MeshVertex& third = candidate.vertices[static_cast<std::size_t>(face.v[2])];
            MeshVertex center;
            center.x = (first.x + second.x + third.x) / 3.0f;
            center.y = (first.y + second.y + third.y) / 3.0f;
            center.z = (first.z + second.z + third.z) / 3.0f;
            validate_sample(center);
        }

        result.accepted = result.acceptedSampleCount == sample_count;
        if (result.minimumSupportingViewCount == std::numeric_limits<int>::max())
        {
            result.minimumSupportingViewCount = 0;
        }
        return result;
    }

    VisibilityHoleProtectionResult
    visibilityConstrainedHoleProtection(const TriMesh& mesh,
                                        const QVector<DepthTsdfFrame>& frames,
                                        const QVector<cv::Mat>& effectiveDepthValidMasks,
                                        const std::vector<std::uint8_t>& silhouetteProtectedVertices,
                                        int maximumBoundaryEdges,
                                        int minimumSupportingViews,
                                        int maximumConflictViews,
                                        float depthToleranceVoxels,
                                        float strongSilhouetteRatio,
                                        float maximumVoxelSize,
                                        float minimumConfidence)
    {
        VisibilityHoleProtectionResult result;
        result.protectedVertices = silhouetteProtectedVertices;
        if (mesh.empty() || frames.empty() || silhouetteProtectedVertices.size() != mesh.vertices.size())
        {
            return result;
        }

        std::unordered_map<std::uint64_t, int> edge_counts;
        edge_counts.reserve(mesh.faces.size() * 3);
        const auto add_edge = [&edge_counts](int first, int second)
        {
            const auto low = static_cast<std::uint32_t>(std::min(first, second));
            const auto high = static_cast<std::uint32_t>(std::max(first, second));
            const std::uint64_t key = (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
            ++edge_counts[key];
            return key;
        };
        for (const Triangle& face : mesh.faces)
        {
            add_edge(face.v[0], face.v[1]);
            add_edge(face.v[1], face.v[2]);
            add_edge(face.v[2], face.v[0]);
        }
        const auto edge_key = [](int first, int second)
        {
            const auto low = static_cast<std::uint32_t>(std::min(first, second));
            const auto high = static_cast<std::uint32_t>(std::max(first, second));
            return (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
        };
        std::vector<std::vector<int>> boundary_neighbors(mesh.vertices.size());
        for (const Triangle& face : mesh.faces)
        {
            const std::array<std::array<int, 2>, 3> edges{
                {{{face.v[0], face.v[1]}}, {{face.v[1], face.v[2]}}, {{face.v[2], face.v[0]}}}};
            for (const auto& edge : edges)
            {
                if (edge_counts[edge_key(edge[0], edge[1])] != 1)
                {
                    continue;
                }
                boundary_neighbors[static_cast<std::size_t>(edge[0])].push_back(edge[1]);
                boundary_neighbors[static_cast<std::size_t>(edge[1])].push_back(edge[0]);
            }
        }

        std::unordered_set<std::uint64_t> visited_edges;
        visited_edges.reserve(edge_counts.size());
        const int required_supporting_views = std::max(1, minimumSupportingViews);
        const int allowed_conflict_views = std::max(0, maximumConflictViews);
        const float absolute_tolerance = std::max(1.0f, depthToleranceVoxels) * std::max(maximumVoxelSize, 1.0e-8f);
        const float strong_ratio = std::clamp(strongSilhouetteRatio, 0.0f, 1.0f);
        for (int start = 0; start < static_cast<int>(boundary_neighbors.size()); ++start)
        {
            if (boundary_neighbors[static_cast<std::size_t>(start)].size() != 2)
            {
                continue;
            }
            for (const int first_neighbor : boundary_neighbors[static_cast<std::size_t>(start)])
            {
                if (visited_edges.find(edge_key(start, first_neighbor)) != visited_edges.cend())
                {
                    continue;
                }
                std::vector<int> loop{start};
                int previous = start;
                int current = first_neighbor;
                bool closed = false;
                visited_edges.insert(edge_key(previous, current));
                while (static_cast<int>(loop.size()) <= maximumBoundaryEdges)
                {
                    if (current == start)
                    {
                        closed = true;
                        break;
                    }
                    loop.push_back(current);
                    const std::vector<int>& neighbors = boundary_neighbors[static_cast<std::size_t>(current)];
                    if (neighbors.size() != 2)
                    {
                        break;
                    }
                    const int next = neighbors[0] == previous ? neighbors[1] : neighbors[0];
                    const std::uint64_t next_key = edge_key(current, next);
                    if (next != start && visited_edges.find(next_key) != visited_edges.cend())
                    {
                        break;
                    }
                    previous = current;
                    current = next;
                    visited_edges.insert(next_key);
                }
                if (!closed || loop.size() < 3 || static_cast<int>(loop.size()) > maximumBoundaryEdges)
                {
                    continue;
                }
                const int silhouette_vertex_count = static_cast<int>(
                    std::count_if(loop.cbegin(),
                                  loop.cend(),
                                  [&silhouetteProtectedVertices](int vertex)
                                  { return silhouetteProtectedVertices[static_cast<std::size_t>(vertex)] != 0; }));
                // Visibility-constrained filling is fail-closed for every eligible
                // boundary loop.  Missing a silhouette tag is not positive evidence
                // that an opening is safe to cap.
                ++result.consideredLoopCount;

                MeshVertex center;
                for (const int vertex_index : loop)
                {
                    const MeshVertex& vertex = mesh.vertices[static_cast<std::size_t>(vertex_index)];
                    center.x += vertex.x;
                    center.y += vertex.y;
                    center.z += vertex.z;
                }
                const float inverse_count = 1.0f / static_cast<float>(loop.size());
                center.x *= inverse_count;
                center.y *= inverse_count;
                center.z *= inverse_count;
                std::array<MeshVertex, 5> samples{};
                samples[0] = center;
                for (int sample_index = 1; sample_index < 5; ++sample_index)
                {
                    const std::size_t loop_index = (loop.size() * static_cast<std::size_t>(sample_index - 1)) / 4;
                    const MeshVertex& boundary = mesh.vertices[static_cast<std::size_t>(loop[loop_index])];
                    samples[static_cast<std::size_t>(sample_index)].x = 0.5f * (center.x + boundary.x);
                    samples[static_cast<std::size_t>(sample_index)].y = 0.5f * (center.y + boundary.y);
                    samples[static_cast<std::size_t>(sample_index)].z = 0.5f * (center.z + boundary.z);
                }

                int supporting_views = 0;
                int conflict_views = 0;
                for (int frame_index = 0; frame_index < frames.size(); ++frame_index)
                {
                    const DepthTsdfFrame& frame = frames[frame_index];
                    if (frame.auxiliarySurfaceOnly)
                    {
                        continue;
                    }
                    const cv::Mat& valid_mask = effectiveDepthValidMasks.size() == frames.size()
                                                    ? effectiveDepthValidMasks[frame_index]
                                                    : frame.depthValidMask;
                    int projected_sample_count = 0;
                    int consistent_sample_count = 0;
                    int outside_support_sample_count = 0;
                    for (const MeshVertex& sample : samples)
                    {
                        const double world[3] = {sample.x, sample.y, sample.z};
                        double pixel[2]{};
                        double camera_depth = 0.0;
                        if (!frame.camera.projectWorldPointWithDepth(world, pixel, camera_depth))
                        {
                            continue;
                        }
                        const int column = static_cast<int>(std::lround(pixel[0]));
                        const int row = static_cast<int>(std::lround(pixel[1]));
                        if (row < 0 || column < 0 || row >= frame.supportMask.rows || column >= frame.supportMask.cols)
                        {
                            continue;
                        }
                        ++projected_sample_count;
                        if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
                        {
                            ++outside_support_sample_count;
                            continue;
                        }
                        if (valid_mask.at<std::uint8_t>(row, column) == 0)
                        {
                            continue;
                        }
                        const float observed_depth = frame.depth.at<float>(row, column);
                        const float confidence = frame.confidence.at<float>(row, column);
                        const float tolerance =
                            std::max(absolute_tolerance, 0.008f * std::abs(static_cast<float>(camera_depth)));
                        if (std::isfinite(observed_depth) && observed_depth > 0.0f && std::isfinite(confidence) &&
                            confidence >= minimumConfidence &&
                            std::abs(observed_depth - static_cast<float>(camera_depth)) <= tolerance)
                        {
                            ++consistent_sample_count;
                        }
                    }
                    if (projected_sample_count >= 3 && consistent_sample_count * 2 >= projected_sample_count)
                    {
                        ++supporting_views;
                    }
                    if (projected_sample_count >= 3 && outside_support_sample_count * 2 >= projected_sample_count)
                    {
                        ++conflict_views;
                    }
                }

                const bool release_hole =
                    DepthTsdfSurfaceBuilder::shouldReleaseVisibilityConstrainedHole(static_cast<int>(loop.size()),
                                                                                    silhouette_vertex_count,
                                                                                    supporting_views,
                                                                                    conflict_views,
                                                                                    required_supporting_views,
                                                                                    allowed_conflict_views,
                                                                                    strong_ratio);
                if (release_hole)
                {
                    for (const int vertex : loop)
                    {
                        result.protectedVertices[static_cast<std::size_t>(vertex)] = 0;
                    }
                    ++result.releasedLoopCount;
                }
                else
                {
                    for (const int vertex : loop)
                    {
                        result.protectedVertices[static_cast<std::size_t>(vertex)] = 1;
                    }
                    if (conflict_views > allowed_conflict_views)
                    {
                        ++result.rejectedConflictLoopCount;
                    }
                    else
                    {
                        ++result.rejectedSupportLoopCount;
                    }
                }
            }
        }
        return result;
    }
} // namespace xjw::mesh::tsdf_detail
