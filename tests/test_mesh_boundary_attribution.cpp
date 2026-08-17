#include "MeshBoundaryAttribution.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace
{

xjw::mesh::TriMesh makeOpenQuad()
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices = {
        {0.1f, 0.1f, 0.5f},
        {0.9f, 0.1f, 0.5f},
        {0.9f, 0.9f, 0.5f},
        {0.1f, 0.9f, 0.5f}};
    mesh.faces = {
        {{0, 1, 2}},
        {{0, 2, 3}}};
    return mesh;
}

xjw::mesh::DepthTsdfLayout makeUnitLayout()
{
    xjw::mesh::DepthTsdfLayout layout;
    layout.ok = true;
    layout.boundsMin = {0.0f, 0.0f, 0.0f};
    layout.boundsMax = {1.0f, 1.0f, 1.0f};
    layout.cells = {1, 1, 1};
    layout.voxelSize = {1.0f, 1.0f, 1.0f};
    layout.sampleCount = 8;
    return layout;
}

} // namespace

TEST(MeshBoundaryAttributionTest, AssignsEveryBoundaryEdgeToExtraction)
{
    const xjw::mesh::TriMesh mesh = makeOpenQuad();
    const xjw::mesh::DepthTsdfLayout layout = makeUnitLayout();
    const std::vector<float> tsdf(8, 0.1f);
    const std::vector<float> weight(8, 1.0f);
    const std::vector<float> surface_weight(8, 1.0f);
    const std::vector<xjw::mesh::DepthGeometrySourceMask> source_masks(8, 0x3);
    const std::vector<std::uint16_t> spread(8, 100);
    const std::vector<std::uint8_t> supported(8, 1);

    std::vector<xjw::mesh::MeshBoundaryAttributionReason> reasons;
    std::vector<xjw::mesh::MeshBoundaryEdgeAttribution> edge_attributions;
    const auto statistics = xjw::mesh::attributeMeshBoundaryEdges(
        mesh,
        layout,
        tsdf,
        weight,
        surface_weight,
        source_masks,
        spread,
        supported,
        {},
        &reasons,
        &edge_attributions);

    EXPECT_EQ(statistics.boundaryEdgeCount, 4U);
    EXPECT_EQ(statistics.extractionOrPostprocessEdgeCount, 4U);
    EXPECT_EQ(statistics.unclassifiedEdgeCount, 0U);
    ASSERT_EQ(edge_attributions.size(), 4U);
    for (const auto &edge : edge_attributions)
    {
        EXPECT_EQ(
            edge.reason,
            xjw::mesh::MeshBoundaryAttributionReason::ExtractionOrPostprocess);
        EXPECT_EQ(edge.evidenceReason, edge.reason);
        EXPECT_EQ(edge.sourceMask, 0x3);
        EXPECT_GT(edge.length, 0.0f);
    }
    ASSERT_EQ(reasons.size(), mesh.vertices.size());
    for (const auto reason : reasons)
    {
        EXPECT_EQ(
            reason,
            xjw::mesh::MeshBoundaryAttributionReason::ExtractionOrPostprocess);
    }

    xjw::mesh::TriMesh colored_mesh = mesh;
    xjw::mesh::applyMeshBoundaryAttributionColors(&colored_mesh, reasons);
    for (const xjw::mesh::MeshVertex &vertex : colored_mesh.vertices)
    {
        EXPECT_EQ(vertex.r, 255);
        EXPECT_EQ(vertex.g, 128);
        EXPECT_EQ(vertex.b, 0);
    }
}

TEST(MeshBoundaryAttributionTest, DistinguishesMissingAndWeakSources)
{
    const xjw::mesh::TriMesh mesh = makeOpenQuad();
    const xjw::mesh::DepthTsdfLayout layout = makeUnitLayout();
    const std::vector<float> tsdf(8, 0.1f);
    std::vector<float> weight(8, 0.0f);
    const std::vector<float> surface_weight(8, 1.0f);
    const std::vector<xjw::mesh::DepthGeometrySourceMask> source_masks(8, 0x1);
    const std::vector<std::uint16_t> spread(
        8, std::numeric_limits<std::uint16_t>::max());
    const std::vector<std::uint8_t> supported(8, 0);

    const auto missing = xjw::mesh::attributeMeshBoundaryEdges(
        mesh,
        layout,
        tsdf,
        weight,
        surface_weight,
        source_masks,
        spread,
        supported);
    EXPECT_EQ(missing.noObservationEdgeCount, 4U);

    weight.assign(8, 1.0f);
    const auto weak_sources = xjw::mesh::attributeMeshBoundaryEdges(
        mesh,
        layout,
        tsdf,
        weight,
        surface_weight,
        source_masks,
        spread,
        supported);
    EXPECT_EQ(weak_sources.insufficientSourceEdgeCount, 4U);
    EXPECT_EQ(weak_sources.unclassifiedEdgeCount, 0U);
}

TEST(MeshBoundaryAttributionTest, TreatsPartialCellSupportAsSupportGateBoundary)
{
    const xjw::mesh::TriMesh mesh = makeOpenQuad();
    const xjw::mesh::DepthTsdfLayout layout = makeUnitLayout();
    const std::vector<float> tsdf(8, 0.1f);
    const std::vector<float> weight(8, 1.0f);
    const std::vector<float> surface_weight(8, 1.0f);
    const std::vector<xjw::mesh::DepthGeometrySourceMask> source_masks(8, 0x3);
    const std::vector<std::uint16_t> spread(8, 100);
    std::vector<std::uint8_t> supported(8, 1);
    supported[0] = 0;

    std::vector<xjw::mesh::MeshBoundaryAttributionReason> reasons;
    const auto statistics = xjw::mesh::attributeMeshBoundaryEdges(
        mesh,
        layout,
        tsdf,
        weight,
        surface_weight,
        source_masks,
        spread,
        supported,
        {},
        &reasons);

    EXPECT_EQ(statistics.boundaryEdgeCount, 4U);
    EXPECT_EQ(statistics.supportGateRejectedEdgeCount, 4U);
    EXPECT_EQ(statistics.extractionOrPostprocessEdgeCount, 0U);
    for (const auto reason : reasons)
    {
        EXPECT_EQ(
            reason,
            xjw::mesh::MeshBoundaryAttributionReason::SupportGateRejected);
    }
}
