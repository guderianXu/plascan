#include <gtest/gtest.h>

#include "DepthTsdfFinalQualityGate.h"

namespace
{

xjw::mesh::TriMesh makeOpenQuad()
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(4);
    mesh.vertices[0] = {0.0f, 0.0f, 0.0f};
    mesh.vertices[1] = {1.0f, 0.0f, 0.0f};
    mesh.vertices[2] = {1.0f, 1.0f, 0.0f};
    mesh.vertices[3] = {0.0f, 1.0f, 0.0f};
    mesh.faces = {{{0, 1, 2}}, {{0, 2, 3}}};
    return mesh;
}

xjw::mesh::TriMesh makeTetrahedron()
{
    xjw::mesh::TriMesh mesh;
    mesh.vertices.resize(4);
    mesh.vertices[0] = {1.0f, 1.0f, 1.0f};
    mesh.vertices[1] = {-1.0f, -1.0f, 1.0f};
    mesh.vertices[2] = {-1.0f, 1.0f, -1.0f};
    mesh.vertices[3] = {1.0f, -1.0f, -1.0f};
    mesh.faces = {
        {{0, 2, 1}},
        {{0, 1, 3}},
        {{0, 3, 2}},
        {{1, 2, 3}}
    };
    return mesh;
}

xjw::mesh::MeshTopologyQualityStatistics makeLatest222LikeQuality()
{
    xjw::mesh::MeshTopologyQualityStatistics quality;
    quality.validFaceCount = 779170;
    quality.referencedVertexCount = 389587;
    quality.uniqueEdgeCount = 1168755;
    quality.componentCount = 1;
    quality.componentEulerCharacteristics = {2};
    quality.largestComponentFaceCount = quality.validFaceCount;
    quality.largestComponentFaceRatio = 1.0;
    quality.eulerCharacteristic = 2;
    quality.topologicalComplexity = 0;
    quality.closedTopologyEvaluated = true;
    quality.closedTwoManifold = true;
    quality.closedGenusEstimate = 0.0;
    quality.highAspectFaceRatio = 0.0440545708895363;
    quality.extremeAspectFaceRatio = 0.022545785900381175;
    quality.strictGatePassed = false;
    return quality;
}

} // namespace

TEST(DepthTsdfFinalQualityGateTest, StrictPolicyAcceptsRegularClosedSurface)
{
    const auto result = xjw::mesh::evaluateDepthTsdfFinalQualityGate(
        makeTetrahedron(),
        xjw::mesh::DepthTsdfFinalQualityPolicy::Strict);

    EXPECT_TRUE(result.passed) << result.reason.toStdString();
    EXPECT_TRUE(result.reason.isEmpty());
    EXPECT_EQ(
        result.diagnostics.value(
            QStringLiteral("final_topology_quality_gate_policy")).toString(),
        QStringLiteral("strict"));
    EXPECT_TRUE(result.diagnostics.value(
        QStringLiteral("final_topology_quality_gate_passed")).toBool());
}

TEST(DepthTsdfFinalQualityGateTest, StrictPolicyHonorsAnUpstreamGateFailure)
{
    xjw::mesh::MeshTopologyQualityStatistics quality =
        xjw::mesh::evaluateMeshTopologyQuality(makeTetrahedron());
    ASSERT_TRUE(quality.strictGatePassed);
    quality.strictGatePassed = false;

    const auto result = xjw::mesh::evaluateDepthTsdfFinalQualityGate(
        quality,
        xjw::mesh::DepthTsdfFinalQualityPolicy::Strict);

    EXPECT_FALSE(result.passed);
    EXPECT_TRUE(result.reason.contains(QStringLiteral("上游严格网格质量门")));
}

TEST(DepthTsdfFinalQualityGateTest,
     ObservationOnlyRelaxesOnlyTheOpenBoundaryRatio)
{
    const xjw::mesh::MeshTopologyQualityStatistics quality =
        xjw::mesh::evaluateMeshTopologyQuality(makeOpenQuad());
    ASSERT_FALSE(quality.strictGatePassed);
    ASSERT_GT(quality.boundaryEdgeRatio, 0.01);

    const auto strict = xjw::mesh::evaluateDepthTsdfFinalQualityGate(
        quality,
        xjw::mesh::DepthTsdfFinalQualityPolicy::Strict);
    const auto observation_only =
        xjw::mesh::evaluateDepthTsdfFinalQualityGate(
            quality,
            xjw::mesh::DepthTsdfFinalQualityPolicy::ObservationOnly);

    EXPECT_FALSE(strict.passed);
    EXPECT_TRUE(strict.reason.contains(QStringLiteral("边界边比例")));
    EXPECT_TRUE(observation_only.passed)
        << observation_only.reason.toStdString();
    EXPECT_TRUE(observation_only.reason.isEmpty());
    EXPECT_TRUE(observation_only.diagnostics.value(QStringLiteral(
        "final_topology_quality_boundary_ratio_relaxed")).toBool());
    EXPECT_DOUBLE_EQ(
        observation_only.diagnostics.value(QStringLiteral(
            "final_topology_quality_effective_maximum_boundary_edge_ratio"))
            .toDouble(),
        1.0);
}

TEST(DepthTsdfFinalQualityGateTest,
     ObservationOnlyStillRejectsComponentsAndNonManifoldGeometry)
{
    xjw::mesh::MeshTopologyQualityStatistics quality =
        xjw::mesh::evaluateMeshTopologyQuality(makeOpenQuad());
    quality.componentCount = 2;
    quality.largestComponentFaceRatio = 0.75;

    const auto disconnected = xjw::mesh::evaluateDepthTsdfFinalQualityGate(
        quality,
        xjw::mesh::DepthTsdfFinalQualityPolicy::ObservationOnly);
    EXPECT_FALSE(disconnected.passed);
    EXPECT_TRUE(disconnected.reason.contains(QStringLiteral("连通分量")));
    EXPECT_TRUE(disconnected.reason.contains(
        QStringLiteral("最大连通分量面比例")));

    quality.componentCount = 1;
    quality.largestComponentFaceRatio = 1.0;
    quality.nonManifoldEdgeCount = 1;
    const auto non_manifold = xjw::mesh::evaluateDepthTsdfFinalQualityGate(
        quality,
        xjw::mesh::DepthTsdfFinalQualityPolicy::ObservationOnly);
    EXPECT_FALSE(non_manifold.passed);
    EXPECT_TRUE(non_manifold.reason.contains(QStringLiteral("非流形边")));
}

TEST(DepthTsdfFinalQualityGateTest,
     RejectsLatest222LikeTriangleQualityUnderBothPolicies)
{
    const auto quality = makeLatest222LikeQuality();

    for (const auto policy : {
             xjw::mesh::DepthTsdfFinalQualityPolicy::Strict,
             xjw::mesh::DepthTsdfFinalQualityPolicy::ObservationOnly})
    {
        const auto result = xjw::mesh::evaluateDepthTsdfFinalQualityGate(
            quality,
            policy);
        EXPECT_FALSE(result.passed);
        EXPECT_TRUE(result.reason.contains(
            QStringLiteral("高长宽比三角面比例")));
        EXPECT_TRUE(result.reason.contains(
            QStringLiteral("极端长宽比三角面比例")));
        EXPECT_FALSE(result.diagnostics.value(QStringLiteral(
            "final_topology_quality_gate_passed")).toBool());
    }
}

TEST(DepthTsdfFinalQualityGateTest,
     ObservationOnlyStillRejectsExcessiveTopologicalComplexity)
{
    xjw::mesh::MeshTopologyQualityStatistics quality =
        xjw::mesh::evaluateMeshTopologyQuality(makeOpenQuad());
    quality.topologicalComplexity = 129;

    const auto result = xjw::mesh::evaluateDepthTsdfFinalQualityGate(
        quality,
        xjw::mesh::DepthTsdfFinalQualityPolicy::ObservationOnly);

    EXPECT_FALSE(result.passed);
    EXPECT_TRUE(result.reason.contains(QStringLiteral("拓扑复杂度")));
    EXPECT_EQ(result.diagnostics.value(QStringLiteral(
        "final_topology_quality_topological_complexity")).toInt(), 129);
}
