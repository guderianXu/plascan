#include "DepthTsdfFinalQualityGate.h"

#include <QJsonArray>
#include <QStringList>

namespace xjw::mesh
{

namespace
{

QString policyName(DepthTsdfFinalQualityPolicy policy)
{
    return policy == DepthTsdfFinalQualityPolicy::ObservationOnly
        ? QStringLiteral("observation_only")
        : QStringLiteral("strict");
}

QJsonArray componentEulersToJson(const std::vector<int> &values)
{
    QJsonArray result;
    for (const int value : values)
    {
        result.append(value);
    }
    return result;
}

QString ratioFailure(const QString &label, double value, double limit)
{
    return QStringLiteral("%1 %2 > %3")
        .arg(label)
        .arg(value, 0, 'f', 6)
        .arg(limit, 0, 'f', 6);
}

QJsonObject makeDiagnostics(
    const MeshTopologyQualityStatistics &statistics,
    DepthTsdfFinalQualityPolicy policy,
    const MeshTopologyQualityThresholds &effectiveThresholds,
    bool passed,
    const QString &reason)
{
    return {
        {QStringLiteral("final_topology_quality_gate_policy"),
         policyName(policy)},
        {QStringLiteral("final_topology_quality_gate_passed"), passed},
        {QStringLiteral("final_topology_quality_gate_reason"), reason},
        {QStringLiteral("final_topology_quality_input_strict_gate_passed"),
         statistics.strictGatePassed},
        {QStringLiteral("final_topology_quality_boundary_ratio_relaxed"),
         policy == DepthTsdfFinalQualityPolicy::ObservationOnly},
        {QStringLiteral(
             "final_topology_quality_effective_maximum_boundary_edge_ratio"),
         effectiveThresholds.maximumBoundaryEdgeRatio},
        {QStringLiteral("final_topology_quality_valid_face_count"),
         statistics.validFaceCount},
        {QStringLiteral("final_topology_quality_boundary_edge_count"),
         statistics.boundaryEdgeCount},
        {QStringLiteral("final_topology_quality_boundary_edge_ratio"),
         statistics.boundaryEdgeRatio},
        {QStringLiteral("final_topology_quality_non_manifold_edge_count"),
         statistics.nonManifoldEdgeCount},
        {QStringLiteral("final_topology_quality_non_manifold_vertex_count"),
         statistics.nonManifoldVertexCount},
        {QStringLiteral("final_topology_quality_component_count"),
         statistics.componentCount},
        {QStringLiteral("final_topology_quality_component_eulers"),
         componentEulersToJson(statistics.componentEulerCharacteristics)},
        {QStringLiteral("final_topology_quality_largest_component_face_ratio"),
         statistics.largestComponentFaceRatio},
        {QStringLiteral("final_topology_quality_euler_characteristic"),
         statistics.eulerCharacteristic},
        {QStringLiteral("final_topology_quality_topological_complexity"),
         statistics.topologicalComplexity},
        {QStringLiteral("final_topology_quality_closed_topology_evaluated"),
         statistics.closedTopologyEvaluated},
        {QStringLiteral("final_topology_quality_closed_two_manifold"),
         statistics.closedTwoManifold},
        {QStringLiteral("final_topology_quality_closed_genus_estimate"),
         statistics.closedGenusEstimate},
        {QStringLiteral("final_topology_quality_high_aspect_face_ratio"),
         statistics.highAspectFaceRatio},
        {QStringLiteral("final_topology_quality_extreme_aspect_face_ratio"),
         statistics.extremeAspectFaceRatio}
    };
}

} // namespace

DepthTsdfFinalQualityGateResult evaluateDepthTsdfFinalQualityGate(
    const MeshTopologyQualityStatistics &statistics,
    DepthTsdfFinalQualityPolicy policy,
    const MeshTopologyQualityThresholds &strictThresholds)
{
    MeshTopologyQualityThresholds effective_thresholds = strictThresholds;
    const bool observation_only =
        policy == DepthTsdfFinalQualityPolicy::ObservationOnly;
    if (observation_only)
    {
        effective_thresholds.maximumBoundaryEdgeRatio = 1.0;
    }

    QStringList failures;
    if (statistics.validFaceCount <= 0)
    {
        failures.push_back(QStringLiteral("有效三角面数量为 0"));
    }
    if (statistics.boundaryEdgeRatio >
        effective_thresholds.maximumBoundaryEdgeRatio)
    {
        failures.push_back(ratioFailure(
            QStringLiteral("边界边比例"),
            statistics.boundaryEdgeRatio,
            effective_thresholds.maximumBoundaryEdgeRatio));
    }
    if (statistics.nonManifoldEdgeCount >
        effective_thresholds.maximumNonManifoldEdgeCount)
    {
        failures.push_back(QStringLiteral("非流形边 %1 > %2")
                               .arg(statistics.nonManifoldEdgeCount)
                               .arg(effective_thresholds
                                        .maximumNonManifoldEdgeCount));
    }
    if (statistics.nonManifoldVertexCount >
        effective_thresholds.maximumNonManifoldVertexCount)
    {
        failures.push_back(QStringLiteral("非流形顶点 %1 > %2")
                               .arg(statistics.nonManifoldVertexCount)
                               .arg(effective_thresholds
                                        .maximumNonManifoldVertexCount));
    }
    if (statistics.componentCount > effective_thresholds.maximumComponentCount)
    {
        failures.push_back(QStringLiteral("连通分量 %1 > %2")
                               .arg(statistics.componentCount)
                               .arg(effective_thresholds.maximumComponentCount));
    }
    if (statistics.largestComponentFaceRatio <
        effective_thresholds.minimumLargestComponentFaceRatio)
    {
        failures.push_back(QStringLiteral("最大连通分量面比例 %1 < %2")
                               .arg(statistics.largestComponentFaceRatio,
                                    0,
                                    'f',
                                    6)
                               .arg(effective_thresholds
                                        .minimumLargestComponentFaceRatio,
                                    0,
                                    'f',
                                    6));
    }
    if (statistics.highAspectFaceRatio >
        effective_thresholds.maximumHighAspectFaceRatio)
    {
        failures.push_back(ratioFailure(
            QStringLiteral("高长宽比三角面比例"),
            statistics.highAspectFaceRatio,
            effective_thresholds.maximumHighAspectFaceRatio));
    }
    if (statistics.extremeAspectFaceRatio >
        effective_thresholds.maximumExtremeAspectFaceRatio)
    {
        failures.push_back(ratioFailure(
            QStringLiteral("极端长宽比三角面比例"),
            statistics.extremeAspectFaceRatio,
            effective_thresholds.maximumExtremeAspectFaceRatio));
    }
    if (statistics.topologicalComplexity >
        effective_thresholds.maximumTopologicalComplexity)
    {
        failures.push_back(QStringLiteral("拓扑复杂度 %1 > %2")
                               .arg(statistics.topologicalComplexity)
                               .arg(effective_thresholds
                                        .maximumTopologicalComplexity));
    }
    if (statistics.closedTopologyEvaluated &&
        statistics.closedGenusEstimate >
            effective_thresholds.maximumClosedGenus)
    {
        failures.push_back(QStringLiteral("闭合曲面亏格 %1 > %2")
                               .arg(statistics.closedGenusEstimate,
                                    0,
                                    'f',
                                    3)
                               .arg(effective_thresholds.maximumClosedGenus,
                                    0,
                                    'f',
                                    3));
    }

    const bool effective_gate_passed =
        passesMeshTopologyQualityGate(statistics, effective_thresholds);
    if (!observation_only && !statistics.strictGatePassed && failures.isEmpty())
    {
        failures.push_back(QStringLiteral("上游严格网格质量门标记为失败"));
    }

    DepthTsdfFinalQualityGateResult result;
    result.passed = effective_gate_passed &&
        (observation_only || statistics.strictGatePassed);
    result.reason = failures.join(QStringLiteral("；"));
    if (!result.passed && result.reason.isEmpty())
    {
        result.reason = QStringLiteral("最终网格未满足质量门的全部约束");
    }
    result.diagnostics = makeDiagnostics(
        statistics,
        policy,
        effective_thresholds,
        result.passed,
        result.reason);
    return result;
}

DepthTsdfFinalQualityGateResult evaluateDepthTsdfFinalQualityGate(
    const TriMesh &mesh,
    DepthTsdfFinalQualityPolicy policy,
    const MeshTopologyQualityThresholds &strictThresholds)
{
    return evaluateDepthTsdfFinalQualityGate(
        evaluateMeshTopologyQuality(mesh, strictThresholds),
        policy,
        strictThresholds);
}

} // namespace xjw::mesh
