#include "DepthTsdfRecoveryTransaction.h"

#include <QStringList>

#include <algorithm>
#include <cmath>

namespace xjw::mesh
{
namespace
{

bool validStatistics(const MeshTopologyQualityStatistics &statistics,
                     double tolerance)
{
    const auto valid_ratio = [tolerance](double value)
    {
        return std::isfinite(value) && value >= -tolerance &&
            value <= 1.0 + tolerance;
    };
    return statistics.validFaceCount > 0 &&
        statistics.boundaryEdgeCount >= 0 &&
        statistics.nonManifoldEdgeCount >= 0 &&
        statistics.nonManifoldVertexCount >= 0 &&
        statistics.componentCount > 0 &&
        statistics.topologicalComplexity >= 0 &&
        valid_ratio(statistics.largestComponentFaceRatio) &&
        valid_ratio(statistics.extremeAspectFaceRatio);
}

void addFailure(bool failed,
                std::uint32_t flag,
                const QString &reason,
                std::uint32_t *flags,
                QStringList *reasons)
{
    if (!failed)
    {
        return;
    }
    *flags |= flag;
    reasons->push_back(reason);
}

QJsonObject diagnostics(
    const MeshTopologyQualityStatistics &baseline,
    const MeshTopologyQualityStatistics &candidate,
    double ratioTolerance,
    bool accepted,
    std::uint32_t rejectionFlags,
    const QString &reason)
{
    return {
        {QStringLiteral("depth_tsdf_recovery_transaction_accepted"), accepted},
        {QStringLiteral("depth_tsdf_recovery_transaction_rejection_flags"),
         static_cast<double>(rejectionFlags)},
        {QStringLiteral("depth_tsdf_recovery_transaction_reason"), reason},
        {QStringLiteral("depth_tsdf_recovery_transaction_ratio_tolerance"),
         ratioTolerance},
        {QStringLiteral("depth_tsdf_recovery_transaction_baseline_face_count"),
         baseline.validFaceCount},
        {QStringLiteral("depth_tsdf_recovery_transaction_candidate_face_count"),
         candidate.validFaceCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_baseline_boundary_edge_count"),
         baseline.boundaryEdgeCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_candidate_boundary_edge_count"),
         candidate.boundaryEdgeCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_baseline_component_count"),
         baseline.componentCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_candidate_component_count"),
         candidate.componentCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_baseline_non_manifold_edge_count"),
         baseline.nonManifoldEdgeCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_candidate_non_manifold_edge_count"),
         candidate.nonManifoldEdgeCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_baseline_non_manifold_vertex_count"),
         baseline.nonManifoldVertexCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_candidate_non_manifold_vertex_count"),
         candidate.nonManifoldVertexCount},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_baseline_topological_complexity"),
         baseline.topologicalComplexity},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_candidate_topological_complexity"),
         candidate.topologicalComplexity},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_baseline_largest_component_face_ratio"),
         baseline.largestComponentFaceRatio},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_candidate_largest_component_face_ratio"),
         candidate.largestComponentFaceRatio},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_baseline_extreme_aspect_face_ratio"),
         baseline.extremeAspectFaceRatio},
        {QStringLiteral(
             "depth_tsdf_recovery_transaction_candidate_extreme_aspect_face_ratio"),
         candidate.extremeAspectFaceRatio}
    };
}

} // namespace

DepthTsdfRecoveryTransactionEvaluation evaluateDepthTsdfRecoveryTransaction(
    const MeshTopologyQualityStatistics &baseline,
    const MeshTopologyQualityStatistics &candidate,
    const DepthTsdfRecoveryTransactionOptions &options)
{
    const double tolerance = std::max(0.0, options.ratioTolerance);
    std::uint32_t flags = DepthTsdfRecoveryTransactionAccepted;
    QStringList reasons;
    addFailure(
        !validStatistics(baseline, tolerance),
        DepthTsdfRecoveryTransactionInvalidBaseline,
        QStringLiteral("基线网格质量统计无效"),
        &flags,
        &reasons);
    addFailure(
        !validStatistics(candidate, tolerance),
        DepthTsdfRecoveryTransactionInvalidCandidate,
        QStringLiteral("候选网格质量统计无效"),
        &flags,
        &reasons);
    addFailure(
        candidate.boundaryEdgeCount >= baseline.boundaryEdgeCount,
        DepthTsdfRecoveryTransactionBoundaryNotReduced,
        QStringLiteral("候选边界边未严格减少"),
        &flags,
        &reasons);
    addFailure(
        candidate.componentCount > baseline.componentCount,
        DepthTsdfRecoveryTransactionComponentGrowth,
        QStringLiteral("候选连通分量数量增加"),
        &flags,
        &reasons);
    addFailure(
        candidate.nonManifoldEdgeCount > baseline.nonManifoldEdgeCount,
        DepthTsdfRecoveryTransactionNonManifoldEdgeGrowth,
        QStringLiteral("候选非流形边数量增加"),
        &flags,
        &reasons);
    addFailure(
        candidate.nonManifoldVertexCount > baseline.nonManifoldVertexCount,
        DepthTsdfRecoveryTransactionNonManifoldVertexGrowth,
        QStringLiteral("候选非流形顶点数量增加"),
        &flags,
        &reasons);
    addFailure(
        candidate.topologicalComplexity > baseline.topologicalComplexity,
        DepthTsdfRecoveryTransactionTopologicalComplexityGrowth,
        QStringLiteral("候选拓扑复杂度增加"),
        &flags,
        &reasons);
    addFailure(
        candidate.largestComponentFaceRatio + tolerance <
            baseline.largestComponentFaceRatio,
        DepthTsdfRecoveryTransactionLargestComponentRatioRegression,
        QStringLiteral("候选最大连通分量占比下降"),
        &flags,
        &reasons);
    addFailure(
        candidate.extremeAspectFaceRatio >
            baseline.extremeAspectFaceRatio + tolerance,
        DepthTsdfRecoveryTransactionExtremeAspectRatioRegression,
        QStringLiteral("候选极端长宽比三角面占比恶化"),
        &flags,
        &reasons);

    DepthTsdfRecoveryTransactionEvaluation result;
    result.accepted = flags == DepthTsdfRecoveryTransactionAccepted;
    result.rejectionFlags = flags;
    result.reason = reasons.join(QStringLiteral("；"));
    result.diagnostics = diagnostics(
        baseline,
        candidate,
        tolerance,
        result.accepted,
        flags,
        result.reason);
    return result;
}

DepthTsdfRecoveryTransactionEvaluation evaluateDepthTsdfRecoveryTransaction(
    const TriMesh &baseline,
    const TriMesh &candidate,
    const DepthTsdfRecoveryTransactionOptions &options)
{
    return evaluateDepthTsdfRecoveryTransaction(
        evaluateMeshTopologyQuality(baseline),
        evaluateMeshTopologyQuality(candidate),
        options);
}

bool commitDepthTsdfRecoveryTransaction(
    const DepthTsdfRecoveryTransactionEvaluation &evaluation,
    const std::vector<std::uint8_t> &baselineSupport,
    std::vector<std::uint8_t> *candidateSupport)
{
    if (!candidateSupport)
    {
        return false;
    }
    if (!evaluation.accepted ||
        candidateSupport->size() != baselineSupport.size())
    {
        *candidateSupport = baselineSupport;
        return false;
    }
    return true;
}

} // namespace xjw::mesh
